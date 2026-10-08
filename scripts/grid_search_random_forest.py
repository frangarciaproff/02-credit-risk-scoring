"""Grid search over Random Forest hyperparameters — SPEC.md Plan step 3
hyperparameter-selection follow-up (D20).

train_random_forest.py's own docstring states every hyperparameter is a
CLI flag "not searched" — a human runs it repeatedly and compares CV
results by hand. This script is a documented, scoped exception (D20):
covering a real grid by hand doesn't scale, so this automates the
repetition, but keeps the non-negotiable part of that rule intact —
every combination is scored by CV only, the held-out test split (D6) is
never touched here, and every run is logged individually to
results/model_runs.csv exactly as a manual invocation would be. The
"winner" is a data-driven readout of mean CV PR-AUC — this project's
stated primary metric for the imbalanced target (SPEC.md, Success
criteria; D4) — never a black-box search picking a final answer by
itself; ROC-AUC is reported alongside for every combination, not used to
rank them.

Imports (not duplicates) train_random_forest.py's own cross_validate,
build_model, load_dataset, split_holdout, summarize_scores, and
append_run_log — unlike this project's other scripts, which duplicate
small loaders deliberately to stay independently runnable (see
clean_features.py's load_train docstring), cross_validate here is ~150
lines that has already changed three times this week (D17 train/val
split, D18 fold parallelism, D19 max_features fix); duplicating it would
be a live drift risk this script's job doesn't justify. The trade-off:
this only works run as a module (see Usage), not `python scripts/
grid_search_random_forest.py` directly — `-m` puts the project root
(cwd) on sys.path, which a direct script invocation does not, and that
root is what a `scripts.<name>` import needs to resolve.

Combinations run sequentially, one at a time; each combination uses
cross_validate's own fold-level parallelism (--n-jobs) internally.
Combinations are deliberately NOT parallelized against each other — that
would oversubscribe the same cores twice over, the same reasoning D18
already applied to folds vs. per-fit threads.

Usage:
    uv run python -m scripts.grid_search_random_forest --dataset raw
    uv run python -m scripts.grid_search_random_forest --dataset raw --n-jobs 4
"""

import argparse
import itertools
import json
import time
from datetime import UTC, datetime
from typing import NamedTuple

from scripts.train_random_forest import (
    RUN_LOG_PATH,
    append_run_log,
    cross_validate,
    load_dataset,
    split_holdout,
    summarize_scores,
)

# The literal grid decided on for the raw dataset (D20). min_samples_split
# and class_weight are held fixed at train_random_forest.py's own D3/D20
# values (25, None) rather than swept — D20 already found
# class_weight="balanced" hurts PR-AUC for RF on this data, and
# min_samples_split wasn't part of what was still undecided.
RAW_GRID: dict[str, list] = {
    "max_depth": [8, 10, 15, None],
    "min_samples_leaf": [4, 8, 12],
    "max_features": ["sqrt", 0.2, 0.3],
    "n_estimators": [250],
}
FIXED_HYPERPARAMS = {"min_samples_split": 25, "class_weight": None}

GRIDS: dict[str, dict[str, list]] = {"raw": RAW_GRID}


class GridResult(NamedTuple):
    hyperparams: dict
    cv_pr_auc_mean: float
    cv_roc_auc_mean: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=sorted(GRIDS), default="raw")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--n-jobs", type=int, default=-1)
    return parser.parse_args()


def _combo_namespace(combo: dict, args: argparse.Namespace) -> argparse.Namespace:
    """Build the argparse.Namespace-shaped object cross_validate/
    build_model expect, from one grid combination plus the fixed values.
    """
    return argparse.Namespace(
        n_estimators=combo["n_estimators"],
        max_depth=combo["max_depth"],
        min_samples_split=FIXED_HYPERPARAMS["min_samples_split"],
        min_samples_leaf=combo["min_samples_leaf"],
        max_features=combo["max_features"],
        class_weight=FIXED_HYPERPARAMS["class_weight"],
        n_folds=args.n_folds,
        n_jobs=args.n_jobs,
    )


def main() -> None:
    args = parse_args()
    grid = GRIDS[args.dataset]
    keys = list(grid.keys())
    combos = [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*grid.values())
    ]
    print(
        f"Random Forest grid search | dataset={args.dataset} | "
        f"{len(combos)} combinations | fixed={FIXED_HYPERPARAMS} | "
        f"selecting by mean CV PR-AUC (D4/SPEC.md primary metric)"
    )

    df = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = split_holdout(df)
    print(
        f"Held out {len(X_test):,} rows for test (D6, untouched by this "
        f"script), {len(X_train):,} available for CV."
    )

    leaderboard: list[GridResult] = []
    for i, combo in enumerate(combos, start=1):
        hyperparams = {**combo, **FIXED_HYPERPARAMS}
        combo_args = _combo_namespace(combo, args)
        print(f"\n[{i}/{len(combos)}] {hyperparams}")

        start = time.monotonic()
        cv_scores = cross_validate(X_train, y_train, combo_args)
        elapsed = time.monotonic() - start

        train_roc_summary = summarize_scores(cv_scores["train_roc_auc"])
        train_pr_summary = summarize_scores(cv_scores["train_pr_auc"])
        roc_summary = summarize_scores(cv_scores["roc_auc"])
        pr_summary = summarize_scores(cv_scores["pr_auc"])
        print(
            f"  Val ROC-AUC={roc_summary['mean']:.4f} +/- "
            f"{roc_summary['ci95_half_width']:.4f}  |  "
            f"Val PR-AUC={pr_summary['mean']:.4f} +/- "
            f"{pr_summary['ci95_half_width']:.4f}  ({elapsed:.0f}s)"
        )

        append_run_log(
            {
                "timestamp": datetime.now(tz=UTC).isoformat(),
                "model": "random_forest",
                "dataset": args.dataset,
                "n_folds": args.n_folds,
                "final_eval": False,
                "calibration": None,
                "hyperparams": json.dumps(hyperparams),
                "train_roc_auc_mean": train_roc_summary["mean"],
                "train_roc_auc_ci95": train_roc_summary["ci95_half_width"],
                "train_pr_auc_mean": train_pr_summary["mean"],
                "train_pr_auc_ci95": train_pr_summary["ci95_half_width"],
                "cv_roc_auc_mean": roc_summary["mean"],
                "cv_roc_auc_ci95": roc_summary["ci95_half_width"],
                "cv_pr_auc_mean": pr_summary["mean"],
                "cv_pr_auc_ci95": pr_summary["ci95_half_width"],
                "val_roc_auc_folds": json.dumps(cv_scores["roc_auc"].tolist()),
                "val_pr_auc_folds": json.dumps(cv_scores["pr_auc"].tolist()),
                "test_roc_auc": None,
                "test_pr_auc": None,
                "test_brier": None,
                "test_log_loss": None,
                # D30: bootstrap test-metric CIs — never computed here,
                # this script never touches the test set (D16).
                "test_roc_auc_ci95": None,
                "test_pr_auc_ci95": None,
                "test_brier_ci95": None,
                "test_log_loss_ci95": None,
                "reliability_plot": None,
            }
        )
        leaderboard.append(
            GridResult(
                hyperparams=hyperparams,
                cv_pr_auc_mean=pr_summary["mean"],
                cv_roc_auc_mean=roc_summary["mean"],
            )
        )

    leaderboard.sort(key=lambda r: r.cv_pr_auc_mean, reverse=True)
    print(f"\n{'=' * 70}\nTop 5 by mean CV PR-AUC (D4/SPEC.md primary metric):")
    for rank, r in enumerate(leaderboard[:5], start=1):
        print(
            f"  {rank}. PR-AUC={r.cv_pr_auc_mean:.4f}  "
            f"ROC-AUC={r.cv_roc_auc_mean:.4f}  {r.hyperparams}"
        )
    print(f"\nAll {len(combos)} runs appended to {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
