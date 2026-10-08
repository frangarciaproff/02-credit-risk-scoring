"""Random Forest benchmark — SPEC.md Plan step 3.

Loads one of the D10 raw-vs-treated tables (train_raw.csv /
train_treated_rf.csv), carves out one held-out test split exactly once
(D6, fixed RANDOM_STATE/TEST_SIZE — never CLI-overridable, so the same
rows are held out on every run of this script or train_lightgbm.py on
the same dataset), then runs stratified k-fold CV over the remaining
train+val pool with the given hyperparameters (D3's tuned values as
defaults) and reports ROC-AUC + PR-AUC across folds (mean, std, 95% CI).

Every hyperparameter is a CLI flag, not searched — the point of this
script is for a human to run it repeatedly and compare CV results across
runs (results/model_runs.csv, shared with train_lightgbm.py) to decide
values, not to have a search pick a final answer.

D16: by default (no --final) this script never touches the held-out test
split — only the CV metrics inform hyperparameter choice. Evaluating a
held-out test set on every hyperparameter iteration would just relocate
the "tuning against test data" problem D4 already flagged for the tuning
metric; passing --final fits once on the full train+val pool and reports
the held-out test metrics exactly once, after the values are already
decided from CV — do this only when you're done iterating.

D21: --final also fits CalibratedClassifierCV (isotonic and/or sigmoid,
see --calibration) on the same train+val pool and reports ROC-AUC,
PR-AUC, Brier score, and log loss for every variant (uncalibrated +
each calibration method) on the held-out test set, plus one shared
reliability-curve figure with quantile bins and per-bin counts.

Usage:
    uv run python scripts/train_random_forest.py
    uv run python scripts/train_random_forest.py --dataset raw --n-folds 10
    uv run python scripts/train_random_forest.py --n-estimators 300 \
        --max-depth 20 --final
"""

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
RUN_LOG_PATH = RESULTS_DIR / "model_runs.csv"

RAW_PATH = PROCESSED_DIR / "train_raw.csv"
TREATED_PATH = PROCESSED_DIR / "train_treated_rf.csv"

# D21: folds for CalibratedClassifierCV's internal cross-fitting on the
# 261k train+val pool under --final. Fixed (not a CLI flag) — this is a
# one-time evaluation step, not something to be iterated on like the
# training hyperparameters above.
CALIBRATION_CV_FOLDS = 5

# D30: nonparametric bootstrap for a CI on each test-set metric reported
# under --final. The test set is touched exactly once (D6/D16) — the
# point estimate is still that one evaluation — but a single number from
# ~46k rows says nothing about how much it would move on a different
# sample of applicants; the bootstrap resamples that one held-out set
# with replacement to characterize that, not to produce a second "real"
# estimate. Fixed, not a CLI flag, for the same reason CALIBRATION_CV_FOLDS
# is fixed above.
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# D6/D16: fixed, not CLI-overridable — this is what makes every run of
# this script (and train_lightgbm.py) comparable in model_runs.csv: the
# same 15% of rows are held out regardless of model, dataset choice, or
# hyperparameters, since every treated/raw table preserves the original
# application_train.csv row order (no shuffle anywhere upstream).
RANDOM_STATE = 42
TEST_SIZE = 0.15
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

RUN_LOG_COLUMNS = [
    "timestamp",
    "model",
    "dataset",
    "n_folds",
    "final_eval",
    "calibration",
    "hyperparams",
    "train_roc_auc_mean",
    "train_roc_auc_ci95",
    "train_pr_auc_mean",
    "train_pr_auc_ci95",
    "cv_roc_auc_mean",
    "cv_roc_auc_ci95",
    "cv_pr_auc_mean",
    "cv_pr_auc_ci95",
    # D26: raw per-fold validation scores (JSON list of length n_folds),
    # not just the mean/CI already above — needed for a paired fold-by-
    # fold comparison between two runs (e.g. RF vs. LightGBM at matching
    # n_folds), which a mean/CI pair alone can't support.
    "val_roc_auc_folds",
    "val_pr_auc_folds",
    "test_roc_auc",
    "test_pr_auc",
    "test_brier",
    "test_log_loss",
    # D30: nonparametric bootstrap 95% CI per test metric, JSON [lo, hi] —
    # None under CV-only runs, same as the point-estimate columns above.
    "test_roc_auc_ci95",
    "test_pr_auc_ci95",
    "test_brier_ci95",
    "test_log_loss_ci95",
    "reliability_plot",
]


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load the chosen D10 cell (raw or RF-treated). Duplicated
    load-by-path pattern rather than imported — see clean_features.py's
    load_train docstring for why each script stays self-contained.
    """
    path = RAW_PATH if dataset == "raw" else TREATED_PATH
    return pd.read_csv(path)


def split_holdout(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carve out the one D6 held-out test split, stratified on TARGET,
    with the fixed RANDOM_STATE/TEST_SIZE above.
    """
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL, ID_COL])
    return train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )


def build_model(args: argparse.Namespace, n_jobs: int = -1) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        class_weight=args.class_weight,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
    )


def _fit_and_score_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> tuple[float, float, float, float]:
    """Fit and score one fold in isolation — the unit of work handed to
    each parallel worker by cross_validate. Each worker's own model uses
    n_jobs=1: the folds themselves are what's parallelized (one process
    per fold), so letting each fold's forest also spawn its own thread
    pool would oversubscribe the same cores twice over and net out
    slower, not faster.
    """
    model = build_model(args, n_jobs=1)
    model.fit(X.iloc[train_idx], y.iloc[train_idx])

    train_proba = model.predict_proba(X.iloc[train_idx])[:, 1]
    train_roc = roc_auc_score(y.iloc[train_idx], train_proba)
    train_pr = average_precision_score(y.iloc[train_idx], train_proba)

    val_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
    val_roc = roc_auc_score(y.iloc[val_idx], val_proba)
    val_pr = average_precision_score(y.iloc[val_idx], val_proba)

    return train_roc, train_pr, val_roc, val_pr


def cross_validate(
    X: pd.DataFrame, y: pd.Series, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Stratified k-fold CV over the train+val pool. A fresh model is fit
    per fold with the same hyperparameters; only the fold split changes.
    Folds are fit concurrently (joblib, one process per fold, capped at
    --n-jobs) rather than one after another — each fold is fully
    independent (fresh model, disjoint data), so there's no correctness
    cost to running them at once, only a wall-clock win bounded by your
    core count and --n-folds (5 folds across 8+ cores: close to 5x;
    beyond that, diminishing returns from having more workers than folds).

    Scores both the fold's own training data and its validation data —
    the gap between the two is the overfitting diagnostic: RF trees are
    grown deep and largely unpruned (bagging reduces variance at the
    ensemble level, not by preventing each tree from fitting its own
    bootstrap sample closely), so some train-vs-val gap is normal for RF
    and not on its own evidence of a problem. A *val* score that's weak
    *and* far below train is the actual warning sign; a small gap with
    strong val performance is fine.
    """
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(skf.split(X, y))
    results = Parallel(n_jobs=args.n_jobs)(
        delayed(_fit_and_score_fold)(X, y, train_idx, val_idx, args)
        for train_idx, val_idx in splits
    )

    train_roc_scores = np.array([r[0] for r in results])
    train_pr_scores = np.array([r[1] for r in results])
    val_roc_scores = np.array([r[2] for r in results])
    val_pr_scores = np.array([r[3] for r in results])

    for fold in range(args.n_folds):
        print(
            f"  fold {fold + 1}/{args.n_folds}: "
            f"train ROC-AUC={train_roc_scores[fold]:.4f} "
            f"PR-AUC={train_pr_scores[fold]:.4f}  |  "
            f"val ROC-AUC={val_roc_scores[fold]:.4f} "
            f"PR-AUC={val_pr_scores[fold]:.4f}"
        )
    return {
        "train_roc_auc": train_roc_scores,
        "train_pr_auc": train_pr_scores,
        "roc_auc": val_roc_scores,
        "pr_auc": val_pr_scores,
    }


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    """Mean, std, and a 95% CI half-width via the t-distribution — correct
    for the small number of fold-level observations this actually is (5
    by default), unlike a normal-approximation shortcut. Report as
    mean +/- ci95_half_width.
    """
    n = len(scores)
    mean = float(scores.mean())
    std = float(scores.std(ddof=1)) if n > 1 else 0.0
    ci_half = (
        float(stats.t.ppf(0.975, df=n - 1) * std / np.sqrt(n))
        if n > 1
        else float("nan")
    )
    return {"mean": mean, "std": std, "ci95_half_width": ci_half}


def _quantile_reliability_curve(
    y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reliability curve using quantile (equal-count) bins instead of
    sklearn's calibration_curve default equal-width bins. RF's predicted
    probabilities are compressed toward the low end (min_samples_leaf,
    an 8% base rate) — equal-width bins leave the upper bins almost
    empty (confirmed on a real --final run: 5 of 10 bins had zero test
    points). Equal-count bins guarantee every bin gets roughly the same
    number of test points, at the cost of unequal bin widths.

    Returns (mean_predicted, observed_rate, count) per bin, ascending by
    predicted probability. Ties in y_prob (RF's compressed low end again)
    can collapse requested quantile edges into fewer *effective* bins —
    that's reported via the returned arrays' length, not hidden.
    """
    y_true_arr = np.asarray(y_true)
    edges = np.unique(np.quantile(y_prob, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        raise ValueError("y_prob has too little spread to form any bin.")

    bin_idx = np.digitize(y_prob, edges[1:-1], right=True)
    n_effective_bins = len(edges) - 1
    mean_pred = np.array([y_prob[bin_idx == i].mean() for i in range(n_effective_bins)])
    observed = np.array(
        [y_true_arr[bin_idx == i].mean() for i in range(n_effective_bins)]
    )
    counts = np.array([int((bin_idx == i).sum()) for i in range(n_effective_bins)])
    return mean_pred, observed, counts


def _bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = N_BOOTSTRAP,
    ci: float = BOOTSTRAP_CI,
    random_state: int = RANDOM_STATE,
) -> tuple[float, float]:
    """Nonparametric (percentile) bootstrap CI for one test-set metric.
    Resamples (y_true, y_prob) with replacement, same size as the test
    set, `n_boot` times; recomputes `metric_fn` on each resample; returns
    the `ci`-percentile interval (e.g. the 2.5th/97.5th percentile for
    ci=0.95). D30.

    A resample that happens to contain only one class makes roc_auc/
    pr_auc undefined — redrawn rather than counted, not propagated as a
    NaN into the interval. At an 8% base rate and ~46k test rows this is
    astronomically unlikely to ever actually trigger; the guard exists so
    a pathological resample can't silently corrupt the interval if it
    ever does.
    """
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    boot_scores = np.empty(n_boot)
    i = 0
    while i < n_boot:
        idx = rng.integers(0, n, size=n)
        sample_y = y_true[idx]
        if sample_y.min() == sample_y.max():
            continue
        boot_scores[i] = metric_fn(sample_y, y_prob[idx])
        i += 1
    lower_pct = (1 - ci) / 2 * 100
    upper_pct = (1 + ci) / 2 * 100
    return (
        float(np.percentile(boot_scores, lower_pct)),
        float(np.percentile(boot_scores, upper_pct)),
    )


def _test_metrics_and_curve(
    y_test: pd.Series, proba: np.ndarray
) -> tuple[
    dict[str, float], dict[str, tuple[float, float]], np.ndarray, np.ndarray, np.ndarray
]:
    """Point estimate plus a bootstrap CI (D30) for each of the four test
    metrics, computed on the real (unresampled) test set and on resamples
    of it respectively — see _bootstrap_ci's docstring for what the CI
    does and doesn't mean.
    """
    y_true_arr = np.asarray(y_test)
    metrics = {
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
        "brier": brier_score_loss(y_test, proba),
        "log_loss": log_loss(y_test, proba),
    }
    metric_fns: dict[str, Callable[[np.ndarray, np.ndarray], float]] = {
        "roc_auc": roc_auc_score,
        "pr_auc": average_precision_score,
        "brier": brier_score_loss,
        "log_loss": log_loss,
    }
    cis = {
        name: _bootstrap_ci(y_true_arr, proba, fn) for name, fn in metric_fns.items()
    }
    mean_pred, observed, counts = _quantile_reliability_curve(y_test, proba)
    return metrics, cis, mean_pred, observed, counts


_CALIBRATION_LABELS = {
    "none": "Uncalibrated RF",
    "isotonic": "Isotonic",
    "sigmoid": "Sigmoid (Platt)",
}


def evaluate_holdout(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    args: argparse.Namespace,
) -> tuple[
    dict[str, dict[str, float]], dict[str, dict[str, tuple[float, float]]], Path
]:
    """Fit once on the full train+val pool, evaluate once on the D6
    held-out test split (D16/D21) — the uncalibrated model plus, per
    --calibration, one CalibratedClassifierCV(cv=CALIBRATION_CV_FOLDS)
    wrapper per method ("isotonic", "sigmoid", or both). Each variant's
    ROC-AUC, PR-AUC, Brier score, and log loss are reported with a
    bootstrap 95% CI (D30), and every variant's quantile-binned
    reliability curve is drawn on one shared figure for direct
    comparison. Only called under --final.

    Each CalibratedClassifierCV wraps a *fresh, unfit* estimator (not the
    one already fit above) with cv=CALIBRATION_CV_FOLDS: with an integer
    cv and no --calibration cv=prefit path, it fits its own internal
    clones per fold and calibrates each on that fold's own held-out
    slice, rather than reusing predictions the base model already saw
    during its own training — reusing those would bias the calibration
    curve optimistically. Its inner estimator is built with n_jobs=1 and
    the outer cv folds parallelized via --n-jobs, the same
    oversubscription-avoiding split cross_validate uses for CV folds.
    """
    variants: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    print("  fitting the uncalibrated model on the full train+val pool...")
    raw_model = build_model(args)
    raw_model.fit(X_train, y_train)
    raw_proba = raw_model.predict_proba(X_test)[:, 1]
    variants["none"], cis["none"], *curve = _test_metrics_and_curve(y_test, raw_proba)
    curves["none"] = tuple(curve)

    methods_by_flag = {
        "none": [],
        "isotonic": ["isotonic"],
        "sigmoid": ["sigmoid"],
        "both": ["isotonic", "sigmoid"],
    }
    calibration_cv = StratifiedKFold(
        n_splits=CALIBRATION_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )
    for method in methods_by_flag[args.calibration]:
        print(
            f"  fitting CalibratedClassifierCV(method={method!r}, "
            f"cv={CALIBRATION_CV_FOLDS}) on the full train+val pool..."
        )
        calibrated = CalibratedClassifierCV(
            estimator=build_model(args, n_jobs=1),
            method=method,
            cv=calibration_cv,
            n_jobs=args.n_jobs,
        )
        calibrated.fit(X_train, y_train)
        proba = calibrated.predict_proba(X_test)[:, 1]
        variants[method], cis[method], *curve = _test_metrics_and_curve(y_test, proba)
        curves[method] = tuple(curve)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fig_path = FIGURES_DIR / f"reliability_random_forest_{args.dataset}_{timestamp}.png"
    fig, ax = plt.subplots(figsize=(6, 6))
    for name, (mean_pred, observed, counts) in curves.items():
        ax.plot(
            mean_pred,
            observed,
            marker="o",
            label=f"{_CALIBRATION_LABELS[name]} ({len(counts)} bins)",
        )
        print(f"  {_CALIBRATION_LABELS[name]} per-bin test counts: {counts.tolist()}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Mean predicted probability (quantile bin)")
    ax.set_ylabel("Observed frequency")
    ax.set_title(f"Reliability curve — Random Forest ({args.dataset}, held-out test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return variants, cis, fig_path


def append_run_log(row: dict, path: Path = RUN_LOG_PATH) -> None:
    """Append one run's results to the shared model-comparison log,
    creating it with a fixed column order if it doesn't exist yet.
    Reindexed to RUN_LOG_COLUMNS on every write, so a log started under
    an older schema (e.g. before the train-vs-val columns existed) still
    ends up with the current column order instead of having new columns
    tacked onto wherever pd.concat happens to put them; old rows just
    get NaN in columns that didn't exist when they were written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    row_df = pd.DataFrame([row], columns=RUN_LOG_COLUMNS)
    if path.exists():
        existing = pd.read_csv(path)
        combined = pd.concat([existing, row_df], ignore_index=True)
        combined = combined.reindex(columns=RUN_LOG_COLUMNS)
    else:
        combined = row_df
    combined.to_csv(path, index=False)


def _int_or_none(value: str) -> int | None:
    return None if value.lower() == "none" else int(value)


def _max_features_value(value: str) -> str | int | float | None:
    if value.lower() == "none":
        return None
    # Try int first: sklearn's max_features takes an int as an absolute
    # feature count (e.g. 9 -> consider 9 features) versus a float only
    # for a fraction in (0.0, 1.0]. int("9") succeeds and returns 9; a
    # naive float(value) would return 9.0, which sklearn rejects outright
    # since 9.0 is outside (0.0, 1.0].
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _class_weight_value(value: str) -> str | None:
    return None if value.lower() == "none" else value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dataset", choices=["raw", "treated"], default="treated")
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Worker processes for cross_validate's fold-level "
        "parallelism (default -1 = all cores). Each fold's own forest "
        "runs single-threaded regardless (see _fit_and_score_fold) — "
        "this flag is the only knob for how many folds run at once.",
    )
    parser.add_argument("--final", action="store_true")
    parser.add_argument(
        "--calibration",
        choices=["none", "isotonic", "sigmoid", "both"],
        default="both",
        help="Only used under --final. Which CalibratedClassifierCV "
        "method(s) to fit and evaluate on the held-out test set alongside "
        "the uncalibrated model (D21). Default 'both' is what a real "
        "--final run should use; 'none'/single-method exist for faster "
        "dev/debug runs, not for iterating on this choice per model.",
    )
    # D3 values as defaults.
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=_int_or_none, default=15)
    parser.add_argument("--min-samples-split", type=int, default=25)
    parser.add_argument("--min-samples-leaf", type=int, default=8)
    parser.add_argument("--max-features", type=_max_features_value, default="sqrt")
    parser.add_argument("--class-weight", type=_class_weight_value, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hyperparams = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_split": args.min_samples_split,
        "min_samples_leaf": args.min_samples_leaf,
        "max_features": args.max_features,
        "class_weight": args.class_weight,
    }
    print(f"Random Forest | dataset={args.dataset} | hyperparams={hyperparams}")

    df = load_dataset(args.dataset)
    X_train, X_test, y_train, y_test = split_holdout(df)
    print(
        f"Held out {len(X_test):,} rows for test (D6), {len(X_train):,} "
        f"available for CV/final fit."
    )

    print(
        f"Running {args.n_folds}-fold stratified CV over the train+val pool "
        f"(--n-jobs={args.n_jobs}, folds run concurrently — "
        "per-fold lines print together once all folds finish, not live)..."
    )
    cv_scores = cross_validate(X_train, y_train, args)
    train_roc_summary = summarize_scores(cv_scores["train_roc_auc"])
    train_pr_summary = summarize_scores(cv_scores["train_pr_auc"])
    roc_summary = summarize_scores(cv_scores["roc_auc"])
    pr_summary = summarize_scores(cv_scores["pr_auc"])
    print(
        f"Train ROC-AUC: {train_roc_summary['mean']:.4f} +/- "
        f"{train_roc_summary['ci95_half_width']:.4f}  |  "
        f"Val (CV) ROC-AUC: {roc_summary['mean']:.4f} +/- "
        f"{roc_summary['ci95_half_width']:.4f}  |  "
        f"gap: {train_roc_summary['mean'] - roc_summary['mean']:.4f}"
    )
    print(
        f"Train PR-AUC:  {train_pr_summary['mean']:.4f} +/- "
        f"{train_pr_summary['ci95_half_width']:.4f}  |  "
        f"Val (CV) PR-AUC:  {pr_summary['mean']:.4f} +/- "
        f"{pr_summary['ci95_half_width']:.4f}  |  "
        f"gap: {train_pr_summary['mean'] - pr_summary['mean']:.4f}"
    )

    def _log_row(
        calibration: str | None,
        test_metrics: dict[str, float] | None,
        test_cis: dict[str, tuple[float, float]] | None,
        fig_path: Path | None,
    ) -> dict:
        def _ci_json(name: str) -> str | None:
            return json.dumps(list(test_cis[name])) if test_cis else None

        return {
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "model": "random_forest",
            "dataset": args.dataset,
            "n_folds": args.n_folds,
            "final_eval": args.final,
            "calibration": calibration,
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
            "test_roc_auc": test_metrics.get("roc_auc") if test_metrics else None,
            "test_pr_auc": test_metrics.get("pr_auc") if test_metrics else None,
            "test_brier": test_metrics.get("brier") if test_metrics else None,
            "test_log_loss": test_metrics.get("log_loss") if test_metrics else None,
            "test_roc_auc_ci95": _ci_json("roc_auc"),
            "test_pr_auc_ci95": _ci_json("pr_auc"),
            "test_brier_ci95": _ci_json("brier"),
            "test_log_loss_ci95": _ci_json("log_loss"),
            "reliability_plot": str(fig_path) if fig_path else None,
        }

    if args.final:
        print(
            "Fitting on the full train+val pool and evaluating the "
            "held-out test set (once) ..."
        )
        variants, cis, fig_path = evaluate_holdout(
            X_train, y_train, X_test, y_test, args
        )
        for name, metrics in variants.items():
            roc_lo, roc_hi = cis[name]["roc_auc"]
            pr_lo, pr_hi = cis[name]["pr_auc"]
            print(
                f"[{_CALIBRATION_LABELS[name]}] "
                f"Test ROC-AUC={metrics['roc_auc']:.4f} "
                f"[{roc_lo:.4f}, {roc_hi:.4f}]  "
                f"PR-AUC={metrics['pr_auc']:.4f} [{pr_lo:.4f}, {pr_hi:.4f}]  "
                f"Brier={metrics['brier']:.4f}  LogLoss={metrics['log_loss']:.4f}  "
                f"(95% bootstrap CI, {N_BOOTSTRAP} resamples of the test set, D30)"
            )
            append_run_log(_log_row(name, metrics, cis[name], fig_path))
        print(f"Reliability curve saved to {fig_path}")
        print(f"{len(variants)} run(s) appended to {RUN_LOG_PATH}")
    else:
        print(
            "Skipping the held-out test set "
            "(pass --final once you've decided on values)."
        )
        append_run_log(_log_row(None, None, None, None))
        print(f"Run appended to {RUN_LOG_PATH}")


if __name__ == "__main__":
    main()
