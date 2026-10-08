"""LightGBM benchmark — SPEC.md Plan step 4.

Same CV/held-out-test protocol as train_random_forest.py (D6/D16: fixed
RANDOM_STATE/TEST_SIZE, --final gates the one-time held-out test
evaluation) but for LightGBM, tuned against average_precision (PR-AUC)
per D4, not roc_auc. Uses the same StratifiedKFold(n_splits, shuffle=True,
random_state=RANDOM_STATE) as train_random_forest.py on the same
row-order-identical tables (D6/D22) — fold assignments are identical
across scripts for the same dataset choice, so CV results are directly
comparable fold-for-fold, not just by their means.

On the "treated" dataset, every remaining object-dtype column
(NAME_CONTRACT_TYPE, CODE_GENDER, OCCUPATION_TYPE, ORGANIZATION_TYPE, and
others — see clean_features.py, D15) is cast to pandas "category" dtype
at load time here, which LightGBM's sklearn wrapper auto-detects and
splits on natively, instead of one-hot or ordinal encoding. This cast is
a no-op on the "raw" dataset, which has no object columns left
(raw_baseline.py already label-encoded them to plain integers — no
native categorical treatment there, on purpose, matching "raw"'s
deliberately naive baseline).

The unexplained-group NaNs (EXT_SOURCE_1/2/3, AMT_REQ_CREDIT_BUREAU_*,
social-circle residual) are left as real NaN and handled by LightGBM's
native missing-value split direction — no imputation happens in this
script, unlike train_random_forest.py's RF-treated table.

D23 — early stopping (--early-stopping-rounds): when set, --n-estimators
becomes a ceiling, not the actual tree count. Each fit carves its own
early-stopping holdout from its *own training portion only* (never from
the fold's validation split, or — under --final — from the held-out test
split): early-stopping on the same data a fold is scored on would choose
the tree count using the data being used to judge it, biasing that score
optimistically. This is the single most common leakage bug in LightGBM
CV pipelines and the reason this holdout exists as its own carve-out at
all, rather than reusing an existing split.

D22 — calibration (--check-calibration during CV, --calibration under
--final): decided by CV, never by comparing on the held-out test set
(unlike the RF version of this feature, D21 — see D22's DECISIONS.md
entry for why that matters and why RF's approach is being revisited).
The same early-stopping holdout above doubles as the calibration-fit
set when both are requested together — a slice already unseen by the
boosted trees is exactly what isotonic calibration needs too, so this
avoids a second carve-out out of the same pool.

D24 — feature importance (--final only): gain (total loss reduction
from a feature's splits), not the sklearn-API default of split count,
saved as both a CSV and a bar chart.

Usage:
    uv run python scripts/train_lightgbm.py
    uv run python scripts/train_lightgbm.py --dataset raw --n-folds 10
    uv run python scripts/train_lightgbm.py --learning-rate 0.05 \
        --n-estimators 2000 --early-stopping-rounds 50 --check-calibration
    uv run python scripts/train_lightgbm.py --learning-rate 0.05 \
        --n-estimators 2000 --early-stopping-rounds 50 \
        --calibration isotonic --final
"""

import argparse
import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
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
TREATED_PATH = PROCESSED_DIR / "train_treated_lightgbm.csv"

# D6/D16: fixed, not CLI-overridable — matches train_random_forest.py
# exactly, so both scripts hold out the same rows on either dataset.
RANDOM_STATE = 42
TEST_SIZE = 0.15
TARGET_COL = "TARGET"
ID_COL = "SK_ID_CURR"

# D30: nonparametric bootstrap for a CI on each test-set metric reported
# under --final — see train_random_forest.py's N_BOOTSTRAP/BOOTSTRAP_CI
# comment for the full rationale; identical here by design, so the two
# scripts' test-set CIs are computed the same way.
N_BOOTSTRAP = 1000
BOOTSTRAP_CI = 0.95

# D23/D22: fraction of a fit's *own training portion* reserved for early
# stopping's eval set and/or isotonic calibration's fit set — never for
# CV scoring or the held-out test split. Fixed like TEST_SIZE, not a CLI
# flag: this is a protocol detail, not a model hyperparameter to iterate.
HOLDOUT_FRACTION = 0.15

# D22: threshold for "isotonic calibration meaningfully helps", reusing
# the same ~0.005 magnitude the user set for comparing RF vs LightGBM CV
# means — applied here to mean out-of-fold Brier improvement instead.
# The OR is deliberate: either a mean improvement past this threshold, or
# an improvement in every single fold (small but consistent) counts.
CALIBRATION_IMPROVEMENT_THRESHOLD = 0.005

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
    # D26: raw per-fold validation scores (JSON list of length n_folds) —
    # see train_random_forest.py's RUN_LOG_COLUMNS for why (paired
    # fold-by-fold comparison, which a mean/CI pair alone can't support).
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


def cast_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Cast every remaining object-dtype column to pandas "category"
    dtype so LightGBM's sklearn wrapper auto-detects and splits on it
    natively (D15). A no-op when there are none (the raw dataset).
    """
    out = df.copy()
    # Explicit ["object", "str"]: pandas currently includes "str"-dtype
    # columns under include="object" too, but warns that this backward-
    # compat behavior is being removed — listing both silences the
    # warning now and keeps catching the same columns once that default
    # actually changes upstream.
    for col in out.select_dtypes(include=["object", "str"]).columns:
        out[col] = out[col].astype("category")
    return out


def load_dataset(dataset: str) -> pd.DataFrame:
    """Load the chosen D10 cell (raw or LightGBM-treated), with the
    category-dtype cast applied. Duplicated load-by-path pattern rather
    than imported — see clean_features.py's load_train docstring.
    """
    path = RAW_PATH if dataset == "raw" else TREATED_PATH
    return cast_categoricals(pd.read_csv(path))


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


def build_model(args: argparse.Namespace, n_jobs: int = -1) -> LGBMClassifier:
    return LGBMClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=args.subsample,
        subsample_freq=args.subsample_freq,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        is_unbalance=args.is_unbalance,
        random_state=RANDOM_STATE,
        n_jobs=n_jobs,
        verbosity=-1,
    )


def _fit_with_optional_early_stopping(
    model: LGBMClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    idx: np.ndarray,
    args: argparse.Namespace,
    extra_holdout_needed: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Fit `model` on `idx`, carving a HOLDOUT_FRACTION holdout out of
    `idx` itself (never touching whatever the caller uses for scoring)
    when early stopping is on and/or the caller needs a holdout for its
    own reason (`extra_holdout_needed` — isotonic calibration, from
    either --check-calibration during CV or --calibration under --final;
    the caller decides which flag is relevant to it, this function only
    needs to know whether a holdout is wanted). Returns (fit_idx,
    holdout_idx_or_None) — `fit_idx` is what was actually trained on,
    needed by the caller to score "train" metrics on the right rows once
    early stopping shrinks that set.
    """
    needs_holdout = args.early_stopping_rounds is not None or extra_holdout_needed
    if not needs_holdout:
        model.fit(X.iloc[idx], y.iloc[idx])
        return idx, None

    fit_idx, holdout_idx = train_test_split(
        idx,
        test_size=HOLDOUT_FRACTION,
        stratify=y.iloc[idx],
        random_state=RANDOM_STATE,
    )
    if args.early_stopping_rounds is not None:
        model.fit(
            X.iloc[fit_idx],
            y.iloc[fit_idx],
            eval_X=X.iloc[holdout_idx],
            eval_y=y.iloc[holdout_idx],
            eval_metric="average_precision",
            callbacks=[
                lgb.early_stopping(
                    stopping_rounds=args.early_stopping_rounds, verbose=False
                )
            ],
        )
    else:
        model.fit(X.iloc[fit_idx], y.iloc[fit_idx])
    return fit_idx, holdout_idx


class FoldResult(NamedTuple):
    train_roc: float
    train_pr: float
    val_roc: float
    val_pr: float
    best_iteration: int
    raw_brier: float
    raw_log_loss: float
    cal_brier: float | None
    cal_log_loss: float | None


def _fit_and_score_fold(
    X: pd.DataFrame,
    y: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    args: argparse.Namespace,
) -> FoldResult:
    """Fit and score one fold in isolation — see
    train_random_forest.py's _fit_and_score_fold for why n_jobs=1 here
    (folds are the parallel unit, not each model's own boosting threads).

    When --check-calibration is set, also fits an isotonic-calibrated
    variant on this fold's own holdout (CalibratedClassifierCV, cv=
    "prefit" — the boosted model is not refit, so this is a single
    coherent calibrated model, not an ensemble) and scores it on the
    fold's val split too, entirely separate from the raw model's score.
    """
    model = build_model(args, n_jobs=1)
    fit_idx, holdout_idx = _fit_with_optional_early_stopping(
        model,
        X,
        y,
        train_idx,
        args,
        extra_holdout_needed=getattr(args, "check_calibration", False),
    )
    best_iteration = model.n_estimators_

    train_proba = model.predict_proba(X.iloc[fit_idx])[:, 1]
    train_roc = roc_auc_score(y.iloc[fit_idx], train_proba)
    train_pr = average_precision_score(y.iloc[fit_idx], train_proba)

    val_proba = model.predict_proba(X.iloc[val_idx])[:, 1]
    val_roc = roc_auc_score(y.iloc[val_idx], val_proba)
    val_pr = average_precision_score(y.iloc[val_idx], val_proba)
    raw_brier = brier_score_loss(y.iloc[val_idx], val_proba)
    raw_log_loss = log_loss(y.iloc[val_idx], val_proba)

    cal_brier = cal_log_loss = None
    if getattr(args, "check_calibration", False):
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method="isotonic"
        )
        calibrated.fit(X.iloc[holdout_idx], y.iloc[holdout_idx])
        cal_proba = calibrated.predict_proba(X.iloc[val_idx])[:, 1]
        cal_brier = brier_score_loss(y.iloc[val_idx], cal_proba)
        cal_log_loss = log_loss(y.iloc[val_idx], cal_proba)

    return FoldResult(
        train_roc,
        train_pr,
        val_roc,
        val_pr,
        best_iteration,
        raw_brier,
        raw_log_loss,
        cal_brier,
        cal_log_loss,
    )


def cross_validate(
    X: pd.DataFrame, y: pd.Series, args: argparse.Namespace
) -> dict[str, np.ndarray]:
    """Stratified k-fold CV over the train+val pool, tuned/reported
    against PR-AUC per D4 (ROC-AUC still reported alongside it, matching
    SPEC.md's Success criteria, but PR-AUC is the metric that matters
    here). Folds are fit concurrently (joblib, one process per fold,
    capped at --n-jobs) — see train_random_forest.py's cross_validate for
    why that's free correctness-wise.

    Scores both the fold's own training data and its validation data —
    the gap is a diagnostic only, never a selection criterion (D20/D22):
    boosted trees fit residuals stagewise and train AUC/PR-AUC rises
    with every additional tree by construction, so — unlike RF, where a
    gap is expected ensemble behavior — a large or growing gap here
    usually does mean too many boosting rounds or too little
    regularization. Either way, what decides a hyperparameter choice is
    the validation score alone, never the gap itself, for both models.
    """
    skf = StratifiedKFold(
        n_splits=args.n_folds, shuffle=True, random_state=RANDOM_STATE
    )
    splits = list(skf.split(X, y))
    results: list[FoldResult] = Parallel(n_jobs=args.n_jobs)(
        delayed(_fit_and_score_fold)(X, y, train_idx, val_idx, args)
        for train_idx, val_idx in splits
    )

    train_roc_scores = np.array([r.train_roc for r in results])
    train_pr_scores = np.array([r.train_pr for r in results])
    val_roc_scores = np.array([r.val_roc for r in results])
    val_pr_scores = np.array([r.val_pr for r in results])
    best_iterations = np.array([r.best_iteration for r in results])

    for fold in range(args.n_folds):
        r = results[fold]
        suffix = (
            f"  |  trees={r.best_iteration}"
            if args.early_stopping_rounds is not None
            else ""
        )
        print(
            f"  fold {fold + 1}/{args.n_folds}: "
            f"train ROC-AUC={r.train_roc:.4f} PR-AUC={r.train_pr:.4f}  |  "
            f"val ROC-AUC={r.val_roc:.4f} PR-AUC={r.val_pr:.4f}{suffix}"
        )

    if args.early_stopping_rounds is not None:
        print(
            f"  trees used per fold: mean={best_iterations.mean():.0f}  "
            f"min={best_iterations.min()}  max={best_iterations.max()}  "
            f"(ceiling was --n-estimators={args.n_estimators})"
        )

    out = {
        "train_roc_auc": train_roc_scores,
        "train_pr_auc": train_pr_scores,
        "roc_auc": val_roc_scores,
        "pr_auc": val_pr_scores,
        "best_iteration": best_iterations,
    }

    if args.check_calibration:
        raw_brier = np.array([r.raw_brier for r in results])
        raw_ll = np.array([r.raw_log_loss for r in results])
        cal_brier = np.array([r.cal_brier for r in results])
        cal_ll = np.array([r.cal_log_loss for r in results])
        brier_diff = raw_brier - cal_brier  # positive => calibration helped
        ll_diff = raw_ll - cal_ll
        folds_improved_brier = int((brier_diff > 0).sum())
        verdict = (
            brier_diff.mean() > CALIBRATION_IMPROVEMENT_THRESHOLD
            or folds_improved_brier == args.n_folds
        )
        print(
            f"  isotonic calibration check (out-of-fold, never touching "
            f"test): Brier improved in {folds_improved_brier}/{args.n_folds} "
            f"folds (mean delta={brier_diff.mean():+.6f})  |  "
            f"LogLoss mean delta={ll_diff.mean():+.6f}"
        )
        print(
            "  sign convention: delta = raw - calibrated. Positive means "
            "calibration lowered the loss (helped); negative means it "
            "raised the loss (hurt). 'N/5 improved' counts folds with "
            "delta > 0, so 0/5 improved forces every fold's delta <= 0, "
            "which forces the mean <= 0 too -- a negative mean alongside "
            "0/5 improved is required by this arithmetic, not a conflict."
        )
        print(
            f"  per-fold Brier delta: "
            f"{[f'{v:+.6f}' for v in brier_diff]}  |  "
            f"per-fold LogLoss delta: {[f'{v:+.6f}' for v in ll_diff]}"
        )
        print(
            f"  verdict: {'APPLY' if verdict else 'DO NOT APPLY'} isotonic "
            f"(rule: mean Brier delta > {CALIBRATION_IMPROVEMENT_THRESHOLD} "
            "OR improved in every fold)"
        )
        out["raw_brier"] = raw_brier
        out["raw_log_loss"] = raw_ll
        out["cal_brier"] = cal_brier
        out["cal_log_loss"] = cal_ll

    return out


def summarize_scores(scores: np.ndarray) -> dict[str, float]:
    """Mean, std, and a 95% CI half-width via the t-distribution — see
    train_random_forest.py's summarize_scores for why not a normal
    approximation.
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
    """Reliability curve using quantile (equal-count) bins — see
    train_random_forest.py's _quantile_reliability_curve (D21) for the
    full rationale; duplicated here rather than imported since, unlike
    cross_validate, this is small and stable, matching this project's
    default self-contained-script convention.
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
    """Nonparametric (percentile) bootstrap CI for one test-set metric —
    identical to train_random_forest.py's version (D30): resamples
    (y_true, y_prob) with replacement, same size as the test set,
    `n_boot` times, recomputes `metric_fn` on each resample, and returns
    the `ci`-percentile interval. A resample with only one class is
    redrawn, not counted (undefined for roc_auc/pr_auc; astronomically
    unlikely at this test set's ~8% base rate and ~46k rows, but guarded
    rather than assumed).
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
    metrics — see _bootstrap_ci's docstring for what the CI does and
    doesn't mean.
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


_CALIBRATION_LABELS = {"none": "Uncalibrated LightGBM", "isotonic": "Isotonic"}


def save_feature_importance(
    model: LGBMClassifier, dataset: str, top_n: int = 20
) -> tuple[Path, Path]:
    """Gain importance (total loss reduction from a feature's splits) —
    not the sklearn-API default of split count, which just measures how
    often a feature was used, not how much it actually helped (D24).
    """
    booster = model.booster_
    gains = booster.feature_importance(importance_type="gain")
    names = booster.feature_name()
    df = pd.DataFrame({"feature": names, "gain": gains})
    df = df.sort_values("gain", ascending=False).reset_index(drop=True)
    total = df["gain"].sum()
    df["gain_pct"] = 100 * df["gain"] / total if total > 0 else 0.0

    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / f"feature_importance_lightgbm_{dataset}_{timestamp}.csv"
    df.to_csv(csv_path, index=False)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = FIGURES_DIR / f"feature_importance_lightgbm_{dataset}_{timestamp}.png"
    top = df.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(top) + 1))
    ax.barh(top["feature"], top["gain"])
    ax.set_xlabel("Total gain (loss reduction from this feature's splits)")
    ax.set_title(f"LightGBM feature importance — gain ({dataset}, top {len(top)})")
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    return csv_path, fig_path


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
    held-out test split (D16). Only called under --final.

    --calibration isotonic wraps the SAME already-fit model with
    CalibratedClassifierCV(estimator=FrozenEstimator(model)) on a
    dedicated holdout carved out of the train+val pool (the same slice
    early stopping's eval set uses,
    when both are on) — one coherent calibrated model, not an ensemble of
    refits (contrast train_random_forest.py's cv=5 approach, D21/D22).
    Which method to apply here is a decision --check-calibration already
    made via CV before this ever runs — not decided by comparing the
    "none" vs "isotonic" rows this function returns (D22). Each variant's
    test metrics get a bootstrap 95% CI too (D30).
    """
    X_train_r = X_train.reset_index(drop=True)
    y_train_r = y_train.reset_index(drop=True)
    all_idx = np.arange(len(X_train_r))

    model = build_model(args)
    fit_idx, holdout_idx = _fit_with_optional_early_stopping(
        model,
        X_train_r,
        y_train_r,
        all_idx,
        args,
        extra_holdout_needed=(args.calibration != "none"),
    )
    if args.early_stopping_rounds is not None:
        print(
            f"  early stopping: best_iteration={model.n_estimators_} "
            f"(ceiling was --n-estimators={args.n_estimators})"
        )

    variants: dict[str, dict[str, float]] = {}
    cis: dict[str, dict[str, tuple[float, float]]] = {}
    curves: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    raw_proba = model.predict_proba(X_test)[:, 1]
    variants["none"], cis["none"], *curve = _test_metrics_and_curve(y_test, raw_proba)
    curves["none"] = tuple(curve)

    if args.calibration == "isotonic":
        holdout_X = X_train_r.iloc[holdout_idx]
        holdout_y = y_train_r.iloc[holdout_idx]
        print(
            "  fitting CalibratedClassifierCV(estimator=FrozenEstimator("
            "model), method='isotonic') on the early-stopping/calibration "
            "holdout (never used to train the boosted trees)..."
        )
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(model), method="isotonic"
        )
        calibrated.fit(holdout_X, holdout_y)
        cal_proba = calibrated.predict_proba(X_test)[:, 1]
        variants["isotonic"], cis["isotonic"], *curve = _test_metrics_and_curve(
            y_test, cal_proba
        )
        curves["isotonic"] = tuple(curve)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    fig_path = FIGURES_DIR / f"reliability_lightgbm_{args.dataset}_{timestamp}.png"
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
    ax.set_title(f"Reliability curve — LightGBM ({args.dataset}, held-out test)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    fi_csv, fi_fig = save_feature_importance(model, args.dataset)
    print(f"Feature importance (gain) saved to {fi_csv} and {fi_fig}")

    return variants, cis, fig_path


def append_run_log(row: dict, path: Path = RUN_LOG_PATH) -> None:
    """Append one run's results to the shared model-comparison log
    (shared with train_random_forest.py), creating it with a fixed
    column order if it doesn't exist yet. Reindexed to RUN_LOG_COLUMNS
    on every write — see train_random_forest.py's append_run_log for why.
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
        "parallelism (default -1 = all cores). Each fold's own model "
        "runs single-threaded regardless — this flag is the only knob "
        "for how many folds run at once.",
    )
    parser.add_argument("--final", action="store_true")
    parser.add_argument(
        "--calibration",
        choices=["none", "isotonic"],
        default="none",
        help="Only used under --final. Applies CalibratedClassifierCV"
        "(method='isotonic', cv='prefit') on a dedicated holdout (D22). "
        "Sigmoid is not offered here — LightGBM's own probabilities "
        "should already be close to calibrated (it optimizes log loss "
        "directly), so isotonic is the only correction being tested. "
        "Decide this from --check-calibration's CV verdict first, not "
        "by comparing 'none' vs 'isotonic' on the test set itself.",
    )
    parser.add_argument(
        "--check-calibration",
        action="store_true",
        help="During CV (not --final): also fits an isotonic-calibrated "
        "variant per fold on that fold's own holdout and compares "
        "out-of-fold Brier/log loss against the uncalibrated model — "
        "never touches the held-out test set (D22). Adds real per-fold "
        "compute (an extra holdout carve-out + isotonic fit) and slightly "
        "shrinks the data each fold's model actually trains on, so treat "
        "this as a deliberate check once hyperparameters are close to "
        "settled, not something to leave on for every routine CV run.",
    )
    # LightGBM's own library defaults — no prior tuned run exists for
    # this project (unlike RF's D3), so these are the starting point for
    # manual iteration, not a finding.
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="A ceiling, not the actual tree count, when "
        "--early-stopping-rounds is set — see that flag.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=_int_or_none, default=-1)
    parser.add_argument("--min-child-samples", type=int, default=20)
    parser.add_argument("--subsample", type=float, default=1.0)
    parser.add_argument(
        "--subsample-freq",
        type=int,
        default=0,
        help="LightGBM's own default (0) DISABLES bagging outright, "
        "regardless of --subsample: bagging_fraction only takes effect "
        "when bagging_freq (this flag) is a positive integer. Set both "
        "together, e.g. --subsample 0.8 --subsample-freq 1.",
    )
    parser.add_argument("--colsample-bytree", type=float, default=1.0)
    parser.add_argument("--reg-alpha", type=float, default=0.0)
    parser.add_argument("--reg-lambda", type=float, default=0.0)
    parser.add_argument("--is-unbalance", action="store_true")
    parser.add_argument(
        "--early-stopping-rounds",
        type=_int_or_none,
        default=None,
        help="Enables early stopping, monitored on average_precision "
        "(PR-AUC, D4) on a per-fit holdout carved from that fit's own "
        "training portion only (D23) — set --n-estimators to a high "
        "ceiling (e.g. 2000+) when this is on, since it now decides the "
        "real tree count, not --n-estimators directly.",
    )
    args = parser.parse_args()
    if args.subsample < 1.0 and args.subsample_freq == 0:
        print(
            f"WARNING: --subsample={args.subsample} has NO effect with "
            "--subsample-freq=0 (LightGBM's own default disables bagging "
            "at freq=0) — pass --subsample-freq with a positive value too."
        )
    return args


def main() -> None:
    args = parse_args()
    hyperparams = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": args.max_depth,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "subsample_freq": args.subsample_freq,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "is_unbalance": args.is_unbalance,
        "early_stopping_rounds": args.early_stopping_rounds,
    }
    print(f"LightGBM | dataset={args.dataset} | hyperparams={hyperparams}")

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
        f"gap: {train_pr_summary['mean'] - pr_summary['mean']:.4f}  "
        "<- D4's primary metric (gap is a diagnostic only, never the "
        "selection criterion — D20/D22)"
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
            "model": "lightgbm",
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
