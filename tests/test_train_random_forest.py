import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.train_random_forest import (
    RUN_LOG_COLUMNS,
    TEST_SIZE,
    _bootstrap_ci,
    _class_weight_value,
    _int_or_none,
    _max_features_value,
    _quantile_reliability_curve,
    _test_metrics_and_curve,
    append_run_log,
    split_holdout,
    summarize_scores,
)


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    target = (rng.random(n) < 0.1).astype(int)  # imbalanced, like the real TARGET
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n + 1),
            "TARGET": target,
            "FEATURE_A": rng.normal(size=n),
            "FEATURE_B": rng.integers(0, 5, size=n),
        }
    )


def test_split_holdout_respects_test_size_and_drops_id_and_target() -> None:
    df = _sample_df()
    X_train, X_test, y_train, y_test = split_holdout(df)

    assert len(X_test) == round(len(df) * TEST_SIZE)
    assert len(X_train) + len(X_test) == len(df)
    assert "SK_ID_CURR" not in X_train.columns
    assert "TARGET" not in X_train.columns
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_split_holdout_is_stratified() -> None:
    df = _sample_df(n=2000)
    _, _, y_train, y_test = split_holdout(df)

    overall_rate = df["TARGET"].mean()
    # Stratified split keeps the positive rate close across splits.
    assert abs(y_train.mean() - overall_rate) < 0.03
    assert abs(y_test.mean() - overall_rate) < 0.03


def test_split_holdout_is_deterministic_across_calls() -> None:
    df = _sample_df()
    _, X_test_1, _, _ = split_holdout(df)
    _, X_test_2, _, _ = split_holdout(df)

    assert X_test_1.index.tolist() == X_test_2.index.tolist()


def test_summarize_scores_zero_variance_gives_zero_ci() -> None:
    scores = np.array([0.8, 0.8, 0.8, 0.8, 0.8])
    summary = summarize_scores(scores)

    assert summary["mean"] == pytest.approx(0.8)
    assert summary["std"] == pytest.approx(0.0, abs=1e-9)
    assert summary["ci95_half_width"] == pytest.approx(0.0, abs=1e-9)


def test_summarize_scores_single_value_has_nan_ci() -> None:
    summary = summarize_scores(np.array([0.75]))

    assert summary["mean"] == 0.75
    assert summary["std"] == 0.0
    assert np.isnan(summary["ci95_half_width"])


def test_summarize_scores_ci_shrinks_with_more_folds_same_spread() -> None:
    # Same std, more observations -> narrower CI (sqrt(n) in the denominator).
    few = summarize_scores(np.array([0.70, 0.74, 0.78]))
    many = summarize_scores(np.array([0.70, 0.74, 0.78, 0.70, 0.74, 0.78]))

    assert many["ci95_half_width"] < few["ci95_half_width"]


def test_int_or_none_parses_none_and_integers() -> None:
    assert _int_or_none("None") is None
    assert _int_or_none("none") is None
    assert _int_or_none("15") == 15


def test_max_features_value_parses_none_float_and_string() -> None:
    assert _max_features_value("none") is None
    assert _max_features_value("0.5") == 0.5
    assert isinstance(_max_features_value("0.5"), float)
    assert _max_features_value("sqrt") == "sqrt"


def test_max_features_value_parses_int_as_int_not_float() -> None:
    # Regression: sklearn's max_features takes an int as an absolute
    # feature count, but only a float in (0.0, 1.0] as a fraction.
    # float("9") == 9.0 fails that range check outright, so whole-number
    # strings must parse as int, not float.
    result = _max_features_value("9")
    assert result == 9
    assert isinstance(result, int)


def test_class_weight_value_parses_none_and_string() -> None:
    assert _class_weight_value("None") is None
    assert _class_weight_value("balanced") == "balanced"


def test_quantile_reliability_curve_bins_have_near_equal_counts() -> None:
    rng = np.random.default_rng(0)
    n = 1000
    y_prob = rng.random(n)  # continuous, no ties -> 10 clean bins
    y_true = (rng.random(n) < y_prob).astype(int)

    mean_pred, observed, counts = _quantile_reliability_curve(y_true, y_prob, n_bins=10)

    assert len(mean_pred) == len(observed) == len(counts) == 10
    assert counts.sum() == n
    # Quantile bins: no bin should be far off from the even n/10 split.
    assert counts.min() >= n // 10 - 5
    assert counts.max() <= n // 10 + 5
    # Bins are returned in ascending predicted-probability order.
    assert (np.diff(mean_pred) >= 0).all()


def test_quantile_reliability_curve_collapses_bins_under_heavy_ties() -> None:
    # RF's proba output is often heavily tied at the low end (many trees
    # agreeing on the same leaf average) -> fewer than n_bins quantile
    # edges are distinct, so bins collapse instead of erroring.
    y_prob = np.array([0.05] * 900 + [0.1, 0.2, 0.3, 0.4, 0.5] * 20)
    y_true = np.zeros_like(y_prob, dtype=int)
    y_true[-5:] = 1

    mean_pred, observed, counts = _quantile_reliability_curve(y_true, y_prob, n_bins=10)

    assert len(mean_pred) < 10
    assert counts.sum() == len(y_prob)


def test_append_run_log_creates_then_appends(tmp_path) -> None:
    log_path = tmp_path / "model_runs.csv"
    row = dict.fromkeys(RUN_LOG_COLUMNS)
    row["model"] = "random_forest"
    row["dataset"] = "treated"

    append_run_log(row, path=log_path)
    append_run_log({**row, "dataset": "raw"}, path=log_path)

    result = pd.read_csv(log_path)
    assert len(result) == 2
    assert list(result.columns) == RUN_LOG_COLUMNS
    assert result["dataset"].tolist() == ["treated", "raw"]


# D30: bootstrap CI regression tests.


def test_bootstrap_ci_brackets_point_estimate_and_is_ordered() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)  # genuine signal, not noise

    point = roc_auc_score(y_true, y_prob)
    lo, hi = _bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=300)

    assert lo < hi
    # Not a strict guarantee for every bootstrap, but true for a 95% CI
    # around a stable metric on 2000 points in the overwhelming majority
    # of draws — a seeded test, so this is deterministic, not flaky.
    assert lo <= point <= hi


def test_bootstrap_ci_is_reproducible_with_same_random_state() -> None:
    rng = np.random.default_rng(2)
    n = 500
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)

    first = _bootstrap_ci(y_true, y_prob, average_precision_score, random_state=42)
    second = _bootstrap_ci(y_true, y_prob, average_precision_score, random_state=42)
    assert first == second


def test_bootstrap_ci_widens_on_a_smaller_test_set() -> None:
    """A CI from fewer rows should be no narrower than one from more rows
    drawn the same way — the whole reason to report a CI instead of a
    bare point estimate.
    """
    rng = np.random.default_rng(3)
    y_prob_big = rng.random(5000)
    y_true_big = (rng.random(5000) < y_prob_big).astype(int)
    y_prob_small = y_prob_big[:200]
    y_true_small = y_true_big[:200]

    lo_big, hi_big = _bootstrap_ci(y_true_big, y_prob_big, roc_auc_score, n_boot=300)
    lo_small, hi_small = _bootstrap_ci(
        y_true_small, y_prob_small, roc_auc_score, n_boot=300
    )
    assert (hi_small - lo_small) > (hi_big - lo_big)


def test_test_metrics_and_curve_returns_a_ci_per_metric() -> None:
    rng = np.random.default_rng(4)
    n = 500
    y_prob = rng.random(n)
    y_true = pd.Series((rng.random(n) < y_prob).astype(int))

    metrics, cis, mean_pred, observed, counts = _test_metrics_and_curve(y_true, y_prob)

    assert set(cis) == {"roc_auc", "pr_auc", "brier", "log_loss"}
    for name in metrics:
        lo, hi = cis[name]
        assert lo <= hi
