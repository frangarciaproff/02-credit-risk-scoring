import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.train_lightgbm import (
    RUN_LOG_COLUMNS,
    TEST_SIZE,
    _bootstrap_ci,
    _test_metrics_and_curve,
    append_run_log,
    cast_categoricals,
    split_holdout,
    summarize_scores,
)


def _sample_df(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    target = (rng.random(n) < 0.1).astype(int)
    return pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n + 1),
            "TARGET": target,
            "FEATURE_A": rng.normal(size=n),
            "OCCUPATION_TYPE": rng.choice(["Laborers", "Drivers", "Missing"], size=n),
        }
    )


def test_cast_categoricals_converts_object_columns_only() -> None:
    df = _sample_df()
    result = cast_categoricals(df)

    assert isinstance(result["OCCUPATION_TYPE"].dtype, pd.CategoricalDtype)
    assert not isinstance(result["FEATURE_A"].dtype, pd.CategoricalDtype)
    assert len(result.select_dtypes(include="object").columns) == 0


def test_cast_categoricals_is_noop_with_no_object_columns() -> None:
    df = pd.DataFrame({"A": [1, 2, 3], "B": [1.0, 2.0, 3.0]})
    result = cast_categoricals(df)

    assert result["A"].dtype == df["A"].dtype
    assert result["B"].dtype == df["B"].dtype


def test_split_holdout_respects_test_size_and_drops_id_and_target() -> None:
    df = cast_categoricals(_sample_df())
    X_train, X_test, y_train, y_test = split_holdout(df)

    assert len(X_test) == round(len(df) * TEST_SIZE)
    assert "SK_ID_CURR" not in X_train.columns
    assert "TARGET" not in X_train.columns
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)


def test_summarize_scores_zero_variance_gives_zero_ci() -> None:
    summary = summarize_scores(np.array([0.8, 0.8, 0.8]))

    assert summary["mean"] == pytest.approx(0.8)
    assert summary["ci95_half_width"] == pytest.approx(0.0, abs=1e-9)


def test_append_run_log_creates_then_appends(tmp_path) -> None:
    log_path = tmp_path / "model_runs.csv"
    row = dict.fromkeys(RUN_LOG_COLUMNS)
    row["model"] = "lightgbm"
    row["dataset"] = "treated"

    append_run_log(row, path=log_path)
    append_run_log({**row, "dataset": "raw"}, path=log_path)

    result = pd.read_csv(log_path)
    assert len(result) == 2
    assert list(result.columns) == RUN_LOG_COLUMNS


# D30: bootstrap CI regression tests — same checks as
# test_train_random_forest.py's, since this is a byte-for-byte duplicate
# of the same helper (self-contained-scripts convention).


def test_bootstrap_ci_brackets_point_estimate_and_is_ordered() -> None:
    rng = np.random.default_rng(1)
    n = 2000
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)

    point = roc_auc_score(y_true, y_prob)
    lo, hi = _bootstrap_ci(y_true, y_prob, roc_auc_score, n_boot=300)

    assert lo < hi
    assert lo <= point <= hi


def test_bootstrap_ci_is_reproducible_with_same_random_state() -> None:
    rng = np.random.default_rng(2)
    n = 500
    y_prob = rng.random(n)
    y_true = (rng.random(n) < y_prob).astype(int)

    first = _bootstrap_ci(y_true, y_prob, average_precision_score, random_state=42)
    second = _bootstrap_ci(y_true, y_prob, average_precision_score, random_state=42)
    assert first == second


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
