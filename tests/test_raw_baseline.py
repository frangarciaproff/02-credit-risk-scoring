import numpy as np
import pandas as pd

from scripts.raw_baseline import (
    build_raw_baseline,
    build_summary,
    fit_train_index,
    label_encode_categoricals,
    median_impute_numeric,
    write_outputs,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 0],
            "DAYS_EMPLOYED": [-100, 365243, -300, 365243],
            "CODE_GENDER": ["M", "F", "F", np.nan],
            "AMT_INCOME_TOTAL": [100.0, 200.0, np.nan, 400.0],
            "FLAG_DOCUMENT_3": [1, 0, 1, 1],
        }
    )


def test_label_encode_categoricals_assigns_codes_and_minus_one_for_nan() -> None:
    result = label_encode_categoricals(_sample_df())

    assert pd.api.types.is_integer_dtype(result["CODE_GENDER"].dtype)
    assert result["CODE_GENDER"].iloc[3] == -1
    assert result["CODE_GENDER"].iloc[0] != result["CODE_GENDER"].iloc[1]


def test_label_encode_categoricals_leaves_numeric_columns_untouched() -> None:
    df = _sample_df()
    result = label_encode_categoricals(df)

    assert result["DAYS_EMPLOYED"].tolist() == df["DAYS_EMPLOYED"].tolist()
    assert (result["DAYS_EMPLOYED"] == 365243).sum() == 2


def test_median_impute_numeric_fills_only_numeric_nans() -> None:
    df = _sample_df()
    result = median_impute_numeric(df)

    assert result["AMT_INCOME_TOTAL"].isna().sum() == 0
    non_missing = df["AMT_INCOME_TOTAL"].dropna()
    assert result.loc[2, "AMT_INCOME_TOTAL"] == non_missing.median()


def test_build_raw_baseline_leaves_sentinel_and_shape_untouched() -> None:
    df = _sample_df()
    result = build_raw_baseline(df)

    assert result.shape == df.shape
    assert (result["DAYS_EMPLOYED"] == 365243).sum() == 2
    assert result.isna().sum().sum() == 0


def test_build_summary_reports_encoded_columns_and_sentinel_count() -> None:
    df = _sample_df()
    baseline = build_raw_baseline(df)
    summary = build_summary(df, baseline)

    assert "CODE_GENDER" in summary
    assert "raw shape" in summary
    assert "365243" in summary


def test_write_outputs_creates_csv_and_summary(tmp_path) -> None:
    df = _sample_df()
    baseline = build_raw_baseline(df)
    output_path = tmp_path / "train_raw.csv"
    summary_path = tmp_path / "raw_baseline_summary.md"

    write_outputs(df, baseline, output_path=output_path, summary_path=summary_path)

    assert output_path.exists()
    assert summary_path.exists()
    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == len(baseline)


# D29: regression tests for the leak fix.


def test_median_impute_numeric_respects_train_index() -> None:
    df = _sample_df()
    train_index = pd.Index([0, 1])  # excludes row 2 (NaN) and row 3 (400.0)
    result = median_impute_numeric(df, train_index=train_index)

    train_only_median = df.loc[train_index, "AMT_INCOME_TOTAL"].median()
    full_df_median = df["AMT_INCOME_TOTAL"].dropna().median()
    assert train_only_median != full_df_median
    assert result.loc[2, "AMT_INCOME_TOTAL"] == train_only_median


def test_fit_train_index_matches_downstream_split_holdout_partition() -> None:
    """Same invariant clean_features.py's version of this test checks:
    the partition computed here must match split_holdout()'s own split,
    computed later on the already-written CSV, row for row.
    """
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "SK_ID_CURR": range(1, n + 1),
            "TARGET": (rng.random(n) < 0.1).astype(int),
            "FEATURE_A": rng.normal(size=n),
        }
    )
    train_index = fit_train_index(df)

    from sklearn.model_selection import train_test_split

    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    assert set(train_index) == set(X_train.index)
