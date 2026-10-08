import pandas as pd

from scripts.eda import (
    cardinality_report,
    categorical_summary,
    missing_report,
    numeric_summary,
    split_by_dtype,
    write_report,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "SK_ID_CURR": [1, 2, 3, 4],
            "TARGET": [0, 1, 0, 0],
            "AMT_CREDIT": [100.0, 200.0, None, 400.0],
            "CODE_GENDER": ["M", "F", "F", "F"],
            "CONSTANT_FLAG": [1, 1, 1, 1],
        }
    )


def test_split_by_dtype_buckets_correctly() -> None:
    groups = split_by_dtype(_sample_df())

    assert set(groups["int"]) == {"SK_ID_CURR", "TARGET", "CONSTANT_FLAG"}
    assert groups["float"] == ["AMT_CREDIT"]
    assert groups["categorical"] == ["CODE_GENDER"]


def test_missing_report_only_lists_columns_with_gaps() -> None:
    report = missing_report(_sample_df())

    assert list(report.index) == ["AMT_CREDIT"]
    assert report.loc["AMT_CREDIT", "n_missing"] == 1
    assert report.loc["AMT_CREDIT", "pct_missing"] == 25.0


def test_cardinality_report_flags_constant_column() -> None:
    report = cardinality_report(_sample_df())

    assert report.loc["CONSTANT_FLAG", "nunique"] == 1
    # sorted ascending, so the constant column should be first
    assert report.index[0] == "CONSTANT_FLAG"


def test_numeric_summary_has_expected_stats() -> None:
    summary = numeric_summary(_sample_df(), ["AMT_CREDIT"])

    assert "mean" in summary.columns
    assert "50%" in summary.columns  # median
    assert summary.loc["AMT_CREDIT", "count"] == 3  # one NaN excluded


def test_categorical_summary_has_expected_stats() -> None:
    summary = categorical_summary(_sample_df(), ["CODE_GENDER"])

    assert summary.loc["CODE_GENDER", "unique"] == 2
    assert summary.loc["CODE_GENDER", "top"] == "F"
    assert summary.loc["CODE_GENDER", "freq"] == 3


def test_categorical_summary_empty_columns_returns_empty_frame() -> None:
    assert categorical_summary(_sample_df(), []).empty


def test_write_report_creates_file(tmp_path) -> None:
    output_path = tmp_path / "report.md"
    write_report(_sample_df(), output_path=output_path)

    assert output_path.exists()
    content = output_path.read_text(encoding="utf-8")
    assert "AMT_CREDIT" in content
    assert "CODE_GENDER" in content
