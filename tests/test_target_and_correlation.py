import pandas as pd
import pytest

from scripts.target_and_correlation import (
    document_flags_summary,
    documents_provided_count,
    housing_block_correlation,
    plot_histogram,
    tail_percentiles,
    target_rate_by_value,
)


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TARGET": [0, 1, 0, 0, 1, 0],
            "FLAG_MOBIL": [1, 1, 1, 1, 1, 0],
            "FLAG_DOCUMENT_2": [0, 0, 0, 0, 1, 0],
            "FLAG_DOCUMENT_3": [1, 1, 0, 1, 1, 0],
            "APARTMENTS_AVG": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "APARTMENTS_MODE": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "APARTMENTS_MEDI": [0.15, 0.25, 0.25, 0.45, 0.55, 0.65],
            "AMT_INCOME_TOTAL": [100.0, 200.0, 150.0, 300.0, 250.0, 1000000.0],
            "OBS_30_CNT_SOCIAL_CIRCLE": [0.0, 1.0, 2.0, 0.0, 3.0, 50.0],
        }
    )


def test_target_rate_by_value() -> None:
    result = target_rate_by_value(_sample_df(), "FLAG_MOBIL")

    assert result.loc[0, "n"] == 1
    assert result.loc[0, "target_rate"] == 0.0
    assert result.loc[1, "n"] == 5
    assert result.loc[1, "target_rate"] == pytest.approx(0.4)


def test_document_flags_summary_reports_prevalence_and_rate() -> None:
    result = document_flags_summary(
        _sample_df(), ["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3"]
    )

    assert result.loc["FLAG_DOCUMENT_2", "n_provided"] == 1
    assert result.loc["FLAG_DOCUMENT_2", "target_rate_when_provided"] == pytest.approx(
        1.0
    )
    assert result.loc["FLAG_DOCUMENT_3", "n_provided"] == 4
    assert result.loc["FLAG_DOCUMENT_3", "target_rate_when_provided"] == pytest.approx(
        0.5
    )


def test_documents_provided_count_sums_correctly() -> None:
    result = documents_provided_count(
        _sample_df(), ["FLAG_DOCUMENT_2", "FLAG_DOCUMENT_3"]
    )

    assert list(result) == [1, 1, 0, 1, 2, 0]


def test_housing_block_correlation_computes_expected_pairs() -> None:
    df = _sample_df()
    result = housing_block_correlation(df, ["APARTMENTS"])

    expected_avg_mode = df["APARTMENTS_AVG"].corr(df["APARTMENTS_MODE"])
    expected_avg_medi = df["APARTMENTS_AVG"].corr(df["APARTMENTS_MEDI"])
    assert result.loc["APARTMENTS", "avg_mode"] == pytest.approx(expected_avg_mode)
    assert result.loc["APARTMENTS", "avg_medi"] == pytest.approx(expected_avg_medi)


def test_housing_block_correlation_skips_missing_attribute() -> None:
    result = housing_block_correlation(_sample_df(), ["NOT_A_REAL_ATTRIBUTE"])

    assert result.empty


def test_tail_percentiles_returns_expected_quantiles() -> None:
    series = pd.Series(range(1, 101))
    result = tail_percentiles(series)

    assert list(result.index) == [0.90, 0.95, 0.99, 0.999, 1.0]
    assert result.loc[1.0] == series.max()


def test_plot_histogram_creates_file(tmp_path) -> None:
    output_path = tmp_path / "hist.png"
    plot_histogram(pd.Series([1, 2, 2, 3, 3, 3]), output_path, title="test")

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_histogram_log_x_creates_file(tmp_path) -> None:
    output_path = tmp_path / "hist_log.png"
    series = pd.Series([1.0, 10.0, 100.0, 1000.0, 1_000_000.0])
    plot_histogram(series, output_path, title="test", log_x=True)

    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_plot_histogram_xlim_excludes_values_outside_range(tmp_path) -> None:
    output_path = tmp_path / "hist_clipped.png"
    series = pd.Series([1.0, 2.0, 3.0, 1_000_000.0])
    # Should not raise even though 1_000_000 falls outside xlim.
    plot_histogram(series, output_path, title="test", xlim=(0, 10))

    assert output_path.exists()
    assert output_path.stat().st_size > 0
