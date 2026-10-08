"""Target-relationship and redundancy checks, following up on
results/pre_process_results.md: TARGET rate for the near-constant flag
columns, the FLAG_DOCUMENT_* block reduced to one count feature, pairwise
correlation within the _AVG/_MODE/_MEDI housing block, and tail percentiles
for AMT_INCOME_TOTAL and the social-circle columns.

Writes a text report to results/target_correlation_results.md and
histograms to results/figures/.

Usage:
    uv run python scripts/target_and_correlation.py
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "target_correlation_results.md"
)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "results" / "figures"

NEAR_CONSTANT_FLAGS = ["FLAG_MOBIL", "FLAG_CONT_MOBILE"]
DOCUMENT_FLAGS = [f"FLAG_DOCUMENT_{i}" for i in range(2, 22)]
HOUSING_ATTRIBUTES = [
    "APARTMENTS",
    "BASEMENTAREA",
    "YEARS_BEGINEXPLUATATION",
    "YEARS_BUILD",
    "COMMONAREA",
    "ELEVATORS",
    "ENTRANCES",
    "FLOORSMAX",
    "FLOORSMIN",
    "LANDAREA",
    "LIVINGAPARTMENTS",
    "LIVINGAREA",
    "NONLIVINGAPARTMENTS",
    "NONLIVINGAREA",
]
SOCIAL_CIRCLE_COLUMNS = [
    "OBS_30_CNT_SOCIAL_CIRCLE",
    "DEF_30_CNT_SOCIAL_CIRCLE",
    "OBS_60_CNT_SOCIAL_CIRCLE",
    "DEF_60_CNT_SOCIAL_CIRCLE",
]


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load application_train.csv.

    Duplicated from eda.py rather than imported: `python scripts/<name>.py`
    puts only `scripts/` on sys.path, not the project root, so
    `from scripts.eda import ...` fails with exactly the ModuleNotFoundError
    this duplication avoids. Each script stays independently runnable the
    same way get_data.py and eda.py already are.
    """
    path = data_dir / "application_train.csv"
    return pd.read_csv(filepath_or_buffer=path)


def target_rate_by_value(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Row count and TARGET rate for each distinct value of column."""
    grouped = df.groupby(column)["TARGET"].agg(n="count", target_rate="mean")
    return grouped.sort_index()


def document_flags_summary(df: pd.DataFrame, doc_columns: list[str]) -> pd.DataFrame:
    """Prevalence and TARGET rate when each FLAG_DOCUMENT_X == 1, next to
    the overall TARGET rate for comparison.
    """
    overall_rate = df["TARGET"].mean()
    rows = []
    for col in doc_columns:
        provided = df[col] == 1
        rows.append(
            {
                "column": col,
                "n_provided": int(provided.sum()),
                "pct_provided": round(100 * provided.mean(), 3),
                "target_rate_when_provided": df.loc[provided, "TARGET"].mean(),
                "target_rate_overall": overall_rate,
            }
        )
    return pd.DataFrame(rows).set_index("column")


def documents_provided_count(df: pd.DataFrame, doc_columns: list[str]) -> pd.Series:
    """Row-wise count of FLAG_DOCUMENT_* columns equal to 1."""
    return df[doc_columns].sum(axis=1).rename("DOCUMENTS_PROVIDED_COUNT")


def housing_block_correlation(df: pd.DataFrame, attributes: list[str]) -> pd.DataFrame:
    """Pairwise correlation between the _AVG/_MODE/_MEDI variants of each
    housing attribute.
    """
    rows = []
    for attr in attributes:
        cols = [f"{attr}_AVG", f"{attr}_MODE", f"{attr}_MEDI"]
        if not all(c in df.columns for c in cols):
            continue
        corr = df[cols].corr()
        rows.append(
            {
                "attribute": attr,
                "avg_mode": corr.loc[cols[0], cols[1]],
                "avg_medi": corr.loc[cols[0], cols[2]],
                "mode_medi": corr.loc[cols[1], cols[2]],
            }
        )
    if not rows:
        return pd.DataFrame(columns=["avg_mode", "avg_medi", "mode_medi"])
    return pd.DataFrame(rows).set_index("attribute")


def tail_percentiles(series: pd.Series) -> pd.Series:
    """90th through 100th (max) percentiles of series."""
    return series.quantile([0.90, 0.95, 0.99, 0.999, 1.0])


def plot_histogram(
    series: pd.Series,
    output_path: Path,
    title: str,
    bins: int = 100,
    log_x: bool = False,
    log_y: bool = False,
    xlim: tuple[float, float] | None = None,
) -> None:
    """Save a histogram of series to output_path.

    log_x computes log-spaced bin edges, not just a log-scaled axis on
    linearly-spaced bins — the latter puts ~all of the mass in a single
    bin when one value dominates the range, which defeats the point of a
    log view. xlim clips the data to a fixed range first, independent of
    log_x, for a zoomed-in look past a dominant outlier.
    """
    data = series.dropna()
    if xlim is not None:
        data = data[(data >= xlim[0]) & (data <= xlim[1])]

    fig, ax = plt.subplots(figsize=(6, 4))
    if log_x:
        positive = data[data > 0]
        bin_edges = np.logspace(
            np.log10(positive.min()), np.log10(positive.max()), bins
        )
        ax.hist(positive, bins=bin_edges)
        ax.set_xscale("log")
    else:
        ax.hist(data, bins=bins)
    if log_y:
        ax.set_yscale("log")
    ax.set_title(title)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def build_report(df: pd.DataFrame) -> str:
    """Assemble the full report as a single string."""
    doc_count = documents_provided_count(df, DOCUMENT_FLAGS)
    with_doc_count = df.assign(DOCUMENTS_PROVIDED_COUNT=doc_count)

    sections = [
        "=== near-constant flag columns: TARGET rate by value ===\n"
        + "\n\n".join(
            f"{col}\n{target_rate_by_value(df, col).to_string()}"
            for col in NEAR_CONSTANT_FLAGS
        ),
        "=== FLAG_DOCUMENT_*: prevalence and TARGET rate when provided ===\n"
        + document_flags_summary(df, DOCUMENT_FLAGS).to_string(),
        "=== DOCUMENTS_PROVIDED_COUNT (sum of FLAG_DOCUMENT_*): TARGET rate by value ===\n"
        + target_rate_by_value(with_doc_count, "DOCUMENTS_PROVIDED_COUNT").to_string(),
        "=== housing block (_AVG/_MODE/_MEDI) pairwise correlation ===\n"
        + housing_block_correlation(df, HOUSING_ATTRIBUTES).to_string(),
        "=== AMT_INCOME_TOTAL tail percentiles ===\n"
        + tail_percentiles(df["AMT_INCOME_TOTAL"]).to_string(),
        "=== social-circle columns: tail percentiles ===\n"
        + "\n\n".join(
            f"{col}\n{tail_percentiles(df[col]).to_string()}"
            for col in SOCIAL_CIRCLE_COLUMNS
        ),
    ]
    return "\n\n".join(sections)


def write_report(df: pd.DataFrame, output_path: Path = OUTPUT_PATH) -> None:
    """Write build_report's output to output_path, creating parent dirs."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(build_report(df), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Where application_train.csv lives.",
    )
    parser.add_argument(
        "--output", type=Path, default=OUTPUT_PATH, help="Where to write the report."
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=FIGURES_DIR,
        help="Where to save plots.",
    )
    args = parser.parse_args()

    df = load_train(data_dir=args.data_dir)

    income_p999 = df["AMT_INCOME_TOTAL"].quantile(0.999)
    plot_histogram(
        df["AMT_INCOME_TOTAL"],
        args.figures_dir / "amt_income_total_linear.png",
        "AMT_INCOME_TOTAL (linear, up to 99.9th pct)",
        xlim=(0, income_p999),
    )
    plot_histogram(
        df["AMT_INCOME_TOTAL"],
        args.figures_dir / "amt_income_total_log.png",
        "AMT_INCOME_TOTAL (log x, full range)",
        log_x=True,
    )
    for col in SOCIAL_CIRCLE_COLUMNS:
        plot_histogram(
            df[col], args.figures_dir / f"{col.lower()}.png", col, log_y=True
        )

    write_report(df, output_path=args.output)
    print(f"Report written to {args.output}; figures written to {args.figures_dir}")


if __name__ == "__main__":
    main()
