"""Exploratory data analysis for the Home Credit application-level data.

Loads application_train.csv and writes summary diagnostics (head, dtype
split, descriptive stats, cardinality, missing-value report) to a report
file (SPEC.md, Plan step 2).

Usage:
    uv run python scripts/eda.py
    uv run python scripts/eda.py --data-dir path/to/data/raw --output path/to/report.md
"""

import argparse
import io
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "pre_process_results.md"
)


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load application_train.csv."""
    path = data_dir / "application_train.csv"
    return pd.read_csv(filepath_or_buffer=path)


def split_by_dtype(df: pd.DataFrame) -> dict[str, list[str]]:
    """Split columns into int, float, and categorical by pandas dtype."""
    groups: dict[str, list[str]] = {"int": [], "float": [], "categorical": []}
    for column in df.columns:
        dtype = df[column].dtype
        if pd.api.types.is_integer_dtype(dtype):
            groups["int"].append(column)
        elif pd.api.types.is_float_dtype(dtype):
            groups["float"].append(column)
        else:
            groups["categorical"].append(column)
    return groups


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage missing per column, worst first. Excludes
    columns with zero missing values.
    """
    counts = df.isna().sum()
    pct = (counts / len(df) * 100).round(2)
    report = pd.DataFrame({"n_missing": counts, "pct_missing": pct})
    return report[report["n_missing"] > 0].sort_values(
        by="pct_missing", ascending=False
    )


def cardinality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Number of unique non-null values per column, ascending. nunique == 1
    flags a constant column.
    """
    nunique = df.nunique(dropna=True)
    return nunique.sort_values().rename("nunique").to_frame()


def numeric_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Mean, median, std, quartiles, min/max for numeric columns."""
    return df[columns].describe().T


def categorical_summary(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Count, unique, top value, and frequency for categorical columns."""
    if not columns:
        return pd.DataFrame()
    return df[columns].describe().T


def build_report(df: pd.DataFrame) -> str:
    """Assemble the full report as a single string."""
    buf = io.StringIO()
    df.info(buf=buf)
    info_text = buf.getvalue()

    groups = split_by_dtype(df)
    numeric_cols = groups["int"] + groups["float"]
    missing = missing_report(df)

    sections = [
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        "=== .info() ===\n" + info_text,
        "=== head ===\n" + df.head().to_string(),
        "=== dtype split ===\n"
        + "\n".join(f"{name} ({len(cols)}): {cols}" for name, cols in groups.items()),
        "=== numeric summary (int + float) ===\n"
        + numeric_summary(df, numeric_cols).to_string(),
        "=== categorical summary ===\n"
        + categorical_summary(df, groups["categorical"]).to_string(),
        "=== cardinality (nunique per column, ascending) ===\n"
        + cardinality_report(df).to_string(),
        f"=== missing values ({len(missing)} of {df.shape[1]} columns) ===\n"
        + missing.to_string(),
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
    args = parser.parse_args()

    df = load_train(data_dir=args.data_dir)
    write_report(df, output_path=args.output)
    print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
