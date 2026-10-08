"""Raw (untreated) baseline for the D10 raw-vs-treated comparison — the
minimum treatment needed for a model to run at all on this data, nothing
else: no DAYS_EMPLOYED fix (the 365243 sentinel is left exactly as-is), no
missingness flags, no outlier handling, no column drops or aggregation.
Categorical columns are label-encoded as-is (pandas factorize, missing
values become their own -1 code); any numeric column still missing after
that is filled with its own median.

This is the shared "raw" cell of D10's 2x2 grid — one table, used for both
Random Forest's and LightGBM's raw baseline. The "treated" cells come from
scripts/clean_features.py instead.

D29: the per-column median used to fill a numeric NaN is now fit only on
the rows that will become the D6 train+val pool downstream (RANDOM_STATE/
TEST_SIZE below are the exact same values train_random_forest.py/
train_lightgbm.py use for their own holdout split, so the partition
computed here lines up row-for-row with theirs). Fitting the median over
all 307,511 rows, as the previous version did, let the test split
influence a value baked into every row of the output CSV, including the
test rows themselves — a real, if small (median over ~260k values), leak.
Every function below still works with no `train_index` (falls back to
the whole `df`) so the unit tests, which build tiny synthetic frames with
no train/test concept, are unaffected — `main()` is the only caller
required to pass the real one.

Usage:
    uv run python scripts/raw_baseline.py
    uv run python scripts/raw_baseline.py --data-dir path/to/data/raw \
        --output path/to/train_raw.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
OUTPUT_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "train_raw.csv"
)
SUMMARY_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "raw_baseline_summary.md"
)

# D29: duplicated from train_random_forest.py/train_lightgbm.py, on
# purpose — must stay byte-identical to both, since the whole point is
# that the train/test partition computed here (on the full
# application_train.csv row order, before any column is dropped) lines
# up exactly with the one split_holdout() computes later on the already-
# written CSV. Verified empirically, not just assumed: with no shuffle
# and no row drop/reorder anywhere between here and there, the two
# splits produce the identical set of row positions, in the identical
# order.
RANDOM_STATE = 42
TEST_SIZE = 0.15
TARGET_COL = "TARGET"


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load application_train.csv.

    Duplicated from the other scripts/ modules rather than imported — see
    target_and_correlation.py's load_train docstring for why.
    """
    path = data_dir / "application_train.csv"
    return pd.read_csv(filepath_or_buffer=path)


def fit_train_index(df: pd.DataFrame) -> pd.Index:
    """D29: the D6 train+val row positions, computed on `df` as-is (full
    columns, original row order) so it matches split_holdout()'s own
    split downstream exactly. Call this once in main(), before any
    fitting step, and thread the result through every function below
    that fits a statistic.
    """
    train_index, _ = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_index


def label_encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every object-dtype column as-is: pandas factorize
    assigns each distinct string its own integer code and -1 to any NaN,
    so this is also how a categorical column's missingness gets handled
    on the raw path — no "Missing" category, no indicator, just a code.

    Not data-dependent in the leakage sense (the codes are arbitrary
    labels, not a statistic a reviewer would call a leak), so this needs
    no train_index.
    """
    out = df.copy()
    categorical_columns = out.select_dtypes(include=["object", "str"]).columns
    for col in categorical_columns:
        codes, _ = pd.factorize(out[col])
        out[col] = codes
    return out


def median_impute_numeric(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Fill any remaining missing value in a numeric column with that
    column's own median. Deliberately generic and column-agnostic — the
    point of the raw path is minimum treatment, not the hand-picked
    per-column reasoning in the treated path (D11/D12/D13).

    D29: when `train_index` is given, the median is computed from
    `df.loc[train_index, col]` only — the rows that will become the
    train+val pool — and then used to fill the *whole* column, test rows
    included. With `train_index=None` (the default, used by every
    existing unit test), the median is computed from the whole `df`,
    exactly as before.
    """
    out = df.copy()
    fit_source = out if train_index is None else out.loc[train_index]
    numeric_columns = out.select_dtypes(include="number").columns
    for col in numeric_columns:
        if out[col].isna().any():
            out[col] = out[col].fillna(fit_source[col].median())
    return out


def build_raw_baseline(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Apply the full raw-path pipeline (D10): label-encode categoricals,
    then median-impute whatever numeric NaNs remain. Nothing else — the
    365243 DAYS_EMPLOYED sentinel, the FLAG_DOCUMENT_* block, the housing
    _AVG/_MODE/_MEDI redundancy, and every outlier all pass through
    untouched, on purpose.

    `train_index` is threaded straight to median_impute_numeric (D29) —
    see that docstring for what passing or omitting it does.
    """
    out = label_encode_categoricals(df)
    out = median_impute_numeric(out, train_index=train_index)
    return out


def build_summary(raw: pd.DataFrame, baseline: pd.DataFrame) -> str:
    """Short before/after audit trail: shape (unchanged), which columns
    got label-encoded, and confirmation that no missing values remain.
    """
    encoded_columns = sorted(raw.select_dtypes(include=["object", "str"]).columns)
    remaining_missing = baseline.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0]
    sections = [
        f"raw shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns",
        f"baseline shape: {baseline.shape[0]:,} rows x {baseline.shape[1]} columns "
        "(same columns, no drops or additions)",
        f"label-encoded columns ({len(encoded_columns)}): {encoded_columns}",
        f"DAYS_EMPLOYED sentinel (365243) count, left untouched: "
        f"{int((baseline['DAYS_EMPLOYED'] == 365243).sum())}",
        f"remaining missing values after median-impute: {len(remaining_missing)} "
        "columns (should be 0)",
    ]
    return "\n\n".join(sections)


def write_outputs(
    raw: pd.DataFrame,
    baseline: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
) -> None:
    """Write the raw-baseline feature table and its summary, creating
    parent dirs.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(output_path, index=False)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(raw, baseline), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Where application_train.csv lives.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_PATH,
        help="Where to write the raw-baseline feature table.",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=SUMMARY_PATH,
        help="Where to write the before/after summary.",
    )
    args = parser.parse_args()

    raw = load_train(data_dir=args.data_dir)
    train_index = fit_train_index(raw)
    baseline = build_raw_baseline(raw, train_index=train_index)
    write_outputs(raw, baseline, output_path=args.output, summary_path=args.summary)
    print(
        f"Raw baseline written to {args.output} (medians fit on "
        f"{len(train_index):,} train+val rows only, D29); "
        f"summary written to {args.summary}"
    )


if __name__ == "__main__":
    main()
