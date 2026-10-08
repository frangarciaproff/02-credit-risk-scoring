"""Feature-treatment pipeline for the Home Credit application-level data,
implementing the cleaning decisions in docs/DECISIONS.md (D2, D11, D12,
D13, D15): DAYS_EMPLOYED sentinel fix + imputation, binary-flag encoding,
FLAG_DOCUMENT_* block reduction, housing-block collapse, OWN_CAR_AGE
imputation, outlier capping for AMT_INCOME_TOTAL and the social-circle
columns, every remaining missing-value column split by whether the
missingness is explained by an observed column (D12, D13), and label
encoding for Random Forest's remaining categorical columns (D15).

This is the "treated" path of the D10 raw-vs-treated comparison —
build_clean_features() is meant to be imported by the RF and LightGBM
benchmark scripts (steps 3-4) rather than having its logic duplicated
there. Its output is ready for LightGBM after the training script casts
the remaining object-dtype columns to pandas "category" dtype (D15 — a
CSV round-trip can't preserve dtype, so that cast happens at load time in
the training script, not here); Random Forest needs the additional
finalize_for_random_forest() step, since it can't accept NaN or string
input at all.

D29: every function below that fits a statistic from the data (a median
fill, a percentile cap) now takes an optional `train_index` and computes
that statistic from `df.loc[train_index]` only, never from the full
307,511-row dataset. RANDOM_STATE/TEST_SIZE below are the exact same
values train_random_forest.py/train_lightgbm.py use for their own D6
holdout split, so fit_train_index()'s partition lines up row-for-row
with split_holdout()'s, computed later on the already-written CSV.
Before this, every median and every percentile cap baked into the output
CSVs was computed over all 307,511 rows, including the ~46,127 that
later become the held-out test split — a real, if small (a median or a
0.999-quantile over roughly 260k values barely moves for losing ~46k of
them), leak. Every function still works with no `train_index` (falls
back to the whole `df`), so the unit tests, which build tiny synthetic
frames with no train/test concept, are unaffected — `main()` is the only
caller required to pass the real one.

Usage:
    uv run python scripts/clean_features.py
    uv run python scripts/clean_features.py --data-dir path/to/data/raw \
        --output path/to/train_treated.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
OUTPUT_PATH = PROCESSED_DIR / "train_treated_lightgbm.csv"
RF_OUTPUT_PATH = PROCESSED_DIR / "train_treated_rf.csv"
SUMMARY_PATH = (
    Path(__file__).resolve().parent.parent / "results" / "cleaning_summary.md"
)

DAYS_EMPLOYED_SENTINEL = 365243

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

# D11: FLAG_DOCUMENT_2 .. FLAG_DOCUMENT_21, partitioned by the two-proportion
# z-test + Bonferroni correction (alpha = 0.05/20) applied in
# results/target_correlation_results.md.
ROBUST_DOCUMENT_FLAGS = [6, 8, 9, 11, 13, 14, 15, 16, 18]
STANDALONE_DOCUMENT_FLAGS = [3, 21]
NOISE_DOCUMENT_FLAGS = [2, 4, 5, 7, 10, 12, 17, 19, 20]

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

CAP_PERCENTILE = 0.999

# D12: remaining missing-value groups after D11, split by whether an
# observed column explains the missingness (see docs/DECISIONS.md, D12).
#
# "Partially explained" — NAME_INCOME_TYPE (Pensioner/Unemployed) or a
# shared cross-column co-occurrence (the housing block) accounts for most
# of the missingness — imputed for both RF and LightGBM.
HOUSING_NUMERIC_COLUMNS = [f"{attr}_MEDI" for attr in HOUSING_ATTRIBUTES] + [
    "TOTALAREA_MODE"
]
HOUSING_CATEGORICAL_COLUMNS = [
    "FONDKAPREMONT_MODE",
    "HOUSETYPE_MODE",
    "WALLSMATERIAL_MODE",
    "EMERGENCYSTATE_MODE",
]

# "Unexplained" — no observed column predicts the missingness (D12) —
# indicator columns are added for both models, but the raw value is only
# imputed for Random Forest; LightGBM keeps the real NaN and uses its
# native missing-value handling.
CREDIT_BUREAU_REQUEST_COLUMNS = [
    f"AMT_REQ_CREDIT_BUREAU_{period}"
    for period in ("HOUR", "DAY", "WEEK", "MON", "QRT", "YEAR")
]
EXT_SOURCE_COLUMNS = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]

# D13: closes two gaps D12 flagged and deferred. DAYS_EMPLOYED (from D2's
# sentinel fix) is imputed for both models — it's fully explained by
# construction (DAYS_EMPLOYED_ANOMALY is 1 exactly where it's NaN).
# The five columns below are all under 0.5% missing and weren't
# investigated for a mechanism — imputed for both models given the
# negligible size either way.
MINOR_NUMERIC_COLUMNS = [
    "AMT_GOODS_PRICE",
    "AMT_ANNUITY",
    "CNT_FAM_MEMBERS",
    "DAYS_LAST_PHONE_CHANGE",
]
MINOR_CATEGORICAL_COLUMNS = ["NAME_TYPE_SUITE"]


def load_train(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Load application_train.csv.

    Duplicated from eda.py/target_and_correlation.py rather than imported —
    `python scripts/<name>.py` only puts `scripts/` on sys.path, not the
    project root, so a cross-script import fails with the same
    ModuleNotFoundError target_and_correlation.py hit. Each script stays
    independently runnable the same way.
    """
    path = data_dir / "application_train.csv"
    return pd.read_csv(filepath_or_buffer=path)


def fit_train_index(df: pd.DataFrame) -> pd.Index:
    """D29: the D6 train+val row positions, computed on `df` as-is (full
    columns, original row order) so it matches split_holdout()'s own
    split downstream exactly. Call this once in main(), before any
    fitting step, and thread the result through build_clean_features()
    and finalize_for_random_forest().
    """
    train_index, _ = train_test_split(
        df.index,
        test_size=TEST_SIZE,
        stratify=df[TARGET_COL],
        random_state=RANDOM_STATE,
    )
    return train_index


def fix_days_employed(df: pd.DataFrame) -> pd.DataFrame:
    """Replace the 365243 sentinel in DAYS_EMPLOYED with NaN and add a
    binary DAYS_EMPLOYED_ANOMALY indicator before the value is dropped.

    D2: the anomaly is concentrated in a specific applicant segment, not
    random noise, so it's flagged rather than silently discarded.
    """
    out = df.copy()
    is_anomaly = out["DAYS_EMPLOYED"] == DAYS_EMPLOYED_SENTINEL
    out["DAYS_EMPLOYED_ANOMALY"] = is_anomaly.astype(int)
    out.loc[is_anomaly, "DAYS_EMPLOYED"] = np.nan
    return out


def encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Encode FLAG_OWN_CAR / FLAG_OWN_REALTY as 1/0 instead of Y/N. D11."""
    out = df.copy()
    for col in ("FLAG_OWN_CAR", "FLAG_OWN_REALTY"):
        out[col] = out[col].map({"Y": 1, "N": 0}).astype(int)
    return out


def drop_near_constant_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Drop FLAG_MOBIL and FLAG_CONT_MOBILE — neither shows a real
    TARGET-rate difference by value (two-proportion z-test). D11.
    """
    return df.drop(columns=["FLAG_MOBIL", "FLAG_CONT_MOBILE"])


def aggregate_document_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce the FLAG_DOCUMENT_* block per D11: FLAG_DOCUMENT_3 and
    FLAG_DOCUMENT_21 stay as their own columns, untouched; the
    statistically robust subset (ROBUST_DOCUMENT_FLAGS) collapses into one
    DOCUMENTS_PROVIDED_COUNT feature; the noise subset
    (NOISE_DOCUMENT_FLAGS) is dropped outright — their apparent
    target-rate deviation doesn't survive Bonferroni correction.
    """
    out = df.copy()
    robust_cols = [f"FLAG_DOCUMENT_{i}" for i in ROBUST_DOCUMENT_FLAGS]
    noise_cols = [f"FLAG_DOCUMENT_{i}" for i in NOISE_DOCUMENT_FLAGS]
    out["DOCUMENTS_PROVIDED_COUNT"] = out[robust_cols].sum(axis=1)
    return out.drop(columns=robust_cols + noise_cols)


def collapse_housing_block(
    df: pd.DataFrame, attributes: list[str] | None = None
) -> pd.DataFrame:
    """Keep only the _MEDI aggregation of each housing attribute, drop
    _AVG/_MODE — pairwise correlation between the three is 0.96-0.998 for
    every attribute (results/target_correlation_results.md). D11.
    """
    attributes = attributes if attributes is not None else HOUSING_ATTRIBUTES
    drop_cols = [
        f"{attr}{suffix}"
        for attr in attributes
        for suffix in ("_AVG", "_MODE")
        if f"{attr}{suffix}" in df.columns
    ]
    return df.drop(columns=drop_cols)


def impute_own_car_age(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing OWN_CAR_AGE with 0 — its 66% missingness lines up
    almost exactly with FLAG_OWN_CAR == 'N' (structural, not MNAR). D11.

    The fill value (0) is a constant, not a statistic computed from the
    data, so this has no train_index — there's nothing for a test split
    to leak into.
    """
    out = df.copy()
    out["OWN_CAR_AGE"] = out["OWN_CAR_AGE"].fillna(0)
    return out


def cap_at_percentile(
    series: pd.Series,
    percentile: float = CAP_PERCENTILE,
    threshold: float | None = None,
    fit_index: pd.Index | None = None,
) -> tuple[pd.Series, float]:
    """Cap series at its own `percentile`-th value, or at an explicit
    `threshold` — so a cap fit on a train split can be reused on val/test
    without recomputing it from the split being capped, avoiding leakage.

    D29: when neither `threshold` nor `fit_index` is given, the quantile
    is computed from the whole `series` (the old, leaky-when-called-on-
    the-full-dataset behavior — kept as the default so the existing unit
    tests, which pass small synthetic series with no train/test concept,
    are unaffected). Pass `fit_index` to compute the quantile from
    `series.loc[fit_index]` only; the clip is always applied to the whole
    `series` regardless of which rows the threshold came from. `threshold`
    takes priority over `fit_index` when both are given.

    Returns the capped series and the threshold actually used.
    """
    if threshold is not None:
        cap = threshold
    elif fit_index is not None:
        cap = series.loc[fit_index].quantile(percentile)
    else:
        cap = series.quantile(percentile)
    return series.clip(upper=cap), cap


def cap_income_outliers(
    df: pd.DataFrame,
    threshold: float | None = None,
    train_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Cap AMT_INCOME_TOTAL at its 99.9th percentile by default. The
    uncapped max (117,000,000) is a data-entry error, not a real income —
    it pairs an ordinary loan/annuity/occupation/education profile with a
    9-figure income — so the value is capped rather than the row dropped,
    keeping the row's TARGET label intact. D11.

    D29: `train_index`, when given, restricts which rows the 99.9th
    percentile itself is computed from (see cap_at_percentile) — every
    row is still capped, train and test alike, using that one threshold.
    """
    out = df.copy()
    out["AMT_INCOME_TOTAL"], _ = cap_at_percentile(
        out["AMT_INCOME_TOTAL"], threshold=threshold, fit_index=train_index
    )
    return out


def cap_social_circle_outliers(
    df: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
    train_index: pd.Index | None = None,
) -> pd.DataFrame:
    """Cap each social-circle column at its own 99.9th percentile by
    default. Treated as genuine extreme values that need bounding for
    modeling, not data errors — see D11 for the row-level and
    dataset-wide consistency checks behind that distinction.

    D29: `train_index`, when given, restricts which rows each column's
    99.9th percentile is computed from (see cap_at_percentile) — an
    explicit entry in `thresholds` still takes priority per column.
    """
    out = df.copy()
    thresholds = thresholds or {}
    for col in SOCIAL_CIRCLE_COLUMNS:
        out[col], _ = cap_at_percentile(
            out[col], threshold=thresholds.get(col), fit_index=train_index
        )
    return out


def add_housing_info_missing_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add one shared HOUSING_INFO_MISSING indicator for the whole housing
    block, derived from APARTMENTS_MEDI — the ~14 numeric and 4 categorical
    housing columns are missing together (Jaccard 0.72-0.96 between their
    missing-row sets), one shared data-collection gap rather than 18
    independent ones. Must run before impute_housing_block fills the NaNs
    it reads. D12.
    """
    out = df.copy()
    out["HOUSING_INFO_MISSING"] = out["APARTMENTS_MEDI"].isna().astype(int)
    return out


def impute_housing_block(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Fill the remaining housing-block NaNs: median for the numeric
    _MEDI/TOTALAREA_MODE columns, a "Missing" category for the categorical
    MODE columns. NAME_HOUSING_TYPE doesn't explain this missingness on its
    own, but the block explaining itself (co-occurring almost entirely)
    counts as "partially explained" per D12 — imputed for both models,
    with HOUSING_INFO_MISSING carrying the signal a raw fill would erase.

    D29: each numeric column's median is computed from
    `out.loc[train_index, col]` when `train_index` is given, else from
    the whole column (default, used by the unit tests). The categorical
    "Missing" fill is a constant, not a statistic — no leak possible
    there either way.
    """
    out = df.copy()
    fit_source = out if train_index is None else out.loc[train_index]
    for col in HOUSING_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(fit_source[col].median())
    for col in HOUSING_CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("Missing")
    return out


def impute_occupation_type(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing OCCUPATION_TYPE with a "Missing" category. Pensioner
    and Unemployed applicants are 100% missing here (structurally — no
    occupation because not working), but ~16-17% of Working/Commercial
    associate/State servant rows are missing too with no explanation —
    "partially explained" per D12, imputed for both models regardless,
    since the categorical fill preserves the distinction either way (a
    tree model can still split on the "Missing" category itself).

    Constant fill, no train_index needed.
    """
    out = df.copy()
    out["OCCUPATION_TYPE"] = out["OCCUPATION_TYPE"].fillna("Missing")
    return out


def add_unexplained_missing_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add missing-value indicators for the columns D12 found no observed
    explanation for (EXT_SOURCE_1/2/3, the AMT_REQ_CREDIT_BUREAU_* block).
    The indicators themselves have no NaNs, so they're safe for both
    models; the raw values are left untouched here — only
    finalize_for_random_forest imputes them, since LightGBM handles the
    real NaN natively and shouldn't have that signal papered over.
    """
    out = df.copy()
    for col in EXT_SOURCE_COLUMNS:
        out[f"{col}_MISSING"] = out[col].isna().astype(int)
    out["CREDIT_BUREAU_REQUEST_MISSING"] = (
        out["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().astype(int)
    )
    return out


def impute_days_employed(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Fill the DAYS_EMPLOYED NaNs left by fix_days_employed with the
    median of the genuine (non-anomalous) values. D13: fully explained by
    construction — DAYS_EMPLOYED_ANOMALY is 1 exactly where this is NaN —
    so it's imputed for both models, closing the gap D12 flagged but
    deferred.

    D29: the median is computed from `out.loc[train_index,
    "DAYS_EMPLOYED"]` when `train_index` is given, else from the whole
    column (default, used by the unit tests).
    """
    out = df.copy()
    fit_source = out if train_index is None else out.loc[train_index]
    out["DAYS_EMPLOYED"] = out["DAYS_EMPLOYED"].fillna(
        fit_source["DAYS_EMPLOYED"].median()
    )
    return out


def impute_minor_missing(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Fill the five columns D12 left uninvestigated (each under 0.5%
    missing): median for the numeric ones (MINOR_NUMERIC_COLUMNS), a
    "Missing" category for NAME_TYPE_SUITE. D13: imputed for both models —
    the size is too small for the RF-vs-LightGBM split to matter either
    way.

    D29: each numeric column's median is computed from
    `out.loc[train_index, col]` when `train_index` is given, else from
    the whole column (default, used by the unit tests).
    """
    out = df.copy()
    fit_source = out if train_index is None else out.loc[train_index]
    for col in MINOR_NUMERIC_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna(fit_source[col].median())
    for col in MINOR_CATEGORICAL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].fillna("Missing")
    return out


def build_clean_features(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Apply the common treated-path pipeline (D2, D11, D12's "partially
    explained" group, D13), in the decided sequence: DAYS_EMPLOYED fix +
    imputation, binary-flag encoding, near-constant-flag drop,
    document-flag reduction, housing-block collapse, OWN_CAR_AGE
    imputation, the two outlier caps, the housing-block indicator +
    imputation, OCCUPATION_TYPE imputation, the five minor columns, and
    indicators (not values) for the "unexplained" group.

    D29: `train_index` is threaded through to every step that fits a
    statistic (impute_days_employed, cap_income_outliers,
    cap_social_circle_outliers, impute_housing_block,
    impute_minor_missing) — see fit_train_index() for how it's computed
    and why it has to be computed before any column is dropped. Leaving
    it as None (the default) reproduces the pre-D29 behavior, which is
    what every existing unit test still exercises on tiny synthetic
    frames with no real train/test split to speak of.

    Missingness-wise this is done: EXT_SOURCE_1/2/3 and
    AMT_REQ_CREDIT_BUREAU_* are the only columns still carrying real NaNs,
    which LightGBM handles natively. But ~14 columns are still
    object-dtype strings (NAME_CONTRACT_TYPE, CODE_GENDER, OCCUPATION_TYPE,
    ORGANIZATION_TYPE, and others) — LightGBM's sklearn wrapper needs
    those cast to pandas "category" dtype to use its native categorical
    splitting, which the training script does at load time (D15), not
    here. Random Forest needs a different, additional step:
    finalize_for_random_forest, which can't accept NaN or string input at
    all.
    """
    out = fix_days_employed(df)
    out = impute_days_employed(out, train_index=train_index)
    out = encode_binary_flags(out)
    out = drop_near_constant_flags(out)
    out = aggregate_document_flags(out)
    out = collapse_housing_block(out)
    out = impute_own_car_age(out)
    out = cap_income_outliers(out, train_index=train_index)
    out = cap_social_circle_outliers(out, train_index=train_index)
    out = add_housing_info_missing_indicator(out)
    out = impute_housing_block(out, train_index=train_index)
    out = impute_occupation_type(out)
    out = impute_minor_missing(out, train_index=train_index)
    out = add_unexplained_missing_indicators(out)
    return out


def encode_categoricals_for_random_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode every remaining object-dtype column with
    pandas.factorize — same convention as raw_baseline.py's
    label_encode_categoricals. D15: build_clean_features() never actually
    encoded most categoricals (only FLAG_OWN_CAR/FLAG_OWN_REALTY, done
    earlier); RandomForestClassifier.fit() rejects string input outright,
    so this was a gap, not a stylistic choice deferred on purpose.

    Ordinal codes are a defensible choice specifically because both
    Random Forest and LightGBM are tree ensembles: a split on an
    arbitrary integer threshold partitions the categories into two
    groups however they were numbered, with no assumption that the codes
    are ordered or evenly spaced. That's not true for a linear or
    distance-based model, which would need one-hot or target encoding
    instead — this function is not a general-purpose categorical encoder.

    Called from finalize_for_random_forest, after the "unexplained" group
    is imputed. By this point every categorical column with missingness
    (OCCUPATION_TYPE, the housing _MODE columns, NAME_TYPE_SUITE) was
    already filled with "Missing" earlier in build_clean_features, so
    factorize's -1-for-NaN behavior never actually triggers here — it's
    only doing the string-to-integer mapping.
    """
    out = df.copy()
    categorical_columns = out.select_dtypes(include=["object", "str"]).columns
    for col in categorical_columns:
        codes, _ = pd.factorize(out[col])
        out[col] = codes
    return out


def finalize_for_random_forest(
    df: pd.DataFrame, train_index: pd.Index | None = None
) -> pd.DataFrame:
    """Impute the D12 "unexplained" group for Random Forest, which can't
    accept NaN input at all: median-fill EXT_SOURCE_1/2/3,
    AMT_REQ_CREDIT_BUREAU_*, and whatever social-circle NaNs survived
    capping. Then label-encode every remaining categorical column (D15) —
    RandomForestClassifier can't accept string input either. Call this on
    build_clean_features's output — never for the LightGBM path, which
    uses that output directly (after its own category-dtype cast) and
    relies on native NaN handling for exactly the unexplained group.

    D29: each column's median is computed from `out.loc[train_index,
    col]` when `train_index` is given (the same index build_clean_features
    was called with), else from the whole column (default, used by the
    unit tests).
    """
    out = df.copy()
    fit_source = out if train_index is None else out.loc[train_index]
    unexplained_numeric = (
        EXT_SOURCE_COLUMNS + CREDIT_BUREAU_REQUEST_COLUMNS + SOCIAL_CIRCLE_COLUMNS
    )
    for col in unexplained_numeric:
        if col in out.columns:
            out[col] = out[col].fillna(fit_source[col].median())
    out = encode_categoricals_for_random_forest(out)
    return out


def build_summary(raw: pd.DataFrame, treated: pd.DataFrame) -> str:
    """Short before/after audit trail for this transformation step: shape
    change, dropped/added columns, and remaining missingness. Not a full
    EDA report — those already exist in results/pre_process_results.md
    and results/target_correlation_results.md.
    """
    dropped = sorted(set(raw.columns) - set(treated.columns))
    added = sorted(set(treated.columns) - set(raw.columns))
    remaining_missing = treated.isna().sum()
    remaining_missing = remaining_missing[remaining_missing > 0].sort_values(
        ascending=False
    )
    sections = [
        f"raw shape: {raw.shape[0]:,} rows x {raw.shape[1]} columns",
        f"treated shape: {treated.shape[0]:,} rows x {treated.shape[1]} columns",
        f"columns dropped ({len(dropped)}): {dropped}",
        f"columns added ({len(added)}): {added}",
        "=== remaining missing values after treatment ===\n"
        + remaining_missing.to_string(),
    ]
    return "\n\n".join(sections)


def write_outputs(
    raw: pd.DataFrame,
    treated: pd.DataFrame,
    output_path: Path = OUTPUT_PATH,
    summary_path: Path = SUMMARY_PATH,
    rf_output_path: Path = RF_OUTPUT_PATH,
    train_index: pd.Index | None = None,
) -> None:
    """Write both treated feature tables (LightGBM-ready as given, plus
    the Random Forest variant via finalize_for_random_forest) and one
    shared before/after summary, creating parent dirs.

    D29: `train_index` is passed straight to finalize_for_random_forest —
    `treated` (the LightGBM-ready table) is assumed already built with
    the same `train_index` via build_clean_features, so nothing further
    needs fitting for that one here.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    treated.to_csv(output_path, index=False)
    rf_output_path.parent.mkdir(parents=True, exist_ok=True)
    finalize_for_random_forest(treated, train_index=train_index).to_csv(
        rf_output_path, index=False
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(build_summary(raw, treated), encoding="utf-8")


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
        help="Where to write the LightGBM-ready treated feature table.",
    )
    parser.add_argument(
        "--rf-output",
        type=Path,
        default=RF_OUTPUT_PATH,
        help="Where to write the Random Forest-ready treated feature table.",
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
    treated = build_clean_features(raw, train_index=train_index)
    write_outputs(
        raw,
        treated,
        output_path=args.output,
        summary_path=args.summary,
        rf_output_path=args.rf_output,
        train_index=train_index,
    )
    print(
        f"LightGBM-ready features written to {args.output}; "
        f"Random Forest-ready features written to {args.rf_output}; "
        f"summary written to {args.summary} "
        f"(medians/caps fit on {len(train_index):,} train+val rows only, D29)"
    )


if __name__ == "__main__":
    main()
