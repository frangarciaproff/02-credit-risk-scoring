import numpy as np
import pandas as pd

from scripts.clean_features import (
    NOISE_DOCUMENT_FLAGS,
    ROBUST_DOCUMENT_FLAGS,
    STANDALONE_DOCUMENT_FLAGS,
    add_housing_info_missing_indicator,
    add_unexplained_missing_indicators,
    aggregate_document_flags,
    build_clean_features,
    build_summary,
    cap_at_percentile,
    cap_income_outliers,
    cap_social_circle_outliers,
    collapse_housing_block,
    drop_near_constant_flags,
    encode_binary_flags,
    encode_categoricals_for_random_forest,
    finalize_for_random_forest,
    fit_train_index,
    fix_days_employed,
    impute_days_employed,
    impute_housing_block,
    impute_minor_missing,
    impute_occupation_type,
    impute_own_car_age,
    write_outputs,
)


def _sample_df() -> pd.DataFrame:
    n = 6
    data = {
        "SK_ID_CURR": list(range(1, n + 1)),
        "TARGET": [0, 1, 0, 0, 1, 0],
        "DAYS_EMPLOYED": [-100, -200, 365243, -300, 365243, -50],
        "FLAG_OWN_CAR": ["Y", "N", "N", "Y", "N", "N"],
        "FLAG_OWN_REALTY": ["N", "Y", "Y", "Y", "N", "Y"],
        "FLAG_MOBIL": [1, 1, 1, 1, 1, 0],
        "FLAG_CONT_MOBILE": [1, 1, 0, 1, 1, 1],
        "APARTMENTS_AVG": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "APARTMENTS_MODE": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "APARTMENTS_MEDI": [0.15, 0.25, np.nan, 0.45, 0.55, 0.65],
        "TOTALAREA_MODE": [0.2, np.nan, 0.3, 0.4, 0.5, 0.6],
        "FONDKAPREMONT_MODE": ["A", "B", np.nan, "A", "B", "A"],
        "HOUSETYPE_MODE": ["A", "B", np.nan, "A", "B", "A"],
        "WALLSMATERIAL_MODE": ["A", "B", np.nan, "A", "B", "A"],
        "EMERGENCYSTATE_MODE": ["No", "No", np.nan, "No", "Yes", "No"],
        "OCCUPATION_TYPE": ["Laborers", np.nan, np.nan, "Drivers", "Laborers", np.nan],
        "OWN_CAR_AGE": [5.0, np.nan, np.nan, 3.0, np.nan, np.nan],
        "AMT_INCOME_TOTAL": [100.0, 200.0, 150.0, 300.0, 250.0, 1_000_000.0],
        "OBS_30_CNT_SOCIAL_CIRCLE": [0.0, 1.0, 2.0, 0.0, 3.0, 50.0],
        "DEF_30_CNT_SOCIAL_CIRCLE": [0.0, 0.0, 1.0, 0.0, 1.0, 10.0],
        "OBS_60_CNT_SOCIAL_CIRCLE": [0.0, 1.0, 2.0, 0.0, 3.0, 48.0],
        "DEF_60_CNT_SOCIAL_CIRCLE": [0.0, 0.0, 1.0, 0.0, 1.0, 8.0],
        "EXT_SOURCE_1": [0.1, np.nan, 0.3, np.nan, 0.5, 0.6],
        "EXT_SOURCE_2": [0.2, 0.3, np.nan, 0.5, 0.6, 0.7],
        "EXT_SOURCE_3": [np.nan, 0.3, 0.4, 0.5, np.nan, 0.7],
        "AMT_REQ_CREDIT_BUREAU_HOUR": [0.0, np.nan, 0.0, 1.0, np.nan, 0.0],
        "AMT_REQ_CREDIT_BUREAU_DAY": [0.0, np.nan, 0.0, 1.0, np.nan, 0.0],
        "AMT_REQ_CREDIT_BUREAU_WEEK": [0.0, np.nan, 0.0, 1.0, np.nan, 0.0],
        "AMT_REQ_CREDIT_BUREAU_MON": [0.0, np.nan, 1.0, 1.0, np.nan, 0.0],
        "AMT_REQ_CREDIT_BUREAU_QRT": [0.0, np.nan, 1.0, 2.0, np.nan, 0.0],
        "AMT_REQ_CREDIT_BUREAU_YEAR": [1.0, np.nan, 2.0, 3.0, np.nan, 1.0],
        "NAME_TYPE_SUITE": [
            "Unaccompanied",
            np.nan,
            "Family",
            "Family",
            "Unaccompanied",
            "Family",
        ],
        "AMT_GOODS_PRICE": [90000.0, np.nan, 110000.0, 120000.0, 95000.0, 105000.0],
        "AMT_ANNUITY": [9000.0, 9500.0, np.nan, 10000.0, 9200.0, 9800.0],
        "CNT_FAM_MEMBERS": [2.0, 3.0, 1.0, np.nan, 2.0, 4.0],
        "DAYS_LAST_PHONE_CHANGE": [-500.0, -600.0, -700.0, -800.0, np.nan, -900.0],
    }
    for i in range(2, 22):
        # A couple of 1s scattered around so aggregation has something to sum.
        data[f"FLAG_DOCUMENT_{i}"] = [
            1 if (i + row) % 5 == 0 else 0 for row in range(n)
        ]
    return pd.DataFrame(data)


def test_fix_days_employed_replaces_sentinel_and_flags() -> None:
    result = fix_days_employed(_sample_df())

    assert result["DAYS_EMPLOYED_ANOMALY"].tolist() == [0, 0, 1, 0, 1, 0]
    assert result["DAYS_EMPLOYED"].isna().sum() == 2
    assert not (result["DAYS_EMPLOYED"] == 365243).any()


def test_encode_binary_flags_maps_yn_to_10() -> None:
    result = encode_binary_flags(_sample_df())

    assert result["FLAG_OWN_CAR"].tolist() == [1, 0, 0, 1, 0, 0]
    assert result["FLAG_OWN_REALTY"].tolist() == [0, 1, 1, 1, 0, 1]
    assert result["FLAG_OWN_CAR"].dtype == int


def test_drop_near_constant_flags_removes_columns() -> None:
    result = drop_near_constant_flags(_sample_df())

    assert "FLAG_MOBIL" not in result.columns
    assert "FLAG_CONT_MOBILE" not in result.columns


def test_document_flag_groups_partition_all_20_flags() -> None:
    all_flags = sorted(
        ROBUST_DOCUMENT_FLAGS + STANDALONE_DOCUMENT_FLAGS + NOISE_DOCUMENT_FLAGS
    )

    assert all_flags == list(range(2, 22))
    assert len(set(all_flags)) == 20  # no overlaps between the three groups


def test_aggregate_document_flags_builds_count_and_drops_correctly() -> None:
    df = _sample_df()
    result = aggregate_document_flags(df)

    robust_cols = [f"FLAG_DOCUMENT_{i}" for i in ROBUST_DOCUMENT_FLAGS]
    noise_cols = [f"FLAG_DOCUMENT_{i}" for i in NOISE_DOCUMENT_FLAGS]
    expected_count = df[robust_cols].sum(axis=1)

    assert list(result["DOCUMENTS_PROVIDED_COUNT"]) == list(expected_count)
    for col in robust_cols + noise_cols:
        assert col not in result.columns
    assert "FLAG_DOCUMENT_3" in result.columns
    assert "FLAG_DOCUMENT_21" in result.columns


def test_collapse_housing_block_keeps_medi_drops_avg_mode() -> None:
    result = collapse_housing_block(_sample_df(), attributes=["APARTMENTS"])

    assert "APARTMENTS_AVG" not in result.columns
    assert "APARTMENTS_MODE" not in result.columns
    assert "APARTMENTS_MEDI" in result.columns


def test_impute_own_car_age_fills_zero() -> None:
    result = impute_own_car_age(_sample_df())

    assert result["OWN_CAR_AGE"].isna().sum() == 0
    assert result.loc[1, "OWN_CAR_AGE"] == 0.0


def test_cap_at_percentile_caps_upper_tail() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 1000.0])
    capped, threshold = cap_at_percentile(series, percentile=0.5)

    assert threshold == series.quantile(0.5)
    assert capped.max() == threshold


def test_cap_at_percentile_uses_explicit_threshold_over_percentile() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 1000.0])
    capped, threshold = cap_at_percentile(series, threshold=2.5)

    assert threshold == 2.5
    assert capped.max() == 2.5


def test_cap_income_outliers_uses_default_percentile() -> None:
    df = _sample_df()
    result = cap_income_outliers(df)

    expected_cap = df["AMT_INCOME_TOTAL"].quantile(0.999)
    assert result["AMT_INCOME_TOTAL"].max() == expected_cap
    assert result["AMT_INCOME_TOTAL"].max() < df["AMT_INCOME_TOTAL"].max()


def test_cap_social_circle_outliers_respects_explicit_thresholds() -> None:
    df = _sample_df()
    result = cap_social_circle_outliers(
        df, thresholds={"OBS_30_CNT_SOCIAL_CIRCLE": 5.0}
    )

    assert result["OBS_30_CNT_SOCIAL_CIRCLE"].max() == 5.0
    # A column with no explicit threshold still falls back to its own 99.9th pct.
    assert result["DEF_30_CNT_SOCIAL_CIRCLE"].max() == df[
        "DEF_30_CNT_SOCIAL_CIRCLE"
    ].quantile(0.999)


def test_add_housing_info_missing_indicator_flags_apartments_medi_nan() -> None:
    result = add_housing_info_missing_indicator(_sample_df())

    assert result["HOUSING_INFO_MISSING"].tolist() == [0, 0, 1, 0, 0, 0]


def test_impute_housing_block_fills_numeric_median_and_categorical_missing() -> None:
    df = _sample_df()
    result = impute_housing_block(df)

    assert result["APARTMENTS_MEDI"].isna().sum() == 0
    assert result.loc[2, "APARTMENTS_MEDI"] == df["APARTMENTS_MEDI"].median()
    assert result["TOTALAREA_MODE"].isna().sum() == 0
    assert result["FONDKAPREMONT_MODE"].isna().sum() == 0
    assert result.loc[2, "FONDKAPREMONT_MODE"] == "Missing"


def test_impute_occupation_type_fills_missing_category() -> None:
    result = impute_occupation_type(_sample_df())

    assert result["OCCUPATION_TYPE"].isna().sum() == 0
    assert result.loc[1, "OCCUPATION_TYPE"] == "Missing"


def test_add_unexplained_missing_indicators_flags_without_imputing_values() -> None:
    df = _sample_df()
    result = add_unexplained_missing_indicators(df)

    assert result["EXT_SOURCE_1_MISSING"].tolist() == [0, 1, 0, 1, 0, 0]
    assert result["EXT_SOURCE_2_MISSING"].tolist() == [0, 0, 1, 0, 0, 0]
    assert result["EXT_SOURCE_3_MISSING"].tolist() == [1, 0, 0, 0, 1, 0]
    assert result["CREDIT_BUREAU_REQUEST_MISSING"].tolist() == [0, 1, 0, 0, 1, 0]
    # Values themselves are untouched here — NaNs still present.
    assert result["EXT_SOURCE_1"].isna().sum() == 2


def test_finalize_for_random_forest_imputes_unexplained_group_only() -> None:
    df = build_clean_features(_sample_df())
    result = finalize_for_random_forest(df)

    assert result["EXT_SOURCE_1"].isna().sum() == 0
    assert result["EXT_SOURCE_2"].isna().sum() == 0
    assert result["EXT_SOURCE_3"].isna().sum() == 0
    assert result["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().sum() == 0
    # The indicators added earlier are untouched by this step.
    assert (
        result["EXT_SOURCE_1_MISSING"].tolist() == df["EXT_SOURCE_1_MISSING"].tolist()
    )
    # D15: finalize_for_random_forest must also close every remaining
    # categorical column and every remaining NaN, full stop —
    # RandomForestClassifier.fit() rejects both.
    assert len(result.select_dtypes(include="object").columns) == 0
    assert result.isna().sum().sum() == 0


def test_encode_categoricals_for_random_forest_assigns_integer_codes() -> None:
    df = impute_occupation_type(_sample_df())
    result = encode_categoricals_for_random_forest(df)

    assert len(result.select_dtypes(include="object").columns) == 0
    assert pd.api.types.is_integer_dtype(result["OCCUPATION_TYPE"].dtype)
    # Same string -> same code, distinct strings -> distinct codes.
    assert result.loc[0, "OCCUPATION_TYPE"] == result.loc[4, "OCCUPATION_TYPE"]
    assert result.loc[0, "OCCUPATION_TYPE"] != result.loc[3, "OCCUPATION_TYPE"]
    # No NaN survives to this step in the real pipeline (impute_occupation_type
    # already filled it with "Missing"), so factorize never emits -1 here.
    assert (result["OCCUPATION_TYPE"] == -1).sum() == 0


def test_impute_days_employed_fills_with_median_of_genuine_values() -> None:
    df = fix_days_employed(_sample_df())
    result = impute_days_employed(df)

    genuine = df["DAYS_EMPLOYED"].dropna()
    assert result["DAYS_EMPLOYED"].isna().sum() == 0
    assert result.loc[2, "DAYS_EMPLOYED"] == genuine.median()


def test_impute_minor_missing_fills_numeric_median_and_suite_category() -> None:
    df = _sample_df()
    result = impute_minor_missing(df)

    assert result["AMT_GOODS_PRICE"].isna().sum() == 0
    assert result.loc[1, "AMT_GOODS_PRICE"] == df["AMT_GOODS_PRICE"].median()
    assert result["AMT_ANNUITY"].isna().sum() == 0
    assert result["CNT_FAM_MEMBERS"].isna().sum() == 0
    assert result["DAYS_LAST_PHONE_CHANGE"].isna().sum() == 0
    assert result["NAME_TYPE_SUITE"].isna().sum() == 0
    assert result.loc[1, "NAME_TYPE_SUITE"] == "Missing"


def test_build_clean_features_leaves_unexplained_group_as_nan_for_lightgbm() -> None:
    df = _sample_df()
    result = build_clean_features(df)

    # LightGBM path: these stay NaN on purpose, handled by native missing support.
    assert result["EXT_SOURCE_1"].isna().sum() == 2
    assert result["AMT_REQ_CREDIT_BUREAU_YEAR"].isna().sum() == 2
    # "Partially explained" group is fully imputed regardless of model.
    assert result["APARTMENTS_MEDI"].isna().sum() == 0
    assert result["OCCUPATION_TYPE"].isna().sum() == 0
    assert "HOUSING_INFO_MISSING" in result.columns
    assert "EXT_SOURCE_1_MISSING" in result.columns
    # D13: DAYS_EMPLOYED and the five minor columns are fully closed out.
    assert result["DAYS_EMPLOYED"].isna().sum() == 0
    assert result["AMT_GOODS_PRICE"].isna().sum() == 0
    assert result["NAME_TYPE_SUITE"].isna().sum() == 0


def test_build_clean_features_end_to_end_on_synthetic_df() -> None:
    df = _sample_df()
    result = build_clean_features(df)

    assert not (result["DAYS_EMPLOYED"] == 365243).any()
    assert result["FLAG_OWN_CAR"].dtype == int
    assert "FLAG_MOBIL" not in result.columns
    assert "APARTMENTS_AVG" not in result.columns
    assert result["OWN_CAR_AGE"].isna().sum() == 0
    assert result["AMT_INCOME_TOTAL"].max() < df["AMT_INCOME_TOTAL"].max()
    assert "DOCUMENTS_PROVIDED_COUNT" in result.columns


def test_build_summary_lists_dropped_and_added_columns() -> None:
    df = _sample_df()
    treated = build_clean_features(df)
    summary = build_summary(df, treated)

    assert "FLAG_MOBIL" in summary
    assert "DOCUMENTS_PROVIDED_COUNT" in summary
    assert "raw shape" in summary
    assert "treated shape" in summary


def test_write_outputs_creates_both_csvs_and_summary(tmp_path) -> None:
    df = _sample_df()
    treated = build_clean_features(df)
    output_path = tmp_path / "train_treated_lightgbm.csv"
    rf_output_path = tmp_path / "train_treated_rf.csv"
    summary_path = tmp_path / "cleaning_summary.md"

    write_outputs(
        df,
        treated,
        output_path=output_path,
        summary_path=summary_path,
        rf_output_path=rf_output_path,
    )

    assert output_path.exists()
    assert rf_output_path.exists()
    assert summary_path.exists()
    reloaded = pd.read_csv(output_path)
    assert len(reloaded) == len(treated)
    rf_reloaded = pd.read_csv(rf_output_path)
    assert rf_reloaded["EXT_SOURCE_1"].isna().sum() == 0
    # D15: the RF csv must be fully numeric on disk, not just NaN-free.
    assert len(rf_reloaded.select_dtypes(include="object").columns) == 0


# D29: regression tests proving the leak fix actually restricts the fit —
# each one picks a train_index that excludes a row which would visibly
# move the full-df statistic, then asserts the filled/capped value came
# from the restricted set, not the whole frame.


def test_impute_days_employed_respects_train_index() -> None:
    df = fix_days_employed(_sample_df())
    train_index = pd.Index([0, 1, 3])  # excludes row 5's genuine value (-50)
    result = impute_days_employed(df, train_index=train_index)

    train_only_median = df.loc[train_index, "DAYS_EMPLOYED"].median()
    full_df_median = df["DAYS_EMPLOYED"].median()
    assert train_only_median != full_df_median
    assert result.loc[2, "DAYS_EMPLOYED"] == train_only_median


def test_cap_at_percentile_respects_fit_index() -> None:
    series = pd.Series([1.0, 2.0, 3.0, 1000.0])
    fit_index = pd.Index([0, 1, 2])  # excludes the outlier itself
    capped, threshold = cap_at_percentile(series, percentile=0.5, fit_index=fit_index)

    assert threshold == series.loc[fit_index].quantile(0.5)
    assert threshold != series.quantile(0.5)
    # The row excluded from fitting is still capped using that threshold.
    assert capped.max() == threshold


def test_cap_income_outliers_respects_train_index() -> None:
    df = _sample_df()
    train_index = df.index[df.index != 5]  # excludes the 1,000,000 income row
    result = cap_income_outliers(df, train_index=train_index)

    train_only_cap = df.loc[train_index, "AMT_INCOME_TOTAL"].quantile(0.999)
    full_df_cap = df["AMT_INCOME_TOTAL"].quantile(0.999)
    assert train_only_cap != full_df_cap
    assert result["AMT_INCOME_TOTAL"].max() == train_only_cap


def test_impute_housing_block_respects_train_index() -> None:
    df = _sample_df()
    train_index = pd.Index([0, 1, 3, 4])  # excludes row 2 (NaN) and row 5
    result = impute_housing_block(df, train_index=train_index)

    train_only_median = df.loc[train_index, "APARTMENTS_MEDI"].median()
    full_df_median = df["APARTMENTS_MEDI"].median()
    assert train_only_median != full_df_median
    assert result.loc[2, "APARTMENTS_MEDI"] == train_only_median


def test_impute_minor_missing_respects_train_index() -> None:
    df = _sample_df()
    train_index = pd.Index([0, 2, 4, 5])  # excludes row 1 (NaN) and row 3
    result = impute_minor_missing(df, train_index=train_index)

    train_only_median = df.loc[train_index, "AMT_GOODS_PRICE"].median()
    full_df_median = df["AMT_GOODS_PRICE"].median()
    assert train_only_median != full_df_median
    assert result.loc[1, "AMT_GOODS_PRICE"] == train_only_median


def test_finalize_for_random_forest_respects_train_index() -> None:
    df = build_clean_features(_sample_df())
    train_index = pd.Index([0, 2, 5])  # the 3 rows with a non-null EXT_SOURCE_1
    result = finalize_for_random_forest(df, train_index=train_index)

    train_only_median = df.loc[train_index, "EXT_SOURCE_1"].median()
    full_df_median = df["EXT_SOURCE_1"].median()
    assert train_only_median != full_df_median
    assert result.loc[1, "EXT_SOURCE_1"] == train_only_median


def test_fit_train_index_matches_downstream_split_holdout_partition() -> None:
    """The whole D29 fix depends on this holding: the partition computed
    here, on the raw frame before any column is dropped, must be the
    identical set of row positions that train_random_forest.py's/
    train_lightgbm.py's own split_holdout() computes later on the
    already-written, column-reshuffled CSV.
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

    # Mirrors split_holdout(): drop ID/TARGET, then stratified-split X/y
    # with the same RANDOM_STATE/TEST_SIZE.
    y = df["TARGET"]
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    from sklearn.model_selection import train_test_split

    X_train, _, _, _ = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )
    assert set(train_index) == set(X_train.index)
