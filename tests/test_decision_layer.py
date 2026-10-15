import numpy as np
import pytest

from scripts.decision_layer import (
    _realized_cost,
    bayes_threshold,
    cost_curve,
    pricing_validation,
    required_rate,
)


def test_bayes_threshold_matches_closed_form() -> None:
    assert bayes_threshold(k=4) == pytest.approx(0.2)
    assert bayes_threshold(k=9) == pytest.approx(0.1)
    assert bayes_threshold(k=1) == pytest.approx(0.5)


def test_bayes_threshold_is_a_pure_function_of_k() -> None:
    # The leakage guard (D31/D29) depends on this literally being true:
    # bayes_threshold takes no data argument at all, so there is nothing
    # for y_test to leak into even by accident.
    import inspect

    params = list(inspect.signature(bayes_threshold).parameters)
    assert params == ["k"]


def test_required_rate_at_zero_risk_equals_margin() -> None:
    r = required_rate(np.array([0.0]), lgd=0.75, margin=0.05)
    assert r[0] == pytest.approx(0.05)


def test_required_rate_increases_with_p_and_with_lgd() -> None:
    p = np.array([0.05, 0.2, 0.5])
    low_lgd = required_rate(p, lgd=0.5, margin=0.05)
    high_lgd = required_rate(p, lgd=1.0, margin=0.05)

    assert (np.diff(low_lgd) > 0).all()  # increasing in p
    assert (high_lgd > low_lgd).all()  # increasing in LGD, same p


def test_required_rate_handles_p_near_one_without_raising() -> None:
    r = required_rate(np.array([1.0, 1 - 1e-12]), lgd=0.75, margin=0.05)
    assert np.isfinite(r).all()
    assert (r > 0).all()


def test_realized_cost_matches_manual_calculation() -> None:
    # 4 applicants: approve iff proba < 0.2 (k=4).
    y_true = np.array([1, 0, 1, 0])
    proba = np.array([0.1, 0.1, 0.5, 0.5])
    t = 0.2
    k = 4.0
    # approved: idx 0,1 (proba<0.2); declined: idx 2,3.
    # idx0: approved & defaulted -> costs k=4. idx1: approved & good -> 0.
    # idx2: declined & defaulted -> 0. idx3: declined & good -> costs 1.
    # total = 4 + 1 = 5, over n=4 -> 1.25
    cost = _realized_cost(y_true, proba, t, k)
    assert cost == pytest.approx(5.0 / 4.0)


def test_cost_curve_approval_rate_is_non_increasing_in_k() -> None:
    rng = np.random.default_rng(0)
    n = 2000
    proba = rng.random(n)
    y_true = (rng.random(n) < proba).astype(int)

    curve = cost_curve(y_true, proba, k_values=[3.0, 5.0, 10.0, 20.0])
    approval_rates = curve.sort_values("k")["approval_rate"].to_numpy()
    # Higher k -> costlier default relative to a missed good loan ->
    # stricter (lower) threshold -> approve no more than before.
    assert (np.diff(approval_rates) <= 1e-12).all()


def test_cost_curve_has_expected_columns() -> None:
    y_true = np.array([0, 1, 0, 1, 0])
    proba = np.array([0.1, 0.4, 0.2, 0.6, 0.05])
    curve = cost_curve(y_true, proba, k_values=[5.0])
    expected = {
        "k",
        "t_star",
        "approval_rate",
        "approval_rate_ci95",
        "bad_rate_among_approved",
        "good_rate_among_declined",
        "realized_cost_per_applicant",
        "realized_cost_ci95",
        "baseline_approve_all_cost",
        "baseline_decline_all_cost",
        "n_test",
    }
    assert set(curve.columns) == expected


def test_pricing_validation_recovers_perfect_agreement() -> None:
    # Construct real_rate to exactly equal required_rate(p, lgd, margin)
    # for every row -> correlation must come out ~1.0.
    rng = np.random.default_rng(1)
    p = rng.uniform(0.01, 0.3, size=200)
    lgd, margin = 0.75, 0.05
    real_rate = required_rate(p, lgd, margin)
    amt_credit = np.ones(200)
    amt_annuity = real_rate * amt_credit

    result = pricing_validation(p, amt_credit, amt_annuity, [lgd], margin)
    row = result.iloc[0]
    assert row["pearson_corr"] == pytest.approx(1.0, abs=1e-6)
    assert row["spearman_corr"] == pytest.approx(1.0, abs=1e-6)
    assert row["n_dropped_invalid_amt"] == 0


def test_pricing_validation_drops_invalid_amt_credit() -> None:
    p = np.array([0.1, 0.1, 0.1, 0.1, 0.2])
    amt_credit = np.array([1.0, 0.0, -5.0, np.nan, 2.0])
    amt_annuity = np.array([0.1, 0.1, 0.1, 0.1, 0.2])

    result = pricing_validation(p, amt_credit, amt_annuity, [0.75], 0.05)
    row = result.iloc[0]
    assert row["n_applicants"] == 2
    assert row["n_dropped_invalid_amt"] == 3


def test_pricing_validation_below_two_valid_rows_reports_nan_not_crash() -> None:
    # pearsonr/spearmanr require length >= 2 -- a single valid row must
    # degrade to NaN correlations, never raise.
    p = np.array([0.1, 0.1])
    amt_credit = np.array([1.0, np.nan])
    amt_annuity = np.array([0.1, 0.1])

    result = pricing_validation(p, amt_credit, amt_annuity, [0.75], 0.05)
    row = result.iloc[0]
    assert row["n_applicants"] == 1
    assert np.isnan(row["pearson_corr"])
    assert np.isnan(row["spearman_corr"])
