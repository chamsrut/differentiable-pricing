"""Task 9H predeclared pricing tolerance: its arithmetic, checked not asserted.

The tolerance's two derivations are numerical claims -- the shortest-maturity
at-the-money vega transmission, and the Bonferroni bar's crossover row count --
so they are verified here rather than left in prose. Nothing opens a partition,
trains anything or prices a lattice.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.representation import (
    discounted_spot,
    european_price,
)
from differentiable_pricing.ml.american_dev.tolerance import (
    CALIBRATION_FAMILY_WISE_ALPHA,
    CALIBRATION_RATIO,
    DECLARED_BEFORE_ANALYSIS,
    DOMAIN_GAP_MAXIMUM_POSITIVE,
    DOMAIN_GAP_MEDIAN,
    DOMAIN_GAP_MINIMUM_SIGNED,
    GATED_SLICE_FAMILY,
    PASS_RATE_BAR,
    RESOLVABILITY_SAFETY_FACTOR,
    SECONDARY_NORMALIZED_TOLERANCE,
    VOLATILITY_TOLERANCE,
    ToleranceError,
    assert_slice_family_matches,
    bias_spread,
    black_scholes_vega,
    bonferroni_z,
    calibration_bar,
    calibration_statistics,
    comparison_floors,
    declaration,
    iv_in_scope,
    label_discretization_proxy,
    meets_primary,
    meets_secondary,
    volatility_equivalent_error,
)

#: The declared domain's shortest maturity: seven calendar days.
SHORTEST_MATURITY: Final = 0.019178082191780823
#: The declared domain's tightest lower bound on ``A = S * exp(-q * T)``.
MINIMUM_DISCOUNTED_SPOT: Final = 50.0 * math.exp(-0.12 * 3.0)


def _states() -> torch.Tensor:
    return torch.tensor(
        [
            [-1.0, 100.0, 100.0, 1.0, 0.05, 0.02, 0.20],
            [1.0, 120.0, 90.0, 0.10, 0.03, 0.00, 0.40],
            [-1.0, 60.0, 140.0, 2.50, -0.01, 0.08, 0.70],
            [1.0, 100.0, 100.0, SHORTEST_MATURITY, 0.05, 0.00, 0.05],
        ],
        dtype=torch.float64,
    )


# ---------------------------------------------------------------------------
# The declaration is a constant, not an argument
# ---------------------------------------------------------------------------


def test_the_tolerance_is_declared_before_the_analysis() -> None:
    assert DECLARED_BEFORE_ANALYSIS is True
    record = declaration()
    assert record["declared_before_analysis"] is True
    assert record["declared_before_h1_was_evaluated"] is True
    assert VOLATILITY_TOLERANCE == 0.01
    assert PASS_RATE_BAR == 0.99
    assert RESOLVABILITY_SAFETY_FACTOR == 3.0
    assert SECONDARY_NORMALIZED_TOLERANCE == 5.5e-4


def test_the_declaration_records_what_it_deliberately_does_not_use() -> None:
    record = declaration()
    assert "no market data" in record["deliberately_not_used"]
    assert "non-binding" in record["deliberately_not_used"]
    assert "unchanged and unrevised" in record["relation_to_existing_gates"]
    assert "not a calibration-readiness test" in record["calibration"]["status"]


def test_no_absolute_price_floor_below_the_domains_own_bound_could_ever_bind() -> None:
    """The reason an absolute floor was dropped, as arithmetic rather than prose."""
    assert MINIMUM_DISCOUNTED_SPOT > 34.0
    assert SECONDARY_NORMALIZED_TOLERANCE * MINIMUM_DISCOUNTED_SPOT > 0.01


# ---------------------------------------------------------------------------
# Vega, and the secondary screen's derivation
# ---------------------------------------------------------------------------


def test_vega_is_the_derivative_of_the_deployed_european_leg() -> None:
    """One definition: the denominator and the floor's European leg agree."""
    states = _states()
    vega = black_scholes_vega(states)
    bumped = states.clone()
    step = 1.0e-6
    bumped[:, 6] += step
    lowered = states.clone()
    lowered[:, 6] -= step
    with torch.no_grad():
        central = (european_price(bumped) - european_price(lowered)) / (2.0 * step)
    assert torch.allclose(vega, central, rtol=1e-6, atol=1e-8)
    assert bool((vega > 0.0).all())
    assert not vega.requires_grad


def test_the_secondary_screen_is_sufficient_at_the_money_at_every_maturity() -> None:
    """5.5e-4 normalized transmits to at most 0.01 volatility at the money.

    The binding case is the domain's shortest maturity, where at-the-money
    normalized vega is smallest. This asserts the claim the screen rests on
    instead of restating it.
    """
    maturities = np.concatenate(
        [np.array([SHORTEST_MATURITY]), np.linspace(0.05, 3.0, 40)]
    )
    worst = 0.0
    for maturity in maturities:
        for volatility in (0.05, 0.2, 0.5, 0.8):
            state = torch.tensor(
                [[1.0, 100.0, 100.0, float(maturity), 0.0, 0.0, float(volatility)]],
                dtype=torch.float64,
            )
            normalized_vega = float(
                black_scholes_vega(state) / discounted_spot(state).detach()
            )
            worst = max(worst, SECONDARY_NORMALIZED_TOLERANCE / normalized_vega)
    assert worst <= VOLATILITY_TOLERANCE


def test_the_minimum_at_the_money_normalized_vega_is_the_declared_value() -> None:
    state = torch.tensor(
        [[1.0, 100.0, 100.0, SHORTEST_MATURITY, 0.0, 0.0, 0.2]], dtype=torch.float64
    )
    normalized_vega = float(black_scholes_vega(state) / discounted_spot(state).detach())
    assert normalized_vega == pytest.approx(0.0552, abs=5e-4)


# ---------------------------------------------------------------------------
# The primary tolerance and the per-row eligibility rule
# ---------------------------------------------------------------------------


def test_the_volatility_equivalent_error_is_a_price_error_over_vega() -> None:
    reference = np.array([10.0, 4.0])
    prediction = np.array([10.5, 3.9])
    vega = np.array([50.0, 20.0])
    assert np.allclose(
        volatility_equivalent_error(prediction, reference, vega), [0.01, 0.005]
    )
    assert list(meets_primary(prediction, reference, vega)) == [True, True]
    assert list(meets_primary(np.array([11.0, 4.0]), reference, vega)) == [False, True]


def test_a_non_positive_vega_is_refused_rather_than_dividing_by_zero() -> None:
    with pytest.raises(ToleranceError, match="iv_in_scope"):
        volatility_equivalent_error(np.array([1.0]), np.array([1.0]), np.array([0.0]))


def test_eligibility_is_evaluated_per_row_against_that_rows_own_gap() -> None:
    states = _states()
    analytic = label_discretization_proxy(states, np.zeros(states.shape[0]))
    assert analytic.shape == (states.shape[0],)
    with torch.no_grad():
        european = european_price(states).numpy()
    # A stored comparator equal to the analytic value leaves a zero gap, which
    # every positive vega resolves; a hugely inflated one resolves nowhere.
    resolved, terms = iv_in_scope(states, european)
    assert bool(resolved.all())
    assert terms["safety_factor"] == RESOLVABILITY_SAFETY_FACTOR
    assert terms["model_free"] is True
    assert terms["label_free"] is False
    unresolved, _ = iv_in_scope(states, european + 1.0e3)
    assert not bool(unresolved.any())


def test_the_two_comparison_floors_are_reported_side_by_side() -> None:
    """One-sided and two-sided differ by 6.7x; reporting one alone would mislead."""
    floors = comparison_floors(_states())
    assert set(floors) == {"one_sided", "two_sided"}
    ratio = float((floors["two_sided"] / floors["one_sided"])[0])
    assert ratio == pytest.approx(
        abs(DOMAIN_GAP_MINIMUM_SIGNED) / DOMAIN_GAP_MAXIMUM_POSITIVE
    )
    assert ratio > 6.0


def test_the_recorded_domain_statistics_reproduce_the_over_strictness_ratio() -> None:
    record = declaration()["domain_gap_statistics"]
    assert record["maximum_over_median"] == pytest.approx(
        DOMAIN_GAP_MAXIMUM_POSITIVE / DOMAIN_GAP_MEDIAN
    )
    assert record["maximum_over_median"] == pytest.approx(134.09, abs=0.01)
    assert "not over partition rows" in record["note"]


def test_the_secondary_screen_is_flat_in_normalized_units() -> None:
    errors = np.array([0.0, 5.4e-4, 5.5e-4, 5.6e-4, -5.4e-4])
    assert list(meets_secondary(errors)) == [True, True, True, False, True]


# ---------------------------------------------------------------------------
# The calibration objective
# ---------------------------------------------------------------------------


def test_the_gated_slice_family_is_frozen_and_excludes_endogenous_slices() -> None:
    assert len(GATED_SLICE_FAMILY) == 19
    assert len(set(GATED_SLICE_FAMILY)) == len(GATED_SLICE_FAMILY)
    assert "iv_scope:in_scope" in GATED_SLICE_FAMILY
    for name in GATED_SLICE_FAMILY:
        assert not name.startswith("projection:")
        assert not name.startswith("floor_leg:")


def test_the_frozen_family_must_match_the_slices_actually_produced() -> None:
    assert_slice_family_matches(GATED_SLICE_FAMILY)
    with pytest.raises(ToleranceError, match="did not produce"):
        assert_slice_family_matches(GATED_SLICE_FAMILY[:-1])


def test_the_bonferroni_bar_is_two_sided_over_the_frozen_family() -> None:
    z = bonferroni_z(len(GATED_SLICE_FAMILY), CALIBRATION_FAMILY_WISE_ALPHA)
    assert z == pytest.approx(3.0078, abs=1e-3)
    assert bonferroni_z(1, 0.05) == pytest.approx(1.95996, abs=1e-4)
    with pytest.raises(ToleranceError):
        bonferroni_z(0)


def test_the_noise_term_binds_below_the_declared_row_count() -> None:
    """Substantive above ~144 rows, noise-aware below, with no discontinuity."""
    crossover = declaration()["calibration"]["noise_term_binds_below_rows"]
    assert crossover == 145
    assert calibration_bar(crossover - 1) > CALIBRATION_RATIO
    assert calibration_bar(crossover + 200) == CALIBRATION_RATIO
    assert calibration_bar(25_000) == CALIBRATION_RATIO


def test_the_noise_aware_bar_delivers_the_family_wise_rate_it_claims() -> None:
    """A zero-bias slice fails at the corrected per-slice rate, not more.

    The bar controls family-wise error at 5% over 19 slices, so one slice's
    false-failure rate is ``0.05 / 19 = 0.26%`` -- about one draw in 400, not
    zero. What matters is that the correction is doing real work: judged against
    the substantive 0.25 alone, a 40-row slice with no bias at all would fail
    roughly one time in nine.
    """
    draws, rows = 4_000, 40
    generator = np.random.default_rng(20260823)
    corrected = uncorrected = 0
    for _ in range(draws):
        statistics = calibration_statistics(generator.normal(0.0, 1.0e-3, size=rows))
        corrected += not statistics["passes"]
        uncorrected += statistics["bias_to_rmse"] > CALIBRATION_RATIO
    expected = CALIBRATION_FAMILY_WISE_ALPHA / len(GATED_SLICE_FAMILY)
    assert corrected / draws == pytest.approx(expected, abs=3.0e-3)
    assert uncorrected > 20 * corrected


def test_a_genuinely_biased_slice_fails() -> None:
    generator = np.random.default_rng(7)
    errors = generator.normal(1.0e-3, 1.0e-3, size=5_000)
    statistics = calibration_statistics(errors)
    assert statistics["bias_to_rmse"] > CALIBRATION_RATIO
    assert statistics["passes"] is False
    assert statistics["bar_is_noise_limited"] is False
    assert statistics["noise_floor"] == pytest.approx(1.0 / math.sqrt(5_000))


def test_the_bias_spread_reports_the_surface_distorting_statistic() -> None:
    statistics = {
        "moneyness:-inf:-0.2": {"rows": 100, "signed_bias": -2.0e-4},
        "moneyness:-0.2:0.2": {"rows": 100, "signed_bias": 1.0e-4},
        "moneyness:0.2:inf": {"rows": 100, "signed_bias": 3.0e-4},
        "expiry:-inf:0.25": {"rows": 0, "signed_bias": 0.0},
    }
    spread = bias_spread(statistics, "moneyness")
    assert spread["members"] == 3
    assert spread["spread"] == pytest.approx(5.0e-4)
    assert spread["maximum"]["slice"] == "moneyness:0.2:inf"
    assert spread["minimum"]["slice"] == "moneyness:-inf:-0.2"
    assert bias_spread(statistics, "expiry")["spread"] is None


def test_an_empty_slice_is_refused_rather_than_scored() -> None:
    with pytest.raises(ToleranceError):
        calibration_statistics(np.array([]))
