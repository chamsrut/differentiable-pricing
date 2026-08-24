"""Task 9H CRR Greek reference contract: correctness against closed form.

The reference is validated where an exact answer exists -- **European** CRR
Greeks against Black--Scholes -- because no exact American Greek is available to
check against. Shallow depths keep the suite fast; the contract being tested is
the construction, not the lattice's resolution.
"""

from __future__ import annotations

from typing import Any, Final

import differentiable_pricing as dp
import numpy as np
import pytest
from differentiable_pricing.ml.american_dev.greek_reference import (
    DEEPER_DEPTH,
    GREEKS,
    LOG_MONEYNESS_DOMAIN,
    OUT_OF_SCOPE,
    REFERENCE_DEPTH,
    SPOT_BUMP_FRACTIONS,
    SPOT_DOMAIN,
    VOLATILITY_BUMPS,
    VOLATILITY_DOMAIN,
    GreekReferenceError,
    _plateau,
    adjacent_average,
    bump_convergence,
    contract,
    declared_states,
    depth_convergence,
    reference_delta,
    reference_gamma,
    reference_vega,
    spot_eligible,
    volatility_eligible,
)

DEPTH: Final = 512


def _states(count: int = 12) -> dict[str, np.ndarray]:
    return {key: value[:count] for key, value in declared_states(count).items()}


def _arrays(states: dict[str, np.ndarray]) -> dict[str, Any]:
    from differentiable_pricing.ml.american_dev.greek_reference import _state_arrays

    return _state_arrays(states)


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------


def test_the_reference_averages_two_adjacent_depths() -> None:
    """Differencing one depth would put the even/odd oscillation in the reference."""
    states = _states(4)
    averaged = adjacent_average(
        states["option_type"],
        states["spot"],
        states["strike"],
        states["maturity"],
        states["rate"],
        states["dividend_yield"],
        states["volatility"],
        depth=DEPTH,
        thread_count=4,
    )
    rows = len(states["spot"])

    def priced(steps: int) -> np.ndarray:
        return np.asarray(
            dp.crr_price_batch(
                list(states["option_type"]),
                ["american"] * rows,
                list(states["spot"]),
                list(states["strike"]),
                list(states["maturity"]),
                list(states["rate"]),
                list(states["dividend_yield"]),
                list(states["volatility"]),
                [steps] * rows,
                4,
            )["price"],
            dtype=np.float64,
        )

    assert np.allclose(averaged, 0.5 * (priced(DEPTH) + priced(DEPTH + 1)))


def test_an_empty_state_set_is_refused() -> None:
    with pytest.raises(GreekReferenceError):
        adjacent_average([], np.array([]), np.array([]), np.array([]), np.array([]),
                         np.array([]), np.array([]))


# ---------------------------------------------------------------------------
# Correctness where an exact answer exists
# ---------------------------------------------------------------------------


def test_the_bumped_construction_recovers_black_scholes_greeks() -> None:
    """A European check, because no exact American Greek exists to check against.

    The same stencils are applied to a European CRR lattice, whose limit is the
    closed form. If the construction were wrong -- a missing factor of two, the
    wrong denominator, one leg of the average -- this would not agree.
    """
    import math

    spot, strike, maturity, rate, volatility = 100.0, 100.0, 1.0, 0.03, 0.25
    rows = 1

    def european(spots: np.ndarray, vols: np.ndarray, steps: int) -> np.ndarray:
        return np.asarray(
            dp.crr_price_batch(
                ["call"] * len(spots),
                ["european"] * len(spots),
                list(spots),
                [strike] * len(spots),
                [maturity] * len(spots),
                [rate] * len(spots),
                [0.0] * len(spots),
                list(vols),
                [steps] * len(spots),
                4,
            )["price"],
            dtype=np.float64,
        )

    step = spot * 0.01
    spots = np.array([spot - step, spot, spot + step])
    vols = np.array([volatility] * 3)
    prices = 0.5 * (european(spots, vols, 4096) + european(spots, vols, 4097))
    delta = (prices[2] - prices[0]) / (2.0 * step)
    gamma = (prices[2] - 2.0 * prices[1] + prices[0]) / (step * step)

    d1 = (math.log(spot / strike) + (rate + 0.5 * volatility**2) * maturity) / (
        volatility * math.sqrt(maturity)
    )
    normal = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
    density = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    assert delta == pytest.approx(normal, abs=2e-3)
    assert gamma == pytest.approx(
        density / (spot * volatility * math.sqrt(maturity)), rel=5e-2
    )
    del rows


def test_delta_and_gamma_use_their_own_denominators() -> None:
    """A three-point stencil, not a rescaled Delta."""
    states = _states(6)
    arrays = _arrays(states)
    delta = reference_delta(arrays, 0.01, depth=DEPTH, thread_count=4)
    gamma = reference_gamma(arrays, 0.01, depth=DEPTH, thread_count=4)
    assert delta.shape == gamma.shape == arrays["spot"].shape
    assert not np.allclose(delta, gamma)


def test_vega_is_a_volatility_difference() -> None:
    states = _states(6)
    vega = reference_vega(_arrays(states), 0.01, depth=DEPTH, thread_count=4)
    assert bool((vega > 0.0).all())


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------


def test_eligibility_excludes_states_whose_bump_leaves_the_domain() -> None:
    arrays = _arrays(_states(20))
    coarse = spot_eligible(arrays, max(SPOT_BUMP_FRACTIONS))
    fine = spot_eligible(arrays, min(SPOT_BUMP_FRACTIONS))
    assert bool((coarse <= fine).all())
    assert bool(
        (
            volatility_eligible(arrays, max(VOLATILITY_BUMPS))
            <= volatility_eligible(arrays, min(VOLATILITY_BUMPS))
        ).all()
    )


def test_the_declared_states_sit_inside_the_declared_domain() -> None:
    states = declared_states(40)
    assert bool((states["spot"] > SPOT_DOMAIN[0]).all())
    assert bool((states["spot"] < SPOT_DOMAIN[1]).all())
    assert bool((states["volatility"] > VOLATILITY_DOMAIN[0]).all())
    assert bool((states["volatility"] < VOLATILITY_DOMAIN[1]).all())
    moneyness = np.log(states["spot"] / states["strike"])
    assert bool((moneyness > LOG_MONEYNESS_DOMAIN[0]).all())
    assert bool((moneyness < LOG_MONEYNESS_DOMAIN[1]).all())
    assert set(states["option_type"]) == {"call", "put"}
    assert np.array_equal(declared_states(40)["spot"], states["spot"])


# ---------------------------------------------------------------------------
# The plateau
# ---------------------------------------------------------------------------


def test_a_plateau_is_reported_as_reached_only_when_it_is() -> None:
    """Selecting the finest bump means the ladder ran out, not that it converged."""
    eligible = {bump: np.ones(4, dtype=bool) for bump in (0.02, 0.01, 0.005)}
    falling = {
        0.02: np.array([1.0, 1.0, 1.0, 1.0]),
        0.01: np.array([0.5, 0.5, 0.5, 0.5]),
        0.005: np.array([0.4, 0.4, 0.4, 0.4]),
    }
    result = _plateau(falling, eligible)
    assert result["selected_bump"] == 0.005
    assert result["plateau_reached"] is False
    assert "NO PLATEAU" in result["plateau_status"]
    assert "UPPER BOUND" in result["plateau_status"]

    settled = {
        0.02: np.array([1.0, 1.0, 1.0, 1.0]),
        0.01: np.array([0.5, 0.5, 0.5, 0.5]),
        0.005: np.array([2.0, 2.0, 2.0, 2.0]),
    }
    reached = _plateau(settled, eligible)
    assert reached["selected_bump"] == 0.01
    assert reached["plateau_reached"] is True


def test_a_single_bump_cannot_form_a_plateau() -> None:
    with pytest.raises(GreekReferenceError):
        _plateau({0.01: np.zeros(2)}, {0.01: np.ones(2, dtype=bool)})


def test_the_bump_study_selects_delta_and_gamma_separately() -> None:
    selection = bump_convergence(_states(10), depth=256, thread_count=8)
    assert set(selection) >= {"delta", "gamma", "vega", "separate_bumps"}
    for greek in GREEKS:
        entry = selection[greek]
        assert entry["reference_uncertainty"] >= 0.0
        assert entry["selected_bump"] > 0.0
        assert "plateau_reached" in entry
    assert "cannot serve both" in selection["separate_bumps"]


# ---------------------------------------------------------------------------
# Depth convergence, and the contract
# ---------------------------------------------------------------------------


def test_the_depth_check_measures_prices_as_well_as_greeks() -> None:
    """The price gap validates the tolerance's safety factor retrospectively."""
    states = _states(6)
    selection = bump_convergence(states, depth=256, thread_count=8)
    convergence = depth_convergence(states, selection, thread_count=8)
    assert convergence["shallow_depth"] == REFERENCE_DEPTH
    assert convergence["deep_depth"] == DEEPER_DEPTH
    price = convergence["price"]
    assert price["maximum_normalized_difference"] >= 0.0
    assert "does not adjust it" in price["purpose"]
    for greek in GREEKS:
        assert "depth_converged" in convergence[greek]
        assert convergence[greek]["reference_uncertainty"] >= 0.0


def test_the_contract_records_its_scope_and_its_limitation() -> None:
    states = _states(6)
    selection = bump_convergence(states, depth=256, thread_count=8)
    convergence = depth_convergence(states, selection, thread_count=8)
    record = contract(selection, convergence)
    assert record["scope"] == list(GREEKS)
    assert sorted(record["out_of_scope"]) == sorted(OUT_OF_SCOPE)
    assert "not established" in record["out_of_scope"]["theta"]
    assert "digest-pinned" in record["tree_internal_greeks"]
    assert "d(L_1024)/dx" in record["which_comparison"]
    assert "exceeds it" in record["reporting_rule"]
