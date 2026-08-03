"""Cross-language tests for the task 9C-A finite-difference price oracle.

These exercise the pybind11 boundary of the `_pde` extension: the deterministic
solver itself is tested again in `cpp/tests/test_main.cpp`, and neither suite
substitutes a Python reimplementation of the pricing recursion for it.
"""

from __future__ import annotations

import itertools
import math
import threading
from collections.abc import Callable

import pytest
from differentiable_pricing import (
    PsorConvergenceError,
    black_scholes,
    crr_price,
    pde_discount_factor,
    pde_log_discount,
    pde_post_dividend_spot,
    pde_price,
    pde_segment_rate,
)

# The PSOR tolerance bounds the residual of one implicit solve, so the solver's
# contribution to a price accumulates over time steps. At the settings below it
# is observed near 1e-7. Comparisons that are exact in exact arithmetic are
# given this band; a real exercise premium in these fixtures is far larger.
SOLVER_BAND = 1.0e-6


def _adaptive_simpson(
    function: Callable[[float], float],
    left: float,
    right: float,
    *,
    absolute_tolerance: float,
    relative_tolerance: float,
    maximum_depth: int = 32,
) -> float:
    """Integrate a smooth finite interval without using any PDE grid state."""

    def simpson(
        a: float,
        b: float,
        left_value: float,
        midpoint_value: float,
        right_value: float,
    ) -> float:
        return (b - a) * (left_value + 4.0 * midpoint_value + right_value) / 6.0

    left_value = function(left)
    midpoint = 0.5 * (left + right)
    midpoint_value = function(midpoint)
    right_value = function(right)
    whole = simpson(left, right, left_value, midpoint_value, right_value)

    def recurse(
        a: float,
        b: float,
        fa: float,
        fm: float,
        fb: float,
        estimate: float,
        tolerance: float,
        depth: int,
    ) -> float:
        middle = 0.5 * (a + b)
        left_middle = 0.5 * (a + middle)
        right_middle = 0.5 * (middle + b)
        flm = function(left_middle)
        frm = function(right_middle)
        left_estimate = simpson(a, middle, fa, flm, fm)
        right_estimate = simpson(middle, b, fm, frm, fb)
        refined = left_estimate + right_estimate
        error = abs(refined - estimate)
        if depth <= 0 or error <= 15.0 * tolerance:
            return refined + (refined - estimate) / 15.0
        return recurse(
            a, middle, fa, flm, fm, left_estimate, 0.5 * tolerance, depth - 1
        ) + recurse(
            middle, b, fm, frm, fb, right_estimate, 0.5 * tolerance, depth - 1
        )

    tolerance = max(absolute_tolerance, relative_tolerance * abs(whole))
    return recurse(
        left,
        right,
        left_value,
        midpoint_value,
        right_value,
        whole,
        tolerance,
        maximum_depth,
    )


def _normal_expectation(
    function: Callable[[float], float],
    *,
    breakpoints: list[float],
    sigma_limit: float,
    absolute_tolerance: float,
    relative_tolerance: float,
) -> float:
    normalizer = 1.0 / math.sqrt(2.0 * math.pi)

    def density_weighted(value: float) -> float:
        return function(value) * normalizer * math.exp(-0.5 * value * value)

    points = [-sigma_limit]
    points.extend(point for point in breakpoints if -sigma_limit < point < sigma_limit)
    points.append(sigma_limit)
    points = sorted(set(points))
    return sum(
        _adaptive_simpson(
            density_weighted,
            left,
            right,
            absolute_tolerance=absolute_tolerance / max(1, len(points) - 1),
            relative_tolerance=relative_tolerance,
        )
        for left, right in itertools.pairwise(points)
    )


def _european_one_cash_dividend_reference(
    option_type: str,
    *,
    spot: float,
    strike: float,
    expiry_time: float,
    rate: float,
    continuous_carry: float,
    volatility: float,
    dividend_time: float,
    dividend_amount: float,
    sigma_limit: float = 10.0,
    absolute_tolerance: float = 1.0e-11,
    relative_tolerance: float = 1.0e-11,
) -> float:
    """Independent flat-parameter reference for one European cash dividend."""
    assert 0.0 < dividend_time < expiry_time
    assert dividend_amount > 0.0
    log_mean = math.log(spot) + (
        rate - continuous_carry - 0.5 * volatility * volatility
    ) * dividend_time
    log_width = volatility * math.sqrt(dividend_time)
    continuation_time = expiry_time - dividend_time

    def continuation(standard_normal: float) -> float:
        spot_before_dividend = math.exp(log_mean + log_width * standard_normal)
        spot_after_dividend = max(spot_before_dividend - dividend_amount, 0.0)
        if spot_after_dividend == 0.0:
            if option_type == "call":
                return 0.0
            return strike * math.exp(-rate * continuation_time)
        return float(
            black_scholes(
                option_type,
                spot_after_dividend,
                strike,
                continuation_time,
                rate,
                continuous_carry,
                volatility,
            )["price"]
        )

    dividend_kink = (math.log(dividend_amount) - log_mean) / log_width
    expectation = _normal_expectation(
        continuation,
        breakpoints=[dividend_kink],
        sigma_limit=sigma_limit,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    )
    return math.exp(-rate * dividend_time) * expectation


def _price(
    *,
    option_type: str = "call",
    exercise_style: str = "european",
    spot: float = 100.0,
    strike: float = 100.0,
    valuation_time: float = 0.0,
    expiry_time: float = 1.0,
    volatility: float = 0.2,
    continuous_carry: float | None = 0.0,
    rate: float = 0.05,
    curve_times: list[float] | None = None,
    curve_log_discounts: list[float] | None = None,
    dividends: list[tuple[float, float]] | None = None,
    settlement: str = "cash",
    contract_multiplier: float = 100.0,
    spot_intervals: int = 800,
    time_steps: int = 400,
    spot_maximum: float = 400.0,
    rannacher_steps: int = 2,
    psor_tolerance: float = 1.0e-11,
    psor_relaxation: float = 1.2,
    psor_maximum_iterations: int = 50_000,
) -> dict[str, object]:
    """Price one contract on a flat curve unless explicit knots are supplied."""
    if curve_times is None:
        curve_times = [0.0, expiry_time]
        curve_log_discounts = [0.0, -rate * expiry_time]
    assert curve_log_discounts is not None
    return pde_price(
        option_type=option_type,
        exercise_style=exercise_style,
        spot=spot,
        strike=strike,
        valuation_time=valuation_time,
        expiry_time=expiry_time,
        volatility=volatility,
        continuous_carry=continuous_carry,
        curve_times=curve_times,
        curve_log_discounts=curve_log_discounts,
        dividends=[] if dividends is None else dividends,
        settlement=settlement,
        contract_multiplier=contract_multiplier,
        spot_intervals=spot_intervals,
        time_steps=time_steps,
        spot_maximum=spot_maximum,
        rannacher_steps=rannacher_steps,
        psor_tolerance=psor_tolerance,
        psor_relaxation=psor_relaxation,
        psor_maximum_iterations=psor_maximum_iterations,
    )


# --------------------------------------------------------------------------
# 1. European prices against analytic Black-Scholes
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expiry", "rate", "carry", "volatility"),
    [
        ("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2),
        ("put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2),
        ("call", 110.0, 100.0, 0.5, 0.03, 0.02, 0.3),
        ("put", 90.0, 100.0, 2.0, 0.01, 0.04, 0.25),
        ("call", 100.0, 120.0, 1.5, -0.01, 0.0, 0.15),
    ],
)
def test_european_prices_match_black_scholes(
    option_type: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    carry: float,
    volatility: float,
) -> None:
    result = _price(
        option_type=option_type,
        spot=spot,
        strike=strike,
        expiry_time=expiry,
        rate=rate,
        continuous_carry=carry,
        volatility=volatility,
        spot_intervals=1600,
        time_steps=800,
    )
    reference = black_scholes(option_type, spot, strike, expiry, rate, carry, volatility)
    assert result["price"] == pytest.approx(reference["price"], abs=5.0e-4)
    assert result["solver_status"] == "discrete_system_converged"
    assert result["discretization_accuracy"] == "not_assessed"
    assert result["psor_solves"] == 0
    assert result["dividend_events"] == []


# --------------------------------------------------------------------------
# 2-4. American prices: CRR cross-check, European parity, and bounds
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("option_type", "spot", "strike", "expiry", "rate", "carry", "volatility"),
    [
        ("put", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2),
        ("put", 90.0, 100.0, 0.5, 0.03, 0.0, 0.35),
        ("put", 130.0, 100.0, 2.0, 0.02, 0.0, 0.25),
        ("call", 100.0, 100.0, 1.0, 0.05, 0.06, 0.2),
    ],
)
def test_american_prices_cross_check_the_crr_reference(
    option_type: str,
    spot: float,
    strike: float,
    expiry: float,
    rate: float,
    carry: float,
    volatility: float,
) -> None:
    result = _price(
        option_type=option_type,
        exercise_style="american",
        spot=spot,
        strike=strike,
        expiry_time=expiry,
        rate=rate,
        continuous_carry=carry,
        volatility=volatility,
        spot_intervals=1600,
        time_steps=800,
    )
    reference = crr_price(
        option_type, "american", spot, strike, expiry, rate, carry, volatility, 8192
    )
    # Both engines carry their own discretization error. This is a conservative
    # cross-engine regression band, not a convergence claim about either.
    assert result["price"] == pytest.approx(reference["price"], abs=2.0e-3)
    assert result["maximum_relative_lcp_residual"] <= result["psor_tolerance"]
    assert result["psor_solves"] == (
        result["crank_nicolson_steps"] + result["damped_half_steps"]
    )


@pytest.mark.parametrize("rate", [0.0, 0.05])
def test_american_call_without_carry_or_dividends_matches_european(rate: float) -> None:
    european = _price(exercise_style="european", rate=rate, continuous_carry=0.0)
    american = _price(exercise_style="american", rate=rate, continuous_carry=0.0)
    assert american["price"] == pytest.approx(european["price"], abs=SOLVER_BAND)


@pytest.mark.parametrize("option_type", ["call", "put"])
@pytest.mark.parametrize("spot", [70.0, 100.0, 140.0])
def test_american_value_is_never_below_intrinsic_or_european(
    option_type: str, spot: float
) -> None:
    dividends = [(0.5, 4.0)]
    european = _price(
        option_type=option_type,
        exercise_style="european",
        spot=spot,
        continuous_carry=0.01,
        dividends=dividends,
    )
    american = _price(
        option_type=option_type,
        exercise_style="american",
        spot=spot,
        continuous_carry=0.01,
        dividends=dividends,
    )
    intrinsic = max(spot - 100.0, 0.0) if option_type == "call" else max(100.0 - spot, 0.0)
    assert american["price"] >= intrinsic - SOLVER_BAND
    assert american["price"] >= european["price"] - SOLVER_BAND


# --------------------------------------------------------------------------
# 5-8. Cash dividends
# --------------------------------------------------------------------------


@pytest.mark.parametrize("amount", [0.0, -1.0])
def test_nonpositive_cash_dividends_are_rejected(amount: float) -> None:
    # A zero dividend is an event that does nothing, which is a different
    # statement from declaring no event at all.
    with pytest.raises(ValueError, match="amount must be finite and positive"):
        _price(dividends=[(0.5, amount)])


@pytest.mark.parametrize("ex_time", [0.0, 1.0, 1.5, -0.25])
def test_dividends_outside_the_open_interval_are_rejected(ex_time: float) -> None:
    with pytest.raises(ValueError, match="strictly inside"):
        _price(dividends=[(ex_time, 1.0)])


def test_unsorted_and_duplicated_dividend_schedules_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        _price(dividends=[(0.6, 1.0), (0.3, 1.0)])
    with pytest.raises(ValueError, match="repeat"):
        _price(dividends=[(0.5, 1.0), (0.5, 1.0)])


def test_an_empty_dividend_list_is_an_explicit_declaration() -> None:
    # `dividends` has no default at the binding boundary, so reaching the empty
    # case requires the caller to write it.
    with pytest.raises(TypeError):
        pde_price(
            option_type="call",
            exercise_style="european",
            spot=100.0,
            strike=100.0,
            valuation_time=0.0,
            expiry_time=1.0,
            volatility=0.2,
            continuous_carry=0.0,
            curve_times=[0.0, 1.0],
            curve_log_discounts=[0.0, -0.05],
            settlement="cash",
            contract_multiplier=100.0,
            spot_intervals=200,
            time_steps=100,
            spot_maximum=400.0,
            rannacher_steps=2,
            psor_tolerance=1.0e-11,
            psor_relaxation=1.2,
            psor_maximum_iterations=1000,
        )
    assert _price(dividends=[])["dividend_events"] == []


def test_larger_dividends_lower_calls_and_raise_puts() -> None:
    amounts = (0.5, 1.0, 2.0, 4.0)
    calls = [_price(option_type="call", dividends=[(0.5, d)])["price"] for d in amounts]
    puts = [_price(option_type="put", dividends=[(0.5, d)])["price"] for d in amounts]
    assert all(later < earlier - 1.0e-6 for earlier, later in itertools.pairwise(calls))
    assert all(later > earlier + 1.0e-6 for earlier, later in itertools.pairwise(puts))


def test_one_cash_dividend_european_quadrature_reference_is_stable() -> None:
    kwargs = {
        "spot": 100.0,
        "strike": 95.0,
        "expiry_time": 1.25,
        "rate": 0.04,
        "continuous_carry": 0.01,
        "volatility": 0.25,
        "dividend_time": 0.4,
        "dividend_amount": 6.0,
    }
    baseline = _european_one_cash_dividend_reference("call", **kwargs)
    tighter = _european_one_cash_dividend_reference(
        "call",
        **kwargs,
        sigma_limit=11.0,
        absolute_tolerance=1.0e-12,
        relative_tolerance=1.0e-12,
    )
    assert baseline == pytest.approx(tighter, abs=5.0e-10)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_fine_pde_grid_matches_one_cash_dividend_european_quadrature(
    option_type: str,
) -> None:
    kwargs = {
        "spot": 100.0,
        "strike": 95.0,
        "expiry_time": 1.25,
        "rate": 0.04,
        "continuous_carry": 0.01,
        "volatility": 0.25,
    }
    dividend_time = 0.4
    dividend_amount = 6.0
    reference = _european_one_cash_dividend_reference(
        option_type,
        **kwargs,
        dividend_time=dividend_time,
        dividend_amount=dividend_amount,
    )
    result = _price(
        option_type=option_type,
        exercise_style="european",
        **kwargs,
        dividends=[(dividend_time, dividend_amount)],
        spot_intervals=4_000,
        time_steps=2_000,
        spot_maximum=500.0,
    )
    error = abs(float(result["price"]) - reference)
    assert result["solver_status"] == "discrete_system_converged"
    assert result["discretization_accuracy"] == "not_assessed"
    assert error < 1.5e-3


@pytest.mark.parametrize(
    ("spot", "amount", "expected"),
    [(100.0, 2.5, 97.5), (2.0, 2.0, 0.0), (0.5, 2.0, 0.0)],
)
def test_dividend_jump_mapping(spot: float, amount: float, expected: float) -> None:
    assert pde_post_dividend_spot(spot, amount) == expected


def test_a_put_below_its_dividend_lands_on_the_absorbing_boundary() -> None:
    # max(S - d, 0) sends S = 5 to the S = 0 node, where the put is certain to
    # pay the strike at expiry.
    result = _price(
        option_type="put", spot=5.0, strike=100.0, dividends=[(0.001, 20.0)]
    )
    assert result["price"] == pytest.approx(100.0 * math.exp(-0.05), abs=1.0e-5)
    assert len(result["dividend_events"]) == 1
    assert result["dividend_events"][0]["amount"] == 20.0


def test_exercise_immediately_before_a_dividend_is_available() -> None:
    # A dividend large enough to leave the stock deep out of the money makes
    # holding past the ex-date worthless, so the American call collapses onto a
    # European call expiring at the ex-time.
    dividends = [(0.5, 40.0)]
    american = _price(
        exercise_style="american",
        strike=90.0,
        rate=0.03,
        dividends=dividends,
        spot_intervals=1600,
        time_steps=800,
    )
    european = _price(
        exercise_style="european",
        strike=90.0,
        rate=0.03,
        dividends=dividends,
        spot_intervals=1600,
        time_steps=800,
    )
    stub = _price(
        exercise_style="european",
        strike=90.0,
        expiry_time=0.5,
        rate=0.03,
        spot_intervals=1600,
        time_steps=400,
    )
    assert american["price"] >= stub["price"] - 1.0e-4
    assert american["price"] == pytest.approx(stub["price"], abs=5.0e-3)
    assert american["price"] > european["price"] + 1.0


# --------------------------------------------------------------------------
# 9. Discount-curve knot alignment and refusal to extrapolate
# --------------------------------------------------------------------------


def test_curve_knots_and_dividends_are_aligned_to_the_time_grid() -> None:
    result = _price(
        exercise_style="american",
        option_type="put",
        curve_times=[0.0, 0.25, 0.75, 1.0],
        curve_log_discounts=[0.0, -0.005, -0.02, -0.03],
        dividends=[(0.4, 1.5), (0.9, 1.5)],
        spot_intervals=400,
        time_steps=200,
    )
    assert result["aligned_times"] == [0.0, 0.25, 0.4, 0.75, 0.9, 1.0]
    events = result["dividend_events"]
    assert [event["ex_time"] for event in events] == [0.4, 0.9]
    assert [result["aligned_times"][event["aligned_time_index"]] for event in events] == [
        0.4,
        0.9,
    ]


def test_log_discount_interpolation_and_segment_rates() -> None:
    times = [0.0, 0.25, 0.75, 1.0]
    log_discounts = [0.0, -0.005, -0.02, -0.03]
    assert pde_log_discount(times, log_discounts, 0.5) == pytest.approx(-0.0125, abs=1.0e-15)
    assert pde_discount_factor(times, log_discounts, 0.0, 1.0) == pytest.approx(
        math.exp(-0.03), abs=1.0e-15
    )
    assert pde_segment_rate(times, log_discounts, 0) == pytest.approx(0.02, abs=1.0e-14)
    assert pde_segment_rate(times, log_discounts, 1) == pytest.approx(0.03, abs=1.0e-14)


@pytest.mark.parametrize("time", [1.5, -0.5])
def test_curve_evaluation_refuses_to_extrapolate(time: float) -> None:
    with pytest.raises(ValueError, match="extrapolation is refused"):
        pde_log_discount([0.0, 0.25, 1.0], [0.0, -0.005, -0.03], time)


def test_a_curve_that_does_not_reach_expiry_is_rejected() -> None:
    with pytest.raises(ValueError, match="bracket"):
        _price(curve_times=[0.0, 0.5], curve_log_discounts=[0.0, -0.025])


@pytest.mark.parametrize(
    ("times", "log_discounts", "message"),
    [
        ([0.5, 1.0], [-0.025, -0.05], r"begin at \(0, 0\)"),
        ([0.0, 1.0], [0.01, -0.05], r"begin at \(0, 0\)"),
        ([0.0, 1.0, 0.5], [0.0, -0.05, -0.025], "strictly increasing"),
        ([0.0, 1.0], [0.0], "equal length"),
        ([0.0], [0.0], "at least two knots"),
    ],
)
def test_malformed_discount_curves_are_rejected(
    times: list[float], log_discounts: list[float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _price(curve_times=times, curve_log_discounts=log_discounts)


# --------------------------------------------------------------------------
# 10. Invalid grids, rates, volatility, carry, conventions and PSOR failure
# --------------------------------------------------------------------------


def test_continuous_carry_must_be_stated() -> None:
    with pytest.raises(ValueError, match="stated explicitly"):
        _price(continuous_carry=None)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"spot": 0.0}, "spot must be finite and positive"),
        ({"strike": -1.0}, "strike must be finite and positive"),
        ({"expiry_time": 0.0}, "expiry time must be finite and after"),
        ({"valuation_time": -1.0}, "valuation time must be finite and non-negative"),
        ({"volatility": 0.0}, "volatility must be finite and positive"),
        ({"continuous_carry": math.inf}, "continuous carry must be finite"),
        ({"contract_multiplier": 0.0}, "contract multiplier must be finite and positive"),
        ({"spot_intervals": 3}, "at least four intervals"),
        ({"time_steps": 0}, "time steps must be positive"),
        ({"spot_maximum": 100.0}, "must exceed both the spot and the strike"),
        ({"psor_tolerance": 0.0}, "PSOR tolerance must be finite and positive"),
        ({"psor_relaxation": 2.0}, "PSOR relaxation must lie strictly inside"),
        ({"psor_maximum_iterations": 0}, "PSOR iteration limit must be positive"),
        ({"rannacher_steps": 10_000}, "Rannacher damping exceeds"),
    ],
)
def test_invalid_requests_are_rejected(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _price(**overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "value"),
    [("option_type", "straddle"), ("exercise_style", "bermudan"), ("settlement", "chained")],
)
def test_unknown_conventions_are_rejected(field: str, value: str) -> None:
    with pytest.raises(ValueError, match=value):
        _price(**{field: value})  # type: ignore[arg-type]


def test_a_time_grid_too_coarse_to_align_every_event_is_rejected() -> None:
    with pytest.raises(ValueError, match="cannot align"):
        _price(dividends=[(0.2, 1.0), (0.4, 1.0), (0.6, 1.0)], time_steps=2)


def test_psor_non_convergence_is_an_explicit_failure() -> None:
    with pytest.raises(PsorConvergenceError) as caught:
        _price(
            exercise_style="american",
            option_type="put",
            spot_intervals=400,
            time_steps=200,
            psor_tolerance=1.0e-16,
            psor_maximum_iterations=2,
        )
    assert issubclass(PsorConvergenceError, RuntimeError)
    # pybind11 forwards what() and nothing else, so the diagnostics have to
    # survive in the message for a Python caller to see them at all.
    message = str(caught.value)
    assert "aligned segment" in message
    assert "iterations" in message
    assert "relative residual" in message
    assert "tolerance" in message


def test_an_unresolvable_expiry_span_is_rejected() -> None:
    # Below the time-alignment tolerance the valuation and expiry instants merge,
    # leaving no segment to step over. Unchecked, that returns the undiscounted
    # terminal payoff while reporting a solved discrete system.
    with pytest.raises(ValueError, match="time-alignment tolerance"):
        _price(
            spot=110.0,
            valuation_time=0.05,
            expiry_time=0.05 + 1.0e-13,
            curve_times=[0.0, 0.1, 1.0],
            curve_log_discounts=[0.0, -0.005, -0.5],
        )


def test_a_short_but_resolvable_expiry_still_prices() -> None:
    # The spot step must resolve the diffusion width sigma * S * sqrt(T), here
    # 0.35. The solver does not scale the requested grid to the expiry, so that
    # choice belongs to the caller; at h = 0.02 the step is 6% of the width.
    result = _price(expiry_time=3.0e-4, spot_intervals=10_000, time_steps=50, spot_maximum=200.0)
    reference = black_scholes("call", 100.0, 100.0, 3.0e-4, 0.05, 0.0, 0.2)
    assert result["price"] == pytest.approx(reference["price"], abs=1.0e-4)
    assert result["solver_status"] == "discrete_system_converged"
    assert result["discretization_accuracy"] == "not_assessed"


def test_a_grid_that_cannot_resolve_a_short_expiry_is_visibly_wrong() -> None:
    # Documented limitation, asserted so it cannot regress into a silent claim:
    # at spot_maximum=400 with 800 intervals the step is 1.4x the diffusion
    # width of a 2.6-hour option and the price is badly biased. Nothing in the
    # solver detects this; the refinement ladder is what exposes it.
    coarse = _price(expiry_time=3.0e-4, spot_intervals=800, time_steps=50)
    reference = black_scholes("call", 100.0, 100.0, 3.0e-4, 0.05, 0.0, 0.2)
    assert coarse["solver_status"] == "discrete_system_converged"
    assert coarse["discretization_accuracy"] == "not_assessed"
    assert abs(coarse["price"] - reference["price"]) > 1.0e-2


def test_a_deep_in_the_money_multi_dividend_call_matches_the_forward_closed_form() -> None:
    # Deep in the money a call is worth its discounted forward intrinsic. This is
    # the independent check on the upper boundary formula and on which dividends
    # the solver counts as still ahead: misattributing one across a segment
    # boundary moves the price by O(d), far above the discretization error.
    rate, carry, expiry, strike, spot = 0.05, 0.02, 1.0, 100.0, 3900.0
    dividends = [(0.35, 1.5), (0.75, 2.5)]
    result = _price(
        spot=spot,
        strike=strike,
        expiry_time=expiry,
        rate=rate,
        continuous_carry=carry,
        dividends=dividends,
        spot_intervals=4000,
        time_steps=800,
        spot_maximum=4000.0,
    )
    expected = spot * math.exp(-carry * expiry)
    for ex_time, amount in dividends:
        expected -= amount * math.exp(-rate * ex_time) * math.exp(-carry * (expiry - ex_time))
    expected -= strike * math.exp(-rate * expiry)
    assert result["price"] == pytest.approx(expected, abs=1.0e-5)
    assert len(result["dividend_events"]) == 2


def test_a_dividend_on_a_curve_knot_merges_into_one_aligned_instant() -> None:
    result = _price(
        option_type="put",
        exercise_style="american",
        volatility=0.25,
        rate=0.03,
        curve_times=[0.0, 0.25, 0.6, 1.0],
        curve_log_discounts=[0.0, -0.01, -0.024, -0.04],
        dividends=[(0.6, 2.0)],
    )
    assert result["aligned_times"] == [0.0, 0.25, 0.6, 1.0]
    assert len(result["dividend_events"]) == 1
    assert result["dividend_events"][0]["aligned_time_index"] == 2


# --------------------------------------------------------------------------
# 11-12. Determinism, reporting metadata, and the refinement ladder
# --------------------------------------------------------------------------


def test_prices_are_bitwise_reproducible() -> None:
    kwargs = {
        "option_type": "put",
        "exercise_style": "american",
        "spot": 103.0,
        "continuous_carry": 0.01,
        "volatility": 0.28,
        "rate": 0.04,
        "dividends": [(0.35, 1.25), (0.8, 2.0)],
    }
    first = _price(**kwargs)  # type: ignore[arg-type]
    for _ in range(3):
        again = _price(**kwargs)  # type: ignore[arg-type]
        assert again["price"] == first["price"]
        assert again["psor_total_iterations"] == first["psor_total_iterations"]
        assert again["maximum_lcp_residual"] == first["maximum_lcp_residual"]


def test_the_contract_multiplier_never_enters_the_pricing_arithmetic() -> None:
    per_share = _price(contract_multiplier=1.0)["price"]
    per_contract = _price(contract_multiplier=100.0)["price"]
    assert per_share == per_contract


def test_the_scalar_pde_price_releases_the_python_gil() -> None:
    started = threading.Event()
    stop = threading.Event()
    counter = [0]

    def increment() -> None:
        started.set()
        while not stop.is_set():
            counter[0] += 1

    worker = threading.Thread(target=increment)
    worker.start()
    try:
        started.wait(timeout=5.0)
        before = counter[0]
        _price(exercise_style="american", option_type="put", spot_intervals=800, time_steps=400)
        assert counter[0] > before
    finally:
        stop.set()
        worker.join(timeout=5.0)


def test_european_refinement_ladder_converges() -> None:
    reference = black_scholes("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)["price"]
    errors = []
    for factor in (1, 2, 4, 8):
        result = _price(spot_intervals=200 * factor, time_steps=100 * factor)
        errors.append(abs(result["price"] - reference))
    # Second-order refinement quarters the error. The gate is a conservative
    # factor of three, so it measures convergence rather than a tuned order.
    assert all(
        later < earlier / 3.0 for earlier, later in itertools.pairwise(errors)
    ), errors


def test_american_refinement_ladder_converges() -> None:
    reference = crr_price("put", "american", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2, 8192)["price"]
    errors = []
    for factor in (1, 2, 4):
        result = _price(
            option_type="put",
            exercise_style="american",
            spot_intervals=400 * factor,
            time_steps=200 * factor,
        )
        errors.append(abs(result["price"] - reference))
    assert all(later < earlier for earlier, later in itertools.pairwise(errors)), errors
    assert errors[-1] < 5.0e-4


def test_spot_domain_truncation_is_reported_and_bounded() -> None:
    wide = _price(spot_intervals=1600, spot_maximum=800.0)["price"]
    medium = _price(spot_intervals=800, spot_maximum=400.0)["price"]
    tight = _price(spot_intervals=300, spot_maximum=150.0)["price"]
    assert medium == pytest.approx(wide, abs=1.0e-6)
    assert abs(tight - wide) > abs(medium - wide)


def test_the_used_grid_is_reported_with_the_strike_on_a_node() -> None:
    result = _price(strike=97.0, spot_intervals=800, spot_maximum=400.0)
    assert result["spot_step"] * result["strike_node_index"] == pytest.approx(97.0, abs=1.0e-12)
    assert result["spot_maximum"] == pytest.approx(
        result["spot_step"] * result["spot_intervals"], abs=1.0e-12
    )
    assert result["spot_maximum"] >= 400.0
