"""Cross-language tests for deterministic LSM pricing and diagnostics."""

from __future__ import annotations

import hashlib
import math
import threading
from pathlib import Path

import pytest
from differentiable_pricing import (
    _core,
    black_scholes,
    lsm_price,
    lsm_training_memory_bytes,
)

ARGS = {
    "exercise_steps": 16,
    "training_paths": 2048,
    "valuation_paths": 4096,
    "polynomial_degree": 2,
    "training_seed": 111,
    "valuation_seed": 222,
    "maximum_training_memory_bytes": 8 * 1024 * 1024,
}
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _price(option_type: str = "put", **overrides):
    arguments = ARGS | overrides
    return lsm_price(
        option_type,
        100.0,
        100.0,
        1.0,
        0.05,
        0.0,
        0.2,
        arguments["exercise_steps"],
        arguments["training_paths"],
        arguments["valuation_paths"],
        arguments["polynomial_degree"],
        arguments["training_seed"],
        arguments["valuation_seed"],
        arguments["maximum_training_memory_bytes"],
    )


def test_lsm_binding_reports_provenance_and_operational_limits() -> None:
    assert len(_core.__lsm_header_sha256__) == 64
    assert len(_core.__lsm_source_sha256__) == 64
    assert len(_core.__lsm_implementation_sha256__) == 64
    assert _core.maximum_lsm_exercise_steps == 1024
    assert _core.maximum_lsm_polynomial_degree == 3
    assert _core.maximum_lsm_paths >= 4096
    assert _core.maximum_lsm_training_memory_bytes >= 8 * 1024 * 1024
    assert _core.__lsm_header_sha256__ == hashlib.sha256(
        (
            PROJECT_ROOT / "cpp" / "include" / "dp" / "least_squares_monte_carlo.hpp"
        ).read_bytes()
    ).hexdigest()
    assert _core.__lsm_source_sha256__ == hashlib.sha256(
        (
            PROJECT_ROOT / "cpp" / "src" / "least_squares_monte_carlo.cpp"
        ).read_bytes()
    ).hexdigest()


def test_training_memory_estimator_matches_result() -> None:
    estimated = lsm_training_memory_bytes(*ARGS.values())
    result = _price()
    matrix_bytes = ARGS["training_paths"] * (ARGS["exercise_steps"] + 1) * 8
    assert estimated > matrix_bytes
    assert result["estimated_training_working_set_bytes"] == estimated


def test_lsm_replay_is_bit_deterministic_and_pair_aware() -> None:
    first = _price()
    second = _price()
    assert first == second
    assert first["independent_valuation_pairs"] == ARGS["valuation_paths"] // 2
    assert first["confidence_level"] == 0.95
    assert (
        first["confidence_interval_lower"]
        <= first["price"]
        <= first["confidence_interval_upper"]
    )
    assert len(first["regressions"]) == ARGS["exercise_steps"] - 1
    assert all(
        row["exercise_step"] == index
        for index, row in enumerate(first["regressions"], start=1)
    )


def test_no_dividend_call_control_recovers_black_scholes() -> None:
    result = _price("call")
    analytic = black_scholes("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)
    assert result["price"] == pytest.approx(analytic["price"], abs=1.0e-12)
    assert result["standard_error"] == pytest.approx(0.0, abs=1.0e-14)
    assert result["valuation_early_exercise_paths"] == 0
    # A perfect control variate is a measurement on a sampled valuation stream,
    # so the ratio stays applicable and infinite.
    assert result["variance_reduction_applicable"] is True
    assert result["european_monte_carlo_sampled"] is True
    assert math.isinf(result["variance_reduction_ratio"])


def test_deterministic_raw_estimator_has_no_variance_reduction() -> None:
    # Every valuation pair of a far out-of-the-money put pays exactly zero, so
    # raw and adjusted variances are both zero: the ratio is an undefined 0/0,
    # not complete variance removal.
    result = lsm_price(
        "put",
        100.0,
        1.0e-6,
        1.0,
        0.05,
        0.0,
        0.2,
        *ARGS.values(),
    )
    assert result["exercise_at_zero"] is False
    assert result["raw_standard_error"] == 0.0
    assert result["standard_error"] == 0.0
    assert result["variance_reduction_applicable"] is False
    assert not math.isinf(result["variance_reduction_ratio"])


def test_immediate_exercise_marks_unavailable_fields() -> None:
    result = lsm_price(
        "put",
        50.0,
        100.0,
        1.0,
        0.20,
        0.0,
        0.05,
        *ARGS.values(),
    )
    assert result["exercise_at_zero"] is True
    assert result["price"] == 50.0
    assert result["standard_error"] == 0.0
    # Both raw and adjusted variances are zero, so the ratio is 0/0.
    assert result["variance_reduction_applicable"] is False
    assert result["variance_reduction_ratio"] == 0.0
    # No valuation simulation runs, so no European Monte Carlo value exists.
    assert result["european_monte_carlo_sampled"] is False
    assert result["european_monte_carlo_price"] == 0.0
    assert result["european_standard_error"] == 0.0
    analytic = black_scholes("put", 50.0, 100.0, 1.0, 0.20, 0.0, 0.05)
    assert result["european_analytic_price"] == pytest.approx(
        analytic["price"], abs=1.0e-12
    )


@pytest.mark.parametrize(
    ("option_type", "rate", "dividend_yield", "volatility", "reason"),
    [
        # exp(log_spot) overflows while the log spot stays finite; the matching
        # discount factor underflows to zero, so the product was inf * 0.
        ("call", 750.0, 0.0, 0.2, "spot"),
        # The same linear-spot overflow driven through the dividend term.
        ("call", 0.05, -1000.0, 0.2, "spot"),
        # A finite but extreme volatility makes the drift itself non-finite.
        ("put", 0.05, 0.0, 1.0e160, "drift"),
        # A large negative rate overflows the discount factor instead.
        ("put", -750.0, 0.0, 0.2, "discount"),
    ],
)
def test_non_finite_price_domain_quantities_raise(
    option_type: str,
    rate: float,
    dividend_yield: float,
    volatility: float,
    reason: str,
) -> None:
    with pytest.raises(OverflowError, match="left double precision"):
        lsm_price(
            option_type,
            100.0,
            100.0,
            1.0,
            rate,
            dividend_yield,
            volatility,
            *ARGS.values(),
        )


def test_extreme_but_representable_inputs_stay_finite() -> None:
    result = lsm_price(
        "call",
        100.0,
        100.0,
        1.0,
        0.05,
        -50.0,
        0.2,
        *ARGS.values(),
    )
    for key in (
        "price",
        "standard_error",
        "confidence_interval_lower",
        "confidence_interval_upper",
        "raw_price",
        "raw_standard_error",
        "european_analytic_price",
    ):
        assert math.isfinite(result[key]), key
    assert result["price"] > 0.0
    assert result["standard_error"] >= 0.0


def test_constant_regression_fallback_is_reported() -> None:
    # A deep out-of-the-money put has no in-the-money training paths at any
    # exercise date, so every regression takes the zero-observation fallback.
    result = lsm_price(
        "put",
        100.0,
        40.0,
        1.0,
        0.05,
        0.0,
        0.2,
        *ARGS.values(),
    )
    rows = result["regressions"]
    assert rows
    assert all(row["used_constant_fallback"] for row in rows)
    assert all(row["in_the_money_paths"] == 0 for row in rows)
    assert all(row["regression_rank"] == 0 for row in rows)
    assert all(row["coefficients"] == [0.0] for row in rows)
    assert all(row["minimum_relative_r_diagonal"] == 0.0 for row in rows)

    # A strike just inside the simulated range leaves fewer in-the-money paths
    # than basis columns at late dates: the rank-deficient fallback branch.
    partial = lsm_price(
        "put",
        100.0,
        60.0,
        1.0,
        0.05,
        0.0,
        0.2,
        *ARGS.values(),
    )
    degree = ARGS["polynomial_degree"]
    observed = [
        row
        for row in partial["regressions"]
        if row["used_constant_fallback"] and row["in_the_money_paths"] > 0
    ]
    assert observed, "no rank-deficient fallback was exercised"
    time_step = 1.0 / ARGS["exercise_steps"]
    for row in observed:
        assert row["in_the_money_paths"] < degree + 1
        assert row["regression_rank"] == 1
        # The coefficient is a mean discounted put response, bounded by the
        # strike discounted over at least one exercise interval.
        assert 0.0 <= row["coefficients"][0] <= 60.0 * math.exp(-0.05 * time_step)
    # A well-conditioned at-the-money case fits every date at full rank, so only
    # fallbacks ever report a zero relative diagonal.
    fitted = [row for row in _price()["regressions"] if not row["used_constant_fallback"]]
    assert fitted
    assert all(0.0 < row["minimum_relative_r_diagonal"] <= 1.0 for row in fitted)
    assert all(row["regression_rank"] == degree + 1 for row in fitted)


def test_lsm_releases_the_python_gil() -> None:
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
        assert started.wait(timeout=1.0)
        before = counter[0]
        _price(
            exercise_steps=32,
            training_paths=8192,
            valuation_paths=16384,
            maximum_training_memory_bytes=16 * 1024 * 1024,
        )
        after = counter[0]
    finally:
        stop.set()
        worker.join(timeout=1.0)
    assert not worker.is_alive()
    assert after > before


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"exercise_steps": 1}, "exercise steps"),
        ({"training_paths": 2049}, "training paths"),
        ({"valuation_paths": 3}, "valuation paths"),
        ({"polynomial_degree": 4}, "degree"),
        ({"training_seed": 222}, "seeds"),
        ({"maximum_training_memory_bytes": 1}, "exceeding"),
    ],
)
def test_invalid_lsm_requests_are_rejected(overrides, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _price(**overrides)


def test_invalid_lsm_option_and_physical_inputs_are_rejected() -> None:
    with pytest.raises(ValueError, match="option type"):
        _price("straddle")
    with pytest.raises(ValueError, match="spot"):
        lsm_price(
            "put",
            -1.0,
            100.0,
            1.0,
            0.05,
            0.0,
            0.2,
            *ARGS.values(),
        )
