"""Which representation carries enough state to learn an American CRR price.

Task 9E's feature-sufficiency evidence. The claim under test is not a matter of
taste: the European ``forward_normalized_v1`` representation
``(option_type, log(F/K), sigma*sqrt(T))`` is exactly sufficient for a European
price scaled by ``spot * exp(-q * T)``, and provably **insufficient** for an
American one, because the early-exercise boundary depends on the rate and the
dividend yield separately rather than only through the forward.

These tests measure that with the compiled CRR engine on a handful of scalar
prices. They price nothing else, train nothing, and benchmark nothing.
"""

from __future__ import annotations

import math

import pytest
from differentiable_pricing import _core
from differentiable_pricing.data.american_schema import (
    FEATURE_ORDER as AMERICAN_FEATURE_ORDER,
)
from differentiable_pricing.data.american_schema import (
    REPRESENTATION,
    REPRESENTATIONS_REJECTED_FOR_AMERICAN,
)
from differentiable_pricing.ml.config import (
    FORWARD_NORMALIZED_REPRESENTATION,
    RAW_PHYSICAL_REPRESENTATION,
    REPRESENTATION_FEATURE_ORDER,
)

STEPS = 512
SPOT = 100.0
MATURITY = 1.0
VOLATILITY = 0.30


def _normalized(
    option_type: str, exercise_style: str, rate: float, dividend_yield: float
) -> float:
    """CRR price scaled the way ``forward_normalized_v1`` scales it.

    Strike is chosen so ``log(F / K) == 0``, so every contract compared below
    shares an option type, a forward moneyness and a total volatility.
    """
    strike = SPOT * math.exp((rate - dividend_yield) * MATURITY)
    result = _core.crr_price(
        option_type,
        exercise_style,
        SPOT,
        strike,
        MATURITY,
        rate,
        dividend_yield,
        VOLATILITY,
        STEPS,
    )
    price = float(result["price"] if isinstance(result, dict) else result)
    return price / (SPOT * math.exp(-dividend_yield * MATURITY))


# Carry pairs sharing one forward: rate - dividend_yield == 0.10 throughout.
_MATCHED_CARRY = ((0.10, 0.00), (0.11, 0.01), (0.12, 0.02))


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_forward_normalization_is_exact_for_european_prices(option_type: str) -> None:
    """Same forward, same total volatility, different (r, q): same European price."""
    reference = _normalized(option_type, "european", *_MATCHED_CARRY[0])
    for rate, dividend_yield in _MATCHED_CARRY[1:]:
        other = _normalized(option_type, "european", rate, dividend_yield)
        assert other == pytest.approx(reference, abs=1.0e-12), (rate, dividend_yield)


def test_forward_normalization_loses_state_for_american_puts() -> None:
    """Same forward and total volatility, different (r, q): different American price.

    Both contracts share ``option_type``, ``log(F / K) == 0`` and
    ``sigma * sqrt(T)``, so ``forward_normalized_v1`` cannot tell them apart.
    Their normalized American prices differ by far more than any plausible
    surrogate tolerance, which is what makes that representation inadmissible
    here.
    """
    low_rate = _normalized("put", "american", 0.10, 0.00)
    high_rate = _normalized("put", "american", 0.12, 0.02)

    european_low = _normalized("put", "european", 0.10, 0.00)
    european_high = _normalized("put", "european", 0.12, 0.02)
    assert european_high == pytest.approx(european_low, abs=1.0e-12)

    difference = abs(high_rate - low_rate)
    assert difference > 1.0e-3, difference
    # The gap is early-exercise value, so both American prices must still
    # dominate the shared European one.
    assert low_rate > european_low
    assert high_rate > european_high


def test_forward_normalization_loses_state_for_american_calls() -> None:
    """The same failure on the dividend-driven side, where calls exercise early."""
    low = _normalized("call", "american", 0.02, 0.10)
    high = _normalized("call", "american", 0.04, 0.12)
    european = _normalized("call", "european", 0.02, 0.10)
    assert _normalized("call", "european", 0.04, 0.12) == pytest.approx(
        european, abs=1.0e-12
    )
    assert low > european
    assert abs(high - low) > 1.0e-4, abs(high - low)


def test_raw_physical_state_separates_contracts_the_normalization_merges() -> None:
    """The seven raw features distinguish exactly the pair the normalization merges."""

    def raw(rate: float, dividend_yield: float) -> tuple[float, ...]:
        strike = SPOT * math.exp((rate - dividend_yield) * MATURITY)
        row = {
            "option_type": -1.0,
            "spot": SPOT,
            "strike": strike,
            "maturity": MATURITY,
            "rate": rate,
            "dividend_yield": dividend_yield,
            "volatility": VOLATILITY,
        }
        return tuple(row[name] for name in AMERICAN_FEATURE_ORDER)

    def normalized(rate: float, dividend_yield: float) -> tuple[float, ...]:
        strike = SPOT * math.exp((rate - dividend_yield) * MATURITY)
        forward = SPOT * math.exp((rate - dividend_yield) * MATURITY)
        return (-1.0, math.log(forward / strike), VOLATILITY * math.sqrt(MATURITY))

    first, second = (0.10, 0.00), (0.12, 0.02)
    assert normalized(*first) == normalized(*second)
    assert raw(*first) != raw(*second)
    assert "rate" in AMERICAN_FEATURE_ORDER
    assert "dividend_yield" in AMERICAN_FEATURE_ORDER


def test_american_representation_is_named_and_matches_raw_state() -> None:
    assert REPRESENTATION == "american_raw_physical_v1"
    assert REPRESENTATION_FEATURE_ORDER[
        RAW_PHYSICAL_REPRESENTATION
    ] == AMERICAN_FEATURE_ORDER


def test_forward_normalized_representation_is_recorded_as_rejected() -> None:
    assert FORWARD_NORMALIZED_REPRESENTATION in REPRESENTATIONS_REJECTED_FOR_AMERICAN
    reason = REPRESENTATIONS_REJECTED_FOR_AMERICAN[FORWARD_NORMALIZED_REPRESENTATION]
    assert "rate and dividend yield" in reason
