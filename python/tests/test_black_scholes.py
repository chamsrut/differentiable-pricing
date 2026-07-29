import math

import pytest

from differentiable_pricing import black_scholes


def test_known_call_price_and_greeks() -> None:
    result = black_scholes("call", 100.0, 100.0, 1.0, 0.05, 0.0, 0.2)
    assert result["price"] == pytest.approx(10.450583572185565, abs=1e-12)
    assert result["delta"] == pytest.approx(0.6368306511756191, abs=1e-12)
    assert all(math.isfinite(value) for value in result.values())


def test_invalid_spot_is_rejected() -> None:
    with pytest.raises(ValueError, match="spot"):
        black_scholes("call", -1.0, 100.0, 1.0, 0.05, 0.0, 0.2)

