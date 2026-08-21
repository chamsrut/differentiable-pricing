"""Task 9H American price representation, heads and physical reconstruction.

The representation is Task 9G's ``american_forward_carry_v1`` and the target is
its ``u = V / (S*exp(-q*T))``, unchanged, so a Task 9H candidate stays directly
comparable with the Task 9G control. What varies across attempts is the network,
the head, and whether deterministic conditioning features are appended.

Conditioning features are deterministic functions of inputs the network already
receives: they change conditioning, not information. They are computed inside
the torch graph rather than precomputed, which costs nothing here and keeps the
model a single function of its physical inputs.

This module does **not** modify the frozen Task 9G modules.
``differentiable_pricing.ml.model`` is digest-pinned by
``configs/american_neural_pilot_protocol_v1.toml``; it is imported read-only for
the physical feature order and the shared input-validation contract.
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np
import torch
from torch import nn

from ..config import FEATURE_ORDER
from ..model import PhysicalInputError, Scaling, validate_physical_features
from .attempts import (
    CONDITIONING_FEATURES,
    HEADS,
    NORMALIZED_TARGET,
    PHYSICAL_RECONSTRUCTION,
    REPRESENTATION,
)

#: Re-exported from :mod:`attempts`, which owns the single definition so the
#: PyTorch-free configuration validator and this module cannot disagree about
#: what an attempt is allowed to declare.
__all__ = [
    "NORMALIZED_TARGET",
    "PHYSICAL_RECONSTRUCTION",
    "REPRESENTATION",
    "AmericanDevPriceModel",
    "base_features",
    "conditioning_feature",
    "discounted_spot",
    "european_price",
    "feature_order",
    "intrinsic_value",
    "network_features",
    "representation_arrays",
]

BASE_FEATURE_ORDER: Final = (
    "option_type",
    "log_forward_moneyness",
    "total_volatility",
    "rate_time",
    "yield_time",
)

_INDEX: Final = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}


def _standard_normal_cdf(value: torch.Tensor) -> torch.Tensor:
    return 0.5 * (1.0 + torch.erf(value * (1.0 / math.sqrt(2.0))))


def european_price(physical_features: torch.Tensor) -> torch.Tensor:
    """Analytic continuous-yield Black--Scholes price, inside the graph.

    Used as a conditioning feature and as the explicit anchor of the
    ``premium_over_european`` head.
    """
    omega = physical_features[:, _INDEX["option_type"]]
    spot = physical_features[:, _INDEX["spot"]]
    strike = physical_features[:, _INDEX["strike"]]
    maturity = physical_features[:, _INDEX["maturity"]]
    rate = physical_features[:, _INDEX["rate"]]
    dividend_yield = physical_features[:, _INDEX["dividend_yield"]]
    volatility = physical_features[:, _INDEX["volatility"]]
    total_volatility = volatility * torch.sqrt(maturity)
    discounted = spot * torch.exp(-dividend_yield * maturity)
    discounted_strike = strike * torch.exp(-rate * maturity)
    d1 = torch.log(discounted / discounted_strike) / total_volatility + 0.5 * total_volatility
    d2 = d1 - total_volatility
    return omega * (
        discounted * _standard_normal_cdf(omega * d1)
        - discounted_strike * _standard_normal_cdf(omega * d2)
    )


def intrinsic_value(physical_features: torch.Tensor) -> torch.Tensor:
    """Payoff at the current spot."""
    omega = physical_features[:, _INDEX["option_type"]]
    spot = physical_features[:, _INDEX["spot"]]
    strike = physical_features[:, _INDEX["strike"]]
    return torch.clamp(omega * (spot - strike), min=0.0)


def discounted_spot(physical_features: torch.Tensor) -> torch.Tensor:
    """``A = S * exp(-q * T)``, the reconstruction scale of the normalized target."""
    return physical_features[:, _INDEX["spot"]] * torch.exp(
        -physical_features[:, _INDEX["dividend_yield"]] * physical_features[:, _INDEX["maturity"]]
    )


def base_features(physical_features: torch.Tensor) -> torch.Tensor:
    """The five ``american_forward_carry_v1`` coordinates."""
    spot = physical_features[:, _INDEX["spot"]]
    strike = physical_features[:, _INDEX["strike"]]
    maturity = physical_features[:, _INDEX["maturity"]]
    rate_time = physical_features[:, _INDEX["rate"]] * maturity
    yield_time = physical_features[:, _INDEX["dividend_yield"]] * maturity
    return torch.stack(
        (
            physical_features[:, _INDEX["option_type"]],
            torch.log(spot / strike) + rate_time - yield_time,
            physical_features[:, _INDEX["volatility"]] * torch.sqrt(maturity),
            rate_time,
            yield_time,
        ),
        dim=1,
    )


def conditioning_feature(name: str, physical_features: torch.Tensor) -> torch.Tensor:
    """One named conditioning feature, normalized by the discounted spot."""
    if name not in CONDITIONING_FEATURES:
        raise ValueError(f"unknown conditioning feature {name!r}")
    scale = discounted_spot(physical_features)
    if name == "european_price_ratio":
        return european_price(physical_features) / scale
    if name == "intrinsic_ratio":
        return intrinsic_value(physical_features) / scale
    return (european_price(physical_features) - intrinsic_value(physical_features)) / scale


def feature_order(conditioning: tuple[str, ...]) -> tuple[str, ...]:
    return (*BASE_FEATURE_ORDER, *conditioning)


def network_features(
    physical_features: torch.Tensor, conditioning: tuple[str, ...] = ()
) -> torch.Tensor:
    """Build the full unstandardized network input."""
    validate_physical_features(physical_features)
    columns = [base_features(physical_features)]
    for name in conditioning:
        columns.append(conditioning_feature(name, physical_features).unsqueeze(1))
    features = torch.cat(columns, dim=1)
    if not bool(torch.isfinite(features).all()):
        raise PhysicalInputError(f"{REPRESENTATION} is outside its domain")
    return features


def representation_arrays(
    physical_features: np.ndarray,
    prices: np.ndarray,
    conditioning: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """NumPy view of the same construction, for fitting train-only scaling.

    It calls the torch path rather than reimplementing it, so the fitted scaling
    can never drift from what the model actually consumes.
    """
    validate_physical_features(physical_features)
    rows = physical_features.shape[0]
    if prices.shape != (rows,) or not bool(np.isfinite(prices).all()):
        raise ValueError("prices must be a finite one-dimensional array matching rows")
    tensor = torch.as_tensor(np.ascontiguousarray(physical_features), dtype=torch.float64)
    with torch.no_grad():
        features = network_features(tensor, conditioning).numpy()
        scale = discounted_spot(tensor).numpy()
    if bool((scale <= 0.0).any()) or not bool(np.isfinite(scale).all()):
        raise PhysicalInputError(f"{REPRESENTATION} is outside its domain")
    targets = prices / scale
    if not bool(np.isfinite(targets).all()):
        raise PhysicalInputError("normalized American target is outside its domain")
    return np.ascontiguousarray(features), np.ascontiguousarray(targets)


class AmericanDevPriceModel(nn.Module):
    """Physical float64 wrapper: features, standardization and reconstruction.

    ``forward`` maps the seven physical contract inputs to a physical price, so
    the evaluation path is the deployed one and the Task 9G bound and shape
    diagnostics can be applied to it unchanged.
    """

    def __init__(
        self,
        network: nn.Module,
        scaling: Scaling,
        *,
        head: str = "direct",
        conditioning: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        if head not in HEADS:
            raise ValueError(f"unknown head {head!r}")
        for name in conditioning:
            if name not in CONDITIONING_FEATURES:
                raise ValueError(f"unknown conditioning feature {name!r}")
        if len(set(conditioning)) != len(conditioning):
            raise ValueError("conditioning features must be distinct")
        expected = len(feature_order(conditioning))
        if scaling.feature_mean.shape != (expected,) or scaling.feature_scale.shape != (expected,):
            raise ValueError(f"scaling must have {expected} feature coordinates")
        if (
            not bool(np.isfinite(scaling.feature_mean).all())
            or not bool(np.isfinite(scaling.feature_scale).all())
            or bool((scaling.feature_scale <= 0.0).any())
            or not np.isfinite(scaling.price_mean)
            or not np.isfinite(scaling.price_scale)
            or scaling.price_scale <= 0.0
        ):
            raise ValueError("scaling must be finite with positive scales")
        self.head = head
        self.conditioning = tuple(conditioning)
        self.representation = REPRESENTATION
        self.network = network
        self.register_buffer(
            "feature_mean", torch.as_tensor(scaling.feature_mean, dtype=torch.float64)
        )
        self.register_buffer(
            "feature_scale", torch.as_tensor(scaling.feature_scale, dtype=torch.float64)
        )
        self.register_buffer("price_mean", torch.tensor(scaling.price_mean, dtype=torch.float64))
        self.register_buffer("price_scale", torch.tensor(scaling.price_scale, dtype=torch.float64))

    def normalized_target(self, physical_features: torch.Tensor) -> torch.Tensor:
        """Return ``u = V / A`` with the head's reconstruction applied."""
        features = network_features(physical_features, self.conditioning)
        standardized = (features - self.feature_mean) / self.feature_scale
        if not bool(torch.isfinite(standardized).all()):
            raise PhysicalInputError("standardized features are outside their domain")
        raw = self.network(standardized)
        if raw.ndim == 2 and raw.shape[1] == 1:
            raw = raw.squeeze(-1)
        if self.head == "direct":
            return raw * self.price_scale + self.price_mean
        # premium_over_european: softplus is non-negative, so the normalized
        # premium added to the anchor is non-negative. Two qualifications keep
        # the claim honest. First, the bound is non-strict: softplus underflows
        # to exactly zero in float64 for a very negative pre-activation, which
        # collapses the price onto the anchor rather than below it. Second, the
        # guarantee is *not* bitwise on the reconstructed physical price:
        # ``forward`` multiplies this normalized value by ``A``, so the price
        # carries an ``A * (E / A)`` round-trip that can land one unit in the
        # last place below ``E``. The shortfall is bounded by a rounding error
        # of the anchor itself -- of order 1e-16 relative, immaterial against a
        # 3e-3 normalized RMSE criterion -- but it is a floating-point
        # near-bound, not an exact one.
        anchor = european_price(physical_features) / discounted_spot(physical_features)
        return anchor + self.price_scale * torch.nn.functional.softplus(raw)

    def forward(self, physical_features: torch.Tensor) -> torch.Tensor:
        if physical_features.dtype != torch.float64 or physical_features.device.type != "cpu":
            raise PhysicalInputError("Task 9H American pricing requires a float64 CPU tensor")
        scale = discounted_spot(physical_features)
        price = scale * self.normalized_target(physical_features)
        if not bool(torch.isfinite(price).all()):
            raise PhysicalInputError("reconstructed American price is outside its domain")
        return price
