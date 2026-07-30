"""C++-compatible smooth neural network and physical-unit transforms."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import torch
from torch import nn

from .config import (
    FEATURE_ORDER,
    FORWARD_NORMALIZED_REPRESENTATION,
    RAW_PHYSICAL_REPRESENTATION,
    REPRESENTATION_FEATURE_ORDER,
)

NO_OUTPUT_CONSTRAINT = "none_v1"
EUROPEAN_BOUNDS_CONSTRAINT = "european_bounds_v1"
EUROPEAN_BOUNDS_PROJECTION = (
    "price = min(upper_bound, max(lower_bound, unconstrained_price))"
)
EUROPEAN_BOUNDS_DIFFERENTIABILITY = (
    "Piecewise differentiable. Derivatives are not unique where the "
    "unconstrained price meets a bound or where intrinsic value changes branch."
)
OUTPUT_CONSTRAINTS = {
    NO_OUTPUT_CONSTRAINT,
    EUROPEAN_BOUNDS_CONSTRAINT,
}

# Physical inputs must be strictly positive for the pricing domain to be
# well defined: the forward-normalized representation takes log(spot/strike)
# and sqrt(maturity), and divides by spot * exp(-q * T).
STRICTLY_POSITIVE_FEATURES = ("spot", "strike", "maturity", "volatility")
# The dataset generator encodes calls as +1 and puts as -1; nothing else is a
# meaningful contract input, and a stray 0.0 would silently price as a put.
OPTION_TYPE_ENCODING = {"call": 1.0, "put": -1.0}


class PhysicalInputError(ValueError):
    """Raised when physical pricing inputs fall outside the model's domain.

    Subclasses ``ValueError`` so existing callers that catch the broader type
    keep working.
    """


def validate_physical_features(physical_features: torch.Tensor | np.ndarray) -> None:
    """Reject physical inputs that would silently produce NaN or infinity.

    Checked before any logarithm, square root, division, normalization, or
    network evaluation, so a bad row raises a named, field-specific error
    instead of propagating a quiet NaN into prices and Greeks. Accepts either
    a torch tensor or a NumPy array; the two representation paths share this
    one contract.
    """
    is_tensor = isinstance(physical_features, torch.Tensor)
    isfinite = torch.isfinite if is_tensor else np.isfinite

    if physical_features.ndim != 2:
        raise PhysicalInputError(
            f"physical features must be a 2-D [rows, {len(FEATURE_ORDER)}] array, "
            f"got {physical_features.ndim} dimension(s)"
        )
    if physical_features.shape[1] != len(FEATURE_ORDER):
        raise PhysicalInputError(
            f"physical features must have {len(FEATURE_ORDER)} columns in the order "
            f"{list(FEATURE_ORDER)}, got {physical_features.shape[1]}"
        )
    if physical_features.shape[0] == 0:
        raise PhysicalInputError("physical features must contain at least one row")

    indices = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
    for name in FEATURE_ORDER:
        column = physical_features[:, indices[name]]
        if not bool(isfinite(column).all()):
            raise PhysicalInputError(
                f"physical feature '{name}' contains NaN or infinite value(s)"
            )

    for name in STRICTLY_POSITIVE_FEATURES:
        column = physical_features[:, indices[name]]
        if not bool((column > 0.0).all()):
            raise PhysicalInputError(
                f"physical feature '{name}' must be strictly positive"
            )

    option_type = physical_features[:, indices["option_type"]]
    valid = (option_type == OPTION_TYPE_ENCODING["call"]) | (
        option_type == OPTION_TYPE_ENCODING["put"]
    )
    if not bool(valid.all()):
        raise PhysicalInputError(
            "physical feature 'option_type' must be encoded as "
            f"{OPTION_TYPE_ENCODING['call']} (call) or "
            f"{OPTION_TYPE_ENCODING['put']} (put)"
        )


@dataclass(frozen=True)
class Scaling:
    """Train-fitted affine transforms for inputs and the price target."""

    feature_mean: np.ndarray
    feature_scale: np.ndarray
    price_mean: float
    price_scale: float


def fit_scaling(features: np.ndarray, prices: np.ndarray) -> Scaling:
    """Fit population mean/scale using one training partition only."""
    if features.ndim != 2 or prices.ndim != 1 or features.shape[0] != prices.shape[0]:
        raise ValueError("features must be [rows, features] and prices must match rows")
    if features.shape[0] == 0:
        raise ValueError("cannot fit scaling on an empty training partition")
    feature_mean = features.mean(axis=0, dtype=np.float64)
    feature_scale = features.std(axis=0, ddof=0, dtype=np.float64)
    price_mean = float(prices.mean(dtype=np.float64))
    price_scale = float(prices.std(ddof=0, dtype=np.float64))
    if not bool(np.isfinite(feature_mean).all()) or not bool(np.isfinite(feature_scale).all()):
        raise ValueError("feature scaling is non-finite")
    if bool((feature_scale <= 0.0).any()):
        names = np.flatnonzero(feature_scale <= 0.0).tolist()
        raise ValueError(f"training feature(s) have zero variance at indices {names}")
    if not np.isfinite(price_mean) or not np.isfinite(price_scale) or price_scale <= 0.0:
        raise ValueError("training prices must have finite, positive variance")
    return Scaling(feature_mean, feature_scale, price_mean, price_scale)


class PricingMlp(nn.Module):
    """Dense/tanh/scalar-linear architecture accepted by C++ ``SmoothMlp``."""

    def __init__(self, input_dimension: int, hidden_dimensions: tuple[int, ...]) -> None:
        super().__init__()
        dimensions = (input_dimension, *hidden_dimensions, 1)
        modules: list[nn.Module] = []
        for index, (input_size, output_size) in enumerate(pairwise(dimensions)):
            modules.append(nn.Linear(input_size, output_size, dtype=torch.float64))
            if index + 2 < len(dimensions):
                modules.append(nn.Tanh())
        self.layers = nn.Sequential(*modules)

    def forward(self, normalized_features: torch.Tensor) -> torch.Tensor:
        return self.layers(normalized_features).squeeze(-1)


def representation_arrays(
    physical_features: np.ndarray,
    prices: np.ndarray,
    representation: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return network inputs and targets for a declared representation."""
    validate_physical_features(physical_features)
    if prices.ndim != 1 or prices.shape[0] != physical_features.shape[0]:
        raise ValueError("prices must be one-dimensional and match physical_features")
    if representation == RAW_PHYSICAL_REPRESENTATION:
        return physical_features, prices
    if representation != FORWARD_NORMALIZED_REPRESENTATION:
        raise ValueError(f"unsupported representation {representation!r}")

    indices = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
    spot = physical_features[:, indices["spot"]]
    strike = physical_features[:, indices["strike"]]
    maturity = physical_features[:, indices["maturity"]]
    rate = physical_features[:, indices["rate"]]
    dividend_yield = physical_features[:, indices["dividend_yield"]]
    volatility = physical_features[:, indices["volatility"]]
    log_forward_moneyness = (
        np.log(spot / strike) + (rate - dividend_yield) * maturity
    )
    total_volatility = volatility * np.sqrt(maturity)
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    if not bool(np.isfinite(discounted_spot).all()) or bool(
        (discounted_spot <= 0.0).any()
    ):
        raise ValueError("forward-normalized reconstruction scale is outside its domain")
    network_features = np.column_stack(
        (
            physical_features[:, indices["option_type"]],
            log_forward_moneyness,
            total_volatility,
        )
    ).astype(np.float64, copy=False)
    targets = prices / discounted_spot
    if (
        not bool(np.isfinite(network_features).all())
        or not bool(np.isfinite(targets).all())
    ):
        raise ValueError("forward-normalized representation is outside its domain")
    return np.ascontiguousarray(network_features), targets


def forward_normalized_derivative_targets(
    physical_features: np.ndarray,
    prices: np.ndarray,
    deltas: np.ndarray,
    vegas: np.ndarray,
    scaling: Scaling,
) -> np.ndarray:
    """Return ``d standardized target / d standardized (x, v)`` labels.

    Here ``x = log(F/K)``, ``v = sigma * sqrt(T)``, and the unstandardized
    target is ``u = price / (S * exp(-qT))``.
    """
    network_features, normalized_prices = representation_arrays(
        physical_features,
        prices,
        FORWARD_NORMALIZED_REPRESENTATION,
    )
    rows = physical_features.shape[0]
    if deltas.shape != (rows,) or vegas.shape != (rows,):
        raise ValueError("delta and vega labels must match physical_features")
    if not bool(np.isfinite(deltas).all()) or not bool(np.isfinite(vegas).all()):
        raise ValueError("delta and vega labels must be finite")
    if scaling.feature_mean.shape != (network_features.shape[1],):
        raise ValueError("scaling dimension does not match forward representation")

    indices = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
    spot = physical_features[:, indices["spot"]]
    maturity = physical_features[:, indices["maturity"]]
    dividend_yield = physical_features[:, indices["dividend_yield"]]
    discounted_spot = spot * np.exp(-dividend_yield * maturity)

    derivative_x = (
        np.exp(dividend_yield * maturity) * deltas - normalized_prices
    )
    derivative_v = vegas / (discounted_spot * np.sqrt(maturity))
    physical_derivatives = np.column_stack((derivative_x, derivative_v))
    standardized_derivatives = physical_derivatives * (
        scaling.feature_scale[1:] / scaling.price_scale
    )
    if not bool(np.isfinite(standardized_derivatives).all()):
        raise ValueError("forward-normalized derivative targets are non-finite")
    return np.ascontiguousarray(standardized_derivatives)


class PhysicalPriceModel(nn.Module):
    """Wrap a normalized model so forward and autograd operate in physical units."""

    def __init__(
        self,
        network: PricingMlp,
        scaling: Scaling,
        representation: str = RAW_PHYSICAL_REPRESENTATION,
        output_constraint: str = NO_OUTPUT_CONSTRAINT,
    ) -> None:
        super().__init__()
        if representation not in REPRESENTATION_FEATURE_ORDER:
            raise ValueError(f"unsupported representation {representation!r}")
        if output_constraint not in OUTPUT_CONSTRAINTS:
            raise ValueError(f"unsupported output constraint {output_constraint!r}")
        expected_dimension = len(REPRESENTATION_FEATURE_ORDER[representation])
        if scaling.feature_mean.shape != (expected_dimension,):
            raise ValueError("scaling dimension does not match representation")
        self.representation = representation
        self.output_constraint = output_constraint
        self.network = network
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(scaling.feature_mean, dtype=torch.float64),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(scaling.feature_scale, dtype=torch.float64),
        )
        self.register_buffer(
            "price_mean",
            torch.tensor(scaling.price_mean, dtype=torch.float64),
        )
        self.register_buffer(
            "price_scale",
            torch.tensor(scaling.price_scale, dtype=torch.float64),
        )

    def unconstrained_price(self, physical_features: torch.Tensor) -> torch.Tensor:
        """Return physical price before applying any declared output constraint.

        Raises :class:`PhysicalInputError` for inputs outside the pricing
        domain. Validation happens here, ahead of every transform, so both the
        unconstrained result and the projected :meth:`forward` price are
        covered by one check.
        """
        validate_physical_features(physical_features)
        indices = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
        if self.representation == RAW_PHYSICAL_REPRESENTATION:
            network_features = physical_features
            reconstruction_scale: torch.Tensor | float = 1.0
        else:
            spot = physical_features[:, indices["spot"]]
            strike = physical_features[:, indices["strike"]]
            maturity = physical_features[:, indices["maturity"]]
            rate = physical_features[:, indices["rate"]]
            dividend_yield = physical_features[:, indices["dividend_yield"]]
            volatility = physical_features[:, indices["volatility"]]
            network_features = torch.stack(
                (
                    physical_features[:, indices["option_type"]],
                    torch.log(spot / strike)
                    + (rate - dividend_yield) * maturity,
                    volatility * torch.sqrt(maturity),
                ),
                dim=1,
            )
            reconstruction_scale = spot * torch.exp(-dividend_yield * maturity)
        normalized = (network_features - self.feature_mean) / self.feature_scale
        target = self.network(normalized) * self.price_scale + self.price_mean
        return target * reconstruction_scale

    def forward(self, physical_features: torch.Tensor) -> torch.Tensor:
        price = self.unconstrained_price(physical_features)
        if self.output_constraint == NO_OUTPUT_CONSTRAINT:
            return price

        indices = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
        spot = physical_features[:, indices["spot"]]
        strike = physical_features[:, indices["strike"]]
        maturity = physical_features[:, indices["maturity"]]
        rate = physical_features[:, indices["rate"]]
        dividend_yield = physical_features[:, indices["dividend_yield"]]
        discounted_spot = spot * torch.exp(-dividend_yield * maturity)
        discounted_strike = strike * torch.exp(-rate * maturity)
        calls = physical_features[:, indices["option_type"]] > 0.0
        zero = torch.zeros_like(price)
        lower = torch.where(
            calls,
            torch.maximum(discounted_spot - discounted_strike, zero),
            torch.maximum(discounted_strike - discounted_spot, zero),
        )
        upper = torch.where(calls, discounted_spot, discounted_strike)
        return torch.minimum(torch.maximum(price, lower), upper)
