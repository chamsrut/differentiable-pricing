"""Task-specific American neural representation, training, and exact lift.

This module intentionally is not a transfer-learning framework.  It contains
the one five-coordinate representation and the one European-to-American lift
locked for Task 9G.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import nn

from .config import FEATURE_ORDER, FORWARD_NORMALIZED_REPRESENTATION
from .model import (
    NO_OUTPUT_CONSTRAINT,
    PhysicalInputError,
    PhysicalPriceModel,
    PricingMlp,
    Scaling,
    fit_scaling,
    validate_physical_features,
)

AMERICAN_REPRESENTATION: Final = "american_forward_carry_v1"
AMERICAN_FEATURE_ORDER: Final = (
    "option_type",
    "log_forward_moneyness",
    "total_volatility",
    "rate_time",
    "yield_time",
)
PHYSICAL_RECONSTRUCTION: Final = "price = spot * exp(-dividend_yield * maturity) * u"


def american_representation_arrays(
    physical_features: np.ndarray,
    prices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map physical rows to the exact Task 9G five inputs and target ``u``."""
    validate_physical_features(physical_features)
    rows = physical_features.shape[0]
    if prices.shape != (rows,) or not bool(np.isfinite(prices).all()):
        raise ValueError("prices must be a finite one-dimensional array matching rows")
    index = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
    spot = physical_features[:, index["spot"]]
    strike = physical_features[:, index["strike"]]
    maturity = physical_features[:, index["maturity"]]
    rate = physical_features[:, index["rate"]]
    dividend_yield = physical_features[:, index["dividend_yield"]]
    volatility = physical_features[:, index["volatility"]]
    with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
        rate_time = rate * maturity
        yield_time = dividend_yield * maturity
        discounted_spot = spot * np.exp(-yield_time)
        features = np.column_stack(
            (
                physical_features[:, index["option_type"]],
                np.log(spot / strike) + rate_time - yield_time,
                volatility * np.sqrt(maturity),
                rate_time,
                yield_time,
            )
        ).astype(np.float64, copy=False)
        targets = prices / discounted_spot
    if (
        not bool(np.isfinite(discounted_spot).all())
        or bool((discounted_spot <= 0.0).any())
        or not bool(np.isfinite(features).all())
        or not bool(np.isfinite(targets).all())
    ):
        raise PhysicalInputError("american_forward_carry_v1 is outside its domain")
    return np.ascontiguousarray(features), np.ascontiguousarray(targets)


class AmericanPriceModel(nn.Module):
    """Physical float64 wrapper with reconstruction inside the torch graph."""

    def __init__(self, network: PricingMlp, scaling: Scaling) -> None:
        super().__init__()
        if scaling.feature_mean.shape != (len(AMERICAN_FEATURE_ORDER),):
            raise ValueError("American scaling must have five feature coordinates")
        if scaling.feature_scale.shape != (len(AMERICAN_FEATURE_ORDER),):
            raise ValueError("American scaling must have five feature scales")
        if (
            not bool(np.isfinite(scaling.feature_mean).all())
            or not bool(np.isfinite(scaling.feature_scale).all())
            or bool((scaling.feature_scale <= 0.0).any())
            or not np.isfinite(scaling.price_mean)
            or not np.isfinite(scaling.price_scale)
            or scaling.price_scale <= 0.0
        ):
            raise ValueError("American scaling must be finite with positive scales")
        self.network = network
        self.register_buffer(
            "feature_mean", torch.as_tensor(scaling.feature_mean, dtype=torch.float64)
        )
        self.register_buffer(
            "feature_scale", torch.as_tensor(scaling.feature_scale, dtype=torch.float64)
        )
        self.register_buffer("price_mean", torch.tensor(scaling.price_mean, dtype=torch.float64))
        self.register_buffer("price_scale", torch.tensor(scaling.price_scale, dtype=torch.float64))

    def forward(self, physical_features: torch.Tensor) -> torch.Tensor:
        if physical_features.dtype != torch.float64 or physical_features.device.type != "cpu":
            raise PhysicalInputError("American pricing requires a float64 CPU tensor")
        validate_physical_features(physical_features)
        index = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
        spot = physical_features[:, index["spot"]]
        strike = physical_features[:, index["strike"]]
        maturity = physical_features[:, index["maturity"]]
        rate = physical_features[:, index["rate"]]
        dividend_yield = physical_features[:, index["dividend_yield"]]
        volatility = physical_features[:, index["volatility"]]
        rate_time = rate * maturity
        yield_time = dividend_yield * maturity
        network_features = torch.stack(
            (
                physical_features[:, index["option_type"]],
                torch.log(spot / strike) + rate_time - yield_time,
                volatility * torch.sqrt(maturity),
                rate_time,
                yield_time,
            ),
            dim=1,
        )
        discounted_spot = spot * torch.exp(-yield_time)
        if (
            not bool(torch.isfinite(network_features).all())
            or not bool(torch.isfinite(discounted_spot).all())
            or not bool((discounted_spot > 0.0).all())
        ):
            raise PhysicalInputError("american_forward_carry_v1 is outside its domain")
        standardized = (network_features - self.feature_mean) / self.feature_scale
        if not bool(torch.isfinite(standardized).all()):
            raise PhysicalInputError("American standardized features are outside their domain")
        normalized_price = self.network(standardized) * self.price_scale + self.price_mean
        price = discounted_spot * normalized_price
        if not bool(torch.isfinite(price).all()):
            raise PhysicalInputError("American reconstructed price is outside its domain")
        return price


class ExactLiftError(ValueError):
    """Raised with structured evidence when the task-specific lift is not exact."""

    def __init__(self, evidence: dict[str, object]) -> None:
        super().__init__(
            "exact European-to-American lift failed: maximum physical price "
            f"difference {evidence['maximum_absolute_difference']:.17g}"
        )
        self.evidence = evidence


def scaling_from_arrays(physical_features: np.ndarray, prices: np.ndarray) -> Scaling:
    features, targets = american_representation_arrays(physical_features, prices)
    return fit_scaling(features, targets)


def lift_european_network(
    source: PhysicalPriceModel,
    target_scaling: Scaling,
) -> AmericanPriceModel:
    """Apply the one exact Task 9G 3-input to 5-input algebraic lift."""
    if source.representation != FORWARD_NORMALIZED_REPRESENTATION:
        raise ValueError("source must use forward_normalized_v1")
    if source.output_constraint != NO_OUTPUT_CONSTRAINT:
        raise ValueError("source must be the unconstrained European artifact")
    if target_scaling.feature_mean.shape != (5,) or target_scaling.feature_scale.shape != (5,):
        raise ValueError("target scaling must have five coordinates")
    source_linears = [layer for layer in source.network.layers if isinstance(layer, nn.Linear)]
    if len(source_linears) != 4 or [layer.out_features for layer in source_linears] != [
        64,
        64,
        64,
        1,
    ]:
        raise ValueError("source architecture must be 3->[64,64,64]->1")
    if source_linears[0].in_features != 3:
        raise ValueError("source first layer must have three inputs")

    target_network = PricingMlp(5, (64, 64, 64))
    target_linears = [layer for layer in target_network.layers if isinstance(layer, nn.Linear)]
    source_mean = source.feature_mean.detach().cpu().numpy()
    source_scale = source.feature_scale.detach().cpu().numpy()
    source_target_mean = float(source.price_mean)
    source_target_scale = float(source.price_scale)
    with torch.no_grad():
        source_first_weight = source_linears[0].weight.detach()
        target_linears[0].weight.zero_()
        rebased = source_first_weight * torch.as_tensor(
            target_scaling.feature_scale[:3] / source_scale,
            dtype=torch.float64,
        )
        target_linears[0].weight[:, :3].copy_(rebased)
        bias_shift = source_first_weight @ torch.as_tensor(
            (target_scaling.feature_mean[:3] - source_mean) / source_scale,
            dtype=torch.float64,
        )
        target_linears[0].bias.copy_(source_linears[0].bias + bias_shift)
        # The rate_time and yield_time columns remain exactly zero.
        for source_layer, target_layer in zip(
            source_linears[1:-1], target_linears[1:-1], strict=True
        ):
            target_layer.weight.copy_(source_layer.weight)
            target_layer.bias.copy_(source_layer.bias)
        target_linears[-1].weight.copy_(
            source_linears[-1].weight * (source_target_scale / target_scaling.price_scale)
        )
        target_linears[-1].bias.copy_(
            (
                source_target_scale * source_linears[-1].bias
                + source_target_mean
                - target_scaling.price_mean
            )
            / target_scaling.price_scale
        )
    target_network.eval()
    return AmericanPriceModel(target_network, target_scaling).eval()


def verify_exact_lift(
    source: PhysicalPriceModel,
    lifted: AmericanPriceModel,
    probes: np.ndarray,
    *,
    rtol: float = 1.0e-12,
    atol: float = 1.0e-12,
) -> dict[str, object]:
    """Compare fully reconstructed unconstrained prices at physical probes."""
    if rtol > 1.0e-12 or atol > 1.0e-12 or rtol < 0.0 or atol < 0.0:
        raise ValueError("exact-lift tolerances must be between zero and 1e-12")
    tensor = torch.as_tensor(probes, dtype=torch.float64)
    with torch.no_grad():
        source_prices = source.unconstrained_price(tensor).cpu().numpy()
        lifted_prices = lifted(tensor).cpu().numpy()
    difference = np.abs(source_prices - lifted_prices)
    relative = difference / np.maximum(np.abs(source_prices), np.finfo(np.float64).tiny)
    evidence: dict[str, object] = {
        "passed": bool(np.allclose(lifted_prices, source_prices, rtol=rtol, atol=atol)),
        "probes": int(probes.shape[0]),
        "probe_prices": [
            {"source": float(source_price), "lifted": float(lifted_price)}
            for source_price, lifted_price in zip(source_prices, lifted_prices, strict=True)
        ],
        "maximum_absolute_difference": float(difference.max(initial=0.0)),
        "maximum_relative_difference": float(relative.max(initial=0.0)),
        "relative_tolerance": float(rtol),
        "absolute_tolerance": float(atol),
    }
    if not evidence["passed"]:
        raise ExactLiftError(evidence)
    return evidence


def select_rows_by_hash(sample_ids: np.ndarray, budget: int, salt: str) -> np.ndarray:
    """Return the fixed lowest SHA-256 ranks, with sample ID as tie-breaker."""
    if budget <= 0 or budget > sample_ids.shape[0] or not salt:
        raise ValueError("row budget must be in [1, rows] and salt must be non-empty")
    ranked = sorted(
        (
            hashlib.sha256(f"{salt}\0{sample_id}".encode()).digest(),
            str(sample_id),
            index,
        )
        for index, sample_id in enumerate(sample_ids.tolist())
    )
    return np.asarray([entry[2] for entry in ranked[:budget]], dtype=np.int64)


@dataclass(frozen=True)
class AmericanFitResult:
    model: AmericanPriceModel
    best_epoch: int
    best_validation_mse: float
    history: tuple[dict[str, float | int], ...]


def fit_price_model(
    train_features: np.ndarray,
    train_prices: np.ndarray,
    validation_features: np.ndarray,
    validation_prices: np.ndarray,
    scaling: Scaling,
    *,
    seed: int,
    batch_size: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    amsgrad: bool,
    maximize: bool,
    foreach: bool,
    capturable: bool,
    differentiable: bool,
    fused: bool,
    minimum_learning_rate: float,
    schedule_period_epochs: int,
    schedule_last_epoch: int,
    num_threads: int,
    initial_model: AmericanPriceModel | None = None,
) -> AmericanFitResult:
    """Fit exactly one fixed price-only arm with validation-only selection."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(num_threads)
    train_x, train_u = american_representation_arrays(train_features, train_prices)
    validation_x, validation_u = american_representation_arrays(
        validation_features, validation_prices
    )
    x_train = torch.as_tensor(
        (train_x - scaling.feature_mean) / scaling.feature_scale, dtype=torch.float64
    )
    y_train = torch.as_tensor(
        (train_u - scaling.price_mean) / scaling.price_scale, dtype=torch.float64
    )
    x_validation = torch.as_tensor(
        (validation_x - scaling.feature_mean) / scaling.feature_scale,
        dtype=torch.float64,
    )
    y_validation = torch.as_tensor(
        (validation_u - scaling.price_mean) / scaling.price_scale,
        dtype=torch.float64,
    )
    network = PricingMlp(5, (64, 64, 64)) if initial_model is None else initial_model.network
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=learning_rate,
        betas=(beta1, beta2),
        eps=epsilon,
        weight_decay=weight_decay,
        amsgrad=amsgrad,
        maximize=maximize,
        foreach=foreach,
        capturable=capturable,
        differentiable=differentiable,
        fused=fused,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=schedule_period_epochs,
        eta_min=minimum_learning_rate,
        last_epoch=schedule_last_epoch,
    )
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best_mse = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    for epoch in range(1, epochs + 1):
        network.train()
        order = torch.randperm(x_train.shape[0], generator=generator)
        for start in range(0, x_train.shape[0], batch_size):
            indices = order[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            loss = torch.mean(torch.square(network(x_train[indices]) - y_train[indices]))
            loss.backward()
            optimizer.step()
        scheduler.step()
        network.eval()
        with torch.no_grad():
            validation_mse = float(torch.mean(torch.square(network(x_validation) - y_validation)))
        if not np.isfinite(validation_mse):
            raise ValueError("training produced a non-finite validation loss")
        history.append(
            {
                "epoch": epoch,
                "validation_standardized_target_mse": validation_mse,
                "learning_rate": float(scheduler.get_last_lr()[0]),
            }
        )
        if validation_mse < best_mse:
            best_mse = validation_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().clone() for key, value in network.state_dict().items()
            }
    if best_state is None:
        raise ValueError("training did not produce a checkpoint")
    network.load_state_dict(best_state, strict=True)
    network.eval()
    return AmericanFitResult(
        AmericanPriceModel(network, scaling).eval(),
        best_epoch,
        best_mse,
        tuple(history),
    )
