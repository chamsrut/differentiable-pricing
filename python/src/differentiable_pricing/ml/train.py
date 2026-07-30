"""Train the deterministic European-option neural baseline."""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .artifact import (
    ARTIFACT_MANIFEST,
    CHECKPOINT_WARNING,
    WEIGHTS_FILE,
    save_weights,
    write_json,
)
from .config import (
    CONFIG_SCHEMA_VERSION,
    RAW_PHYSICAL_REPRESENTATION,
    REPRESENTATION_FEATURE_ORDER,
    NeuralBaselineConfig,
    ObjectiveConfig,
    TrainingConfigError,
    load_training_config,
)
from .dataset import (
    DatasetIdentity,
    DatasetLoadError,
    SplitData,
    load_dataset_manifest,
    load_split,
)
from .model import (
    PricingMlp,
    Scaling,
    fit_scaling,
    forward_normalized_derivative_targets,
    representation_arrays,
)


class TrainingError(RuntimeError):
    """Raised when training cannot produce a complete artifact."""


@dataclass(frozen=True)
class TrainingResult:
    network: PricingMlp
    scaling: Scaling
    history: tuple[dict[str, float | int], ...]
    best_epoch: int
    best_validation_price_loss: float
    best_validation_objective: float


@dataclass(frozen=True)
class LossMetrics:
    price_mse: float
    gradient_mse: float | None
    objective: float


def _normalized_tensors(
    split: SplitData,
    scaling: Scaling,
    representation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    network_features, targets = representation_arrays(
        split.features,
        split.columns["price"],
        representation,
    )
    features = (network_features - scaling.feature_mean) / scaling.feature_scale
    prices = (targets - scaling.price_mean) / scaling.price_scale
    return (
        torch.as_tensor(features, dtype=torch.float64),
        torch.as_tensor(prices, dtype=torch.float64),
    )


def _normalized_derivative_tensors(
    split: SplitData,
    scaling: Scaling,
) -> torch.Tensor:
    try:
        delta = split.columns["delta"]
        vega = split.columns["vega"]
    except KeyError as error:
        raise ValueError(
            "first-derivative training requires delta and vega labels"
        ) from error
    derivatives = forward_normalized_derivative_targets(
        split.features,
        split.columns["price"],
        delta,
        vega,
        scaling,
    )
    return torch.as_tensor(derivatives, dtype=torch.float64)


def _loss_metrics(
    network: PricingMlp,
    features: torch.Tensor,
    targets: torch.Tensor,
    derivative_targets: torch.Tensor | None,
    batch_size: int,
    objective: ObjectiveConfig,
) -> LossMetrics:
    price_squared_error = 0.0
    gradient_squared_error = 0.0
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        inputs = features[start:stop]
        if derivative_targets is None:
            with torch.no_grad():
                prediction = network(inputs)
        else:
            inputs = inputs.detach().requires_grad_(True)
            prediction = network(inputs)
            gradient = torch.autograd.grad(prediction.sum(), inputs)[0][:, 1:]
            gradient_residual = gradient - derivative_targets[start:stop]
            gradient_squared_error += float(
                torch.sum(torch.square(gradient_residual))
            )
        residual = prediction.detach() - targets[start:stop]
        price_squared_error += float(torch.dot(residual, residual))

    rows = int(features.shape[0])
    price_mse = price_squared_error / rows
    gradient_mse = (
        None
        if derivative_targets is None
        else gradient_squared_error / (rows * derivative_targets.shape[1])
    )
    combined = objective.price_weight * price_mse
    if gradient_mse is not None:
        combined += objective.gradient_weight * gradient_mse
    return LossMetrics(price_mse, gradient_mse, combined)


def fit_model(
    train: SplitData,
    validation: SplitData,
    config: NeuralBaselineConfig,
) -> TrainingResult:
    """Fit the configured objective with transforms estimated from ``train`` only."""
    torch.manual_seed(config.training.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(config.training.num_threads)

    train_representation, train_prices = representation_arrays(
        train.features,
        train.columns["price"],
        config.representation,
    )
    scaling = fit_scaling(train_representation, train_prices)
    train_features, train_targets = _normalized_tensors(
        train,
        scaling,
        config.representation,
    )
    validation_features, validation_targets = _normalized_tensors(
        validation,
        scaling,
        config.representation,
    )
    train_derivatives = (
        _normalized_derivative_tensors(train, scaling)
        if config.objective.uses_first_derivatives
        else None
    )
    validation_derivatives = (
        _normalized_derivative_tensors(validation, scaling)
        if config.objective.uses_first_derivatives
        else None
    )

    network = PricingMlp(
        len(REPRESENTATION_FEATURE_ORDER[config.representation]),
        config.model.hidden_dimensions,
    )
    optimizer = torch.optim.AdamW(
        network.parameters(),
        lr=config.optimizer.learning_rate,
        weight_decay=config.optimizer.weight_decay,
    )
    mean_squared_error = nn.MSELoss(reduction="mean")
    permutation_generator = torch.Generator(device="cpu")
    permutation_generator.manual_seed(config.training.seed)

    best_loss = float("inf")
    best_price_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.training.max_epochs + 1):
        network.train()
        permutation = torch.randperm(
            train_features.shape[0],
            generator=permutation_generator,
        )
        for start in range(0, train_features.shape[0], config.training.batch_size):
            indices = permutation[start : start + config.training.batch_size]
            optimizer.zero_grad(set_to_none=True)
            inputs = train_features[indices]
            if train_derivatives is not None:
                inputs = inputs.detach().requires_grad_(True)
            prediction = network(inputs)
            price_loss = mean_squared_error(prediction, train_targets[indices])
            loss = config.objective.price_weight * price_loss
            if train_derivatives is not None:
                gradient = torch.autograd.grad(
                    prediction.sum(),
                    inputs,
                    create_graph=True,
                )[0][:, 1:]
                gradient_loss = mean_squared_error(
                    gradient,
                    train_derivatives[indices],
                )
                loss = loss + config.objective.gradient_weight * gradient_loss
            loss.backward()
            optimizer.step()

        network.eval()
        train_metrics = _loss_metrics(
            network,
            train_features,
            train_targets,
            train_derivatives,
            config.training.batch_size,
            config.objective,
        )
        validation_metrics = _loss_metrics(
            network,
            validation_features,
            validation_targets,
            validation_derivatives,
            config.training.batch_size,
            config.objective,
        )
        record: dict[str, float | int] = {
            "epoch": epoch,
            "train_normalized_price_mse": train_metrics.price_mse,
            "validation_normalized_price_mse": validation_metrics.price_mse,
            "train_objective": train_metrics.objective,
            "validation_objective": validation_metrics.objective,
        }
        if (
            train_metrics.gradient_mse is not None
            and validation_metrics.gradient_mse is not None
        ):
            record["train_normalized_first_derivative_mse"] = (
                train_metrics.gradient_mse
            )
            record["validation_normalized_first_derivative_mse"] = (
                validation_metrics.gradient_mse
            )
        history.append(record)

        if (
            validation_metrics.objective
            < best_loss - config.training.early_stopping_min_delta
        ):
            best_loss = validation_metrics.objective
            best_price_loss = validation_metrics.price_mse
            best_epoch = epoch
            best_state = {
                key: value.detach().clone()
                for key, value in network.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

    if best_state is None or not np.isfinite(best_loss):
        raise TrainingError("training did not produce a finite validation loss")
    network.load_state_dict(best_state, strict=True)
    network.eval()
    return TrainingResult(
        network=network,
        scaling=scaling,
        history=tuple(history),
        best_epoch=best_epoch,
        best_validation_price_loss=best_price_loss,
        best_validation_objective=best_loss,
    )


def _artifact_payload(
    config: NeuralBaselineConfig,
    identity: DatasetIdentity,
    manifest: dict[str, Any],
    result: TrainingResult,
    weights_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": config.artifact_schema_version,
        "experiment_name": config.experiment_name,
        "dtype": config.dtype,
        "feature_order": list(config.features),
        "representation": {
            "name": config.representation,
            "network_feature_order": list(
                REPRESENTATION_FEATURE_ORDER[config.representation]
            ),
            "physical_reconstruction": (
                "identity"
                if config.representation == RAW_PHYSICAL_REPRESENTATION
                else "price = discounted_spot * normalized_forward_price"
            ),
        },
        "option_type_encoding": {"call": 1.0, "put": -1.0},
        "architecture": {
            "input_dimension": len(
                REPRESENTATION_FEATURE_ORDER[config.representation]
            ),
            "hidden_dimensions": list(config.model.hidden_dimensions),
            "activation": config.model.activation,
            "output_dimension": config.model.output_dimension,
            "dense_weight_layout": "PyTorch [output,input], row-major when flattened for C++",
        },
        "transform": {
            "fit_partition": "train",
            "feature_mean": result.scaling.feature_mean.tolist(),
            "feature_scale": result.scaling.feature_scale.tolist(),
            "price_mean": result.scaling.price_mean,
            "price_scale": result.scaling.price_scale,
            "population_standard_deviation_ddof": 0,
        },
        "objective": {
            "name": config.objective.name,
            "trained_target": (
                "normalized_price"
                if config.representation == RAW_PHYSICAL_REPRESENTATION
                else "standardized_price_over_discounted_spot"
            ),
            "loss": (
                "mean_squared_error"
                if not config.objective.uses_first_derivatives
                else (
                    "price_weight * standardized_price_mse + gradient_weight * "
                    "mean_standardized_coordinate_derivative_mse"
                )
            ),
            "price_weight": config.objective.price_weight,
            "gradient_weight": config.objective.gradient_weight,
            "differential_coordinates": (
                ["log_forward_moneyness", "total_volatility"]
                if config.objective.uses_first_derivatives
                else []
            ),
            "greek_labels_used_for_training": (
                ["delta", "vega"]
                if config.objective.uses_first_derivatives
                else []
            ),
        },
        "optimizer": {
            "name": config.optimizer.name,
            "learning_rate": config.optimizer.learning_rate,
            "weight_decay": config.optimizer.weight_decay,
        },
        "training": {
            "seed": config.training.seed,
            "batch_size": config.training.batch_size,
            "maximum_epochs": config.training.max_epochs,
            "completed_epochs": len(result.history),
            "early_stopping_patience": config.training.early_stopping_patience,
            "early_stopping_min_delta": config.training.early_stopping_min_delta,
            "best_epoch": result.best_epoch,
            "selection_metric": "validation_objective",
            "best_validation_normalized_price_mse": (
                result.best_validation_price_loss
            ),
            "best_validation_objective": result.best_validation_objective,
            "history": list(result.history),
            "partitions_read": ["train", "validation"],
            "held_out_partition": "interpolation_test",
        },
        "evaluation": {
            "batch_size": config.evaluation.batch_size,
            "core_max_absolute_z": config.evaluation.core_max_absolute_z,
            "tail_max_absolute_z": config.evaluation.tail_max_absolute_z,
        },
        "dataset": {
            "manifest_sha256": identity.manifest_sha256,
            "config_sha256": identity.config_sha256,
            "schema_version": identity.schema_version,
            "generator_version": identity.generator_version,
            "oracle": manifest["oracle"],
            "row_counts": manifest["row_counts"],
        },
        "training_config": {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "sha256": config.sha256,
        },
        "weights": {
            "file": WEIGHTS_FILE,
            "format": "NumPy NPZ, float64 arrays, allow_pickle=False",
            "sha256": weights_sha256,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "device": "cpu",
            "num_threads": config.training.num_threads,
            "deterministic_algorithms": True,
        },
        "checkpoint_security": CHECKPOINT_WARNING,
        "limitations": [
            "Synthetic in-envelope Black-Scholes interpolation only; no market calibration.",
            (
                "Delta and vega labels supervise first coordinate derivatives; "
                "gamma labels are not used."
                if config.objective.uses_first_derivatives
                else (
                    "Price-only training; autograd Greeks are derivatives of the "
                    "learned network."
                )
            ),
            "No boundary, extrapolation, scenario, latency, or C++ export claim is made.",
            "Black-Scholes analytic pricing may be faster than this neural baseline.",
        ],
    }


def train_to_artifact(
    dataset: Path,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    """Load train/validation, fit, and atomically publish one artifact directory."""
    if output.exists():
        raise TrainingError(f"output '{output}' already exists; refusing to overwrite")
    config = load_training_config(config_path)
    manifest, identity = load_dataset_manifest(dataset)
    load_labels = config.objective.uses_first_derivatives
    train = load_split(dataset, manifest, "train", evaluation=load_labels)
    validation = load_split(
        dataset,
        manifest,
        "validation",
        evaluation=load_labels,
    )
    try:
        result = fit_model(train, validation, config)
    except ValueError as error:
        raise TrainingError(f"cannot fit neural baseline: {error}") from error

    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output.name}-", dir=output.parent)
        )
    except OSError as error:
        raise TrainingError(f"cannot prepare artifact output '{output}': {error}") from error
    try:
        weights_sha256 = save_weights(temporary / WEIGHTS_FILE, result.network)
        payload = _artifact_payload(
            config,
            identity,
            manifest,
            result,
            weights_sha256,
        )
        write_json(temporary / ARTIFACT_MANIFEST, payload)
        temporary.rename(output)
    except (OSError, TypeError, ValueError) as error:
        shutil.rmtree(temporary, ignore_errors=True)
        raise TrainingError(f"cannot publish artifact '{output}': {error}") from error
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        payload = train_to_artifact(
            arguments.dataset,
            arguments.config,
            arguments.output,
        )
    except (DatasetLoadError, TrainingConfigError, TrainingError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "artifact": str(arguments.output),
                "best_epoch": payload["training"]["best_epoch"],
                "best_validation_normalized_price_mse": payload["training"][
                    "best_validation_normalized_price_mse"
                ],
                "best_validation_objective": payload["training"][
                    "best_validation_objective"
                ],
                "weights_sha256": payload["weights"]["sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
