"""Evaluate physical prices and autograd Greeks on an explicitly named partition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..data.generate import LABEL_FIELDS
from .artifact import (
    ARTIFACT_MANIFEST,
    ArtifactError,
    load_physical_model,
    write_json_atomic,
)
from .config import FEATURE_ORDER
from .dataset import (
    DatasetLoadError,
    SplitData,
    load_dataset_manifest,
    load_split,
)
from .model import NO_OUTPUT_CONSTRAINT, PhysicalPriceModel

EVALUATION_SCHEMA_VERSION: Final = "european-neural-evaluation/v3"
EVALUATION_PARTITIONS: Final = ("validation", "interpolation_test")
MATERIAL_ARBITRAGE_RELATIVE_TOLERANCE: Final = 1.0e-6
PARTITION_ROLES: Final = {
    "validation": "model_selection",
    "interpolation_test": "locked_final_evaluation",
}
GREEK_FEATURES: Final = {
    "delta": "spot",
    "vega": "volatility",
    "rho": "rate",
}


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot produce a trustworthy report."""


def predict_with_greeks(
    model: PhysicalPriceModel,
    features: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Return price and physical-unit Greeks from autograd.

    Theta follows the dataset convention: calendar-time decay is
    ``-d price / d maturity``.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    indices = {name: FEATURE_ORDER.index(name) for name in FEATURE_ORDER}
    chunks: dict[str, list[np.ndarray]] = {
        "price": [],
        "delta": [],
        "gamma": [],
        "vega": [],
        "theta": [],
        "rho": [],
    }
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for start in range(0, features.shape[0], batch_size):
        stop = min(start + batch_size, features.shape[0])
        inputs = torch.as_tensor(
            features[start:stop],
            dtype=torch.float64,
        ).clone()
        inputs.requires_grad_(True)
        price = model(inputs)
        first = torch.autograd.grad(price.sum(), inputs, create_graph=True)[0]

        chunks["price"].append(price.detach().cpu().numpy())
        for greek, feature in GREEK_FEATURES.items():
            chunks[greek].append(
                first[:, indices[feature]].detach().cpu().numpy()
            )
        chunks["theta"].append(
            (-first[:, indices["maturity"]]).detach().cpu().numpy()
        )
        spot_delta = first[:, indices["spot"]]
        if spot_delta.requires_grad:
            second = torch.autograd.grad(
                spot_delta.sum(),
                inputs,
                allow_unused=True,
            )[0]
            gamma = (
                second[:, indices["spot"]]
                if second is not None
                else torch.zeros_like(spot_delta)
            )
        else:
            gamma = torch.zeros_like(spot_delta)
        chunks["gamma"].append(gamma.detach().cpu().numpy())

    return {
        name: np.concatenate(values).astype(np.float64, copy=False)
        for name, values in chunks.items()
    }


def error_metrics(reference: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    """Summarize finite prediction error without a fragile relative metric."""
    if reference.shape != prediction.shape:
        raise ValueError("reference and prediction shapes differ")
    if reference.size == 0:
        return {
            "count": 0,
            "mae": None,
            "rmse": None,
            "p95_absolute_error": None,
            "p99_absolute_error": None,
            "maximum_absolute_error": None,
        }
    residual = prediction - reference
    if not bool(np.isfinite(residual).all()):
        raise EvaluationError("model produced non-finite predictions")
    absolute = np.abs(residual)
    return {
        "count": int(reference.size),
        "mae": float(absolute.mean()),
        "rmse": float(math.sqrt(float(np.mean(np.square(residual))))),
        "p95_absolute_error": float(np.quantile(absolute, 0.95, method="linear")),
        "p99_absolute_error": float(np.quantile(absolute, 0.99, method="linear")),
        "maximum_absolute_error": float(absolute.max()),
    }


def slice_metrics(
    split: SplitData,
    predictions: Mapping[str, np.ndarray],
    mask: np.ndarray,
) -> dict[str, Any]:
    metrics = {
        label: error_metrics(split.columns[label][mask], predictions[label][mask])
        for label in LABEL_FIELDS
    }
    metrics["price_over_spot"] = error_metrics(
        split.columns["price"][mask] / split.columns["spot"][mask],
        predictions["price"][mask] / split.columns["spot"][mask],
    )
    return {"rows": int(mask.sum()), "metrics": metrics}


def moneyness_masks(
    split: SplitData,
    *,
    core_max: float,
    tail_max: float,
) -> dict[str, np.ndarray]:
    absolute_z = np.abs(split.columns["log_forward_moneyness"]) / (
        split.columns["volatility"] * np.sqrt(split.columns["maturity"])
    )
    return {
        "core": absolute_z <= core_max,
        "tail": (absolute_z > core_max) & (absolute_z <= tail_max),
        "extreme": absolute_z > tail_max,
    }


def no_arbitrage_violations(
    split: SplitData,
    predicted_price: np.ndarray,
    *,
    relative_tolerance: float = 1.0e-12,
) -> dict[str, Any]:
    spot = split.columns["spot"]
    strike = split.columns["strike"]
    maturity = split.columns["maturity"]
    rate = split.columns["rate"]
    dividend_yield = split.columns["dividend_yield"]
    discounted_spot = spot * np.exp(-dividend_yield * maturity)
    discounted_strike = strike * np.exp(-rate * maturity)
    calls = split.option_types == "call"
    lower = np.where(
        calls,
        np.maximum(discounted_spot - discounted_strike, 0.0),
        np.maximum(discounted_strike - discounted_spot, 0.0),
    )
    upper = np.where(calls, discounted_spot, discounted_strike)
    scale = np.maximum(discounted_spot, discounted_strike)
    tolerance = relative_tolerance * scale
    below = predicted_price < lower - tolerance
    above = predicted_price > upper + tolerance
    lower_shortfall = np.maximum(lower - predicted_price, 0.0)
    upper_excess = np.maximum(predicted_price - upper, 0.0)
    maximum_excess = np.maximum(lower_shortfall, upper_excess)
    relative_excess = np.divide(
        maximum_excess,
        scale,
        out=np.zeros_like(maximum_excess),
        where=scale > 0.0,
    )
    material = relative_excess > MATERIAL_ARBITRAGE_RELATIVE_TOLERANCE
    return {
        "relative_tolerance": relative_tolerance,
        "material_relative_tolerance": MATERIAL_ARBITRAGE_RELATIVE_TOLERANCE,
        "below_lower_bound": int(below.sum()),
        "above_upper_bound": int(above.sum()),
        "total_violations": int((below | above).sum()),
        "material_violations": int(material.sum()),
        "maximum_lower_bound_shortfall": float(lower_shortfall.max(initial=0.0)),
        "maximum_upper_bound_excess": float(upper_excess.max(initial=0.0)),
        "maximum_relative_violation": float(relative_excess.max(initial=0.0)),
    }


def output_constraint_diagnostics(
    model: PhysicalPriceModel,
    split: SplitData,
    predictions: Mapping[str, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, Any]:
    """Describe how a declared output projection changed unconstrained prices."""
    if model.output_constraint == NO_OUTPUT_CONSTRAINT:
        unconstrained = predictions["price"]
    else:
        chunks: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, split.rows, batch_size):
                stop = min(start + batch_size, split.rows)
                inputs = torch.as_tensor(
                    split.features[start:stop],
                    dtype=torch.float64,
                )
                chunks.append(
                    model.unconstrained_price(inputs).detach().cpu().numpy()
                )
        unconstrained = np.concatenate(chunks).astype(np.float64, copy=False)
    if not bool(np.isfinite(unconstrained).all()):
        raise EvaluationError("model produced non-finite unconstrained prices")

    adjustment = predictions["price"] - unconstrained
    lower_clipped = adjustment > 0.0
    upper_clipped = adjustment < 0.0
    return {
        "name": model.output_constraint,
        "activation_definition": (
            "Rows whose projected float64 price differs from the unconstrained "
            "price; exact equality at a kink is not counted."
        ),
        "active_rows": int(np.count_nonzero(adjustment)),
        "lower_bound_activations": int(lower_clipped.sum()),
        "upper_bound_activations": int(upper_clipped.sum()),
        "maximum_absolute_adjustment": float(
            np.abs(adjustment).max(initial=0.0)
        ),
        "unconstrained_price_metrics": error_metrics(
            split.columns["price"],
            unconstrained,
        ),
        "unconstrained_price_no_arbitrage": no_arbitrage_violations(
            split,
            unconstrained,
        ),
    }


def evaluate_artifact(
    dataset: Path,
    artifact: Path,
    output: Path,
    *,
    partition: str,
) -> dict[str, Any]:
    """Evaluate one named partition and publish the report at ``output``.

    Reports are never overwritten. An existing ``output`` raises
    ``EvaluationError`` up front, before any dataset or artifact is read, so a
    mistaken re-run costs nothing; the same error is raised if the file appears
    while the evaluation is running, because publication itself is exclusive.
    The report is written atomically, so a failure at any point leaves either
    no file or the untouched previous one — never a partial report.
    """
    if partition not in EVALUATION_PARTITIONS:
        raise EvaluationError(
            f"partition must be one of {list(EVALUATION_PARTITIONS)}, got {partition!r}"
        )
    if output.exists():
        raise EvaluationError(f"output '{output}' already exists; refusing to overwrite")
    manifest, identity = load_dataset_manifest(dataset)
    model, artifact_payload = load_physical_model(artifact)
    recorded_dataset = artifact_payload.get("dataset")
    if not isinstance(recorded_dataset, dict):
        raise EvaluationError("artifact.dataset must be a JSON object")
    if recorded_dataset.get("manifest_sha256") != identity.manifest_sha256:
        raise EvaluationError(
            "dataset manifest SHA-256 does not match the dataset used for training"
        )
    evaluation_config = artifact_payload.get("evaluation")
    if not isinstance(evaluation_config, dict):
        raise EvaluationError("artifact.evaluation must be a JSON object")
    batch_size = evaluation_config.get("batch_size")
    core_max = evaluation_config.get("core_max_absolute_z")
    tail_max = evaluation_config.get("tail_max_absolute_z")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size <= 0
        or isinstance(core_max, bool)
        or not isinstance(core_max, int | float)
        or isinstance(tail_max, bool)
        or not isinstance(tail_max, int | float)
        or not 0.0 < core_max < tail_max
    ):
        raise EvaluationError("artifact evaluation configuration is malformed")

    split = load_split(
        dataset,
        manifest,
        partition,
        evaluation=True,
    )
    predictions = predict_with_greeks(model, split.features, batch_size=batch_size)
    all_rows = np.ones(split.rows, dtype=bool)
    option_masks = {
        "call": split.option_types == "call",
        "put": split.option_types == "put",
    }
    z_masks = moneyness_masks(split, core_max=core_max, tail_max=tail_max)
    artifact_manifest_bytes = (artifact / ARTIFACT_MANIFEST).read_bytes()
    output_constraint = artifact_payload.get(
        "output_constraint",
        {"name": NO_OUTPUT_CONSTRAINT},
    )
    report = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "partition": partition,
        "partition_role": PARTITION_ROLES[partition],
        "dataset": {
            "manifest_sha256": identity.manifest_sha256,
            "config_sha256": identity.config_sha256,
            "schema_version": identity.schema_version,
            "generator_version": identity.generator_version,
        },
        "artifact": {
            "manifest_sha256": hashlib.sha256(artifact_manifest_bytes).hexdigest(),
            "weights_sha256": artifact_payload["weights"]["sha256"],
            "training_config_sha256": artifact_payload["training_config"]["sha256"],
            "best_epoch": artifact_payload["training"]["best_epoch"],
            "output_constraint": output_constraint["name"],
            "source_artifact": output_constraint.get("source_artifact"),
        },
        "definitions": {
            "selection_policy": (
                "Validation reports may guide model selection. The interpolation test "
                "report is a locked final evaluation and must not guide tuning."
            ),
            "autograd": (
                "Derivatives of physical price after input and target transforms are "
                "inside the differentiable graph, including any declared output "
                "constraint."
            ),
            "output_constraint": (
                "The artifact block names the exact output contract used for both "
                "prices and autograd Greeks."
            ),
            "theta": "-d learned price / d maturity, per year",
            "vega": "d learned price / d absolute volatility, not per volatility point",
            "rho": "d learned price / d absolute continuously compounded rate",
            "relative_price_metric": (
                "No division by price is reported because near-zero option prices make it "
                "unstable; price_over_spot is the dimensionless alternative."
            ),
            "moneyness_bands": {
                "absolute_z": (
                    "abs(log_forward_moneyness) / (volatility * sqrt(maturity))"
                ),
                "core": f"absolute_z <= {float(core_max)}",
                "tail": f"{float(core_max)} < absolute_z <= {float(tail_max)}",
                "extreme": f"absolute_z > {float(tail_max)}",
            },
        },
        "slices": {
            "overall": slice_metrics(split, predictions, all_rows),
            "option_type": {
                name: slice_metrics(split, predictions, mask)
                for name, mask in option_masks.items()
            },
            "moneyness_band": {
                name: slice_metrics(split, predictions, mask)
                for name, mask in z_masks.items()
            },
        },
        "learned_price_no_arbitrage": no_arbitrage_violations(
            split,
            predictions["price"],
        ),
        "output_constraint_diagnostics": output_constraint_diagnostics(
            model,
            split,
            predictions,
            batch_size=batch_size,
        ),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": "cpu",
        },
        "limitations": artifact_payload["limitations"],
    }
    # Published atomically and exclusively: the report appears complete or not
    # at all, and an existing report is never overwritten or truncated even if
    # it was created after the early existence check above.
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_atomic(output, report)
    except FileExistsError as error:
        raise EvaluationError(
            f"output '{output}' already exists; refusing to overwrite"
        ) from error
    except (OSError, TypeError, ValueError) as error:
        raise EvaluationError(f"cannot write evaluation report '{output}': {error}") from error
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=EVALUATION_PARTITIONS,
        required=True,
        help="validation for model selection; interpolation_test for locked final evaluation",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        report = evaluate_artifact(
            arguments.dataset,
            arguments.artifact,
            arguments.output,
            partition=arguments.partition,
        )
    except (ArtifactError, DatasetLoadError, EvaluationError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "output": str(arguments.output),
                "partition": report["partition"],
                "partition_role": report["partition_role"],
                "rows": report["slices"]["overall"]["rows"],
                "price_rmse": report["slices"]["overall"]["metrics"]["price"]["rmse"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
