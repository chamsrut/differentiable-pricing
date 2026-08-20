"""Strict, versioned artifact persistence for the Task 9G American pilot."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from .american import (
    AMERICAN_FEATURE_ORDER,
    AMERICAN_REPRESENTATION,
    PHYSICAL_RECONSTRUCTION,
    AmericanPriceModel,
)
from .artifact import (
    ARTIFACT_MANIFEST,
    CHECKPOINT_WARNING,
    WEIGHTS_FILE,
    save_weights,
    write_json,
)
from .model import PricingMlp, Scaling

AMERICAN_ARTIFACT_SCHEMA: Final = "american-neural-artifact/1"
_TOP_KEYS: Final = frozenset(
    {
        "schema_version",
        "experiment_name",
        "arm",
        "dtype",
        "device",
        "feature_order",
        "representation",
        "architecture",
        "transform",
        "output_constraint",
        "weights",
        "dataset",
        "protocol",
        "code",
        "training",
        "source_lineage",
        "checkpoint_security",
        "limitations",
    }
)


class AmericanArtifactError(RuntimeError):
    """Raised when an American pilot artifact is incomplete or inconsistent."""


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise AmericanArtifactError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise AmericanArtifactError(f"{where} keys differ: missing={missing}, unknown={unknown}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scaling(payload: Mapping[str, Any], *, expected_fit_partition: str) -> Scaling:
    transform = payload.get("transform")
    if not isinstance(transform, Mapping):
        raise AmericanArtifactError("artifact.transform must be an object")
    _exact_keys(
        transform,
        frozenset(
            {
                "fit_partition",
                "feature_mean",
                "feature_scale",
                "target_mean",
                "target_scale",
                "population_standard_deviation_ddof",
            }
        ),
        "artifact.transform",
    )
    if (
        expected_fit_partition != "selected train rows only"
        or transform["fit_partition"] != expected_fit_partition
        or transform["population_standard_deviation_ddof"] != 0
    ):
        raise AmericanArtifactError(
            "artifact scaling must be fitted on selected train rows only with ddof=0"
        )
    mean = np.asarray(transform["feature_mean"], dtype=np.float64)
    scale = np.asarray(transform["feature_scale"], dtype=np.float64)
    target_mean = transform["target_mean"]
    target_scale = transform["target_scale"]
    if (
        mean.shape != (5,)
        or scale.shape != (5,)
        or not bool(np.isfinite(mean).all())
        or not bool(np.isfinite(scale).all())
        or bool((scale <= 0.0).any())
        or isinstance(target_mean, bool)
        or isinstance(target_scale, bool)
        or not isinstance(target_mean, int | float)
        or not isinstance(target_scale, int | float)
        or not math.isfinite(float(target_mean))
        or not math.isfinite(float(target_scale))
        or float(target_scale) <= 0.0
    ):
        raise AmericanArtifactError("artifact scaling is malformed")
    return Scaling(mean, scale, float(target_mean), float(target_scale))


def artifact_payload(
    model: AmericanPriceModel,
    *,
    experiment_name: str,
    arm: str,
    weights_sha256: str,
    dataset: Mapping[str, Any],
    protocol: Mapping[str, Any],
    code: Mapping[str, Any],
    training: Mapping[str, Any],
    source_lineage: Mapping[str, Any] | None,
) -> dict[str, Any]:
    linears = [layer for layer in model.network.layers if isinstance(layer, torch.nn.Linear)]
    if [layer.in_features for layer in linears] != [5, 64, 64, 64] or [
        layer.out_features for layer in linears
    ] != [64, 64, 64, 1]:
        raise AmericanArtifactError("artifact model must use 5->[64,64,64]->1")
    return {
        "schema_version": AMERICAN_ARTIFACT_SCHEMA,
        "experiment_name": experiment_name,
        "arm": arm,
        "dtype": "float64",
        "device": "cpu",
        "feature_order": list(AMERICAN_FEATURE_ORDER),
        "representation": {
            "name": AMERICAN_REPRESENTATION,
            "physical_reconstruction": PHYSICAL_RECONSTRUCTION,
            "target": "u = price / (spot * exp(-dividend_yield * maturity))",
        },
        "architecture": {
            "input_dimension": 5,
            "hidden_dimensions": [64, 64, 64],
            "activation": "tanh",
            "output_dimension": 1,
        },
        "transform": {
            "fit_partition": "selected train rows only",
            "feature_mean": model.feature_mean.detach().cpu().tolist(),
            "feature_scale": model.feature_scale.detach().cpu().tolist(),
            "target_mean": float(model.price_mean),
            "target_scale": float(model.price_scale),
            "population_standard_deviation_ddof": 0,
        },
        "output_constraint": {"name": "none_v1", "projection": None},
        "weights": {
            "file": WEIGHTS_FILE,
            "format": "deterministic NumPy NPZ; float64; allow_pickle=False",
            "sha256": weights_sha256,
        },
        "dataset": dict(dataset),
        "protocol": dict(protocol),
        "code": dict(code),
        "training": dict(training),
        "source_lineage": None if source_lineage is None else dict(source_lineage),
        "checkpoint_security": CHECKPOINT_WARNING,
        "limitations": [
            "One-seed, one-budget synthetic continuous-yield CRR feasibility pilot only.",
            "No American Greek, OOD, discrete-dividend, market, or production claim.",
        ],
    }


def save_american_artifact(
    directory: Path,
    model: AmericanPriceModel,
    **metadata: Any,
) -> dict[str, Any]:
    """Atomically publish deterministic weights and their strict manifest."""
    directory = Path(directory)
    if directory.exists():
        raise AmericanArtifactError(f"artifact output already exists: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent))
    try:
        weights_sha256 = save_weights(temporary / WEIGHTS_FILE, model.network)
        payload = artifact_payload(model, weights_sha256=weights_sha256, **metadata)
        write_json(temporary / ARTIFACT_MANIFEST, payload)
        temporary.rename(directory)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return payload


def load_american_artifact(
    directory: Path,
    *,
    expected_row_budget: int,
    expected_fit_partition: str,
) -> tuple[AmericanPriceModel, dict[str, Any]]:
    """Fail closed on schema, identity, scaling, architecture, and NPZ state."""
    if (
        isinstance(expected_row_budget, bool)
        or not isinstance(expected_row_budget, int)
        or expected_row_budget <= 0
    ):
        raise AmericanArtifactError("expected row budget must be a positive integer")
    directory = Path(directory)
    try:
        payload = json.loads((directory / ARTIFACT_MANIFEST).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AmericanArtifactError(f"cannot load American artifact: {error}") from error
    if not isinstance(payload, dict):
        raise AmericanArtifactError("artifact manifest must be an object")
    _exact_keys(payload, _TOP_KEYS, "artifact")
    if payload["schema_version"] != AMERICAN_ARTIFACT_SCHEMA:
        raise AmericanArtifactError("unsupported American artifact schema")
    if payload["arm"] not in {"scratch", "transfer"}:
        raise AmericanArtifactError("artifact.arm must be scratch or transfer")
    if payload["dtype"] != "float64" or payload["device"] != "cpu":
        raise AmericanArtifactError("artifact must be float64 CPU")
    if payload["feature_order"] != list(AMERICAN_FEATURE_ORDER):
        raise AmericanArtifactError("artifact feature order differs from the contract")
    representation = payload["representation"]
    if not isinstance(representation, Mapping) or representation != {
        "name": AMERICAN_REPRESENTATION,
        "physical_reconstruction": PHYSICAL_RECONSTRUCTION,
        "target": "u = price / (spot * exp(-dividend_yield * maturity))",
    }:
        raise AmericanArtifactError("artifact representation differs from the contract")
    if payload["architecture"] != {
        "input_dimension": 5,
        "hidden_dimensions": [64, 64, 64],
        "activation": "tanh",
        "output_dimension": 1,
    }:
        raise AmericanArtifactError("artifact architecture differs from the contract")
    if payload["output_constraint"] != {"name": "none_v1", "projection": None}:
        raise AmericanArtifactError("American artifacts must not apply output projection")
    weights = payload["weights"]
    if not isinstance(weights, Mapping):
        raise AmericanArtifactError("artifact.weights must be an object")
    _exact_keys(weights, frozenset({"file", "format", "sha256"}), "artifact.weights")
    expected_digest = _digest(weights.get("sha256"), "artifact.weights.sha256")
    if weights.get("file") != WEIGHTS_FILE or weights.get("format") != (
        "deterministic NumPy NPZ; float64; allow_pickle=False"
    ):
        raise AmericanArtifactError("artifact weights format is unsupported")
    path = directory / WEIGHTS_FILE
    try:
        actual_digest = _sha256(path)
    except OSError as error:
        raise AmericanArtifactError(f"cannot hash artifact weights: {error}") from error
    if actual_digest != expected_digest:
        raise AmericanArtifactError("artifact weights SHA-256 mismatch")
    network = PricingMlp(5, (64, 64, 64))
    expected = network.state_dict()
    try:
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise AmericanArtifactError("artifact NPZ members differ from architecture")
            state = {}
            for key, reference in expected.items():
                value = np.asarray(archive[key])
                if (
                    value.dtype != np.float64
                    or value.shape != tuple(reference.shape)
                    or not bool(np.isfinite(value).all())
                ):
                    raise AmericanArtifactError(f"artifact weight {key!r} is malformed")
                state[key] = torch.from_numpy(value.copy())
    except (OSError, ValueError) as error:
        raise AmericanArtifactError(f"cannot load artifact weights: {error}") from error
    network.load_state_dict(state, strict=True)
    network.eval()
    if payload["arm"] == "transfer":
        lineage = payload["source_lineage"]
        if not isinstance(lineage, Mapping):
            raise AmericanArtifactError("transfer artifact requires source_lineage")
        _exact_keys(
            lineage,
            frozenset({"manifest_sha256", "weights_sha256", "lift"}),
            "artifact.source_lineage",
        )
        _digest(lineage.get("manifest_sha256"), "artifact.source_lineage.manifest_sha256")
        _digest(lineage.get("weights_sha256"), "artifact.source_lineage.weights_sha256")
        if lineage.get("lift") != "task-9g-exact-european-to-american-v1":
            raise AmericanArtifactError("artifact source lift identity differs")
    elif payload["source_lineage"] is not None:
        raise AmericanArtifactError("scratch artifact source_lineage must be null")
    dataset = payload["dataset"]
    if not isinstance(dataset, Mapping):
        raise AmericanArtifactError("artifact.dataset must be an object")
    _exact_keys(
        dataset,
        frozenset({"manifest_sha256", "train_sha256", "validation_sha256", "selected_train_rows"}),
        "artifact.dataset",
    )
    for key in ("manifest_sha256", "train_sha256", "validation_sha256"):
        _digest(dataset[key], f"artifact.dataset.{key}")
    if dataset["selected_train_rows"] != expected_row_budget:
        raise AmericanArtifactError("artifact selected train-row budget differs")
    protocol = payload["protocol"]
    if not isinstance(protocol, Mapping):
        raise AmericanArtifactError("artifact.protocol must be an object")
    _exact_keys(protocol, frozenset({"sha256", "schema_version"}), "artifact.protocol")
    _digest(protocol["sha256"], "artifact.protocol.sha256")
    if protocol["schema_version"] != "american-neural-pilot-protocol/1":
        raise AmericanArtifactError("artifact protocol schema differs")
    code = payload["code"]
    if not isinstance(code, Mapping):
        raise AmericanArtifactError("artifact.code must be an object")
    _exact_keys(code, frozenset({"tracked_inputs"}), "artifact.code")
    tracked = code["tracked_inputs"]
    if not isinstance(tracked, list) or not tracked:
        raise AmericanArtifactError("artifact code identity must contain tracked inputs")
    for index, entry in enumerate(tracked):
        if not isinstance(entry, Mapping):
            raise AmericanArtifactError(f"artifact.code.tracked_inputs[{index}] must be an object")
        _exact_keys(
            entry,
            frozenset({"path", "sha256"}),
            f"artifact.code.tracked_inputs[{index}]",
        )
        if not isinstance(entry["path"], str) or not entry["path"]:
            raise AmericanArtifactError("artifact tracked-input path is malformed")
        _digest(entry["sha256"], f"artifact.code.tracked_inputs[{index}].sha256")
    training = payload["training"]
    if not isinstance(training, Mapping):
        raise AmericanArtifactError("artifact.training must be an object")
    _exact_keys(
        training,
        frozenset({"seed", "best_epoch", "best_validation_mse", "history"}),
        "artifact.training",
    )
    if (
        isinstance(training["seed"], bool)
        or not isinstance(training["seed"], int)
        or isinstance(training["best_epoch"], bool)
        or not isinstance(training["best_epoch"], int)
        or training["best_epoch"] <= 0
        or isinstance(training["best_validation_mse"], bool)
        or not isinstance(training["best_validation_mse"], int | float)
        or not math.isfinite(float(training["best_validation_mse"]))
        or float(training["best_validation_mse"]) < 0.0
        or not isinstance(training["history"], list)
        or not training["history"]
    ):
        raise AmericanArtifactError("artifact training record is malformed")
    for index, record in enumerate(training["history"]):
        if not isinstance(record, Mapping):
            raise AmericanArtifactError(f"artifact.training.history[{index}] must be an object")
        _exact_keys(
            record,
            frozenset({"epoch", "validation_standardized_target_mse", "learning_rate"}),
            f"artifact.training.history[{index}]",
        )
        if record["epoch"] != index + 1 or not all(
            isinstance(record[key], int | float)
            and not isinstance(record[key], bool)
            and math.isfinite(float(record[key]))
            for key in ("validation_standardized_target_mse", "learning_rate")
        ):
            raise AmericanArtifactError("artifact training history is malformed")
    if training["best_epoch"] > len(training["history"]):
        raise AmericanArtifactError("artifact best epoch is outside its history")
    limitations = payload["limitations"]
    if not isinstance(limitations, list) or not all(
        isinstance(item, str) and item for item in limitations
    ):
        raise AmericanArtifactError("artifact limitations must be non-empty strings")
    if payload["checkpoint_security"] != CHECKPOINT_WARNING:
        raise AmericanArtifactError("artifact checkpoint warning differs")
    return (
        AmericanPriceModel(
            network,
            _scaling(payload, expected_fit_partition=expected_fit_partition),
        ).eval(),
        payload,
    )
