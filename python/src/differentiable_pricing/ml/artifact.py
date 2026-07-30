"""Safe, versioned neural artifact persistence."""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..data.generate import sha256_file
from .config import (
    ARTIFACT_SCHEMA_VERSION,
    CONFIG_SCHEMA_VERSION,
    FEATURE_ORDER,
    MAX_HIDDEN_DIMENSION,
    MAX_MODEL_PARAMETERS,
    RAW_PHYSICAL_REPRESENTATION,
    REPRESENTATION_FEATURE_ORDER,
)
from .model import (
    EUROPEAN_BOUNDS_CONSTRAINT,
    EUROPEAN_BOUNDS_DIFFERENTIABILITY,
    EUROPEAN_BOUNDS_PROJECTION,
    NO_OUTPUT_CONSTRAINT,
    PhysicalPriceModel,
    PricingMlp,
    Scaling,
)

ARTIFACT_MANIFEST: Final = "artifact.json"
WEIGHTS_FILE: Final = "weights.npz"
CHECKPOINT_WARNING: Final = (
    "This artifact uses non-pickled NPZ weights with allow_pickle=False. If a future "
    "workflow emits a PyTorch .pt checkpoint, load it only from a trusted source; "
    "checkpoint deserialization is outside this artifact contract."
)


class ArtifactError(RuntimeError):
    """Raised when a model artifact is malformed or fails integrity checks."""


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ArtifactError(f"{where} is not a lowercase hexadecimal SHA-256 digest")
    return value


def _canonical_json(payload: Mapping[str, Any]) -> str:
    """Serialize canonical, newline-terminated JSON, rejecting NaN and infinities."""
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write canonical, newline-terminated JSON.

    This is the non-atomic writer, used only for files published inside a
    temporary directory that is itself renamed into place (see ``train.py`` and
    ``constrain.py``). To write a standalone file directly at its final path,
    use :func:`write_json_atomic`.
    """
    path.write_text(_canonical_json(payload), encoding="utf-8")


def write_json_atomic(
    path: Path,
    payload: Mapping[str, Any],
    *,
    overwrite: bool = False,
) -> None:
    """Publish canonical JSON at ``path`` atomically, or not at all.

    Serialization happens before any filesystem state changes, so an
    unserializable payload (NaN, infinity, non-JSON type) raises without
    touching the target. The bytes are then written to a temporary file in the
    same directory, flushed and fsynced, and only afterwards linked into place.
    An observer therefore sees either no file or a complete, valid one; a crash
    or exception can never leave a partial report at ``path``.

    ``overwrite=False`` (the default) publishes with :func:`os.link`, which
    fails atomically with ``FileExistsError`` when ``path`` already exists.
    That closes the check-then-write race an up-front ``path.exists()`` test
    leaves open, and guarantees an existing report is never truncated or
    replaced. ``overwrite=True`` uses :func:`os.replace`, which atomically
    swaps the contents; the previous file survives intact until the instant it
    is replaced.

    The temporary file is removed on every failure path — including
    ``KeyboardInterrupt`` and ``SystemExit``, which is why the cleanup catches
    ``BaseException`` — so a retry after a failed write succeeds and no partial
    temporaries accumulate.

    The exclusive path needs ``path``'s directory to support hard links. Every
    supported environment (ext4, APFS, NTFS) does; a filesystem that does not
    (some network and FAT mounts) raises ``OSError`` here, which callers report
    as a failed write. That degrades to a clear error, never to a partial or
    corrupted report.
    """
    serialized = _canonical_json(payload)
    directory = path.parent
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}-",
        suffix=".tmp",
        dir=directory,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        if overwrite:
            os.replace(temporary, path)
        else:
            # Atomic exclusive publish: fails if `path` appeared since any
            # earlier existence check, leaving the existing file untouched.
            os.link(temporary, path)
            temporary.unlink()
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def save_weights(path: Path, network: PricingMlp) -> str:
    """Persist deterministic state arrays without pickle and return SHA-256."""
    arrays = {
        key: tensor.detach().cpu().numpy()
        for key, tensor in sorted(network.state_dict().items())
    }
    with zipfile.ZipFile(path, mode="w") as archive:
        for key, array in arrays.items():
            buffer = io.BytesIO()
            np.lib.format.write_array(buffer, array, allow_pickle=False)
            member = zipfile.ZipInfo(
                filename=f"{key}.npy",
                date_time=(1980, 1, 1, 0, 0, 0),
            )
            member.compress_type = zipfile.ZIP_DEFLATED
            member.external_attr = 0o600 << 16
            archive.writestr(member, buffer.getvalue())
    return sha256_file(path)


def _require(
    table: Mapping[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    where: str,
) -> Any:
    if key not in table:
        raise ArtifactError(f"{where} is missing required key '{key}'")
    value = table[key]
    if isinstance(value, bool) and kind is not bool:
        raise ArtifactError(f"{where}.{key} has unexpected type bool")
    if not isinstance(value, kind):
        raise ArtifactError(f"{where}.{key} has unexpected type {type(value).__name__}")
    return value


def load_artifact_manifest(directory: Path) -> dict[str, Any]:
    path = directory / ARTIFACT_MANIFEST
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactError(f"cannot load artifact manifest '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise ArtifactError("artifact manifest must be a JSON object")
    if _require(payload, "schema_version", str, "artifact") != ARTIFACT_SCHEMA_VERSION:
        raise ArtifactError(
            f"artifact schema must be '{ARTIFACT_SCHEMA_VERSION}', "
            f"got {payload.get('schema_version')!r}"
        )
    if _require(payload, "feature_order", list, "artifact") != list(FEATURE_ORDER):
        raise ArtifactError("artifact feature_order does not match the stage-1 contract")
    _artifact_representation(payload)
    _artifact_output_constraint(payload)
    if _require(payload, "dtype", str, "artifact") != "float64":
        raise ArtifactError("artifact dtype must be float64")
    if _require(payload, "checkpoint_security", str, "artifact") != CHECKPOINT_WARNING:
        raise ArtifactError("artifact checkpoint_security warning is missing or changed")
    dataset = _require(payload, "dataset", dict, "artifact")
    _digest(dataset.get("manifest_sha256"), "artifact.dataset.manifest_sha256")
    _digest(dataset.get("config_sha256"), "artifact.dataset.config_sha256")
    training_config = _require(payload, "training_config", dict, "artifact")
    if training_config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise ArtifactError("artifact training configuration schema is unsupported")
    _digest(training_config.get("sha256"), "artifact.training_config.sha256")
    training = _require(payload, "training", dict, "artifact")
    _require(training, "best_epoch", int, "artifact.training")
    _require(payload, "evaluation", dict, "artifact")
    limitations = _require(payload, "limitations", list, "artifact")
    if not all(isinstance(item, str) and item for item in limitations):
        raise ArtifactError("artifact limitations must be non-empty strings")
    return payload


def _artifact_representation(payload: Mapping[str, Any]) -> str:
    block = payload.get("representation")
    if block is None:
        return RAW_PHYSICAL_REPRESENTATION
    if not isinstance(block, dict):
        raise ArtifactError("artifact.representation must be a JSON object")
    name = _require(block, "name", str, "artifact.representation")
    if name not in REPRESENTATION_FEATURE_ORDER:
        raise ArtifactError(f"artifact representation {name!r} is unsupported")
    network_features = _require(
        block,
        "network_feature_order",
        list,
        "artifact.representation",
    )
    if network_features != list(REPRESENTATION_FEATURE_ORDER[name]):
        raise ArtifactError(
            "artifact representation network_feature_order does not match its name"
        )
    _require(
        block,
        "physical_reconstruction",
        str,
        "artifact.representation",
    )
    return name


def _validated_scaling(
    payload: Mapping[str, Any],
    feature_dimension: int,
) -> Scaling:
    transform = _require(payload, "transform", dict, "artifact")
    mean = np.asarray(
        _require(transform, "feature_mean", list, "artifact.transform"),
        dtype=np.float64,
    )
    scale = np.asarray(
        _require(transform, "feature_scale", list, "artifact.transform"),
        dtype=np.float64,
    )
    if mean.shape != (feature_dimension,) or scale.shape != (feature_dimension,):
        raise ArtifactError("artifact transform arrays do not match network features")
    if not bool(np.isfinite(mean).all()) or not bool(np.isfinite(scale).all()):
        raise ArtifactError("artifact feature transform contains non-finite values")
    if bool((scale <= 0.0).any()):
        raise ArtifactError("artifact feature_scale must be strictly positive")
    price_mean = _require(transform, "price_mean", int | float, "artifact.transform")
    price_scale = _require(transform, "price_scale", int | float, "artifact.transform")
    if not np.isfinite(price_mean) or not np.isfinite(price_scale) or price_scale <= 0.0:
        raise ArtifactError("artifact price transform must be finite with positive scale")
    return Scaling(mean, scale, float(price_mean), float(price_scale))


def _artifact_output_constraint(payload: Mapping[str, Any]) -> str:
    block = payload.get("output_constraint")
    if block is None:
        return NO_OUTPUT_CONSTRAINT
    if not isinstance(block, dict):
        raise ArtifactError("artifact.output_constraint must be a JSON object")
    name = _require(block, "name", str, "artifact.output_constraint")
    if name != EUROPEAN_BOUNDS_CONSTRAINT:
        raise ArtifactError(f"artifact output constraint {name!r} is unsupported")
    if (
        _require(block, "projection", str, "artifact.output_constraint")
        != EUROPEAN_BOUNDS_PROJECTION
    ):
        raise ArtifactError("artifact output constraint projection is unsupported")
    if (
        _require(
            block,
            "differentiability",
            str,
            "artifact.output_constraint",
        )
        != EUROPEAN_BOUNDS_DIFFERENTIABILITY
    ):
        raise ArtifactError("artifact output constraint differentiability is unsupported")
    bounds = _require(block, "bounds", str, "artifact.output_constraint")
    if not bounds:
        raise ArtifactError("artifact output constraint bounds statement is empty")
    source = _require(
        block,
        "source_artifact",
        dict,
        "artifact.output_constraint",
    )
    _digest(
        source.get("manifest_sha256"),
        "artifact.output_constraint.source_artifact.manifest_sha256",
    )
    source_weights_sha256 = _digest(
        source.get("weights_sha256"),
        "artifact.output_constraint.source_artifact.weights_sha256",
    )
    weights = _require(payload, "weights", dict, "artifact")
    artifact_weights_sha256 = _digest(
        weights.get("sha256"),
        "artifact.weights.sha256",
    )
    if source_weights_sha256 != artifact_weights_sha256:
        raise ArtifactError(
            "constrained artifact weights do not match its declared source artifact"
        )
    return name


def load_physical_model(directory: Path) -> tuple[PhysicalPriceModel, dict[str, Any]]:
    """Verify an artifact and reconstruct its physical-unit model."""
    payload = load_artifact_manifest(directory)
    representation = _artifact_representation(payload)
    output_constraint = _artifact_output_constraint(payload)
    input_dimension = len(REPRESENTATION_FEATURE_ORDER[representation])
    architecture = _require(payload, "architecture", dict, "artifact")
    if _require(architecture, "activation", str, "artifact.architecture") != "tanh":
        raise ArtifactError("artifact activation must be tanh")
    if _require(architecture, "output_dimension", int, "artifact.architecture") != 1:
        raise ArtifactError("artifact output_dimension must be one")
    hidden = _require(
        architecture,
        "hidden_dimensions",
        list,
        "artifact.architecture",
    )
    if (
        not hidden
        or any(
            isinstance(item, bool)
            or not isinstance(item, int)
            or not 0 < item <= MAX_HIDDEN_DIMENSION
            for item in hidden
        )
    ):
        raise ArtifactError("artifact hidden_dimensions are outside the supported range")
    declared_input_dimension = architecture.get("input_dimension", input_dimension)
    if (
        isinstance(declared_input_dimension, bool)
        or not isinstance(declared_input_dimension, int)
        or declared_input_dimension != input_dimension
    ):
        raise ArtifactError(
            "artifact architecture input_dimension does not match representation"
        )
    dimensions = (input_dimension, *hidden, 1)
    parameters = sum(
        input_size * output_size + output_size
        for input_size, output_size in pairwise(dimensions)
    )
    if parameters > MAX_MODEL_PARAMETERS:
        raise ArtifactError("artifact model exceeds the supported parameter limit")

    weights = _require(payload, "weights", dict, "artifact")
    name = _require(weights, "file", str, "artifact.weights")
    if name != WEIGHTS_FILE or Path(name).name != name:
        raise ArtifactError(f"artifact weights file must be '{WEIGHTS_FILE}'")
    expected_digest = _digest(weights.get("sha256"), "artifact.weights.sha256")
    path = directory / name
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise ArtifactError(f"cannot hash artifact weights '{path}': {error}") from error
    if digest != expected_digest:
        raise ArtifactError(
            "artifact weights sha256 mismatch: manifest declares "
            f"{expected_digest}, file is {digest}"
        )

    network = PricingMlp(input_dimension, tuple(hidden))
    expected = network.state_dict()
    try:
        with zipfile.ZipFile(path) as zip_archive:
            information = zip_archive.infolist()
            expected_names = {f"{key}.npy" for key in expected}
            if (
                len(information) != len(expected_names)
                or {entry.filename for entry in information} != expected_names
            ):
                raise ArtifactError("weights archive contains unexpected members")
            expected_bytes = sum(tensor.numel() * 8 for tensor in expected.values())
            maximum_uncompressed = expected_bytes + len(expected) * 4096
            if sum(entry.file_size for entry in information) > maximum_uncompressed:
                raise ArtifactError("weights archive exceeds its expected uncompressed size")
            for entry in information:
                key = entry.filename.removesuffix(".npy")
                reference = expected[key]
                with zip_archive.open(entry) as stream:
                    version = np.lib.format.read_magic(stream)
                    if version == (1, 0):
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_1_0(stream)
                        )
                    elif version == (2, 0):
                        shape, fortran_order, dtype = (
                            np.lib.format.read_array_header_2_0(stream)
                        )
                    else:
                        raise ArtifactError(
                            f"weight '{key}' uses unsupported NPY version {version}"
                        )
                    expected_payload_bytes = reference.numel() * 8
                    if (
                        dtype != np.dtype(np.float64)
                        or shape != tuple(reference.shape)
                        or fortran_order
                        or entry.file_size - stream.tell() != expected_payload_bytes
                    ):
                        raise ArtifactError(
                            f"weight '{key}' has a malformed NPY header or payload size"
                        )
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(expected):
                raise ArtifactError(
                    f"weights hold keys {sorted(archive.files)}, expected {sorted(expected)}"
                )
            state: dict[str, torch.Tensor] = {}
            for key, reference in expected.items():
                array = np.asarray(archive[key])
                if array.dtype != np.float64:
                    raise ArtifactError(f"weight '{key}' is {array.dtype}, expected float64")
                if array.shape != tuple(reference.shape):
                    raise ArtifactError(
                        f"weight '{key}' has shape {array.shape}, expected {tuple(reference.shape)}"
                    )
                if not bool(np.isfinite(array).all()):
                    raise ArtifactError(f"weight '{key}' contains non-finite values")
                state[key] = torch.from_numpy(array.copy())
    except (EOFError, OSError, ValueError, zipfile.BadZipFile) as error:
        raise ArtifactError(f"cannot load artifact weights '{path}': {error}") from error
    network.load_state_dict(state, strict=True)
    network.eval()
    return (
        PhysicalPriceModel(
            network,
            _validated_scaling(payload, input_dimension),
            representation,
            output_constraint,
        ).eval(),
        payload,
    )
