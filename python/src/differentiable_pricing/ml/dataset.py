"""Manifest-linked Parquet loading for neural experiments."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..data.diagnose import DiagnosticsError, load_manifest
from ..data.generate import LABEL_FIELDS, MANIFEST_NAME, TABLE_SCHEMA, sha256_file
from .config import FEATURE_ORDER

OPTION_TYPE_ENCODING: Final = {"put": -1.0, "call": 1.0}
EVALUATION_COLUMNS: Final = (
    *FEATURE_ORDER,
    "log_forward_moneyness",
    *LABEL_FIELDS,
)


class DatasetLoadError(RuntimeError):
    """Raised when a declared split cannot be loaded without guessing."""


@dataclass(frozen=True)
class DatasetIdentity:
    manifest_sha256: str
    config_sha256: str
    schema_version: str
    generator_version: str


@dataclass(frozen=True)
class SplitData:
    split: str
    features: np.ndarray
    option_types: np.ndarray
    columns: dict[str, np.ndarray]

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])


def load_dataset_manifest(dataset: Path) -> tuple[dict[str, Any], DatasetIdentity]:
    """Load the manifest without reading any Parquet partition."""
    try:
        manifest = load_manifest(dataset)
        manifest_bytes = (dataset / MANIFEST_NAME).read_bytes()
    except (DiagnosticsError, OSError) as error:
        raise DatasetLoadError(str(error)) from error
    config_block = manifest.get("config")
    config_sha256 = (
        config_block.get("sha256")
        if isinstance(config_block, dict)
        else None
    )
    generator_version = manifest.get("generator_version")
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise DatasetLoadError("manifest.config.sha256 is not a SHA-256 digest")
    if not isinstance(generator_version, str) or not generator_version:
        raise DatasetLoadError("manifest.generator_version must be a non-empty string")
    return manifest, DatasetIdentity(
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        config_sha256=config_sha256,
        schema_version=str(manifest["schema_version"]),
        generator_version=generator_version,
    )


def _entry_for_split(manifest: dict[str, Any], split: str) -> dict[str, Any]:
    matches = [entry for entry in manifest["files"] if entry["split"] == split]
    if len(matches) != 1:
        raise DatasetLoadError(
            f"manifest must declare exactly one file for split '{split}', found {len(matches)}"
        )
    return matches[0]


def _encode_option_type(values: np.ndarray, split: str) -> np.ndarray:
    encoded = np.empty(values.shape[0], dtype=np.float64)
    for name, code in OPTION_TYPE_ENCODING.items():
        encoded[values == name] = code
    valid = np.isin(values, tuple(OPTION_TYPE_ENCODING))
    if not bool(valid.all()):
        unknown = sorted({str(value) for value in values[~valid]})
        raise DatasetLoadError(f"split '{split}' has unknown option_type value(s): {unknown}")
    return encoded


def load_split(
    dataset: Path,
    manifest: dict[str, Any],
    split: str,
    *,
    evaluation: bool,
) -> SplitData:
    """Verify and read one declared split.

    Callers choose the split explicitly. In particular, training loads only
    ``train`` and ``validation``; it neither hashes nor opens the untouched
    interpolation test partition.
    """
    entry = _entry_for_split(manifest, split)
    path = dataset / entry["file"]
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise DatasetLoadError(f"cannot hash declared split '{path}': {error}") from error
    if digest != entry["sha256"]:
        raise DatasetLoadError(
            f"sha256 mismatch for split '{split}': manifest declares "
            f"{entry['sha256']}, file is {digest}"
        )

    columns = EVALUATION_COLUMNS if evaluation else (*FEATURE_ORDER, "price")
    try:
        schema = pq.read_schema(path)
        if not schema.equals(TABLE_SCHEMA, check_metadata=False):
            raise DatasetLoadError(
                f"split '{split}' schema does not match the generated-dataset contract"
            )
        table = pq.read_table(path, columns=list(columns))
    except (OSError, pa.ArrowInvalid) as error:
        raise DatasetLoadError(f"cannot read split '{path}': {error}") from error
    if table.num_rows != entry["rows"]:
        raise DatasetLoadError(
            f"split '{split}' has {table.num_rows} rows, manifest declares {entry['rows']}"
        )

    arrays: dict[str, np.ndarray] = {}
    for name in columns:
        values = table[name].to_numpy(zero_copy_only=False)
        if name == "option_type":
            arrays[name] = np.asarray(values, dtype=str)
        else:
            numeric = np.asarray(values, dtype=np.float64)
            if not bool(np.isfinite(numeric).all()):
                raise DatasetLoadError(f"split '{split}' column '{name}' contains non-finite data")
            arrays[name] = numeric

    option_types = arrays["option_type"]
    encoded = _encode_option_type(option_types, split)
    features = np.column_stack(
        [
            encoded if name == "option_type" else arrays[name]
            for name in FEATURE_ORDER
        ]
    ).astype(np.float64, copy=False)
    return SplitData(
        split=split,
        features=np.ascontiguousarray(features),
        option_types=option_types,
        columns=arrays,
    )
