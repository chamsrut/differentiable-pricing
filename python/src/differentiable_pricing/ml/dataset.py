"""Manifest-linked Parquet loading for neural experiments.

Datasets are loaded under an **explicitly named, versioned schema**. A file is
read because its manifest declares a schema this module has registered, never
because the bytes happen to parse. An unregistered ``schema_version`` is an
error, and there is no permissive compatibility fallback: the alternative to a
registered schema is a refusal, not a guess.

Two schemas are registered today — ``european-option-dataset/1`` (task 8's
European surrogate, unchanged) and ``american-option-dataset/1`` (the
continuous-yield American CRR tables admitted by task 9E).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ..data import american_schema
from ..data.american_schema import AmericanManifestError, load_american_manifest
from ..data.config import SCHEMA_VERSION as EUROPEAN_SCHEMA_VERSION
from ..data.config import SPLIT_NAMES as EUROPEAN_SPLIT_NAMES
from ..data.diagnose import DiagnosticsError, load_manifest
from ..data.generate import LABEL_FIELDS, MANIFEST_NAME, TABLE_SCHEMA, sha256_file
from .config import FEATURE_ORDER

OPTION_TYPE_ENCODING: Final = {"put": -1.0, "call": 1.0}
EVALUATION_COLUMNS: Final = (
    *FEATURE_ORDER,
    "log_forward_moneyness",
    *LABEL_FIELDS,
)
AMERICAN_EVALUATION_COLUMNS: Final = (
    *american_schema.FEATURE_ORDER,
    american_schema.TARGET_COLUMN,
    american_schema.EUROPEAN_COMPARATOR_COLUMN,
    "early_exercise_premium",
    "adjacent_step_gap",
    "intrinsic_value",
    "european_black_scholes_price",
)


class DatasetLoadError(RuntimeError):
    """Raised when a declared split cannot be loaded without guessing."""


@dataclass(frozen=True)
class DatasetSchema:
    """One registered dataset schema and everything a loader needs for it."""

    schema_version: str
    table_schema: pa.Schema
    split_names: tuple[str, ...]
    feature_order: tuple[str, ...]
    target_column: str
    evaluation_columns: tuple[str, ...]
    manifest_loader: Callable[[Path], dict[str, Any]]
    manifest_errors: tuple[type[Exception], ...]
    comparator_column: str | None = None

    def training_columns(self) -> tuple[str, ...]:
        """Columns read when labels beyond the target are not needed."""
        if self.comparator_column is None:
            return (*self.feature_order, self.target_column)
        return (*self.feature_order, self.target_column, self.comparator_column)


_EUROPEAN_SCHEMA: Final = DatasetSchema(
    schema_version=EUROPEAN_SCHEMA_VERSION,
    table_schema=TABLE_SCHEMA,
    split_names=tuple(EUROPEAN_SPLIT_NAMES),
    feature_order=tuple(FEATURE_ORDER),
    target_column="price",
    evaluation_columns=EVALUATION_COLUMNS,
    manifest_loader=load_manifest,
    manifest_errors=(DiagnosticsError, OSError),
    comparator_column=None,
)

_AMERICAN_SCHEMA: Final = DatasetSchema(
    schema_version=american_schema.SCHEMA_VERSION,
    table_schema=american_schema.TABLE_SCHEMA,
    split_names=tuple(american_schema.SPLIT_NAMES),
    feature_order=tuple(american_schema.FEATURE_ORDER),
    target_column=american_schema.TARGET_COLUMN,
    evaluation_columns=AMERICAN_EVALUATION_COLUMNS,
    manifest_loader=load_american_manifest,
    manifest_errors=(AmericanManifestError, OSError),
    comparator_column=american_schema.EUROPEAN_COMPARATOR_COLUMN,
)

SCHEMA_REGISTRY: Final[dict[str, DatasetSchema]] = {
    _EUROPEAN_SCHEMA.schema_version: _EUROPEAN_SCHEMA,
    _AMERICAN_SCHEMA.schema_version: _AMERICAN_SCHEMA,
}
"""Every schema this repository will load. Anything else is refused."""


def registered_schema(schema_version: str) -> DatasetSchema:
    """Resolve a declared ``schema_version``, or refuse to guess."""
    try:
        return SCHEMA_REGISTRY[schema_version]
    except KeyError:
        known = ", ".join(sorted(SCHEMA_REGISTRY))
        raise DatasetLoadError(
            f"manifest declares unregistered schema_version '{schema_version}'; "
            f"this repository loads only: {known}"
        ) from None


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
    schema_version: str = EUROPEAN_SCHEMA_VERSION
    target_column: str = "price"
    comparator_column: str | None = None

    @property
    def rows(self) -> int:
        return int(self.features.shape[0])

    @property
    def target(self) -> np.ndarray:
        """The supervised target for this split's schema.

        Named rather than positional so a caller never has to know whether the
        label column is ``price`` or ``american_price``.
        """
        try:
            return self.columns[self.target_column]
        except KeyError:
            raise DatasetLoadError(
                f"split '{self.split}' was loaded without its target column "
                f"'{self.target_column}'"
            ) from None

    @property
    def european_comparator(self) -> np.ndarray | None:
        """The paired European price on the same lattice, where one exists.

        ``None`` for a European dataset, whose own label is already European.
        For ``american-option-dataset/1`` this is ``european_crr_price``: the
        comparator the European-to-American transfer question is asked against,
        and the leg that makes ``early_exercise_premium`` exact.
        """
        if self.comparator_column is None:
            return None
        return self.columns.get(self.comparator_column)


def _peek_schema_version(dataset: Path) -> str:
    path = Path(dataset) / MANIFEST_NAME
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise DatasetLoadError(f"cannot read manifest '{path}': {error}") from error
    except json.JSONDecodeError as error:
        raise DatasetLoadError(
            f"manifest '{path}' is not valid JSON: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise DatasetLoadError(f"manifest '{path}' must be a JSON object")
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str) or not schema_version:
        raise DatasetLoadError(
            f"manifest '{path}' does not declare a string 'schema_version'"
        )
    return schema_version


def load_dataset_manifest(dataset: Path) -> tuple[dict[str, Any], DatasetIdentity]:
    """Load the manifest without reading any Parquet partition.

    The declared ``schema_version`` selects the validator. An unregistered
    schema fails here, before any file is opened.
    """
    dataset = Path(dataset)
    schema = registered_schema(_peek_schema_version(dataset))
    try:
        manifest = schema.manifest_loader(dataset)
        manifest_bytes = (dataset / MANIFEST_NAME).read_bytes()
    except schema.manifest_errors as error:
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
    """Verify and read one declared split, under the manifest's named schema.

    Callers choose the split explicitly. In particular, training loads only
    ``train`` and ``validation``; it neither hashes nor opens the untouched
    interpolation test partition.
    """
    schema = registered_schema(str(manifest.get("schema_version", "")))
    if split not in schema.split_names:
        raise DatasetLoadError(
            f"'{split}' is not a partition of schema '{schema.schema_version}'; "
            f"expected one of {list(schema.split_names)}"
        )
    entry = _entry_for_split(manifest, split)
    path = Path(dataset) / entry["file"]
    try:
        digest = sha256_file(path)
    except OSError as error:
        raise DatasetLoadError(f"cannot hash declared split '{path}': {error}") from error
    if digest != entry["sha256"]:
        raise DatasetLoadError(
            f"sha256 mismatch for split '{split}': manifest declares "
            f"{entry['sha256']}, file is {digest}"
        )

    columns = schema.evaluation_columns if evaluation else schema.training_columns()
    try:
        file_schema = pq.read_schema(path)
        if not file_schema.equals(schema.table_schema, check_metadata=False):
            raise DatasetLoadError(
                f"split '{split}' schema does not match the '{schema.schema_version}' "
                "generated-dataset contract"
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
            for name in schema.feature_order
        ]
    ).astype(np.float64, copy=False)
    return SplitData(
        split=split,
        features=np.ascontiguousarray(features),
        option_types=option_types,
        columns=arrays,
        schema_version=schema.schema_version,
        target_column=schema.target_column,
        comparator_column=schema.comparator_column,
    )
