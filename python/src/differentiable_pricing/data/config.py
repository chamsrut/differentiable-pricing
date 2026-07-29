"""Validated, versioned configuration for the European-option dataset.

The configuration file is the single declaration of the sampling domain, the
partition sizes, and the base seed. It is hashed verbatim so that a manifest
can pin the exact assumptions a dataset was generated under.
"""

from __future__ import annotations

import hashlib
import math
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCHEMA_VERSION: Final = "european-option-dataset/1"
"""Only schema version this generator accepts."""

GENERATOR_VERSION: Final = "1.0.0"
"""Version of the sampling and labelling procedure itself."""

SPLIT_NAMES: Final = ("train", "validation", "interpolation_test")
"""Required partitions, in generation order."""

KNOWN_ORACLES: Final = frozenset({"dp::black_scholes"})
"""Oracles this generator can actually call.

The name is recorded in the manifest as provenance, so an unrecognised value
is rejected rather than stored as a plausible-looking but false claim.
"""

DOMAIN_FIELDS: Final = (
    "spot",
    "log_forward_moneyness",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
"""Sampled fields, in the fixed order in which random draws are consumed."""

STRICTLY_POSITIVE_FIELDS: Final = frozenset({"spot", "maturity", "volatility"})
"""Fields whose lower bound must be positive for the oracle to accept them."""

_TOP_LEVEL_KEYS: Final = frozenset({"schema_version", "generator_version", "dataset", "domain"})
_DATASET_KEYS: Final = frozenset({"name", "oracle", "base_seed", "rows"})


class ConfigError(ValueError):
    """Raised when a dataset configuration is missing, malformed, or unusable."""


@dataclass(frozen=True, slots=True)
class Bounds:
    """Sampling bounds for one scalar field.

    Draws are half-open ``[low, high)`` in exact arithmetic, so the interval is
    treated as inclusive when validating generated values: NumPy documents that
    ``high`` can be returned through floating-point rounding.
    """

    low: float
    high: float

    def contains(self, value: float) -> bool:
        """Return whether ``value`` lies within the inclusive interval."""
        return self.low <= value <= self.high


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    """A fully validated dataset configuration plus its provenance hash."""

    schema_version: str
    generator_version: str
    dataset_name: str
    oracle_name: str
    base_seed: int
    rows: Mapping[str, int]
    domain: Mapping[str, Bounds]
    source_name: str
    source_sha256: str


def load_dataset_config(path: Path | str) -> DatasetConfig:
    """Read, hash, and validate the dataset configuration at ``path``."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read configuration '{config_path}': {error}") from error

    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        message = f"configuration '{config_path}' is not valid UTF-8 TOML: {error}"
        raise ConfigError(message) from error

    return parse_dataset_config(document, source_name=config_path.name, source_sha256=digest)


def parse_dataset_config(
    document: Mapping[str, Any],
    *,
    source_name: str,
    source_sha256: str,
) -> DatasetConfig:
    """Validate an already-parsed TOML document. Unknown keys are rejected."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")

    schema_version = _require_str(document, "schema_version", "top-level")
    if schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schema_version '{schema_version}'; this generator only "
            f"accepts '{SCHEMA_VERSION}'"
        )
    generator_version = _require_str(document, "generator_version", "top-level")
    if generator_version != GENERATOR_VERSION:
        raise ConfigError(
            f"configuration declares generator_version '{generator_version}' but this "
            f"generator is '{GENERATOR_VERSION}'"
        )

    dataset = _require_table(document, "dataset", "top-level")
    _reject_unknown(dataset, _DATASET_KEYS, "[dataset]")
    dataset_name = _require_str(dataset, "name", "[dataset]")
    oracle_name = _require_str(dataset, "oracle", "[dataset]")
    if oracle_name not in KNOWN_ORACLES:
        known = ", ".join(sorted(KNOWN_ORACLES))
        raise ConfigError(
            f"[dataset].oracle '{oracle_name}' is not a known oracle; expected one of {known}"
        )
    base_seed = _require_int(dataset, "base_seed", "[dataset]")
    if base_seed < 0:
        raise ConfigError(f"[dataset].base_seed must be non-negative, got {base_seed}")

    rows_table = _require_table(dataset, "rows", "[dataset]")
    _reject_unknown(rows_table, frozenset(SPLIT_NAMES), "[dataset.rows]")
    rows: dict[str, int] = {}
    for split in SPLIT_NAMES:
        count = _require_int(rows_table, split, "[dataset.rows]")
        if count <= 0:
            raise ConfigError(f"[dataset.rows].{split} must be positive, got {count}")
        rows[split] = count

    domain_table = _require_table(document, "domain", "top-level")
    _reject_unknown(domain_table, frozenset(DOMAIN_FIELDS), "[domain]")
    domain = {field: _parse_bounds(domain_table, field) for field in DOMAIN_FIELDS}

    return DatasetConfig(
        schema_version=schema_version,
        generator_version=generator_version,
        dataset_name=dataset_name,
        oracle_name=oracle_name,
        base_seed=base_seed,
        rows=rows,
        domain=domain,
        source_name=source_name,
        source_sha256=source_sha256,
    )


def _parse_bounds(table: Mapping[str, Any], field: str) -> Bounds:
    if field not in table:
        raise ConfigError(f"[domain] is missing required key '{field}'")
    value = table[field]
    if not isinstance(value, list) or len(value) != 2:
        raise ConfigError(f"[domain].{field} must be a two-element [low, high] array")

    bounds: list[float] = []
    for index, item in enumerate(value):
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ConfigError(f"[domain].{field}[{index}] must be a number, got {item!r}")
        number = float(item)
        if not math.isfinite(number):
            raise ConfigError(f"[domain].{field}[{index}] must be finite, got {number!r}")
        bounds.append(number)

    low, high = bounds
    if not low < high:
        raise ConfigError(f"[domain].{field} requires low < high, got [{low}, {high}]")
    if field in STRICTLY_POSITIVE_FIELDS and low <= 0.0:
        raise ConfigError(
            f"[domain].{field} lower bound must be positive for the Black-Scholes "
            f"oracle, got {low}"
        )
    return Bounds(low=low, high=high)


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{where} contains unknown keys: {', '.join(unknown)}")


def _require_table(table: Mapping[str, Any], key: str, where: str) -> Mapping[str, Any]:
    if key not in table:
        raise ConfigError(f"{where} is missing required table '{key}'")
    value = table[key]
    if not isinstance(value, dict):
        raise ConfigError(f"{where}.{key} must be a table, got {type(value).__name__}")
    return value


def _require_str(table: Mapping[str, Any], key: str, where: str) -> str:
    if key not in table:
        raise ConfigError(f"{where} is missing required key '{key}'")
    value = table[key]
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{where}.{key} must be a non-empty string, got {value!r}")
    return value


def _require_int(table: Mapping[str, Any], key: str, where: str) -> int:
    if key not in table:
        raise ConfigError(f"{where} is missing required key '{key}'")
    value = table[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{where}.{key} must be an integer, got {value!r}")
    return value
