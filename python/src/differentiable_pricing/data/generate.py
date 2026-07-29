"""Deterministic European-option dataset generation.

Every label in the emitted tables comes from the compiled C++ analytic
Black-Scholes oracle exposed by :mod:`differentiable_pricing._core`. No pricing
formula is reimplemented in Python; this module only samples inputs, derives
the strike from the forward relation, calls the oracle, and records provenance.

Determinism
-----------
One base seed produces one independent stream per partition:

    SeedSequence(entropy=base_seed, spawn_key=(sha256(split)[:8],)) -> PCG64

Keying the stream on the split name rather than on an ordinal means adding or
resizing a partition never perturbs the others, and the partitions are drawn
independently rather than by random-splitting one previously generated table.
Within a stream the draws are consumed in a fixed order: the balanced
call/put assignment first, then :data:`~.config.DOMAIN_FIELDS` in order.

The reproducibility guarantee is scoped to a toolchain, not absolute: NumPy
gives no cross-version stream guarantee for ``Generator`` methods, and PyArrow
stamps its own version into the Parquet footer. The manifest therefore records
the NumPy, PyArrow, and Python versions used. Regenerating with the recorded
toolchain reproduces byte-identical files; a different toolchain must be
treated as a new dataset until its hashes are compared.

Domain caveat
-------------
The sampling box is a provisional synthetic engineering range, not a
calibrated market distribution. Draws are independent and uniform, so the
joint structure of real markets (skew, term structure, spot/volatility
correlation) is absent by construction.

Because ``strike = forward * exp(-log_forward_moneyness)``, the model collapses
to the standardized moneyness ``z = log_forward_moneyness / (volatility *
sqrt(maturity))``. Independent uniform draws therefore reach very large ``|z|``
in the short-maturity/low-volatility corner, where prices underflow towards
zero and delta saturates. :data:`LABEL_DIAGNOSTICS_THRESHOLDS` quantifies that
tail per partition in the manifest so downstream metrics can stratify it out.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .._core import __version__ as _ORACLE_VERSION
from .._core import black_scholes
from .config import (
    DOMAIN_FIELDS,
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    ConfigError,
    DatasetConfig,
    load_dataset_config,
)

MANIFEST_NAME: Final = "manifest.json"
PARQUET_COMPRESSION: Final = "snappy"
PARQUET_VERSION: Final = "2.6"
ORACLE_SYMBOL: Final = "differentiable_pricing._core.black_scholes"

LABEL_FIELDS: Final = ("price", "delta", "gamma", "vega", "theta", "rho")
"""Oracle outputs copied verbatim into the table, in column order."""

LABEL_DIAGNOSTICS_THRESHOLDS: Final[Mapping[str, float]] = {
    "near_zero_price": 1.0e-6,
    "saturated_delta": 1.0e-6,
}
"""Absolute cut-offs used to count degenerate rows; recorded in the manifest.

A price at or below the first cut-off carries almost no training signal and
makes relative price error meaningless. Delta is saturated when it sits within
the second cut-off of either end of its attainable range, ``0`` to
``exp(-dividend_yield * maturity)`` in absolute value.
"""

_STRING_COLUMNS: Final = ("sample_id", "split", "option_type")
_FLOAT_COLUMNS: Final = (
    "spot",
    "strike",
    "log_forward_moneyness",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
    *LABEL_FIELDS,
)

TABLE_SCHEMA: Final = pa.schema(
    [pa.field(name, pa.string(), nullable=False) for name in _STRING_COLUMNS]
    + [pa.field(name, pa.float64(), nullable=False) for name in _FLOAT_COLUMNS]
)
"""Column order and types of every emitted Parquet file."""

UNITS_AND_CONVENTIONS: Final[Mapping[str, str]] = {
    "currency": "single unspecified currency; spot, strike, and price share one unit",
    "floating_point": "float64 for every numeric column, feature and label alike",
    "exercise": "European, single expiry, no early exercise and no discrete dividends",
    "model": "Black-Scholes with a flat continuously compounded rate and dividend yield",
    "spot": "price of one unit of the underlying at valuation time, currency units",
    "strike": "forward * exp(-log_forward_moneyness), currency units",
    "forward": "spot * exp((rate - dividend_yield) * maturity); derived, not stored",
    "log_forward_moneyness": "log(forward / strike), dimensionless",
    "maturity": "time to expiry in years; the 7/365 lower bound uses ACT/365F",
    "rate": "continuously compounded annualised risk-free rate, flat term structure",
    "dividend_yield": "continuously compounded annualised dividend yield, flat",
    "volatility": "annualised lognormal volatility, decimal (0.20 = 20%)",
    "price": "present value at valuation time, currency units",
    "delta": "d price / d spot, dimensionless",
    "gamma": "d^2 price / d spot^2, per currency unit",
    "vega": "d price / d volatility, per 1.0 of absolute volatility, not per point",
    "theta": "d price / d calendar time = -d price / d maturity, per year, not per day",
    "rho": "d price / d rate, per 1.0 of absolute continuously compounded rate",
    "sampling": (
        "independent uniform draws over the configured box, half-open [low, high) "
        "in exact arithmetic; the upper bound is attainable only by rounding"
    ),
    "sample_id": "'<split>-<row index>'; stable for a fixed config hash, seed, and row count",
}

CAVEATS: Final[Sequence[str]] = (
    "Sampling ranges are provisional synthetic engineering ranges, not a calibrated "
    "market distribution; independent uniform draws omit real joint structure.",
    "Labels carry only the floating-point error of the analytic oracle; they contain "
    "no model risk, no bid/ask, and no market-data uncertainty.",
    "Only an in-envelope interpolation test is produced here. Boundary, "
    "extrapolation, and scenario partitions are out of scope for this version.",
    "Independent uniform draws reach large standardized moneyness "
    "|log_forward_moneyness| / (volatility * sqrt(maturity)), so a minority of rows "
    "have prices at or near zero and saturated delta. See label_diagnostics: "
    "relative price error is not a stable metric over the whole box, and these rows "
    "should be stratified rather than averaged away.",
    "Byte-identical regeneration is guaranteed only for the toolchain recorded under "
    "'runtime'; NumPy does not guarantee generator streams across versions and "
    "PyArrow stamps its version into the Parquet footer.",
)


class GenerationError(RuntimeError):
    """Raised when generation cannot produce a trustworthy dataset."""


def split_generator(base_seed: int, split: str) -> np.random.Generator:
    """Return the independent PCG64 stream assigned to ``split``."""
    key = int.from_bytes(hashlib.sha256(split.encode("utf-8")).digest()[:8], "big")
    sequence = np.random.SeedSequence(entropy=base_seed, spawn_key=(key,))
    return np.random.Generator(np.random.PCG64(sequence))


def sample_inputs(config: DatasetConfig, split: str) -> dict[str, np.ndarray]:
    """Draw the primitive inputs for ``split`` and derive the strike.

    The call/put assignment is exactly balanced up to one row when the row
    count is odd, then permuted so option type is independent of row position.
    """
    rows = config.rows[split]
    rng = split_generator(config.base_seed, split)

    is_call = np.zeros(rows, dtype=bool)
    is_call[: (rows + 1) // 2] = True
    rng.shuffle(is_call)

    columns: dict[str, np.ndarray] = {}
    for field in DOMAIN_FIELDS:
        bounds = config.domain[field]
        columns[field] = rng.uniform(bounds.low, bounds.high, size=rows)

    forward = columns["spot"] * np.exp(
        (columns["rate"] - columns["dividend_yield"]) * columns["maturity"]
    )
    columns["strike"] = forward * np.exp(-columns["log_forward_moneyness"])
    columns["option_type"] = np.where(is_call, "call", "put")
    return columns


def label_with_oracle(
    columns: Mapping[str, np.ndarray],
    *,
    split: str = "",
) -> dict[str, np.ndarray]:
    """Price every row with the compiled C++ oracle, one scalar call per row.

    A rejection by the oracle's input validation is re-raised as a
    :class:`GenerationError` naming the offending row, so a configuration whose
    fields are individually valid but jointly pathological (an overflowing
    forward, say) still fails through the documented error contract.
    """
    option_type = columns["option_type"]
    spot = columns["spot"]
    strike = columns["strike"]
    maturity = columns["maturity"]
    rate = columns["rate"]
    dividend_yield = columns["dividend_yield"]
    volatility = columns["volatility"]

    rows = int(spot.size)
    labels = {name: np.empty(rows, dtype=np.float64) for name in LABEL_FIELDS}
    for index in range(rows):
        try:
            result = black_scholes(
                str(option_type[index]),
                float(spot[index]),
                float(strike[index]),
                float(maturity[index]),
                float(rate[index]),
                float(dividend_yield[index]),
                float(volatility[index]),
            )
        except ValueError as error:
            location = f"split '{split}' row {index}" if split else f"row {index}"
            raise GenerationError(f"oracle rejected {location}: {error}") from error
        for name in LABEL_FIELDS:
            labels[name][index] = result[name]
    return labels


def build_table(config: DatasetConfig, split: str) -> pa.Table:
    """Sample, label, and assemble one partition as an Arrow table."""
    columns = sample_inputs(config, split)
    columns.update(label_with_oracle(columns, split=split))

    rows = config.rows[split]
    columns["sample_id"] = np.array([f"{split}-{index:09d}" for index in range(rows)])
    columns["split"] = np.full(rows, split)

    for name in _FLOAT_COLUMNS:
        values = columns[name]
        if values.dtype != np.float64:
            raise GenerationError(f"column '{name}' is {values.dtype}, expected float64")
        if not np.isfinite(values).all():
            raise GenerationError(f"column '{name}' contains non-finite values in split '{split}'")

    for field in DOMAIN_FIELDS:
        bounds = config.domain[field]
        values = columns[field]
        if not (bounds.contains(float(values.min())) and bounds.contains(float(values.max()))):
            raise GenerationError(
                f"sampled '{field}' leaves the configured domain "
                f"[{bounds.low}, {bounds.high}] in split '{split}'"
            )
    if not (columns["strike"] > 0.0).all():
        raise GenerationError(f"derived strike is not positive in split '{split}'")

    return pa.Table.from_pydict(
        {name: columns[name] for name in TABLE_SCHEMA.names},
        schema=TABLE_SCHEMA,
    )


def label_diagnostics(
    table: pa.Table,
    thresholds: Mapping[str, float] = LABEL_DIAGNOSTICS_THRESHOLDS,
) -> dict[str, int]:
    """Count degenerate rows so the manifest exposes the low-information tail.

    ``thresholds`` defaults to :data:`LABEL_DIAGNOSTICS_THRESHOLDS`, the values
    generation records in the manifest. It is a parameter so that a consumer
    reading an already-written dataset can recompute these counts under the
    thresholds that dataset was published with, rather than under whatever the
    installed generator currently declares.
    """
    price = table.column("price").to_numpy()
    delta = np.abs(table.column("delta").to_numpy())
    dividend_yield = table.column("dividend_yield").to_numpy()
    maturity = table.column("maturity").to_numpy()

    delta_bound = np.exp(-dividend_yield * maturity)
    tolerance = thresholds["saturated_delta"]
    saturated = (delta <= tolerance) | (delta_bound - delta <= tolerance)
    return {
        "rows": table.num_rows,
        "zero_price_rows": int((price == 0.0).sum()),
        "near_zero_price_rows": int((price <= thresholds["near_zero_price"]).sum()),
        "saturated_delta_rows": int(saturated.sum()),
    }


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of ``path``, read in bounded chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_manifest(config: DatasetConfig, files: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator_version": GENERATOR_VERSION,
        "dataset_name": config.dataset_name,
        "seed": {
            "base_seed": config.base_seed,
            "bit_generator": "numpy.random.PCG64",
            "stream_derivation": (
                "SeedSequence(entropy=base_seed, "
                "spawn_key=(int(sha256(split_name)[:8]),)) per partition"
            ),
            "draw_order": ["option_type", *DOMAIN_FIELDS],
        },
        "config": {"file": config.source_name, "sha256": config.source_sha256},
        "oracle": {
            "name": config.oracle_name,
            "symbol": ORACLE_SYMBOL,
            "kind": "analytic-closed-form",
            "language": "C++",
            "version": _ORACLE_VERSION,
            "numerical_error": (
                "IEEE-754 float64 rounding only; no discretization, truncation, or "
                "Monte Carlo error. Greeks are closed-form, not bumped."
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
        },
        "units_and_conventions": dict(UNITS_AND_CONVENTIONS),
        "domain": {
            field: [config.domain[field].low, config.domain[field].high]
            for field in DOMAIN_FIELDS
        },
        "columns": list(TABLE_SCHEMA.names),
        "row_counts": {split: config.rows[split] for split in SPLIT_NAMES},
        "label_diagnostics": {
            "thresholds": dict(LABEL_DIAGNOSTICS_THRESHOLDS),
            "splits": {entry["split"]: entry["diagnostics"] for entry in files},
        },
        "files": list(files),
        "parquet": {"compression": PARQUET_COMPRESSION, "format_version": PARQUET_VERSION},
        "caveats": list(CAVEATS),
    }


def generate_dataset(
    config: DatasetConfig,
    output_dir: Path | str,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write one Parquet file per partition plus a manifest, and return it.

    Existing files are only replaced when ``overwrite`` is set, so a stale
    dataset is never silently destroyed. The manifest is written last, and is
    therefore the marker that a directory holds a complete dataset.
    """
    directory = Path(output_dir)
    targets = [directory / f"{split}.parquet" for split in SPLIT_NAMES]
    targets.append(directory / MANIFEST_NAME)
    if not overwrite:
        existing = sorted(str(path) for path in targets if path.exists())
        if existing:
            raise GenerationError(
                f"refusing to overwrite existing files: {', '.join(existing)}; "
                "pass overwrite to replace them"
            )

    try:
        directory.mkdir(parents=True, exist_ok=True)
        files: list[dict[str, Any]] = []
        for split in SPLIT_NAMES:
            table = build_table(config, split)
            path = directory / f"{split}.parquet"
            pq.write_table(
                table,
                path,
                compression=PARQUET_COMPRESSION,
                version=PARQUET_VERSION,
                write_statistics=True,
            )
            files.append(
                {
                    "split": split,
                    "file": path.name,
                    "rows": table.num_rows,
                    "sha256": sha256_file(path),
                    "diagnostics": label_diagnostics(table),
                }
            )

        manifest = _build_manifest(config, files)
        (directory / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as error:
        raise GenerationError(f"cannot write dataset to '{directory}': {error}") from error
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m differentiable_pricing.data.generate",
        description="Generate the deterministic European-option dataset.",
    )
    parser.add_argument("--config", required=True, type=Path, help="dataset configuration TOML")
    parser.add_argument("--output", required=True, type=Path, help="output directory")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace existing Parquet files and manifest in the output directory",
    )
    arguments = parser.parse_args(argv)

    try:
        config = load_dataset_config(arguments.config)
        manifest = generate_dataset(config, arguments.output, overwrite=arguments.overwrite)
    except (ConfigError, GenerationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    total = sum(manifest["row_counts"].values())
    print(f"wrote {total} rows to {arguments.output} (config sha256 {config.source_sha256})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
