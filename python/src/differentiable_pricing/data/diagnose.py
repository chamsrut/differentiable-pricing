"""Deterministic diagnostics for an already-generated European-option dataset.

This tool describes a dataset. It never trains, samples, prices, or plots
anything, and it says nothing about model quality: every number below is a
property of the stored bytes, not of a surrogate fitted to them.

Trust order
-----------
The manifest is the dataset's declaration of what it contains, and the
per-file SHA-256 digests are the only link between that declaration and the
bytes on disk. Diagnostics therefore run in two strictly ordered phases:

1. validate the manifest, then hash *every* declared Parquet file and compare
   with the recorded digest;
2. only then open any Parquet file.

A failure in phase 1, a schema that does not match the pinned table contract,
an unknown ``option_type``, or a manifest row count that disagrees with the
Parquet footer all raise :class:`DiagnosticsError`. Those are corruption, not
measurements: reporting statistics over them would dress up untrusted bytes as
analysis. ``manifest.json`` is not itself hashed, so the row-count cross-check
is the one guard against a manifest edited in isolation.

Everything that *is* a measurement -- distribution statistics, standardized
moneyness bands, no-arbitrage bound violations, duplicated or shared sample
identifiers and economic states, coverage of the declared sampling domain,
disagreement with the manifest's recorded label diagnostics -- is reported in
the JSON body and summarised under ``findings``.

Determinism
-----------
The report carries a schema version, no wall-clock field, and no run
identifier. Every list is emitted in a sorted order and the document is dumped
with sorted keys, so two runs over identical input bytes on one installation
produce byte-identical output. That guarantee is scoped to a fixed toolchain
for the same reason the dataset's own is: reductions such as ``mean`` and
``std`` are not required to associate identically across NumPy builds or SIMD
widths, and the last unit in the last place of a summary statistic may move.
Counts, proportions of counts, and digests are exact everywhere.

Thresholds
----------
The near-zero-price and saturated-delta cut-offs are read from the manifest of
the dataset under examination, never from the installed generator's constants,
so a dataset published under older thresholds is described under those
thresholds. The recomputed counts are compared with the ones the manifest
recorded; a disagreement is a finding.

Command line::

    python -m differentiable_pricing.data.diagnose \\
        --dataset data/european-option-v1 \\
        --output data/european-option-v1/diagnostics.json

Exit status is ``0`` when no findings were raised, ``3`` when the dataset is
readable but findings were reported, and ``2`` when it could not be trusted
enough to describe.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .config import DOMAIN_FIELDS, SCHEMA_VERSION, SPLIT_NAMES
from .generate import MANIFEST_NAME, TABLE_SCHEMA, label_diagnostics, sha256_file

DIAGNOSTICS_SCHEMA_VERSION: Final = "european-option-diagnostics/1"
"""Version of the emitted report layout. Bump on any breaking key change."""

DIAGNOSTICS_TOOL_VERSION: Final = "1.0.0"
"""Version of the diagnostic procedure itself."""

MONEYNESS_BAND_EDGES: Final[Mapping[str, float]] = {"core_max": 4.0, "tail_max": 8.0}
"""Upper edges, in absolute standardized moneyness, of the core and tail bands.

``absolute_z = |log_forward_moneyness| / (volatility * sqrt(maturity))`` is the
single quantity Black-Scholes depends on once the strike is derived from the
forward, so it is the honest axis on which to stratify this dataset. Rows with
``absolute_z > tail_max`` price to within float64 noise of their intrinsic
value: they carry almost no gradient signal and relative error over them is
meaningless. They are counted here, not filtered, and must be evaluated as a
separate stratum rather than averaged into a headline metric.
"""

QUANTILE_LEVELS: Final = (0.01, 0.05, 0.50, 0.95, 0.99)
QUANTILE_KEYS: Final = ("p01", "p05", "p50", "p95", "p99")
QUANTILE_METHOD: Final = "linear"
"""Interpolating quantile estimator; pinned so the report is reproducible."""

STANDARD_DEVIATION_DDOF: Final = 0
"""Population standard deviation: these tables are the population of interest."""

ARBITRAGE_RELATIVE_TOLERANCE: Final = 1.0e-12
"""Bound-check slack, relative to ``max(discounted_spot, discounted_strike)``.

Both bounds and the price itself are assembled from a handful of float64
operations on quantities of the order of the discounted spot or strike, so the
representable disagreement is a few units in the last place of that scale,
around ``1e-16`` relative. ``1e-12`` leaves four orders of magnitude of head
room above rounding while remaining far below any economically meaningful
violation: on a spot of 100 it is a tenth of a nanocurrency-unit. The reported
maximum violation magnitudes are raw, tolerance-free, so the residual head room
is visible rather than hidden by the cut-off.
"""

MAX_REPORTED_EXAMPLES: Final = 10
"""Cap on example identifiers echoed into the report, sorted for stability."""

FLOAT_COLUMNS: Final = tuple(field.name for field in TABLE_SCHEMA if field.type == pa.float64())
"""Every float64 column of the pinned table: inputs, price, and Greeks."""

STATE_FIELDS: Final = (
    "option_type",
    "spot",
    "strike",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
"""Economic inputs that fully determine a row's label under the oracle.

Two rows agreeing on all of these are the same priced contract whatever their
identifiers say, which is the leakage the research contract actually cares
about. Comparison is on exact float64 bit patterns: this finds duplicated
states, not *near*-duplicate ones. Near-duplicate detection would need a
declared quantization of the input space, which this project has not fixed
yet, and inventing one here would make the count an artefact of that choice.
"""

_LABEL_DIAGNOSTIC_KEYS: Final = (
    "rows",
    "zero_price_rows",
    "near_zero_price_rows",
    "saturated_delta_rows",
)

_OPTION_TYPES: Final = ("call", "put")


class DiagnosticsError(RuntimeError):
    """Raised when a dataset cannot be trusted enough to be described."""


# --- manifest --------------------------------------------------------------


def _require(table: Mapping[str, Any], key: str, kind: type | tuple[type, ...], where: str) -> Any:
    if not isinstance(table, Mapping) or key not in table:
        raise DiagnosticsError(f"{where} is missing required key '{key}'")
    value = table[key]
    if isinstance(value, bool) and kind is not bool:
        raise DiagnosticsError(f"{where}.{key} has unexpected type bool")
    if not isinstance(value, kind):
        raise DiagnosticsError(f"{where}.{key} has unexpected type {type(value).__name__}")
    return value


def load_manifest(directory: Path) -> dict[str, Any]:
    """Read and structurally validate ``manifest.json`` under ``directory``.

    Only the parts this tool relies on are validated. The manifest remains the
    generator's contract; this is a consumer refusing to guess.
    """
    path = directory / MANIFEST_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise DiagnosticsError(f"cannot read manifest '{path}': {error}") from error
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as error:
        raise DiagnosticsError(f"manifest '{path}' is not valid JSON: {error}") from error
    if not isinstance(manifest, dict):
        raise DiagnosticsError(f"manifest '{path}' must be a JSON object")

    schema_version = _require(manifest, "schema_version", str, "manifest")
    if schema_version != SCHEMA_VERSION:
        raise DiagnosticsError(
            f"manifest declares schema_version '{schema_version}'; this tool only "
            f"understands '{SCHEMA_VERSION}'"
        )

    columns = _require(manifest, "columns", list, "manifest")
    if columns != list(TABLE_SCHEMA.names):
        raise DiagnosticsError(
            "manifest 'columns' does not match the pinned table schema: "
            f"expected {list(TABLE_SCHEMA.names)}, got {columns}"
        )

    row_counts = _require(manifest, "row_counts", dict, "manifest")
    if sorted(row_counts) != sorted(SPLIT_NAMES):
        raise DiagnosticsError(
            f"manifest 'row_counts' covers {sorted(row_counts)}, expected {sorted(SPLIT_NAMES)}"
        )
    for split in SPLIT_NAMES:
        count = _require(row_counts, split, int, "manifest.row_counts")
        if count < 0:
            raise DiagnosticsError(f"manifest.row_counts.{split} is negative: {count}")

    files = _require(manifest, "files", list, "manifest")
    seen: set[str] = set()
    declared_names: set[str] = set()
    for index, entry in enumerate(files):
        where = f"manifest.files[{index}]"
        if not isinstance(entry, dict):
            raise DiagnosticsError(f"{where} must be a JSON object, got {type(entry).__name__}")
        split = _require(entry, "split", str, where)
        if split not in SPLIT_NAMES:
            raise DiagnosticsError(f"{where}.split '{split}' is not a known partition")
        if split in seen:
            raise DiagnosticsError(f"manifest.files declares split '{split}' more than once")
        seen.add(split)
        name = _require(entry, "file", str, where)
        if not name.endswith(".parquet") or Path(name).name != name:
            raise DiagnosticsError(f"{where}.file '{name}' is not a plain Parquet file name")
        # Two splits pointing at one file would collapse the per-file checks
        # below into a single verification and quietly drop a partition.
        if name in declared_names:
            raise DiagnosticsError(
                f"manifest.files declares '{name}' for more than one partition"
            )
        declared_names.add(name)
        rows = _require(entry, "rows", int, where)
        if rows != row_counts[split]:
            raise DiagnosticsError(
                f"{where}.rows is {rows} but manifest.row_counts.{split} is {row_counts[split]}"
            )
        digest = _require(entry, "sha256", str, where)
        if len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
            raise DiagnosticsError(f"{where}.sha256 is not a lowercase hex SHA-256 digest")
    if seen != set(SPLIT_NAMES):
        missing = ", ".join(sorted(set(SPLIT_NAMES) - seen))
        raise DiagnosticsError(f"manifest.files does not declare partition(s): {missing}")

    domain = _require(manifest, "domain", dict, "manifest")
    if sorted(domain) != sorted(DOMAIN_FIELDS):
        raise DiagnosticsError(
            f"manifest 'domain' covers {sorted(domain)}, expected {sorted(DOMAIN_FIELDS)}"
        )
    for field in DOMAIN_FIELDS:
        bounds = domain[field]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise DiagnosticsError(f"manifest.domain.{field} must be a [low, high] pair")
        for value in bounds:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise DiagnosticsError(f"manifest.domain.{field} must hold numbers, got {value!r}")
            if not np.isfinite(value):
                raise DiagnosticsError(f"manifest.domain.{field} must be finite, got {value!r}")
        if not float(bounds[0]) < float(bounds[1]):
            raise DiagnosticsError(f"manifest.domain.{field} requires low < high, got {bounds}")

    diagnostics = _require(manifest, "label_diagnostics", dict, "manifest")
    thresholds = _require(diagnostics, "thresholds", dict, "manifest.label_diagnostics")
    for name in ("near_zero_price", "saturated_delta"):
        value = _require(thresholds, name, (int, float), "manifest.label_diagnostics.thresholds")
        if not np.isfinite(value) or value <= 0.0:
            raise DiagnosticsError(
                f"manifest.label_diagnostics.thresholds.{name} must be finite and positive, "
                f"got {value!r}"
            )
    recorded = _require(diagnostics, "splits", dict, "manifest.label_diagnostics")
    if sorted(recorded) != sorted(SPLIT_NAMES):
        raise DiagnosticsError(
            f"manifest.label_diagnostics.splits covers {sorted(recorded)}, "
            f"expected {sorted(SPLIT_NAMES)}"
        )
    for split in SPLIT_NAMES:
        where = f"manifest.label_diagnostics.splits.{split}"
        entry = _require(recorded, split, dict, "manifest.label_diagnostics.splits")
        if sorted(entry) != sorted(_LABEL_DIAGNOSTIC_KEYS):
            raise DiagnosticsError(
                f"{where} holds {sorted(entry)}, expected {sorted(_LABEL_DIAGNOSTIC_KEYS)}"
            )
        for key in _LABEL_DIAGNOSTIC_KEYS:
            count = _require(entry, key, int, where)
            if count < 0:
                raise DiagnosticsError(f"{where}.{key} is negative: {count}")
    return manifest


def verify_files(directory: Path, manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Hash every declared Parquet file and reject any mismatch.

    Runs to completion before a single row is read. Parquet files present in
    the directory but absent from the manifest are rejected too: a stray
    partition is either a half-finished regeneration or a file nobody has
    accounted for, and both make the manifest a false description. Non-Parquet
    files are ignored so that writing ``diagnostics.json`` into the dataset
    directory, as the documented command line does, is not self-defeating.
    """
    declared = {str(entry["file"]): str(entry["split"]) for entry in manifest["files"]}
    try:
        present = sorted(path.name for path in directory.glob("*.parquet") if path.is_file())
    except OSError as error:
        raise DiagnosticsError(f"cannot list dataset directory '{directory}': {error}") from error

    missing = sorted(set(declared) - set(present))
    if missing:
        raise DiagnosticsError(f"declared file(s) missing from '{directory}': {', '.join(missing)}")
    unexpected = sorted(set(present) - set(declared))
    if unexpected:
        raise DiagnosticsError(
            f"undeclared Parquet file(s) in '{directory}': {', '.join(unexpected)}"
        )

    recorded = {str(entry["file"]): entry for entry in manifest["files"]}
    verified: list[dict[str, Any]] = []
    for name in present:
        path = directory / name
        try:
            digest = sha256_file(path)
            size = path.stat().st_size
        except OSError as error:
            raise DiagnosticsError(f"cannot read '{path}': {error}") from error
        expected = str(recorded[name]["sha256"])
        if digest != expected:
            raise DiagnosticsError(
                f"sha256 mismatch for '{path}': manifest declares {expected}, file is {digest}"
            )
        verified.append(
            {
                "file": name,
                "split": declared[name],
                "bytes": int(size),
                "sha256": digest,
                "sha256_verified": True,
            }
        )
    return verified


# --- statistics ------------------------------------------------------------


def _finite_or_none(value: float) -> float | None:
    """Map a non-finite reduction to ``null`` so the report stays valid JSON."""
    number = float(value)
    return number if np.isfinite(number) else None


def column_statistics(values: np.ndarray) -> dict[str, Any]:
    """Summarise one float64 column.

    Finite and non-finite values are counted separately and every location
    statistic is computed over the finite subset only, so one NaN cannot
    silently turn an entire summary into ``null``.
    """
    total = int(values.size)
    finite = values[np.isfinite(values)]
    count = int(finite.size)
    statistics: dict[str, Any] = {
        "count": total,
        "finite": count,
        "non_finite": total - count,
    }
    keys = ("min", "max", "mean", "standard_deviation", *QUANTILE_KEYS)
    if count == 0:
        statistics.update(dict.fromkeys(keys))
        return statistics

    statistics["min"] = _finite_or_none(finite.min())
    statistics["max"] = _finite_or_none(finite.max())
    statistics["mean"] = _finite_or_none(finite.mean())
    statistics["standard_deviation"] = _finite_or_none(finite.std(ddof=STANDARD_DEVIATION_DDOF))
    quantiles = np.quantile(finite, QUANTILE_LEVELS, method=QUANTILE_METHOD)
    for key, quantile in zip(QUANTILE_KEYS, quantiles, strict=True):
        statistics[key] = _finite_or_none(quantile)
    return statistics


def absolute_standardized_moneyness(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return ``|log_forward_moneyness| / (volatility * sqrt(maturity))``."""
    return np.abs(columns["log_forward_moneyness"]) / (
        columns["volatility"] * np.sqrt(columns["maturity"])
    )


def moneyness_bands(absolute_z: np.ndarray) -> dict[str, Any]:
    """Partition rows into core, tail, and extreme standardized-moneyness bands.

    The three bands plus ``non_finite`` are exhaustive and disjoint, so their
    counts sum to the row count by construction.
    """
    core_max = MONEYNESS_BAND_EDGES["core_max"]
    tail_max = MONEYNESS_BAND_EDGES["tail_max"]
    finite = np.isfinite(absolute_z)
    counts = {
        "core": int((finite & (absolute_z <= core_max)).sum()),
        "tail": int((finite & (absolute_z > core_max) & (absolute_z <= tail_max)).sum()),
        "extreme": int((finite & (absolute_z > tail_max)).sum()),
        "non_finite": int((~finite).sum()),
    }
    rows = int(absolute_z.size)
    return {
        band: {
            "count": count,
            "proportion": (count / rows) if rows else None,
        }
        for band, count in counts.items()
    }


def arbitrage_bounds(columns: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Check the rowwise European no-arbitrage price bounds.

    With ``discounted_spot = S * exp(-q * T)`` and
    ``discounted_strike = K * exp(-r * T)``::

        call:  max(discounted_spot - discounted_strike, 0)
                   <= price <= discounted_spot
        put:   max(discounted_strike - discounted_spot, 0)
                   <= price <= discounted_strike

    A row counts as a violation only when it exceeds
    :data:`ARBITRAGE_RELATIVE_TOLERANCE` times
    ``max(discounted_spot, discounted_strike)``. The reported magnitudes are
    the raw excesses, so a clean dataset still shows how much head room it has.
    """
    discounted_spot = columns["spot"] * np.exp(-columns["dividend_yield"] * columns["maturity"])
    discounted_strike = columns["strike"] * np.exp(-columns["rate"] * columns["maturity"])
    is_call = columns["option_type"] == "call"

    lower = np.where(
        is_call,
        np.maximum(discounted_spot - discounted_strike, 0.0),
        np.maximum(discounted_strike - discounted_spot, 0.0),
    )
    upper = np.where(is_call, discounted_spot, discounted_strike)
    scale = np.maximum(discounted_spot, discounted_strike)
    tolerance = ARBITRAGE_RELATIVE_TOLERANCE * scale

    price = columns["price"]
    comparable = np.isfinite(price) & np.isfinite(lower) & np.isfinite(upper)
    lower_excess = np.where(comparable, np.maximum(lower - price, 0.0), 0.0)
    upper_excess = np.where(comparable, np.maximum(price - upper, 0.0), 0.0)
    excess = np.maximum(lower_excess, upper_excess)

    lower_violations = int((comparable & (lower - price > tolerance)).sum())
    upper_violations = int((comparable & (price - upper > tolerance)).sum())
    relative = np.divide(excess, scale, out=np.zeros_like(excess), where=scale > 0.0)
    rows = int(price.size)
    return {
        "rows_checked": int(comparable.sum()),
        "rows_not_comparable": rows - int(comparable.sum()),
        "lower_bound_violations": lower_violations,
        "upper_bound_violations": upper_violations,
        "violations": lower_violations + upper_violations,
        "max_lower_violation": float(lower_excess.max()) if rows else 0.0,
        "max_upper_violation": float(upper_excess.max()) if rows else 0.0,
        "max_violation": float(excess.max()) if rows else 0.0,
        "max_relative_violation": float(relative.max()) if rows else 0.0,
    }


# --- per-split diagnostics -------------------------------------------------


def _read_table(path: Path, split: str, manifest_rows: int) -> pa.Table:
    """Open one verified Parquet file and enforce the pinned table contract."""
    try:
        table = pq.read_table(path)
    except (OSError, pa.ArrowInvalid) as error:
        raise DiagnosticsError(f"cannot read Parquet file '{path}': {error}") from error

    if list(table.schema.names) != list(TABLE_SCHEMA.names):
        raise DiagnosticsError(
            f"'{path}' has columns {list(table.schema.names)}, "
            f"expected {list(TABLE_SCHEMA.names)}"
        )
    for field in TABLE_SCHEMA:
        actual = table.schema.field(field.name).type
        if actual != field.type:
            raise DiagnosticsError(
                f"'{path}' column '{field.name}' has type {actual}, expected {field.type}"
            )
    if table.num_rows != manifest_rows:
        raise DiagnosticsError(
            f"'{path}' holds {table.num_rows} rows but the manifest declares "
            f"{manifest_rows} for split '{split}'"
        )
    return table


def _columns(table: pa.Table) -> dict[str, np.ndarray]:
    return {
        name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names
    }


def _duplicate_ids(sample_id: np.ndarray) -> tuple[int, list[str]]:
    values, counts = np.unique(sample_id, return_counts=True)
    repeated = sorted(str(value) for value in values[counts > 1])
    duplicate_rows = int((counts[counts > 1] - 1).sum())
    return duplicate_rows, repeated[:MAX_REPORTED_EXAMPLES]


def state_keys(columns: Mapping[str, np.ndarray]) -> np.ndarray:
    """Return one opaque, exactly comparable key per row over :data:`STATE_FIELDS`.

    The option type is encoded as a float flag so the whole state becomes a
    contiguous float64 matrix, then each row is viewed as a single opaque
    ``void`` scalar. Comparison is bytewise, which is what "the same priced
    contract" means for float64 inputs and, unlike float comparison, is also
    well defined if a corrupted file carries NaNs.
    """
    is_call = (columns["option_type"] == "call").astype(np.float64)
    matrix = np.ascontiguousarray(
        np.column_stack([is_call, *(columns[field] for field in STATE_FIELDS[1:])])
    )
    width = matrix.shape[1] * matrix.itemsize
    return matrix.view(np.dtype((np.void, width))).ravel()


def _duplicate_states(keys: np.ndarray, sample_id: np.ndarray) -> tuple[int, list[str]]:
    """Count rows repeating an economic state already present in the split."""
    _, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
    repeated = counts > 1
    duplicate_rows = int((counts[repeated] - 1).sum())
    if not duplicate_rows:
        return 0, []
    involved = sorted(str(value) for value in sample_id[repeated[inverse.ravel()]])
    return duplicate_rows, involved[:MAX_REPORTED_EXAMPLES]


def domain_coverage(
    columns: Mapping[str, np.ndarray],
    domain: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    """Compare each sampled column with the bounds the manifest declares.

    The manifest is not hashed, so its declared domain is exactly the kind of
    claim that can drift from the bytes it describes. Any later in-domain or
    out-of-distribution statement rests on this block agreeing with the data.
    Bounds are treated as inclusive, matching the generator's own validation:
    draws are half-open in exact arithmetic but rounding can return ``high``.
    """
    coverage: dict[str, Any] = {}
    for field in DOMAIN_FIELDS:
        low, high = (float(domain[field][0]), float(domain[field][1]))
        values = columns[field]
        finite = values[np.isfinite(values)]
        coverage[field] = {
            "declared": [low, high],
            "observed_min": _finite_or_none(finite.min()) if finite.size else None,
            "observed_max": _finite_or_none(finite.max()) if finite.size else None,
            "rows_below": int((values < low).sum()),
            "rows_above": int((values > high).sum()),
        }
    return coverage


def diagnose_split(
    table: pa.Table,
    split: str,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Produce the diagnostic block for one verified partition."""
    columns = _columns(table)
    option_type = columns["option_type"]

    unexpected = sorted({str(value) for value in np.unique(option_type)} - set(_OPTION_TYPES))
    if unexpected:
        raise DiagnosticsError(
            f"split '{split}' contains unknown option_type value(s): {', '.join(unexpected)}"
        )

    thresholds = manifest["label_diagnostics"]["thresholds"]
    recomputed = label_diagnostics(table, thresholds)
    recorded = manifest["label_diagnostics"]["splits"].get(split)

    absolute_z = absolute_standardized_moneyness(columns)
    duplicate_rows, duplicate_examples = _duplicate_ids(columns["sample_id"])
    state_rows, state_examples = _duplicate_states(state_keys(columns), columns["sample_id"])
    mislabelled = int((columns["split"] != split).sum())

    return {
        "rows": int(table.num_rows),
        "manifest_rows": int(manifest["row_counts"][split]),
        "option_types": {
            "call": int((option_type == "call").sum()),
            "put": int((option_type == "put").sum()),
        },
        "statistics": {name: column_statistics(columns[name]) for name in FLOAT_COLUMNS},
        "derived_statistics": {"absolute_z": column_statistics(absolute_z)},
        "moneyness_bands": moneyness_bands(absolute_z),
        "no_arbitrage": arbitrage_bounds(columns),
        "label_diagnostics": {
            "recomputed": recomputed,
            "manifest": recorded,
            "matches_manifest": recorded == recomputed,
        },
        "sample_ids": {
            "rows": int(columns["sample_id"].size),
            "unique": int(np.unique(columns["sample_id"]).size),
            "duplicate_rows": duplicate_rows,
            "duplicate_examples": duplicate_examples,
        },
        "states": {
            "fields": list(STATE_FIELDS),
            "unique": int(np.unique(state_keys(columns)).size),
            "duplicate_rows": state_rows,
            "duplicate_examples": state_examples,
        },
        "domain_coverage": domain_coverage(columns, manifest["domain"]),
        "split_column_mismatches": mislabelled,
        "non_finite_values": sum(
            int((~np.isfinite(columns[name])).sum()) for name in FLOAT_COLUMNS
        ),
    }


def _cross_split_intersections(ids: Mapping[str, set[str]]) -> list[dict[str, Any]]:
    """Report shared sample identifiers for every unordered pair of splits.

    Identifiers are ``'<split>-<index>'`` by construction, so in a correctly
    generated dataset they can never collide across partitions. This check
    therefore detects a corrupted or hand-edited identifier column, not the
    train/test leakage the research contract is about; see
    :func:`_cross_split_state_intersections` for that.
    """
    intersections: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(ids), 2):
        shared = sorted(ids[left] & ids[right])
        intersections.append(
            {
                "splits": [left, right],
                "count": len(shared),
                "examples": shared[:MAX_REPORTED_EXAMPLES],
            }
        )
    return intersections


def _cross_split_state_intersections(
    keys: Mapping[str, np.ndarray],
    sample_ids: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    """Report economic states shared between partitions.

    This is the leakage check that matters: a state appearing in two splits is
    the same priced contract evaluated twice, whatever its identifier says. The
    partitions are drawn from independent PCG64 streams over a continuous box,
    so a genuine collision is astronomically unlikely -- which is exactly why a
    non-zero count here means the generation protocol was violated rather than
    that the sampler got unlucky.
    """
    intersections: list[dict[str, Any]] = []
    for left, right in itertools.combinations(sorted(keys), 2):
        shared = np.intersect1d(keys[left], keys[right])
        examples = sorted(
            str(value) for value in sample_ids[left][np.isin(keys[left], shared)]
        )
        intersections.append(
            {
                "splits": [left, right],
                "count": int(shared.size),
                "examples": examples[:MAX_REPORTED_EXAMPLES],
            }
        )
    return intersections


def _collect_findings(
    splits: Mapping[str, Mapping[str, Any]],
    intersections: Sequence[Mapping[str, Any]],
    state_intersections: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Flatten reportable problems into a sorted, machine-readable list."""
    findings: list[dict[str, str]] = []

    def add(check: str, scope: str, detail: str) -> None:
        findings.append({"check": check, "scope": scope, "detail": detail})

    for split in sorted(splits):
        report = splits[split]
        arbitrage = report["no_arbitrage"]
        if arbitrage["violations"]:
            add(
                "no_arbitrage",
                split,
                f"{arbitrage['violations']} row(s) outside the European price bounds; "
                f"maximum violation {arbitrage['max_violation']!r}",
            )
        if report["sample_ids"]["duplicate_rows"]:
            add(
                "duplicate_sample_ids",
                split,
                f"{report['sample_ids']['duplicate_rows']} duplicate row(s); "
                f"examples {report['sample_ids']['duplicate_examples']}",
            )
        if report["states"]["duplicate_rows"]:
            add(
                "duplicate_states",
                split,
                f"{report['states']['duplicate_rows']} row(s) repeat an economic state "
                f"already in the split; examples {report['states']['duplicate_examples']}",
            )
        for field, coverage in sorted(report["domain_coverage"].items()):
            outside = coverage["rows_below"] + coverage["rows_above"]
            if outside:
                add(
                    "domain_coverage",
                    f"{split}/{field}",
                    f"{outside} row(s) outside the declared domain "
                    f"{coverage['declared']}; observed "
                    f"[{coverage['observed_min']!r}, {coverage['observed_max']!r}]",
                )
        if report["non_finite_values"]:
            add(
                "non_finite_values",
                split,
                f"{report['non_finite_values']} non-finite value(s) across float columns",
            )
        if report["split_column_mismatches"]:
            add(
                "split_column",
                split,
                f"{report['split_column_mismatches']} row(s) whose 'split' column "
                f"does not equal '{split}'",
            )
        if not report["label_diagnostics"]["matches_manifest"]:
            add(
                "label_diagnostics",
                split,
                "recomputed label diagnostics differ from the manifest: "
                f"recomputed {report['label_diagnostics']['recomputed']}, "
                f"manifest {report['label_diagnostics']['manifest']}",
            )
    for entry in intersections:
        if entry["count"]:
            add(
                "sample_id_intersection",
                "/".join(entry["splits"]),
                f"{entry['count']} shared sample id(s); examples {entry['examples']}",
            )
    for entry in state_intersections:
        if entry["count"]:
            add(
                "state_intersection",
                "/".join(entry["splits"]),
                f"{entry['count']} economic state(s) present in both splits; "
                f"examples {entry['examples']}",
            )
    findings.sort(key=lambda item: (item["check"], item["scope"], item["detail"]))
    return findings


def diagnose_dataset(dataset_dir: Path | str) -> dict[str, Any]:
    """Validate, hash-verify, and describe the dataset under ``dataset_dir``."""
    directory = Path(dataset_dir)
    if not directory.is_dir():
        raise DiagnosticsError(f"dataset directory '{directory}' does not exist")

    manifest = load_manifest(directory)
    files = verify_files(directory, manifest)

    splits: dict[str, dict[str, Any]] = {}
    identifiers: dict[str, set[str]] = {}
    sample_ids: dict[str, np.ndarray] = {}
    keys: dict[str, np.ndarray] = {}
    for entry in files:
        split = str(entry["split"])
        table = _read_table(
            directory / str(entry["file"]), split, int(manifest["row_counts"][split])
        )
        splits[split] = diagnose_split(table, split, manifest)
        columns = _columns(table)
        sample_ids[split] = columns["sample_id"]
        identifiers[split] = {str(value) for value in columns["sample_id"]}
        keys[split] = state_keys(columns)

    # Defence in depth: no partition may fall out of the report between
    # manifest validation and here without the tool saying so.
    if set(splits) != set(SPLIT_NAMES):
        missing = ", ".join(sorted(set(SPLIT_NAMES) - set(splits)))
        raise DiagnosticsError(f"no diagnostics were produced for partition(s): {missing}")

    intersections = _cross_split_intersections(identifiers)
    state_intersections = _cross_split_state_intersections(keys, sample_ids)
    findings = _collect_findings(splits, intersections, state_intersections)
    return {
        "diagnostics_schema_version": DIAGNOSTICS_SCHEMA_VERSION,
        "diagnostics_tool_version": DIAGNOSTICS_TOOL_VERSION,
        "dataset": {
            "name": manifest.get("dataset_name"),
            "schema_version": manifest["schema_version"],
            "generator_version": manifest.get("generator_version"),
            "config": manifest.get("config"),
            "oracle": manifest.get("oracle"),
            "runtime": manifest.get("runtime"),
            "row_counts": dict(manifest["row_counts"]),
            "domain": {field: list(manifest["domain"][field]) for field in DOMAIN_FIELDS},
        },
        "thresholds": {
            "near_zero_price": manifest["label_diagnostics"]["thresholds"]["near_zero_price"],
            "saturated_delta": manifest["label_diagnostics"]["thresholds"]["saturated_delta"],
            "threshold_source": f"{MANIFEST_NAME}:label_diagnostics.thresholds",
            "arbitrage_relative_tolerance": ARBITRAGE_RELATIVE_TOLERANCE,
            "moneyness_band_edges": dict(MONEYNESS_BAND_EDGES),
            "quantile_method": QUANTILE_METHOD,
            "standard_deviation_ddof": STANDARD_DEVIATION_DDOF,
        },
        "files": files,
        "splits": splits,
        "cross_split": {
            "sample_id_intersections": intersections,
            "state_intersections": state_intersections,
        },
        "findings": findings,
        "status": "findings" if findings else "ok",
        "interpretation": [
            "These are dataset diagnostics, not model-performance metrics. Nothing "
            "here says a surrogate can price anything.",
            "absolute_z = |log_forward_moneyness| / (volatility * sqrt(maturity)). "
            "Rows in the 'extreme' band price to within float64 noise of intrinsic "
            "value; evaluate them as a separate stratum, never averaged into a "
            "headline error.",
            "Statistics are exact for counts and digests; float reductions are "
            "reproducible for a fixed toolchain, matching the dataset's own "
            "determinism guarantee.",
            "Statistics are pooled over the whole split, not conditioned on the "
            "moneyness band. Recompute them per band before quoting a tail "
            "quantile of price, gamma, or vega as a property of the tradable "
            "region.",
            "State leakage is checked on exact float64 input tuples. Near-duplicate "
            "states would need a declared quantization of the input space, which "
            "this project has not fixed.",
        ],
    }


def render_report(report: Mapping[str, Any]) -> str:
    """Serialise the report deterministically.

    ``allow_nan=False`` is a guard rather than a formatting choice: a NaN would
    both invalidate the JSON and betray a statistic that escaped the finite
    filtering in :func:`column_statistics`.
    """
    return json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        prog="python -m differentiable_pricing.data.diagnose",
        description="Describe a generated European-option dataset deterministically.",
    )
    parser.add_argument("--dataset", required=True, type=Path, help="dataset directory")
    parser.add_argument("--output", required=True, type=Path, help="diagnostics JSON to write")
    arguments = parser.parse_args(argv)

    try:
        report = diagnose_dataset(arguments.dataset)
        text = render_report(report)
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(text, encoding="utf-8", newline="\n")
    except DiagnosticsError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except (OSError, ValueError) as error:
        print(f"error: cannot write diagnostics to '{arguments.output}': {error}", file=sys.stderr)
        return 2

    rows = sum(int(entry["rows"]) for entry in report["splits"].values())
    findings = report["findings"]
    print(f"diagnosed {rows} rows across {len(report['splits'])} splits -> {arguments.output}")
    if findings:
        for finding in findings:
            print(f"finding: [{finding['check']}] {finding['scope']}: {finding['detail']}")
        return 3
    print("no findings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
