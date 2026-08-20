"""Recovered schema contract for the ``american-option-dataset/1`` tables.

Task 9E ported this module from the generator on the unmerged branch
``feat/american-dataset-v1`` (commit ``49ef72a``), which produced
``data/american-option-v1/``. It carries the **schema contract only** — column
names, order, types, nullability, the label definitions, the sampled domain,
and a strict manifest validator. It deliberately does **not** carry the
generation machinery: nothing here samples, prices, or writes a row, and this
repository still cannot regenerate that dataset from tracked sources alone.

What this module is for is the other direction: reading an existing
``american-option-dataset/1`` table under an explicit, versioned contract
instead of by coincidence.

Two things this schema is not
-----------------------------
``dividend_yield`` is a **flat, continuously compounded dividend yield**. It is
not a discrete cash-dividend schedule, and no column here can represent one:
there is no ex-date, no amount, and no dividend count. A dataset in this schema
therefore **cannot model SPY-style discrete cash dividends**.

There is no Greek column. No delta, gamma, vega, theta or rho is labelled, so
nothing loaded through this schema supplies a supervised sensitivity target.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa

SCHEMA_VERSION: Final = "american-option-dataset/1"
"""The only dataset schema version this contract describes."""

GENERATOR_VERSION: Final = "1.0.0"
"""Version of the sampling and labelling procedure that produced the tables."""

MANIFEST_NAME: Final = "manifest.json"

SPLIT_NAMES: Final = ("train", "validation", "interpolation_test")
"""Required partitions, in generation order."""

LABEL_POLICY_NAME: Final = "american-crr-adjacent-average/1"
"""The one label policy this schema version admits.

``price = 0.5 * (CRR(N) + CRR(N + 1))`` for the American contract, with the
European CRR leg averaged identically on the same lattice. Documented in
``docs/american-crr-contract.md``, "Label policy v1".
"""

KNOWN_ORACLES: Final = frozenset({"dp::crr_binomial"})
"""Engines that may legitimately appear as the manifest's oracle name."""

DOMAIN_FIELDS: Final = (
    "spot",
    "log_moneyness",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
"""Sampled fields, in the fixed order in which random draws were consumed."""

STRING_COLUMNS: Final = (
    "sample_id",
    "split",
    "stratum",
    "exercise_regime",
    "option_type",
    "label_policy",
)
INPUT_COLUMNS: Final = (
    "spot",
    "strike",
    "log_moneyness",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
LABEL_COLUMNS: Final = (
    "american_price",
    "american_price_at_steps",
    "american_price_at_steps_plus_one",
    "adjacent_step_gap",
    "european_crr_price",
    "european_crr_price_at_steps",
    "european_crr_price_at_steps_plus_one",
    "early_exercise_premium",
    "european_black_scholes_price",
    "intrinsic_value",
    "risk_neutral_probability",
    "exercise_activity_fraction",
)
INTEGER_COLUMNS: Final = (
    "label_steps",
    "early_exercise_nodes",
    "earliest_exercise_step",
    "exercise_boundary_layers",
)
FLOAT_COLUMNS: Final = (*INPUT_COLUMNS, *LABEL_COLUMNS)

TABLE_SCHEMA: Final = pa.schema(
    [pa.field(name, pa.string(), nullable=False) for name in STRING_COLUMNS]
    + [pa.field(name, pa.float64(), nullable=False) for name in FLOAT_COLUMNS]
    + [pa.field(name, pa.int64(), nullable=False) for name in INTEGER_COLUMNS]
)
"""Column order, types and nullability of every emitted Parquet file."""

OPTION_TYPES: Final = ("call", "put")

NO_EARLY_EXERCISE_STEP: Final = -1
"""Sentinel in ``earliest_exercise_step`` when no node materially exercises.

An integer sentinel rather than a null keeps every column non-nullable, which
makes "no non-finite values anywhere" an exact, checkable property.
"""

TARGET_COLUMN: Final = "american_price"
"""The supervised target: the American CRR label."""

EUROPEAN_COMPARATOR_COLUMN: Final = "european_crr_price"
"""The paired European price, on the same lattice as the target.

Exposed explicitly because it is the comparator the European-to-American
transfer question is asked against, and because
``american_price - european_crr_price`` is exactly the stored
``early_exercise_premium``.
"""

FEATURE_ORDER: Final = (
    "option_type",
    "spot",
    "strike",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
"""State variables of the ``american_raw_physical_v1`` representation.

Every state variable a constant-parameter American CRR price depends on, and
nothing derived. See :data:`REPRESENTATION` and
:data:`REPRESENTATIONS_REJECTED_FOR_AMERICAN`.
"""

REPRESENTATION: Final = "american_raw_physical_v1"
"""Name of the minimal admissible American representation.

It carries the same seven raw state variables as the European
``raw_physical_v1`` representation. It is named and versioned separately so a
later task declares it deliberately rather than inheriting a European
configuration, and so that the rejection below is explicit rather than
implicit.
"""

REPRESENTATIONS_REJECTED_FOR_AMERICAN: Final[Mapping[str, str]] = {
    "forward_normalized_v1": (
        "loses state: (option_type, log(F/K), sigma*sqrt(T)) determines a European "
        "price scaled by spot*exp(-q*T), but not an American one. Two contracts "
        "sharing a forward, a total volatility and an option type can differ in "
        "rate and dividend yield, and their early-exercise boundaries then differ. "
        "See test_american_feature_sufficiency.py, which measures the gap with the "
        "CRR engine."
    ),
}
"""Representations that must not be used to learn this schema's labels."""

UNITS_AND_CONVENTIONS: Final[Mapping[str, str]] = {
    "currency": (
        "single unspecified currency; spot, strike, and every price share one unit"
    ),
    "floating_point": "float64 for every price column, feature and label alike",
    "exercise": (
        "American, exercisable at every lattice time layer; continuous exercise is "
        "approximated by the discrete layer grid and requires step convergence"
    ),
    "model": (
        "constant-parameter lognormal dynamics with a flat continuously compounded "
        "rate and dividend yield, discretized by a Cox-Ross-Rubinstein lattice"
    ),
    "spot": "price of one unit of the underlying at valuation time, currency units",
    "strike": "spot * exp(-log_moneyness), currency units",
    "log_moneyness": (
        "log(spot / strike), dimensionless; spot moneyness, not forward moneyness"
    ),
    "maturity": "time to expiry in years; the 7/365 lower bound uses ACT/365F",
    "rate": "continuously compounded annualised risk-free rate, flat term structure",
    "dividend_yield": (
        "continuously compounded annualised dividend yield, flat; NOT a discrete "
        "cash-dividend schedule and unable to represent one"
    ),
    "volatility": "annualised lognormal volatility, decimal (0.20 = 20%)",
    "american_price": (
        "label: 0.5 * (american CRR at label_steps + american CRR at label_steps + 1), "
        "present value at valuation time, currency units"
    ),
    "american_price_at_steps": "american CRR price at label_steps, currency units",
    "american_price_at_steps_plus_one": (
        "american CRR price at label_steps + 1, currency units"
    ),
    "adjacent_step_gap": (
        "|american_price_at_steps - american_price_at_steps_plus_one|; an "
        "adjacent-refinement discrepancy, not a truncation-error bound"
    ),
    "european_crr_price": (
        "0.5 * (european CRR at label_steps + european CRR at label_steps + 1) for the "
        "same contract and the same lattice, currency units"
    ),
    "european_crr_price_at_steps": "european CRR price at label_steps, currency units",
    "european_crr_price_at_steps_plus_one": (
        "european CRR price at label_steps + 1, currency units"
    ),
    "early_exercise_premium": (
        "american_price - european_crr_price; both legs share one lattice, so this "
        "is non-negative exactly rather than up to a tolerance"
    ),
    "european_black_scholes_price": (
        "analytic European Black-Scholes price; the one genuinely independent control "
        "column, not part of the label policy"
    ),
    "intrinsic_value": (
        "max(spot - strike, 0) for a call, max(strike - spot, 0) for a put"
    ),
    "risk_neutral_probability": (
        "CRR up-move probability at label_steps; strictly inside (0, 1) or the row "
        "is rejected, never clipped"
    ),
    "exercise_activity_fraction": (
        "early_exercise_nodes / (label_steps * (label_steps + 1) / 2), the fraction of "
        "pre-expiry lattice nodes at which exercise is materially preferable"
    ),
    "label_steps": (
        "CRR step count N of the label policy; the adjacent tree uses N + 1"
    ),
    "early_exercise_nodes": (
        "pre-expiry nodes of the label_steps tree where exercise beats continuation by "
        "more than the engine's scale-relative indifference band"
    ),
    "earliest_exercise_step": (
        "earliest time layer with a materially preferable exercise node, 0 denoting "
        "the valuation date; -1 when no such node exists"
    ),
    "exercise_boundary_layers": (
        "count of pre-expiry time layers holding at least one exercise node"
    ),
    "option_type": "'call' or 'put'; the only two contract types in this schema",
    "split": (
        "name of the partition this row belongs to; equals the file's partition"
    ),
    "sampling": (
        "stratified mixture: exact largest-remainder row quotas per stratum, then "
        "independent uniform draws inside that stratum's box, half-open [low, high) "
        "in exact arithmetic"
    ),
    "stratum": "name of the sampling-mixture component the row was drawn from",
    "exercise_regime": "documentary regime label of that stratum, never a gate",
    "label_policy": "identifier of the labelling procedure, versioned with the schema",
    "sample_id": (
        "'<split>-<row index>'; stable for a fixed config hash, seed, and row count"
    ),
}
"""Recovered definition of every column, verbatim from the generator."""

CAVEATS: Final[Sequence[str]] = (
    "Labels are synthetic CRR model values, not market prices. They carry lattice "
    "discretization error and no model, liquidity, or market-data uncertainty.",
    "The adjacent-step gap is a refinement diagnostic. Neither it nor half of it is a "
    "certified truncation-error bound for an American price.",
    "The European CRR leg shares the label's lattice, so the early-exercise premium is "
    "exactly non-negative but is not the premium over an exact European value.",
    "Sampling ranges are a synthetic engineering design, not a calibrated market "
    "distribution. Stratum weights are a coverage choice, not market probabilities.",
    "No Greek is labelled. A finite difference of these prices is not a validated Greek.",
    "A flat continuous dividend yield is not a discrete cash-dividend schedule, and "
    "this schema cannot represent one.",
)
"""Recovered caveats. These are properties of the schema, not of one dataset."""

_LABEL_DIAGNOSTIC_KEYS: Final = (
    "rows",
    "zero_price_rows",
    "near_zero_price_rows",
    "zero_premium_rows",
    "no_early_exercise_rows",
    "at_intrinsic_rows",
    "large_adjacent_gap_rows",
)


class AmericanManifestError(RuntimeError):
    """Raised when an ``american-option-dataset/1`` manifest cannot be trusted."""


def _require(
    mapping: Mapping[str, Any],
    key: str,
    expected: type | tuple[type, ...],
    where: str,
) -> Any:
    if key not in mapping:
        raise AmericanManifestError(f"{where} is missing required key '{key}'")
    value = mapping[key]
    if isinstance(value, bool) and expected is not bool:
        raise AmericanManifestError(f"{where}.{key} must not be a boolean")
    if not isinstance(value, expected):
        names = (
            expected.__name__
            if isinstance(expected, type)
            else "/".join(item.__name__ for item in expected)
        )
        raise AmericanManifestError(
            f"{where}.{key} must be {names}, got {type(value).__name__}"
        )
    return value


def _require_digest(mapping: Mapping[str, Any], key: str, where: str) -> str:
    digest = _require(mapping, key, str, where)
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise AmericanManifestError(f"{where}.{key} is not a lowercase hex SHA-256")
    return digest


def load_american_manifest(directory: Path) -> dict[str, Any]:
    """Read and strictly validate an ``american-option-dataset/1`` manifest.

    A consumer refusing to guess: every field this repository relies on is
    checked, and anything that does not match the pinned schema is an error
    rather than a warning. The manifest remains the generator's contract.
    """
    path = Path(directory) / MANIFEST_NAME
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise AmericanManifestError(f"cannot read manifest '{path}': {error}") from error
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as error:
        raise AmericanManifestError(
            f"manifest '{path}' is not valid JSON: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise AmericanManifestError(f"manifest '{path}' must be a JSON object")

    schema_version = _require(manifest, "schema_version", str, "manifest")
    if schema_version != SCHEMA_VERSION:
        raise AmericanManifestError(
            f"manifest declares schema_version '{schema_version}'; this contract "
            f"only understands '{SCHEMA_VERSION}'"
        )

    columns = _require(manifest, "columns", list, "manifest")
    if columns != list(TABLE_SCHEMA.names):
        raise AmericanManifestError(
            "manifest 'columns' does not match the pinned table schema: "
            f"expected {list(TABLE_SCHEMA.names)}, got {columns}"
        )

    policy = _require(manifest, "label_policy", dict, "manifest")
    policy_name = _require(policy, "name", str, "manifest.label_policy")
    if policy_name != LABEL_POLICY_NAME:
        raise AmericanManifestError(
            f"manifest.label_policy.name is '{policy_name}'; this contract only "
            f"understands '{LABEL_POLICY_NAME}'"
        )
    steps = _require(policy, "steps", int, "manifest.label_policy")
    if steps <= 0:
        raise AmericanManifestError(
            f"manifest.label_policy.steps must be positive, got {steps}"
        )

    oracle = _require(manifest, "oracle", dict, "manifest")
    oracle_name = _require(oracle, "name", str, "manifest.oracle")
    if oracle_name not in KNOWN_ORACLES:
        raise AmericanManifestError(
            f"manifest.oracle.name '{oracle_name}' is not a recognised engine"
        )
    for key in ("crr_header_sha256", "crr_implementation_sha256", "crr_source_sha256"):
        _require_digest(oracle, key, "manifest.oracle")

    config = _require(manifest, "config", dict, "manifest")
    _require(config, "file", str, "manifest.config")
    _require_digest(config, "sha256", "manifest.config")

    generator_version = _require(manifest, "generator_version", str, "manifest")
    if not generator_version:
        raise AmericanManifestError("manifest.generator_version must be non-empty")

    row_counts = _require(manifest, "row_counts", dict, "manifest")
    if sorted(row_counts) != sorted(SPLIT_NAMES):
        raise AmericanManifestError(
            f"manifest 'row_counts' covers {sorted(row_counts)}, "
            f"expected {sorted(SPLIT_NAMES)}"
        )
    for split in SPLIT_NAMES:
        count = _require(row_counts, split, int, "manifest.row_counts")
        if count < 0:
            raise AmericanManifestError(
                f"manifest.row_counts.{split} is negative: {count}"
            )

    files = _require(manifest, "files", list, "manifest")
    seen: set[str] = set()
    declared_names: set[str] = set()
    for index, entry in enumerate(files):
        where = f"manifest.files[{index}]"
        if not isinstance(entry, dict):
            raise AmericanManifestError(
                f"{where} must be a JSON object, got {type(entry).__name__}"
            )
        split = _require(entry, "split", str, where)
        if split not in SPLIT_NAMES:
            raise AmericanManifestError(f"{where}.split '{split}' is not a partition")
        if split in seen:
            raise AmericanManifestError(
                f"manifest.files declares split '{split}' more than once"
            )
        seen.add(split)
        name = _require(entry, "file", str, where)
        if not name.endswith(".parquet") or Path(name).name != name:
            raise AmericanManifestError(
                f"{where}.file '{name}' is not a plain Parquet file name"
            )
        # Two splits pointing at one file would collapse the per-file checks
        # into a single verification and quietly drop a partition.
        if name in declared_names:
            raise AmericanManifestError(
                f"manifest.files declares '{name}' for more than one partition"
            )
        declared_names.add(name)
        rows = _require(entry, "rows", int, where)
        if rows != row_counts[split]:
            raise AmericanManifestError(
                f"{where}.rows is {rows} but manifest.row_counts.{split} is "
                f"{row_counts[split]}"
            )
        _require_digest(entry, "sha256", where)
    if seen != set(SPLIT_NAMES):
        missing = ", ".join(sorted(set(SPLIT_NAMES) - seen))
        raise AmericanManifestError(
            f"manifest.files does not declare partition(s): {missing}"
        )

    domain = _require(manifest, "domain", dict, "manifest")
    if sorted(domain) != sorted(DOMAIN_FIELDS):
        raise AmericanManifestError(
            f"manifest 'domain' covers {sorted(domain)}, "
            f"expected {sorted(DOMAIN_FIELDS)}"
        )
    for field in DOMAIN_FIELDS:
        bounds = domain[field]
        if not isinstance(bounds, list) or len(bounds) != 2:
            raise AmericanManifestError(
                f"manifest.domain.{field} must be a [low, high] pair"
            )
        for value in bounds:
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise AmericanManifestError(
                    f"manifest.domain.{field} must hold numbers, got {value!r}"
                )
            if not np.isfinite(value):
                raise AmericanManifestError(
                    f"manifest.domain.{field} must be finite, got {value!r}"
                )
        if not float(bounds[0]) < float(bounds[1]):
            raise AmericanManifestError(
                f"manifest.domain.{field} requires low < high, got {bounds}"
            )

    diagnostics = _require(manifest, "label_diagnostics", dict, "manifest")
    recorded = _require(diagnostics, "splits", dict, "manifest.label_diagnostics")
    if sorted(recorded) != sorted(SPLIT_NAMES):
        raise AmericanManifestError(
            f"manifest.label_diagnostics.splits covers {sorted(recorded)}, "
            f"expected {sorted(SPLIT_NAMES)}"
        )
    for split in SPLIT_NAMES:
        where = f"manifest.label_diagnostics.splits.{split}"
        entry = _require(recorded, split, dict, "manifest.label_diagnostics.splits")
        if sorted(entry) != sorted(_LABEL_DIAGNOSTIC_KEYS):
            raise AmericanManifestError(
                f"{where} holds {sorted(entry)}, expected {sorted(_LABEL_DIAGNOSTIC_KEYS)}"
            )
        for key in _LABEL_DIAGNOSTIC_KEYS:
            count = _require(entry, key, int, where)
            if count < 0:
                raise AmericanManifestError(f"{where}.{key} is negative: {count}")
        if entry["rows"] != row_counts[split]:
            raise AmericanManifestError(
                f"{where}.rows is {entry['rows']} but manifest.row_counts.{split} "
                f"is {row_counts[split]}"
            )
    return manifest
