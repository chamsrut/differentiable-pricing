"""Task 9C-C2a: leakage-safe harvesting of spot rows from PDE valuation surfaces.

Task 9C-C1 exposed the whole valuation-time slice of one backward induction.
Turning that slice into several dataset rows is cheap and is exactly where a
dataset silently loses its independence assumption: **one surface yielding many
rows does not make those rows independent observations.** They are correlated
numerical outputs of one solve.

This module implements the smallest machinery that keeps that fact structural
rather than documentary:

* three versioned canonical identities -- ``partition_group_id`` for the base
  non-spot economic state, ``surface_id`` for one actual solve, ``row_id`` for
  one harvested exact node -- each a SHA-256 digest of a canonical JSON payload
  with no dependence on ``hash()``, dictionary order, locale, platform float
  formatting, absolute paths, or insertion order;
* a deterministic grouped partition assignment that consumes *only* candidate
  group identities and therefore provably runs before any PDE solve;
* exact-node harvesting inside a predeclared interior window, with the surface
  contract's structural admissibility rules carried through untouched and
  predeclared per-regime density quotas applied only after a group's partition
  is already fixed;
* a report that states raw row count, independent design-group count and actual
  surface-solve count as three separate numbers.

It computes no vega, runs no worker pool, replaces no LCP solver, trains
nothing, generates no production dataset, and selects no label policy. Task 9C-B
remains ``no_policy_selected``; nothing here reads, reruns or reinterprets it.

Command line::

    python -m differentiable_pricing.american.pde_surface_harvest \\
        --config configs/pde_surface_harvest_demo_v1.toml \\
        --output-directory artifacts/pde-surface-harvest-demo-v1
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

from differentiable_pricing import _pde, pde_valuation_surface

SCHEMA_VERSION: Final = "pde-surface-harvest/2"
REPORT_SCHEMA_VERSION: Final = "pde-surface-harvest-report/2"
ROW_SCHEMA_VERSION: Final = "pde-surface-harvest-row/2"
IDENTITY_VERSION: Final = "pde-surface-harvest-identity/2"
PARTITION_ALGORITHM: Final = "grouped_quota_by_sorted_digest/1"
SELECTION_RULE: Final = "evenly_spaced_by_ascending_node_index/1"
NODE_SOURCE: Final = "exact_grid_nodes"
ENGINE_NAME: Final = "dp::finite_difference_valuation_surface/v1"

PARTITION_NAMES: Final = ("train", "validation", "interpolation_test")
"""Partitions, in declared order.

This mirrors :data:`differentiable_pricing.data.config.SPLIT_NAMES` so the two
generators name the same three partitions. A test asserts the equality rather
than importing across stages.
"""

EXERCISE_STATES: Final = ("continuation", "exercise", "numerically_indifferent", "no_obstacle")
"""Exercise classifications the task 9C-C1 surface can report, in report order."""

APPLICABLE_EXERCISE_STATES: Final = {
    "american": ("continuation", "exercise", "numerically_indifferent"),
    "european": ("no_obstacle",),
}
"""Which regimes a contract style can actually produce.

A European contract poses no complementarity problem, so every node is
``no_obstacle``; an American one never reports that state. A quota for a regime
the style cannot produce is *inapplicable*, not a shortfall, and is reported as
such: calling it a shortfall would bury the shortfalls that are real.
"""

GREEK_INELIGIBILITY_REASONS: Final = (
    "centered_stencil_unavailable",
    "inside_domain_boundary_buffer",
    "non_finite_stencil_value",
    "unresolved_exercise_state",
    "regime_stencil_not_uniform",
)
"""Reason vocabulary of the surface contract, excluding ``eligible``."""

SURFACE_ROLES: Final = ("base", "sigma_down", "sigma_up")
"""Volatility-bump roles of one partition group.

Only ``base`` changes anything today. The other two exist so that the three
surfaces a future vega estimate needs are already, structurally, members of one
partition group. **No vega is computed anywhere in this module.**
"""

OPTION_TYPES: Final = frozenset({"call", "put"})
EXERCISE_STYLES: Final = frozenset({"european", "american"})

PARTITION_GROUP_EXCLUDES: Final = (
    "queried_spot",
    "harvested_node_index",
    "option_type",
    "exercise_style",
    "volatility_bump_role",
    "numerical_grid_and_solver_settings",
    "reporting_only_contract_multiplier",
)
"""What a partition group deliberately ignores.

Spot and node index are excluded because harvesting many spots from one state is
the whole point. The volatility-bump role is excluded because a base/down/up
triple must not straddle partitions. Option type and exercise style are excluded
as the *conservative* choice: a call and a put on one base state are linked by
parity, and a European contract is the natural dominance control of the American
one on the same state, so all four are solver siblings of one scenario and are
kept together. The numerical grid is excluded because the same economic state
solved on two grids is still one state; excluding it can only merge groups,
never split them.

``contract_multiplier`` is excluded because the PDE contract states that it is
carried for the task 9B input interface and **never enters pricing arithmetic**:
prices are per share, and a regression test asserts that changing the multiplier
does not change the price. Two candidates differing only in it are therefore the
same pricing state, and letting the multiplier split them would manufacture a
second design point out of a reporting convention -- and, worse, could place two
identical pricing states in different partitions. It stays in row and report
metadata, where it is a reporting fact rather than an identity.

``settlement`` is deliberately *not* excluded even though the current engine does
not use it either. It names a genuinely different contract term whose economic
content a later engine may price, whereas a multiplier can only ever scale a
reported number.
"""

ROWS_CORRELATED_STATEMENT: Final = (
    "Rows sharing a surface_id are correlated numerical outputs of one backward "
    "induction, and rows sharing a partition_group_id are correlated outputs of "
    "shared numerical work on one base economic state. They are not independent "
    "observations and must not be counted as such."
)

DESIGN_GROUP_STATEMENT: Final = (
    "independent_design_group_count is the number of distinct base economic "
    "states that produced rows. It is a design count, not a formally estimated "
    "statistical effective sample size, and no estimator has been fitted to it."
)

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "study",
        "identity",
        "partitioning",
        "solver",
        "grid",
        "surfaces",
        "harvest",
        "scenarios",
    }
)
_STUDY_KEYS: Final = frozenset(
    {"name", "status", "engine", "price_units", "greek_method", "curve_construction"}
)
_IDENTITY_KEYS: Final = frozenset({"version", "digest", "partition_group_excludes"})
_PARTITIONING_KEYS: Final = frozenset(
    {"algorithm", "seed", "minimum_groups_per_partition", "weights"}
)
_SOLVER_KEYS: Final = frozenset(
    {
        "spot_maximum_strike_multiple",
        "rannacher_steps",
        "psor_tolerance",
        "psor_relaxation",
        "psor_maximum_iterations",
        "settlement",
        "contract_multiplier",
    }
)
_GRID_KEYS: Final = frozenset({"spot_intervals", "time_steps", "boundary_exclusion_nodes"})
_SURFACES_KEYS: Final = frozenset(
    {"roles", "option_types", "exercise_styles", "volatility_bump"}
)
_HARVEST_KEYS: Final = frozenset(
    {
        "status",
        "node_source",
        "moneyness_window_low",
        "moneyness_window_high",
        "selection_rule",
        "quota",
    }
)
_SCENARIO_KEYS: Final = frozenset(
    {
        "name",
        "strike",
        "valuation_time",
        "expiry_time",
        "rate",
        "base_volatility",
        "continuous_carry",
        "dividends",
    }
)


class HarvestError(RuntimeError):
    """Raised when the harvesting contract, execution, or publication fails."""


# ---------------------------------------------------------------------------
# Canonical identities
# ---------------------------------------------------------------------------


def canonical_payload(value: Any) -> str:
    """Return the canonical JSON text used for every identity digest.

    Floats are rendered as ``.17g`` tagged strings rather than left to a JSON
    float writer, so no repr, locale or platform formatting difference can reach
    a digest. Negative zero is normalized to positive zero first: IEEE-754 makes
    ``-0.0 == 0.0`` true while ``.17g`` prints them differently, so without the
    normalization two states that compare equal everywhere -- and price
    identically -- would receive different identities. A carry of ``-0.0``
    arrives easily, for example as ``-rate * expiry_time`` at a zero rate.
    Mappings are emitted with sorted keys, so no insertion order can reach a
    digest either. Sequences keep their order, because a declared order *is*
    part of the state being identified.
    """
    return json.dumps(_canonicalize(value), sort_keys=True, separators=(",", ":"))


def _canonicalize(value: Any) -> Any:
    if isinstance(value, bool):
        return "b:1" if value else "b:0"
    if isinstance(value, int):
        return f"i:{value:d}"
    if isinstance(value, float):
        if not math.isfinite(value):
            raise HarvestError("an identity payload may not contain a non-finite number")
        # `value == 0.0` is true for both signed zeros and for nothing else, so
        # this normalizes -0.0 and leaves every other magnitude untouched.
        return f"f:{0.0 if value == 0.0 else value:.17g}"
    if isinstance(value, str):
        return f"s:{value}"
    if value is None:
        return "n:"
    if isinstance(value, Mapping):
        keys = list(value)
        if any(not isinstance(key, str) for key in keys):
            raise HarvestError("an identity payload may only use string keys")
        if len(set(keys)) != len(keys):
            raise HarvestError("an identity payload may not repeat a key")
        return {key: _canonicalize(value[key]) for key in keys}
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    raise HarvestError(f"an identity payload may not contain {type(value).__name__}")


def _digest(prefix: str, payload: Mapping[str, Any]) -> str:
    text = canonical_payload(payload)
    return f"{prefix}-{hashlib.sha256(text.encode('utf-8')).hexdigest()}"


def partition_group_identity(scenario: Scenario) -> str:
    """Return the canonical ID of one base non-spot economic state.

    See :data:`PARTITION_GROUP_EXCLUDES` for what is deliberately absent --
    including ``contract_multiplier``, which never enters pricing arithmetic.
    Note that the scenario's *name* is absent too: a name is documentation, and
    letting it enter the digest would let two identical economic states claim to
    be different design points and defeat duplicate detection.
    """
    return _digest("pg", _partition_group_payload(scenario))


def _partition_group_payload(scenario: Scenario) -> dict[str, Any]:
    return {
        "identity_version": IDENTITY_VERSION,
        "kind": "partition_group",
        "strike": scenario.strike,
        "valuation_time": scenario.valuation_time,
        "expiry_time": scenario.expiry_time,
        "base_volatility": scenario.base_volatility,
        "continuous_carry": scenario.continuous_carry,
        "curve_times": list(scenario.curve_times),
        "curve_log_discounts": list(scenario.curve_log_discounts),
        "dividends": [list(dividend) for dividend in scenario.dividends],
        "settlement": scenario.settlement,
    }


def solver_input_identity(descriptor: SolverInputDescriptor) -> str:
    """Return the canonical identity of the inputs affecting one PDE solve.

    Partition/group provenance, scenario names, surface roles and the reporting-
    only contract multiplier are deliberately absent. Consequently two calls
    with identical pricing and numerical inputs receive the same identity even
    when they were requested for different downstream purposes.
    """
    return _digest(
        "si",
        {
            "identity_version": IDENTITY_VERSION,
            "kind": "solver_input",
            **descriptor.payload(),
        },
    )


def surface_identity(descriptor: SolverInputDescriptor) -> str:
    """Return the actual-solve identity used as ``surface_id``."""
    return solver_input_identity(descriptor)


def row_identity(*, surface_id: str, node_index: int) -> str:
    """Return the canonical ID of one harvested exact node of one surface.

    The node index is the whole spot coordinate: the grid is already pinned by
    ``settings_digest`` inside ``surface_id``, so the index determines the spot
    exactly and no float need enter the digest.
    """
    return _digest(
        "rw",
        {
            "identity_version": IDENTITY_VERSION,
            "kind": "row",
            "surface_id": surface_id,
            "node_index": node_index,
        },
    )


def settings_identity(settings: Mapping[str, Any]) -> str:
    """Return the canonical ID of one numerical grid and solver configuration."""
    return _digest(
        "st", {"identity_version": IDENTITY_VERSION, "kind": "settings", **dict(settings)}
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SolverInputDescriptor:
    """Every pricing and numerical input that can affect one PDE surface solve."""

    option_type: str
    exercise_style: str
    strike: float
    valuation_time: float
    expiry_time: float
    volatility: float
    continuous_carry: float
    curve_times: tuple[float, ...]
    curve_log_discounts: tuple[float, ...]
    dividends: tuple[tuple[float, float], ...]
    settlement: str
    spot_intervals: int
    time_steps: int
    spot_maximum: float
    rannacher_steps: int
    psor_tolerance: float
    psor_relaxation: float
    psor_maximum_iterations: int
    boundary_exclusion_nodes: int

    def payload(self) -> dict[str, Any]:
        return {
            "option_type": self.option_type,
            "exercise_style": self.exercise_style,
            "strike": self.strike,
            "valuation_time": self.valuation_time,
            "expiry_time": self.expiry_time,
            "volatility": self.volatility,
            "continuous_carry": self.continuous_carry,
            "curve_times": list(self.curve_times),
            "curve_log_discounts": list(self.curve_log_discounts),
            "dividends": [list(item) for item in self.dividends],
            "settlement": self.settlement,
            "spot_intervals": self.spot_intervals,
            "time_steps": self.time_steps,
            "spot_maximum": self.spot_maximum,
            "rannacher_steps": self.rannacher_steps,
            "psor_tolerance": self.psor_tolerance,
            "psor_relaxation": self.psor_relaxation,
            "psor_maximum_iterations": self.psor_maximum_iterations,
            "boundary_exclusion_nodes": self.boundary_exclusion_nodes,
        }


@dataclass(frozen=True, slots=True)
class Scenario:
    """One base non-spot economic state, before any option type or bump role."""

    name: str
    strike: float
    valuation_time: float
    expiry_time: float
    rate: float
    base_volatility: float
    continuous_carry: float
    dividends: tuple[tuple[float, float], ...]
    settlement: str
    contract_multiplier: float

    @property
    def curve_times(self) -> tuple[float, float]:
        return (0.0, self.expiry_time)

    @property
    def curve_log_discounts(self) -> tuple[float, float]:
        return (0.0, -self.rate * self.expiry_time)


@dataclass(frozen=True, slots=True)
class SolverSettings:
    spot_maximum_strike_multiple: float
    rannacher_steps: int
    psor_tolerance: float
    psor_relaxation: float
    psor_maximum_iterations: int
    settlement: str
    contract_multiplier: float


@dataclass(frozen=True, slots=True)
class GridSettings:
    spot_intervals: int
    time_steps: int
    boundary_exclusion_nodes: int


@dataclass(frozen=True, slots=True)
class SurfaceDesign:
    roles: tuple[str, ...]
    option_types: tuple[str, ...]
    exercise_styles: tuple[str, ...]
    volatility_bump: float | None


@dataclass(frozen=True, slots=True)
class HarvestRules:
    status: str
    node_source: str
    moneyness_window_low: float
    moneyness_window_high: float
    selection_rule: str
    quota: tuple[tuple[str, int], ...]

    def quota_for(self, exercise_state: str) -> int:
        for name, value in self.quota:
            if name == exercise_state:
                return value
        raise HarvestError(f"no quota declared for exercise state '{exercise_state}'")


@dataclass(frozen=True, slots=True)
class PartitionSettings:
    algorithm: str
    seed: int
    minimum_groups_per_partition: int
    weights: tuple[tuple[str, float], ...]


@dataclass(frozen=True, slots=True)
class HarvestConfig:
    study_name: str
    study_status: str
    engine: str
    price_units: str
    greek_method: str
    curve_construction: str
    identity_version: str
    identity_digest: str
    partitioning: PartitionSettings
    solver: SolverSettings
    grid: GridSettings
    surfaces: SurfaceDesign
    harvest: HarvestRules
    scenarios: tuple[Scenario, ...]
    source_name: str
    raw_config_sha256: str

    @property
    def semantic_config_sha256(self) -> str:
        payload = canonical_payload(_semantic_config_payload(self)).encode()
        return hashlib.sha256(payload).hexdigest()


def _scenario_payload(scenario: Scenario) -> dict[str, Any]:
    return {
        "name": scenario.name,
        "strike": scenario.strike,
        "valuation_time": scenario.valuation_time,
        "expiry_time": scenario.expiry_time,
        "rate": scenario.rate,
        "base_volatility": scenario.base_volatility,
        "continuous_carry": scenario.continuous_carry,
        "dividends": [list(item) for item in scenario.dividends],
        "settlement": scenario.settlement,
        "contract_multiplier": scenario.contract_multiplier,
    }


def _semantic_config_payload(config: HarvestConfig) -> dict[str, Any]:
    """Return parsed configuration semantics, independent of TOML byte ordering."""
    scenarios = sorted(
        (_scenario_payload(scenario) for scenario in config.scenarios),
        key=canonical_payload,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "study": {
            "name": config.study_name,
            "status": config.study_status,
            "engine": config.engine,
            "price_units": config.price_units,
            "greek_method": config.greek_method,
            "curve_construction": config.curve_construction,
        },
        "identity": {
            "version": config.identity_version,
            "digest": config.identity_digest,
            "partition_group_excludes": list(PARTITION_GROUP_EXCLUDES),
        },
        "partitioning": {
            "algorithm": config.partitioning.algorithm,
            "seed": config.partitioning.seed,
            "minimum_groups_per_partition": config.partitioning.minimum_groups_per_partition,
            "weights": {name: value for name, value in config.partitioning.weights},
        },
        "solver": {
            "spot_maximum_strike_multiple": config.solver.spot_maximum_strike_multiple,
            "rannacher_steps": config.solver.rannacher_steps,
            "psor_tolerance": config.solver.psor_tolerance,
            "psor_relaxation": config.solver.psor_relaxation,
            "psor_maximum_iterations": config.solver.psor_maximum_iterations,
            "settlement": config.solver.settlement,
            "contract_multiplier": config.solver.contract_multiplier,
        },
        "grid": {
            "spot_intervals": config.grid.spot_intervals,
            "time_steps": config.grid.time_steps,
            "boundary_exclusion_nodes": config.grid.boundary_exclusion_nodes,
        },
        "surfaces": {
            "roles": sorted(config.surfaces.roles),
            "option_types": sorted(config.surfaces.option_types),
            "exercise_styles": sorted(config.surfaces.exercise_styles),
            "volatility_bump": config.surfaces.volatility_bump,
        },
        "harvest": {
            "status": config.harvest.status,
            "node_source": config.harvest.node_source,
            "moneyness_window_low": config.harvest.moneyness_window_low,
            "moneyness_window_high": config.harvest.moneyness_window_high,
            "selection_rule": config.harvest.selection_rule,
            "quota": {name: value for name, value in config.harvest.quota},
        },
        "scenarios": scenarios,
    }


def load_harvest_config(path: Path) -> HarvestConfig:
    """Read, validate, and hash one versioned harvesting configuration."""
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise HarvestError(f"cannot read configuration '{path}': {error}") from error
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise HarvestError(f"configuration '{path}' is not valid TOML: {error}") from error
    return parse_harvest_config(
        document, raw_config_sha256=hashlib.sha256(raw).hexdigest(), source_name=path.name
    )


def parse_harvest_config(
    document: Mapping[str, Any], *, source_name: str, raw_config_sha256: str
) -> HarvestConfig:
    """Validate one already-decoded configuration document."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "configuration")
    version = _require_string(document, "schema_version", "configuration")
    if version != SCHEMA_VERSION:
        raise HarvestError(
            f"configuration schema_version '{version}' is not '{SCHEMA_VERSION}'"
        )

    study = _require_table(document, "study", "configuration")
    _reject_unknown(study, _STUDY_KEYS, "study")

    identity = _require_table(document, "identity", "configuration")
    _reject_unknown(identity, _IDENTITY_KEYS, "identity")
    identity_version = _require_string(identity, "version", "identity")
    if identity_version != IDENTITY_VERSION:
        raise HarvestError(f"identity.version '{identity_version}' is not '{IDENTITY_VERSION}'")
    identity_digest = _require_string(identity, "digest", "identity")
    if identity_digest != "sha256_over_canonical_json":
        raise HarvestError("identity.digest must be 'sha256_over_canonical_json'")
    declared_excludes = _require_string_array(
        identity, "partition_group_excludes", "identity"
    )
    if declared_excludes != PARTITION_GROUP_EXCLUDES:
        raise HarvestError(
            "identity.partition_group_excludes must be exactly "
            f"{list(PARTITION_GROUP_EXCLUDES)}: it is a contract statement, not a setting"
        )

    partitioning = _parse_partitioning(_require_table(document, "partitioning", "configuration"))
    solver = _parse_solver(_require_table(document, "solver", "configuration"))
    grid = _parse_grid(_require_table(document, "grid", "configuration"))
    surfaces = _parse_surfaces(_require_table(document, "surfaces", "configuration"))
    harvest = _parse_harvest(_require_table(document, "harvest", "configuration"))
    scenarios = _parse_scenarios(document, solver)

    if grid.boundary_exclusion_nodes < int(_pde.pde_regime_stencil_radius):
        raise HarvestError(
            "grid.boundary_exclusion_nodes must be at least the regime stencil radius "
            f"({int(_pde.pde_regime_stencil_radius)})"
        )
    if len(scenarios) < len(PARTITION_NAMES) * partitioning.minimum_groups_per_partition:
        raise HarvestError(
            f"{len(scenarios)} scenarios cannot fill {len(PARTITION_NAMES)} partitions with at "
            f"least {partitioning.minimum_groups_per_partition} group(s) each"
        )
    for role in surfaces.roles:
        if role != "base" and surfaces.volatility_bump is None:
            raise HarvestError(
                "surfaces.volatility_bump is required when a non-base role is declared"
            )
    if surfaces.volatility_bump is not None:
        for scenario in scenarios:
            if scenario.base_volatility - surfaces.volatility_bump <= 0.0:
                raise HarvestError(
                    f"scenario '{scenario.name}' has a sigma_down volatility that is not positive"
                )

    return HarvestConfig(
        study_name=_require_string(study, "name", "study"),
        study_status=_require_string(study, "status", "study"),
        engine=_require_string(study, "engine", "study"),
        price_units=_require_string(study, "price_units", "study"),
        greek_method=_require_string(study, "greek_method", "study"),
        curve_construction=_require_string(study, "curve_construction", "study"),
        identity_version=identity_version,
        identity_digest=identity_digest,
        partitioning=partitioning,
        solver=solver,
        grid=grid,
        surfaces=surfaces,
        harvest=harvest,
        scenarios=scenarios,
        source_name=source_name,
        raw_config_sha256=raw_config_sha256,
    )


def _parse_partitioning(table: Mapping[str, Any]) -> PartitionSettings:
    _reject_unknown(table, _PARTITIONING_KEYS, "partitioning")
    algorithm = _require_string(table, "algorithm", "partitioning")
    if algorithm != PARTITION_ALGORITHM:
        raise HarvestError(f"partitioning.algorithm '{algorithm}' is not '{PARTITION_ALGORITHM}'")
    weights_table = _require_table(table, "weights", "partitioning")
    declared = tuple(weights_table)
    if declared != PARTITION_NAMES:
        raise HarvestError(
            f"partitioning.weights must declare exactly {list(PARTITION_NAMES)} in that order"
        )
    weights = tuple(
        (name, _require_positive_float(weights_table, name, "partitioning.weights"))
        for name in declared
    )
    total = math.fsum(weight for _name, weight in weights)
    if abs(total - 1.0) > 1.0e-12:
        raise HarvestError(f"partitioning.weights must sum to 1.0, not {total!r}")
    return PartitionSettings(
        algorithm=algorithm,
        seed=_require_nonnegative_int(table, "seed", "partitioning"),
        minimum_groups_per_partition=_require_positive_int(
            table, "minimum_groups_per_partition", "partitioning"
        ),
        weights=weights,
    )


def _parse_solver(table: Mapping[str, Any]) -> SolverSettings:
    _reject_unknown(table, _SOLVER_KEYS, "solver")
    settlement = _require_string(table, "settlement", "solver")
    if settlement not in {"cash", "physical"}:
        raise HarvestError("solver.settlement must be 'cash' or 'physical'")
    return SolverSettings(
        spot_maximum_strike_multiple=_require_positive_float(
            table, "spot_maximum_strike_multiple", "solver"
        ),
        rannacher_steps=_require_nonnegative_int(table, "rannacher_steps", "solver"),
        psor_tolerance=_require_positive_float(table, "psor_tolerance", "solver"),
        psor_relaxation=_require_positive_float(table, "psor_relaxation", "solver"),
        psor_maximum_iterations=_require_positive_int(table, "psor_maximum_iterations", "solver"),
        settlement=settlement,
        contract_multiplier=_require_positive_float(table, "contract_multiplier", "solver"),
    )


def _parse_grid(table: Mapping[str, Any]) -> GridSettings:
    _reject_unknown(table, _GRID_KEYS, "grid")
    return GridSettings(
        spot_intervals=_require_positive_int(table, "spot_intervals", "grid"),
        time_steps=_require_positive_int(table, "time_steps", "grid"),
        boundary_exclusion_nodes=_require_positive_int(
            table, "boundary_exclusion_nodes", "grid"
        ),
    )


def _parse_surfaces(table: Mapping[str, Any]) -> SurfaceDesign:
    _reject_unknown(table, _SURFACES_KEYS, "surfaces")
    roles = _require_string_array(table, "roles", "surfaces")
    option_types = _require_string_array(table, "option_types", "surfaces")
    exercise_styles = _require_string_array(table, "exercise_styles", "surfaces")
    for name, values, allowed in (
        ("roles", roles, frozenset(SURFACE_ROLES)),
        ("option_types", option_types, OPTION_TYPES),
        ("exercise_styles", exercise_styles, EXERCISE_STYLES),
    ):
        if len(set(values)) != len(values):
            raise HarvestError(f"surfaces.{name} repeats a value")
        unknown = sorted(set(values) - allowed)
        if unknown:
            raise HarvestError(f"surfaces.{name} has unknown values: {', '.join(unknown)}")
    bump: float | None = None
    if "volatility_bump" in table:
        bump = _require_positive_float(table, "volatility_bump", "surfaces")
    return SurfaceDesign(
        roles=roles,
        option_types=option_types,
        exercise_styles=exercise_styles,
        volatility_bump=bump,
    )


def _parse_harvest(table: Mapping[str, Any]) -> HarvestRules:
    _reject_unknown(table, _HARVEST_KEYS, "harvest")
    node_source = _require_string(table, "node_source", "harvest")
    if node_source != NODE_SOURCE:
        raise HarvestError(
            f"harvest.node_source '{node_source}' is not '{NODE_SOURCE}': dataset harvesting "
            "never uses off-grid interpolation"
        )
    selection_rule = _require_string(table, "selection_rule", "harvest")
    if selection_rule != SELECTION_RULE:
        raise HarvestError(f"harvest.selection_rule '{selection_rule}' is not '{SELECTION_RULE}'")
    low = _require_positive_float(table, "moneyness_window_low", "harvest")
    high = _require_positive_float(table, "moneyness_window_high", "harvest")
    if not low < high:
        raise HarvestError("harvest.moneyness_window_low must be below moneyness_window_high")
    quota_table = _require_table(table, "quota", "harvest")
    missing = sorted(set(EXERCISE_STATES) - set(quota_table))
    if missing:
        raise HarvestError(f"harvest.quota must declare every exercise state; missing: {missing}")
    _reject_unknown(quota_table, frozenset(EXERCISE_STATES), "harvest.quota")
    quota = tuple(
        (state, _require_nonnegative_int(quota_table, state, "harvest.quota"))
        for state in EXERCISE_STATES
    )
    return HarvestRules(
        status=_require_string(table, "status", "harvest"),
        node_source=node_source,
        moneyness_window_low=low,
        moneyness_window_high=high,
        selection_rule=selection_rule,
        quota=quota,
    )


def _parse_scenarios(
    document: Mapping[str, Any], solver: SolverSettings
) -> tuple[Scenario, ...]:
    value = document.get("scenarios")
    if not isinstance(value, list) or not value:
        raise HarvestError("configuration.scenarios must be a non-empty array of tables")
    scenarios: list[Scenario] = []
    seen: set[str] = set()
    for index, entry in enumerate(value):
        if not isinstance(entry, Mapping):
            raise HarvestError(f"scenarios[{index}] must be a table")
        context = f"scenarios[{index}]"
        _reject_unknown(entry, _SCENARIO_KEYS, context)
        name = _require_string(entry, "name", context)
        if name in seen:
            raise HarvestError(f"scenario name '{name}' is declared more than once")
        seen.add(name)
        valuation_time = _require_nonnegative_float(entry, "valuation_time", context)
        expiry_time = _require_positive_float(entry, "expiry_time", context)
        if expiry_time <= valuation_time:
            raise HarvestError(f"{context}.expiry_time must exceed valuation_time")
        scenarios.append(
            Scenario(
                name=name,
                strike=_require_positive_float(entry, "strike", context),
                valuation_time=valuation_time,
                expiry_time=expiry_time,
                rate=_require_float(entry, "rate", context),
                base_volatility=_require_positive_float(entry, "base_volatility", context),
                continuous_carry=_require_float(entry, "continuous_carry", context),
                dividends=_parse_dividends(entry, context, valuation_time, expiry_time),
                settlement=solver.settlement,
                contract_multiplier=solver.contract_multiplier,
            )
        )
    return tuple(scenarios)


def _parse_dividends(
    entry: Mapping[str, Any], context: str, valuation_time: float, expiry_time: float
) -> tuple[tuple[float, float], ...]:
    value = entry.get("dividends")
    if not isinstance(value, list):
        raise HarvestError(f"{context}.dividends must be an array (possibly empty)")
    parsed: list[tuple[float, float]] = []
    previous = -math.inf
    for index, item in enumerate(value):
        if not isinstance(item, list) or len(item) != 2:
            raise HarvestError(f"{context}.dividends[{index}] must be [ex_time, amount]")
        ex_time = _as_finite_float(item[0], f"{context}.dividends[{index}].ex_time")
        amount = _as_finite_float(item[1], f"{context}.dividends[{index}].amount")
        if not valuation_time < ex_time < expiry_time:
            raise HarvestError(
                f"{context}.dividends[{index}].ex_time must lie strictly inside the valuation "
                "interval"
            )
        if ex_time <= previous:
            raise HarvestError(f"{context}.dividends must be strictly increasing in ex_time")
        if amount <= 0.0:
            raise HarvestError(f"{context}.dividends[{index}].amount must be positive")
        previous = ex_time
        parsed.append((ex_time, amount))
    return tuple(parsed)


# ---------------------------------------------------------------------------
# Planning: identities and partition assignment, both before any solve
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SurfacePlan:
    """One planned PDE solve, with its partition already fixed."""

    surface_id: str
    solver_input_id: str
    solver_input: SolverInputDescriptor
    partition_group_id: str
    partition: str
    scenario_name: str
    option_type: str
    exercise_style: str
    surface_role: str
    volatility: float
    settings_digest: str
    group_ordinal: int
    surface_ordinal: int


@dataclass(frozen=True, slots=True)
class GroupPlan:
    partition_group_id: str
    partition: str
    scenario: Scenario
    ordinal: int
    assignment_digest: str
    surfaces: tuple[SurfacePlan, ...]


@dataclass(frozen=True, slots=True)
class HarvestPlan:
    """The complete pre-solve design: every group assigned, no solve performed."""

    algorithm: str
    identity_version: str
    seed: int
    semantic_config_sha256: str
    partition_counts: tuple[tuple[str, int], ...]
    groups: tuple[GroupPlan, ...]
    settings: tuple[tuple[str, dict[str, Any]], ...]

    @property
    def surfaces(self) -> tuple[SurfacePlan, ...]:
        return tuple(surface for group in self.groups for surface in group.surfaces)

    def partition_of(self, partition_group_id: str) -> str:
        for group in self.groups:
            if group.partition_group_id == partition_group_id:
                return group.partition
        raise HarvestError(f"partition group '{partition_group_id}' is not in the plan")


def assignment_digest(partition_group_id: str, *, seed: int) -> str:
    """Return the seeded ordering digest of one candidate partition group.

    It depends on the group identity, the algorithm version and the seed, and on
    nothing else -- not on the group's position in the configuration, not on how
    many groups exist, and not on any solved value.
    """
    return _digest(
        "as",
        {
            "identity_version": IDENTITY_VERSION,
            "kind": "assignment",
            "algorithm": PARTITION_ALGORITHM,
            "seed": seed,
            "partition_group_id": partition_group_id,
        },
    )


def partition_counts(total: int, settings: PartitionSettings) -> tuple[tuple[str, int], ...]:
    """Return the largest-remainder group count per partition, minima enforced.

    Ties in the fractional remainder are broken by declared partition order, so
    the result depends only on ``total`` and the configuration.
    """
    if total < len(settings.weights) * settings.minimum_groups_per_partition:
        raise HarvestError(
            f"{total} candidate group(s) cannot fill {len(settings.weights)} partitions with at "
            f"least {settings.minimum_groups_per_partition} each"
        )
    exact = [(name, weight * total) for name, weight in settings.weights]
    counts = {name: math.floor(value) for name, value in exact}
    remainder = total - sum(counts.values())
    order = sorted(
        range(len(exact)),
        key=lambda index: (-(exact[index][1] - math.floor(exact[index][1])), index),
    )
    for position in range(remainder):
        counts[exact[order[position % len(order)]][0]] += 1

    minimum = settings.minimum_groups_per_partition
    names = [name for name, _weight in settings.weights]
    # Repair deficits deterministically: take one from the largest partition that
    # can spare it, ties broken by declared order. The total is at least the sum
    # of the minima, so this terminates.
    for name in names:
        while counts[name] < minimum:
            donors = [donor for donor in names if counts[donor] > minimum]
            if not donors:
                raise HarvestError("partition minima cannot be satisfied deterministically")
            donor = min(donors, key=lambda item: (-counts[item], names.index(item)))
            counts[donor] -= 1
            counts[name] += 1
    return tuple((name, counts[name]) for name in names)


def assign_partitions(
    partition_group_ids: Sequence[str], settings: PartitionSettings
) -> dict[str, str]:
    """Assign every candidate group to exactly one partition.

    The only inputs are the candidate identities and the configuration, so the
    result cannot depend on any solved value, on the order the candidates were
    supplied in, or on how the run is later batched.
    """
    occurrences: dict[str, int] = {}
    for group_id in partition_group_ids:
        occurrences[group_id] = occurrences.get(group_id, 0) + 1
    duplicates = sorted(key for key, count in occurrences.items() if count > 1)
    if duplicates:
        raise HarvestError(f"duplicate candidate partition group(s): {duplicates}")
    unique = sorted(occurrences)
    ordered = sorted(
        unique, key=lambda group_id: (assignment_digest(group_id, seed=settings.seed), group_id)
    )
    counts = partition_counts(len(ordered), settings)
    assignments: dict[str, str] = {}
    position = 0
    for name, count in counts:
        for group_id in ordered[position : position + count]:
            assignments[group_id] = name
        position += count
    if position != len(ordered):
        raise HarvestError("partition counts do not cover every candidate group")
    return assignments


def plan_harvest(config: HarvestConfig) -> HarvestPlan:
    """Build the whole design and assign partitions. **No PDE is solved here.**

    This function takes no solver argument and calls none: the impossibility of
    an outcome-dependent assignment is structural, not a matter of ordering
    statements inside one function.
    """
    settings_by_digest: dict[str, dict[str, Any]] = {}
    SurfaceCandidate = tuple[str, str, str, SolverInputDescriptor, str, str]
    candidates: list[tuple[Scenario, str, list[SurfaceCandidate]]] = []
    seen: dict[str, str] = {}
    seen_solver_inputs: dict[str, tuple[str, str]] = {}
    for scenario in config.scenarios:
        group_id = partition_group_identity(scenario)
        if group_id in seen:
            raise HarvestError(
                f"scenarios '{seen[group_id]}' and '{scenario.name}' are the same economic state "
                f"({group_id})"
            )
        seen[group_id] = scenario.name
        numerical = _numerical_settings(scenario, config)
        settings_digest = settings_identity(numerical)
        settings_by_digest.setdefault(settings_digest, dict(numerical))
        planned: list[SurfaceCandidate] = []
        for role in config.surfaces.roles:
            volatility = _role_volatility(scenario, role, config.surfaces)
            for option_type in config.surfaces.option_types:
                for exercise_style in config.surfaces.exercise_styles:
                    descriptor = _solver_input_descriptor(
                        scenario=scenario,
                        option_type=option_type,
                        exercise_style=exercise_style,
                        volatility=volatility,
                        numerical=numerical,
                    )
                    solver_id = solver_input_identity(descriptor)
                    if solver_id in seen_solver_inputs:
                        other_group, other_claim = seen_solver_inputs[solver_id]
                        claim = f"{scenario.name}/{role}/{option_type}/{exercise_style}"
                        raise HarvestError(
                            "duplicate actual PDE solver input "
                            f"'{solver_id}' claimed by '{other_claim}' in group "
                            f"'{other_group}' and '{claim}' in group '{group_id}'; "
                            "task 9C-C2a rejects aliases before partition assignment or solving"
                        )
                    seen_solver_inputs[solver_id] = (
                        group_id,
                        f"{scenario.name}/{role}/{option_type}/{exercise_style}",
                    )
                    planned.append(
                        (role, option_type, exercise_style, descriptor, solver_id, settings_digest)
                    )
        planned.sort(key=lambda item: item[4])
        candidates.append((scenario, group_id, planned))

    assignments = assign_partitions(
        [group_id for _scenario, group_id, _planned in candidates], config.partitioning
    )

    # Execution order is canonical, not configuration order: the plan is sorted
    # by partition, then by the group's own assignment digest. Reordering the
    # declared scenarios in the same parsed configuration therefore changes no
    # ordinal or execution order. Reordering the raw TOML deliberately changes
    # only exact-source provenance and report/manifest hashes, never rows or IDs.
    candidates.sort(
        key=lambda item: (
            PARTITION_NAMES.index(assignments[item[1]]),
            assignment_digest(item[1], seed=config.partitioning.seed),
            item[1],
        )
    )

    groups: list[GroupPlan] = []
    for ordinal, (scenario, group_id, planned) in enumerate(candidates):
        surfaces: list[SurfacePlan] = []
        for role, option_type, exercise_style, descriptor, solver_id, digest in planned:
            surfaces.append(
                SurfacePlan(
                    surface_id=solver_id,
                    solver_input_id=solver_id,
                    solver_input=descriptor,
                    partition_group_id=group_id,
                    partition=assignments[group_id],
                    scenario_name=scenario.name,
                    option_type=option_type,
                    exercise_style=exercise_style,
                    surface_role=role,
                    volatility=descriptor.volatility,
                    settings_digest=digest,
                    group_ordinal=ordinal,
                    surface_ordinal=len(surfaces),
                )
            )
        groups.append(
            GroupPlan(
                partition_group_id=group_id,
                partition=assignments[group_id],
                scenario=scenario,
                ordinal=ordinal,
                assignment_digest=assignment_digest(group_id, seed=config.partitioning.seed),
                surfaces=tuple(surfaces),
            )
        )
    return HarvestPlan(
        algorithm=config.partitioning.algorithm,
        identity_version=IDENTITY_VERSION,
        seed=config.partitioning.seed,
        semantic_config_sha256=config.semantic_config_sha256,
        partition_counts=partition_counts(len(groups), config.partitioning),
        groups=tuple(groups),
        settings=tuple(sorted(settings_by_digest.items())),
    )


def _role_volatility(scenario: Scenario, role: str, design: SurfaceDesign) -> float:
    """Return the volatility one surface role is actually solved at.

    ``sigma_down`` and ``sigma_up`` exist so a future vega triple already shares
    one partition group. **This module computes no vega from them.**
    """
    if role == "base":
        return scenario.base_volatility
    if design.volatility_bump is None:
        raise HarvestError(f"role '{role}' needs surfaces.volatility_bump")
    if role == "sigma_down":
        return scenario.base_volatility - design.volatility_bump
    if role == "sigma_up":
        return scenario.base_volatility + design.volatility_bump
    raise HarvestError(f"unknown surface role '{role}'")


def _numerical_settings(scenario: Scenario, config: HarvestConfig) -> dict[str, Any]:
    return {
        "spot_intervals": config.grid.spot_intervals,
        "time_steps": config.grid.time_steps,
        "spot_maximum": scenario.strike * config.solver.spot_maximum_strike_multiple,
        "rannacher_steps": config.solver.rannacher_steps,
        "psor_tolerance": config.solver.psor_tolerance,
        "psor_relaxation": config.solver.psor_relaxation,
        "psor_maximum_iterations": config.solver.psor_maximum_iterations,
        "boundary_exclusion_nodes": config.grid.boundary_exclusion_nodes,
    }


def _solver_input_descriptor(
    *,
    scenario: Scenario,
    option_type: str,
    exercise_style: str,
    volatility: float,
    numerical: Mapping[str, Any],
) -> SolverInputDescriptor:
    return SolverInputDescriptor(
        option_type=option_type,
        exercise_style=exercise_style,
        strike=scenario.strike,
        valuation_time=scenario.valuation_time,
        expiry_time=scenario.expiry_time,
        volatility=volatility,
        continuous_carry=scenario.continuous_carry,
        curve_times=scenario.curve_times,
        curve_log_discounts=scenario.curve_log_discounts,
        dividends=scenario.dividends,
        settlement=scenario.settlement,
        spot_intervals=int(numerical["spot_intervals"]),
        time_steps=int(numerical["time_steps"]),
        spot_maximum=float(numerical["spot_maximum"]),
        rannacher_steps=int(numerical["rannacher_steps"]),
        psor_tolerance=float(numerical["psor_tolerance"]),
        psor_relaxation=float(numerical["psor_relaxation"]),
        psor_maximum_iterations=int(numerical["psor_maximum_iterations"]),
        boundary_exclusion_nodes=int(numerical["boundary_exclusion_nodes"]),
    )


# ---------------------------------------------------------------------------
# Node selection
# ---------------------------------------------------------------------------


def evenly_spaced_selection(candidates: Sequence[int], quota: int) -> tuple[int, ...]:
    """Select at most ``quota`` candidates, evenly spaced by ascending index.

    The rule is predeclared and outcome-independent within a regime: it looks at
    positions, never at prices, Greeks or errors. With ``quota <= len``, the
    spacing is at least one position, so the selected indices are strictly
    increasing and no duplicate can be produced.
    """
    if quota < 0:
        raise HarvestError("a regime quota may not be negative")
    count = len(candidates)
    if quota == 0 or count == 0:
        return ()
    if quota >= count:
        return tuple(candidates)
    if quota == 1:
        return (candidates[(count - 1) // 2],)
    step = (count - 1) / (quota - 1)
    return tuple(candidates[math.floor(position * step)] for position in range(quota))


def deduplicate_node_indices(indices: Iterable[int]) -> tuple[tuple[int, ...], int]:
    """Return ``(unique_indices, rejected_duplicate_count)``, order preserved.

    Zero by construction under :data:`SELECTION_RULE`; kept as a live check so a
    future selection rule cannot quietly emit one node twice.
    """
    seen: set[int] = set()
    kept: list[int] = []
    rejected = 0
    for index in indices:
        if index in seen:
            rejected += 1
            continue
        seen.add(index)
        kept.append(index)
    return tuple(kept), rejected


@dataclass(frozen=True, slots=True)
class SurfaceHarvest:
    """Rows and accounting from one solved surface."""

    rows: tuple[dict[str, Any], ...]
    rejected: dict[str, int]
    candidates_by_state: dict[str, int]
    requested_by_state: dict[str, int]
    selected_by_state: dict[str, int]
    applicable_states: tuple[str, ...]
    shortfalls: tuple[dict[str, Any], ...]


def harvest_surface(
    surface: Mapping[str, Any], plan: SurfacePlan, scenario: Scenario, rules: HarvestRules
) -> SurfaceHarvest:
    """Harvest exact interior nodes of one solved surface.

    Structural admissibility comes from the surface contract and is copied, never
    recomputed or relaxed: ``exercise_state``, ``greek_eligible`` and
    ``greek_eligibility_reason`` are carried through as reported, and a node the
    engine refused as a Greek label is never promoted to an eligible one.
    """
    spots = surface["spot_nodes"]
    values = surface["values"]
    deltas = surface["deltas"]
    gammas = surface["gammas"]
    eligible = surface["greek_eligible"]
    states = surface["exercise_states"]
    reasons = surface["greek_eligibility_reasons"]
    intervals = int(surface["spot_intervals"])
    buffer = int(surface["boundary_exclusion_nodes"])
    if len(spots) != intervals + 1:
        raise HarvestError(
            f"surface '{plan.surface_id}' reports {len(spots)} nodes for {intervals} intervals"
        )

    rejected = {
        "truncation_endpoint": 0,
        "boundary_buffer": 0,
        "outside_moneyness_window": 0,
        "non_finite_price": 0,
        "quota": 0,
        "duplicate_node": 0,
    }
    admissible: dict[str, list[int]] = {state: [] for state in EXERCISE_STATES}
    for index in range(len(spots)):
        # Precedence is fixed and first-applicable, so every rejected node is
        # counted exactly once and the counters reconcile against the node total.
        if index == 0 or index == intervals:
            rejected["truncation_endpoint"] += 1
            continue
        if index < buffer or index > intervals - buffer:
            rejected["boundary_buffer"] += 1
            continue
        moneyness = spots[index] / scenario.strike
        if not rules.moneyness_window_low <= moneyness <= rules.moneyness_window_high:
            rejected["outside_moneyness_window"] += 1
            continue
        if not math.isfinite(values[index]):
            rejected["non_finite_price"] += 1
            continue
        state = states[index]
        if state not in admissible:
            raise HarvestError(f"surface '{plan.surface_id}' reports unknown state '{state}'")
        admissible[state].append(index)

    applicable = APPLICABLE_EXERCISE_STATES[plan.exercise_style]
    requested_by_state: dict[str, int] = {}
    selected_by_state: dict[str, int] = {}
    shortfalls: list[dict[str, Any]] = []
    chosen: list[int] = []
    for state in EXERCISE_STATES:
        quota = rules.quota_for(state) if state in applicable else 0
        candidates = admissible[state]
        if state not in applicable and candidates:
            raise HarvestError(
                f"surface '{plan.surface_id}' is {plan.exercise_style} and reports "
                f"{len(candidates)} '{state}' node(s), which that style cannot produce"
            )
        picked = evenly_spaced_selection(candidates, quota)
        unique, duplicates = deduplicate_node_indices(picked)
        rejected["duplicate_node"] += duplicates
        rejected["quota"] += len(candidates) - len(unique)
        requested_by_state[state] = quota
        selected_by_state[state] = len(unique)
        if len(unique) < quota:
            shortfalls.append(
                {
                    "partition": plan.partition,
                    "partition_group_id": plan.partition_group_id,
                    "surface_id": plan.surface_id,
                    "exercise_style": plan.exercise_style,
                    "option_type": plan.option_type,
                    "exercise_state": state,
                    "requested": quota,
                    "achieved": len(unique),
                    "shortfall": quota - len(unique),
                    "admissible_candidates": len(candidates),
                }
            )
        chosen.extend(unique)

    chosen.sort()
    rows = tuple(
        _build_row(
            index=index,
            surface=surface,
            plan=plan,
            scenario=scenario,
            spot=spots[index],
            value=values[index],
            delta=deltas[index],
            gamma=gammas[index],
            greek_eligible=bool(eligible[index]),
            exercise_state=states[index],
            reason=reasons[index],
        )
        for index in chosen
    )
    return SurfaceHarvest(
        rows=rows,
        rejected=rejected,
        candidates_by_state={state: len(admissible[state]) for state in EXERCISE_STATES},
        requested_by_state=requested_by_state,
        selected_by_state=selected_by_state,
        applicable_states=applicable,
        shortfalls=tuple(shortfalls),
    )


def _build_row(
    *,
    index: int,
    surface: Mapping[str, Any],
    plan: SurfacePlan,
    scenario: Scenario,
    spot: float,
    value: float,
    delta: float | None,
    gamma: float | None,
    greek_eligible: bool,
    exercise_state: str,
    reason: str,
) -> dict[str, Any]:
    # Invariants of the surface contract, asserted rather than assumed. A silent
    # promotion of an uncertified node to an eligible training label is exactly
    # the failure this module exists to make impossible.
    if exercise_state == "numerically_indifferent" and greek_eligible:
        raise HarvestError(
            f"surface '{plan.surface_id}' node {index} is numerically indifferent and Greek "
            "eligible; the surface contract refuses that combination"
        )
    if greek_eligible:
        if reason != "eligible":
            raise HarvestError(
                f"surface '{plan.surface_id}' node {index} is eligible with reason '{reason}'"
            )
        if delta is None or gamma is None:
            raise HarvestError(
                f"surface '{plan.surface_id}' node {index} is eligible with a missing derivative"
            )
        if not (math.isfinite(delta) and math.isfinite(gamma)):
            raise HarvestError(
                f"surface '{plan.surface_id}' node {index} is eligible with a non-finite "
                "derivative"
            )
    elif reason == "eligible":
        raise HarvestError(
            f"surface '{plan.surface_id}' node {index} is ineligible with reason 'eligible'"
        )
    if not math.isfinite(value):
        raise HarvestError(f"surface '{plan.surface_id}' node {index} has a non-finite price")

    # The immutable half of the row is *derived*, by the same function that
    # reconciliation uses to rebuild it. The returned surface supplies only the
    # numerical half below.
    immutable = expected_row_metadata(planned_surface_view(plan, scenario), index)
    if canonical_payload(immutable["spot"]) != canonical_payload(spot):
        raise HarvestError(
            f"surface '{plan.surface_id}' node {index} spot is not its planned exact grid node"
        )
    return {
        **immutable,
        "price": value,
        # The numerical derivatives are reported whatever their admissibility;
        # the three eligibility flags below, not the presence of a number, decide
        # what may be supervised.
        "delta": delta,
        "gamma": gamma,
        "exercise_state": exercise_state,
        "price_label_eligible": True,
        "delta_label_eligible": bool(greek_eligible and delta is not None),
        "gamma_label_eligible": bool(greek_eligible and gamma is not None),
        "greek_eligible": bool(greek_eligible),
        "greek_eligibility_reason": reason,
    }


ROW_COLUMNS: Final = (
    "schema_version",
    "partition",
    "partition_group_id",
    "surface_id",
    "row_id",
    "scenario_name",
    "surface_role",
    "node_index",
    "spot",
    "moneyness_spot_over_strike",
    "option_type",
    "exercise_style",
    "strike",
    "valuation_time",
    "expiry_time",
    "volatility",
    "base_volatility",
    "continuous_carry",
    "rate",
    "settlement",
    "contract_multiplier",
    "dividend_count",
    "price",
    "delta",
    "gamma",
    "exercise_state",
    "price_label_eligible",
    "delta_label_eligible",
    "gamma_label_eligible",
    "greek_eligible",
    "greek_eligibility_reason",
    "settings_digest",
    "spot_intervals",
    "time_steps",
    "spot_maximum",
    "spot_step",
    "boundary_exclusion_nodes",
)
"""Published row schema, in fixed column order."""


def check_unique_row_ids(rows: Sequence[Mapping[str, Any]]) -> None:
    """Fail loudly on any repeated ``row_id``: that would be an identity collision."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in seen:
            duplicates.append(row_id)
        seen.add(row_id)
    if duplicates:
        raise HarvestError(f"duplicate row_id(s): {sorted(set(duplicates))}")


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

Solver = Callable[..., Mapping[str, Any]]


def _chunked(items: Sequence[Any], size: int | None) -> Iterator[Sequence[Any]]:
    if size is None:
        yield items
        return
    if size <= 0:
        raise HarvestError("chunk_size must be positive")
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _solve_surface(
    plan: SurfacePlan, scenario: Scenario, config: HarvestConfig, solver: Solver
) -> Mapping[str, Any]:
    descriptor = plan.solver_input
    return solver(
        option_type=descriptor.option_type,
        exercise_style=descriptor.exercise_style,
        strike=descriptor.strike,
        valuation_time=descriptor.valuation_time,
        expiry_time=descriptor.expiry_time,
        volatility=descriptor.volatility,
        continuous_carry=descriptor.continuous_carry,
        curve_times=list(descriptor.curve_times),
        curve_log_discounts=list(descriptor.curve_log_discounts),
        dividends=[list(dividend) for dividend in descriptor.dividends],
        settlement=descriptor.settlement,
        contract_multiplier=scenario.contract_multiplier,
        spot_intervals=descriptor.spot_intervals,
        time_steps=descriptor.time_steps,
        spot_maximum=descriptor.spot_maximum,
        rannacher_steps=descriptor.rannacher_steps,
        psor_tolerance=descriptor.psor_tolerance,
        psor_relaxation=descriptor.psor_relaxation,
        psor_maximum_iterations=descriptor.psor_maximum_iterations,
        boundary_exclusion_nodes=descriptor.boundary_exclusion_nodes,
        query_spots=[],
    )


def _llround_positive(value: float) -> int:
    return math.floor(value + 0.5)


def _expected_spot_grid_payload(descriptor: Mapping[str, Any]) -> tuple[int, float, float, int]:
    """Reproduce the solver's node placement from a canonical descriptor payload."""
    spot_maximum = float(descriptor["spot_maximum"])
    strike = float(descriptor["strike"])
    target_step = spot_maximum / int(descriptor["spot_intervals"])
    strike_index = max(1, _llround_positive(strike / target_step))
    step = strike / strike_index
    intervals = math.ceil((spot_maximum / step) - 1.0e-9)
    if intervals <= strike_index:
        intervals = strike_index + 1
    maximum = intervals * step
    return intervals, maximum, step, strike_index


def _expected_time_grid_payload(descriptor: Mapping[str, Any]) -> tuple[tuple[float, ...], int]:
    """Reproduce the solver's time alignment from a canonical descriptor payload."""
    valuation_time = float(descriptor["valuation_time"])
    expiry_time = float(descriptor["expiry_time"])
    interior_curve = [
        float(value)
        for value in descriptor["curve_times"]
        if valuation_time < float(value) < expiry_time
    ]
    aligned = tuple(
        sorted(
            {
                valuation_time,
                expiry_time,
                *interior_curve,
                *(float(item[0]) for item in descriptor["dividends"]),
            }
        )
    )
    span = expiry_time - valuation_time
    total = 0
    for left, right in pairwise(aligned):
        total += max(1, _llround_positive(int(descriptor["time_steps"]) * (right - left) / span))
    return aligned, total


def _expected_spot_grid(descriptor: SolverInputDescriptor) -> tuple[int, float, float, int]:
    return _expected_spot_grid_payload(descriptor.payload())


def _expected_time_grid(descriptor: SolverInputDescriptor) -> tuple[tuple[float, ...], int]:
    return _expected_time_grid_payload(descriptor.payload())


def _require_result_equal(
    surface: Mapping[str, Any], key: str, expected: Any, surface_id: str
) -> None:
    if key not in surface:
        raise HarvestError(f"surface '{surface_id}' is missing required result field '{key}'")
    if canonical_payload(surface[key]) != canonical_payload(expected):
        raise HarvestError(
            f"surface '{surface_id}' result field '{key}' does not match its planned solver input"
        )


SURFACE_INPUT_KEY: Final = "surface_input"
"""Where the task 9C-C1 result carries its mandatory normalized input echo."""

SURFACE_INPUT_IDENTITY_FIELDS: Final = (
    "option_type",
    "exercise_style",
    "strike",
    "valuation_time",
    "expiry_time",
    "volatility",
    "continuous_carry",
    "curve_times",
    "curve_log_discounts",
    "dividends",
    "settlement",
    "spot_intervals",
    "time_steps",
    "spot_maximum",
    "rannacher_steps",
    "psor_tolerance",
    "psor_relaxation",
    "psor_maximum_iterations",
    "boundary_exclusion_nodes",
)
"""Echo fields that select the numerical solve and therefore form its identity."""

SURFACE_INPUT_METADATA_FIELDS: Final = ("contract_multiplier", "dividends_declared")
"""Echo fields carried for reporting that must never enter the identity.

``contract_multiplier`` never enters pricing arithmetic, so admitting it would
split one pricing state into two. ``dividends_declared`` distinguishes "no cash
dividends, stated" from an undeclared schedule; the solver refuses the latter
outright, so it is checked rather than hashed.
"""


def solver_input_id_from_echo(echo: Mapping[str, Any], *, context: str) -> str:
    """Rebuild the actual-solve identity from a returned input echo.

    The echo must carry exactly the declared fields: a missing one is a failure,
    not a field to skip, because skipping is precisely how an unchecked echo
    becomes decoration.
    """
    if not isinstance(echo, Mapping):
        raise HarvestError(f"{context} has no '{SURFACE_INPUT_KEY}' mapping")
    expected_keys = set(SURFACE_INPUT_IDENTITY_FIELDS) | set(SURFACE_INPUT_METADATA_FIELDS)
    missing = sorted(expected_keys - set(echo))
    if missing:
        raise HarvestError(f"{context} input echo is missing required field(s): {missing}")
    unknown = sorted(set(echo) - expected_keys)
    if unknown:
        raise HarvestError(f"{context} input echo has unknown field(s): {unknown}")
    return _digest(
        "si",
        {
            "identity_version": IDENTITY_VERSION,
            "kind": "solver_input",
            **{key: echo[key] for key in SURFACE_INPUT_IDENTITY_FIELDS},
        },
    )


def validate_surface_input_echo(
    surface: Mapping[str, Any], plan: SurfacePlan, *, contract_multiplier: float
) -> None:
    """Require the returned input echo to reconstruct the planned solver identity.

    The echo binds the API result to the inputs the solver accepted. It is
    **not** independent evidence that the numerical algorithm used them
    correctly; that remains established by the task 9C-C1 validation suite.
    """
    context = f"surface '{plan.surface_id}'"
    if SURFACE_INPUT_KEY not in surface:
        raise HarvestError(f"{context} result has no mandatory '{SURFACE_INPUT_KEY}' section")
    echo = surface[SURFACE_INPUT_KEY]
    returned_id = solver_input_id_from_echo(echo, context=context)

    # Field-by-field first, so a mismatch names the offending input rather than
    # only reporting that two digests differ.
    expected = plan.solver_input.payload()
    for key in SURFACE_INPUT_IDENTITY_FIELDS:
        if canonical_payload(echo[key]) != canonical_payload(expected[key]):
            raise HarvestError(
                f"{context} input echo field '{key}' does not match its planned solver input"
            )
    if returned_id != plan.solver_input_id:
        raise HarvestError(
            f"{context} input echo reconstructs '{returned_id}', not its planned solver_input_id"
        )

    if echo["dividends_declared"] is not True:
        raise HarvestError(f"{context} input echo reports an undeclared dividend schedule")
    if canonical_payload(echo["contract_multiplier"]) != canonical_payload(contract_multiplier):
        raise HarvestError(
            f"{context} input echo reports a contract_multiplier the plan did not declare"
        )


def validate_surface_result(
    surface: Mapping[str, Any], plan: SurfacePlan, *, contract_multiplier: float
) -> None:
    """Bind every exposed result field and exact node coordinate to the plan."""
    descriptor = plan.solver_input
    intervals, maximum, step, strike_index = _expected_spot_grid(descriptor)
    aligned_times, time_steps = _expected_time_grid(descriptor)
    required = {
        "valuation_time": descriptor.valuation_time,
        "spot_intervals": intervals,
        "spot_maximum": maximum,
        "spot_step": step,
        "strike_node_index": strike_index,
        "time_steps": time_steps,
        "rannacher_steps": descriptor.rannacher_steps,
        "psor_tolerance": descriptor.psor_tolerance,
        "psor_relaxation": descriptor.psor_relaxation,
        "boundary_exclusion_nodes": descriptor.boundary_exclusion_nodes,
        "regime_stencil_radius": int(_pde.pde_regime_stencil_radius),
        "backward_inductions": 1,
        "query_count": 0,
        "queries": [],
        "solver_status": "discrete_system_converged",
        "discretization_accuracy": "not_assessed",
        "aligned_times": list(aligned_times),
    }
    for key, expected in required.items():
        _require_result_equal(surface, key, expected, plan.surface_id)

    validate_surface_input_echo(surface, plan, contract_multiplier=contract_multiplier)

    events = surface.get("dividend_events")
    if not isinstance(events, Sequence) or isinstance(events, (str, bytes)):
        raise HarvestError(f"surface '{plan.surface_id}' has invalid dividend_events")
    returned_dividends = []
    for event in events:
        if not isinstance(event, Mapping):
            raise HarvestError(f"surface '{plan.surface_id}' has an invalid dividend event")
        returned_dividends.append([event.get("ex_time"), event.get("amount")])
    if canonical_payload(returned_dividends) != canonical_payload(
        [list(item) for item in descriptor.dividends]
    ):
        raise HarvestError(f"surface '{plan.surface_id}' dividend events do not match the plan")

    vector_keys = (
        "spot_nodes",
        "values",
        "deltas",
        "gammas",
        "greek_eligible",
        "exercise_states",
        "greek_eligibility_reasons",
    )
    for key in vector_keys:
        value = surface.get(key)
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            raise HarvestError(f"surface '{plan.surface_id}' has invalid vector '{key}'")
        if len(value) != intervals + 1:
            raise HarvestError(
                f"surface '{plan.surface_id}' vector '{key}' has {len(value)} entries; "
                f"expected {intervals + 1}"
            )
    spots = surface["spot_nodes"]
    for index, spot in enumerate(spots):
        expected = index * step
        if canonical_payload(spot) != canonical_payload(expected):
            raise HarvestError(
                f"surface '{plan.surface_id}' spot node {index} is not its planned exact grid node"
            )


def execute_plan(
    plan: HarvestPlan,
    config: HarvestConfig,
    *,
    solver: Solver = pde_valuation_surface,
    chunk_size: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Solve every planned surface and harvest its nodes.

    ``plan`` arrives already assigned. Chunking only groups the loop; it changes
    no identity, no assignment, and no output byte.
    """
    if plan.semantic_config_sha256 != config.semantic_config_sha256:
        raise HarvestError(
            "plan/config semantic digest mismatch; refusing to solve under stale identities"
        )
    group_records: list[dict[str, Any]] = []
    surface_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    shortfalls: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    attempted_surface_solve_count = 0
    successful_surface_solve_count = 0
    failed_surface_solve_count = 0

    for chunk in _chunked(plan.groups, chunk_size):
        for group in chunk:
            group_rows: list[dict[str, Any]] = []
            group_surfaces: list[dict[str, Any]] = []
            group_shortfalls: list[dict[str, Any]] = []
            group_failures: list[dict[str, Any]] = []
            group_attempted = 0
            for surface_plan in group.surfaces:
                attempted_surface_solve_count += 1
                group_attempted += 1
                stage = "pde_solve"
                try:
                    surface = _solve_surface(surface_plan, group.scenario, config, solver)
                    stage = "surface_validation"
                    validate_surface_result(
                        surface,
                        surface_plan,
                        contract_multiplier=group.scenario.contract_multiplier,
                    )
                    stage = "surface_harvest"
                    harvest = harvest_surface(
                        surface, surface_plan, group.scenario, config.harvest
                    )
                except Exception as error:
                    # Isolation is the point: one failed solve is recorded with
                    # its full group and surface identity and never aborts the
                    # run or contaminates another group.
                    group_failures.append(
                        {
                            "partition": surface_plan.partition,
                            "partition_group_id": surface_plan.partition_group_id,
                            "surface_id": surface_plan.surface_id,
                            "scenario_name": surface_plan.scenario_name,
                            "option_type": surface_plan.option_type,
                            "exercise_style": surface_plan.exercise_style,
                            "surface_role": surface_plan.surface_role,
                            "stage": stage,
                            "error_type": type(error).__name__,
                            "error_message": str(error),
                        }
                    )
                    failed_surface_solve_count += 1
                    continue
                successful_surface_solve_count += 1
                group_rows.extend(harvest.rows)
                group_surfaces.append(
                    _surface_record(surface_plan, group.scenario, surface, harvest)
                )
                group_shortfalls.extend(harvest.shortfalls)

            failed = bool(group_failures)
            failures.extend(group_failures)
            # A partially solved group is not a coherent design point, so its
            # rows are dropped whole rather than entering a partition as a
            # silently thinner state. The solves it did complete are still
            # reported as solves, with `retained = false`.
            for record in group_surfaces:
                record["retained"] = not failed
                record["retained_row_count"] = 0 if failed else record["harvested_row_count"]
            surface_records.extend(group_surfaces)
            if not failed:
                rows.extend(group_rows)
                shortfalls.extend(group_shortfalls)
            group_records.append(
                {
                    "partition_group_id": group.partition_group_id,
                    "partition": group.partition,
                    "scenario_name": group.scenario.name,
                    "ordinal": group.ordinal,
                    "status": "failed" if failed else "succeeded",
                    "planned_surface_count": len(group.surfaces),
                    "attempted_surface_solve_count": group_attempted,
                    "successful_surface_solve_count": len(group_surfaces),
                    "failed_surface_solve_count": len(group_failures),
                    "retained_surface_count": 0 if failed else len(group_surfaces),
                    "discarded_surface_count": len(group_surfaces) if failed else 0,
                    "harvested_row_count": 0 if failed else len(group_rows),
                    "planned_surface_ids": [item.surface_id for item in group.surfaces],
                    "successful_surface_ids": [item["surface_id"] for item in group_surfaces],
                    "failed_surface_ids": [item["surface_id"] for item in group_failures],
                }
            )

    execution_counts = {
        "planned_surface_count": len(plan.surfaces),
        "attempted_surface_solve_count": attempted_surface_solve_count,
        "successful_surface_solve_count": successful_surface_solve_count,
        "failed_surface_solve_count": failed_surface_solve_count,
    }
    check_unique_row_ids(rows)
    rows.sort(key=_row_sort_key)
    report = _build_report(
        plan=plan,
        config=config,
        rows=rows,
        group_records=group_records,
        surface_records=surface_records,
        failures=failures,
        shortfalls=shortfalls,
        execution_counts=execution_counts,
    )
    reconcile_report(report, rows, plan)
    return report, rows


def run_harvest(
    config: HarvestConfig,
    *,
    solver: Solver = pde_valuation_surface,
    chunk_size: int | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Plan (assign partitions), then solve and harvest."""
    return execute_plan(plan_harvest(config), config, solver=solver, chunk_size=chunk_size)


def _record_key(record: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        PARTITION_NAMES.index(str(record["partition"])),
        str(record["partition_group_id"]),
        str(record.get("surface_id", "")),
    )


def _row_sort_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        PARTITION_NAMES.index(str(row["partition"])),
        str(row["partition_group_id"]),
        str(row["surface_id"]),
        int(row["node_index"]),
    )


def _surface_record(
    plan: SurfacePlan,
    scenario: Scenario,
    surface: Mapping[str, Any],
    harvest: SurfaceHarvest,
) -> dict[str, Any]:
    return {
        # Derived by the same function reconciliation uses to rebuild it.
        **expected_surface_metadata(planned_surface_view(plan, scenario)),
        "status": "solved",
        "retained": True,
        "retained_row_count": len(harvest.rows),
        "backward_inductions": int(surface["backward_inductions"]),
        "solver_status": str(surface["solver_status"]),
        "discretization_accuracy": str(surface["discretization_accuracy"]),
        "exercise_classification_scale": float(surface["exercise_classification_scale"]),
        "maximum_relative_lcp_residual": float(surface["maximum_relative_lcp_residual"]),
        "node_count": len(surface["spot_nodes"]),
        "harvested_row_count": len(harvest.rows),
        "applicable_exercise_states": list(harvest.applicable_states),
        "admissible_by_exercise_state": dict(harvest.candidates_by_state),
        "requested_by_exercise_state": dict(harvest.requested_by_state),
        "selected_by_exercise_state": dict(harvest.selected_by_state),
        "rejected": dict(harvest.rejected),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _counter(values: Iterable[str], names: Sequence[str]) -> dict[str, int]:
    counts = dict.fromkeys(names, 0)
    for value in values:
        if value not in counts:
            raise HarvestError(f"unexpected reported value '{value}'")
        counts[value] += 1
    return counts


def _row_totals(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    eligible = sum(1 for row in rows if row["greek_eligible"])
    return {
        "raw_row_count": len(rows),
        "counts_by_exercise_state": _counter(
            (str(row["exercise_state"]) for row in rows), EXERCISE_STATES
        ),
        "counts_by_greek_eligibility": {
            "eligible": eligible,
            "ineligible": len(rows) - eligible,
        },
        "counts_by_greek_ineligibility_reason": _counter(
            (
                str(row["greek_eligibility_reason"])
                for row in rows
                if not row["greek_eligible"]
            ),
            GREEK_INELIGIBILITY_REASONS,
        ),
        "counts_by_label_eligibility": {
            "price": sum(1 for row in rows if row["price_label_eligible"]),
            "delta": sum(1 for row in rows if row["delta_label_eligible"]),
            "gamma": sum(1 for row in rows if row["gamma_label_eligible"]),
        },
    }


def _rejection_totals(surface_records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    keys = (
        "truncation_endpoint",
        "boundary_buffer",
        "outside_moneyness_window",
        "non_finite_price",
        "quota",
        "duplicate_node",
    )
    return {
        key: sum(int(record["rejected"][key]) for record in surface_records) for key in keys
    }


YIELD_DEFINITIONS: Final = {
    "raw_rows_per_group": (
        "Dataset expansion per independent design group: harvested rows divided "
        "by the number of distinct base economic states that produced rows. It "
        "measures how much labelled data one design point yields. It is NOT a "
        "measure of numerical work saved and must never be quoted as a "
        "computational speedup: a group may contain several separate solves."
    ),
    "raw_rows_per_attempted_surface_solve": (
        "Numerical-work reuse per attempted solve: harvested rows divided by "
        "every solver call attempted, including raising calls, invalid returned "
        "surfaces and successful solves later discarded with a failed group. This is the "
        "honest rows-per-solve "
        "multiplier -- how many rows one PDE solve bought on average across the "
        "whole run."
    ),
    "raw_rows_per_retained_surface": (
        "Numerical-work reuse per retained solve: harvested rows divided by the "
        "solves whose rows were kept. It equals raw_rows_per_attempted_surface_solve when "
        "nothing was discarded, and exceeds it otherwise. Reported beside the "
        "attempted-solve figure so the flattering denominator can never be "
        "quoted alone."
    ),
    "not_an_effective_sample_size": (
        "Neither ratio is a statistical effective sample size. Both are design "
        "and work accounting over correlated rows, and no estimator has been "
        "fitted to either."
    ),
}
"""What each published yield ratio does and does not mean."""


def _yield_ratio(numerator: int, denominator: int) -> dict[str, Any]:
    """Return a ratio that carries its own integer numerator and denominator.

    Every published ratio is auditable: ``value`` is exactly
    ``numerator / denominator`` in float64, both terms are integer counts that
    appear elsewhere in the same scope, and a zero denominator yields ``0.0``
    rather than a non-finite number that canonical JSON would reject.
    """
    return {
        "numerator": numerator,
        "denominator": denominator,
        "value": 0.0 if denominator == 0 else numerator / denominator,
    }


def _scope_totals(
    *,
    candidate_groups: Sequence[str],
    assigned_groups: Sequence[str],
    group_records: Sequence[Mapping[str, Any]],
    surface_records: Sequence[Mapping[str, Any]],
    failures: Sequence[Mapping[str, Any]],
    shortfalls: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    succeeded = [record for record in group_records if record["status"] == "succeeded"]
    failed = [record for record in group_records if record["status"] == "failed"]
    retained = [record for record in surface_records if record["retained"]]
    producing = sorted({str(row["partition_group_id"]) for row in rows})
    planned_surfaces = sum(int(record["planned_surface_count"]) for record in group_records)
    attempted = sum(
        int(record["attempted_surface_solve_count"]) for record in group_records
    )
    successful = len(surface_records)
    failed_surfaces = len(failures)
    totals: dict[str, Any] = {
        "candidate_group_count": len(candidate_groups),
        "assigned_group_count": len(assigned_groups),
        "successful_group_count": len(succeeded),
        "failed_group_count": len(failed),
        "independent_design_group_count": len(producing),
        "planned_surface_count": planned_surfaces,
        "attempted_surface_solve_count": attempted,
        "successful_surface_solve_count": successful,
        "failed_surface_solve_count": failed_surfaces,
        "retained_surface_count": len(retained),
        "discarded_surface_count": successful - len(retained),
        "backward_induction_count": sum(
            int(record["backward_inductions"]) for record in surface_records
        ),
        # Three separate denominators, always reported together so nobody can
        # quote rows-per-group as a computational multiplier.
        "raw_rows_per_group": _yield_ratio(len(rows), len(producing)),
        "raw_rows_per_attempted_surface_solve": _yield_ratio(len(rows), attempted),
        "raw_rows_per_retained_surface": _yield_ratio(len(rows), len(retained)),
        "quota_shortfall_count": len(shortfalls),
        "quota_shortfall_rows": sum(int(item["shortfall"]) for item in shortfalls),
        "rejected": _rejection_totals(surface_records),
    }
    totals.update(_row_totals(rows))
    return totals


def _integrity(
    rows: Sequence[Mapping[str, Any]], group_records: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    group_partitions: dict[str, set[str]] = {}
    for record in group_records:
        group_partitions.setdefault(str(record["partition_group_id"]), set()).add(
            str(record["partition"])
        )
    straddling = sorted(
        group_id for group_id, partitions in group_partitions.items() if len(partitions) > 1
    )

    rows_by_partition: dict[str, set[str]] = {name: set() for name in PARTITION_NAMES}
    groups_by_partition: dict[str, set[str]] = {name: set() for name in PARTITION_NAMES}
    for row in rows:
        rows_by_partition[str(row["partition"])].add(str(row["row_id"]))
        groups_by_partition[str(row["partition"])].add(str(row["partition_group_id"]))

    group_intersections = 0
    row_intersections = 0
    for left in range(len(PARTITION_NAMES)):
        for right in range(left + 1, len(PARTITION_NAMES)):
            a, b = PARTITION_NAMES[left], PARTITION_NAMES[right]
            group_intersections += len(groups_by_partition[a] & groups_by_partition[b])
            row_intersections += len(rows_by_partition[a] & rows_by_partition[b])

    row_ids = [str(row["row_id"]) for row in rows]
    group_ids = [str(record["partition_group_id"]) for record in group_records]
    return {
        "duplicate_partition_group_count": len(group_ids) - len(set(group_ids)),
        "duplicate_row_id_count": len(row_ids) - len(set(row_ids)),
        "groups_in_more_than_one_partition": straddling,
        "cross_partition_group_intersection_count": group_intersections,
        "cross_partition_row_intersection_count": row_intersections,
        "distinct_economic_states_per_partition": {
            name: len(groups_by_partition[name]) for name in PARTITION_NAMES
        },
    }


def _scenario_metadata(scenario: Scenario) -> dict[str, Any]:
    """Reporting-only scenario facts a row echoes but no identity consumes.

    They are published so that every immutable row field is derivable from the
    plan alone, which is what lets reconciliation rebuild a row rather than
    trust it.
    """
    return {
        "scenario_name": scenario.name,
        "rate": scenario.rate,
        "base_volatility": scenario.base_volatility,
        "contract_multiplier": scenario.contract_multiplier,
    }


def _published_surface(surface: SurfacePlan) -> dict[str, Any]:
    return {
        "surface_id": surface.surface_id,
        "solver_input_id": surface.solver_input_id,
        "solver_input": surface.solver_input.payload(),
        "surface_role": surface.surface_role,
        "settings_digest": surface.settings_digest,
    }


def planned_surface_view(surface: SurfacePlan, scenario: Scenario) -> dict[str, Any]:
    """One surface's immutable planned facts, in exactly the published shape.

    ``execute_plan`` builds this from live plan objects and ``_plan_indexes``
    rebuilds the identical shape from ``report.plan``, so the *same* derivation
    functions serve both the writing and the reconciling path.
    """
    return {
        **_published_surface(surface),
        "partition_group_id": surface.partition_group_id,
        "partition": surface.partition,
        "scenario_metadata": _scenario_metadata(scenario),
    }


def _published_plan(plan: HarvestPlan) -> dict[str, Any]:
    return {
        "identity_version": plan.identity_version,
        "semantic_config_sha256": plan.semantic_config_sha256,
        "groups": [
            {
                "partition_group_id": group.partition_group_id,
                "partition_group_payload": _partition_group_payload(group.scenario),
                "partition": group.partition,
                "scenario_name": group.scenario.name,
                "scenario_metadata": _scenario_metadata(group.scenario),
                "surface_ids": [surface.surface_id for surface in group.surfaces],
                "surfaces": [_published_surface(surface) for surface in group.surfaces],
            }
            for group in plan.groups
        ],
    }


# ---------------------------------------------------------------------------
# Canonical expected metadata: one derivation, used to write and to reconcile
# ---------------------------------------------------------------------------

SURFACE_IMMUTABLE_FIELDS: Final = (
    "surface_id",
    "solver_input_id",
    "partition_group_id",
    "partition",
    "scenario_name",
    "surface_role",
    "option_type",
    "exercise_style",
    "strike",
    "volatility",
    "settings_digest",
    "solver_input",
)
"""Surface-record fields the plan alone determines."""

ROW_IMMUTABLE_COLUMNS: Final = (
    "schema_version",
    "partition",
    "partition_group_id",
    "surface_id",
    "row_id",
    "scenario_name",
    "surface_role",
    "node_index",
    "spot",
    "moneyness_spot_over_strike",
    "option_type",
    "exercise_style",
    "strike",
    "valuation_time",
    "expiry_time",
    "volatility",
    "base_volatility",
    "continuous_carry",
    "rate",
    "settlement",
    "contract_multiplier",
    "dividend_count",
    "settings_digest",
    "spot_intervals",
    "time_steps",
    "spot_maximum",
    "spot_step",
    "boundary_exclusion_nodes",
)
"""Row columns the plan and the node index alone determine."""

ROW_NUMERICAL_COLUMNS: Final = (
    "price",
    "delta",
    "gamma",
    "exercise_state",
    "price_label_eligible",
    "delta_label_eligible",
    "gamma_label_eligible",
    "greek_eligible",
    "greek_eligibility_reason",
)
"""Row columns only the validated solve can supply."""


def expected_surface_metadata(view: Mapping[str, Any]) -> dict[str, Any]:
    """Derive one surface record's immutable metadata from the planned view."""
    descriptor = view["solver_input"]
    if not isinstance(descriptor, Mapping):
        raise HarvestError("planned surface view has no solver descriptor")
    metadata = view["scenario_metadata"]
    if not isinstance(metadata, Mapping):
        raise HarvestError("planned surface view has no scenario metadata")
    return {
        "surface_id": str(view["surface_id"]),
        "solver_input_id": str(view["solver_input_id"]),
        "partition_group_id": str(view["partition_group_id"]),
        "partition": str(view["partition"]),
        "scenario_name": str(metadata["scenario_name"]),
        "surface_role": str(view["surface_role"]),
        "option_type": str(descriptor["option_type"]),
        "exercise_style": str(descriptor["exercise_style"]),
        "strike": float(descriptor["strike"]),
        "volatility": float(descriptor["volatility"]),
        "settings_digest": str(view["settings_digest"]),
        "solver_input": dict(descriptor),
    }


def expected_row_metadata(view: Mapping[str, Any], node_index: int) -> dict[str, Any]:
    """Derive one row's immutable metadata from the planned view and node index.

    Nothing here reads the returned surface. The spot is the exact grid node the
    plan implies, and :func:`validate_surface_result` has already required the
    returned grid to be bitwise that grid, so deriving rather than copying is
    safe *and* is what makes an altered row detectable.
    """
    descriptor = view["solver_input"]
    metadata = view["scenario_metadata"]
    surface_id = str(view["surface_id"])
    strike = float(descriptor["strike"])
    intervals, maximum, step, _strike_index = _expected_spot_grid_payload(descriptor)
    _aligned, time_steps = _expected_time_grid_payload(descriptor)
    if not 0 <= node_index <= intervals:
        raise HarvestError(f"node index {node_index} is outside surface '{surface_id}'")
    spot = node_index * step
    return {
        "schema_version": ROW_SCHEMA_VERSION,
        "partition": str(view["partition"]),
        "partition_group_id": str(view["partition_group_id"]),
        "surface_id": surface_id,
        "row_id": row_identity(surface_id=surface_id, node_index=node_index),
        "scenario_name": str(metadata["scenario_name"]),
        "surface_role": str(view["surface_role"]),
        "node_index": node_index,
        "spot": spot,
        "moneyness_spot_over_strike": spot / strike,
        "option_type": str(descriptor["option_type"]),
        "exercise_style": str(descriptor["exercise_style"]),
        "strike": strike,
        "valuation_time": float(descriptor["valuation_time"]),
        "expiry_time": float(descriptor["expiry_time"]),
        "volatility": float(descriptor["volatility"]),
        "base_volatility": float(metadata["base_volatility"]),
        "continuous_carry": float(descriptor["continuous_carry"]),
        "rate": float(metadata["rate"]),
        "settlement": str(descriptor["settlement"]),
        "contract_multiplier": float(metadata["contract_multiplier"]),
        "dividend_count": len(descriptor["dividends"]),
        "settings_digest": str(view["settings_digest"]),
        "spot_intervals": intervals,
        "time_steps": time_steps,
        "spot_maximum": maximum,
        "spot_step": step,
        "boundary_exclusion_nodes": int(descriptor["boundary_exclusion_nodes"]),
    }


def _build_report(
    *,
    plan: HarvestPlan,
    config: HarvestConfig,
    rows: Sequence[dict[str, Any]],
    group_records: Sequence[dict[str, Any]],
    surface_records: Sequence[dict[str, Any]],
    failures: Sequence[dict[str, Any]],
    shortfalls: Sequence[dict[str, Any]],
    execution_counts: Mapping[str, int],
) -> dict[str, Any]:
    # Every published list is emitted in canonical order, so no configuration
    # ordering can reach an output byte.
    group_records = sorted(group_records, key=_record_key)
    surface_records = sorted(surface_records, key=_record_key)
    failures = sorted(failures, key=_record_key)
    shortfalls = sorted(
        shortfalls,
        key=lambda item: (*_record_key(item), EXERCISE_STATES.index(str(item["exercise_state"]))),
    )
    candidates = [group.partition_group_id for group in plan.groups]
    if execution_counts != {
        "planned_surface_count": len(plan.surfaces),
        "attempted_surface_solve_count": len(surface_records) + len(failures),
        "successful_surface_solve_count": len(surface_records),
        "failed_surface_solve_count": len(failures),
    }:
        raise HarvestError("execution counters do not match surface and failure records")
    per_partition: dict[str, Any] = {}
    for name in PARTITION_NAMES:
        partition_groups = [g for g in group_records if g["partition"] == name]
        partition_surfaces = [s for s in surface_records if s["partition"] == name]
        partition_rows = [r for r in rows if r["partition"] == name]
        per_partition[name] = _scope_totals(
            candidate_groups=[g["partition_group_id"] for g in partition_groups],
            assigned_groups=[g["partition_group_id"] for g in partition_groups],
            group_records=partition_groups,
            surface_records=partition_surfaces,
            failures=[f for f in failures if f["partition"] == name],
            shortfalls=[s for s in shortfalls if s["partition"] == name],
            rows=partition_rows,
        )

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "row_schema_version": ROW_SCHEMA_VERSION,
        "study": {
            "name": config.study_name,
            "status": config.study_status,
            "task": "9C-C2a",
            "engine": config.engine,
            "price_units": config.price_units,
            "greek_method": config.greek_method,
            "curve_construction": config.curve_construction,
        },
        "identity": {
            "version": IDENTITY_VERSION,
            "digest": "sha256_over_canonical_json",
            "partition_group_excludes": list(PARTITION_GROUP_EXCLUDES),
            "surface_id_is_solver_input_id": True,
            "surface_id_includes": [
                "strike",
                "option_type",
                "exercise_style",
                "valuation_time",
                "expiry_time",
                "curves",
                "dividends",
                "continuous_carry",
                "settlement",
                "volatility_actually_solved",
                "numerical_grid_and_solver_settings",
            ],
            "surface_id_excludes": [
                "partition_group_id",
                "scenario_name",
                "surface_role",
                "reporting_only_contract_multiplier",
                "descriptive_metadata",
            ],
            "row_id_includes": ["surface_id", "node_index"],
            "canonical_representation": (
                "sorted-key compact JSON with .17g tagged floats; independent of hash(), "
                "dictionary order, locale, platform float repr, paths and insertion order; "
                "negative zero normalized to positive zero"
            ),
        },
        "partitioning": {
            "algorithm": plan.algorithm,
            "seed": plan.seed,
            "partition_names": list(PARTITION_NAMES),
            "weights": {name: weight for name, weight in config.partitioning.weights},
            "minimum_groups_per_partition": config.partitioning.minimum_groups_per_partition,
            "target_group_counts": {name: count for name, count in plan.partition_counts},
            "assignment_precedes_solving": True,
            "assignment_inputs": (
                "candidate partition_group_id digests, the algorithm version and the seed; no "
                "solved value, no input order and no batch size"
            ),
            "incremental": False,
            "actual_solve_alias_policy": (
                "reject_before_partition_assignment; task 9C-C2b may introduce explicit reuse"
            ),
        },
        "harvest_rules": {
            "status": config.harvest.status,
            "node_source": config.harvest.node_source,
            "selection_rule": config.harvest.selection_rule,
            "selection_is_outcome_independent": True,
            "moneyness_window": [
                config.harvest.moneyness_window_low,
                config.harvest.moneyness_window_high,
            ],
            "quota_per_surface_by_exercise_state": {
                state: quota for state, quota in config.harvest.quota
            },
            "surface_roles": list(config.surfaces.roles),
            "option_types": list(config.surfaces.option_types),
            "exercise_styles": list(config.surfaces.exercise_styles),
            "vega_computed": False,
        },
        "determinism": {
            "byte_identity_preconditions": list(BYTE_IDENTITY_PRECONDITIONS),
            "same_bytes_different_filename": (
                "semantic_config_sha256, raw_config_sha256, every identity, every partition "
                "assignment and rows.csv are unchanged; config_name differs, so report.json and "
                "the manifest that pins it differ. That is recorded provenance, not "
                "non-determinism."
            ),
        },
        "plan": _published_plan(plan),
        "numerical_settings": {digest: values for digest, values in plan.settings},
        "totals": _scope_totals(
            candidate_groups=candidates,
            assigned_groups=candidates,
            group_records=group_records,
            surface_records=surface_records,
            failures=failures,
            shortfalls=shortfalls,
            rows=rows,
        ),
        "partition_totals": per_partition,
        "groups": list(group_records),
        "surfaces": list(surface_records),
        "failures": list(failures),
        "quota_shortfalls": list(shortfalls),
        "integrity": _integrity(rows, group_records),
        "interpretation": {
            "rows_within_a_group_are_correlated": ROWS_CORRELATED_STATEMENT,
            "independent_design_group_count": DESIGN_GROUP_STATEMENT,
            "yield_definitions": dict(YIELD_DEFINITIONS),
            "not_a_label_policy": (
                "This run selects no label policy and validates none. Task 9C-B remains "
                "no_policy_selected and is not revisited here."
            ),
            "not_a_production_dataset": (
                "Quotas, windows and grids here are exploratory demonstration values."
            ),
        },
        "provenance": _provenance(config),
    }


def _provenance(config: HarvestConfig) -> dict[str, Any]:
    module_path = Path(__file__)
    return {
        "config_name": config.source_name,
        "raw_config_sha256": config.raw_config_sha256,
        "semantic_config_sha256": config.semantic_config_sha256,
        "runner_name": module_path.name,
        "runner_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "pde_header_sha256": str(_pde.__pde_header_sha256__),
        "pde_source_sha256": str(_pde.__pde_source_sha256__),
        "pde_implementation_sha256": str(_pde.__pde_implementation_sha256__),
        "data_provenance": "fixed_synthetic_public_configuration_only",
        "private_market_data_used": False,
    }


def _reconcile_yields(scope: Mapping[str, Any], context: str) -> None:
    """Fail unless every yield ratio matches the integer counts of its own scope."""
    expected = {
        "raw_rows_per_group": ("raw_row_count", "independent_design_group_count"),
        "raw_rows_per_attempted_surface_solve": (
            "raw_row_count",
            "attempted_surface_solve_count",
        ),
        "raw_rows_per_retained_surface": ("raw_row_count", "retained_surface_count"),
    }
    for name, (numerator_key, denominator_key) in expected.items():
        ratio = scope[name]
        numerator = int(scope[numerator_key])
        denominator = int(scope[denominator_key])
        if int(ratio["numerator"]) != numerator:
            raise HarvestError(
                f"{context}.{name}.numerator does not equal {numerator_key}"
            )
        if int(ratio["denominator"]) != denominator:
            raise HarvestError(
                f"{context}.{name}.denominator does not equal {denominator_key}"
            )
        recomputed = 0.0 if denominator == 0 else numerator / denominator
        if float(ratio["value"]) != recomputed:
            raise HarvestError(f"{context}.{name}.value does not equal its own quotient")
    if int(scope["retained_surface_count"]) + int(scope["discarded_surface_count"]) != int(
        scope["successful_surface_solve_count"]
    ):
        raise HarvestError(
            f"{context} retained and discarded surfaces do not sum to successful solves"
        )


def _reconcile_solve_counts(scope: Mapping[str, Any], context: str) -> None:
    planned = int(scope["planned_surface_count"])
    attempted = int(scope["attempted_surface_solve_count"])
    successful = int(scope["successful_surface_solve_count"])
    failed = int(scope["failed_surface_solve_count"])
    retained = int(scope["retained_surface_count"])
    discarded = int(scope["discarded_surface_count"])
    if planned != attempted:
        raise HarvestError(f"{context} planned surfaces do not equal attempted solves")
    if attempted != successful + failed:
        raise HarvestError(f"{context} successful and failed solves do not equal attempts")
    if successful != retained + discarded:
        raise HarvestError(f"{context} retained and discarded surfaces do not equal successes")


def _unique_records(
    records: Sequence[Mapping[str, Any]], key: str, context: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for record in records:
        identity = str(record[key])
        if identity in indexed:
            raise HarvestError(f"duplicate {context} '{identity}'")
        indexed[identity] = record
    return indexed


def _plan_indexes(
    published_plan: Mapping[str, Any],
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    groups_value = published_plan.get("groups")
    if not isinstance(groups_value, list):
        raise HarvestError("report plan.groups must be a list")
    groups = _unique_records(groups_value, "partition_group_id", "planned group")
    surfaces: dict[str, Mapping[str, Any]] = {}
    for group_id, group in groups.items():
        partition = str(group["partition"])
        if partition not in PARTITION_NAMES:
            raise HarvestError(f"planned group '{group_id}' has an unknown partition")
        payload = group["partition_group_payload"]
        if not isinstance(payload, Mapping) or _digest("pg", dict(payload)) != group_id:
            raise HarvestError(f"planned group '{group_id}' does not match its canonical payload")
        metadata = group.get("scenario_metadata")
        if not isinstance(metadata, Mapping):
            raise HarvestError(f"planned group '{group_id}' has no scenario metadata")
        if set(metadata) != {"scenario_name", "rate", "base_volatility", "contract_multiplier"}:
            raise HarvestError(f"planned group '{group_id}' scenario metadata has the wrong shape")
        if str(metadata["scenario_name"]) != str(group["scenario_name"]):
            raise HarvestError(f"planned group '{group_id}' scenario names disagree")
        # The base volatility appears in the hashed group payload as well, so the
        # reporting copy cannot drift away from the identity-bearing one.
        if canonical_payload(metadata["base_volatility"]) != canonical_payload(
            payload["base_volatility"]
        ):
            raise HarvestError(f"planned group '{group_id}' base volatilities disagree")
        surface_values = group.get("surfaces")
        if not isinstance(surface_values, list):
            raise HarvestError(f"planned group '{group_id}' surfaces must be a list")
        declared_ids = [str(value) for value in group.get("surface_ids", [])]
        actual_ids: list[str] = []
        for surface in surface_values:
            if not isinstance(surface, Mapping):
                raise HarvestError(f"planned group '{group_id}' has an invalid surface")
            surface_id = str(surface["surface_id"])
            if surface_id in surfaces:
                raise HarvestError(f"duplicate planned surface '{surface_id}'")
            descriptor = surface["solver_input"]
            if not isinstance(descriptor, Mapping):
                raise HarvestError(f"planned surface '{surface_id}' has no solver descriptor")
            expected = _digest(
                "si",
                {
                    "identity_version": IDENTITY_VERSION,
                    "kind": "solver_input",
                    **dict(descriptor),
                },
            )
            if surface_id != expected or str(surface["solver_input_id"]) != expected:
                raise HarvestError(
                    f"planned surface '{surface_id}' does not match its actual solver input"
                )
            surfaces[surface_id] = {
                **surface,
                "partition_group_id": group_id,
                "partition": partition,
                "scenario_metadata": dict(metadata),
            }
            actual_ids.append(surface_id)
        if declared_ids != actual_ids:
            raise HarvestError(f"planned group '{group_id}' surface_ids do not match its surfaces")
    return groups, surfaces


def _validate_row_numerics(row: Mapping[str, Any]) -> None:
    """Check the solve-supplied half of one row for internal consistency."""
    row_id = str(row["row_id"])
    state = str(row["exercise_state"])
    reason = str(row["greek_eligibility_reason"])
    eligible = bool(row["greek_eligible"])
    if state not in EXERCISE_STATES:
        raise HarvestError(f"row '{row_id}' has unknown exercise state '{state}'")
    if reason not in (*GREEK_INELIGIBILITY_REASONS, "eligible"):
        raise HarvestError(f"row '{row_id}' has unknown eligibility reason '{reason}'")
    if eligible != (reason == "eligible"):
        raise HarvestError(f"row '{row_id}' eligibility disagrees with its reason")
    if state == "numerically_indifferent" and eligible:
        raise HarvestError(f"row '{row_id}' is numerically indifferent and Greek eligible")
    if bool(row["price_label_eligible"]) is not True:
        raise HarvestError(f"row '{row_id}' is published without a usable price label")
    if not isinstance(row["price"], float) or not math.isfinite(float(row["price"])):
        raise HarvestError(f"row '{row_id}' has a non-finite price while price-eligible")
    for name, flag in (("delta", "delta_label_eligible"), ("gamma", "gamma_label_eligible")):
        value = row[name]
        if bool(row[flag]):
            if not eligible:
                raise HarvestError(f"row '{row_id}' has an eligible {name} on an ineligible node")
            if value is None or not math.isfinite(float(value)):
                raise HarvestError(f"row '{row_id}' has a non-finite eligible {name}")
        elif eligible and value is not None:
            raise HarvestError(f"row '{row_id}' is Greek eligible but refuses its own {name}")
        elif value is not None and not math.isfinite(float(value)):
            raise HarvestError(f"row '{row_id}' reports a non-finite {name}")


def _recomputed_surface_regimes(
    surface_rows: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    counts = dict.fromkeys(EXERCISE_STATES, 0)
    for row in surface_rows:
        counts[str(row["exercise_state"])] += 1
    return counts


def reconcile_report(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    plan: HarvestPlan | None = None,
) -> None:
    """Recompute every identity, membership and count from plan, records and rows."""
    published_plan = report.get("plan")
    if not isinstance(published_plan, Mapping):
        raise HarvestError("report has no immutable published plan")
    if str(published_plan.get("identity_version")) != IDENTITY_VERSION:
        raise HarvestError("report plan has the wrong identity version")
    if str(published_plan.get("semantic_config_sha256")) != str(
        report.get("provenance", {}).get("semantic_config_sha256")
    ):
        raise HarvestError("published plan and provenance semantic digests disagree")
    if plan is not None and canonical_payload(published_plan) != canonical_payload(
        _published_plan(plan)
    ):
        raise HarvestError("report plan does not match the immutable execution plan")
    groups, planned_surfaces = _plan_indexes(published_plan)
    group_records = _unique_records(report["groups"], "partition_group_id", "group record")
    surface_records = _unique_records(report["surfaces"], "surface_id", "surface record")
    failures = _unique_records(report["failures"], "surface_id", "failure record")

    if set(group_records) != set(groups):
        raise HarvestError("group records do not match the planned groups")
    if set(surface_records) & set(failures):
        raise HarvestError("a planned surface is both successful and failed")
    if set(surface_records) | set(failures) != set(planned_surfaces):
        raise HarvestError("successful and failed records do not cover every planned surface")

    successes_by_group = {group_id: [] for group_id in groups}
    failures_by_group = {group_id: [] for group_id in groups}
    for surface_id, record in surface_records.items():
        expected = planned_surfaces[surface_id]
        group_id = str(expected["partition_group_id"])
        # Every duplicated identity-linked field, rebuilt from the plan and
        # compared as a whole rather than spot-checked.
        derived = expected_surface_metadata(expected)
        for key in SURFACE_IMMUTABLE_FIELDS:
            if key not in record:
                raise HarvestError(f"surface '{surface_id}' record is missing '{key}'")
            if canonical_payload(record[key]) != canonical_payload(derived[key]):
                raise HarvestError(
                    f"surface '{surface_id}' record field '{key}' does not match the plan"
                )
        if str(record["status"]) != "solved":
            raise HarvestError(f"surface '{surface_id}' record has a non-solved status")
        successes_by_group[group_id].append(surface_id)
    for surface_id, record in failures.items():
        expected = planned_surfaces[surface_id]
        group_id = str(expected["partition_group_id"])
        if str(record["partition_group_id"]) != group_id:
            raise HarvestError(f"failure '{surface_id}' belongs to the wrong group")
        if str(record["partition"]) != str(expected["partition"]):
            raise HarvestError(f"failure '{surface_id}' belongs to the wrong partition")
        failures_by_group[group_id].append(surface_id)

    rows_by_surface = {surface_id: [] for surface_id in surface_records}
    rows_by_group = {group_id: [] for group_id in groups}
    row_ids: set[str] = set()
    for row in rows:
        row_id = str(row["row_id"])
        if row_id in row_ids:
            raise HarvestError(f"duplicate row '{row_id}'")
        row_ids.add(row_id)
        surface_id = str(row["surface_id"])
        if surface_id not in surface_records:
            raise HarvestError(f"row '{row_id}' names an unknown or failed surface")
        expected = planned_surfaces[surface_id]
        group_id = str(expected["partition_group_id"])
        # Rebuild every immutable column from the plan and the node index, then
        # compare the whole set. A regenerated hash cannot survive this: the
        # expected value comes from the plan, never from the row.
        if set(row) != set(ROW_COLUMNS):
            raise HarvestError(f"row '{row_id}' does not carry exactly the published schema")
        derived = expected_row_metadata(expected, int(row["node_index"]))
        for key in ROW_IMMUTABLE_COLUMNS:
            if canonical_payload(row[key]) != canonical_payload(derived[key]):
                raise HarvestError(
                    f"row '{row_id}' field '{key}' does not match the plan and node index"
                )
        _validate_row_numerics(row)
        if not bool(surface_records[surface_id]["retained"]):
            raise HarvestError(f"discarded surface '{surface_id}' contributes a row")
        rows_by_surface[surface_id].append(row)
        rows_by_group[group_id].append(row)

    for group_id, planned_group in groups.items():
        record = group_records[group_id]
        planned_ids = [str(value) for value in planned_group["surface_ids"]]
        failed = bool(failures_by_group[group_id])
        expected_status = "failed" if failed else "succeeded"
        if str(record["status"]) != expected_status:
            raise HarvestError(f"group '{group_id}' has the wrong status")
        successful_ids = sorted(successes_by_group[group_id])
        failed_ids = sorted(failures_by_group[group_id])
        expected_counts = {
            "planned_surface_count": len(planned_ids),
            "attempted_surface_solve_count": len(planned_ids),
            "successful_surface_solve_count": len(successful_ids),
            "failed_surface_solve_count": len(failed_ids),
            "retained_surface_count": 0 if failed else len(successful_ids),
            "discarded_surface_count": len(successful_ids) if failed else 0,
            "harvested_row_count": len(rows_by_group[group_id]),
        }
        for key, expected in expected_counts.items():
            if int(record[key]) != expected:
                raise HarvestError(f"group '{group_id}' has the wrong {key}")
        if sorted(str(value) for value in record["planned_surface_ids"]) != sorted(planned_ids):
            raise HarvestError(f"group '{group_id}' has the wrong planned surface identities")
        if sorted(str(value) for value in record["successful_surface_ids"]) != successful_ids:
            raise HarvestError(f"group '{group_id}' has the wrong successful surface identities")
        if sorted(str(value) for value in record["failed_surface_ids"]) != failed_ids:
            raise HarvestError(f"group '{group_id}' has the wrong failed surface identities")
        for surface_id in successful_ids:
            retained = bool(surface_records[surface_id]["retained"])
            if retained == failed:
                raise HarvestError(
                    f"surface '{surface_id}' retention disagrees with its group status"
                )

    for surface_id, record in surface_records.items():
        selected = sum(int(value) for value in record["selected_by_exercise_state"].values())
        # The rejected tally is a recorded surface audit: the nodes it counts
        # were discarded and are not republished, so it is reconciled as integer
        # accounting against the node total, not independently reconstructed.
        rejected = sum(int(value) for value in record["rejected"].values())
        if selected + rejected != int(record["node_count"]):
            raise HarvestError(f"surface '{surface_id}' node accounting does not reconcile")
        if selected != int(record["harvested_row_count"]):
            raise HarvestError(f"surface '{surface_id}' selected rows do not reconcile")
        surface_rows = rows_by_surface[surface_id]
        retained_rows = len(surface_rows)
        if retained_rows != int(record["retained_row_count"]):
            raise HarvestError(f"surface '{surface_id}' retained rows do not reconcile")
        if not bool(record["retained"]) and retained_rows:
            raise HarvestError(f"discarded surface '{surface_id}' contributes rows")
        if bool(record["retained"]):
            # Selected counts are reconciled directly against the rows that
            # actually belong to this surface, per regime, not merely in total.
            if retained_rows != selected:
                raise HarvestError(
                    f"surface '{surface_id}' selected count does not equal its actual rows"
                )
            recomputed_regimes = _recomputed_surface_regimes(surface_rows)
            declared_regimes = {
                state: int(value)
                for state, value in record["selected_by_exercise_state"].items()
            }
            if recomputed_regimes != declared_regimes:
                raise HarvestError(
                    f"surface '{surface_id}' selected_by_exercise_state does not match its rows"
                )
            indices = sorted(int(row["node_index"]) for row in surface_rows)
            if len(set(indices)) != len(indices):
                raise HarvestError(f"surface '{surface_id}' repeats a harvested node index")
            node_count = int(record["node_count"])
            if indices and (indices[0] < 0 or indices[-1] >= node_count):
                raise HarvestError(f"surface '{surface_id}' harvested an out-of-range node")
            for state, quota in record["requested_by_exercise_state"].items():
                if recomputed_regimes[state] > int(quota):
                    raise HarvestError(
                        f"surface '{surface_id}' harvested more '{state}' rows than requested"
                    )

    group_record_values = list(group_records.values())
    surface_record_values = list(surface_records.values())
    failure_values = list(failures.values())
    shortfalls = report["quota_shortfalls"]
    candidates = list(groups)
    recomputed_totals = _scope_totals(
        candidate_groups=candidates,
        assigned_groups=candidates,
        group_records=group_record_values,
        surface_records=surface_record_values,
        failures=failure_values,
        shortfalls=shortfalls,
        rows=rows,
    )
    if canonical_payload(report["totals"]) != canonical_payload(recomputed_totals):
        raise HarvestError("report totals do not match recomputed totals")
    _reconcile_yields(recomputed_totals, "totals")
    _reconcile_solve_counts(recomputed_totals, "totals")

    for name in PARTITION_NAMES:
        partition_groups = [record for record in group_record_values if record["partition"] == name]
        partition_surfaces = [
            record for record in surface_record_values if record["partition"] == name
        ]
        partition_failures = [record for record in failure_values if record["partition"] == name]
        partition_rows = [row for row in rows if row["partition"] == name]
        partition_shortfalls = [item for item in shortfalls if item["partition"] == name]
        recomputed = _scope_totals(
            candidate_groups=[record["partition_group_id"] for record in partition_groups],
            assigned_groups=[record["partition_group_id"] for record in partition_groups],
            group_records=partition_groups,
            surface_records=partition_surfaces,
            failures=partition_failures,
            shortfalls=partition_shortfalls,
            rows=partition_rows,
        )
        if canonical_payload(report["partition_totals"][name]) != canonical_payload(recomputed):
            raise HarvestError(f"partition_totals.{name} do not match actual records and rows")
        _reconcile_yields(recomputed, f"partition_totals.{name}")
        _reconcile_solve_counts(recomputed, f"partition_totals.{name}")

    recomputed_integrity = _integrity(rows, group_record_values)
    if canonical_payload(report["integrity"]) != canonical_payload(recomputed_integrity):
        raise HarvestError("reported integrity counters do not match recomputed integrity")
    if any(
        int(recomputed_integrity[key])
        for key in (
            "duplicate_partition_group_count",
            "duplicate_row_id_count",
            "cross_partition_group_intersection_count",
            "cross_partition_row_intersection_count",
        )
    ) or recomputed_integrity["groups_in_more_than_one_partition"]:
        raise HarvestError("recomputed partition integrity is not clean")


# ---------------------------------------------------------------------------
# Publication
# ---------------------------------------------------------------------------

PUBLISHED_FILES: Final = ("report.json", "rows.csv")
MANIFEST_NAME: Final = "manifest.json"


def write_outputs(
    report: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    output_directory: Path,
    *,
    overwrite: bool = False,
) -> list[Path]:
    """Publish deterministic outputs, manifest last.

    Every payload is rendered in memory before anything is written, so a failure
    in rendering publishes nothing at all. ``manifest.json`` carries the SHA-256
    of every other file and is written last, so a run interrupted mid-publication
    leaves a directory that ``verify_publication`` rejects rather than one that
    reads as a completed dataset.
    """
    reconcile_report(report, rows)
    payloads = {
        "report.json": _canonical_json(report).encode("utf-8"),
        "rows.csv": rows_csv(rows).encode("utf-8"),
    }
    manifest = {
        "schema_version": "pde-surface-harvest-manifest/2",
        "publication_complete": True,
        "row_count": len(rows),
        "files": {
            name: {
                "sha256": hashlib.sha256(payload).hexdigest(),
                "bytes": len(payload),
            }
            for name, payload in sorted(payloads.items())
        },
    }
    payloads[MANIFEST_NAME] = _canonical_json(manifest).encode("utf-8")

    if output_directory.exists() and not output_directory.is_dir():
        raise HarvestError(f"output path '{output_directory}' is not a directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    # The manifest is deliberately last.
    for name in (*PUBLISHED_FILES, MANIFEST_NAME):
        destination = output_directory / name
        if destination.exists() and not overwrite:
            raise HarvestError(f"refusing to overwrite existing output '{destination}'")
        _write_atomic(destination, payloads[name])
        written.append(destination)
    return written


def verify_publication(directory: Path) -> None:
    """Reject a publication that is incomplete, stale or semantically inconsistent."""
    manifest_path = directory / MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise HarvestError(f"cannot read '{manifest_path}': {error}") from error
    if manifest.get("publication_complete") is not True:
        raise HarvestError(f"'{manifest_path}' does not record a complete publication")
    if manifest.get("schema_version") != "pde-surface-harvest-manifest/2":
        raise HarvestError(f"'{manifest_path}' has the wrong schema version")
    declared = manifest.get("files")
    if not isinstance(declared, Mapping) or set(declared) != set(PUBLISHED_FILES):
        raise HarvestError(f"'{manifest_path}' does not declare exactly {list(PUBLISHED_FILES)}")
    payloads: dict[str, bytes] = {}
    for name, record in sorted(declared.items()):
        try:
            payload = (directory / name).read_bytes()
        except OSError as error:
            raise HarvestError(f"cannot read published file '{name}': {error}") from error
        if hashlib.sha256(payload).hexdigest() != record["sha256"]:
            raise HarvestError(f"published file '{name}' does not match its manifest digest")
        if len(payload) != int(record["bytes"]):
            raise HarvestError(f"published file '{name}' does not match its manifest size")
        payloads[name] = payload
    try:
        report = json.loads(payloads["report.json"])
        rows = _parse_rows_csv(payloads["rows.csv"].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError, csv.Error, ValueError) as error:
        raise HarvestError(f"published payload cannot be parsed: {error}") from error
    if int(manifest.get("row_count", -1)) != len(rows):
        raise HarvestError("manifest row_count does not match rows.csv")
    reconcile_report(report, rows)


BYTE_IDENTITY_PRECONDITIONS: Final = (
    "identical configuration TOML bytes",
    "identical source-name provenance (the same config_name)",
    "identical runner code and compiled engine",
    "differences confined to harmless candidate ordering or chunk size",
)
"""Everything that must hold before two runs are required to match byte for byte.

The source *name* is a precondition, not an afterthought: `config_name` is
recorded provenance, so the same bytes loaded from a differently named file
produce the same semantic digest, identities, assignments and `rows.csv`, and the
same `raw_config_sha256` -- but a different `report.json`, and therefore a
different `manifest.json`. That is a provenance difference, not a determinism
failure, and :func:`verify_semantic_identity` is the check for it.
"""


def verify_semantic_identity(left: Path, right: Path) -> None:
    """Require two publications to agree on everything except source provenance.

    Use this when the same configuration bytes were loaded under different
    filenames. ``rows.csv`` must still match byte for byte, and the report must
    match once the recorded ``config_name`` is set aside.
    """
    reports: list[dict[str, Any]] = []
    for directory in (left, right):
        verify_publication(directory)
        try:
            reports.append(json.loads((directory / "report.json").read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as error:
            raise HarvestError(f"cannot read report in '{directory}': {error}") from error
    if (left / "rows.csv").read_bytes() != (right / "rows.csv").read_bytes():
        raise HarvestError("semantically identical runs disagree on rows.csv")
    stripped: list[dict[str, Any]] = []
    for report in reports:
        copied = json.loads(json.dumps(report))
        copied["provenance"].pop("config_name", None)
        stripped.append(copied)
    if canonical_payload(stripped[0]) != canonical_payload(stripped[1]):
        raise HarvestError("semantically identical runs disagree outside source provenance")
    if str(reports[0]["provenance"]["raw_config_sha256"]) != str(
        reports[1]["provenance"]["raw_config_sha256"]
    ):
        raise HarvestError("semantically identical runs disagree on their raw config digest")


def verify_byte_identity(left: Path, right: Path) -> None:
    """Require every published artifact to be byte-identical between two runs.

    Only valid under every entry of :data:`BYTE_IDENTITY_PRECONDITIONS`. In
    particular the two runs must record the same ``config_name``; for identical
    bytes under a different filename use :func:`verify_semantic_identity`.
    """
    for name in (*PUBLISHED_FILES, MANIFEST_NAME):
        try:
            left_bytes = (left / name).read_bytes()
            right_bytes = (right / name).read_bytes()
        except OSError as error:
            raise HarvestError(f"cannot compare artifact '{name}': {error}") from error
        if left_bytes != right_bytes:
            raise HarvestError(f"byte-identity repeat differs for '{name}'")


def rows_csv(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render the published rows in the fixed column order."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(ROW_COLUMNS)
    for row in rows:
        writer.writerow([_cell(row[column]) for column in ROW_COLUMNS])
    return buffer.getvalue()


_ROW_INTEGER_COLUMNS: Final = frozenset(
    {"node_index", "dividend_count", "spot_intervals", "time_steps", "boundary_exclusion_nodes"}
)
_ROW_FLOAT_COLUMNS: Final = frozenset(
    {
        "spot",
        "moneyness_spot_over_strike",
        "strike",
        "valuation_time",
        "expiry_time",
        "volatility",
        "base_volatility",
        "continuous_carry",
        "rate",
        "contract_multiplier",
        "price",
        "delta",
        "gamma",
        "spot_maximum",
        "spot_step",
    }
)
_ROW_BOOLEAN_COLUMNS: Final = frozenset(
    {
        "price_label_eligible",
        "delta_label_eligible",
        "gamma_label_eligible",
        "greek_eligible",
    }
)


def _parse_rows_csv(payload: str) -> list[dict[str, Any]]:
    reader = csv.DictReader(io.StringIO(payload))
    if tuple(reader.fieldnames or ()) != ROW_COLUMNS:
        raise HarvestError("rows.csv columns do not match the row schema")
    rows: list[dict[str, Any]] = []
    for raw in reader:
        row: dict[str, Any] = {}
        for key in ROW_COLUMNS:
            value = raw[key]
            if key in _ROW_INTEGER_COLUMNS:
                row[key] = int(value)
            elif key in _ROW_FLOAT_COLUMNS:
                row[key] = None if value == "" else float(value)
            elif key in _ROW_BOOLEAN_COLUMNS:
                if value not in {"true", "false"}:
                    raise ValueError(f"row boolean '{key}' is not true or false")
                row[key] = value == "true"
            else:
                row[key] = value
        rows.append(row)
    return rows


def _cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return f"{value:.17g}"
    return str(value)


def _canonical_json(document: Any) -> str:
    try:
        return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as error:
        raise HarvestError(f"document is not finite canonical JSON: {error}") from error


def _write_atomic(destination: Path, payload: bytes) -> None:
    temporary: Path | None = None
    try:
        descriptor, name = tempfile.mkstemp(
            prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
        )
        temporary = Path(name)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise HarvestError(f"cannot publish '{destination}': {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="group batch size; it changes no assignment, no identity and no output byte",
    )
    parser.add_argument(
        "--verify-identical-to",
        type=Path,
        default=None,
        help="after publication, require all outputs to match this directory byte for byte",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    try:
        config = load_harvest_config(arguments.config)
        report, rows = run_harvest(config, chunk_size=arguments.chunk_size)
        paths = write_outputs(
            report, rows, arguments.output_directory, overwrite=arguments.overwrite
        )
        verify_publication(arguments.output_directory)
        if arguments.verify_identical_to is not None:
            verify_byte_identity(arguments.verify_identical_to, arguments.output_directory)
    except HarvestError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    totals = report["totals"]
    print(
        f"wrote {len(paths)} artifacts; {totals['raw_row_count']} raw rows from "
        f"{totals['attempted_surface_solve_count']} attempted surface solves over "
        f"{totals['independent_design_group_count']} independent design groups"
    )
    return 0


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise HarvestError(f"{context} has unknown keys: {', '.join(unknown)}")


def _require_table(table: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, Mapping):
        raise HarvestError(f"{context}.{key} must be a table")
    return value


def _require_string(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise HarvestError(f"{context}.{key} must be a non-empty string")
    return value


def _require_string_array(table: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = table.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise HarvestError(f"{context}.{key} must be a non-empty string array")
    return tuple(value)


def _as_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarvestError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise HarvestError(f"{context} must be finite")
    return result


def _require_float(table: Mapping[str, Any], key: str, context: str) -> float:
    if key not in table:
        raise HarvestError(f"{context}.{key} is required")
    return _as_finite_float(table[key], f"{context}.{key}")


def _require_positive_float(table: Mapping[str, Any], key: str, context: str) -> float:
    value = _require_float(table, key, context)
    if value <= 0.0:
        raise HarvestError(f"{context}.{key} must be positive")
    return value


def _require_nonnegative_float(table: Mapping[str, Any], key: str, context: str) -> float:
    value = _require_float(table, key, context)
    if value < 0.0:
        raise HarvestError(f"{context}.{key} must be non-negative")
    return value


def _require_positive_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise HarvestError(f"{context}.{key} must be a positive integer")
    return value


def _require_nonnegative_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise HarvestError(f"{context}.{key} must be a non-negative integer")
    return value


if __name__ == "__main__":
    sys.exit(main())
