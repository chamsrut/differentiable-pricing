"""Controlled task 9C-B PDE label-policy pilot.

The expensive pilot is deliberately separate from CI.  CI exercises the
configuration contract, centered-difference and Richardson formulas, units,
threshold logic, deterministic serialization, and regular/stress separation
with synthetic values.  Only this runner calls the compiled PDE oracle for the
full versioned design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import itertools
import json
import math
import os
import platform
import resource
import sys
import tempfile
import time
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

from differentiable_pricing import _pde, pde_price

SCHEMA_VERSION: Final = "pde-label-policy-pilot/1"
REPORT_SCHEMA_VERSION: Final = "pde-label-policy-report/1"
ENGINE_NAME: Final = "dp::finite_difference_price/v1"
CLASSIFICATIONS: Final = frozenset({"regular", "stress"})
OPTION_TYPES: Final = frozenset({"call", "put"})
EXERCISE_STYLES: Final = frozenset({"european", "american"})
GRID_ROLES: Final = frozenset({"candidate", "reference", "anchor"})
RAW_GRID_NAMES: Final = (
    "grid_800x400",
    "grid_1600x800",
    "grid_3200x1600",
    "grid_6400x3200",
)
RICHARDSON_CANDIDATE: Final = "richardson_800x400_1600x800"
RICHARDSON_REFERENCE: Final = "richardson_1600x800_3200x1600"
RICHARDSON_ANCHOR: Final = "richardson_3200x1600_6400x3200"

_TOP_LEVEL_KEYS: Final = frozenset(
    {"schema_version", "study", "solver", "grids", "bumps", "richardson", "criteria", "cases"}
)
_STUDY_KEYS: Final = frozenset(
    {
        "name",
        "engine",
        "price_units",
        "greek_method",
        "curve_construction",
        "selection_order",
        "anchor_cases",
        "projected_label_count",
        "projected_worker_counts",
    }
)
_SOLVER_KEYS: Final = frozenset(
    {
        "spot_maximum",
        "rannacher_steps",
        "psor_tolerance",
        "psor_relaxation",
        "psor_maximum_iterations",
        "settlement",
        "contract_multiplier",
    }
)
_GRID_KEYS: Final = frozenset({"name", "spot_intervals", "time_steps", "role"})
_BUMP_KEYS: Final = frozenset(
    {"spot", "primary_spot", "volatility", "primary_volatility"}
)
_RICHARDSON_KEYS: Final = frozenset(
    {
        "coarse_to_fine_ratio",
        "assumed_order",
        "minimum_supported_observed_order",
        "maximum_supported_observed_order",
    }
)
_CRITERIA_KEYS: Final = frozenset(
    {
        "price_absolute_error",
        "delta_absolute_error",
        "gamma_absolute_error",
        "vega_absolute_error_per_unit_volatility",
        "delta_bump_variation",
        "gamma_bump_variation",
        "vega_bump_variation_per_unit_volatility",
        "shape_tolerance",
    }
)
_CASE_KEYS: Final = frozenset(
    {
        "name",
        "classification",
        "description",
        "option_type",
        "exercise_style",
        "spot",
        "strike",
        "expiry_time",
        "rate",
        "continuous_carry",
        "volatility",
        "dividends",
    }
)


class LabelPolicyError(RuntimeError):
    """Raised when the pilot contract, execution, or publication fails."""


@dataclass(frozen=True, slots=True)
class Grid:
    name: str
    spot_intervals: int
    time_steps: int
    role: str


@dataclass(frozen=True, slots=True)
class PilotCase:
    name: str
    classification: str
    description: str
    option_type: str
    exercise_style: str
    spot: float
    strike: float
    expiry_time: float
    rate: float
    continuous_carry: float
    volatility: float
    dividends: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class SolverSettings:
    spot_maximum: float
    rannacher_steps: int
    psor_tolerance: float
    psor_relaxation: float
    psor_maximum_iterations: int
    settlement: str
    contract_multiplier: float


@dataclass(frozen=True, slots=True)
class BumpSettings:
    spot: tuple[float, ...]
    primary_spot: float
    volatility: tuple[float, ...]
    primary_volatility: float


@dataclass(frozen=True, slots=True)
class RichardsonSettings:
    coarse_to_fine_ratio: int
    assumed_order: float
    minimum_supported_observed_order: float
    maximum_supported_observed_order: float


@dataclass(frozen=True, slots=True)
class Criteria:
    price_absolute_error: float
    delta_absolute_error: float
    gamma_absolute_error: float
    vega_absolute_error_per_unit_volatility: float
    delta_bump_variation: float
    gamma_bump_variation: float
    vega_bump_variation_per_unit_volatility: float
    shape_tolerance: float


@dataclass(frozen=True, slots=True)
class PilotConfig:
    name: str
    selection_order: tuple[str, ...]
    anchor_cases: tuple[str, ...]
    projected_label_count: int
    projected_worker_counts: tuple[int, ...]
    solver: SolverSettings
    grids: tuple[Grid, ...]
    bumps: BumpSettings
    richardson: RichardsonSettings
    criteria: Criteria
    cases: tuple[PilotCase, ...]
    source_name: str
    source_sha256: str


@dataclass(frozen=True, slots=True)
class BumpRequest:
    name: str
    spot: float
    volatility: float
    kind: str
    size: float | None
    direction: int


def centered_delta(up: float, down: float, bump: float) -> float:
    """Return dV/dS from a centered absolute spot bump."""
    _require_positive_finite_number(bump, "delta bump")
    return (up - down) / (2.0 * bump)


def centered_gamma(up: float, center: float, down: float, bump: float) -> float:
    """Return d2V/dS2 from a centered absolute spot bump."""
    _require_positive_finite_number(bump, "gamma bump")
    return (up - 2.0 * center + down) / (bump * bump)


def centered_vega(up: float, down: float, bump: float) -> float:
    """Return dV/dsigma per unit absolute volatility."""
    _require_positive_finite_number(bump, "vega bump")
    return (up - down) / (2.0 * bump)


def vega_per_volatility_point(vega_per_unit: float) -> float:
    """Convert per-unit absolute-volatility vega to one percentage point."""
    return vega_per_unit / 100.0


def richardson_extrapolate(coarse: float, fine: float, assumed_order: float = 2.0) -> float:
    """Extrapolate a factor-two pair: fine + (fine - coarse)/(2**p - 1)."""
    _require_positive_finite_number(assumed_order, "Richardson assumed order")
    denominator = 2.0**assumed_order - 1.0
    return fine + (fine - coarse) / denominator


def observed_order(coarse: float, medium: float, fine: float) -> float | None:
    """Return log2(|coarse-medium| / |medium-fine|), or null if undefined."""
    numerator = abs(coarse - medium)
    denominator = abs(medium - fine)
    if numerator == 0.0 or denominator == 0.0:
        return None
    value = math.log2(numerator / denominator)
    return value if math.isfinite(value) else None


def threshold_pass(error: float, threshold: float) -> bool:
    """Inclusive predeclared absolute-error threshold."""
    return math.isfinite(error) and error <= threshold


def select_accuracy_policy(
    selection_order: Sequence[str], outcomes: Mapping[str, Mapping[str, Any]]
) -> str:
    """Select the first policy whose every regular row passed, or refuse."""
    for policy in selection_order:
        outcome = outcomes[policy]
        if bool(outcome["all_regular_cases_pass"]):
            return policy
    return "no_policy_selected"


def load_pilot_config(path: Path | str) -> PilotConfig:
    """Load and strictly validate the versioned pilot configuration."""
    source = Path(path)
    try:
        raw = source.read_bytes()
    except OSError as error:
        raise LabelPolicyError(f"cannot read configuration '{source}': {error}") from error
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise LabelPolicyError(f"configuration is not valid UTF-8 TOML: {error}") from error
    return parse_pilot_config(
        document, source_name=source.name, source_sha256=hashlib.sha256(raw).hexdigest()
    )


def parse_pilot_config(
    document: Mapping[str, Any], *, source_name: str, source_sha256: str
) -> PilotConfig:
    """Validate an already parsed task 9C-B configuration."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")
    if _require_string(document, "schema_version", "top-level") != SCHEMA_VERSION:
        raise LabelPolicyError(f"schema_version must be '{SCHEMA_VERSION}'")

    study = _require_table(document, "study", "top-level")
    _reject_unknown(study, _STUDY_KEYS, "[study]")
    if _require_string(study, "engine", "[study]") != ENGINE_NAME:
        raise LabelPolicyError(f"[study].engine must be '{ENGINE_NAME}'")
    if _require_string(study, "price_units", "[study]") != "currency_per_share":
        raise LabelPolicyError("[study].price_units must be 'currency_per_share'")
    if _require_string(study, "greek_method", "[study]") != "centered_price_bumps":
        raise LabelPolicyError("[study].greek_method must be 'centered_price_bumps'")
    if (
        _require_string(study, "curve_construction", "[study]")
        != "flat_continuously_compounded_from_case_rate"
    ):
        raise LabelPolicyError("unsupported [study].curve_construction")
    selection_order = _require_string_array(study, "selection_order", "[study]")
    expected_candidates = {RAW_GRID_NAMES[0], RAW_GRID_NAMES[1], RICHARDSON_CANDIDATE}
    if set(selection_order) != expected_candidates or len(selection_order) != 3:
        raise LabelPolicyError("[study].selection_order must contain each candidate exactly once")
    anchor_cases = _require_string_array(study, "anchor_cases", "[study]")
    if not 1 <= len(anchor_cases) <= 3:
        raise LabelPolicyError("[study].anchor_cases must contain one to three cases")
    projected_label_count = _require_positive_int(study, "projected_label_count", "[study]")
    worker_counts = _require_int_array(study, "projected_worker_counts", "[study]")
    if worker_counts != (8, 16):
        raise LabelPolicyError("[study].projected_worker_counts must be [8, 16]")

    solver_table = _require_table(document, "solver", "top-level")
    _reject_unknown(solver_table, _SOLVER_KEYS, "[solver]")
    solver = SolverSettings(
        spot_maximum=_require_positive_float(solver_table, "spot_maximum", "[solver]"),
        rannacher_steps=_require_nonnegative_int(solver_table, "rannacher_steps", "[solver]"),
        psor_tolerance=_require_positive_float(solver_table, "psor_tolerance", "[solver]"),
        psor_relaxation=_require_positive_float(solver_table, "psor_relaxation", "[solver]"),
        psor_maximum_iterations=_require_positive_int(
            solver_table, "psor_maximum_iterations", "[solver]"
        ),
        settlement=_require_string(solver_table, "settlement", "[solver]"),
        contract_multiplier=_require_positive_float(
            solver_table, "contract_multiplier", "[solver]"
        ),
    )
    if solver.settlement not in {"cash", "physical"}:
        raise LabelPolicyError("[solver].settlement must be cash or physical")
    if not 0.0 < solver.psor_relaxation < 2.0:
        raise LabelPolicyError("[solver].psor_relaxation must lie in (0, 2)")

    raw_grids = document.get("grids")
    if not isinstance(raw_grids, list) or len(raw_grids) != 4:
        raise LabelPolicyError("top-level grids must contain exactly four tables")
    grids = tuple(_parse_grid(item, index) for index, item in enumerate(raw_grids))
    if tuple(grid.name for grid in grids) != RAW_GRID_NAMES:
        raise LabelPolicyError(f"grid names and order must be {RAW_GRID_NAMES}")
    expected_dimensions = ((800, 400), (1600, 800), (3200, 1600), (6400, 3200))
    if tuple((grid.spot_intervals, grid.time_steps) for grid in grids) != expected_dimensions:
        raise LabelPolicyError(
            "grid dimensions must be the declared 800x400 through 6400x3200 ladder"
        )
    if tuple(grid.role for grid in grids) != ("candidate", "candidate", "reference", "anchor"):
        raise LabelPolicyError("grid roles must be candidate, candidate, reference, anchor")

    bump_table = _require_table(document, "bumps", "top-level")
    _reject_unknown(bump_table, _BUMP_KEYS, "[bumps]")
    bumps = BumpSettings(
        spot=_require_positive_float_array(bump_table, "spot", "[bumps]"),
        primary_spot=_require_positive_float(bump_table, "primary_spot", "[bumps]"),
        volatility=_require_positive_float_array(bump_table, "volatility", "[bumps]"),
        primary_volatility=_require_positive_float(
            bump_table, "primary_volatility", "[bumps]"
        ),
    )
    if len(bumps.spot) < 2 or len(bumps.volatility) < 2:
        raise LabelPolicyError("both bump ladders must contain at least two sizes")
    if bumps.primary_spot not in bumps.spot or bumps.primary_volatility not in bumps.volatility:
        raise LabelPolicyError("primary bumps must be members of their bump ladders")

    rich_table = _require_table(document, "richardson", "top-level")
    _reject_unknown(rich_table, _RICHARDSON_KEYS, "[richardson]")
    richardson = RichardsonSettings(
        coarse_to_fine_ratio=_require_positive_int(
            rich_table, "coarse_to_fine_ratio", "[richardson]"
        ),
        assumed_order=_require_positive_float(rich_table, "assumed_order", "[richardson]"),
        minimum_supported_observed_order=_require_float(
            rich_table, "minimum_supported_observed_order", "[richardson]"
        ),
        maximum_supported_observed_order=_require_float(
            rich_table, "maximum_supported_observed_order", "[richardson]"
        ),
    )
    if richardson.coarse_to_fine_ratio != 2 or richardson.assumed_order != 2.0:
        raise LabelPolicyError("task 9C-B Richardson must use a factor-two, second-order formula")
    if not (
        richardson.minimum_supported_observed_order
        < richardson.maximum_supported_observed_order
    ):
        raise LabelPolicyError("Richardson observed-order support interval is empty")

    criteria_table = _require_table(document, "criteria", "top-level")
    _reject_unknown(criteria_table, _CRITERIA_KEYS, "[criteria]")
    criteria = Criteria(
        **{
            key: _require_positive_float(criteria_table, key, "[criteria]")
            for key in _CRITERIA_KEYS
        }
    )
    expected_thresholds = (5.0e-4, 1.0e-3, 2.0e-4, 5.0e-2)
    actual_thresholds = (
        criteria.price_absolute_error,
        criteria.delta_absolute_error,
        criteria.gamma_absolute_error,
        criteria.vega_absolute_error_per_unit_volatility,
    )
    if actual_thresholds != expected_thresholds:
        raise LabelPolicyError("the four predeclared regular-case accuracy thresholds changed")

    raw_cases = document.get("cases")
    if not isinstance(raw_cases, list) or not 24 <= len(raw_cases) <= 36:
        raise LabelPolicyError("top-level cases must contain between 24 and 36 case tables")
    cases = tuple(_parse_case(item, index, solver, bumps) for index, item in enumerate(raw_cases))
    names = tuple(case.name for case in cases)
    if len(set(names)) != len(names):
        raise LabelPolicyError("case names must be unique")
    if not set(anchor_cases).issubset(names):
        raise LabelPolicyError("[study].anchor_cases contains an unknown case")
    classifications = {case.classification for case in cases}
    if classifications != CLASSIFICATIONS:
        raise LabelPolicyError("the design must keep distinct regular and stress cases")
    if sum(case.classification == "stress" for case in cases) < 3:
        raise LabelPolicyError("the design must retain several explicitly labelled stress cases")
    if {case.option_type for case in cases} != OPTION_TYPES:
        raise LabelPolicyError("the design must cover calls and puts")
    if {case.exercise_style for case in cases} != EXERCISE_STYLES:
        raise LabelPolicyError("the design must cover European and American exercise")

    return PilotConfig(
        name=_require_string(study, "name", "[study]"),
        selection_order=selection_order,
        anchor_cases=anchor_cases,
        projected_label_count=projected_label_count,
        projected_worker_counts=worker_counts,
        solver=solver,
        grids=grids,
        bumps=bumps,
        richardson=richardson,
        criteria=criteria,
        cases=cases,
        source_name=source_name,
        source_sha256=source_sha256,
    )


def _parse_grid(value: Any, index: int) -> Grid:
    context = f"grids[{index}]"
    if not isinstance(value, Mapping):
        raise LabelPolicyError(f"{context} must be a table")
    _reject_unknown(value, _GRID_KEYS, context)
    role = _require_string(value, "role", context)
    if role not in GRID_ROLES:
        raise LabelPolicyError(f"{context}.role is invalid")
    return Grid(
        name=_require_string(value, "name", context),
        spot_intervals=_require_positive_int(value, "spot_intervals", context),
        time_steps=_require_positive_int(value, "time_steps", context),
        role=role,
    )


def _parse_case(
    value: Any, index: int, solver: SolverSettings, bumps: BumpSettings
) -> PilotCase:
    context = f"cases[{index}]"
    if not isinstance(value, Mapping):
        raise LabelPolicyError(f"{context} must be a table")
    _reject_unknown(value, _CASE_KEYS, context)
    classification = _require_string(value, "classification", context)
    option_type = _require_string(value, "option_type", context)
    exercise_style = _require_string(value, "exercise_style", context)
    if classification not in CLASSIFICATIONS:
        raise LabelPolicyError(f"{context}.classification must be regular or stress")
    if option_type not in OPTION_TYPES:
        raise LabelPolicyError(f"{context}.option_type must be call or put")
    if exercise_style not in EXERCISE_STYLES:
        raise LabelPolicyError(f"{context}.exercise_style must be european or american")
    spot = _require_positive_float(value, "spot", context)
    strike = _require_positive_float(value, "strike", context)
    expiry = _require_positive_float(value, "expiry_time", context)
    volatility = _require_positive_float(value, "volatility", context)
    rate = _require_float(value, "rate", context)
    carry = _require_float(value, "continuous_carry", context)
    if spot - max(bumps.spot) <= 0.0:
        raise LabelPolicyError(f"{context} spot bump ladder crosses S <= 0")
    if volatility - max(bumps.volatility) <= 0.0:
        raise LabelPolicyError(f"{context} volatility bump ladder crosses sigma <= 0")
    if solver.spot_maximum <= max(spot + max(bumps.spot), strike):
        raise LabelPolicyError(f"{context} bumped spot or strike reaches the spatial boundary")
    raw_dividends = value.get("dividends")
    if not isinstance(raw_dividends, list):
        raise LabelPolicyError(f"{context}.dividends must be an explicit array")
    dividends: list[tuple[float, float]] = []
    for dividend_index, raw in enumerate(raw_dividends):
        if not isinstance(raw, list) or len(raw) != 2:
            raise LabelPolicyError(f"{context}.dividends[{dividend_index}] must be [time, amount]")
        ex_time = _as_finite_float(raw[0], f"{context}.dividends[{dividend_index}][0]")
        amount = _as_finite_float(raw[1], f"{context}.dividends[{dividend_index}][1]")
        if not 0.0 < ex_time < expiry or amount <= 0.0:
            raise LabelPolicyError(f"{context} dividend must have 0 < time < expiry and amount > 0")
        dividends.append((ex_time, amount))
    if any(later[0] <= earlier[0] for earlier, later in itertools.pairwise(dividends)):
        raise LabelPolicyError(f"{context}.dividends must be strictly time ordered")
    return PilotCase(
        name=_require_string(value, "name", context),
        classification=classification,
        description=_require_string(value, "description", context),
        option_type=option_type,
        exercise_style=exercise_style,
        spot=spot,
        strike=strike,
        expiry_time=expiry,
        rate=rate,
        continuous_carry=carry,
        volatility=volatility,
        dividends=tuple(dividends),
    )


def bump_requests(case: PilotCase, bumps: BumpSettings) -> tuple[BumpRequest, ...]:
    """Return the deterministic centered-bump state ladder for one case."""
    requests = [BumpRequest("center", case.spot, case.volatility, "center", None, 0)]
    for bump in bumps.spot:
        if case.spot - bump <= 0.0:
            raise LabelPolicyError(f"case '{case.name}' spot bump {bump} crosses S <= 0")
        token = _number_token(bump)
        requests.extend(
            (
                BumpRequest(
                    f"spot_{token}_down",
                    case.spot - bump,
                    case.volatility,
                    "spot",
                    bump,
                    -1,
                ),
                BumpRequest(f"spot_{token}_up", case.spot + bump, case.volatility, "spot", bump, 1),
            )
        )
    for bump in bumps.volatility:
        if case.volatility - bump <= 0.0:
            raise LabelPolicyError(
                f"case '{case.name}' volatility bump {bump} crosses sigma <= 0"
            )
        token = _number_token(bump)
        requests.extend(
            (
                BumpRequest(
                    f"volatility_{token}_down",
                    case.spot,
                    case.volatility - bump,
                    "volatility",
                    bump,
                    -1,
                ),
                BumpRequest(
                    f"volatility_{token}_up",
                    case.spot,
                    case.volatility + bump,
                    "volatility",
                    bump,
                    1,
                ),
            )
        )
    return tuple(requests)


def labels_from_prices(
    prices: Mapping[str, float], bumps: BumpSettings
) -> dict[str, Any]:
    """Derive price, delta, gamma, and both vega unit conventions."""
    center = float(prices["center"])
    deltas: dict[str, float] = {}
    gammas: dict[str, float] = {}
    vegas: dict[str, float] = {}
    vegas_point: dict[str, float] = {}
    for bump in bumps.spot:
        token = _number_token(bump)
        up = float(prices[f"spot_{token}_up"])
        down = float(prices[f"spot_{token}_down"])
        deltas[token] = centered_delta(up, down, bump)
        gammas[token] = centered_gamma(up, center, down, bump)
    for bump in bumps.volatility:
        token = _number_token(bump)
        up = float(prices[f"volatility_{token}_up"])
        down = float(prices[f"volatility_{token}_down"])
        vegas[token] = centered_vega(up, down, bump)
        vegas_point[token] = vega_per_volatility_point(vegas[token])
    return {
        "price": center,
        "delta_by_spot_bump": deltas,
        "gamma_by_spot_bump": gammas,
        "vega_per_unit_by_volatility_bump": vegas,
        "vega_per_point_by_volatility_bump": vegas_point,
        "bump_variation": {
            "delta": _variation(deltas.values()),
            "gamma": _variation(gammas.values()),
            "vega_per_unit_volatility": _variation(vegas.values()),
        },
    }


def _richardson_labels(
    coarse: Mapping[str, Any], fine: Mapping[str, Any], settings: RichardsonSettings
) -> dict[str, Any]:
    def combine(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
        return {
            key: richardson_extrapolate(float(left[key]), float(right[key]), settings.assumed_order)
            for key in left
        }

    delta = combine(coarse["delta_by_spot_bump"], fine["delta_by_spot_bump"])
    gamma = combine(coarse["gamma_by_spot_bump"], fine["gamma_by_spot_bump"])
    vega = combine(
        coarse["vega_per_unit_by_volatility_bump"],
        fine["vega_per_unit_by_volatility_bump"],
    )
    return {
        "price": richardson_extrapolate(
            float(coarse["price"]), float(fine["price"]), settings.assumed_order
        ),
        "delta_by_spot_bump": delta,
        "gamma_by_spot_bump": gamma,
        "vega_per_unit_by_volatility_bump": vega,
        "vega_per_point_by_volatility_bump": {
            key: vega_per_volatility_point(value) for key, value in vega.items()
        },
        "bump_variation": {
            "delta": _variation(delta.values()),
            "gamma": _variation(gamma.values()),
            "vega_per_unit_volatility": _variation(vega.values()),
        },
    }


def _orders(
    coarse: Mapping[str, Any], medium: Mapping[str, Any], fine: Mapping[str, Any]
) -> dict[str, Any]:
    def values(key: str) -> dict[str, float | None]:
        return {
            bump: observed_order(
                float(coarse[key][bump]), float(medium[key][bump]), float(fine[key][bump])
            )
            for bump in coarse[key]
        }

    return {
        "price": observed_order(
            float(coarse["price"]), float(medium["price"]), float(fine["price"])
        ),
        "delta_by_spot_bump": values("delta_by_spot_bump"),
        "gamma_by_spot_bump": values("gamma_by_spot_bump"),
        "vega_per_unit_by_volatility_bump": values(
            "vega_per_unit_by_volatility_bump"
        ),
    }


def _order_supported(value: float | None, settings: RichardsonSettings) -> bool:
    return value is not None and (
        settings.minimum_supported_observed_order
        <= value
        <= settings.maximum_supported_observed_order
    )


def _reference_labels(
    medium: Mapping[str, Any],
    fine: Mapping[str, Any],
    orders: Mapping[str, Any],
    config: PilotConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    rich = _richardson_labels(medium, fine, config.richardson)
    sources: dict[str, Any] = {}

    price_supported = _order_supported(orders["price"], config.richardson)
    output: dict[str, Any] = {
        "price": rich["price"] if price_supported else fine["price"],
    }
    sources["price"] = RICHARDSON_REFERENCE if price_supported else RAW_GRID_NAMES[2]
    for label_key, order_key in (
        ("delta_by_spot_bump", "delta_by_spot_bump"),
        ("gamma_by_spot_bump", "gamma_by_spot_bump"),
        ("vega_per_unit_by_volatility_bump", "vega_per_unit_by_volatility_bump"),
    ):
        output[label_key] = {}
        sources[label_key] = {}
        for bump, fine_value in fine[label_key].items():
            supported = _order_supported(orders[order_key][bump], config.richardson)
            output[label_key][bump] = rich[label_key][bump] if supported else fine_value
            sources[label_key][bump] = (
                RICHARDSON_REFERENCE if supported else RAW_GRID_NAMES[2]
            )
    output["vega_per_point_by_volatility_bump"] = {
        bump: vega_per_volatility_point(value)
        for bump, value in output["vega_per_unit_by_volatility_bump"].items()
    }
    output["bump_variation"] = {
        "delta": _variation(output["delta_by_spot_bump"].values()),
        "gamma": _variation(output["gamma_by_spot_bump"].values()),
        "vega_per_unit_volatility": _variation(
            output["vega_per_unit_by_volatility_bump"].values()
        ),
    }
    return output, sources


def _price_one(
    case: PilotCase,
    request: BumpRequest,
    grid: Grid,
    solver: SolverSettings,
    exercise_style: str,
) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    try:
        result = pde_price(
            option_type=case.option_type,
            exercise_style=exercise_style,
            spot=request.spot,
            strike=case.strike,
            valuation_time=0.0,
            expiry_time=case.expiry_time,
            volatility=request.volatility,
            continuous_carry=case.continuous_carry,
            curve_times=[0.0, case.expiry_time],
            curve_log_discounts=[0.0, -case.rate * case.expiry_time],
            dividends=list(case.dividends),
            settlement=solver.settlement,
            contract_multiplier=solver.contract_multiplier,
            spot_intervals=grid.spot_intervals,
            time_steps=grid.time_steps,
            spot_maximum=solver.spot_maximum,
            rannacher_steps=solver.rannacher_steps,
            psor_tolerance=solver.psor_tolerance,
            psor_relaxation=solver.psor_relaxation,
            psor_maximum_iterations=solver.psor_maximum_iterations,
        )
    except (RuntimeError, ValueError, OverflowError) as error:
        raise LabelPolicyError(
            f"PDE solve failed for case={case.name}, grid={grid.name}, "
            f"bump={request.name}, style={exercise_style}: {error}"
        ) from error
    elapsed = time.perf_counter() - started
    return dict(result), elapsed


def run_pilot(config: PilotConfig) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the full expensive PDE pilot and return report plus measured performance."""
    started = time.perf_counter()
    initial_peak = _peak_rss_bytes()
    timings: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    for case in config.cases:
        requests = bump_requests(case, config.bumps)
        required_grids = list(config.grids[:3])
        if case.name in config.anchor_cases:
            required_grids.append(config.grids[3])
        raw: dict[str, dict[str, Any]] = {}
        control_raw: dict[str, dict[str, Any]] = {}
        detailed_grids: list[dict[str, Any]] = []
        for grid in required_grids:
            price_rows: list[dict[str, Any]] = []
            prices: dict[str, float] = {}
            control_prices: dict[str, float] = {}
            grid_signatures: set[tuple[Any, ...]] = set()
            diagnostic_totals = {
                "psor_total_iterations": 0,
                "linear_solves": 0,
                "psor_solves": 0,
            }
            for request in requests:
                result, elapsed = _price_one(
                    case, request, grid, config.solver, case.exercise_style
                )
                timings.append(
                    {
                        "case": case.name,
                        "grid": grid.name,
                        "bump": request.name,
                        "style": case.exercise_style,
                        "seconds": elapsed,
                    }
                )
                prices[request.name] = float(result["price"])
                signature = (
                    int(result["spot_intervals"]),
                    float(result["spot_maximum"]),
                    float(result["spot_step"]),
                    int(result["time_steps"]),
                    tuple(float(value) for value in result["aligned_times"]),
                )
                grid_signatures.add(signature)
                for key in diagnostic_totals:
                    diagnostic_totals[key] += int(result[key])
                row = {
                    "name": request.name,
                    "kind": request.kind,
                    "size": request.size,
                    "direction": request.direction,
                    "spot": request.spot,
                    "volatility": request.volatility,
                    "price": float(result["price"]),
                    "solver_status": str(result["solver_status"]),
                    "discretization_accuracy": str(result["discretization_accuracy"]),
                    "psor_total_iterations": int(result["psor_total_iterations"]),
                    "psor_maximum_iterations_used": int(
                        result["psor_maximum_iterations_used"]
                    ),
                    "maximum_relative_lcp_residual": float(
                        result["maximum_relative_lcp_residual"]
                    ),
                }
                price_rows.append(row)
                if case.exercise_style == "american":
                    control, control_elapsed = _price_one(
                        case, request, grid, config.solver, "european"
                    )
                    timings.append(
                        {
                            "case": case.name,
                            "grid": grid.name,
                            "bump": request.name,
                            "style": "european_dominance_control",
                            "seconds": control_elapsed,
                        }
                    )
                    control_prices[request.name] = float(control["price"])
            if len(grid_signatures) != 1:
                raise LabelPolicyError(
                    f"case '{case.name}' grid '{grid.name}' changed domain, nodes, or time grid "
                    "across centered bumps"
                )
            signature = next(iter(grid_signatures))
            labels = labels_from_prices(prices, config.bumps)
            raw[grid.name] = labels
            if control_prices:
                control_raw[grid.name] = labels_from_prices(control_prices, config.bumps)
            detailed_grids.append(
                {
                    "grid": asdict(grid),
                    "actual_grid": {
                        "spot_intervals": signature[0],
                        "spot_maximum": signature[1],
                        "spot_step": signature[2],
                        "time_steps": signature[3],
                        "aligned_times": list(signature[4]),
                    },
                    "diagnostic_totals": diagnostic_totals,
                    "bumped_prices": price_rows,
                    "labels": labels,
                    "european_dominance_control_labels": (
                        control_raw.get(grid.name) if control_prices else None
                    ),
                }
            )

        candidate_labels: dict[str, dict[str, Any]] = {
            RAW_GRID_NAMES[0]: raw[RAW_GRID_NAMES[0]],
            RAW_GRID_NAMES[1]: raw[RAW_GRID_NAMES[1]],
            RICHARDSON_CANDIDATE: _richardson_labels(
                raw[RAW_GRID_NAMES[0]], raw[RAW_GRID_NAMES[1]], config.richardson
            ),
        }
        control_candidates: dict[str, dict[str, Any]] = {}
        if control_raw:
            control_candidates = {
                RAW_GRID_NAMES[0]: control_raw[RAW_GRID_NAMES[0]],
                RAW_GRID_NAMES[1]: control_raw[RAW_GRID_NAMES[1]],
                RICHARDSON_CANDIDATE: _richardson_labels(
                    control_raw[RAW_GRID_NAMES[0]],
                    control_raw[RAW_GRID_NAMES[1]],
                    config.richardson,
                ),
            }
        orders = _orders(
            raw[RAW_GRID_NAMES[0]], raw[RAW_GRID_NAMES[1]], raw[RAW_GRID_NAMES[2]]
        )
        reference, reference_sources = _reference_labels(
            raw[RAW_GRID_NAMES[1]], raw[RAW_GRID_NAMES[2]], orders, config
        )
        outcomes = {
            policy: _case_outcome(
                case,
                policy,
                labels,
                reference,
                candidate_labels,
                control_candidates,
                orders,
                config,
            )
            for policy, labels in candidate_labels.items()
        }
        anchor = None
        if RAW_GRID_NAMES[3] in raw:
            anchor_orders = _orders(
                raw[RAW_GRID_NAMES[1]], raw[RAW_GRID_NAMES[2]], raw[RAW_GRID_NAMES[3]]
            )
            anchor_reference, anchor_sources = _reference_labels(
                raw[RAW_GRID_NAMES[2]], raw[RAW_GRID_NAMES[3]], anchor_orders, config
            )
            anchor = {
                "observed_orders_1600_3200_6400": anchor_orders,
                "anchor_reference": anchor_reference,
                "anchor_reference_sources": {
                    key: _rename_anchor_source(value) for key, value in anchor_sources.items()
                },
                "main_reference_absolute_gaps": _label_errors(
                    reference, anchor_reference, config.bumps
                ),
            }
        case_reports.append(
            {
                "state": _case_state(case),
                "grid_results": detailed_grids,
                "candidate_labels": candidate_labels,
                "reference_labels": reference,
                "reference_sources": reference_sources,
                "observed_orders_800_1600_3200": orders,
                "candidate_outcomes": outcomes,
                "greek_supervision": _greek_supervision(case, reference, outcomes, config),
                "anchor_validation": anchor,
            }
        )

    policy_outcomes = _policy_outcomes(case_reports, config)
    recommendation = select_accuracy_policy(config.selection_order, policy_outcomes)
    measured = _performance(
        timings=timings,
        started=started,
        initial_peak=initial_peak,
        config=config,
        selected_policy=recommendation,
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "study": {
            "name": config.name,
            "objective": "controlled_pde_price_delta_gamma_vega_label_policy_pilot",
            "classification_counts": {
                "regular": sum(case.classification == "regular" for case in config.cases),
                "stress": sum(case.classification == "stress" for case in config.cases),
            },
            "stress_cases_are_descriptive_and_retained": True,
            "expensive_pilot_is_excluded_from_ci": True,
        },
        "conventions": {
            "value": "currency_per_share",
            "delta": "dV/dS",
            "gamma": "d2V/dS2_per_currency_unit",
            "vega_per_unit_volatility": "dV/dsigma where sigma is absolute volatility",
            "vega_per_volatility_point": "vega_per_unit_volatility/100",
            "spot_bumps": list(config.bumps.spot),
            "primary_spot_bump": config.bumps.primary_spot,
            "volatility_bumps": list(config.bumps.volatility),
            "primary_volatility_bump": config.bumps.primary_volatility,
            "theta_and_rho_implemented": False,
        },
        "predeclared_criteria": asdict(config.criteria),
        "richardson_contract": {
            **asdict(config.richardson),
            "formula": "fine + (fine - coarse) / 3 = (4*fine - coarse) / 3",
            "used_only_when_observed_order_is_supported": True,
        },
        "source": _source_provenance(config),
        "solver": asdict(config.solver),
        "grids": [asdict(grid) for grid in config.grids],
        "cases": case_reports,
        "policy_outcomes": policy_outcomes,
        "recommendation": {
            "selected_accuracy_policy": recommendation,
            "selection_order": list(config.selection_order),
            "regular_cases_only_decide_selection": True,
            "criteria_were_not_loosened": True,
        },
        "limitations": [
            (
                "The 3200x1600 PDE, selectively Richardson-extrapolated only where "
                "observed order supports it, is an internal numerical reference rather "
                "than independent truth."
            ),
            (
                "The 6400x3200 rung is restricted to declared anchors and is not run "
                "across the design."
            ),
            (
                "Stress cases at genuine exercise kinks may remain usable for price "
                "evaluation while being unsuitable for Greek supervision."
            ),
            (
                "Parallel cost projections assume ideal independent-worker scaling and "
                "do not establish production feasibility."
            ),
            (
                "No private market data, dataset generation, network training, theta, "
                "rho, or PSOR optimization is part of this pilot."
            ),
        ],
        "performance": measured,
    }
    return report, measured


def _case_outcome(
    case: PilotCase,
    policy: str,
    labels: Mapping[str, Any],
    reference: Mapping[str, Any],
    candidate_labels: Mapping[str, Mapping[str, Any]],
    control_candidates: Mapping[str, Mapping[str, Any]],
    orders: Mapping[str, Any],
    config: PilotConfig,
) -> dict[str, Any]:
    errors = _label_errors(labels, reference, config.bumps)
    violations = _shape_violations(
        case,
        policy,
        labels,
        candidate_labels,
        control_candidates,
        config,
    )
    criteria = config.criteria
    checks = {
        "price_absolute_error": threshold_pass(
            errors["price"], criteria.price_absolute_error
        ),
        "delta_absolute_error": threshold_pass(
            errors["delta"], criteria.delta_absolute_error
        ),
        "gamma_absolute_error": threshold_pass(
            errors["gamma"], criteria.gamma_absolute_error
        ),
        "vega_absolute_error_per_unit_volatility": threshold_pass(
            errors["vega_per_unit_volatility"],
            criteria.vega_absolute_error_per_unit_volatility,
        ),
        "delta_bump_variation": threshold_pass(
            float(labels["bump_variation"]["delta"]), criteria.delta_bump_variation
        ),
        "gamma_bump_variation": threshold_pass(
            float(labels["bump_variation"]["gamma"]), criteria.gamma_bump_variation
        ),
        "vega_bump_variation_per_unit_volatility": threshold_pass(
            float(labels["bump_variation"]["vega_per_unit_volatility"]),
            criteria.vega_bump_variation_per_unit_volatility,
        ),
        "shape_and_bounds": not violations,
    }
    richardson_support = None
    if policy == RICHARDSON_CANDIDATE:
        spot_key = _number_token(config.bumps.primary_spot)
        vol_key = _number_token(config.bumps.primary_volatility)
        richardson_support = {
            "price": _order_supported(orders["price"], config.richardson),
            "delta": _order_supported(
                orders["delta_by_spot_bump"][spot_key], config.richardson
            ),
            "gamma": _order_supported(
                orders["gamma_by_spot_bump"][spot_key], config.richardson
            ),
            "vega": _order_supported(
                orders["vega_per_unit_by_volatility_bump"][vol_key], config.richardson
            ),
        }
        checks["richardson_observed_order_support"] = all(richardson_support.values())
    return {
        "classification": case.classification,
        "errors_to_main_reference": errors,
        "checks": checks,
        "violations": violations,
        "richardson_support": richardson_support,
        "passes": all(checks.values()),
        "counts_toward_selection": case.classification == "regular",
    }


def _label_errors(
    labels: Mapping[str, Any], reference: Mapping[str, Any], bumps: BumpSettings
) -> dict[str, float]:
    spot_key = _number_token(bumps.primary_spot)
    vol_key = _number_token(bumps.primary_volatility)
    return {
        "price": abs(float(labels["price"]) - float(reference["price"])),
        "delta": abs(
            float(labels["delta_by_spot_bump"][spot_key])
            - float(reference["delta_by_spot_bump"][spot_key])
        ),
        "gamma": abs(
            float(labels["gamma_by_spot_bump"][spot_key])
            - float(reference["gamma_by_spot_bump"][spot_key])
        ),
        "vega_per_unit_volatility": abs(
            float(labels["vega_per_unit_by_volatility_bump"][vol_key])
            - float(reference["vega_per_unit_by_volatility_bump"][vol_key])
        ),
    }


def _shape_violations(
    case: PilotCase,
    policy: str,
    labels: Mapping[str, Any],
    candidate_labels: Mapping[str, Mapping[str, Any]],
    control_candidates: Mapping[str, Mapping[str, Any]],
    config: PilotConfig,
) -> list[str]:
    del candidate_labels  # Kept in the signature to make policy provenance explicit.
    tolerance = config.criteria.shape_tolerance
    violations: list[str] = []
    numeric_values = [
        float(labels["price"]),
        *map(float, labels["delta_by_spot_bump"].values()),
        *map(float, labels["gamma_by_spot_bump"].values()),
        *map(float, labels["vega_per_unit_by_volatility_bump"].values()),
    ]
    if not all(math.isfinite(value) for value in numeric_values):
        violations.append("non_finite_value")
    intrinsic = (
        max(case.spot - case.strike, 0.0)
        if case.option_type == "call"
        else max(case.strike - case.spot, 0.0)
    )
    if case.exercise_style == "american" and float(labels["price"]) < intrinsic - tolerance:
        violations.append("american_intrinsic_bound")
    if case.exercise_style == "american" and (
        float(labels["price"]) < float(control_candidates[policy]["price"]) - tolerance
    ):
        violations.append("american_dominance")
    primary_delta = float(
        labels["delta_by_spot_bump"][_number_token(config.bumps.primary_spot)]
    )
    if case.option_type == "call" and primary_delta < -tolerance:
        violations.append("spot_monotonicity")
    if case.option_type == "put" and primary_delta > tolerance:
        violations.append("spot_monotonicity")
    if any(float(gamma) < -tolerance for gamma in labels["gamma_by_spot_bump"].values()):
        violations.append("spot_convexity")
    return violations


def _greek_supervision(
    case: PilotCase,
    reference: Mapping[str, Any],
    outcomes: Mapping[str, Mapping[str, Any]],
    config: PilotConfig,
) -> dict[str, Any]:
    variation = reference["bump_variation"]
    reasons: list[str] = []
    if float(variation["delta"]) > config.criteria.delta_bump_variation:
        reasons.append("reference_delta_varies_across_spot_bumps")
    if float(variation["gamma"]) > config.criteria.gamma_bump_variation:
        reasons.append("reference_gamma_varies_across_spot_bumps")
    if (
        float(variation["vega_per_unit_volatility"])
        > config.criteria.vega_bump_variation_per_unit_volatility
    ):
        reasons.append("reference_vega_varies_across_volatility_bumps")
    if case.classification == "stress" and any(
        token in case.description.lower() for token in ("boundary", "kink", "exercise transition")
    ):
        reasons.append("predeclared_exercise_or_payoff_kink_stress")
    if not any(bool(outcome["checks"]["shape_and_bounds"]) for outcome in outcomes.values()):
        reasons.append("all_candidate_policies_have_shape_or_bound_violations")
    return {
        "suitable": not reasons,
        "reasons": reasons,
        "price_row_is_retained_regardless": True,
    }


def _policy_outcomes(
    cases: Sequence[Mapping[str, Any]], config: PilotConfig
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for policy in config.selection_order:
        regular_failures: list[str] = []
        stress_failures: list[str] = []
        for row in cases:
            name = str(row["state"]["name"])
            outcome = row["candidate_outcomes"][policy]
            if not outcome["passes"]:
                if row["state"]["classification"] == "regular":
                    regular_failures.append(name)
                else:
                    stress_failures.append(name)
        output[policy] = {
            "regular_case_count": sum(
                row["state"]["classification"] == "regular" for row in cases
            ),
            "regular_failure_count": len(regular_failures),
            "regular_failures": regular_failures,
            "all_regular_cases_pass": not regular_failures,
            "stress_failure_count_descriptive_only": len(stress_failures),
            "stress_failures_descriptive_only": stress_failures,
        }
    return output


def _performance(
    *,
    timings: Sequence[Mapping[str, Any]],
    started: float,
    initial_peak: int,
    config: PilotConfig,
    selected_policy: str,
) -> dict[str, Any]:
    elapsed = time.perf_counter() - started
    final_peak = _peak_rss_bytes()
    projections: dict[str, Any] = {}
    primary_names = {
        "center",
        f"spot_{_number_token(config.bumps.primary_spot)}_down",
        f"spot_{_number_token(config.bumps.primary_spot)}_up",
        f"volatility_{_number_token(config.bumps.primary_volatility)}_down",
        f"volatility_{_number_token(config.bumps.primary_volatility)}_up",
    }
    by_grid: dict[str, float] = {}
    for grid_name in RAW_GRID_NAMES[:2]:
        seconds = sum(
            float(row["seconds"])
            for row in timings
            if row["grid"] == grid_name
            and row["bump"] in primary_names
            and row["style"] != "european_dominance_control"
        )
        by_grid[grid_name] = seconds / len(config.cases)
    seconds_per_label = {
        RAW_GRID_NAMES[0]: by_grid[RAW_GRID_NAMES[0]],
        RAW_GRID_NAMES[1]: by_grid[RAW_GRID_NAMES[1]],
        RICHARDSON_CANDIDATE: by_grid[RAW_GRID_NAMES[0]] + by_grid[RAW_GRID_NAMES[1]],
    }
    for policy, per_label in seconds_per_label.items():
        serial = per_label * config.projected_label_count
        projections[policy] = {
            "measured_mean_seconds_per_four-label_state": per_label,
            "serial_seconds_for_250000_labels": serial,
            "serial_hours_for_250000_labels": serial / 3600.0,
            "idealized_8_worker_hours": serial / (8.0 * 3600.0),
            "idealized_16_worker_hours": serial / (16.0 * 3600.0),
        }
    return {
        "timing_protocol": (
            "first-run wall timings are reusable for a numerical byte-identity repeat"
        ),
        "wall_seconds": elapsed,
        "solve_count": len(timings),
        "timed_solve_seconds": sum(float(row["seconds"]) for row in timings),
        "peak_resident_memory_bytes": final_peak,
        "peak_resident_memory_increase_bytes": max(0, final_peak - initial_peak),
        # Filled from deterministic solver diagnostics once the report's case
        # rows have been assembled; unlike wall time this count is bit-stable.
        "psor_iterations": 0,
        "projected_label_count": config.projected_label_count,
        "projections": projections,
        "selected_policy_projection": (
            projections.get(selected_policy) if selected_policy != "no_policy_selected" else None
        ),
        "projection_caveat": (
            "Ideal independent-worker division only; excludes scheduling, serialization, memory "
            "contention, failures, and dataset I/O, and is not a production-feasibility claim."
        ),
        "host": {
            "system": platform.system(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
    }


def _inject_psor_total(report: dict[str, Any]) -> None:
    report["performance"]["psor_iterations"] = sum(
        int(grid["diagnostic_totals"]["psor_total_iterations"])
        for case in report["cases"]
        for grid in case["grid_results"]
    )


def _source_provenance(config: PilotConfig) -> dict[str, Any]:
    module_path = Path(__file__)
    return {
        "config_name": config.source_name,
        "config_sha256": config.source_sha256,
        "runner_name": module_path.name,
        "runner_sha256": hashlib.sha256(module_path.read_bytes()).hexdigest(),
        "pde_header_sha256": str(_pde.__pde_header_sha256__),
        "pde_source_sha256": str(_pde.__pde_source_sha256__),
        "pde_implementation_sha256": str(_pde.__pde_implementation_sha256__),
        "build_configuration": str(_pde.__build_configuration__),
        "cxx_compiler": str(_pde.__cxx_compiler__),
        "data_provenance": "fixed_synthetic_public_configuration_only",
        "private_market_data_used": False,
    }


def _case_state(case: PilotCase) -> dict[str, Any]:
    state = asdict(case)
    state["dividends"] = [list(dividend) for dividend in case.dividends]
    state["valuation_time"] = 0.0
    state["curve_times"] = [0.0, case.expiry_time]
    state["curve_log_discounts"] = [0.0, -case.rate * case.expiry_time]
    state["moneyness_spot_over_strike"] = case.spot / case.strike
    return state


def _rename_anchor_source(value: Any) -> Any:
    if isinstance(value, str):
        if value == RICHARDSON_REFERENCE:
            return RICHARDSON_ANCHOR
        if value == RAW_GRID_NAMES[2]:
            return RAW_GRID_NAMES[3]
    if isinstance(value, Mapping):
        return {key: _rename_anchor_source(item) for key, item in value.items()}
    return value


def write_outputs(report: dict[str, Any], output_directory: Path, *, overwrite: bool) -> list[Path]:
    """Atomically publish deterministic JSON and concise CSV views."""
    _inject_psor_total(report)
    if output_directory.exists() and not output_directory.is_dir():
        raise LabelPolicyError(f"output path '{output_directory}' is not a directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    outputs = {
        "report.json": _canonical_json(report),
        "labels.csv": _labels_csv(report),
        "sensitivity.csv": _sensitivity_csv(report),
        "failures.csv": _failures_csv(report),
        "runtime.csv": _runtime_csv(report),
    }
    paths: list[Path] = []
    for name, payload in outputs.items():
        destination = output_directory / name
        if destination.exists() and not overwrite:
            raise LabelPolicyError(f"refusing to overwrite existing output '{destination}'")
        _write_atomic(destination, payload.encode("utf-8"))
        paths.append(destination)
    return paths


def _labels_csv(report: Mapping[str, Any]) -> str:
    header = [
        "case",
        "classification",
        "policy",
        "price",
        "delta",
        "gamma",
        "vega_per_unit_volatility",
        "vega_per_volatility_point",
        "price_error",
        "delta_error",
        "gamma_error",
        "vega_error",
        "passes",
        "greek_supervision_suitable",
    ]
    rows: list[list[Any]] = []
    primary_spot = _number_token(float(report["conventions"]["primary_spot_bump"]))
    primary_vol = _number_token(float(report["conventions"]["primary_volatility_bump"]))
    for case in report["cases"]:
        for policy, labels in case["candidate_labels"].items():
            errors = case["candidate_outcomes"][policy]["errors_to_main_reference"]
            rows.append(
                [
                    case["state"]["name"],
                    case["state"]["classification"],
                    policy,
                    labels["price"],
                    labels["delta_by_spot_bump"][primary_spot],
                    labels["gamma_by_spot_bump"][primary_spot],
                    labels["vega_per_unit_by_volatility_bump"][primary_vol],
                    labels["vega_per_point_by_volatility_bump"][primary_vol],
                    errors["price"],
                    errors["delta"],
                    errors["gamma"],
                    errors["vega_per_unit_volatility"],
                    case["candidate_outcomes"][policy]["passes"],
                    case["greek_supervision"]["suitable"],
                ]
            )
    return _csv_text(header, rows)


def _sensitivity_csv(report: Mapping[str, Any]) -> str:
    header = ["case", "classification", "policy", "metric", "bump", "value", "observed_order"]
    rows: list[list[Any]] = []
    for case in report["cases"]:
        orders = case["observed_orders_800_1600_3200"]
        for policy, labels in case["candidate_labels"].items():
            rows.append(
                [
                    case["state"]["name"],
                    case["state"]["classification"],
                    policy,
                    "price",
                    "center",
                    labels["price"],
                    orders["price"],
                ]
            )
            for metric, order_metric in (
                ("delta_by_spot_bump", "delta_by_spot_bump"),
                ("gamma_by_spot_bump", "gamma_by_spot_bump"),
                ("vega_per_unit_by_volatility_bump", "vega_per_unit_by_volatility_bump"),
            ):
                for bump, value in labels[metric].items():
                    rows.append(
                        [
                            case["state"]["name"],
                            case["state"]["classification"],
                            policy,
                            metric,
                            bump,
                            value,
                            orders[order_metric][bump],
                        ]
                    )
    return _csv_text(header, rows)


def _failures_csv(report: Mapping[str, Any]) -> str:
    header = [
        "case",
        "classification",
        "policy",
        "failed_checks",
        "violations",
        "counts_toward_selection",
    ]
    rows: list[list[Any]] = []
    for case in report["cases"]:
        for policy, outcome in case["candidate_outcomes"].items():
            failed = [key for key, passed in outcome["checks"].items() if not passed]
            if failed:
                rows.append(
                    [
                        case["state"]["name"],
                        case["state"]["classification"],
                        policy,
                        ";".join(failed),
                        ";".join(outcome["violations"]),
                        outcome["counts_toward_selection"],
                    ]
                )
    return _csv_text(header, rows)


def _runtime_csv(report: Mapping[str, Any]) -> str:
    header = [
        "policy",
        "seconds_per_four_label_state",
        "serial_hours_250000",
        "idealized_8_worker_hours",
        "idealized_16_worker_hours",
        "selected",
    ]
    selected = report["recommendation"]["selected_accuracy_policy"]
    rows = []
    # JSON is serialized with sorted keys, so a performance block loaded for
    # the byte-identity repeat has a different mapping insertion order from the
    # first in-memory run.  The versioned selection order is the canonical CSV
    # row order in both cases.
    for policy in report["recommendation"]["selection_order"]:
        projection = report["performance"]["projections"][policy]
        rows.append(
            [
                policy,
                projection["measured_mean_seconds_per_four-label_state"],
                projection["serial_hours_for_250000_labels"],
                projection["idealized_8_worker_hours"],
                projection["idealized_16_worker_hours"],
                policy == selected,
            ]
        )
    return _csv_text(header, rows)


def _csv_text(header: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buffer.getvalue()


def _canonical_json(document: Any) -> str:
    try:
        return json.dumps(document, allow_nan=False, indent=2, sort_keys=True) + "\n"
    except (TypeError, ValueError) as error:
        raise LabelPolicyError(f"report is not finite canonical JSON: {error}") from error


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
        raise LabelPolicyError(f"cannot publish '{destination}': {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def replace_performance_from(report: dict[str, Any], source: Path) -> None:
    """Reuse first-run timing so a full numerical repeat is byte-identical."""
    try:
        previous = json.loads(source.read_text(encoding="utf-8"))
        performance = previous["performance"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise LabelPolicyError(f"cannot reuse performance from '{source}': {error}") from error
    report["performance"] = performance


def verify_byte_identity(left: Path, right: Path) -> None:
    """Require every expected artifact to be byte-identical."""
    for name in ("report.json", "labels.csv", "sensitivity.csv", "failures.csv", "runtime.csv"):
        left_path = left / name
        right_path = right / name
        try:
            left_bytes = left_path.read_bytes()
            right_bytes = right_path.read_bytes()
        except OSError as error:
            raise LabelPolicyError(f"cannot compare identity artifact '{name}': {error}") from error
        if left_bytes != right_bytes:
            raise LabelPolicyError(f"byte-identity repeat differs for '{name}'")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--performance-source",
        type=Path,
        default=None,
        help="reuse timing from a first report after rerunning every numerical solve",
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
        if arguments.output_directory.exists() and not arguments.overwrite:
            existing = list(arguments.output_directory.iterdir())
            if existing:
                raise LabelPolicyError(
                    f"refusing to run expensive pilot into non-empty '{arguments.output_directory}'"
                )
        config = load_pilot_config(arguments.config)
        report, _performance_document = run_pilot(config)
        _inject_psor_total(report)
        if arguments.performance_source is not None:
            replace_performance_from(report, arguments.performance_source)
        paths = write_outputs(report, arguments.output_directory, overwrite=arguments.overwrite)
        if arguments.verify_identical_to is not None:
            verify_byte_identity(arguments.verify_identical_to, arguments.output_directory)
    except LabelPolicyError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(
        f"wrote {len(paths)} artifacts; selected_accuracy_policy="
        f"{report['recommendation']['selected_accuracy_policy']}"
    )
    return 0


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == "darwin" else peak * 1024


def _variation(values: Any) -> float:
    finite = [float(value) for value in values]
    return max(finite) - min(finite)


def _number_token(value: float) -> str:
    return format(value, ".17g")


def _require_positive_finite_number(value: float, context: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise LabelPolicyError(f"{context} must be finite and positive")


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], context: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise LabelPolicyError(f"{context} has unknown keys: {', '.join(unknown)}")


def _require_table(table: Mapping[str, Any], key: str, context: str) -> Mapping[str, Any]:
    value = table.get(key)
    if not isinstance(value, Mapping):
        raise LabelPolicyError(f"{context}.{key} must be a table")
    return value


def _require_string(table: Mapping[str, Any], key: str, context: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise LabelPolicyError(f"{context}.{key} must be a non-empty string")
    return value


def _require_string_array(table: Mapping[str, Any], key: str, context: str) -> tuple[str, ...]:
    value = table.get(key)
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
    ):
        raise LabelPolicyError(f"{context}.{key} must be a non-empty string array")
    return tuple(value)


def _as_finite_float(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise LabelPolicyError(f"{context} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise LabelPolicyError(f"{context} must be finite")
    return result


def _require_float(table: Mapping[str, Any], key: str, context: str) -> float:
    if key not in table:
        raise LabelPolicyError(f"{context}.{key} is required")
    return _as_finite_float(table[key], f"{context}.{key}")


def _require_positive_float(table: Mapping[str, Any], key: str, context: str) -> float:
    value = _require_float(table, key, context)
    if value <= 0.0:
        raise LabelPolicyError(f"{context}.{key} must be positive")
    return value


def _require_positive_float_array(
    table: Mapping[str, Any], key: str, context: str
) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise LabelPolicyError(f"{context}.{key} must be a non-empty numeric array")
    parsed = tuple(_as_finite_float(item, f"{context}.{key}") for item in value)
    if any(item <= 0.0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise LabelPolicyError(f"{context}.{key} must contain unique positive values")
    return parsed


def _require_positive_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise LabelPolicyError(f"{context}.{key} must be a positive integer")
    return value


def _require_nonnegative_int(table: Mapping[str, Any], key: str, context: str) -> int:
    value = table.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise LabelPolicyError(f"{context}.{key} must be a non-negative integer")
    return value


def _require_int_array(table: Mapping[str, Any], key: str, context: str) -> tuple[int, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise LabelPolicyError(f"{context}.{key} must be a non-empty integer array")
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise LabelPolicyError(f"{context}.{key} must contain positive integers")
    return tuple(value)


if __name__ == "__main__":
    sys.exit(main())
