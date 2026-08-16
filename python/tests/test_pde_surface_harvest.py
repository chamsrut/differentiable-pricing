"""Task 9C-C2a: identities, grouped partitioning, and exact-node harvesting.

Most requirements here are structural, so most tests drive the harvester with a
synthetic surface builder rather than the compiled solver: the point is that
assignment cannot see an outcome and that a group cannot straddle a partition,
neither of which needs a real backward induction. A small number of tests do
call the real engine, to check that eligibility, regimes and exact node spots
survive the round trip and that the scalar and task 9C-C1 APIs are unchanged.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import math
import tomllib
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest
from differentiable_pricing import pde_price, pde_valuation_surface
from differentiable_pricing.american import pde_surface_harvest as harvest
from differentiable_pricing.data.config import SPLIT_NAMES

CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "pde_surface_harvest_demo_v1.toml"

STRIKE = 100.0


def _scenario(
    name: str,
    *,
    strike: float = STRIKE,
    expiry_time: float = 1.0,
    rate: float = 0.05,
    base_volatility: float = 0.2,
    continuous_carry: float = 0.0,
    valuation_time: float = 0.0,
    dividends: list[list[float]] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "strike": strike,
        "valuation_time": valuation_time,
        "expiry_time": expiry_time,
        "rate": rate,
        "base_volatility": base_volatility,
        "continuous_carry": continuous_carry,
        "dividends": [] if dividends is None else dividends,
    }


def _document(
    *,
    scenarios: list[dict[str, Any]] | None = None,
    roles: list[str] | None = None,
    option_types: list[str] | None = None,
    exercise_styles: list[str] | None = None,
    quota: dict[str, int] | None = None,
    window: tuple[float, float] = (0.7, 1.4),
    spot_intervals: int = 200,
    time_steps: int = 100,
    boundary_exclusion_nodes: int = 4,
    volatility_bump: float | None = None,
) -> dict[str, Any]:
    surfaces: dict[str, Any] = {
        "roles": ["base"] if roles is None else roles,
        "option_types": ["put"] if option_types is None else option_types,
        "exercise_styles": ["american"] if exercise_styles is None else exercise_styles,
    }
    if volatility_bump is not None:
        surfaces["volatility_bump"] = volatility_bump
    return {
        "schema_version": harvest.SCHEMA_VERSION,
        "study": {
            "name": "unit-test-design",
            "status": "exploratory_infrastructure_demonstration",
            "engine": harvest.ENGINE_NAME,
            "price_units": "currency_per_share",
            "greek_method": "nodewise_discrete_derivatives_of_one_solved_slice",
            "curve_construction": "flat_continuously_compounded_from_scenario_rate",
        },
        "identity": {
            "version": harvest.IDENTITY_VERSION,
            "digest": "sha256_over_canonical_json",
            "partition_group_excludes": list(harvest.PARTITION_GROUP_EXCLUDES),
        },
        "partitioning": {
            "algorithm": harvest.PARTITION_ALGORITHM,
            "seed": 7,
            "minimum_groups_per_partition": 1,
            "weights": {"train": 0.5, "validation": 0.25, "interpolation_test": 0.25},
        },
        "solver": {
            "spot_maximum_strike_multiple": 4.0,
            "rannacher_steps": 2,
            "psor_tolerance": 1.0e-11,
            "psor_relaxation": 1.2,
            "psor_maximum_iterations": 50000,
            "settlement": "cash",
            "contract_multiplier": 100.0,
        },
        "grid": {
            "spot_intervals": spot_intervals,
            "time_steps": time_steps,
            "boundary_exclusion_nodes": boundary_exclusion_nodes,
        },
        "surfaces": surfaces,
        "harvest": {
            "status": "exploratory_demonstration_values",
            "node_source": harvest.NODE_SOURCE,
            "moneyness_window_low": window[0],
            "moneyness_window_high": window[1],
            "selection_rule": harvest.SELECTION_RULE,
            "quota": (
                {
                    "continuation": 5,
                    "exercise": 3,
                    "numerically_indifferent": 2,
                    "no_obstacle": 5,
                }
                if quota is None
                else quota
            ),
        },
        "scenarios": (
            [
                _scenario("a"),
                _scenario("b", base_volatility=0.3),
                _scenario("c", expiry_time=2.0),
                _scenario("d", rate=0.03),
            ]
            if scenarios is None
            else scenarios
        ),
    }


def _config(document: dict[str, Any] | None = None) -> harvest.HarvestConfig:
    return harvest.parse_harvest_config(
        _document() if document is None else document,
        source_name="unit-test.toml",
        raw_config_sha256="0" * 64,
    )


# ---------------------------------------------------------------------------
# Synthetic surfaces
# ---------------------------------------------------------------------------


def _regimes(spot_intervals: int, exercise_style: str) -> list[str]:
    if exercise_style == "european":
        return ["no_obstacle"] * (spot_intervals + 1)
    states: list[str] = []
    for index in range(spot_intervals + 1):
        if index < spot_intervals // 4:
            states.append("exercise")
        elif index < (3 * spot_intervals) // 4:
            states.append("continuation")
        else:
            states.append("numerically_indifferent")
    return states


def _synthetic_surface(
    *,
    spot_intervals: int = 200,
    spot_maximum: float = 400.0,
    exercise_style: str = "american",
    buffer: int = 4,
    states: list[str] | None = None,
    values: list[float] | None = None,
    valuation_time: float = 0.0,
    expiry_time: float = 1.0,
    time_steps: int = 100,
    rannacher_steps: int = 2,
    psor_tolerance: float = 1.0e-11,
    psor_relaxation: float = 1.2,
    strike: float = STRIKE,
    volatility: float = 0.2,
    continuous_carry: float = 0.0,
    curve_times: list[float] | None = None,
    curve_log_discounts: list[float] | None = None,
    dividends: list[list[float]] | None = None,
    settlement: str = "cash",
    option_type: str = "put",
    psor_maximum_iterations: int = 50000,
    contract_multiplier: float = 100.0,
    requested_spot_intervals: int | None = None,
    requested_spot_maximum: float | None = None,
    requested_time_steps: int | None = None,
) -> dict[str, Any]:
    """Build one surface obeying the task 9C-C1 eligibility precedence exactly."""
    step = spot_maximum / spot_intervals
    spots = [index * step for index in range(spot_intervals + 1)]
    states = _regimes(spot_intervals, exercise_style) if states is None else states
    values = [max(STRIKE - spot, 0.0) + 1.0 for spot in spots] if values is None else values

    deltas: list[float | None] = []
    gammas: list[float | None] = []
    eligible: list[bool] = []
    reasons: list[str] = []
    for index in range(spot_intervals + 1):
        endpoint = index in (0, spot_intervals)
        deltas.append(None if endpoint else -0.5 + index * 1.0e-3)
        gammas.append(None if endpoint else 1.0e-3)
        if endpoint:
            reason = "centered_stencil_unavailable"
        elif index < buffer or index > spot_intervals - buffer:
            reason = "inside_domain_boundary_buffer"
        elif not math.isfinite(values[index]):
            reason = "non_finite_stencil_value"
        elif states[index] == "numerically_indifferent":
            reason = "unresolved_exercise_state"
        elif any(
            states[neighbour] != states[index]
            for neighbour in range(index - 2, index + 3)
            if 0 <= neighbour <= spot_intervals
        ):
            reason = "regime_stencil_not_uniform"
        else:
            reason = "eligible"
        reasons.append(reason)
        eligible.append(reason == "eligible")

    dividends = [] if dividends is None else dividends
    aligned_times = sorted({valuation_time, expiry_time, *(item[0] for item in dividends)})
    span = expiry_time - valuation_time
    used_time_steps = sum(
        max(1, math.floor(time_steps * (right - left) / span + 0.5))
        for left, right in pairwise(aligned_times)
    )
    return {
        # The mandatory normalized input echo the production binding now always
        # returns. A fake solver that omitted it would be exercising a path the
        # real one cannot take.
        "surface_input": {
            "option_type": option_type,
            "exercise_style": exercise_style,
            "strike": strike,
            "valuation_time": valuation_time,
            "expiry_time": expiry_time,
            "volatility": volatility,
            "continuous_carry": continuous_carry,
            "curve_times": (
                [valuation_time, expiry_time] if curve_times is None else curve_times
            ),
            "curve_log_discounts": (
                [0.0, 0.0] if curve_log_discounts is None else curve_log_discounts
            ),
            "dividends": dividends,
            "dividends_declared": True,
            "settlement": settlement,
            "contract_multiplier": contract_multiplier,
            "spot_intervals": (
                spot_intervals if requested_spot_intervals is None else requested_spot_intervals
            ),
            "time_steps": time_steps if requested_time_steps is None else requested_time_steps,
            "spot_maximum": (
                spot_maximum if requested_spot_maximum is None else requested_spot_maximum
            ),
            "rannacher_steps": rannacher_steps,
            "psor_tolerance": psor_tolerance,
            "psor_relaxation": psor_relaxation,
            "psor_maximum_iterations": psor_maximum_iterations,
            "boundary_exclusion_nodes": buffer,
        },
        # The real result carries the valuation time at top level too.
        "valuation_time": valuation_time,
        "spot_nodes": spots,
        "values": values,
        "deltas": deltas,
        "gammas": gammas,
        "greek_eligible": eligible,
        "exercise_states": states,
        "greek_eligibility_reasons": reasons,
        "spot_intervals": spot_intervals,
        "spot_maximum": spot_maximum,
        "spot_step": step,
        "time_steps": used_time_steps,
        "rannacher_steps": rannacher_steps,
        "psor_tolerance": psor_tolerance,
        "psor_relaxation": psor_relaxation,
        "boundary_exclusion_nodes": buffer,
        "regime_stencil_radius": int(harvest._pde.pde_regime_stencil_radius),
        "backward_inductions": 1,
        "query_count": 0,
        "queries": [],
        "aligned_times": aligned_times,
        "dividend_events": [
            {"ex_time": item[0], "amount": item[1], "aligned_time_index": index + 1}
            for index, item in enumerate(dividends)
        ],
        "strike_node_index": round(strike / step),
        "solver_status": "discrete_system_converged",
        "discretization_accuracy": "not_assessed",
        "exercise_classification_scale": 1.0e-9,
        "maximum_relative_lcp_residual": 1.0e-12,
    }


def _surface_from_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    return _synthetic_surface(
        spot_intervals=int(kwargs["spot_intervals"]),
        spot_maximum=float(kwargs["spot_maximum"]),
        exercise_style=str(kwargs["exercise_style"]),
        buffer=int(kwargs["boundary_exclusion_nodes"]),
        valuation_time=float(kwargs["valuation_time"]),
        expiry_time=float(kwargs["expiry_time"]),
        time_steps=int(kwargs["time_steps"]),
        rannacher_steps=int(kwargs["rannacher_steps"]),
        psor_tolerance=float(kwargs["psor_tolerance"]),
        psor_relaxation=float(kwargs["psor_relaxation"]),
        strike=float(kwargs["strike"]),
        volatility=float(kwargs["volatility"]),
        continuous_carry=float(kwargs["continuous_carry"]),
        curve_times=list(kwargs["curve_times"]),
        curve_log_discounts=list(kwargs["curve_log_discounts"]),
        dividends=list(kwargs["dividends"]),
        settlement=str(kwargs["settlement"]),
        option_type=str(kwargs["option_type"]),
        psor_maximum_iterations=int(kwargs["psor_maximum_iterations"]),
        contract_multiplier=float(kwargs["contract_multiplier"]),
    )


class _SyntheticSolver:
    """A solver stand-in that records every call it receives."""

    def __init__(self, *, failing_surface: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.failing_surface = failing_surface

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(dict(kwargs))
        if self.failing_surface is not None and len(self.calls) - 1 == self.failing_surface:
            raise RuntimeError("synthetic solve failure")
        return _surface_from_kwargs(kwargs)


# ---------------------------------------------------------------------------
# 1-2. Canonical identities
# ---------------------------------------------------------------------------


def test_partition_group_id_is_stable_across_repeated_calls() -> None:
    config = _config()
    scenario = config.scenarios[0]
    first = harvest.partition_group_identity(scenario)
    for _repeat in range(5):
        assert harvest.partition_group_identity(scenario) == first
    assert first.startswith("pg-") and len(first) == 3 + 64


def test_surface_and_row_ids_are_stable_across_repeated_calls() -> None:
    descriptor = harvest.plan_harvest(_config()).surfaces[0].solver_input
    surface = harvest.surface_identity(descriptor)
    assert surface == harvest.surface_identity(descriptor)
    assert surface == harvest.solver_input_identity(descriptor)
    row = harvest.row_identity(surface_id=surface, node_index=17)
    assert row == harvest.row_identity(surface_id=surface, node_index=17)
    assert row != harvest.row_identity(surface_id=surface, node_index=18)


def test_identity_payload_is_independent_of_mapping_insertion_order() -> None:
    forward = {"alpha": 1.0, "beta": [1, 2], "gamma": "x"}
    reversed_order = {"gamma": "x", "beta": [1, 2], "alpha": 1.0}
    assert harvest.canonical_payload(forward) == harvest.canonical_payload(reversed_order)
    # Sequence order is preserved, because a declared order is part of the state.
    assert harvest.canonical_payload([1, 2]) != harvest.canonical_payload([2, 1])


def test_group_ids_are_independent_of_scenario_declaration_order() -> None:
    document = _document()
    reversed_document = copy.deepcopy(document)
    reversed_document["scenarios"] = list(reversed(document["scenarios"]))
    forward = harvest.plan_harvest(_config(document))
    backward = harvest.plan_harvest(_config(reversed_document))
    assert [group.partition_group_id for group in forward.groups] == [
        group.partition_group_id for group in backward.groups
    ]
    assert [group.partition for group in forward.groups] == [
        group.partition for group in backward.groups
    ]
    assert [surface.surface_id for surface in forward.surfaces] == [
        surface.surface_id for surface in backward.surfaces
    ]


def test_reporting_only_contract_multiplier_does_not_distinguish_a_pricing_state() -> None:
    config = _config()
    base = config.scenarios[0]
    rescaled = dataclasses.replace(base, contract_multiplier=base.contract_multiplier * 7.5)
    assert rescaled.contract_multiplier != base.contract_multiplier
    assert harvest.partition_group_identity(rescaled) == harvest.partition_group_identity(base)

    numerical = harvest._numerical_settings(base, config)
    identities = {
        harvest.surface_identity(
            harvest._solver_input_descriptor(
                scenario=scenario,
                numerical=numerical,
                option_type="put",
                exercise_style="american",
                volatility=scenario.base_volatility,
            )
        )
        for scenario in (base, rescaled)
    }
    assert len(identities) == 1
    only = identities.pop()
    assert harvest.row_identity(surface_id=only, node_index=11) == harvest.row_identity(
        surface_id=only, node_index=11
    )
    assert "reporting_only_contract_multiplier" in harvest.PARTITION_GROUP_EXCLUDES


def test_surface_role_and_partition_group_do_not_enter_actual_solve_identity() -> None:
    descriptor = harvest.plan_harvest(_config()).surfaces[0].solver_input
    assert "surface_role" not in descriptor.payload()
    assert "partition_group_id" not in descriptor.payload()
    assert "contract_multiplier" not in descriptor.payload()
    assert harvest.surface_identity(descriptor) == harvest.solver_input_identity(descriptor)


def test_different_actual_volatilities_have_distinct_solver_input_ids() -> None:
    descriptor = harvest.plan_harvest(_config()).surfaces[0].solver_input
    changed = dataclasses.replace(descriptor, volatility=descriptor.volatility + 0.01)
    assert harvest.solver_input_identity(descriptor) != harvest.solver_input_identity(changed)


def test_candidates_differing_only_in_contract_multiplier_are_duplicate_groups() -> None:
    config = _config()
    base = config.scenarios[0]
    twin = dataclasses.replace(base, name="same_state_other_multiplier", contract_multiplier=1.0)
    design = dataclasses.replace(
        config, scenarios=(base, twin, *config.scenarios[1:])
    )
    solver = _SyntheticSolver()
    with pytest.raises(harvest.HarvestError, match="same economic state"):
        harvest.run_harvest(design, solver=solver)
    # Rejected before the first backward induction, like any duplicate state.
    assert solver.calls == []


def test_contract_multiplier_does_not_change_the_pde_price() -> None:
    arguments = dict(_ENGINE_ARGUMENTS)
    arguments.pop("contract_multiplier")
    first = pde_price(
        option_type="put",
        exercise_style="american",
        spot=95.0,
        contract_multiplier=1.0,
        **arguments,
    )
    second = pde_price(
        option_type="put",
        exercise_style="american",
        spot=95.0,
        contract_multiplier=250.0,
        **arguments,
    )
    # Bitwise, not to a tolerance: prices are per share and the multiplier is
    # carried for reporting only.
    assert first["price"] == second["price"]
    assert first == second


def test_contract_multiplier_survives_in_row_and_report_metadata() -> None:
    config = _config()
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    assert rows
    assert "contract_multiplier" in harvest.ROW_COLUMNS
    assert all(row["contract_multiplier"] == 100.0 for row in rows)
    assert (
        "reporting_only_contract_multiplier" in report["identity"]["partition_group_excludes"]
    )


def test_signed_zero_is_canonicalized_before_hashing() -> None:
    assert harvest.canonical_payload({"x": -0.0}) == harvest.canonical_payload({"x": 0.0})
    # Nested, as curve and dividend inputs arrive.
    assert harvest.canonical_payload(
        {"curve": [-0.0, 1.0], "dividends": [[-0.0, 2.0]]}
    ) == harvest.canonical_payload({"curve": [0.0, 1.0], "dividends": [[0.0, 2.0]]})
    # Genuinely different small magnitudes stay distinct, including subnormals.
    for tiny in (1.0e-320, -1.0e-320, 5.0e-324, -5.0e-324):
        assert harvest.canonical_payload({"x": tiny}) != harvest.canonical_payload({"x": 0.0})
    assert harvest.canonical_payload({"x": 1.0e-320}) != harvest.canonical_payload(
        {"x": -1.0e-320}
    )
    # Non-finite values are still refused.
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(harvest.HarvestError):
            harvest.canonical_payload({"x": bad})


def test_signed_zero_does_not_split_an_economic_state() -> None:
    config = _config()
    base = config.scenarios[0]
    negative_zero = dataclasses.replace(base, continuous_carry=-0.0, valuation_time=-0.0)
    assert math.copysign(1.0, negative_zero.continuous_carry) == -1.0
    assert harvest.partition_group_identity(negative_zero) == harvest.partition_group_identity(
        base
    )
    # A zero rate makes the curve's own log discount -0.0 by construction.
    zero_rate = dataclasses.replace(base, rate=0.0)
    flipped = dataclasses.replace(base, rate=-0.0)
    assert math.copysign(1.0, zero_rate.curve_log_discounts[1]) == -1.0
    assert harvest.partition_group_identity(zero_rate) == harvest.partition_group_identity(
        flipped
    )


def test_identity_payload_rejects_non_finite_and_unsupported_values() -> None:
    with pytest.raises(harvest.HarvestError):
        harvest.canonical_payload({"x": float("nan")})
    with pytest.raises(harvest.HarvestError):
        harvest.canonical_payload({"x": {1: "int key"}})
    with pytest.raises(harvest.HarvestError):
        harvest.canonical_payload({"x": object()})


def test_partition_group_excludes_spot_role_option_type_and_grid() -> None:
    config = _config()
    scenario = config.scenarios[0]
    group_id = harvest.partition_group_identity(scenario)
    changed = dataclasses.replace(scenario, contract_multiplier=1.0)
    assert harvest.partition_group_identity(changed) == group_id
    assert "option_type" in harvest.PARTITION_GROUP_EXCLUDES
    assert "exercise_style" in harvest.PARTITION_GROUP_EXCLUDES
    assert "volatility_bump_role" in harvest.PARTITION_GROUP_EXCLUDES
    assert "numerical_grid_and_solver_settings" in harvest.PARTITION_GROUP_EXCLUDES


# ---------------------------------------------------------------------------
# 3. Assignment precedes solving
# ---------------------------------------------------------------------------


def test_planning_assigns_every_group_without_a_solver() -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    assert len(plan.groups) == len(config.scenarios)
    assert all(group.partition in harvest.PARTITION_NAMES for group in plan.groups)
    # plan_harvest has no solver parameter at all, so no outcome can reach it.
    assert "solver" not in harvest.plan_harvest.__code__.co_varnames


def test_every_solver_call_sees_an_already_assigned_plan() -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    assigned = {group.partition_group_id: group.partition for group in plan.groups}
    observed: list[str] = []

    def spy(**kwargs: Any) -> dict[str, Any]:
        assert len(assigned) == len(plan.groups)
        assert all(partition in harvest.PARTITION_NAMES for partition in assigned.values())
        observed.append(str(kwargs["exercise_style"]))
        return _surface_from_kwargs(kwargs)

    report, _rows = harvest.execute_plan(plan, config, solver=spy)
    assert len(observed) == len(plan.surfaces)
    assert report["partitioning"]["assignment_precedes_solving"] is True


def test_execute_plan_rejects_changed_semantic_config_before_solving() -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    changed = dataclasses.replace(
        config,
        grid=dataclasses.replace(config.grid, spot_intervals=config.grid.spot_intervals + 4),
    )
    solver = _SyntheticSolver()
    with pytest.raises(harvest.HarvestError, match="semantic digest mismatch"):
        harvest.execute_plan(plan, changed, solver=solver)
    assert solver.calls == []


@pytest.mark.parametrize(
    ("mutate", "field"),
    [
        (lambda surface: surface.__setitem__("spot_intervals", 100), "spot_intervals"),
        (lambda surface: surface.__setitem__("spot_maximum", 123.0), "spot_maximum"),
        (lambda surface: surface.__setitem__("boundary_exclusion_nodes", 7), "boundary"),
        (lambda surface: surface.__setitem__("valuation_time", 0.125), "valuation_time"),
        (
            lambda surface: surface.__setitem__(
                "solver_status", "psor_iteration_limit_exceeded"
            ),
            "solver_status",
        ),
        # The mandatory input echo: a missing section, a missing field, and one
        # wrong value of every identity-defining kind.
        (lambda surface: surface.pop("surface_input"), "surface_input"),
        (lambda surface: surface["surface_input"].pop("volatility"), "missing required field"),
        (lambda surface: surface["surface_input"].pop("settlement"), "missing required field"),
        (lambda surface: surface["surface_input"].__setitem__("volatility", 0.99), "volatility"),
        (
            lambda surface: surface["surface_input"].__setitem__(
                "curve_log_discounts", [0.0, -0.4]
            ),
            "curve_log_discounts",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("curve_times", [0.0, 3.0]),
            "curve_times",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("option_type", "call"),
            "option_type",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("exercise_style", "european"),
            "exercise_style",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("settlement", "physical"),
            "settlement",
        ),
        (lambda surface: surface["surface_input"].__setitem__("strike", 123.0), "strike"),
        (
            lambda surface: surface["surface_input"].__setitem__("continuous_carry", 0.33),
            "continuous_carry",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("dividends", [[0.5, 1.0]]),
            "dividends",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("spot_intervals", 111),
            "spot_intervals",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("psor_tolerance", 1.0e-6),
            "psor_tolerance",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("psor_maximum_iterations", 7),
            "psor_maximum_iterations",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("rannacher_steps", 9),
            "rannacher_steps",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("boundary_exclusion_nodes", 9),
            "boundary_exclusion_nodes",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("dividends_declared", False),
            "undeclared dividend schedule",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("contract_multiplier", 3.0),
            "contract_multiplier",
        ),
        (
            lambda surface: surface["surface_input"].__setitem__("unexpected_field", 1),
            "unknown field",
        ),
    ],
)
def test_mismatched_returned_surface_is_a_failed_attempt_with_no_rows(
    mutate: Any, field: str
) -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    calls = 0

    def solver(**kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        surface = _surface_from_kwargs(kwargs)
        if calls == 0:
            mutate(surface)
        calls += 1
        return surface

    report, rows = harvest.execute_plan(plan, config, solver=solver)
    failed_group = plan.groups[0]
    assert calls == len(plan.surfaces)
    assert report["totals"]["attempted_surface_solve_count"] == len(plan.surfaces)
    assert report["totals"]["failed_surface_solve_count"] == 1
    assert report["failures"][0]["stage"] == "surface_validation"
    assert field in report["failures"][0]["error_message"]
    assert all(row["partition_group_id"] != failed_group.partition_group_id for row in rows)


def test_matching_returned_surface_is_bound_and_accepted() -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    report, rows = harvest.execute_plan(plan, config, solver=_SyntheticSolver())
    assert rows
    assert report["totals"]["successful_surface_solve_count"] == len(plan.surfaces)
    assert report["totals"]["failed_surface_solve_count"] == 0


def test_assignment_ignores_input_order_and_batch_size() -> None:
    settings = _config().partitioning
    ids = [f"pg-{index:064d}" for index in range(9)]
    forward = harvest.assign_partitions(ids, settings)
    backward = harvest.assign_partitions(list(reversed(ids)), settings)
    assert forward == backward


# ---------------------------------------------------------------------------
# 4-6, 15. Grouping guarantees
# ---------------------------------------------------------------------------


def test_all_rows_of_one_group_land_in_one_partition() -> None:
    config = _config(
        _document(option_types=["call", "put"], exercise_styles=["american", "european"])
    )
    _report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    partitions: dict[str, set[str]] = {}
    for row in rows:
        partitions.setdefault(row["partition_group_id"], set()).add(row["partition"])
    assert partitions
    assert all(len(values) == 1 for values in partitions.values())


def test_bump_roles_share_a_partition_group_with_distinct_surface_ids() -> None:
    config = _config(
        _document(roles=["base", "sigma_down", "sigma_up"], volatility_bump=0.01)
    )
    plan = harvest.plan_harvest(config)
    for group in plan.groups:
        roles = {surface.surface_role for surface in group.surfaces}
        assert roles == {"base", "sigma_down", "sigma_up"}
        assert len({surface.surface_id for surface in group.surfaces}) == len(group.surfaces)
        assert {surface.partition for surface in group.surfaces} == {group.partition}
        volatilities = {surface.surface_role: surface.volatility for surface in group.surfaces}
        assert volatilities["sigma_down"] < volatilities["base"] < volatilities["sigma_up"]


def test_cross_group_actual_solve_alias_is_rejected_before_assignment_or_solving(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    document = _document(
        scenarios=[
            _scenario("group_a", base_volatility=0.20),
            _scenario("group_b", base_volatility=0.25),
            _scenario("group_c", base_volatility=0.40),
        ],
        roles=["base", "sigma_up"],
        volatility_bump=0.05,
    )
    config = _config(document)
    numerical_a = harvest._numerical_settings(config.scenarios[0], config)
    numerical_b = harvest._numerical_settings(config.scenarios[1], config)
    a_up = harvest._solver_input_descriptor(
        scenario=config.scenarios[0],
        option_type="put",
        exercise_style="american",
        volatility=0.25,
        numerical=numerical_a,
    )
    b_base = harvest._solver_input_descriptor(
        scenario=config.scenarios[1],
        option_type="put",
        exercise_style="american",
        volatility=0.25,
        numerical=numerical_b,
    )
    assert harvest.solver_input_identity(a_up) == harvest.solver_input_identity(b_base)

    assigned = False

    def forbidden_assignment(*_args: Any, **_kwargs: Any) -> dict[str, str]:
        nonlocal assigned
        assigned = True
        raise AssertionError("partition assignment must not run")

    monkeypatch.setattr(harvest, "assign_partitions", forbidden_assignment)
    solver = _SyntheticSolver()
    with pytest.raises(harvest.HarvestError, match="duplicate actual PDE solver input"):
        harvest.run_harvest(config, solver=solver)
    assert assigned is False
    assert solver.calls == []


def test_duplicate_base_groups_are_rejected_before_any_solve() -> None:
    document = _document(
        scenarios=[_scenario("a"), _scenario("b"), _scenario("c"), _scenario("a_again")]
    )
    # 'a_again' repeats every economic field of 'a'; only the name differs, and a
    # name is documentation rather than state.
    solver = _SyntheticSolver()
    with pytest.raises(harvest.HarvestError, match="same economic state"):
        harvest.run_harvest(_config(document), solver=solver)
    assert solver.calls == []


def test_no_group_or_row_appears_in_more_than_one_partition() -> None:
    config = _config(_document(option_types=["call", "put"]))
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    integrity = report["integrity"]
    assert integrity["cross_partition_group_intersection_count"] == 0
    assert integrity["cross_partition_row_intersection_count"] == 0
    assert integrity["duplicate_partition_group_count"] == 0
    assert integrity["duplicate_row_id_count"] == 0
    assert integrity["groups_in_more_than_one_partition"] == []
    assert len({row["row_id"] for row in rows}) == len(rows)


def test_partition_counts_fill_every_partition_and_sum_to_the_total() -> None:
    settings = _config().partitioning
    for total in range(3, 40):
        counts = harvest.partition_counts(total, settings)
        assert sum(count for _name, count in counts) == total
        assert all(count >= settings.minimum_groups_per_partition for _name, count in counts)
    with pytest.raises(harvest.HarvestError):
        harvest.partition_counts(2, settings)


def test_partition_names_match_the_repository_dataset_convention() -> None:
    assert harvest.PARTITION_NAMES == SPLIT_NAMES


# ---------------------------------------------------------------------------
# 7-11. Node selection
# ---------------------------------------------------------------------------


def test_duplicate_node_selection_is_detected_and_rejected() -> None:
    kept, rejected = harvest.deduplicate_node_indices([4, 7, 4, 9, 7, 7])
    assert kept == (4, 7, 9)
    assert rejected == 3
    with pytest.raises(harvest.HarvestError, match="duplicate row_id"):
        harvest.check_unique_row_ids([{"row_id": "rw-a"}, {"row_id": "rw-a"}])


def test_selection_is_evenly_spaced_strictly_increasing_and_deterministic() -> None:
    candidates = list(range(10, 30))
    for quota in range(0, len(candidates) + 3):
        picked = harvest.evenly_spaced_selection(candidates, quota)
        assert len(picked) == min(quota, len(candidates))
        assert list(picked) == sorted(set(picked))
        assert picked == harvest.evenly_spaced_selection(candidates, quota)
        assert set(picked) <= set(candidates)
    assert harvest.evenly_spaced_selection(candidates, 2) == (10, 29)


def test_harvested_rows_are_exact_grid_nodes_only() -> None:
    config = _config()
    solver = _SyntheticSolver()
    _report, rows = harvest.run_harvest(config, solver=solver)
    assert rows
    for row in rows:
        assert row["spot"] == row["node_index"] * row["spot_step"]
        assert float(row["node_index"]).is_integer()
    # No off-grid query is ever requested: the task 9C-C1 query API is untouched.
    assert all(call["query_spots"] == [] for call in solver.calls)


def test_off_grid_node_source_is_rejected_by_the_configuration() -> None:
    document = _document()
    document["harvest"]["node_source"] = "interpolated_query_spots"
    with pytest.raises(harvest.HarvestError, match="off-grid interpolation"):
        _config(document)


def test_boundary_endpoints_and_buffer_nodes_are_excluded() -> None:
    config = _config(_document(window=(0.0001, 100.0), boundary_exclusion_nodes=6))
    plan = harvest.plan_harvest(config)
    surface_plan = plan.groups[0].surfaces[0]
    surface = _synthetic_surface(spot_intervals=200, spot_maximum=400.0, buffer=6)
    result = harvest.harvest_surface(
        surface, surface_plan, plan.groups[0].scenario, config.harvest
    )
    indices = [row["node_index"] for row in result.rows]
    assert indices
    assert min(indices) >= 6
    assert max(indices) <= 200 - 6
    assert result.rejected["truncation_endpoint"] == 2
    assert result.rejected["boundary_buffer"] == 2 * 6 - 2


def test_interior_moneyness_window_is_enforced() -> None:
    config = _config(_document(window=(0.9, 1.1), spot_intervals=200))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    surface = _synthetic_surface(
        spot_intervals=200, spot_maximum=group.scenario.strike * 4.0
    )
    result = harvest.harvest_surface(surface, group.surfaces[0], group.scenario, config.harvest)
    assert result.rows
    for row in result.rows:
        assert 0.9 <= row["moneyness_spot_over_strike"] <= 1.1
    assert result.rejected["outside_moneyness_window"] > 0


def test_regime_quotas_are_deterministic_and_shortfalls_are_reported() -> None:
    quota = {
        "continuation": 4,
        "exercise": 3,
        "numerically_indifferent": 2,
        "no_obstacle": 4,
    }
    config = _config(_document(window=(0.0001, 100.0), quota=quota))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    surface = _synthetic_surface(spot_intervals=200, spot_maximum=400.0)
    first = harvest.harvest_surface(surface, group.surfaces[0], group.scenario, config.harvest)
    second = harvest.harvest_surface(surface, group.surfaces[0], group.scenario, config.harvest)
    assert [row["row_id"] for row in first.rows] == [row["row_id"] for row in second.rows]
    assert first.selected_by_state["continuation"] == 4
    assert first.selected_by_state["exercise"] == 3
    assert first.selected_by_state["numerically_indifferent"] == 2
    assert first.selected_by_state["no_obstacle"] == 0
    assert first.shortfalls == ()

    # A quota larger than the admissible population is a reported shortfall, not
    # a silent truncation.
    starved = _config(_document(window=(0.0001, 100.0), quota={**quota, "exercise": 500}))
    result = harvest.harvest_surface(
        surface, group.surfaces[0], group.scenario, starved.harvest
    )
    shortfall = next(
        item for item in result.shortfalls if item["exercise_state"] == "exercise"
    )
    assert shortfall["requested"] == 500
    assert shortfall["achieved"] < 500
    assert shortfall["shortfall"] == 500 - shortfall["achieved"]


def test_a_regime_the_style_cannot_produce_is_inapplicable_not_a_shortfall() -> None:
    config = _config(_document(exercise_styles=["european"], window=(0.0001, 100.0)))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    surface = _synthetic_surface(spot_intervals=200, exercise_style="european")
    result = harvest.harvest_surface(surface, group.surfaces[0], group.scenario, config.harvest)
    assert result.applicable_states == ("no_obstacle",)
    assert result.requested_by_state["exercise"] == 0
    assert [item["exercise_state"] for item in result.shortfalls] == []
    assert all(row["exercise_state"] == "no_obstacle" for row in result.rows)


# ---------------------------------------------------------------------------
# 12-13. Eligibility carry-through
# ---------------------------------------------------------------------------


def test_indifferent_nodes_keep_their_price_but_never_a_greek_label() -> None:
    config = _config(_document(window=(0.0001, 100.0)))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    surface = _synthetic_surface(spot_intervals=200, spot_maximum=400.0)
    result = harvest.harvest_surface(surface, group.surfaces[0], group.scenario, config.harvest)
    indifferent = [
        row for row in result.rows if row["exercise_state"] == "numerically_indifferent"
    ]
    assert indifferent
    for row in indifferent:
        assert math.isfinite(row["price"])
        assert row["price_label_eligible"] is True
        assert row["greek_eligible"] is False
        assert row["delta_label_eligible"] is False
        assert row["gamma_label_eligible"] is False
        assert row["greek_eligibility_reason"] == "unresolved_exercise_state"
        # The stencil number is still exposed; only its admissibility is denied.
        assert row["delta"] is not None


def test_greek_eligibility_and_reason_are_copied_from_the_surface() -> None:
    config = _config(_document(window=(0.0001, 100.0)))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    surface = _synthetic_surface(spot_intervals=200, spot_maximum=400.0)
    result = harvest.harvest_surface(surface, group.surfaces[0], group.scenario, config.harvest)
    assert result.rows
    for row in result.rows:
        index = row["node_index"]
        assert row["greek_eligible"] == surface["greek_eligible"][index]
        assert row["greek_eligibility_reason"] == surface["greek_eligibility_reasons"][index]
        assert row["exercise_state"] == surface["exercise_states"][index]
        assert row["delta"] == surface["deltas"][index]
        assert row["gamma"] == surface["gammas"][index]
        assert row["delta_label_eligible"] is (
            row["greek_eligible"] and row["delta"] is not None
        )
    assert any(not row["greek_eligible"] for row in result.rows)
    assert any(row["greek_eligible"] for row in result.rows)


def test_an_ineligible_node_may_not_be_promoted_to_an_eligible_label() -> None:
    config = _config(_document(window=(0.0001, 100.0)))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    surface = _synthetic_surface(spot_intervals=200, spot_maximum=400.0)
    # Forge exactly the contradiction the harvester must refuse.
    forged = copy.deepcopy(surface)
    forged_any = False
    for index, state in enumerate(forged["exercise_states"]):
        if state == "numerically_indifferent":
            forged["greek_eligible"][index] = True
            forged["greek_eligibility_reasons"][index] = "eligible"
            forged_any = True
    assert forged_any
    with pytest.raises(harvest.HarvestError, match="numerically indifferent"):
        harvest.harvest_surface(forged, group.surfaces[0], group.scenario, config.harvest)


def test_an_eligible_node_without_a_derivative_is_refused() -> None:
    config = _config(_document(window=(0.0001, 100.0)))
    plan = harvest.plan_harvest(config)
    group = plan.groups[0]
    forged = _synthetic_surface(spot_intervals=200, spot_maximum=400.0)
    index = forged["greek_eligible"].index(True)
    forged["deltas"][index] = None
    with pytest.raises(harvest.HarvestError, match="missing derivative"):
        harvest.harvest_surface(forged, group.surfaces[0], group.scenario, config.harvest)


# ---------------------------------------------------------------------------
# 14. Failure isolation
# ---------------------------------------------------------------------------


def test_one_failing_surface_is_isolated_without_corrupting_other_groups() -> None:
    config = _config(_document(option_types=["call", "put"]))
    plan = harvest.plan_harvest(config)
    failing = plan.groups[1].surfaces[0]
    solver = _SyntheticSolver(failing_surface=2)
    report, rows = harvest.execute_plan(plan, config, solver=solver)

    assert report["totals"]["failed_group_count"] == 1
    assert report["totals"]["successful_group_count"] == len(plan.groups) - 1
    failure = report["failures"][0]
    assert failure["surface_id"] == failing.surface_id
    assert failure["partition_group_id"] == failing.partition_group_id
    assert failure["partition"] == failing.partition
    assert failure["error_type"] == "RuntimeError"
    # The failed group contributes no rows, and every other group is untouched.
    assert all(row["partition_group_id"] != failing.partition_group_id for row in rows)
    assert rows
    harvest.reconcile_report(report, rows)


# ---------------------------------------------------------------------------
# 16-18. Determinism, publication and reconciliation
# ---------------------------------------------------------------------------


def _publish(config: harvest.HarvestConfig, directory: Path, **kwargs: Any) -> dict[str, bytes]:
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver(), **kwargs)
    harvest.write_outputs(report, rows, directory, overwrite=True)
    harvest.verify_publication(directory)
    return {
        name: (directory / name).read_bytes()
        for name in (*harvest.PUBLISHED_FILES, harvest.MANIFEST_NAME)
    }


def test_shuffled_candidate_input_produces_byte_identical_output(tmp_path: Path) -> None:
    document = _document(option_types=["call", "put"])
    shuffled = copy.deepcopy(document)
    shuffled["scenarios"] = [document["scenarios"][index] for index in (2, 0, 3, 1)]
    first = _publish(_config(document), tmp_path / "forward")
    second = _publish(_config(shuffled), tmp_path / "shuffled")
    assert first == second


def test_chunk_size_changes_no_output_byte(tmp_path: Path) -> None:
    config = _config(_document(option_types=["call", "put"]))
    whole = _publish(config, tmp_path / "whole")
    for index, chunk_size in enumerate((1, 2, 3, 97)):
        chunked = _publish(config, tmp_path / f"chunk{index}", chunk_size=chunk_size)
        assert chunked == whole
    with pytest.raises(harvest.HarvestError):
        harvest.run_harvest(config, solver=_SyntheticSolver(), chunk_size=0)


def test_repeated_runs_are_byte_identical(tmp_path: Path) -> None:
    config = _config()
    first = _publish(config, tmp_path / "one")
    second = _publish(config, tmp_path / "two")
    assert first == second
    harvest.verify_byte_identity(tmp_path / "one", tmp_path / "two")


def test_published_output_carries_no_time_host_or_path_information(tmp_path: Path) -> None:
    config = _config()
    payloads = _publish(config, tmp_path / "run")
    text = b"".join(payloads.values()).decode("utf-8")
    for forbidden in ("elapsed", "timestamp", "hostname", "wall_clock", str(tmp_path)):
        assert forbidden not in text
    report = json.loads(payloads["report.json"])
    assert set(report["provenance"]) == {
        "config_name",
        "raw_config_sha256",
        "semantic_config_sha256",
        "runner_name",
        "runner_sha256",
        "pde_header_sha256",
        "pde_source_sha256",
        "pde_implementation_sha256",
        "data_provenance",
        "private_market_data_used",
    }


def test_reordered_toml_preserves_semantics_ids_assignments_and_rows_only(
    tmp_path: Path,
) -> None:
    document = _document(option_types=["call", "put"])
    reordered = copy.deepcopy(document)
    reordered["scenarios"] = list(reversed(reordered["scenarios"]))
    first_path = tmp_path / "a" / "config.toml"
    second_path = tmp_path / "b" / "config.toml"
    first_path.parent.mkdir()
    second_path.parent.mkdir()
    first_path.write_text(_toml(document), encoding="utf-8")
    second_path.write_text(_toml(reordered), encoding="utf-8")
    first_config = harvest.load_harvest_config(first_path)
    second_config = harvest.load_harvest_config(second_path)
    assert first_config.raw_config_sha256 != second_config.raw_config_sha256
    assert first_config.semantic_config_sha256 == second_config.semantic_config_sha256
    first_plan = harvest.plan_harvest(first_config)
    second_plan = harvest.plan_harvest(second_config)
    assert [group.partition_group_id for group in first_plan.groups] == [
        group.partition_group_id for group in second_plan.groups
    ]
    assert [group.partition for group in first_plan.groups] == [
        group.partition for group in second_plan.groups
    ]
    assert [surface.surface_id for surface in first_plan.surfaces] == [
        surface.surface_id for surface in second_plan.surfaces
    ]
    first_report, first_rows = harvest.execute_plan(
        first_plan, first_config, solver=_SyntheticSolver()
    )
    second_report, second_rows = harvest.execute_plan(
        second_plan, second_config, solver=_SyntheticSolver()
    )
    assert harvest.rows_csv(first_rows) == harvest.rows_csv(second_rows)
    assert first_report != second_report
    first_output = tmp_path / "first-output"
    second_output = tmp_path / "second-output"
    harvest.write_outputs(first_report, first_rows, first_output)
    harvest.write_outputs(second_report, second_rows, second_output)
    assert (first_output / "rows.csv").read_bytes() == (second_output / "rows.csv").read_bytes()
    assert (first_output / "report.json").read_bytes() != (
        second_output / "report.json"
    ).read_bytes()
    first_manifest = json.loads((first_output / harvest.MANIFEST_NAME).read_text())
    second_manifest = json.loads((second_output / harvest.MANIFEST_NAME).read_text())
    assert first_manifest["files"]["rows.csv"] == second_manifest["files"]["rows.csv"]
    assert first_manifest["files"]["report.json"] != second_manifest["files"]["report.json"]
    first_manifest["files"].pop("report.json")
    second_manifest["files"].pop("report.json")
    assert first_manifest == second_manifest
    first_raw = first_report["provenance"].pop("raw_config_sha256")
    second_raw = second_report["provenance"].pop("raw_config_sha256")
    assert first_raw != second_raw
    assert first_report == second_report


def test_a_partial_publication_is_rejected_rather_than_read_as_complete(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _publish(_config(), directory)
    harvest.verify_publication(directory)
    (directory / harvest.MANIFEST_NAME).unlink()
    with pytest.raises(harvest.HarvestError, match="cannot read"):
        harvest.verify_publication(directory)


def test_a_tampered_published_file_fails_its_manifest_digest(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    _publish(_config(), directory)
    (directory / "rows.csv").write_text("corrupted\n", encoding="utf-8")
    with pytest.raises(harvest.HarvestError, match="manifest digest"):
        harvest.verify_publication(directory)


def test_regenerated_hashes_cannot_legitimize_semantically_inconsistent_rows(
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"
    _publish(_config(), directory)
    rows_path = directory / "rows.csv"
    payload = rows_path.read_text(encoding="utf-8")
    original = next(name for name in harvest.PARTITION_NAMES if f",{name}," in payload)
    replacement = next(name for name in harvest.PARTITION_NAMES if name != original)
    changed = payload.replace(f",{original},", f",{replacement},", 1)
    rows_path.write_text(changed, encoding="utf-8")
    manifest_path = directory / harvest.MANIFEST_NAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    encoded = changed.encode("utf-8")
    manifest["files"]["rows.csv"] = {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "bytes": len(encoded),
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    with pytest.raises(harvest.HarvestError, match="'partition' does not match the plan"):
        harvest.verify_publication(directory)


def _republish(
    directory: Path, report: dict[str, Any], rows: list[dict[str, Any]]
) -> None:
    """Write a mutated publication with every hash regenerated.

    Hash verification therefore passes, and only the semantic reconciliation
    stands between a tampered dataset and acceptance.
    """
    payloads = {
        "report.json": (json.dumps(report, indent=2, sort_keys=True) + "\n").encode("utf-8"),
        "rows.csv": harvest.rows_csv(rows).encode("utf-8"),
    }
    manifest = {
        "schema_version": "pde-surface-harvest-manifest/2",
        "publication_complete": True,
        "row_count": len(rows),
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in payloads.items()
        },
    }
    directory.mkdir(parents=True, exist_ok=True)
    for name, data in payloads.items():
        (directory / name).write_bytes(data)
    (directory / harvest.MANIFEST_NAME).write_bytes(
        (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )


def _mutate_first_row(rows: list[dict[str, Any]], key: str, value: Any) -> None:
    rows[0][key] = value


def _other_state(row: Mapping[str, Any]) -> str:
    return next(
        state for state in harvest.EXERCISE_STATES if state != row["exercise_state"]
    )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "volatility", 0.99),
            "'volatility' does not match",
            id="row-volatility",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "strike", 12.5),
            "'strike' does not match",
            id="row-strike",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "option_type", "call"),
            "'option_type' does not match",
            id="row-option-type",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "exercise_style", "european"),
            "'exercise_style' does not match",
            id="row-exercise-style",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "node_index", 999999),
            "outside surface",
            id="row-node-index",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "spot", 1.5),
            "'spot' does not match",
            id="row-spot",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "settings_digest", "st-0"),
            "'settings_digest' does not match",
            id="row-settings-digest",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "spot_step", 0.125),
            "'spot_step' does not match",
            id="row-grid",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "contract_multiplier", 7.0),
            "'contract_multiplier' does not match",
            id="row-contract-multiplier",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "rate", 0.99),
            "'rate' does not match",
            id="row-rate",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(
                rows, "greek_eligible", not rows[0]["greek_eligible"]
            ),
            "eligibility disagrees with its reason",
            id="row-eligibility",
        ),
        pytest.param(
            lambda report, rows: _mutate_first_row(rows, "exercise_state", _other_state(rows[0])),
            "does not match its rows",
            id="row-exercise-state",
        ),
        pytest.param(
            lambda report, rows: report["surfaces"][0].__setitem__("volatility", 0.99),
            "'volatility' does not match the plan",
            id="surface-volatility",
        ),
        pytest.param(
            lambda report, rows: report["surfaces"][0].__setitem__("settings_digest", "st-0"),
            "'settings_digest' does not match the plan",
            id="surface-settings",
        ),
        pytest.param(
            lambda report, rows: report["surfaces"][0]["solver_input"].__setitem__(
                "psor_tolerance", 1.0
            ),
            "'solver_input' does not match the plan",
            id="surface-solver-input",
        ),
        pytest.param(
            lambda report, rows: report["totals"]["counts_by_exercise_state"].__setitem__(
                "continuation", 99999
            ),
            "totals do not match recomputed totals",
            id="report-classification-count",
        ),
        pytest.param(
            lambda report, rows: report["totals"]["counts_by_greek_eligibility"].__setitem__(
                "eligible", 0
            ),
            "totals do not match recomputed totals",
            id="report-eligibility-count",
        ),
        pytest.param(
            lambda report, rows: report["totals"][
                "counts_by_greek_ineligibility_reason"
            ].__setitem__("regime_stencil_not_uniform", 4242),
            "totals do not match recomputed totals",
            id="report-reason-count",
        ),
        pytest.param(
            lambda report, rows: report["surfaces"][0]["selected_by_exercise_state"].__setitem__(
                "continuation", 0
            ),
            "node accounting does not reconcile",
            id="surface-selected-count",
        ),
    ],
)
def test_regenerated_hashes_cannot_legitimize_mutated_metadata(
    tmp_path: Path, mutate: Any, message: str
) -> None:
    config = _config(_document(option_types=["call", "put"]))
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    harvest.reconcile_report(report, rows)
    broken_report = copy.deepcopy(report)
    broken_rows = copy.deepcopy(rows)
    mutate(broken_report, broken_rows)
    directory = tmp_path / "tampered"
    _republish(directory, broken_report, broken_rows)
    with pytest.raises(harvest.HarvestError, match=message):
        harvest.verify_publication(directory)


def test_identical_bytes_under_a_different_filename_stay_semantically_identical(
    tmp_path: Path,
) -> None:
    document = _document(
        spot_intervals=200,
        time_steps=100,
        scenarios=[_scenario("a"), _scenario("b", base_volatility=0.3), _scenario("c")],
    )
    document["scenarios"][2]["expiry_time"] = 2.0
    payload = _toml(document)
    first_source = tmp_path / "design.toml"
    second_source = tmp_path / "renamed-copy.toml"
    first_source.write_text(payload, encoding="utf-8")
    second_source.write_text(payload, encoding="utf-8")
    assert first_source.read_bytes() == second_source.read_bytes()

    first = tmp_path / "run1"
    second = tmp_path / "run2"
    assert harvest.main(["--config", str(first_source), "--output-directory", str(first)]) == 0
    assert harvest.main(["--config", str(second_source), "--output-directory", str(second)]) == 0

    first_report = json.loads((first / "report.json").read_text(encoding="utf-8"))
    second_report = json.loads((second / "report.json").read_text(encoding="utf-8"))

    # Identical bytes: the raw digest, the semantic digest, every identity, every
    # assignment and rows.csv are unchanged.
    assert (
        first_report["provenance"]["raw_config_sha256"]
        == second_report["provenance"]["raw_config_sha256"]
    )
    assert (
        first_report["provenance"]["semantic_config_sha256"]
        == second_report["provenance"]["semantic_config_sha256"]
    )
    assert (first / "rows.csv").read_bytes() == (second / "rows.csv").read_bytes()
    assert harvest.canonical_payload(first_report["plan"]) == harvest.canonical_payload(
        second_report["plan"]
    )

    # Different filename: config_name differs, so report.json and the manifest
    # that pins it differ. That is provenance, not non-determinism.
    assert first_report["provenance"]["config_name"] == "design.toml"
    assert second_report["provenance"]["config_name"] == "renamed-copy.toml"
    assert (first / "report.json").read_bytes() != (second / "report.json").read_bytes()
    assert (
        first / harvest.MANIFEST_NAME
    ).read_bytes() != (second / harvest.MANIFEST_NAME).read_bytes()
    with pytest.raises(harvest.HarvestError, match="byte-identity repeat differs"):
        harvest.verify_byte_identity(first, second)

    # The narrower guarantee that does hold.
    harvest.verify_semantic_identity(first, second)
    assert "identical source-name provenance (the same config_name)" in (
        first_report["determinism"]["byte_identity_preconditions"]
    )


def test_semantic_identity_still_rejects_a_genuine_difference(tmp_path: Path) -> None:
    base = _document(
        spot_intervals=200,
        time_steps=100,
        scenarios=[_scenario("a"), _scenario("b", base_volatility=0.3), _scenario("c")],
    )
    base["scenarios"][2]["expiry_time"] = 2.0
    changed = copy.deepcopy(base)
    changed["partitioning"]["seed"] = base["partitioning"]["seed"] + 1

    directories = []
    for index, document in enumerate((base, changed)):
        source = tmp_path / f"design{index}.toml"
        source.write_text(_toml(document), encoding="utf-8")
        directory = tmp_path / f"run{index}"
        assert harvest.main(["--config", str(source), "--output-directory", str(directory)]) == 0
        directories.append(directory)
    with pytest.raises(harvest.HarvestError, match="disagree"):
        harvest.verify_semantic_identity(*directories)


def test_write_outputs_refuses_to_overwrite_without_the_flag(tmp_path: Path) -> None:
    directory = tmp_path / "run"
    config = _config()
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    harvest.write_outputs(report, rows, directory)
    with pytest.raises(harvest.HarvestError, match="refusing to overwrite"):
        harvest.write_outputs(report, rows, directory)


def test_report_counts_reconcile_exactly() -> None:
    config = _config(
        _document(option_types=["call", "put"], exercise_styles=["american", "european"])
    )
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    harvest.reconcile_report(report, rows)

    totals = report["totals"]
    assert totals["candidate_group_count"] == len(config.scenarios)
    assert totals["assigned_group_count"] == len(config.scenarios)
    assert totals["planned_surface_count"] == len(config.scenarios) * 4
    assert totals["attempted_surface_solve_count"] == len(config.scenarios) * 4
    assert totals["successful_surface_solve_count"] == len(config.scenarios) * 4
    assert totals["failed_surface_solve_count"] == 0
    assert totals["raw_row_count"] == len(rows)
    assert totals["raw_row_count"] > totals["independent_design_group_count"]
    assert totals["backward_induction_count"] == totals["successful_surface_solve_count"]
    assert sum(totals["counts_by_exercise_state"].values()) == totals["raw_row_count"]
    eligibility = totals["counts_by_greek_eligibility"]
    assert eligibility["eligible"] + eligibility["ineligible"] == totals["raw_row_count"]
    assert (
        sum(totals["counts_by_greek_ineligibility_reason"].values()) == eligibility["ineligible"]
    )

    for scope in ("train", "validation", "interpolation_test"):
        partition = report["partition_totals"][scope]
        assert partition["raw_row_count"] == sum(
            1 for row in rows if row["partition"] == scope
        )
    assert sum(
        report["partition_totals"][name]["raw_row_count"] for name in harvest.PARTITION_NAMES
    ) == len(rows)

    # A tampered report must not reconcile.
    broken = copy.deepcopy(report)
    broken["totals"]["raw_row_count"] += 1
    with pytest.raises(harvest.HarvestError):
        harvest.reconcile_report(broken, rows)


def test_yield_ratios_carry_their_own_integer_numerator_and_denominator() -> None:
    config = _config(
        _document(option_types=["call", "put"], exercise_styles=["american", "european"])
    )
    report, _rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    scopes = [report["totals"]] + [
        report["partition_totals"][name] for name in harvest.PARTITION_NAMES
    ]
    for scope in scopes:
        for name, denominator_key in (
            ("raw_rows_per_group", "independent_design_group_count"),
            ("raw_rows_per_attempted_surface_solve", "attempted_surface_solve_count"),
            ("raw_rows_per_retained_surface", "retained_surface_count"),
        ):
            ratio = scope[name]
            assert isinstance(ratio["numerator"], int)
            assert isinstance(ratio["denominator"], int)
            assert ratio["numerator"] == scope["raw_row_count"]
            assert ratio["denominator"] == scope[denominator_key]
            expected = (
                0.0
                if ratio["denominator"] == 0
                else ratio["numerator"] / ratio["denominator"]
            )
            assert ratio["value"] == expected


def test_rows_per_surface_solve_is_the_work_multiplier_not_rows_per_group() -> None:
    # Four surfaces per group, so the two ratios differ by exactly that factor
    # and rows/group must never be quoted as the computational multiplier.
    config = _config(
        _document(option_types=["call", "put"], exercise_styles=["american", "european"])
    )
    report, _rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    totals = report["totals"]
    per_group = totals["raw_rows_per_group"]
    per_solve = totals["raw_rows_per_attempted_surface_solve"]
    assert per_solve["denominator"] == 4 * per_group["denominator"]
    assert per_group["value"] == pytest.approx(4.0 * per_solve["value"])
    assert per_solve["value"] < per_group["value"]
    # The definitions travel with the numbers.
    definitions = report["interpretation"]["yield_definitions"]
    assert "never be quoted as a computational speedup" in definitions["raw_rows_per_group"]
    assert "effective sample size" in definitions["not_an_effective_sample_size"]


def test_discarded_solves_stay_in_the_attempted_denominator() -> None:
    config = _config(_document(option_types=["call", "put"]))
    plan = harvest.plan_harvest(config)
    # Fail the second surface of one group: its sibling solve still happened.
    report, rows = harvest.execute_plan(
        plan, config, solver=_SyntheticSolver(failing_surface=3)
    )
    totals = report["totals"]
    assert totals["failed_group_count"] == 1
    assert totals["planned_surface_count"] == 8
    assert totals["attempted_surface_solve_count"] == 8
    assert totals["successful_surface_solve_count"] == 7
    assert totals["failed_surface_solve_count"] == 1
    assert totals["discarded_surface_count"] == 1
    assert totals["retained_surface_count"] == 6
    per_solve = totals["raw_rows_per_attempted_surface_solve"]
    per_retained = totals["raw_rows_per_retained_surface"]
    # Attempted-solve yield is the conservative one and is reported beside it.
    assert per_solve["denominator"] == totals["attempted_surface_solve_count"]
    assert per_retained["denominator"] == totals["retained_surface_count"]
    assert per_solve["value"] < per_retained["value"]
    harvest.reconcile_report(report, rows)


def _failed_report() -> tuple[harvest.HarvestPlan, dict[str, Any], list[dict[str, Any]]]:
    config = _config(_document(option_types=["call", "put"]))
    plan = harvest.plan_harvest(config)
    report, rows = harvest.execute_plan(
        plan, config, solver=_SyntheticSolver(failing_surface=3)
    )
    return plan, report, rows


def test_reconciliation_rejects_a_row_partition_mutation_with_zero_integrity_counters() -> None:
    plan, report, rows = _failed_report()
    broken_rows = copy.deepcopy(rows)
    broken_rows[0]["partition"] = next(
        name for name in harvest.PARTITION_NAMES if name != broken_rows[0]["partition"]
    )
    assert all(value == 0 for key, value in report["integrity"].items() if isinstance(value, int))
    with pytest.raises(harvest.HarvestError, match="'partition' does not match the plan"):
        harvest.reconcile_report(report, broken_rows, plan)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda report: report["failures"].pop(),
        lambda report: report["totals"].__setitem__("failed_surface_solve_count", 99),
        lambda report: report["totals"].__setitem__("planned_surface_count", 99),
        lambda report: next(
            surface for surface in report["surfaces"] if not surface["retained"]
        ).__setitem__("retained", True),
    ],
)
def test_reconciliation_rejects_failure_count_and_retention_mutations(mutate: Any) -> None:
    plan, report, rows = _failed_report()
    broken = copy.deepcopy(report)
    mutate(broken)
    with pytest.raises(harvest.HarvestError):
        harvest.reconcile_report(broken, rows, plan)


def test_reconciliation_rejects_a_discarded_surface_contributing_rows() -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    report, rows = harvest.execute_plan(plan, config, solver=_SyntheticSolver())
    broken = copy.deepcopy(report)
    broken["surfaces"][0]["retained"] = False
    with pytest.raises(harvest.HarvestError, match="discarded surface"):
        harvest.reconcile_report(broken, rows, plan)


def test_reconciliation_rejects_unknown_surface_and_row_records() -> None:
    config = _config()
    plan = harvest.plan_harvest(config)
    report, rows = harvest.execute_plan(plan, config, solver=_SyntheticSolver())
    unknown_surface = copy.deepcopy(report)
    extra = copy.deepcopy(unknown_surface["surfaces"][0])
    extra["surface_id"] = "si-" + "f" * 64
    extra["solver_input_id"] = extra["surface_id"]
    unknown_surface["surfaces"].append(extra)
    with pytest.raises(harvest.HarvestError, match="cover every planned surface"):
        harvest.reconcile_report(unknown_surface, rows, plan)

    unknown_rows = copy.deepcopy(rows)
    extra_row = copy.deepcopy(unknown_rows[0])
    extra_row["surface_id"] = "si-" + "e" * 64
    extra_row["row_id"] = harvest.row_identity(surface_id=extra_row["surface_id"], node_index=999)
    extra_row["node_index"] = 999
    unknown_rows.append(extra_row)
    with pytest.raises(harvest.HarvestError, match="unknown or failed surface"):
        harvest.reconcile_report(report, unknown_rows, plan)


def test_a_tampered_yield_ratio_fails_reconciliation() -> None:
    config = _config()
    report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    for mutate in (
        lambda doc: doc["totals"]["raw_rows_per_attempted_surface_solve"].__setitem__(
            "value", 42.0
        ),
        lambda doc: doc["totals"]["raw_rows_per_attempted_surface_solve"].__setitem__(
            "denominator", 1
        ),
        lambda doc: doc["totals"]["raw_rows_per_group"].__setitem__("numerator", 0),
    ):
        broken = copy.deepcopy(report)
        mutate(broken)
        with pytest.raises(harvest.HarvestError):
            harvest.reconcile_report(broken, rows)


def test_row_schema_columns_match_the_published_rows() -> None:
    config = _config()
    _report, rows = harvest.run_harvest(config, solver=_SyntheticSolver())
    assert rows
    for row in rows:
        assert set(row) == set(harvest.ROW_COLUMNS)
        assert row["schema_version"] == harvest.ROW_SCHEMA_VERSION
    header = harvest.rows_csv(rows).splitlines()[0]
    assert header == ",".join(harvest.ROW_COLUMNS)


# ---------------------------------------------------------------------------
# Configuration contract
# ---------------------------------------------------------------------------


def test_the_shipped_demonstration_configuration_parses() -> None:
    config = harvest.load_harvest_config(CONFIG_PATH)
    assert config.study_status == "exploratory_infrastructure_demonstration"
    assert config.harvest.node_source == harvest.NODE_SOURCE
    assert len(config.scenarios) >= len(harvest.PARTITION_NAMES)
    plan = harvest.plan_harvest(config)
    assert {name for name, count in plan.partition_counts} == set(harvest.PARTITION_NAMES)
    assert all(count >= 1 for _name, count in plan.partition_counts)


def test_the_shipped_configuration_hash_matches_its_bytes() -> None:
    config = harvest.load_harvest_config(CONFIG_PATH)
    assert config.raw_config_sha256 == hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest()
    with CONFIG_PATH.open("rb") as stream:
        tomllib.load(stream)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda doc: doc.__setitem__("schema_version", "other/1"), "schema_version"),
        (lambda doc: doc["identity"].__setitem__("partition_group_excludes", ["spot"]),
         "contract statement"),
        (lambda doc: doc["partitioning"].__setitem__("algorithm", "random/1"), "algorithm"),
        (lambda doc: doc["partitioning"]["weights"].__setitem__("train", 0.9), "sum to 1.0"),
        (lambda doc: doc["harvest"].__setitem__("moneyness_window_low", 9.0), "below"),
        (lambda doc: doc["harvest"]["quota"].pop("exercise"), "every exercise state"),
        (lambda doc: doc["grid"].__setitem__("boundary_exclusion_nodes", 1), "stencil radius"),
        (lambda doc: doc["surfaces"].__setitem__("roles", ["sigma_up"]), "volatility_bump"),
        (lambda doc: doc["surfaces"].__setitem__("option_types", ["straddle"]), "unknown values"),
        (lambda doc: doc.__setitem__("scenarios", []), "non-empty array"),
        (lambda doc: doc["study"].__setitem__("unexpected", 1), "unknown keys"),
    ],
)
def test_invalid_configurations_are_rejected(mutate: Any, message: str) -> None:
    document = _document()
    mutate(document)
    with pytest.raises(harvest.HarvestError, match=message):
        _config(document)


def test_too_few_scenarios_for_the_declared_partitions_is_rejected() -> None:
    document = _document(scenarios=[_scenario("a"), _scenario("b")])
    with pytest.raises(harvest.HarvestError, match="cannot fill"):
        _config(document)


# ---------------------------------------------------------------------------
# 19. The compiled engine, and the untouched task 9C-A/9C-C1 APIs
# ---------------------------------------------------------------------------

_ENGINE_ARGUMENTS: dict[str, Any] = {
    "strike": STRIKE,
    "valuation_time": 0.0,
    "expiry_time": 1.0,
    "volatility": 0.2,
    "continuous_carry": 0.0,
    "curve_times": [0.0, 1.0],
    "curve_log_discounts": [0.0, -0.05],
    "dividends": [],
    "settlement": "cash",
    "contract_multiplier": 100.0,
    "spot_intervals": 400,
    "time_steps": 200,
    "spot_maximum": 400.0,
    "rannacher_steps": 2,
    "psor_tolerance": 1.0e-11,
    "psor_relaxation": 1.2,
    "psor_maximum_iterations": 50000,
}


def test_scalar_and_surface_apis_are_unchanged_by_this_module() -> None:
    scalar = pde_price(
        option_type="put", exercise_style="american", spot=95.0, **_ENGINE_ARGUMENTS
    )
    surface = pde_valuation_surface(
        option_type="put",
        exercise_style="american",
        boundary_exclusion_nodes=4,
        query_spots=[95.0],
        **_ENGINE_ARGUMENTS,
    )
    assert scalar["price"] == surface["queries"][0]["value"]
    assert surface["backward_inductions"] == 1
    assert surface["spot_intervals"] == scalar["spot_intervals"]
    # An empty query list, which is what harvesting asks for, is accepted and
    # leaves the solved slice unchanged.
    empty = pde_valuation_surface(
        option_type="put",
        exercise_style="american",
        boundary_exclusion_nodes=4,
        query_spots=[],
        **_ENGINE_ARGUMENTS,
    )
    assert empty["query_count"] == 0
    assert empty["values"] == surface["values"]


def test_real_surface_harvest_keeps_indifferent_rows_price_only() -> None:
    document = _document(
        window=(0.5, 3.9),
        spot_intervals=400,
        time_steps=200,
        scenarios=[_scenario("a"), _scenario("b", base_volatility=0.3), _scenario("c")],
    )
    document["scenarios"][2]["expiry_time"] = 2.0
    config = _config(document)
    report, rows = harvest.run_harvest(config)
    harvest.reconcile_report(report, rows)

    indifferent = [
        row for row in rows if row["exercise_state"] == "numerically_indifferent"
    ]
    assert indifferent, "the reference put fixture must expose the truncation tail"
    for row in indifferent:
        assert math.isfinite(row["price"])
        assert row["price_label_eligible"] is True
        assert row["greek_eligible"] is False
        assert row["delta_label_eligible"] is False
        assert row["greek_eligibility_reason"] == "unresolved_exercise_state"

    exercise_rows = [row for row in rows if row["exercise_state"] == "exercise"]
    assert exercise_rows
    for row in exercise_rows:
        # Pure exercise-region put nodes are correct and information-light.
        assert row["price"] == pytest.approx(max(row["strike"] - row["spot"], 0.0), abs=1.0e-6)
        if row["delta_label_eligible"]:
            assert row["delta"] == pytest.approx(-1.0, abs=1.0e-9)


def test_real_run_reconciles_and_repeats_bitwise(tmp_path: Path) -> None:
    document = _document(
        spot_intervals=200,
        time_steps=100,
        scenarios=[_scenario("a"), _scenario("b", base_volatility=0.3), _scenario("c")],
    )
    document["scenarios"][2]["expiry_time"] = 2.0
    config = _config(document)
    first_report, first_rows = harvest.run_harvest(config)
    second_report, second_rows = harvest.run_harvest(config, chunk_size=1)
    assert first_rows == second_rows
    assert first_report == second_report

    harvest.write_outputs(first_report, first_rows, tmp_path / "a")
    harvest.write_outputs(second_report, second_rows, tmp_path / "b")
    harvest.verify_byte_identity(tmp_path / "a", tmp_path / "b")
    assert first_report["totals"]["attempted_surface_solve_count"] == 3
    assert first_report["totals"]["raw_row_count"] > 3


def test_command_line_runs_the_shipped_demonstration_design(tmp_path: Path) -> None:
    document = _document(
        spot_intervals=200,
        time_steps=100,
        scenarios=[_scenario("a"), _scenario("b", base_volatility=0.3), _scenario("c")],
    )
    document["scenarios"][2]["expiry_time"] = 2.0
    source = tmp_path / "design.toml"
    source.write_text(_toml(document), encoding="utf-8")
    first = tmp_path / "run1"
    assert harvest.main(["--config", str(source), "--output-directory", str(first)]) == 0
    harvest.verify_publication(first)
    second = tmp_path / "run2"
    assert (
        harvest.main(
            [
                "--config",
                str(source),
                "--output-directory",
                str(second),
                "--chunk-size",
                "2",
                "--verify-identical-to",
                str(first),
            ]
        )
        == 0
    )


def _toml(document: dict[str, Any]) -> str:
    """Render one test document as TOML without adding a writer dependency."""

    def scalar(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, str):
            return json.dumps(value)
        if isinstance(value, float):
            return repr(value)
        if isinstance(value, int):
            return str(value)
        if isinstance(value, list):
            return "[" + ", ".join(scalar(item) for item in value) + "]"
        raise TypeError(type(value))

    lines = [f"schema_version = {scalar(document['schema_version'])}"]
    for name in ("study", "identity", "partitioning", "solver", "grid", "surfaces", "harvest"):
        table = document[name]
        lines.append(f"\n[{name}]")
        for key, value in table.items():
            if not isinstance(value, dict):
                lines.append(f"{key} = {scalar(value)}")
        for key, value in table.items():
            if isinstance(value, dict):
                lines.append(f"\n[{name}.{key}]")
                lines.extend(f"{inner} = {scalar(item)}" for inner, item in value.items())
    for entry in document["scenarios"]:
        lines.append("\n[[scenarios]]")
        lines.extend(f"{key} = {scalar(value)}" for key, value in entry.items())
    return "\n".join(lines) + "\n"
