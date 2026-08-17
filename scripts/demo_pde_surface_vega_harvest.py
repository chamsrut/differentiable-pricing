#!/usr/bin/env python3
"""Task 9C-C2b1 demonstration: three-surface vega and authoritative verification.

One small synthetic design is planned, partitioned, solved three times per
contract leg and harvested. It shows

* that every group costs exactly three planned surfaces and that rows come only
  from the base one;
* the predeclared centered vega, per unit absolute volatility, checked against
  Black--Scholes for a European call and against a twice-refined PDE control for
  an American put;
* the difference between self-contained consistency verification and
  authoritative verification against an externally supplied configuration.

This is an infrastructure demonstration on exploratory settings. It is **not** a
production dataset, **not** a throughput measurement, **not** a label policy and
**not** a stability decision about vega: task 9C-B remains
``no_policy_selected`` and is untouched here, and whether a vega may be
supervised is task 9C-C3's question. The elapsed lines are wall clock on one
core of whatever machine ran them and appear on the terminal only; no
deterministic output file contains a time.

    python scripts/demo_pde_surface_vega_harvest.py
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

from differentiable_pricing import black_scholes, pde_valuation_surface
from differentiable_pricing.american import pde_surface_harvest as harvest

CONFIG = Path("configs/pde_surface_vega_harvest_demo_v1.toml")
REPEAT_CHUNK_SIZE = 4
CONTROL_REFINEMENT = 2
CONTROL_GROUPS = 0  # 0 means every planned group


def _print_scope(title: str, scope: dict[str, Any]) -> None:
    per_group = scope["raw_rows_per_group"]
    per_solve = scope["raw_rows_per_attempted_surface"]
    per_retained = scope["raw_rows_per_retained_surface"]
    vega = scope["counts_by_vega_numerical_availability"]
    regimes = scope["counts_by_vega_bump_exercise_regime"]
    print(f"  {title}")
    print(
        f"    groups {scope['independent_design_group_count']:>3}"
        f"   planned/attempted/returned/pipeline-successful "
        f"{scope['planned_surface_count']:>3}/{scope['attempted_surface_count']:>3}/"
        f"{scope['solver_returned_surface_count']:>3}/"
        f"{scope['pipeline_successful_surface_count']:>3}"
        f"   solver-failed/post-solve-failed "
        f"{scope['solver_failed_surface_count']:>2}/"
        f"{scope['post_solve_pipeline_failed_surface_count']:>2}"
        f"   retained/discarded {scope['retained_surface_count']:>3}/"
        f"{scope['discarded_surface_count']:>2}"
        f"   rows {scope['raw_row_count']:>4}"
    )
    print(
        f"    rows/group {per_group['value']:>7.4f}"
        f" ({per_group['numerator']}/{per_group['denominator']}, dataset expansion)"
        f"   rows/attempted-surface {per_solve['value']:>7.4f}"
        f" ({per_solve['numerator']}/{per_solve['denominator']}, work reuse)"
        f"   rows/retained-solve {per_retained['value']:>7.4f}"
    )
    print(
        f"    vega available {vega['available']:>4}  unavailable {vega['unavailable']:>3}"
        f"   bump regime unchanged {regimes['unchanged_across_the_bump']:>4}"
        f"  changed {regimes['changed_across_the_bump']:>3}"
    )


def _european_design(config: harvest.HarvestConfig) -> harvest.HarvestConfig:
    """The same scenarios as a European call, so Black--Scholes is a comparator."""
    return dataclasses.replace(
        config,
        surfaces=dataclasses.replace(
            config.surfaces, option_types=("call",), exercise_styles=("european",)
        ),
    )


def _analytic_vegas(row: dict[str, Any]) -> tuple[float, float]:
    arguments = {
        "option_type": str(row["option_type"]),
        "spot": float(row["spot"]),
        "strike": float(row["strike"]),
        "maturity": float(row["expiry_time"]) - float(row["valuation_time"]),
        "rate": float(row["rate"]),
        "dividend_yield": float(row["continuous_carry"]),
    }
    sigma = float(row["volatility"])
    bump = float(row["vega_bump"])
    up = black_scholes(volatility=sigma + bump, **arguments)["price"]
    down = black_scholes(volatility=sigma - bump, **arguments)["price"]
    return black_scholes(volatility=sigma, **arguments)["vega"], (up - down) / (2.0 * bump)


def _refined_surface(descriptor: dict[str, Any], refinement: int) -> dict[str, Any]:
    """Solve one planned surface on a grid with ``refinement`` times the nodes."""
    return pde_valuation_surface(
        option_type=descriptor["option_type"],
        exercise_style=descriptor["exercise_style"],
        strike=descriptor["strike"],
        valuation_time=descriptor["valuation_time"],
        expiry_time=descriptor["expiry_time"],
        volatility=descriptor["volatility"],
        continuous_carry=descriptor["continuous_carry"],
        curve_times=list(descriptor["curve_times"]),
        curve_log_discounts=list(descriptor["curve_log_discounts"]),
        dividends=[list(item) for item in descriptor["dividends"]],
        settlement=descriptor["settlement"],
        # Reporting-only and excluded from the solver identity, so any positive
        # value gives bitwise the same slice.
        contract_multiplier=1.0,
        spot_intervals=int(descriptor["spot_intervals"]) * refinement,
        time_steps=int(descriptor["time_steps"]) * refinement,
        spot_maximum=float(descriptor["spot_maximum"]),
        rannacher_steps=int(descriptor["rannacher_steps"]),
        psor_tolerance=float(descriptor["psor_tolerance"]),
        psor_relaxation=float(descriptor["psor_relaxation"]),
        psor_maximum_iterations=int(descriptor["psor_maximum_iterations"]),
        boundary_exclusion_nodes=int(descriptor["boundary_exclusion_nodes"]),
        query_spots=[],
    )


def _american_control(
    report: dict[str, Any], rows: list[dict[str, Any]], refinement: int, groups: int
) -> tuple[float, float, int, int, str]:
    """Compare every harvested vega against a twice-refined centered control.

    The refined grid places a node on the strike exactly as the coarse one does,
    so every coarse node index ``i`` is refined node ``refinement * i`` at
    bitwise the same spot. That is asserted, not assumed.
    """
    rows_by_group: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_group.setdefault(str(row["partition_group_id"]), []).append(row)

    worst = 0.0
    worst_eligible = 0.0
    worst_reason = "eligible"
    compared = 0
    solves = 0
    planned = report["plan"]["groups"]
    for group in planned if groups <= 0 else planned[:groups]:
        group_rows = rows_by_group.get(str(group["partition_group_id"]), [])
        if not group_rows:
            continue
        descriptors = {
            str(surface["surface_role"]): dict(surface["solver_input"])
            for surface in group["surfaces"]
        }
        if set(descriptors) != set(harvest.SURFACE_ROLES):
            continue
        refined = {}
        for role in ("sigma_down", "sigma_up"):
            refined[role] = _refined_surface(descriptors[role], refinement)
            solves += 1
        for row in group_rows:
            index = int(row["node_index"]) * refinement
            for surface in refined.values():
                if surface["spot_nodes"][index] != row["spot"]:
                    raise SystemExit("refined control grid does not contain the coarse node")
            bump = float(row["vega_bump"])
            control = (
                refined["sigma_up"]["values"][index] - refined["sigma_down"]["values"][index]
            ) / (2.0 * bump)
            difference = abs(float(row["vega"]) - control)
            if difference > worst:
                worst = difference
                worst_reason = str(row["greek_eligibility_reason"])
            if row["greek_eligible"]:
                worst_eligible = max(worst_eligible, difference)
            compared += 1
    return worst, worst_eligible, compared, solves, worst_reason


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-directory", type=Path, default=None)
    parser.add_argument("--control-refinement", type=int, default=CONTROL_REFINEMENT)
    parser.add_argument(
        "--control-groups",
        type=int,
        default=CONTROL_GROUPS,
        help="how many planned groups to compare against the refined control (0 = all)",
    )
    arguments = parser.parse_args()

    print(__doc__.splitlines()[0])
    print()

    config = harvest.load_harvest_config(arguments.config)

    # 1. Plan. This call takes no solver argument and calls none, so every
    #    partition is fixed before the first backward induction, and a group's
    #    three volatility surfaces are members of one partition by construction.
    plan = harvest.plan_harvest(config)
    per_group = {len(group.surfaces) for group in plan.groups}
    print(
        f"planned {len(plan.groups)} candidate groups and {len(plan.surfaces)} surfaces; "
        f"surfaces per group {sorted(per_group)}"
    )
    print(
        f"roles {list(config.surfaces.roles)}   vega bump eta = "
        f"{config.surfaces.vega_bump!r} (absolute volatility)"
    )
    print(f"convention  {harvest.VEGA_CONVENTION}   units {harvest.VEGA_UNITS}")
    print(f"algorithm {plan.algorithm}  seed {plan.seed}  targets {dict(plan.partition_counts)}")
    print()

    solves = 0

    def counting_solver(**kwargs: Any) -> dict[str, Any]:
        nonlocal solves
        solves += 1
        return harvest.pde_valuation_surface(**kwargs)  # type: ignore[arg-type]

    started = time.perf_counter()
    report, rows = harvest.execute_plan(plan, config, solver=counting_solver)
    harvest_elapsed = time.perf_counter() - started

    totals = report["totals"]
    print("global")
    _print_scope("all partitions", totals)
    print()
    for name in harvest.PARTITION_NAMES:
        _print_scope(name, report["partition_totals"][name])
    print()

    integrity = report["integrity"]
    print(
        f"integrity  duplicate groups {integrity['duplicate_partition_group_count']}"
        f"  duplicate rows {integrity['duplicate_row_id_count']}"
        f"  straddling groups {len(integrity['groups_in_more_than_one_partition'])}"
        f"  cross-partition group/row intersections "
        f"{integrity['cross_partition_group_intersection_count']}/"
        f"{integrity['cross_partition_row_intersection_count']}"
    )
    print(f"distinct states per partition  {integrity['distinct_economic_states_per_partition']}")
    surfaces_per_partition = Counter(record["partition"] for record in report["surfaces"])
    row_sources = Counter(
        record["surface_role"] for record in report["surfaces"] if record["is_row_source"]
    )
    print(f"surfaces per partition   {dict(sorted(surfaces_per_partition.items()))}")
    print(f"row-source roles         {dict(row_sources)} (price, delta and gamma come from base)")
    print(
        f"solver calls observed    {solves} "
        f"(report says {totals['attempted_surface_count']})"
    )
    print()

    # 2. American vega against a twice-refined centered PDE control.
    started = time.perf_counter()
    worst, worst_eligible, compared, control_solves, worst_reason = _american_control(
        report, rows, arguments.control_refinement, arguments.control_groups
    )
    control_elapsed = time.perf_counter() - started
    print(
        f"american vega vs {arguments.control_refinement}x-refined centered PDE control "
        f"({compared} rows, {control_solves} extra solves)"
    )
    print(f"  worst absolute difference, all rows            {worst:.3e}")
    print(f"  worst absolute difference, Greek-eligible rows {worst_eligible:.3e}")
    if worst > worst_eligible:
        print(
            f"  the worst row is one the surface contract already refuses as a Greek label, "
            f"with reason '{worst_reason}'."
        )
    print(
        "  vega is not gated by that flag -- it comes from prices, not from a stencil -- which "
        "is exactly the"
    )
    print("  stability question task 9C-C3 must answer. This demonstration answers none of it.")
    print()

    # 3. European vega against Black-Scholes, on the same scenarios.
    started = time.perf_counter()
    european_config = _european_design(config)
    european_report, european_rows = harvest.run_harvest(european_config)
    european_elapsed = time.perf_counter() - started
    worst_derivative = 0.0
    worst_centered = 0.0
    # A cash dividend invalidates Black--Scholes, so those scenarios are excluded
    # from the analytic comparison rather than compared against a formula that
    # does not price them.
    comparable = [row for row in european_rows if int(row["dividend_count"]) == 0]
    for row in comparable:
        analytic, centered = _analytic_vegas(row)
        worst_derivative = max(worst_derivative, abs(float(row["vega"]) - analytic))
        worst_centered = max(worst_centered, abs(float(row["vega"]) - centered))
    print(
        f"european vega vs Black-Scholes ({len(comparable)} of {len(european_rows)} rows "
        f"from {european_report['totals']['attempted_surface_count']} attempted surfaces; "
        "cash-dividend scenarios excluded, Black-Scholes does not price them)"
    )
    print(f"  worst absolute error vs the analytic derivative        {worst_derivative:.3e}")
    print(f"  worst absolute error vs the analytic centered difference {worst_centered:.3e}")
    print("  error against analytic Black--Scholes vega includes both PDE-grid error and")
    print("  finite-bump truncation. The same-eta analytic centered difference isolates the")
    print("  PDE-grid component pointwise: refinement reduces that component but cannot remove")
    print("  the finite-bump component. These worst-case maxima may occur at different rows")
    print("  and must not be subtracted to estimate truncation error.")
    print()

    # 4. Publication, both verification guarantees, and byte identity.
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(arguments.output_directory) if arguments.output_directory else Path(scratch)
        first = base / "run1"
        second = base / "run2"
        harvest.write_outputs(report, rows, first, overwrite=True)

        harvest.verify_publication(first)
        print(f"{harvest.CONSISTENCY_VERIFICATION}: passed (publication agrees with itself)")
        harvest.verify_publication_authoritatively(first, expected_config=config)
        print(
            f"{harvest.AUTHORITATIVE_VERIFICATION}: passed against "
            f"{arguments.config} (replanned, nothing solved)"
        )
        try:
            harvest.verify_training_input_publication(first, expected_config=config)
        except harvest.HarvestError as error:
            print(f"training-input gate refused, as it must: {error}".rsplit(":", 1)[0])

        shuffled = dataclasses.replace(config, scenarios=tuple(reversed(config.scenarios)))
        repeat_report, repeat_rows = harvest.run_harvest(shuffled, chunk_size=REPEAT_CHUNK_SIZE)
        harvest.write_outputs(repeat_report, repeat_rows, second, overwrite=True)
        harvest.verify_publication(second)
        harvest.verify_byte_identity(first, second)
        print(
            "repeat with reversed candidate order and chunk size "
            f"{REPEAT_CHUNK_SIZE}: byte-identical outputs (report.json, rows.csv, manifest.json)"
        )
        if arguments.output_directory:
            print(f"outputs written beneath {arguments.output_directory}")
    print()

    print(
        f"elapsed  harvest {harvest_elapsed:.2f} s   american control {control_elapsed:.2f} s"
        f"   european check {european_elapsed:.2f} s"
        "   (terminal-only, not a throughput claim)"
    )
    print()
    print(report["interpretation"]["vega_numerical_availability"])
    print()
    print(report["interpretation"]["vega_rows_cost_three_solves"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
