#!/usr/bin/env python3
"""Task 9C-C2a demonstration: correlated spot rows under grouped partitioning.

One small synthetic design is planned, partitioned, solved and harvested. It
shows the three counts that a flattened dataset otherwise conflates -- raw
harvested rows, independent design groups, and actual PDE surface solves -- and
it shows that partition assignment happened before the first solve.

This is an infrastructure demonstration on exploratory settings. It is **not** a
production dataset, **not** a throughput measurement, and **not** a validation
of any label policy: task 9C-B remains ``no_policy_selected`` and is untouched
here. The elapsed lines are wall clock on one core of whatever machine ran them
and appear on the terminal only; no deterministic output file contains a time.

    python scripts/demo_pde_surface_harvest.py
"""

from __future__ import annotations

import argparse
import dataclasses
import tempfile
import time
from collections import Counter
from pathlib import Path

from differentiable_pricing.american import pde_surface_harvest as harvest

CONFIG = Path("configs/pde_surface_harvest_demo_v1.toml")
REPEAT_CHUNK_SIZE = 5


def _print_scope(title: str, scope: dict[str, object]) -> None:
    states = scope["counts_by_exercise_state"]
    eligibility = scope["counts_by_greek_eligibility"]
    rejected = scope["rejected"]
    per_group = scope["raw_rows_per_group"]
    per_solve = scope["raw_rows_per_attempted_surface"]
    print(f"  {title}")
    print(
        f"    groups {scope['independent_design_group_count']:>3}"
        f"   planned/attempted/returned/pipeline-successful "
        f"{scope['planned_surface_count']:>3}/{scope['attempted_surface_count']:>3}/"
        f"{scope['solver_returned_surface_count']:>3}/"
        f"{scope['pipeline_successful_surface_count']:>3}"
        f"   rows {scope['raw_row_count']:>4}"
    )
    print(
        f"    rows/group {per_group['value']:>6.4f}"
        f" ({per_group['numerator']}/{per_group['denominator']}, dataset expansion)"
        f"   rows/attempted-surface {per_solve['value']:>7.4f}"
        f" ({per_solve['numerator']}/{per_solve['denominator']}, work reuse)"
    )
    print(
        f"    regimes  continuation {states['continuation']:>4}"
        f"  exercise {states['exercise']:>3}"
        f"  numerically_indifferent {states['numerically_indifferent']:>3}"
        f"  no_obstacle {states['no_obstacle']:>4}"
    )
    print(
        f"    Greeks   eligible {eligibility['eligible']:>4}"
        f"  ineligible {eligibility['ineligible']:>3}"
        f"   quota shortfalls {scope['quota_shortfall_count']:>3}"
        f" ({scope['quota_shortfall_rows']} rows)"
    )
    print(
        f"    rejected endpoint {rejected['truncation_endpoint']:>3}"
        f"  buffer {rejected['boundary_buffer']:>4}"
        f"  window {rejected['outside_moneyness_window']:>6}"
        f"  non-finite {rejected['non_finite_price']:>3}"
        f"  quota {rejected['quota']:>5}"
        f"  duplicate {rejected['duplicate_node']:>3}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output-directory", type=Path, default=None)
    arguments = parser.parse_args()

    print(__doc__.splitlines()[0])
    print()

    config = harvest.load_harvest_config(arguments.config)

    # 1. Plan. This call takes no solver argument and calls none, so every
    #    partition is fixed before the first backward induction by construction.
    plan = harvest.plan_harvest(config)
    assigned = {group.partition_group_id: group.partition for group in plan.groups}
    print(
        f"planned {len(plan.groups)} candidate groups and {len(plan.surfaces)} surfaces; "
        f"{len(assigned)} groups assigned before any solve"
    )
    print(f"algorithm {plan.algorithm}  seed {plan.seed}  targets {dict(plan.partition_counts)}")
    print()

    # 2. Solve, counting the calls so the demonstration can state that no solve
    #    preceded the assignment above.
    solves = 0

    def counting_solver(**kwargs: object) -> dict[str, object]:
        nonlocal solves
        if len(assigned) != len(plan.groups):
            raise AssertionError("a solve began before every group was assigned")
        solves += 1
        return harvest.pde_valuation_surface(**kwargs)  # type: ignore[arg-type]

    started = time.perf_counter()
    report, rows = harvest.execute_plan(plan, config, solver=counting_solver)
    elapsed = time.perf_counter() - started

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
        f"  cross-partition group intersections "
        f"{integrity['cross_partition_group_intersection_count']}"
        f"  cross-partition row intersections "
        f"{integrity['cross_partition_row_intersection_count']}"
    )
    surfaces_per_partition = Counter(record["partition"] for record in report["surfaces"])
    print(f"surfaces per partition   {dict(sorted(surfaces_per_partition.items()))}")
    print(
        f"solver calls observed    {solves} "
        f"(report says {totals['attempted_surface_count']})"
    )
    print("partition assignment preceded solving: true (plan_harvest takes no solver)")
    print()

    # 3. Publish twice and compare bytes: once as planned, once from a reversed
    #    candidate order under a different chunk size.
    with tempfile.TemporaryDirectory() as scratch:
        base = Path(arguments.output_directory) if arguments.output_directory else Path(scratch)
        first = base / "run1"
        second = base / "run2"
        harvest.write_outputs(report, rows, first, overwrite=True)
        harvest.verify_publication(first)

        shuffled = dataclasses.replace(config, scenarios=tuple(reversed(config.scenarios)))
        repeat_report, repeat_rows = harvest.run_harvest(
            shuffled, chunk_size=REPEAT_CHUNK_SIZE
        )
        harvest.write_outputs(repeat_report, repeat_rows, second, overwrite=True)
        harvest.verify_publication(second)
        harvest.verify_byte_identity(first, second)
        print(
            "repeat with reversed candidate order and chunk size "
            f"{REPEAT_CHUNK_SIZE}: byte-identical outputs (report.json, rows.csv, manifest.json)"
        )
        if arguments.output_directory:
            print(f"outputs written beneath {arguments.output_directory}")

    print(f"elapsed                  {elapsed:.2f} s (terminal-only, not a throughput claim)")
    print()
    print(report["interpretation"]["rows_within_a_group_are_correlated"])
    print(report["interpretation"]["independent_design_group_count"])
    definitions = report["interpretation"]["yield_definitions"]
    print()
    print(f"rows/group: {definitions['raw_rows_per_group']}")
    print(
        "rows/attempted-surface: "
        f"{definitions['raw_rows_per_attempted_surface']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
