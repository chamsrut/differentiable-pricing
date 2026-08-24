#!/usr/bin/env python3
"""Task 9H Greek fidelity: `analyze`, `show`.

**Manual, terminal-invoked human command.** Grid 1 prices hundreds of thousands
of binomial lattices and opens a dataset partition, so no test, hook, CI job or
repository check calls it; an agent reports the command and stops.

It **trains nothing and changes no weight**. Every grid computes raw-network and
deployed Greeks by autograd at identical states -- the split that separates "the
learned function has a derivative problem" from "the projection has one", which
lead to different interventions.

Grid 1 is absolute fidelity against the label operator and is genuinely limited
by CRR finite-difference uncertainty; conclusions may be drawn only where the
disagreement exceeds it. Grids 2 and 2b are label-free crossover scans whose
primary statistic is the deployed-minus-raw autograd Gamma difference, which
carries no finite-difference uncertainty at all.

Scope is Delta, Gamma and Vega. Theta and rho are declared non-claims.

There is deliberately no final-evaluation subcommand, and the reserved half is
not a scorable row set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMMANDS: Final = ("analyze", "show")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("analyze")
    run.add_argument("--grid", required=True, choices=("1", "2", "2b"))
    run.add_argument("--row-set", default=None, help="grid 1 only: validation or H1")
    run.add_argument("--output", default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--threads", type=int, default=8)
    run.add_argument("--batch-size", type=int, default=4096)
    show = commands.add_parser("show")
    show.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.frozen import FrozenCheckpointError
    from differentiable_pricing.ml.american_dev.greek_fidelity import (
        GreekFidelityError,
        analyze,
        compact_grid,
    )

    try:
        if arguments.command == "analyze":
            report = analyze(
                PROJECT_ROOT,
                str(arguments.grid),
                row_set=arguments.row_set,
                output=arguments.output,
                overwrite=bool(arguments.overwrite),
                thread_count=int(arguments.threads),
                batch_size=int(arguments.batch_size),
            )
            suffix = f"-{arguments.row_set}" if arguments.grid == "1" else ""
            relative = arguments.output or (
                f"artifacts/task-9h/greeks/grid{arguments.grid}{suffix}-v1.json"
            )
        else:
            relative = str(arguments.output)
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run analyze first", file=sys.stderr)
                return 2
            report = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            "report": relative,
            "report_sha256": sha256_file(PROJECT_ROOT / relative),
            **compact_grid(report),
        }
    except (AttemptError, FrozenCheckpointError, GreekFidelityError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
