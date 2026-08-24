#!/usr/bin/env python3
"""Task 9H price fidelity: `analyze`, `show`.

**Manual, terminal-invoked human command.** It opens a dataset partition and
reads two frozen checkpoints, so no test, hook, CI job or repository check calls
it; an agent reports the command and stops.

It **trains nothing and changes no weight**. It runs the fixed-weight
margin/attempt-variation 2x2 and the B.3 characterization, for the raw network
and the deployed output separately, under the predeclared tolerance.

**It refuses to start without the matching eligibility artifact.** Assessability
is settled before any prediction exists; run the eligibility analysis first.

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
    run.add_argument("--row-set", required=True, help="validation or H1")
    run.add_argument("--output", default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--batch-size", type=int, default=4096)
    show = commands.add_parser("show")
    show.add_argument("--row-set", required=True)
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.eligibility import EligibilityError
    from differentiable_pricing.ml.american_dev.frozen import FrozenCheckpointError
    from differentiable_pricing.ml.american_dev.price_fidelity import (
        DEFAULT_OUTPUT_TEMPLATE,
        PriceFidelityError,
        analyze,
        compact_fidelity,
    )

    row_set = str(arguments.row_set)
    try:
        relative = arguments.output or DEFAULT_OUTPUT_TEMPLATE.format(row_set=row_set)
        if arguments.command == "analyze":
            report = analyze(
                PROJECT_ROOT,
                row_set,
                output=relative,
                overwrite=bool(arguments.overwrite),
                batch_size=int(arguments.batch_size),
            )
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run analyze first", file=sys.stderr)
                return 2
            report = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            "report": relative,
            "report_sha256": sha256_file(PROJECT_ROOT / relative),
            **compact_fidelity(report),
        }
    except (
        AttemptError,
        EligibilityError,
        FrozenCheckpointError,
        PriceFidelityError,
        OSError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
