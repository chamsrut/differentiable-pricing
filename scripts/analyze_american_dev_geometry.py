#!/usr/bin/env python3
"""Task 9H validation-set geometry: `analyze`, `show`.

**Manual, terminal-invoked human command.** `analyze` opens a dataset partition,
so no test, hook, CI job or repository check calls it; an agent reports the
command and stops. It trains nothing, reserves no attempt directory or ledger,
appends nothing to the attempt log and mutates no existing attempt evidence.

The analysis partition is a module constant in
`differentiable_pricing.ml.american_dev.geometry`, not an argument: there is no
flag, subcommand or configuration key here that can point this analysis at any
other partition, and `python3 scripts/american_dev_attempts.py check` re-verifies
that offline. There is deliberately **no** final-evaluation subcommand and no
flag that adds one.

The report it writes is **exploratory and validation-selected**. Task 9H selects
against the same partition, repeatedly, so a geometry measured on it is a
development observation and never a project result.
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
    run.add_argument("--output", default=None)
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing exploratory geometry report; never touches attempt evidence",
    )
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Imported here, not at module scope, so ``--help`` does not need PyTorch,
    # pyarrow or the compiled pricing extensions.
    from differentiable_pricing.ml.american_dev.attempts import AttemptError
    from differentiable_pricing.ml.american_dev.geometry import (
        DEFAULT_OUTPUT,
        GeometryError,
        analyze_validation_geometry,
        compact_geometry,
    )
    from differentiable_pricing.ml.american_dev.workbench import WorkbenchError

    relative = arguments.output or DEFAULT_OUTPUT
    try:
        if arguments.command == "analyze":
            report = analyze_validation_geometry(
                PROJECT_ROOT, output=relative, overwrite=bool(arguments.overwrite)
            )
            payload = {"report": relative, **compact_geometry(report)}
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run analyze first", file=sys.stderr)
                return 2
            payload = compact_geometry(json.loads(path.read_text(encoding="utf-8")))
    except (AttemptError, GeometryError, WorkbenchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
