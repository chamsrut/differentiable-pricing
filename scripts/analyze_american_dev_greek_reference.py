#!/usr/bin/env python3
"""Task 9H CRR Greek reference contract: `analyze`, `show`.

**Manual, terminal-invoked human command.** It prices tens of thousands of
binomial lattices, so no test, hook, CI job or repository check calls it; an
agent reports the command and stops.

It is **label-free and partition-free**: the declared contract set is a fixed
deterministic draw over the domain interior, so the reference is not tuned to
the rows it will later be applied to. It opens no dataset partition and trains
nothing.

It selects **separate** bumps for Delta and Gamma by plateau, records each one's
reference uncertainty, and runs the depth-convergence check at N = 2048/2049 for
prices as well as Greeks.

Scope is Delta, Gamma and Vega. Theta and rho are recorded as declared
non-claims.

There is deliberately no final-evaluation subcommand and no flag that adds one.
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
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--threads", type=int, default=8)
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.greek_reference import (
        DEFAULT_OUTPUT,
        GreekReferenceError,
        analyze,
        compact_reference,
    )

    relative = arguments.output or DEFAULT_OUTPUT
    try:
        if arguments.command == "analyze":
            report = analyze(
                PROJECT_ROOT,
                output=relative,
                overwrite=bool(arguments.overwrite),
                thread_count=int(arguments.threads),
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
            **compact_reference(report),
        }
    except (AttemptError, GreekReferenceError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
