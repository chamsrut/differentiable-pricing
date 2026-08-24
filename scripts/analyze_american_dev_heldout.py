#!/usr/bin/env python3
"""Task 9H H1/H2 carve declaration: `declare`, `show`.

**Manual, terminal-invoked human command.** It opens the `train` partition to
read its sample identifiers, so no test, hook, CI job or repository check calls
it; an agent reports the command and stops.

It **trains nothing and scores nothing**. It carves 50,000 of the 167,232 train
rows no model in this project has ever read, splits them into two disjoint
25,000-row halves in advance, and records counts, digests and verified
disjointness **before** either half is evaluated.

**H2 is never turned into rows.** The declaration hashes it; nothing else can
reach it.

There is deliberately no final-evaluation subcommand and no flag that adds one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMMANDS: Final = ("declare", "show")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("declare")
    run.add_argument("--output", default=None)
    run.add_argument("--overwrite", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.heldout import (
        DEFAULT_OUTPUT,
        HeldoutAccessError,
        compact_carve,
        declare_carve,
    )

    relative = arguments.output or DEFAULT_OUTPUT
    try:
        if arguments.command == "declare":
            report = declare_carve(
                PROJECT_ROOT, output=relative, overwrite=bool(arguments.overwrite)
            )
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run declare first", file=sys.stderr)
                return 2
            report = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            "report": relative,
            "report_sha256": sha256_file(PROJECT_ROOT / relative),
            **compact_carve(report),
        }
    except (AttemptError, HeldoutAccessError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
