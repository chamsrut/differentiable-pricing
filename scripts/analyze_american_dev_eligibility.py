#!/usr/bin/env python3
"""Task 9H assessability artifact: `analyze`, `show`.

**Manual, terminal-invoked human command.** It opens a dataset partition, so no
test, hook, CI job or repository check calls it; an agent reports the command
and stops.

**Run this before scoring anything.** It reads **no model and no prediction**:
whether the primary pass-rate statistic can honestly be reported as a number is
settled by the declared tolerances and the label operator alone.

The price-fidelity analysis refuses to score a row set until this artifact
exists and matches its rows, its declaration digests and its protocol commit.
That ordering is the point: choosing a denominator after seeing a score is
exactly what it prevents.

There is deliberately no final-evaluation subcommand, and the reserved half is
not an assessable row set.
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
    show = commands.add_parser("show")
    show.add_argument("--row-set", required=True)
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.eligibility import (
        DEFAULT_OUTPUT_TEMPLATE,
        EligibilityError,
        compact,
        load,
        load_row_set,
        write,
    )

    row_set = str(arguments.row_set)
    try:
        relative = arguments.output or DEFAULT_OUTPUT_TEMPLATE.format(row_set=row_set)
        if arguments.command == "analyze":
            columns = load_row_set(PROJECT_ROOT, row_set)
            report, _ = write(
                PROJECT_ROOT,
                row_set,
                columns,
                output=relative,
                overwrite=bool(arguments.overwrite),
            )
        else:
            report = load(PROJECT_ROOT, row_set, output=relative)
        payload = {
            "report": relative,
            "report_sha256": sha256_file(PROJECT_ROOT / relative),
            **compact(report),
        }
    except (AttemptError, EligibilityError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
