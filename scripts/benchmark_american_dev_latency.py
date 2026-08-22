#!/usr/bin/env python3
"""Task 9H matched latency diagnostic: `benchmark`, `show`.

**Manual, terminal-invoked human command.** `benchmark` is a timing measurement,
so its numbers depend on the machine being otherwise quiet; no test, hook, CI job
or repository check calls it, and an agent reports the command and stops.

It measures the deployed neural path of one recorded Task 9H checkpoint against
the matched adjacent-average CRR comparator at the label policy's own depth,
under the Task 9G latency contract and its implementation, reused rather than
restated. It opens **no dataset partition**, trains nothing, reserves no attempt
directory or ledger, appends nothing to the attempt log and mutates no existing
attempt evidence.

The result is an **ungated exploratory diagnostic with no validation exposure**.
Task 9H's scope is price only and it makes no latency claim: no gate is applied,
and this measurement neither passes nor fails a Task 9G gate. There is
deliberately **no** evaluation subcommand and no flag that adds one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMMANDS: Final = ("benchmark", "show")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("benchmark")
    run.add_argument("--output", default=None)
    run.add_argument(
        "--attempt-directory",
        default=None,
        help="ignored artifacts/ directory holding the recorded attempt's checkpoint",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing exploratory latency report; never touches attempt evidence",
    )
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Imported here, not at module scope, so ``--help`` does not need PyTorch or
    # the compiled pricing extensions.
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.latency import (
        DEFAULT_ATTEMPT_DIRECTORY,
        DEFAULT_OUTPUT,
        LatencyDiagnosticError,
        benchmark_matched_latency,
        compact_latency,
    )

    relative = arguments.output or DEFAULT_OUTPUT
    try:
        if arguments.command == "benchmark":
            report = benchmark_matched_latency(
                PROJECT_ROOT,
                output=relative,
                attempt_directory=(
                    arguments.attempt_directory or DEFAULT_ATTEMPT_DIRECTORY
                ),
                overwrite=bool(arguments.overwrite),
            )
            payload = {
                "report": relative,
                "report_sha256": sha256_file(PROJECT_ROOT / relative),
                **compact_latency(report),
            }
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run benchmark first", file=sys.stderr)
                return 2
            payload = {
                "report": relative,
                "report_sha256": sha256_file(path),
                **compact_latency(json.loads(path.read_text(encoding="utf-8"))),
            }
    except (AttemptError, LatencyDiagnosticError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
