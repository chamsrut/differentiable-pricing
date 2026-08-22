#!/usr/bin/env python3
"""Task 9H European CRR-versus-Black-Scholes domain characterization: `analyze`, `show`.

**Manual, terminal-invoked human command.** `analyze` prices hundreds of
thousands of binomial lattices, so no test, hook, CI job or repository check
calls it; an agent reports the command and stops.

It is **label-free**: it measures a difference between two European valuations
of the same contract state over the declared input domain. It opens **no dataset
partition**, uses no American label, trains nothing, reserves no attempt
directory or ledger, appends nothing to the attempt log, and mutates no existing
attempt evidence.

It derives a **candidate** additive margin under the rule predeclared in
`differentiable_pricing.ml.american_dev.domain`, and writes one report beneath
the ignored `artifacts/` tree. It modifies no configuration and no source file;
applying the margin would be a separate, separately predeclared attempt. There
is deliberately **no** evaluation subcommand and no flag that adds one.
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
        "--threads",
        type=int,
        default=None,
        help="worker threads for the batch pricing boundary; throughput only, never a value",
    )
    run.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing exploratory domain report; never touches attempt evidence",
    )
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Imported here, not at module scope, so ``--help`` does not need NumPy,
    # PyTorch or the compiled pricing extensions.
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.domain import (
        DEFAULT_OUTPUT,
        DEFAULT_THREAD_COUNT,
        DomainAnalysisError,
        characterize_european_comparator,
        compact_domain,
    )

    relative = arguments.output or DEFAULT_OUTPUT
    try:
        if arguments.command == "analyze":
            report = characterize_european_comparator(
                PROJECT_ROOT,
                output=relative,
                overwrite=bool(arguments.overwrite),
                thread_count=(
                    DEFAULT_THREAD_COUNT if arguments.threads is None else int(arguments.threads)
                ),
            )
            payload = {
                "report": relative,
                "report_sha256": sha256_file(PROJECT_ROOT / relative),
                **compact_domain(report),
            }
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run analyze first", file=sys.stderr)
                return 2
            payload = {
                "report": relative,
                "report_sha256": sha256_file(path),
                **compact_domain(json.loads(path.read_text(encoding="utf-8"))),
            }
    except (AttemptError, DomainAnalysisError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
