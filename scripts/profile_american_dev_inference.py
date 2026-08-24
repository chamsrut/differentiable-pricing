#!/usr/bin/env python3
"""Task 9H frozen-inference profile and bounded optimization: `profile`, `show`.

**Manual, terminal-invoked human command.** It is a timing measurement and needs
a quiet machine, so no test, hook, CI job or repository check calls it; an agent
reports the command and stops.

It **trains nothing, changes no weight and changes no architecture**. It
decomposes E2c's batch-1 and batch-8 wall time into named components, and
measures the closed, predeclared list of implementation variants of the
identical frozen function.

**Section D is not deferrable.** The capacity branch is decided from the
optimized implementation and its counterfactuals, never from the
pre-optimization profile; `--no-variants` leaves that branch explicitly
undecided.

Task 9H makes and measures no latency claim, and Task 9G's 10x figure is a
historical reference here, not a gate.

There is deliberately no final-evaluation subcommand and no flag that adds one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMMANDS: Final = ("profile", "show")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("profile")
    run.add_argument("--output", default=None)
    run.add_argument("--variants-output", default=None)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--warmups", type=int, default=50)
    run.add_argument("--repetitions", type=int, default=400)
    run.add_argument(
        "--no-variants",
        action="store_true",
        help="profile only; leaves the capacity branch explicitly undecided",
    )
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.frozen import FrozenCheckpointError
    from differentiable_pricing.ml.american_dev.inference_profile import (
        DEFAULT_PROFILE_OUTPUT,
        DEFAULT_VARIANTS_OUTPUT,
        InferenceProfileError,
        compact_profile,
        profile,
    )

    relative = arguments.output or DEFAULT_PROFILE_OUTPUT
    try:
        if arguments.command == "profile":
            report = profile(
                PROJECT_ROOT,
                output=relative,
                variants_output=arguments.variants_output or DEFAULT_VARIANTS_OUTPUT,
                overwrite=bool(arguments.overwrite),
                warmups=int(arguments.warmups),
                repetitions=int(arguments.repetitions),
                run_variants=not arguments.no_variants,
            )
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run profile first", file=sys.stderr)
                return 2
            report = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            "report": relative,
            "report_sha256": sha256_file(PROJECT_ROOT / relative),
            **compact_profile(report),
        }
    except (AttemptError, FrozenCheckpointError, InferenceProfileError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
