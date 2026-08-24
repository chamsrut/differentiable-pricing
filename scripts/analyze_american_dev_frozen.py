#!/usr/bin/env python3
"""Task 9H frozen-checkpoint manifest: `freeze`, `show`.

**Manual, terminal-invoked human command.** It reads the ignored artifact tree
two completed attempts wrote, so no test, hook, CI job or repository check calls
it; an agent reports the command and stops.

It **trains nothing, changes no weight, changes no configuration** and opens no
dataset partition. It records what E2b and E2c are: architecture, parameter
count, seeds, best epoch, head parameters, digests, and the deployed function
written out mathematically.

Writing the manifest requires a **clean tracked worktree**, which is how a
decision-bearing artifact cannot be produced before the protocol commit.

There is deliberately no final-evaluation subcommand and no flag that adds one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMMANDS: Final = ("freeze", "show")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("freeze")
    run.add_argument("--output", default=None)
    run.add_argument("--overwrite", action="store_true")
    show = commands.add_parser("show")
    show.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Imported here so ``--help`` needs neither PyTorch nor the compiled
    # pricing extensions.
    from differentiable_pricing.ml.american_dev.attempts import AttemptError, sha256_file
    from differentiable_pricing.ml.american_dev.frozen import (
        DEFAULT_OUTPUT,
        FrozenCheckpointError,
        compact_frozen,
        freeze_checkpoints,
    )

    relative = arguments.output or DEFAULT_OUTPUT
    try:
        if arguments.command == "freeze":
            report = freeze_checkpoints(
                PROJECT_ROOT, output=relative, overwrite=bool(arguments.overwrite)
            )
        else:
            path = PROJECT_ROOT / relative
            if not path.is_file():
                print(f"error: '{relative}' does not exist; run freeze first", file=sys.stderr)
                return 2
            report = json.loads(path.read_text(encoding="utf-8"))
        payload = {
            "report": relative,
            "report_sha256": sha256_file(PROJECT_ROOT / relative),
            **compact_frozen(report),
        }
    except (AttemptError, FrozenCheckpointError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
