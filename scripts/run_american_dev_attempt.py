#!/usr/bin/env python3
"""Task 9H price-training runner: run, status.

Training is manual, terminal-invoked human work. This module is importable for
fixture tests, but no test, hook, CI job or repository check calls the expensive
execution function.

There is deliberately **no** final-evaluation subcommand and no flag that adds
one. Task 9H reaches ``train`` and ``validation`` and nothing else; a candidate
that meets the development criterion needs a separately predeclared confirmation
on a fresh final partition, as its own task.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
COMMANDS: Final = ("run", "status")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        sub = commands.add_parser(name)
        sub.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    # Imported here, not at module scope, so ``--help`` does not need PyTorch or
    # the compiled pricing extensions.
    from differentiable_pricing.ml.american_dev.attempts import AttemptError
    from differentiable_pricing.ml.american_dev.workbench import (
        WorkbenchError,
        attempt_status,
        execute_attempt,
    )

    config = arguments.config.resolve()
    try:
        if arguments.command == "run":
            report = execute_attempt(config, PROJECT_ROOT)
            payload = {
                "attempt_id": report["attempt_id"],
                "commit": report["repository"]["commit"],
                "gate": report["price_metrics"]["gate"],
                "summary": f"{report['attempt_id']}/summary.json",
            }
        else:
            payload = attempt_status(config, PROJECT_ROOT)
    except (AttemptError, WorkbenchError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
