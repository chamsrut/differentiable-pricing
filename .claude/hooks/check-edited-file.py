#!/usr/bin/env python3
"""Validate an edited file without modifying it."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path


def fail(message: str) -> int:
    print(f"post-edit check: {message}", file=sys.stderr)
    return 2


def run(command: list[str]) -> int:
    completed = subprocess.run(command, check=False)
    return completed.returncode


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError as error:
        return fail(f"invalid hook input: {error}")

    tool_input = event.get("tool_input", {})
    raw_path = tool_input.get("file_path") or tool_input.get("path")
    if not raw_path:
        return 0

    root = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd())).resolve()
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()

    try:
        path.relative_to(root)
    except ValueError:
        return fail(f"refusing to inspect a file outside the project: {path}")

    if not path.is_file():
        return 0

    try:
        if path.suffix == ".json":
            with path.open(encoding="utf-8") as stream:
                json.load(stream)
        elif path.suffix == ".toml":
            with path.open("rb") as stream:
                tomllib.load(stream)
        elif path.suffix == ".py":
            if shutil.which("ruff"):
                return run(["ruff", "check", str(path)])
            return run([sys.executable, "-m", "py_compile", str(path)])
        elif path.suffix in {".cc", ".cpp", ".cxx", ".h", ".hpp", ".hxx"}:
            if shutil.which("clang-format"):
                return run(["clang-format", "--dry-run", "--Werror", str(path)])
    except (OSError, ValueError, tomllib.TOMLDecodeError) as error:
        return fail(f"{path}: {error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
