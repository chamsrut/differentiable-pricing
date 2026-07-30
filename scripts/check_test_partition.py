#!/usr/bin/env python3
"""Assert the CI test partition stays a real partition.

Continuous integration splits the Python suite by directory:

* the lightweight job installs ``.[dev,data]`` (no PyTorch) and runs
  ``pytest -q --ignore=python/tests/ml``;
* the ML job installs ``.[dev,train]`` and runs ``pytest -q python/tests/ml``.

Those two commands cover ``python/tests`` exactly once each only while no test
module outside ``python/tests/ml`` reaches PyTorch. A module that imports
``torch`` directly, or that imports the ``differentiable_pricing.ml`` package
(whose modules import ``torch`` at import time), would fail collection in the
lightweight job. This check fails loudly at that moment instead of leaving a
red pipeline to be diagnosed from a collection traceback.

The import scan is static: it walks the AST rather than importing anything, so
it runs in the lightweight environment where PyTorch is absent. That also bounds
what it can see — a dynamic ``importlib.import_module("torch")`` inside a test
body would evade it. Nothing in this repository imports that way, and the lint
configuration keeps imports static, so the check matches how the tests are
actually written; a future dynamic import would surface as a lightweight-job
collection error rather than being caught here.
"""

from __future__ import annotations

import ast
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
TEST_ROOT: Final = PROJECT_ROOT / "python" / "tests"
ML_TEST_ROOT: Final = TEST_ROOT / "ml"

# Importing any of these pulls in PyTorch at module import time.
TORCH_BACKED_ROOTS: Final = frozenset({"torch", "differentiable_pricing.ml"})


def _imported_modules(tree: ast.AST) -> Iterator[str]:
    """Yield every module name bound by an import statement anywhere in a file."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def _requires_torch(path: Path) -> set[str]:
    """Return the torch-backed modules a test file imports, if any."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = set()
    for module in _imported_modules(tree):
        for root in TORCH_BACKED_ROOTS:
            if module == root or module.startswith(f"{root}."):
                offenders.add(module)
    return offenders


def main() -> int:
    if not TEST_ROOT.is_dir():
        print(f"error: test root '{TEST_ROOT}' does not exist", file=sys.stderr)
        return 2

    all_tests = sorted(TEST_ROOT.rglob("test_*.py"))
    if not all_tests:
        print(f"error: no test modules found under '{TEST_ROOT}'", file=sys.stderr)
        return 2

    failures: list[str] = []
    lightweight = 0
    ml = 0
    for path in all_tests:
        relative = path.relative_to(PROJECT_ROOT)
        if ML_TEST_ROOT in path.parents:
            ml += 1
            continue
        lightweight += 1
        offenders = _requires_torch(path)
        if offenders:
            failures.append(
                f"{relative} imports {', '.join(sorted(offenders))} but sits outside "
                f"python/tests/ml, so the lightweight CI job (which has no PyTorch) "
                f"will fail collection; move it under python/tests/ml/"
            )

    if not ml:
        failures.append(
            f"no test modules under '{ML_TEST_ROOT.relative_to(PROJECT_ROOT)}'; the ML "
            f"CI job would run an empty selection and pass vacuously"
        )

    if failures:
        print("error: CI test partition is broken", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(
        f"test partition ok: {lightweight} lightweight + {ml} ml "
        f"= {len(all_tests)} modules, each claimed by exactly one CI job"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
