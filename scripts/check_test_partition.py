#!/usr/bin/env python3
"""Assert the CI test partition stays a real partition.

Continuous integration splits the Python suite by directory:

* the lightweight job installs ``.[dev,data]`` (no PyTorch) and runs
  ``pytest -q --ignore=python/tests/ml``;
* the ML job installs ``.[dev,train]`` and runs ``pytest -q python/tests/ml``.

Those two commands cover ``python/tests`` exactly once each only while no test
module outside ``python/tests/ml`` reaches PyTorch. A module that imports
``torch`` directly, imports the ``differentiable_pricing.ml`` package, or
dynamically executes a repository script that does either would fail collection
in the lightweight job. This check fails loudly at that moment instead of
leaving a red pipeline to be diagnosed from a collection traceback.

The import scan is static: it walks ASTs rather than importing anything, so it
runs in the lightweight environment where PyTorch is absent. The supported
dynamic-script pattern is the repository's normal
``spec_from_file_location``/``exec_module`` idiom with a statically resolvable
repository-relative path. Arbitrary computed imports remain outside this
check's scope and still fail naturally during lightweight collection.
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


def _direct_torch_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = set()
    for module in _imported_modules(tree):
        for root in TORCH_BACKED_ROOTS:
            if module == root or module.startswith(f"{root}."):
                offenders.add(module)
    return offenders


def _attribute_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        prefix = _attribute_name(node.value)
        return node.attr if prefix is None else f"{prefix}.{node.attr}"
    return None


def _scope_for(node: ast.AST, parents: dict[ast.AST, ast.AST], tree: ast.Module) -> ast.AST:
    current = node
    while current in parents:
        current = parents[current]
        if isinstance(current, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return current
    return tree


def _assignments_before(scope: ast.AST, line: int) -> dict[str, ast.AST]:
    assignments: dict[str, ast.AST] = {}

    class LexicalAssignmentVisitor(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            if node.lineno < line:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        assignments[target.id] = node.value

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.lineno < line and isinstance(node.target, ast.Name) and node.value is not None:
                assignments[node.target.id] = node.value

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            pass

        def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
            pass

        def visit_ClassDef(self, node: ast.ClassDef) -> None:
            pass

        def visit_Lambda(self, node: ast.Lambda) -> None:
            pass

    visitor = LexicalAssignmentVisitor()
    body = (
        scope.body if isinstance(scope, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef)) else []
    )
    for statement in body:
        visitor.visit(statement)
    return assignments


def _path_parts(
    node: ast.AST,
    assignments: dict[str, ast.AST],
    *,
    seen: frozenset[str] = frozenset(),
) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, ast.Name) and node.id in assignments and node.id not in seen:
        return _path_parts(assignments[node.id], assignments, seen=seen | frozenset({node.id}))
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return _path_parts(node.left, assignments, seen=seen) + _path_parts(
            node.right, assignments, seen=seen
        )
    if isinstance(node, ast.Call):
        parts: list[str] = []
        for argument in node.args:
            parts.extend(_path_parts(argument, assignments, seen=seen))
        return parts
    if isinstance(node, (ast.Attribute, ast.Subscript)):
        return _path_parts(node.value, assignments, seen=seen)
    return []


def _dynamic_repository_scripts(tree: ast.Module, project_root: Path) -> set[Path]:
    parents = {child: parent for parent in ast.walk(tree) for child in ast.iter_child_nodes(parent)}
    scripts: set[Path] = set()
    global_assignments = _assignments_before(tree, 1 << 30)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if (_attribute_name(node.func) or "").split(".")[-1] != "spec_from_file_location":
            continue
        scope = _scope_for(node, parents, tree)
        if not any(
            isinstance(candidate, ast.Call)
            and (_attribute_name(candidate.func) or "").split(".")[-1] == "exec_module"
            for candidate in ast.walk(scope)
        ):
            continue
        assignments = dict(global_assignments)
        assignments.update(_assignments_before(scope, node.lineno))
        if len(node.args) < 2:
            continue
        parts = _path_parts(node.args[1], assignments)
        if not parts:
            continue
        candidate = Path(*parts)
        if not candidate.is_absolute():
            candidate = project_root / candidate
        candidate = candidate.resolve()
        try:
            candidate.relative_to(project_root.resolve())
        except ValueError:
            continue
        if candidate.is_file() and candidate.suffix == ".py":
            scripts.add(candidate)
    return scripts


def _requires_torch(path: Path, project_root: Path = PROJECT_ROOT) -> set[str]:
    """Return direct and dynamically reached torch-backed imports."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    offenders = _direct_torch_imports(path)
    for script in _dynamic_repository_scripts(tree, project_root):
        for module in _direct_torch_imports(script):
            offenders.add(f"{script.relative_to(project_root)} -> {module}")
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
