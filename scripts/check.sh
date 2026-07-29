#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
quick=false
if [[ "${1:-}" == "--quick" ]]; then
    quick=true
elif [[ $# -ne 0 ]]; then
    echo "usage: $0 [--quick]" >&2
    exit 2
fi

cd "${project_root}"

python3 -m compileall -q python/src python/tests .claude/hooks
python3 - <<'PY'
import json
import tomllib
from pathlib import Path

for path in (Path("pyproject.toml"), *sorted(Path("configs").glob("*.toml"))):
    with path.open("rb") as stream:
        tomllib.load(stream)

with Path(".claude/settings.json").open(encoding="utf-8") as stream:
    json.load(stream)
PY

if command -v ruff >/dev/null 2>&1; then
    ruff check python .claude/hooks
fi

if command -v clang-format >/dev/null 2>&1; then
    find cpp bindings -type f \
        \( -name '*.cpp' -o -name '*.hpp' \) \
        -print0 |
        xargs -0 clang-format --dry-run --Werror
fi

if [[ "${quick}" == "true" ]]; then
    if [[ -d build/check ]]; then
        cmake --build build/check --parallel
        ctest --test-dir build/check --output-on-failure
    fi
    exit 0
fi

cmake -S . -B build/check \
    -DDP_BUILD_PYTHON_BINDINGS=OFF \
    -DDP_WARNINGS_AS_ERRORS=ON
cmake --build build/check --parallel
ctest --test-dir build/check --output-on-failure

# The Python suite covers the pybind11 boundary and the dataset generator, so
# the full gate must run it. It is deliberately not guarded by an availability
# check: a missing pytest or a missing editable install is a failed gate, not a
# silently skipped one.
if ! python3 -c "import pytest" >/dev/null 2>&1; then
    echo "error: pytest is not importable; install the package with" \
        "python -m pip install -e '.[dev,data]'" >&2
    exit 1
fi
python3 -m pytest -q
