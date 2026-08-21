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

python3 -m compileall -q python/src python/tests scripts .claude/hooks
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

python3 scripts/check_test_partition.py
python3 scripts/check_european_replication_protocol.py
# Task 9G protocol validation is offline: it hashes tracked inputs only and
# never touches any dataset partition or launches the locked experiment.
python3 scripts/check_american_neural_pilot_protocol.py
python3 scripts/plot_european_validation_results.py --check
python3 scripts/plot_european_replication_results.py --check
# Both checks use only checked-in files, so they pass with artifacts/ absent.
python3 scripts/freeze_american_lsm_results.py --check
python3 scripts/plot_american_lsm_results.py --check
# The task 9C-C3 v2 snapshot's designated validator. It recomputes every verdict
# from the snapshot's own numbers against the checked-in configuration and never
# reads the ignored raw confirmation report, so it passes with artifacts/ absent.
python3 scripts/freeze_pde_label_policy_v2_results.py --check
# The task 9G snapshot's designated validator. Like the checks above it is
# offline: it reads only the tracked protocol and the tracked snapshot, reruns
# no pricing, training, latency measurement or IV inversion, and never opens a
# dataset partition, so it passes with artifacts/ absent.
python3 scripts/freeze_american_neural_pilot_results.py --check
# Task 9H attempt-log checks. Static and offline: they parse tracked sources,
# attempt configurations and the append-only attempt log, import nothing from
# the project package, and need neither PyTorch nor a compiled extension. No
# pricing, training or dataset access happens here; training stays a manual,
# terminal-invoked human job.
python3 scripts/american_dev_attempts.py check

if command -v ruff >/dev/null 2>&1; then
    ruff check python scripts .claude/hooks
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
if ! python3 -c "import numpy, pyarrow, pytest, torch" >/dev/null 2>&1; then
    echo "error: full Python test dependencies are not importable; install with" \
        "python -m pip install -e '.[dev,train]'" >&2
    exit 1
fi
python3 -m pytest -q
