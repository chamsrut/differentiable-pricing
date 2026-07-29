#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${project_root}"

if ! git rev-parse --git-dir >/dev/null 2>&1; then
    echo "error: initialize the Git repository before installing hooks" >&2
    exit 1
fi

git config core.hooksPath .githooks
echo "Git hooks enabled from .githooks"
