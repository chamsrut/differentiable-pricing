"""Keep tracked-only protocol/result checks aligned between local and CI gates."""

from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
TRACKED_CHECK = re.compile(
    r"python3 (scripts/(?:check_[a-z0-9_]*protocol\.py|"
    r"(?:freeze|plot)_[a-z0-9_]*results\.py --check))"
)


def _entry_points_text(text: str) -> set[str]:
    return set(TRACKED_CHECK.findall(text))


def _entry_points(path: Path) -> set[str]:
    return _entry_points_text(path.read_text(encoding="utf-8"))


def test_clean_environment_ci_runs_every_local_tracked_protocol_and_freeze_check() -> None:
    local = _entry_points(PROJECT_ROOT / "scripts/check.sh")
    ci = _entry_points(PROJECT_ROOT / ".github/workflows/ci.yml")
    assert local
    assert ci == local


def test_task_9g_snapshot_check_runs_in_both_gates() -> None:
    """The task 9G result snapshot must be validated locally and in clean CI.

    Its ``--check`` is offline by construction: it reads only the tracked
    protocol and the tracked snapshot, so it belongs in both gates and cannot
    be dropped from either without failing here.
    """
    command = "python3 scripts/freeze_american_neural_pilot_results.py --check"
    local_text = (PROJECT_ROOT / "scripts/check.sh").read_text(encoding="utf-8")
    ci_text = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    assert command in local_text
    assert command in ci_text
    entry = "scripts/freeze_american_neural_pilot_results.py --check"
    assert entry in _entry_points_text(local_text)
    assert entry in _entry_points_text(ci_text)


def test_reconciliation_detects_a_missing_plot_check() -> None:
    local_text = (PROJECT_ROOT / "scripts/check.sh").read_text(encoding="utf-8")
    ci_text = (PROJECT_ROOT / ".github/workflows/ci.yml").read_text(encoding="utf-8")
    command = "python3 scripts/plot_european_validation_results.py --check"
    assert command in local_text and command in ci_text
    assert _entry_points_text(ci_text.replace(command, "", 1)) != _entry_points_text(local_text)
