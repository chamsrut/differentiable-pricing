"""Publication of JSON reports is all-or-nothing.

These cover the writer itself. The report-level contract (`evaluate_artifact`
refusing to overwrite, and a retry succeeding after a failed write) is exercised
against a real trained artifact in `test_differential_training.py`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from differentiable_pricing.ml.artifact import write_json_atomic

PAYLOAD = {"beta": [1.0, 2.0], "alpha": "value"}


def temporaries(directory: Path) -> list[Path]:
    """Return leftover temporary files, which a correct writer never leaves."""
    return [path for path in directory.iterdir() if path.name.endswith(".tmp")]


def test_successful_write_is_canonical_and_leaves_no_temporaries(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    write_json_atomic(output, PAYLOAD)

    text = output.read_text(encoding="utf-8")
    assert json.loads(text) == PAYLOAD
    assert text.endswith("\n")
    # Canonical: sorted keys, two-space indent.
    assert text == json.dumps(PAYLOAD, indent=2, sort_keys=True) + "\n"
    assert temporaries(tmp_path) == []


def test_unserializable_payload_never_creates_a_file(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float"):
        write_json_atomic(output, {"nan": float("nan")})

    assert not output.exists()
    assert temporaries(tmp_path) == []


def test_infinity_is_rejected_like_nan(tmp_path: Path) -> None:
    output = tmp_path / "report.json"

    with pytest.raises(ValueError, match="Out of range float"):
        write_json_atomic(output, {"inf": float("inf")})

    assert not output.exists()
    assert temporaries(tmp_path) == []


def test_failed_write_leaves_no_partial_file_and_retry_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"

    def failing_fsync(descriptor: int) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="simulated disk failure"):
        write_json_atomic(output, PAYLOAD)

    assert not output.exists()
    assert temporaries(tmp_path) == []

    # The retry, once the fault clears, must succeed rather than trip over
    # residue from the failed attempt.
    monkeypatch.undo()
    write_json_atomic(output, PAYLOAD)
    assert json.loads(output.read_text(encoding="utf-8")) == PAYLOAD
    assert temporaries(tmp_path) == []


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt, SystemExit])
def test_interrupt_during_write_still_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interrupt: type[BaseException],
) -> None:
    """Ctrl-C mid-write must not strand a temporary file or a partial report."""
    output = tmp_path / "report.json"

    def interrupting_fsync(descriptor: int) -> None:
        raise interrupt()

    monkeypatch.setattr(os, "fsync", interrupting_fsync)
    with pytest.raises(interrupt):
        write_json_atomic(output, PAYLOAD)

    assert not output.exists()
    assert temporaries(tmp_path) == []

    monkeypatch.undo()
    write_json_atomic(output, PAYLOAD)
    assert json.loads(output.read_text(encoding="utf-8")) == PAYLOAD


def test_existing_report_is_not_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "report.json"
    original = '{"existing": true}\n'
    output.write_text(original, encoding="utf-8")

    with pytest.raises(FileExistsError):
        write_json_atomic(output, PAYLOAD)

    assert output.read_text(encoding="utf-8") == original
    assert temporaries(tmp_path) == []


def test_failed_overwrite_preserves_the_existing_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "report.json"
    original = '{"existing": true}\n'
    output.write_text(original, encoding="utf-8")

    def failing_fsync(descriptor: int) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(os, "fsync", failing_fsync)
    with pytest.raises(OSError, match="simulated disk failure"):
        write_json_atomic(output, PAYLOAD, overwrite=True)

    assert output.read_text(encoding="utf-8") == original
    assert temporaries(tmp_path) == []

    monkeypatch.undo()
    write_json_atomic(output, PAYLOAD, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == PAYLOAD
    assert temporaries(tmp_path) == []


def test_exclusive_publish_wins_the_race_against_a_late_writer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A file appearing after any prior existence check must not be clobbered."""
    output = tmp_path / "report.json"
    original = '{"written-by-someone-else": true}\n'
    real_fsync = os.fsync

    def racing_fsync(descriptor: int) -> None:
        # Simulate a competing process publishing between the caller's
        # existence check and this call's atomic link.
        if not output.exists():
            output.write_text(original, encoding="utf-8")
        real_fsync(descriptor)

    monkeypatch.setattr(os, "fsync", racing_fsync)
    with pytest.raises(FileExistsError):
        write_json_atomic(output, PAYLOAD)

    assert output.read_text(encoding="utf-8") == original
    assert temporaries(tmp_path) == []
