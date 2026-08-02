"""Observations about what the archive contains, as opposed to what it says.

Two kinds of evidence live here, and neither is a constant in the source.

Declared-source probes
    Every external input the study needs is declared in configuration as a path
    and a kind. :func:`probe_declared_source` reports one of three *observed*
    states — ``absent``, ``present_but_empty``, ``present`` — because those are
    three different facts. "We never acquired it" and "we acquired an empty
    directory" have different remedies, and an audit that renders both as a
    hardcoded ``False`` has stopped observing anything.

    Empty directories are invisible to a SHA-256 manifest: a manifest lists
    files, so a directory holding nothing is neither missing nor unexpected.
    That is exactly the state the real archive's corporate-action and issuer
    directories are in, so it has to be probed for explicitly.

Vendor condition statements
    Each vendor request directory ships a ``condition.json`` recording, per
    calendar date, whether the vendor considers the data available, degraded,
    pending or missing. It is the only authoritative statement about
    completeness and the audit consumes it rather than asserting completeness
    on its own.
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SOURCE_ABSENT: Final = "absent"
SOURCE_PRESENT_BUT_EMPTY: Final = "present_but_empty"
SOURCE_PRESENT: Final = "present"

CONDITION_AVAILABLE: Final = "available"
"""The only vendor condition under which a date may be treated as complete."""


@dataclass(frozen=True, slots=True)
class SourceObservation:
    """What was actually found at a declared source path."""

    key: str
    path: str
    kind: str
    requirement: str
    capability: str
    state: str
    file_count: int
    total_bytes: int
    detail: str
    why: str
    candidate_source: str

    @property
    def satisfied(self) -> bool:
        """Whether the source supplied usable content."""
        return self.state == SOURCE_PRESENT

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready observation."""
        return {
            "key": self.key,
            "path": self.path,
            "kind": self.kind,
            "requirement": self.requirement,
            "capability": self.capability,
            "state": self.state,
            "satisfied": self.satisfied,
            "file_count": self.file_count,
            "total_bytes": self.total_bytes,
            "detail": self.detail,
            "why_the_study_needs_it": self.why,
            "candidate_source": self.candidate_source,
        }


def probe_declared_source(
    archive_root: Path,
    *,
    key: str,
    path: str,
    kind: str,
    requirement: str,
    capability: str,
    why: str,
    candidate_source: str,
) -> SourceObservation:
    """Observe one declared source and report which of the three states it is in."""
    target = archive_root / path
    if kind == "file":
        if not target.is_file():
            state, files, size, detail = (
                SOURCE_ABSENT,
                0,
                0,
                f"no file at '{path}'",
            )
        else:
            size = target.stat().st_size
            if size == 0:
                state, files, detail = (
                    SOURCE_PRESENT_BUT_EMPTY,
                    1,
                    f"'{path}' exists but holds zero bytes",
                )
            else:
                state, files, detail = (
                    SOURCE_PRESENT,
                    1,
                    f"'{path}' holds {size} bytes",
                )
    elif kind == "directory":
        if not target.is_dir():
            state, files, size, detail = (
                SOURCE_ABSENT,
                0,
                0,
                f"no directory at '{path}'",
            )
        else:
            contents = [item for item in target.rglob("*") if item.is_file()]
            files = len(contents)
            size = sum(item.stat().st_size for item in contents)
            if files == 0:
                state, detail = (
                    SOURCE_PRESENT_BUT_EMPTY,
                    (
                        f"'{path}' exists but contains no files; a SHA-256 manifest lists "
                        f"files, so an empty directory is invisible to manifest verification "
                        f"and must be probed for"
                    ),
                )
            else:
                state, detail = (
                    SOURCE_PRESENT,
                    f"'{path}' contains {files} files totalling {size} bytes",
                )
    else:  # pragma: no cover - configuration validation rejects other kinds
        raise ValueError(f"unknown declared-source kind '{kind}'")

    return SourceObservation(
        key=key,
        path=path,
        kind=kind,
        requirement=requirement,
        capability=capability,
        state=state,
        file_count=files,
        total_bytes=size,
        detail=detail,
        why=why,
        candidate_source=candidate_source,
    )


@dataclass(frozen=True, slots=True)
class VendorCondition:
    """The vendor's own statement about one calendar date of one request."""

    request_path: str
    date: dt.date | None
    condition: str
    last_modified_date: str | None
    raw_date: str

    @property
    def is_available(self) -> bool:
        """Whether the vendor calls this date complete."""
        return self.condition == CONDITION_AVAILABLE

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready condition entry."""
        return {
            "request_path": self.request_path,
            "date": self.raw_date,
            "condition": self.condition,
            "last_modified_date": self.last_modified_date,
        }


class ConditionError(RuntimeError):
    """Raised when a vendor condition statement cannot be read."""


def read_vendor_conditions(
    condition_path: Path, *, relative_to: Path
) -> tuple[VendorCondition, ...]:
    """Parse one ``condition.json`` into per-date statements.

    A condition file that exists but cannot be parsed is an error, not an
    absence: silently ignoring it would restore exactly the behaviour of never
    having consulted the vendor at all.
    """
    try:
        document = json.loads(condition_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ConditionError(f"cannot read '{condition_path}': {error}") from error
    if not isinstance(document, list):
        raise ConditionError(
            f"'{condition_path}' is not a list of per-date condition entries"
        )
    request = condition_path.parent.relative_to(relative_to).as_posix()
    entries: list[VendorCondition] = []
    for item in document:
        if not isinstance(item, dict):
            raise ConditionError(f"'{condition_path}' contains a non-object entry")
        raw_date = str(item.get("date", ""))
        try:
            parsed = dt.date.fromisoformat(raw_date)
        except ValueError:
            parsed = None
        entries.append(
            VendorCondition(
                request_path=request,
                date=parsed,
                condition=str(item.get("condition", "")).strip().lower(),
                last_modified_date=(
                    None
                    if item.get("last_modified_date") is None
                    else str(item["last_modified_date"])
                ),
                raw_date=raw_date,
            )
        )
    return tuple(entries)
