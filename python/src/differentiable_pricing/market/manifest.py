"""SHA-256 verification of a raw market archive before anything is decoded.

Nothing downstream may run against an archive whose contents do not match the
manifest that was signed off with it. Verification is strict in both directions:

* every manifest entry must exist on disk and hash to the recorded digest;
* every file on disk must be covered by the manifest, apart from the manifest
  itself, which cannot hash its own contents.

Missing, altered, duplicated and unexpected files are distinct findings. They
are reported separately so a failure says which of the four happened, and the
verifier refuses to proceed on any of them.

Files are hashed in bounded chunks: the archive holds multi-hundred-megabyte
payloads and must never be read into memory whole.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

READ_CHUNK_BYTES: Final = 1 << 20
"""Bytes hashed per read. Bounds memory independently of file size."""

_DIGEST_LENGTH: Final = 64


class ManifestError(RuntimeError):
    """Raised when an archive does not match its manifest."""


@dataclass(frozen=True, slots=True)
class ManifestEntry:
    """One ``<sha256>  <relative path>`` line of a manifest."""

    relative_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ManifestMismatch:
    """One file whose recorded and observed digests disagree."""

    relative_path: str
    expected_sha256: str
    observed_sha256: str


@dataclass(frozen=True, slots=True)
class ManifestVerification:
    """The full outcome of verifying an archive against its manifest."""

    archive_root: str
    manifest_path: str
    manifest_sha256: str
    entry_count: int
    verified_count: int
    total_bytes: int
    missing: tuple[str, ...] = ()
    altered: tuple[ManifestMismatch, ...] = ()
    duplicated: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    unreadable: tuple[str, ...] = ()
    _failures: tuple[str, ...] = field(default=(), repr=False)

    @property
    def ok(self) -> bool:
        """Whether the archive matched the manifest exactly."""
        return not self._failures

    def failure_report(self) -> str:
        """Return a multi-line description of every finding, newest first."""
        return "\n".join(self._failures)


def parse_manifest(text: str, *, manifest_path: str) -> tuple[ManifestEntry, ...]:
    """Parse a ``sha256sum`` manifest.

    Accepts the two formats ``sha256sum`` emits: ``<digest>  <path>`` for text
    mode and ``<digest> *<path>`` for binary mode. Blank lines are skipped;
    anything else is a hard error, because a manifest line that cannot be
    understood is indistinguishable from a file that is not covered.
    """
    entries: list[ManifestEntry] = []
    for number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        digest, separator, remainder = stripped.partition(" ")
        if not separator:
            raise ManifestError(f"{manifest_path}:{number}: line is not '<sha256>  <path>'")
        if len(digest) != _DIGEST_LENGTH or any(char not in "0123456789abcdef" for char in digest):
            raise ManifestError(
                f"{manifest_path}:{number}: '{digest}' is not a lowercase hex SHA-256 digest"
            )
        relative = remainder.lstrip(" ")
        if relative.startswith("*"):
            relative = relative[1:]
        if not relative:
            raise ManifestError(f"{manifest_path}:{number}: manifest entry has an empty path")
        entries.append(ManifestEntry(relative_path=relative, sha256=digest))
    if not entries:
        raise ManifestError(f"{manifest_path}: manifest is empty")
    return tuple(entries)


def hash_file(path: Path, *, chunk_bytes: int = READ_CHUNK_BYTES) -> tuple[str, int]:
    """Return the ``(sha256, size)`` of ``path``, reading it in bounded chunks."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def iter_archive_files(archive_root: Path) -> Iterator[Path]:
    """Yield every regular file under ``archive_root`` in deterministic order."""
    yield from sorted(path for path in archive_root.rglob("*") if path.is_file())


def verify_archive(
    archive_root: Path | str,
    manifest_relative_path: str,
    *,
    expected_entry_count: int | None = None,
    expected_total_files: int | None = None,
    chunk_bytes: int = READ_CHUNK_BYTES,
) -> ManifestVerification:
    """Verify ``archive_root`` against the manifest it contains.

    Returns a verification record. The record's :attr:`ManifestVerification.ok`
    is false when anything disagreed; callers that must not proceed should use
    :func:`require_verified_archive` instead of interpreting the record.
    """
    root = Path(archive_root)
    if not root.is_dir():
        raise ManifestError(f"archive root '{root}' does not exist or is not a directory")

    manifest_path = root / manifest_relative_path
    if not manifest_path.is_file():
        raise ManifestError(f"manifest '{manifest_path}' is missing")
    manifest_sha256, _ = hash_file(manifest_path, chunk_bytes=chunk_bytes)
    entries = parse_manifest(
        manifest_path.read_text(encoding="utf-8"), manifest_path=manifest_relative_path
    )

    seen: dict[str, str] = {}
    duplicated: list[str] = []
    for entry in entries:
        previous = seen.get(entry.relative_path)
        if previous is None:
            seen[entry.relative_path] = entry.sha256
        else:
            # A repeated path is a defect whether or not the two digests agree:
            # the manifest no longer states one digest per file.
            duplicated.append(entry.relative_path)

    missing: list[str] = []
    altered: list[ManifestMismatch] = []
    unreadable: list[str] = []
    verified = 0
    total_bytes = 0
    for relative_path, expected in sorted(seen.items()):
        candidate = root / relative_path
        if not candidate.is_file():
            missing.append(relative_path)
            continue
        try:
            observed, size = hash_file(candidate, chunk_bytes=chunk_bytes)
        except OSError as error:
            unreadable.append(f"{relative_path}: {error}")
            continue
        total_bytes += size
        if observed != expected:
            altered.append(
                ManifestMismatch(
                    relative_path=relative_path,
                    expected_sha256=expected,
                    observed_sha256=observed,
                )
            )
            continue
        verified += 1

    covered = set(seen) | {manifest_relative_path}
    on_disk = {path.relative_to(root).as_posix() for path in iter_archive_files(root)}
    unexpected = sorted(on_disk - covered)

    failures: list[str] = []
    if missing:
        failures.append(f"{len(missing)} manifest entries are missing from the archive:")
        failures.extend(f"  missing: {path}" for path in missing)
    if altered:
        failures.append(f"{len(altered)} files do not match their recorded digest:")
        failures.extend(
            f"  altered: {item.relative_path} expected {item.expected_sha256} "
            f"observed {item.observed_sha256}"
            for item in altered
        )
    if duplicated:
        failures.append(f"{len(duplicated)} paths appear more than once in the manifest:")
        failures.extend(f"  duplicated: {path}" for path in sorted(duplicated))
    if unexpected:
        failures.append(f"{len(unexpected)} files are present but not covered by the manifest:")
        failures.extend(f"  unexpected: {path}" for path in unexpected)
    if unreadable:
        failures.append(f"{len(unreadable)} files could not be read:")
        failures.extend(f"  unreadable: {item}" for item in unreadable)
    if expected_entry_count is not None and len(seen) != expected_entry_count:
        failures.append(
            f"manifest covers {len(seen)} distinct paths but the configuration "
            f"declares {expected_entry_count}"
        )
    if expected_total_files is not None and len(on_disk) != expected_total_files:
        failures.append(
            f"archive holds {len(on_disk)} files but the configuration declares "
            f"{expected_total_files}"
        )

    return ManifestVerification(
        archive_root=str(root),
        manifest_path=manifest_relative_path,
        manifest_sha256=manifest_sha256,
        entry_count=len(seen),
        verified_count=verified,
        total_bytes=total_bytes,
        missing=tuple(missing),
        altered=tuple(altered),
        duplicated=tuple(sorted(duplicated)),
        unexpected=tuple(unexpected),
        unreadable=tuple(unreadable),
        _failures=tuple(failures),
    )


def require_verified_archive(
    archive_root: Path | str,
    manifest_relative_path: str,
    *,
    expected_entry_count: int | None = None,
    expected_total_files: int | None = None,
    chunk_bytes: int = READ_CHUNK_BYTES,
) -> ManifestVerification:
    """Verify an archive and raise :class:`ManifestError` unless it is clean."""
    verification = verify_archive(
        archive_root,
        manifest_relative_path,
        expected_entry_count=expected_entry_count,
        expected_total_files=expected_total_files,
        chunk_bytes=chunk_bytes,
    )
    if not verification.ok:
        raise ManifestError(
            f"archive '{verification.archive_root}' does not match manifest "
            f"'{verification.manifest_path}':\n{verification.failure_report()}"
        )
    return verification
