"""Manifest verification must fail loudly on every way an archive can go wrong."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from differentiable_pricing.market.manifest import (
    ManifestError,
    hash_file,
    parse_manifest,
    require_verified_archive,
    verify_archive,
)

MANIFEST = "manifests/sha256sums.txt"


def test_clean_archive_verifies(synthetic_archive: Path) -> None:
    verification = verify_archive(
        synthetic_archive, MANIFEST, expected_entry_count=2, expected_total_files=3
    )
    assert verification.ok
    assert verification.entry_count == 2
    assert verification.verified_count == 2
    assert verification.total_bytes == len(b"alpha payload") + len(b"beta payload")
    assert verification.failure_report() == ""


def test_missing_file_is_reported_and_raises(synthetic_archive: Path) -> None:
    (synthetic_archive / "raw" / "beta.bin").unlink()
    verification = verify_archive(synthetic_archive, MANIFEST)
    assert not verification.ok
    assert verification.missing == ("raw/beta.bin",)
    assert verification.altered == ()
    with pytest.raises(ManifestError, match=r"raw/beta\.bin"):
        require_verified_archive(synthetic_archive, MANIFEST)


def test_altered_file_is_reported_with_both_digests(synthetic_archive: Path) -> None:
    target = synthetic_archive / "raw" / "alpha.bin"
    target.write_bytes(b"tampered payload")
    verification = verify_archive(synthetic_archive, MANIFEST)
    assert not verification.ok
    assert verification.missing == ()
    assert len(verification.altered) == 1
    mismatch = verification.altered[0]
    assert mismatch.relative_path == "raw/alpha.bin"
    assert mismatch.observed_sha256 == hashlib.sha256(b"tampered payload").hexdigest()
    assert mismatch.expected_sha256 != mismatch.observed_sha256


def test_duplicated_manifest_entry_is_reported(synthetic_archive: Path) -> None:
    manifest = synthetic_archive / MANIFEST
    lines = manifest.read_text(encoding="utf-8").splitlines()
    manifest.write_text("\n".join([*lines, lines[0]]) + "\n", encoding="utf-8")
    verification = verify_archive(synthetic_archive, MANIFEST)
    assert not verification.ok
    assert verification.duplicated == ("raw/alpha.bin",)
    assert "duplicated" in verification.failure_report()


def test_unexpected_file_is_reported(synthetic_archive: Path) -> None:
    (synthetic_archive / "raw" / "gamma.bin").write_bytes(b"not in the manifest")
    verification = verify_archive(synthetic_archive, MANIFEST)
    assert not verification.ok
    assert verification.unexpected == ("raw/gamma.bin",)


def test_declared_counts_are_enforced(synthetic_archive: Path) -> None:
    verification = verify_archive(
        synthetic_archive, MANIFEST, expected_entry_count=99, expected_total_files=99
    )
    assert not verification.ok
    report = verification.failure_report()
    assert "manifest covers 2 distinct paths" in report
    assert "archive holds 3 files" in report


def test_missing_manifest_raises(tmp_path: Path) -> None:
    (tmp_path / "raw").mkdir()
    with pytest.raises(ManifestError, match="is missing"):
        verify_archive(tmp_path, MANIFEST)


def test_missing_archive_root_raises(tmp_path: Path) -> None:
    with pytest.raises(ManifestError, match="does not exist"):
        verify_archive(tmp_path / "absent", MANIFEST)


@pytest.mark.parametrize(
    "text",
    [
        "notadigest  raw/alpha.bin",
        "ABCDEF0123456789" * 4 + "  raw/alpha.bin",
        "0123456789abcdef" * 4,
        "0123456789abcdef" * 4 + "  ",
    ],
)
def test_malformed_manifest_lines_are_rejected(text: str) -> None:
    with pytest.raises(ManifestError):
        parse_manifest(text, manifest_path=MANIFEST)


def test_empty_manifest_is_rejected() -> None:
    with pytest.raises(ManifestError, match="manifest is empty"):
        parse_manifest("\n\n", manifest_path=MANIFEST)


def test_binary_mode_and_blank_lines_are_accepted() -> None:
    digest = "0" * 64
    entries = parse_manifest(f"\n{digest} *raw/alpha.bin\n\n", manifest_path=MANIFEST)
    assert len(entries) == 1
    assert entries[0].relative_path == "raw/alpha.bin"


def test_hashing_is_chunk_size_independent(tmp_path: Path) -> None:
    target = tmp_path / "payload.bin"
    target.write_bytes(bytes(range(256)) * 97)
    reference = hashlib.sha256(target.read_bytes()).hexdigest()
    for chunk in (1, 7, 4096, 1 << 20):
        digest, size = hash_file(target, chunk_bytes=chunk)
        assert digest == reference
        assert size == 256 * 97
