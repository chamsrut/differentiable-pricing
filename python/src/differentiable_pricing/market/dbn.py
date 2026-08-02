"""Streaming, bounded-memory access to Databento DBN files.

Decoding goes through the vendor's own ``databento_dbn`` reader. Hand-rolling a
DBN parser would put an unvalidated binary-format assumption underneath every
number in the audit; the supported reader owns the format, its versions and its
upgrade policy.

Raw archive files are read as they lie: opened read-only, streamed in bounded
chunks, never decompressed in place and never rewritten.

Units
    DBN fixed-point prices are integers scaled by ``FIXED_PRICE_SCALE`` (1e-9).
    Sentinels mark absent values and must be mapped to ``None`` rather than
    scaled, which is what :func:`decode_price` does.

Truncation
    A stream that ends mid-record leaves bytes in the decoder that will never
    become a record. The reader asks the decoder for its residual buffer once
    the file is exhausted and fails loudly if anything is left, so a truncated
    payload cannot masquerade as a short but complete one.

Record flags
    Every DBN record carries a bitfield in which the vendor states what it
    thinks of the record: whether the receive timestamp is trustworthy, whether
    an unrecoverable gap preceded it, whether it came from a snapshot replay.
    Those bits are decoded here and carried all the way into the normalized
    output; dropping them would throw away the only quality statement the
    vendor makes about an individual observation.
"""

from __future__ import annotations

import datetime as dt
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

FIXED_PRICE_SCALE: Final = 1_000_000_000
"""Divisor turning a DBN fixed-point integer price into a decimal price."""

UNDEF_PRICE: Final = 2**63 - 1
"""``INT64_MAX``: DBN's 'no price' sentinel."""

UNDEF_TIMESTAMP: Final = 2**64 - 1
"""``UINT64_MAX``: DBN's 'no timestamp' sentinel."""

UNDEF_INT32: Final = 2**31 - 1
"""``INT32_MAX``: DBN's 'undefined' sentinel for 32-bit integer fields."""

UNDEF_INT64: Final = 2**63 - 1
"""``INT64_MAX``: DBN's 'undefined' sentinel for 64-bit integer fields."""

UNDEF_UINT16: Final = 2**16 - 1
UNDEF_UINT8: Final = 2**8 - 1

FLAG_LAST: Final = 128
"""Last message in the venue packet for this instrument."""

FLAG_TOB: Final = 64
"""Top-of-book message rather than an individual order."""

FLAG_SNAPSHOT: Final = 32
"""Message sourced from a replay or snapshot server rather than the live feed."""

FLAG_MBP: Final = 16
"""Aggregated price-level message rather than an individual order."""

FLAG_BAD_TS_RECV: Final = 8
"""The vendor considers this record's ``ts_recv`` inaccurate."""

FLAG_MAYBE_BAD_BOOK: Final = 4
"""An unrecoverable gap was detected before this record; the book may be wrong."""

SUSPECT_FLAGS: Final = FLAG_BAD_TS_RECV | FLAG_MAYBE_BAD_BOOK
"""Bits that make an individual observation unfit for a synchronized grid."""

_FLAG_NAMES: Final = (
    (FLAG_LAST, "last"),
    (FLAG_TOB, "top_of_book"),
    (FLAG_SNAPSHOT, "snapshot"),
    (FLAG_MBP, "aggregated_level"),
    (FLAG_BAD_TS_RECV, "bad_ts_recv"),
    (FLAG_MAYBE_BAD_BOOK, "maybe_bad_book"),
)

_ZSTD_MAGIC: Final = b"\x28\xb5\x2f\xfd"
_DBN_MAGIC: Final = b"DBN"


class DbnError(RuntimeError):
    """Raised when a DBN input cannot be opened or decoded."""


class DbnTruncationError(DbnError):
    """Raised when a DBN stream ends part-way through a record."""


class DbnDependencyError(DbnError):
    """Raised when the supported DBN reader is not installed."""


def require_dbn_module() -> Any:
    """Return the ``databento_dbn`` module, or explain how to install it.

    The dependency is optional on purpose: the lightweight CI job runs the whole
    test suite without it, because no test may depend on the private archive.
    Only real ingestion needs a decoder.
    """
    try:
        import databento_dbn
    except ImportError as error:  # pragma: no cover - exercised by environments without the dep
        raise DbnDependencyError(
            "decoding DBN requires the 'databento-dbn' reader; install the market extra "
            "with: python -m pip install -e '.[dev,market]'"
        ) from error
    return databento_dbn


def decode_price(raw: int | None) -> float | None:
    """Convert a DBN fixed-point price to a float, mapping sentinels to ``None``."""
    if raw is None or raw == UNDEF_PRICE:
        return None
    return raw / FIXED_PRICE_SCALE


def decode_timestamp(raw: int | None) -> int | None:
    """Return a UTC epoch-nanosecond timestamp, mapping the sentinel to ``None``."""
    if raw is None or raw == UNDEF_TIMESTAMP:
        return None
    return raw


def decode_optional_int(raw: int | None, sentinel: int) -> int | None:
    """Return ``raw`` unless it equals ``sentinel``, in which case ``None``."""
    if raw is None or raw == sentinel:
        return None
    return raw


def decode_quantity(raw: int | None) -> int | None:
    """Return a statistics quantity, mapping either width's sentinel to ``None``.

    ``StatMsg.quantity`` is 32-bit in DBN v1/v2 and 64-bit in v3. The reader's
    upgrade policy decides which one a caller sees, so both sentinels are
    treated as absent rather than trusting the reader to have upgraded.
    """
    if raw is None or raw in (UNDEF_INT32, UNDEF_INT64):
        return None
    return raw


def describe_flags(flags: int) -> tuple[str, ...]:
    """Return the set bit names of a DBN record flag byte, in declared order."""
    return tuple(name for bit, name in _FLAG_NAMES if flags & bit)


def is_suspect(flags: int) -> bool:
    """Whether the vendor flagged this record's timestamp or book as unreliable."""
    return bool(flags & SUSPECT_FLAGS)


def utc_datetime(epoch_nanoseconds: int) -> dt.datetime:
    """Return the UTC datetime for an epoch-nanosecond timestamp.

    Microsecond truncation is acceptable here because this helper only feeds
    calendar-level checks and reporting; the nanosecond integer is what is
    written to the normalized output.
    """
    return dt.datetime.fromtimestamp(epoch_nanoseconds / 1e9, tz=dt.UTC)


def detect_compression(path: Path) -> str:
    """Return ``"zstd"`` or ``"none"`` by inspecting the file's leading bytes.

    File names are provenance, not proof. Sniffing the magic number means a
    mislabelled ``.dbn.zst`` fails with a clear decode error rather than being
    silently mis-read.
    """
    with path.open("rb") as stream:
        prefix = stream.read(4)
    if prefix[:4] == _ZSTD_MAGIC:
        return "zstd"
    if prefix[:3] == _DBN_MAGIC:
        return "none"
    raise DbnError(
        f"'{path}' starts with {prefix!r}, which is neither a zstd frame nor a DBN header"
    )


@dataclass(frozen=True, slots=True)
class DbnFileSummary:
    """Header-level provenance of one decoded DBN file."""

    path: str
    compression: str
    compressed_bytes: int
    dbn_version: int
    dataset: str
    schema: str | None
    stype_in: str | None
    stype_out: str | None
    start_ns: int
    end_ns: int | None
    symbols: tuple[str, ...]
    symbol_mapping_count: int
    record_count: int


def _describe(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def iter_records(path: Path | str, *, chunk_bytes: int = 1 << 20) -> Iterator[Any]:
    """Yield every record of a DBN file, metadata first, without buffering the file.

    The decoder is fed ``chunk_bytes`` of compressed input at a time and the
    records it returns are yielded and released before the next chunk is read,
    so peak memory tracks the chunk size rather than the file size.

    Once the file is exhausted the decoder's residual buffer must be empty. A
    non-empty residue means the payload stopped part-way through a record, which
    raises :class:`DbnTruncationError` rather than ending the iteration quietly.
    """
    module = require_dbn_module()
    source = Path(path)
    compression = detect_compression(source)
    compression_enum = (
        module.Compression.ZSTD if compression == "zstd" else module.Compression.NONE
    )
    decoder = module.DBNDecoder(compression=compression_enum)
    try:
        with source.open("rb") as stream:
            while chunk := stream.read(chunk_bytes):
                yield from decoder.write_and_decode(chunk)
        residue = decoder.buffer()
    except module.DBNError as error:
        raise DbnError(f"cannot decode '{source}': {error}") from error
    if residue:
        raise DbnTruncationError(
            f"'{source}' ends with {len(residue)} undecodable trailing bytes; the payload "
            f"is truncated or corrupt"
        )


class DbnFile:
    """A DBN file opened for one streaming pass.

    Iterating the instance yields data records only. The header summary is
    available immediately after the first record is produced, and its
    ``record_count`` is final once iteration is exhausted.
    """

    __slots__ = ("_chunk_bytes", "_path", "_record_count", "_summary")

    def __init__(self, path: Path | str, *, chunk_bytes: int = 1 << 20) -> None:
        self._path = Path(path)
        self._chunk_bytes = chunk_bytes
        self._summary: DbnFileSummary | None = None
        self._record_count = 0

    @property
    def path(self) -> Path:
        """Path of the file being read."""
        return self._path

    @property
    def summary(self) -> DbnFileSummary:
        """Header provenance. Available only after iteration has started."""
        if self._summary is None:
            raise DbnError(f"'{self._path}' has not been read yet; its metadata is unknown")
        return DbnFileSummary(
            path=self._summary.path,
            compression=self._summary.compression,
            compressed_bytes=self._summary.compressed_bytes,
            dbn_version=self._summary.dbn_version,
            dataset=self._summary.dataset,
            schema=self._summary.schema,
            stype_in=self._summary.stype_in,
            stype_out=self._summary.stype_out,
            start_ns=self._summary.start_ns,
            end_ns=self._summary.end_ns,
            symbols=self._summary.symbols,
            symbol_mapping_count=self._summary.symbol_mapping_count,
            record_count=self._record_count,
        )

    @property
    def has_summary(self) -> bool:
        """Whether the metadata header has been decoded yet."""
        return self._summary is not None

    def __iter__(self) -> Iterator[Any]:
        module = require_dbn_module()
        metadata_type = module.Metadata
        compressed_bytes = self._path.stat().st_size
        compression = detect_compression(self._path)
        for record in iter_records(self._path, chunk_bytes=self._chunk_bytes):
            if isinstance(record, metadata_type):
                if self._summary is not None:
                    raise DbnError(f"'{self._path}' contains more than one metadata header")
                self._summary = DbnFileSummary(
                    path=self._path.as_posix(),
                    compression=compression,
                    compressed_bytes=compressed_bytes,
                    dbn_version=int(record.version),
                    dataset=str(record.dataset),
                    schema=_describe(record.schema),
                    stype_in=_describe(record.stype_in),
                    stype_out=_describe(record.stype_out),
                    start_ns=int(record.start),
                    end_ns=None if record.end is None else int(record.end),
                    symbols=tuple(record.symbols),
                    symbol_mapping_count=len(record.mappings),
                    record_count=0,
                )
                continue
            if self._summary is None:
                raise DbnError(f"'{self._path}' yields records before its metadata header")
            self._record_count += 1
            yield record
        if self._summary is None:
            raise DbnError(f"'{self._path}' contains no DBN metadata header")


def open_dbn(path: Path | str, *, chunk_bytes: int = 1 << 20) -> DbnFile:
    """Open a DBN file for one bounded-memory streaming pass."""
    source = Path(path)
    if not source.is_file():
        raise DbnError(f"DBN input '{source}' does not exist")
    return DbnFile(source, chunk_bytes=chunk_bytes)
