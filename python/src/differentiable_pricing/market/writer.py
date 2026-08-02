"""Atomic, bounded-memory Parquet output for normalized market data.

Two properties matter and both are enforced here rather than left to callers.

Atomicity
    Rows are written to a temporary file beside the destination and moved into
    place with :func:`os.replace` only after the file has been closed cleanly.
    A crashed or aborted run therefore leaves either the previous output or no
    output, never a half-written partition that a later run would mistake for
    complete data. The temporary file is removed on *every* failure path,
    including a failure inside the final flush.

Bounded memory
    Rows are buffered in column-major Python lists up to a fixed batch size and
    flushed as one row group. Peak memory tracks the batch size, not the number
    of rows written, so a hundred-million-row product costs the same as a
    thousand-row one. Note that this buffer, not the decoder's read chunk, is
    the dominant memory term: a batch of N rows over C columns holds N*C boxed
    Python objects at once.

Determinism
    Given the same rows in the same order and the same pyarrow build, the bytes
    written are identical: no timestamps, no run identifiers and no dictionary
    ordering derived from set iteration enter the file.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_BATCH_ROWS: Final = 262_144
"""Rows buffered before a row group is flushed."""


class WriterError(RuntimeError):
    """Raised when normalized output cannot be written."""


class AtomicParquetWriter:
    """Write one Parquet file atomically, one bounded row group at a time.

    Use as a context manager. Leaving the block normally publishes the file;
    leaving it through an exception — or failing during the closing flush —
    removes the temporary file and leaves any previously published output
    untouched.
    """

    __slots__ = ("_batch_rows", "_buffer", "_columns", "_compression", "_destination",
                 "_published", "_rows_written", "_schema", "_temporary", "_writer")

    def __init__(
        self,
        destination: Path | str,
        schema: pa.Schema,
        *,
        batch_rows: int = DEFAULT_BATCH_ROWS,
        compression: str = "zstd",
    ) -> None:
        if batch_rows <= 0:
            raise WriterError(f"batch_rows must be positive, got {batch_rows}")
        self._destination = Path(destination)
        self._schema = schema
        self._batch_rows = batch_rows
        self._compression = compression
        self._columns = tuple(schema.names)
        self._buffer: dict[str, list[Any]] = {name: [] for name in self._columns}
        self._writer: pq.ParquetWriter | None = None
        self._temporary: Path | None = None
        self._rows_written = 0
        self._published = False

    @property
    def destination(self) -> Path:
        """Final path the file is published to."""
        return self._destination

    @property
    def rows_written(self) -> int:
        """Rows accepted so far, buffered or flushed."""
        return self._rows_written

    @property
    def published(self) -> bool:
        """Whether the file reached its destination."""
        return self._published

    def __enter__(self) -> AtomicParquetWriter:
        self._destination.parent.mkdir(parents=True, exist_ok=True)
        # The temporary file must share a directory with the destination so the
        # final move is a same-filesystem rename and therefore atomic.
        self._temporary = self._destination.with_name(
            f".{self._destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
        )
        try:
            self._writer = pq.ParquetWriter(
                self._temporary, self._schema, compression=self._compression
            )
        except (OSError, pa.ArrowException) as error:
            self._discard()
            raise WriterError(f"cannot open '{self._temporary}' for writing: {error}") from error
        return self

    def write_row(self, row: Mapping[str, Any]) -> None:
        """Buffer one row, flushing a row group when the batch is full."""
        buffer = self._buffer
        for name in self._columns:
            try:
                buffer[name].append(row[name])
            except KeyError as error:
                raise WriterError(
                    f"row is missing column '{name}'; expected {list(self._columns)}"
                ) from error
        self._rows_written += 1
        if len(buffer[self._columns[0]]) >= self._batch_rows:
            self.flush()

    def write_rows(self, rows: Iterable[Mapping[str, Any]]) -> None:
        """Buffer many rows."""
        for row in rows:
            self.write_row(row)

    def flush(self) -> None:
        """Write the buffered rows as one row group and clear the buffer."""
        if self._writer is None:
            raise WriterError("writer is not open")
        first = self._columns[0]
        if not self._buffer[first]:
            return
        try:
            table = pa.Table.from_pydict(
                {name: self._buffer[name] for name in self._columns}, schema=self._schema
            )
            self._writer.write_table(table)
        except (pa.ArrowException, OSError, ValueError) as error:
            raise WriterError(
                f"cannot write a row group to '{self._temporary}': {error}"
            ) from error
        for name in self._columns:
            self._buffer[name].clear()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            self._discard()
            return
        try:
            self.flush()
        except BaseException:
            # A failure in the closing flush is still a failed write: the
            # temporary file must go, or an aborted run leaves a stray partial
            # partition next to the destination it never reached.
            self._discard()
            raise
        writer, self._writer = self._writer, None
        if writer is not None:
            try:
                writer.close()
            except BaseException:
                self._discard()
                raise
        assert self._temporary is not None
        try:
            os.replace(self._temporary, self._destination)
        except OSError as error:
            self._discard()
            raise WriterError(
                f"cannot publish '{self._temporary}' to '{self._destination}': {error}"
            ) from error
        self._temporary = None
        self._published = True

    def _discard(self) -> None:
        """Close and delete the temporary file, leaving the destination alone."""
        writer, self._writer = self._writer, None
        if writer is not None:
            # The run is already failing; a close error must not mask it or
            # prevent the temporary file from being removed.
            with contextlib.suppress(OSError, pa.ArrowException, ValueError):
                writer.close()
        temporary, self._temporary = self._temporary, None
        if temporary is not None:
            with contextlib.suppress(OSError):
                temporary.unlink(missing_ok=True)
        for name in self._columns:
            self._buffer[name].clear()


QUOTE_SCHEMA: Final = pa.schema(
    [
        pa.field("ts_recv_ns", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("ts_event_ns", pa.timestamp("ns", tz="UTC")),
        pa.field("instrument_id", pa.uint32(), nullable=False),
        pa.field("publisher_id", pa.uint16(), nullable=False),
        pa.field("bid_price", pa.float64()),
        pa.field("ask_price", pa.float64()),
        pa.field("bid_size", pa.uint32()),
        pa.field("ask_size", pa.uint32()),
        pa.field("mid_price", pa.float64()),
        pa.field("quoted_spread", pa.float64()),
        pa.field("relative_spread", pa.float64()),
        pa.field("bid_venue", pa.uint16()),
        pa.field("ask_venue", pa.uint16()),
        pa.field("in_regular_session", pa.bool_(), nullable=False),
        pa.field("quality", pa.string(), nullable=False),
        pa.field("vendor_flags", pa.uint8(), nullable=False),
        pa.field("vendor_flag_names", pa.string(), nullable=False),
        pa.field("bad_ts_recv", pa.bool_(), nullable=False),
        pa.field("maybe_bad_book", pa.bool_(), nullable=False),
        pa.field("tradable_core", pa.bool_(), nullable=False),
    ]
)
"""Normalized quote schema. float64 throughout, as the numerical rules require.

``vendor_flags`` and its derived booleans are persisted so a consumer of the
Parquet never has to re-decode 700 MB of DBN to learn that the vendor doubted a
record's receive timestamp.
"""

OPTION_DEFINITION_SCHEMA: Final = pa.schema(
    [
        pa.field("instrument_id", pa.uint32(), nullable=False),
        pa.field("raw_symbol", pa.string(), nullable=False),
        pa.field("publisher_id", pa.uint16(), nullable=False),
        pa.field("root", pa.string(), nullable=False),
        pa.field("is_standard_root", pa.bool_(), nullable=False),
        pa.field("underlying", pa.string(), nullable=False),
        pa.field("option_type", pa.string(), nullable=False),
        pa.field("strike", pa.float64(), nullable=False),
        pa.field("expiration_ns", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("expiration_date", pa.date32(), nullable=False),
        pa.field("currency", pa.string()),
        pa.field("exchange", pa.string()),
        pa.field("underlying_id", pa.uint32()),
        pa.field("contract_multiplier", pa.int32()),
        pa.field("unit_of_measure_quantity", pa.float64()),
        pa.field("original_contract_size", pa.int32()),
        pa.field("exercise_style", pa.string()),
        pa.field("security_update_action", pa.string()),
        pa.field("definition_ts_recv_ns", pa.timestamp("ns", tz="UTC")),
    ]
)

FUTURE_DEFINITION_SCHEMA: Final = pa.schema(
    [
        pa.field("instrument_id", pa.uint32(), nullable=False),
        pa.field("raw_symbol", pa.string(), nullable=False),
        pa.field("publisher_id", pa.uint16(), nullable=False),
        pa.field("asset", pa.string()),
        pa.field("instrument_class", pa.string(), nullable=False),
        pa.field("is_outright", pa.bool_(), nullable=False),
        pa.field("expiration_ns", pa.timestamp("ns", tz="UTC")),
        pa.field("expiration_date", pa.date32()),
        pa.field("currency", pa.string()),
        pa.field("exchange", pa.string()),
        pa.field("min_price_increment", pa.float64()),
        pa.field("unit_of_measure", pa.string()),
        pa.field("unit_of_measure_quantity", pa.float64()),
        pa.field("security_update_action", pa.string()),
    ]
)

STATISTIC_SCHEMA: Final = pa.schema(
    [
        pa.field("ts_recv_ns", pa.timestamp("ns", tz="UTC"), nullable=False),
        pa.field("ts_ref_ns", pa.timestamp("ns", tz="UTC")),
        pa.field("instrument_id", pa.uint32(), nullable=False),
        pa.field("publisher_id", pa.uint16(), nullable=False),
        pa.field("stat_type", pa.string(), nullable=False),
        pa.field("price", pa.float64()),
        pa.field("quantity", pa.int64()),
        pa.field("update_action", pa.string(), nullable=False),
        pa.field("stat_flags", pa.uint8(), nullable=False),
        pa.field("is_outright", pa.bool_()),
        pa.field("is_repeat_of_earlier_record", pa.bool_(), nullable=False),
    ]
)
"""Statistics schema.

``is_repeat_of_earlier_record`` marks a record whose
``(instrument, stat type, reference time, price, quantity)`` was already seen in
this partition. CME republishes a settlement several times per session; counting
those republications as observations inflates every statistics count.
"""


def partition_path(
    processed_root: Path | str,
    *,
    product_key: str,
    schema: str,
    date: str,
    part: int = 0,
) -> Path:
    """Return the Hive-style partition path for one product, schema and date."""
    return (
        Path(processed_root)
        / f"product={product_key}"
        / f"schema={schema}"
        / f"date={date}"
        / f"part-{part:05d}.parquet"
    )


def write_table_atomically(
    destination: Path | str,
    rows: Sequence[Mapping[str, Any]],
    schema: pa.Schema,
    *,
    batch_rows: int = DEFAULT_BATCH_ROWS,
    compression: str = "zstd",
) -> int:
    """Write ``rows`` to ``destination`` atomically and return the row count."""
    with AtomicParquetWriter(
        destination, schema, batch_rows=batch_rows, compression=compression
    ) as writer:
        writer.write_rows(rows)
        return writer.rows_written
