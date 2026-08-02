"""Output must be atomic, deterministic and self-cleaning after a failure.

The failure path that matters most is the *closing* flush: rows buffered up to
the end of the block are written when the context manager exits, so a schema
violation surfaces there rather than at ``write_row``. If that path leaks its
temporary file, an aborted run leaves a stray partial partition beside the
destination it never reached.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from differentiable_pricing.market.writer import (
    QUOTE_SCHEMA,
    STATISTIC_SCHEMA,
    AtomicParquetWriter,
    WriterError,
    partition_path,
    write_table_atomically,
)

SCHEMA = pa.schema(
    [
        pa.field("instrument_id", pa.uint32(), nullable=False),
        pa.field("price", pa.float64()),
        pa.field("label", pa.string()),
    ]
)


def rows(count: int) -> list[dict[str, object]]:
    return [
        {"instrument_id": index, "price": index / 8.0, "label": f"row-{index:04d}"}
        for index in range(count)
    ]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def temporary_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.name.endswith(".tmp"))


class TestAtomicity:
    def test_nothing_exists_until_the_block_exits(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        with AtomicParquetWriter(destination, SCHEMA) as writer:
            writer.write_rows(rows(10))
            assert not destination.exists()
            assert len(temporary_files(tmp_path)) == 1
        assert destination.is_file()
        assert temporary_files(tmp_path) == []

    def test_a_failure_removes_the_temporary_and_writes_nothing(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        with (
            pytest.raises(RuntimeError, match="boom"),
            AtomicParquetWriter(destination, SCHEMA) as writer,
        ):
            writer.write_rows(rows(10))
            raise RuntimeError("boom")
        assert not destination.exists()
        assert temporary_files(tmp_path) == []

    def test_a_failure_leaves_a_previous_output_intact(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        write_table_atomically(destination, rows(4), SCHEMA)
        before = digest(destination)
        with (
            pytest.raises(RuntimeError, match="boom"),
            AtomicParquetWriter(destination, SCHEMA) as writer,
        ):
            writer.write_rows(rows(99))
            raise RuntimeError("boom")
        assert digest(destination) == before
        assert pq.read_table(destination).num_rows == 4
        assert temporary_files(tmp_path) == []

    def test_a_missing_column_cleans_up(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        with (
            pytest.raises(WriterError, match="missing column"),
            AtomicParquetWriter(destination, SCHEMA) as writer,
        ):
            writer.write_row({"instrument_id": 1, "price": 1.0})
        assert not destination.exists()
        assert temporary_files(tmp_path) == []

    def test_a_failure_in_the_closing_flush_removes_the_temporary(
        self, tmp_path: Path
    ) -> None:
        """The buffered rows are written on exit, so this is the last failure path."""
        destination = tmp_path / "out.parquet"
        with (
            pytest.raises(WriterError, match="cannot write a row group"),
            AtomicParquetWriter(destination, SCHEMA) as writer,
        ):
            writer.write_row({"instrument_id": 1, "price": "not a number", "label": "x"})
        assert not destination.exists()
        assert temporary_files(tmp_path) == []

    def test_a_closing_flush_failure_leaves_a_previous_output_intact(
        self, tmp_path: Path
    ) -> None:
        destination = tmp_path / "out.parquet"
        write_table_atomically(destination, rows(4), SCHEMA)
        before = digest(destination)
        with (
            pytest.raises(WriterError),
            AtomicParquetWriter(destination, SCHEMA) as writer,
        ):
            writer.write_row({"instrument_id": 2, "price": "nope", "label": "y"})
        assert digest(destination) == before
        assert temporary_files(tmp_path) == []

    def test_publication_is_recorded(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        with AtomicParquetWriter(destination, SCHEMA) as writer:
            writer.write_rows(rows(2))
            assert writer.published is False
        assert destination.is_file()

    def test_parent_directories_are_created(self, tmp_path: Path) -> None:
        destination = tmp_path / "a" / "b" / "c" / "out.parquet"
        write_table_atomically(destination, rows(3), SCHEMA)
        assert destination.is_file()

    def test_a_non_positive_batch_size_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(WriterError, match="batch_rows must be positive"):
            AtomicParquetWriter(tmp_path / "out.parquet", SCHEMA, batch_rows=0)


class TestDeterminism:
    def test_identical_input_produces_identical_bytes(self, tmp_path: Path) -> None:
        payload = rows(5000)
        first = tmp_path / "first.parquet"
        second = tmp_path / "second.parquet"
        write_table_atomically(first, payload, SCHEMA)
        write_table_atomically(second, payload, SCHEMA)
        assert digest(first) == digest(second)

    def test_rewriting_the_same_destination_is_idempotent(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        payload = rows(1000)
        write_table_atomically(destination, payload, SCHEMA)
        before = digest(destination)
        write_table_atomically(destination, payload, SCHEMA)
        assert digest(destination) == before

    def test_row_group_size_does_not_change_the_data(self, tmp_path: Path) -> None:
        payload = rows(1000)
        one_group = tmp_path / "one.parquet"
        many_groups = tmp_path / "many.parquet"
        write_table_atomically(one_group, payload, SCHEMA, batch_rows=100_000)
        write_table_atomically(many_groups, payload, SCHEMA, batch_rows=64)
        assert pq.read_table(one_group).to_pydict() == pq.read_table(many_groups).to_pydict()
        assert pq.ParquetFile(many_groups).num_row_groups > 1

    def test_row_order_is_preserved(self, tmp_path: Path) -> None:
        destination = tmp_path / "out.parquet"
        payload = list(reversed(rows(200)))
        write_table_atomically(destination, payload, SCHEMA, batch_rows=17)
        stored = pq.read_table(destination).to_pydict()
        assert stored["instrument_id"] == [row["instrument_id"] for row in payload]


class TestSchemasAndPaths:
    def test_partition_path_is_hive_style(self, tmp_path: Path) -> None:
        path = partition_path(
            tmp_path, product_key="spy-options", schema="cbbo-1m", date="2026-07-30"
        )
        assert path.relative_to(tmp_path).as_posix() == (
            "product=spy-options/schema=cbbo-1m/date=2026-07-30/part-00000.parquet"
        )

    def test_quote_schema_uses_float64_and_utc_timestamps(self) -> None:
        assert QUOTE_SCHEMA.field("ts_recv_ns").type == pa.timestamp("ns", tz="UTC")
        assert QUOTE_SCHEMA.field("ts_event_ns").type == pa.timestamp("ns", tz="UTC")
        for name in ("bid_price", "ask_price", "mid_price", "quoted_spread", "relative_spread"):
            assert QUOTE_SCHEMA.field(name).type == pa.float64()

    def test_quote_schema_persists_vendor_flags_and_tradability(self) -> None:
        """A consumer must not have to re-decode DBN to recover the vendor's own doubt."""
        for name in ("vendor_flags", "vendor_flag_names", "bad_ts_recv", "maybe_bad_book",
                     "tradable_core"):
            assert name in QUOTE_SCHEMA.names

    def test_statistic_schema_marks_republished_records(self) -> None:
        assert "is_repeat_of_earlier_record" in STATISTIC_SCHEMA.names
        assert "is_outright" in STATISTIC_SCHEMA.names
        assert STATISTIC_SCHEMA.field("ts_ref_ns").type == pa.timestamp("ns", tz="UTC")

    def test_nullable_price_columns_accept_absent_sides(self, tmp_path: Path) -> None:
        destination = tmp_path / "quotes.parquet"
        row = {
            "ts_recv_ns": 1785418260000000000,
            "ts_event_ns": None,
            "instrument_id": 1,
            "publisher_id": 30,
            "bid_price": None,
            "ask_price": 1.1,
            "bid_size": 0,
            "ask_size": 5,
            "mid_price": None,
            "quoted_spread": None,
            "relative_spread": None,
            "bid_venue": None,
            "ask_venue": None,
            "in_regular_session": True,
            "quality": "missing",
            "vendor_flags": 192,
            "vendor_flag_names": "last,top_of_book",
            "bad_ts_recv": False,
            "maybe_bad_book": False,
            "tradable_core": False,
        }
        assert write_table_atomically(destination, [row], QUOTE_SCHEMA) == 1
        stored = pq.read_table(destination).to_pylist()[0]
        assert stored["bid_price"] is None
        assert stored["quality"] == "missing"
        assert stored["tradable_core"] is False
