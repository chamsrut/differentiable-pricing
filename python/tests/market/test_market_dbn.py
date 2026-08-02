"""Decoding is exercised against DBN files built programmatically, never copied data.

Every fixture here is encoded from scratch with the vendor library, so the tests
carry no market records and stay valid on any machine. When the optional reader
is absent the decode-backed tests skip; the sentinel, flag and compression-
sniffing logic is pure Python and always runs.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pyarrow as pa
import pytest
from differentiable_pricing.market.dbn import (
    FLAG_BAD_TS_RECV,
    FLAG_LAST,
    FLAG_MAYBE_BAD_BOOK,
    FLAG_SNAPSHOT,
    FLAG_TOB,
    UNDEF_INT32,
    UNDEF_INT64,
    UNDEF_PRICE,
    UNDEF_TIMESTAMP,
    DbnError,
    DbnTruncationError,
    decode_optional_int,
    decode_price,
    decode_quantity,
    decode_timestamp,
    describe_flags,
    detect_compression,
    is_suspect,
    open_dbn,
)
from differentiable_pricing.market.instruments import build_option_universe

databento_dbn = pytest.importorskip(
    "databento_dbn", reason="the optional 'market' extra is not installed"
)


def encode_dbn(records: list[object], *, dataset: str, schema: object) -> bytes:
    """Encode a metadata header plus records into DBN bytes."""
    metadata = databento_dbn.Metadata(
        dataset=dataset,
        start=0,
        stype_in=databento_dbn.SType.PARENT,
        stype_out=databento_dbn.SType.INSTRUMENT_ID,
        schema=schema,
        symbols=["SPY.OPT"],
    )
    payload = bytearray(metadata.encode())
    for record in records:
        payload += bytes(record)
    return bytes(payload)


def definition_record(
    *, instrument_id: int, raw_symbol: str, right: str, strike: float, expiry: dt.date
) -> object:
    expiration = int(
        dt.datetime.combine(expiry, dt.time(20, 0), tzinfo=dt.UTC).timestamp()
    ) * 1_000_000_000
    option_class = (
        databento_dbn.InstrumentClass.CALL
        if right == "C"
        else databento_dbn.InstrumentClass.PUT
    )
    return databento_dbn.InstrumentDefMsg(
        publisher_id=30,
        instrument_id=instrument_id,
        ts_event=expiration,
        ts_recv=expiration,
        min_price_increment=10_000_000,
        display_factor=1_000_000_000,
        raw_symbol=raw_symbol,
        asset="SPY",
        security_type="OPT",
        instrument_class=option_class,
        security_update_action=databento_dbn.SecurityUpdateAction.ADD,
        expiration=expiration,
        strike_price=round(strike * 1_000_000_000),
        underlying="SPY",
        currency="USD",
        exchange="OPRA",
    )


@pytest.fixture
def definition_bytes() -> bytes:
    return encode_dbn(
        [
            definition_record(
                instrument_id=101,
                raw_symbol="SPY   260918C00600000",
                right="C",
                strike=600.0,
                expiry=dt.date(2026, 9, 18),
            ),
            definition_record(
                instrument_id=102,
                raw_symbol="SPY   260918P00612500",
                right="P",
                strike=612.5,
                expiry=dt.date(2026, 9, 18),
            ),
        ],
        dataset="OPRA.PILLAR",
        schema=databento_dbn.Schema.DEFINITION,
    )


@pytest.fixture
def definition_file(definition_bytes: bytes, tmp_path: Path) -> Path:
    path = tmp_path / "synthetic.definition.dbn"
    path.write_bytes(definition_bytes)
    return path


@pytest.fixture
def compressed_definition_file(definition_file: Path, tmp_path: Path) -> Path:
    compressed = pa.compress(definition_file.read_bytes(), codec="zstd", asbytes=True)
    path = tmp_path / "synthetic.definition.dbn.zst"
    path.write_bytes(compressed)
    return path


class TestSentinelDecoding:
    def test_price_sentinel_becomes_none(self) -> None:
        assert decode_price(UNDEF_PRICE) is None
        assert decode_price(None) is None

    def test_prices_are_scaled_by_one_billion(self) -> None:
        assert decode_price(612_500_000_000) == pytest.approx(612.5)
        assert decode_price(-1_250_000_000) == pytest.approx(-1.25)
        assert decode_price(0) == 0.0

    def test_timestamp_sentinel_becomes_none(self) -> None:
        assert decode_timestamp(UNDEF_TIMESTAMP) is None
        assert decode_timestamp(1_785_418_260_000_000_000) == 1_785_418_260_000_000_000

    def test_integer_sentinels_become_none(self) -> None:
        assert decode_optional_int(UNDEF_INT32, UNDEF_INT32) is None
        assert decode_optional_int(100, UNDEF_INT32) == 100
        assert decode_optional_int(0, 0) is None

    def test_both_quantity_widths_are_treated_as_absent(self) -> None:
        """StatMsg.quantity is 32-bit in DBN v1/v2 and 64-bit in v3."""
        assert decode_quantity(UNDEF_INT32) is None
        assert decode_quantity(UNDEF_INT64) is None
        assert decode_quantity(None) is None
        assert decode_quantity(1234) == 1234
        assert decode_quantity(0) == 0


class TestRecordFlags:
    def test_named_bits_are_decoded(self) -> None:
        assert describe_flags(FLAG_LAST | FLAG_TOB) == ("last", "top_of_book")
        assert describe_flags(FLAG_LAST | FLAG_SNAPSHOT | FLAG_BAD_TS_RECV) == (
            "last",
            "snapshot",
            "bad_ts_recv",
        )
        assert describe_flags(0) == ()

    def test_only_the_quality_bits_make_a_record_suspect(self) -> None:
        assert is_suspect(FLAG_LAST | FLAG_TOB) is False
        assert is_suspect(FLAG_SNAPSHOT) is False
        assert is_suspect(FLAG_BAD_TS_RECV) is True
        assert is_suspect(FLAG_MAYBE_BAD_BOOK) is True
        assert is_suspect(168) is True  # last | snapshot | bad_ts_recv


class TestCompressionSniffing:
    def test_plain_dbn_is_detected(self, definition_file: Path) -> None:
        assert detect_compression(definition_file) == "none"

    def test_zstd_is_detected(self, compressed_definition_file: Path) -> None:
        assert detect_compression(compressed_definition_file) == "zstd"

    def test_a_mislabelled_payload_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "not-really.dbn.zst"
        path.write_bytes(b"PK\x03\x04 this is a zip")
        with pytest.raises(DbnError, match="neither a zstd frame nor a DBN header"):
            detect_compression(path)


class TestStreamingDecode:
    def test_records_and_header_are_recovered(self, definition_file: Path) -> None:
        handle = open_dbn(definition_file)
        records = list(handle)
        assert len(records) == 2
        summary = handle.summary
        assert summary.dataset == "OPRA.PILLAR"
        assert summary.record_count == 2
        assert summary.compression == "none"
        assert summary.compressed_bytes == definition_file.stat().st_size

    def test_compression_is_transparent_to_the_caller(
        self, definition_file: Path, compressed_definition_file: Path
    ) -> None:
        plain = [record.raw_symbol for record in open_dbn(definition_file)]
        compressed = [record.raw_symbol for record in open_dbn(compressed_definition_file)]
        assert plain == compressed

    @pytest.mark.parametrize("chunk_bytes", [1, 13, 512, 1 << 20])
    def test_chunk_size_does_not_change_the_result(
        self, compressed_definition_file: Path, chunk_bytes: int
    ) -> None:
        records = list(open_dbn(compressed_definition_file, chunk_bytes=chunk_bytes))
        assert [record.instrument_id for record in records] == [101, 102]

    def test_resolution_runs_end_to_end_from_an_encoded_file(
        self, compressed_definition_file: Path
    ) -> None:
        universe = build_option_universe(
            open_dbn(compressed_definition_file), product_root="SPY"
        )
        assert universe.failures == []
        contracts = {item.instrument_id: item for item in universe.contracts.values()}
        assert contracts[101].strike == pytest.approx(600.0)
        assert contracts[102].option_type == "P"
        assert contracts[102].expiration_date == dt.date(2026, 9, 18)
        # OPRA leaves these undefined; resolution must not invent them.
        assert contracts[101].contract_multiplier is None
        assert contracts[101].exercise_style is None

    def test_summary_before_reading_is_an_error(self, definition_file: Path) -> None:
        handle = open_dbn(definition_file)
        with pytest.raises(DbnError, match="has not been read yet"):
            _ = handle.summary

    def test_a_missing_file_is_an_error(self, tmp_path: Path) -> None:
        with pytest.raises(DbnError, match="does not exist"):
            open_dbn(tmp_path / "absent.dbn.zst")

    def test_a_garbage_payload_fails_loudly(self, tmp_path: Path) -> None:
        path = tmp_path / "truncated.dbn"
        path.write_bytes(b"DBN\x03short")
        with pytest.raises(DbnError):
            list(open_dbn(path))


class TestTruncationDetection:
    """A stream that stops mid-record must not look like a short but complete one."""

    def test_a_partial_trailing_record_raises(
        self, definition_bytes: bytes, tmp_path: Path
    ) -> None:
        path = tmp_path / "clipped.dbn"
        path.write_bytes(definition_bytes[:-40])
        with pytest.raises(DbnTruncationError, match="undecodable trailing bytes"):
            list(open_dbn(path))

    def test_the_records_before_the_tear_are_still_yielded_before_the_error(
        self, definition_bytes: bytes, tmp_path: Path
    ) -> None:
        path = tmp_path / "clipped.dbn"
        path.write_bytes(definition_bytes[:-40])
        seen = []
        with pytest.raises(DbnTruncationError):
            for record in open_dbn(path):
                seen.append(record.instrument_id)
        assert seen == [101]

    def test_a_truncation_error_is_a_decode_error(
        self, definition_bytes: bytes, tmp_path: Path
    ) -> None:
        path = tmp_path / "clipped.dbn"
        path.write_bytes(definition_bytes[:-40])
        with pytest.raises(DbnError):
            list(open_dbn(path))

    def test_an_intact_payload_does_not_raise(self, definition_file: Path) -> None:
        assert len(list(open_dbn(definition_file))) == 2

    def test_a_compressed_payload_clipped_mid_record_raises(
        self, definition_bytes: bytes, tmp_path: Path
    ) -> None:
        path = tmp_path / "clipped.dbn.zst"
        path.write_bytes(
            pa.compress(definition_bytes[:-40], codec="zstd", asbytes=True)
        )
        with pytest.raises(DbnError):
            list(open_dbn(path))
