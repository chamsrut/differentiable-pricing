"""The whole audit must run end to end on a synthetic archive, and run identically twice.

The archive built here is encoded from scratch: a small option chain, a
cash-settled control whose underlying was not acquired, an equity reference on a
partial-venue feed, a futures product carrying both outrights and a calendar
spread, statistics with republished settlements, and three declared external
sources in three different states. It exercises manifest verification, vendor
condition statements, declared-source probes, definition resolution, quote
normalization and tradability, atomic partitioned output, per-contract
synchronization and the scoped capability results without touching real market
data.

Every assertion is meant to be mutation-sensitive: each one names a specific
failure mechanism from the review and would fail if that mechanism came back.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from differentiable_pricing.market.config import load_market_config
from differentiable_pricing.market.ingest import Auditor, IngestError, discover_inputs
from differentiable_pricing.market.manifest import ManifestError, verify_archive
from market_fixtures import write_condition, write_manifest

databento_dbn = pytest.importorskip(
    "databento_dbn", reason="the optional 'market' extra is not installed"
)

SESSION_DATE = dt.date(2026, 7, 30)
OPEN_NS = int(dt.datetime(2026, 7, 30, 13, 30, tzinfo=dt.UTC).timestamp()) * 1_000_000_000
MINUTE_NS = 60 * 1_000_000_000
SESSION_MINUTES = 5
FLAG_LAST = 128
FLAG_TOB = 64
FLAG_BAD_TS_RECV = 8

CONFIG_TEMPLATE = """
schema_version = "market-feasibility/2"
audit_version = "2.0.0-test"

[archive]
root = "archive"
sha256_manifest = "manifests/sha256sums.txt"
request_inventory = "manifests/requests.csv"
vendor_condition_filename = "condition.json"
expected_manifest_entries = {entries}
expected_total_files = {total}

[output]
processed_root = "processed"
report_root = "reports"

[session]
name = "test-session"
time_zone = "America/New_York"
open_local = "09:30"
close_local = "09:35"
expected_minutes = 5

[synchronization]
frequency_seconds = 60
label = "interval-close"

[[dates]]
date = "2026-07-30"
role = "held_out"

[[products]]
key = "spy-options"
dataset = "OPRA.PILLAR"
product = "SPY"
instrument_kind = "option"
role = "primary_option_chain"
settles_on = "SPY shares, physically delivered"
archive_subpath = "raw/opra/spy"
definition_schema = "definition"
quote_schema = "cbbo-1m"
schemas = ["definition", "cbbo-1m", "statistics"]
feed_class = "consolidated_nbbo"
underlying_product = "spy-underlying"

[[products]]
key = "xsp-options"
dataset = "OPRA.PILLAR"
product = "XSP"
instrument_kind = "option"
role = "cash_settled_control"
settles_on = "an index level that was not acquired"
archive_subpath = "raw/opra/xsp"
definition_schema = "definition"
quote_schema = "cbbo-1m"
schemas = ["definition", "cbbo-1m"]
feed_class = "consolidated_nbbo"
proxy_underlying_product = "spy-underlying"
proxy_caveat = "the settlement index was not acquired; the ETF/index basis is unmeasured"

[[products]]
key = "spy-underlying"
dataset = "EQUS.MINI"
product = "SPY"
instrument_kind = "equity"
role = "underlying_reference"
settles_on = "SPY shares"
archive_subpath = "raw/equs/spy"
quote_schema = "bbo-1m"
schemas = ["bbo-1m"]
feed_class = "partial_venue_consolidated"
reference_selection = "single_instrument"

[[products]]
key = "es-futures"
dataset = "GLBX.MDP3"
product = "ES"
instrument_kind = "future"
role = "{es_role}"
settles_on = "an index, cash settled"
archive_subpath = "raw/glbx/es"
definition_schema = "definition"
quote_schema = "bbo-1m"
schemas = ["definition", "bbo-1m", "statistics"]
feed_class = "exchange_direct"
reference_selection = "nearest_unexpired_outright"

[[synchronization_pairs]]
name = "spy-options-vs-spy-underlying"
option_product = "spy-options"
reference_product = "spy-underlying"
role = "primary"

[[synchronization_pairs]]
name = "spy-options-vs-es-futures"
option_product = "spy-options"
reference_product = "es-futures"
role = "control_proxy"

[[fred_series]]
series_id = "SOFR"
path = "raw/fred/SOFR.csv"
role = "daily_overnight_rate"

[[declared_sources]]
key = "discrete_dividend_schedule"
path = "raw/issuer"
kind = "directory"
requirement = "required"
capability = "spy_american_calibration_readiness"
why = "early exercise is driven by discrete dividends"
candidate_source = "an issuer distribution history"

[[declared_sources]]
key = "corporate_action_and_contract_adjustment_history"
path = "raw/occ"
kind = "directory"
requirement = "required"
capability = "spy_american_calibration_readiness"
why = "adjusted contracts carry non-standard deliverables"
candidate_source = "contract-adjustment memoranda"

[[declared_sources]]
key = "securities_lending_borrow_rate"
path = "raw/borrow"
kind = "directory"
requirement = "required"
capability = "spy_american_calibration_readiness"
why = "the forward embeds the cost of borrowing the underlying"
candidate_source = "a securities-lending fee series"

[tradability]
require_positive_sizes = true
require_positive_bid = true
require_positive_spread = true
exclude_vendor_flagged = true
exclude_stale = false
exploratory_relative_spread_thresholds = {thresholds}

[capabilities]
minimum_sessions = 60
minimum_held_out_sessions = 12
minimum_expiries_with_paired_strikes = 1
minimum_paired_strikes_per_expiry = 2
session_minima_status = "provisional_pilot_target"
session_minima_provenance = "chosen after observing the feasibility archive; provisional"
structural_minima_provenance = "a surface fit needs several same-minute call/put pairs"

[audit]
read_chunk_bytes = 4096
parquet_batch_rows = 64
parquet_compression = "zstd"
moneyness_bucket_edges = [0.9, 1.0, 1.1]
maturity_bucket_edges_days = [0, 30, 365]
extrapolation_trading_dates = 200
"""


def scaled(price: float) -> int:
    return round(price * 1_000_000_000)


def encode(records: list[Any], *, dataset: str, schema: Any, stype_in: Any, symbol: str) -> bytes:
    metadata = databento_dbn.Metadata(
        dataset=dataset,
        start=OPEN_NS,
        stype_in=stype_in,
        stype_out=databento_dbn.SType.INSTRUMENT_ID,
        schema=schema,
        symbols=[symbol],
    )
    payload = bytearray(metadata.encode())
    for record in records:
        payload += bytes(record)
    return bytes(payload)


def option_definition(
    instrument_id: int,
    right: str,
    strike: float,
    *,
    root: str = "SPY",
    ts_recv: int = OPEN_NS,
    action: Any = None,
) -> Any:
    expiry = dt.date(2026, 9, 18)
    expiration = int(
        dt.datetime.combine(expiry, dt.time(20, 0), tzinfo=dt.UTC).timestamp()
    ) * 1_000_000_000
    return databento_dbn.InstrumentDefMsg(
        publisher_id=30,
        instrument_id=instrument_id,
        ts_event=ts_recv,
        ts_recv=ts_recv,
        min_price_increment=10_000_000,
        display_factor=1_000_000_000,
        raw_symbol=f"{root:<6}{expiry:%y%m%d}{right}{round(strike * 1000):08d}",
        asset=root,
        security_type="OPT",
        instrument_class=(
            databento_dbn.InstrumentClass.CALL
            if right == "C"
            else databento_dbn.InstrumentClass.PUT
        ),
        security_update_action=action or databento_dbn.SecurityUpdateAction.ADD,
        expiration=expiration,
        strike_price=scaled(strike),
        underlying=root,
        currency="USD",
        exchange="OPRA",
    )


def future_definition(
    instrument_id: int, symbol: str, klass: Any, expiry: dt.date
) -> Any:
    expiration = int(
        dt.datetime.combine(expiry, dt.time(13, 30), tzinfo=dt.UTC).timestamp()
    ) * 1_000_000_000
    return databento_dbn.InstrumentDefMsg(
        publisher_id=1,
        instrument_id=instrument_id,
        ts_event=OPEN_NS,
        ts_recv=OPEN_NS,
        min_price_increment=250_000_000,
        display_factor=1_000_000_000,
        raw_symbol=symbol,
        asset="ES",
        security_type="FUT",
        instrument_class=klass,
        security_update_action=databento_dbn.SecurityUpdateAction.ADD,
        expiration=expiration,
        strike_price=databento_dbn.UNDEF_PRICE,
        underlying="ES",
        currency="USD",
        exchange="XCME",
    )


def consolidated_quote(
    instrument_id: int, minute: int, bid: float, ask: float, *, size: int = 10
) -> Any:
    return databento_dbn.CBBOMsg(
        rtype=databento_dbn.RType.CBBO_1M,
        publisher_id=30,
        instrument_id=instrument_id,
        ts_event=databento_dbn.UNDEF_TIMESTAMP,
        price=databento_dbn.UNDEF_PRICE,
        size=0,
        side=databento_dbn.Side.NONE,
        ts_recv=OPEN_NS + minute * MINUTE_NS,
        flags=FLAG_LAST | FLAG_TOB,
        levels=databento_dbn.ConsolidatedBidAskPair(
            bid_px=scaled(bid), ask_px=scaled(ask), bid_sz=size, ask_sz=size + 2,
            bid_pb=24, ask_pb=23
        ),
    )


def plain_quote(
    instrument_id: int,
    minute: int,
    bid: float,
    ask: float,
    *,
    publisher_id: int = 95,
    flags: int = FLAG_LAST,
    size: int = 100,
) -> Any:
    return databento_dbn.BBOMsg(
        rtype=databento_dbn.RType.BBO_1M,
        publisher_id=publisher_id,
        instrument_id=instrument_id,
        ts_event=databento_dbn.UNDEF_TIMESTAMP,
        price=databento_dbn.UNDEF_PRICE,
        size=0,
        side=databento_dbn.Side.NONE,
        ts_recv=OPEN_NS + minute * MINUTE_NS,
        flags=flags,
        levels=databento_dbn.BidAskPair(
            bid_px=scaled(bid), ask_px=scaled(ask), bid_sz=size, ask_sz=size,
            bid_ct=1, ask_ct=1
        ),
    )


def statistic(instrument_id: int, price: float, *, ts_ref: int) -> Any:
    return databento_dbn.StatMsg(
        publisher_id=1,
        instrument_id=instrument_id,
        ts_event=OPEN_NS,
        ts_recv=OPEN_NS,
        ts_ref=ts_ref,
        price=scaled(price),
        quantity=databento_dbn.UNDEF_STAT_QUANTITY
        if hasattr(databento_dbn, "UNDEF_STAT_QUANTITY")
        else 2**63 - 1,
        sequence=0,
        ts_in_delta=0,
        stat_type=databento_dbn.StatType.SETTLEMENT_PRICE,
        channel_id=0,
        update_action=databento_dbn.StatUpdateAction.NEW,
        stat_flags=0,
    )


# Instrument ids: 101 call / 102 put on the SPY chain, 201/202 on the control,
# 10 the ES front outright, 11 a deferred outright, 12 a calendar spread.
ES_FRONT = 10
ES_DEFERRED = 11
ES_SPREAD = 12


def build_archive(
    root: Path,
    *,
    option_minutes: range = range(1, SESSION_MINUTES + 1),
    call_minutes: range | None = None,
    put_minutes: range | None = None,
    es_outright_minutes: range | None = None,
    orphan_future_id: int | None = None,
    underlying_flags: int = FLAG_LAST,
    condition: str = "available",
    condition_dates: list[str] | None = None,
) -> None:
    """Encode a complete miniature archive under ``root`` and manifest it."""
    spy_definitions = encode(
        [option_definition(101, "C", 600.0), option_definition(102, "P", 600.0)],
        dataset="OPRA.PILLAR",
        schema=databento_dbn.Schema.DEFINITION,
        stype_in=databento_dbn.SType.PARENT,
        symbol="SPY.OPT",
    )
    calls = call_minutes if call_minutes is not None else option_minutes
    puts = put_minutes if put_minutes is not None else option_minutes
    spy_quotes = encode(
        [
            consolidated_quote(instrument_id, minute, 1.00 + 0.01 * minute,
                               1.10 + 0.01 * minute)
            for instrument_id, minutes in ((101, calls), (102, puts))
            for minute in minutes
        ],
        dataset="OPRA.PILLAR",
        schema=databento_dbn.Schema.CBBO_1M,
        stype_in=databento_dbn.SType.PARENT,
        symbol="SPY.OPT",
    )
    spy_statistics = encode(
        # The same settlement republished three times, as CME and OPRA both do.
        [statistic(101, 1.05, ts_ref=OPEN_NS) for _ in range(3)],
        dataset="OPRA.PILLAR",
        schema=databento_dbn.Schema.STATISTICS,
        stype_in=databento_dbn.SType.PARENT,
        symbol="SPY.OPT",
    )
    xsp_definitions = encode(
        [
            option_definition(201, "C", 600.0, root="XSP"),
            option_definition(202, "P", 600.0, root="XSP"),
        ],
        dataset="OPRA.PILLAR",
        schema=databento_dbn.Schema.DEFINITION,
        stype_in=databento_dbn.SType.PARENT,
        symbol="XSP.OPT",
    )
    xsp_quotes = encode(
        [
            consolidated_quote(instrument_id, minute, 1.00 + 0.01 * minute,
                               1.10 + 0.01 * minute)
            for minute in range(1, SESSION_MINUTES + 1)
            for instrument_id in (201, 202)
        ],
        dataset="OPRA.PILLAR",
        schema=databento_dbn.Schema.CBBO_1M,
        stype_in=databento_dbn.SType.PARENT,
        symbol="XSP.OPT",
    )
    underlying = encode(
        [
            plain_quote(15144, minute, 610.00 + 0.01 * minute, 610.02 + 0.01 * minute,
                        flags=underlying_flags)
            for minute in range(1, SESSION_MINUTES + 1)
        ],
        dataset="EQUS.MINI",
        schema=databento_dbn.Schema.BBO_1M,
        stype_in=databento_dbn.SType.RAW_SYMBOL,
        symbol="SPY",
    )
    es_definitions = encode(
        [
            future_definition(
                ES_SPREAD, "ESU6-ESZ6", databento_dbn.InstrumentClass.FUTURE_SPREAD,
                dt.date(2026, 9, 18),
            ),
            future_definition(
                ES_FRONT, "ESU6", databento_dbn.InstrumentClass.FUTURE, dt.date(2026, 9, 18)
            ),
            future_definition(
                ES_DEFERRED, "ESZ6", databento_dbn.InstrumentClass.FUTURE,
                dt.date(2026, 12, 18),
            ),
        ],
        dataset="GLBX.MDP3",
        schema=databento_dbn.Schema.DEFINITION,
        stype_in=databento_dbn.SType.PARENT,
        symbol="ES.FUT",
    )
    outright_minutes = (
        es_outright_minutes
        if es_outright_minutes is not None
        else range(1, SESSION_MINUTES + 1)
    )
    es_records: list[Any] = []
    for minute in range(1, SESSION_MINUTES + 1):
        # The spread is emitted FIRST in every minute, so an audit that keeps
        # "whichever instrument the decoder yielded first" would pick it, and an
        # audit that pooled populations would let it carry outright coverage.
        es_records.append(plain_quote(ES_SPREAD, minute, 60.00, 60.25, publisher_id=1))
        if orphan_future_id is not None:
            es_records.append(
                plain_quote(orphan_future_id, minute, 6100.0, 6100.25, publisher_id=1)
            )
        if minute in outright_minutes:
            es_records.append(
                plain_quote(ES_FRONT, minute, 6100.00 + minute, 6100.25 + minute,
                            publisher_id=1)
            )
            es_records.append(
                plain_quote(ES_DEFERRED, minute, 6160.00 + minute, 6160.25 + minute,
                            publisher_id=1)
            )
    es_quotes = encode(
        es_records,
        dataset="GLBX.MDP3",
        schema=databento_dbn.Schema.BBO_1M,
        stype_in=databento_dbn.SType.PARENT,
        symbol="ES.FUT",
    )
    es_statistics = encode(
        [
            statistic(ES_FRONT, 6105.0, ts_ref=OPEN_NS),
            statistic(ES_SPREAD, 60.0, ts_ref=OPEN_NS),
        ],
        dataset="GLBX.MDP3",
        schema=databento_dbn.Schema.STATISTICS,
        stype_in=databento_dbn.SType.PARENT,
        symbol="ES.FUT",
    )

    payloads = {
        "raw/opra/spy/definition/2026-07-30/REQ-SPY-DEF/opra-20260730.definition.dbn":
            spy_definitions,
        "raw/opra/spy/cbbo-1m/2026-07-30/REQ-SPY-CBBO/opra-20260730.cbbo-1m.dbn": spy_quotes,
        "raw/opra/spy/statistics/2026-07-30/REQ-SPY-STAT/opra-20260730.statistics.dbn":
            spy_statistics,
        "raw/opra/xsp/definition/2026-07-30/REQ-XSP-DEF/opra-20260730.definition.dbn":
            xsp_definitions,
        "raw/opra/xsp/cbbo-1m/2026-07-30/REQ-XSP-CBBO/opra-20260730.cbbo-1m.dbn": xsp_quotes,
        "raw/equs/spy/bbo-1m/2026-07-30/REQ-BBO/equs-20260730.bbo-1m.dbn": underlying,
        "raw/glbx/es/definition/2026-07-30/REQ-ES-DEF/glbx-20260730.definition.dbn":
            es_definitions,
        "raw/glbx/es/bbo-1m/2026-07-30/REQ-ES-BBO/glbx-20260730.bbo-1m.dbn": es_quotes,
        "raw/glbx/es/statistics/2026-07-30/REQ-ES-STAT/glbx-20260730.statistics.dbn":
            es_statistics,
    }
    for relative, payload in payloads.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        write_condition(
            target.parent, condition_dates or ["2026-07-30"], condition=condition
        )

    fred = root / "raw" / "fred" / "SOFR.csv"
    fred.parent.mkdir(parents=True, exist_ok=True)
    fred.write_text("observation_date,SOFR\n2026-07-30,3.65\n", encoding="utf-8")

    requests = root / "manifests" / "requests.csv"
    requests.parent.mkdir(parents=True, exist_ok=True)
    requests.write_text("request_id\nREQ-SPY-DEF\n", encoding="utf-8")

    # Three declared sources in three different states.
    (root / "raw" / "issuer").mkdir(parents=True)  # present but empty
    borrow = root / "raw" / "borrow"  # present with content
    borrow.mkdir(parents=True)
    (borrow / "fees.csv").write_text("date,fee\n2026-07-30,0.0025\n", encoding="utf-8")
    # raw/occ is deliberately never created: absent.

    covered = [
        path.relative_to(root).as_posix()
        for path in sorted(root.rglob("*"))
        if path.is_file() and "sha256sums" not in path.name
    ]
    write_manifest(root, covered)


def write_config(
    root: Path,
    *,
    entries: int | None = None,
    total: int | None = None,
    es_role: str = "futures_control",
    thresholds: str = "[0.02, 0.10, 0.50]",
) -> Path:
    archive = root / "archive"
    files = [path for path in archive.rglob("*") if path.is_file()]
    resolved_total = total if total is not None else len(files)
    resolved_entries = entries if entries is not None else resolved_total - 1
    path = root / "audit.toml"
    path.write_text(
        CONFIG_TEMPLATE.format(
            entries=resolved_entries,
            total=resolved_total,
            es_role=es_role,
            thresholds=thresholds,
        ),
        encoding="utf-8",
    )
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    build_archive(tmp_path / "archive")
    write_config(tmp_path)
    return tmp_path


def run(project_root: Path) -> dict[str, Any]:
    config = load_market_config(project_root / "audit.toml")
    return Auditor(config, project_root=project_root).run()


def capability(report: dict[str, Any], name: str) -> dict[str, Any]:
    return report["capabilities"]["results"][name]


class TestEndToEnd:
    def test_the_audit_completes_and_reports_every_section(self, project: Path) -> None:
        report = run(project)
        assert report["schema_version"] == "market-feasibility-report/2"
        assert report["manifest"]["altered"] == []
        assert report["inputs"]["payloads_decoded"] == 9
        assert set(report["per_date"]) == {"2026-07-30"}
        assert "capabilities" in report
        assert "declared_sources" in report
        assert "vendor_conditions" in report
        assert "storage" in report

    def test_definitions_resolve_and_are_persisted(self, project: Path) -> None:
        report = run(project)
        block = report["per_date"]["2026-07-30"]["products"]["spy-options"]
        assert block["definition_resolution"]["records"] == 2
        assert block["definition_resolution"]["resolved"] == 2
        assert block["definition_resolution"]["failed"] == 0
        assert block["definition_resolution"]["standard_root"] == 2
        stored = pq.read_table(
            project / "processed" / "product=spy-options" / "schema=definition"
            / "date=2026-07-30" / "part-00000.parquet"
        )
        assert stored.num_rows == 2
        assert sorted(stored.column("option_type").to_pylist()) == ["C", "P"]
        assert stored.column("is_standard_root").to_pylist() == [True, True]

    def test_quotes_are_normalized_graded_and_partitioned(self, project: Path) -> None:
        report = run(project)
        quotes = report["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        assert quotes["records"] == 2 * SESSION_MINUTES
        assert quotes["in_session_records"] == 2 * SESSION_MINUTES
        assert quotes["session_slots"] == SESSION_MINUTES
        assert quotes["tradable_coverage"]["contract_minute_fill_ratio"] == pytest.approx(1.0)
        assert quotes["quality_counts"]["ok"] == 2 * SESSION_MINUTES

        stored = pq.read_table(
            project / "processed" / "product=spy-options" / "schema=cbbo-1m"
            / "date=2026-07-30" / "part-00000.parquet"
        )
        assert stored.num_rows == 2 * SESSION_MINUTES
        assert stored.schema.field("ts_recv_ns").type == pa.timestamp("ns", tz="UTC")
        assert set(stored.column("bid_venue").to_pylist()) == {24}
        assert set(stored.column("tradable_core").to_pylist()) == {True}
        assert set(stored.column("vendor_flags").to_pylist()) == {FLAG_LAST | FLAG_TOB}


class TestCoverageDenominators:
    """C2: the union metric stays descriptive; coverage is per contract-minute."""

    def test_the_union_metric_is_kept_but_labelled_descriptive(self, project: Path) -> None:
        quotes = run(project)["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        liveness = quotes["descriptive_liveness"]
        assert liveness["session_slots_with_any_quote"] == SESSION_MINUTES
        assert "never used as an acceptance gate" in liveness["note"]

    def test_coverage_uses_the_resolved_universe_as_its_denominator(
        self, tmp_path: Path
    ) -> None:
        """A chain where one contract never quotes must not report full coverage."""
        build_archive(tmp_path / "archive")
        write_config(tmp_path)
        report = run(tmp_path)
        quotes = report["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        coverage = quotes["tradable_coverage"]
        assert coverage["denominator_basis"] == "resolved primary universe"
        assert coverage["denominator_contracts"] == 2
        assert coverage["contract_minutes_expected"] == 2 * SESSION_MINUTES

    def test_partial_coverage_is_measured_not_papered_over(self, tmp_path: Path) -> None:
        build_archive(tmp_path / "archive", option_minutes=range(1, 4))
        write_config(tmp_path)
        quotes = run(tmp_path)["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        coverage = quotes["tradable_coverage"]
        assert coverage["contract_minutes_present"] == 6
        assert coverage["contract_minutes_expected"] == 10
        assert coverage["contract_minute_fill_ratio"] == pytest.approx(0.6)
        assert coverage["per_contract_minutes"]["maximum"] == pytest.approx(3.0)

    def test_raw_and_tradable_coverage_are_both_reported(self, project: Path) -> None:
        quotes = run(project)["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        assert "raw_quality_coverage" in quotes
        assert "tradable_coverage" in quotes

    def test_every_exploratory_threshold_is_reported(self, project: Path) -> None:
        quotes = run(project)["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        thresholds = quotes["tradable_coverage_by_relative_spread_threshold"]
        assert [item["relative_spread_at_most"] for item in thresholds] == [0.02, 0.10, 0.50]
        assert "consulted by any capability check" in quotes["threshold_note"]

    def test_synchronization_is_reported_per_contract(self, project: Path) -> None:
        sync = run(project)["per_date"]["2026-07-30"]["synchronization"]
        primary = next(item for item in sync if item["role"] == "primary")
        assert primary["contracts_measured"] == 2
        assert primary["contracts_synchronized_in_every_own_minute"] == 2
        assert primary["per_contract_synchronized_minutes"]["minimum"] == pytest.approx(
            float(SESSION_MINUTES)
        )
        assert "descriptive_liveness" in primary

    def test_a_reference_gap_shows_up_per_contract(self, tmp_path: Path) -> None:
        """A flagged underlying is not tradable, so no option minute is synchronized."""
        build_archive(tmp_path / "archive", underlying_flags=FLAG_LAST | FLAG_BAD_TS_RECV)
        write_config(tmp_path)
        sync = run(tmp_path)["per_date"]["2026-07-30"]["synchronization"]
        primary = next(item for item in sync if item["role"] == "primary")
        assert primary["reference_tradable_minutes"] == 0
        assert primary["contracts_with_no_synchronized_minute"] == 2
        assert primary["per_contract_reference_overlap_fraction"]["maximum"] == pytest.approx(
            0.0
        )


class TestFuturesReference:
    """H1/H2: a spread must never become the futures reference."""

    def test_the_reference_is_the_nearest_unexpired_outright(self, project: Path) -> None:
        block = run(project)["per_date"]["2026-07-30"]["products"]["es-futures"]
        selection = block["reference_selection"]
        assert selection["selection_rule"] == "nearest_unexpired_outright"
        assert selection["selected_instrument"] == "ESU6"
        assert selection["reference_mid"] == pytest.approx(6105.125 + 0.0)

    def test_the_spread_quoted_first_every_minute_and_still_lost(
        self, project: Path
    ) -> None:
        """The decoder emits the spread first; selection must not follow arrival order."""
        block = run(project)["per_date"]["2026-07-30"]["products"]["es-futures"]
        assert block["reference_selection"]["selected_instrument"] != "ESU6-ESZ6"
        assert block["reference_selection"]["reference_mid"] > 1000.0

    def test_outright_and_multi_leg_quote_coverage_are_separated(
        self, project: Path
    ) -> None:
        block = run(project)["per_date"]["2026-07-30"]["products"]["es-futures"]
        split = block["quotes_by_instrument_class"]
        assert split["primary"]["instruments_with_quotes"] == 2
        assert split["multi_leg"]["instruments_with_quotes"] == 1
        assert split["primary"]["symbols"] == ["ESU6", "ESZ6"]
        assert split["multi_leg"]["symbols"] == ["ESU6-ESZ6"]

    def test_the_definition_block_separates_outrights_from_spreads(
        self, project: Path
    ) -> None:
        block = run(project)["per_date"]["2026-07-30"]["products"]["es-futures"]
        assert block["futures"]["outright_contracts"] == 2
        assert block["futures"]["multi_leg_contracts"] == 1
        assert block["futures"]["multi_leg_symbols"] == ["ESU6-ESZ6"]

    def test_futures_statistics_separate_outrights_from_spreads(
        self, project: Path
    ) -> None:
        block = run(project)["per_date"]["2026-07-30"]["products"]["es-futures"]
        settlement = block["statistics"]["by_stat_type"]["SETTLEMENT_PRICE"]
        assert settlement["distinct_outright_records"] == 1
        assert settlement["distinct_multi_leg_records"] == 1
        assert settlement["outright_instruments"] == 1


class TestFuturesPopulation:
    """MEDIUM 1: the futures headline block is the outrights, consistently."""

    def test_a_spread_that_quotes_every_minute_cannot_carry_outright_coverage(
        self, tmp_path: Path
    ) -> None:
        """The reviewer's scenario, end to end."""
        build_archive(tmp_path / "archive", es_outright_minutes=range(1, 3))
        write_config(tmp_path)
        block = run(tmp_path)["per_date"]["2026-07-30"]["products"]["es-futures"]
        quotes = block["quotes"]
        coverage = quotes["tradable_coverage"]
        # Two outrights quoting in two of five minutes each: 4 of 10 cells.
        assert coverage["population"] == "primary"
        assert coverage["denominator_contracts"] == 2
        assert coverage["contract_minutes_present"] == 4
        assert coverage["contract_minutes_expected"] == 10
        assert coverage["contract_minute_fill_ratio"] == pytest.approx(0.4)
        # The spread still quoted every minute; that is descriptive only.
        assert quotes["descriptive_liveness"]["session_slots_with_any_quote"] == 5
        assert quotes["descriptive_liveness"]["session_slots_with_any_primary_quote"] == 2
        assert block["quotes_by_instrument_class"]["multi_leg"][
            "contract_minutes_present"
        ] == 5

    def test_the_headline_spread_distribution_excludes_multi_leg_books(
        self, project: Path
    ) -> None:
        block = run(project)["per_date"]["2026-07-30"]["products"]["es-futures"]
        headline = block["quotes"]["quoted_spread_absolute"]
        multi = block["quotes_by_instrument_class"]["multi_leg"]["quoted_spread_absolute"]
        assert headline["count"] == 2 * SESSION_MINUTES
        assert headline["maximum"] == pytest.approx(0.25)
        assert multi["count"] == SESSION_MINUTES

    def test_a_spread_only_minute_does_not_synchronize_an_option_chain(
        self, tmp_path: Path
    ) -> None:
        build_archive(tmp_path / "archive", es_outright_minutes=range(1, 3))
        write_config(tmp_path)
        sync = run(tmp_path)["per_date"]["2026-07-30"]["synchronization"]
        futures = next(item for item in sync if item["reference_product"] == "es-futures")
        assert futures["reference_instrument"] == "ESU6"
        assert futures["reference_tradable_minutes"] == 2
        assert futures["per_contract_synchronized_minutes"]["maximum"] == pytest.approx(2.0)

    def test_a_futures_instrument_without_a_definition_is_gated(
        self, tmp_path: Path
    ) -> None:
        build_archive(tmp_path / "archive", orphan_future_id=999)
        write_config(tmp_path)
        report = run(tmp_path)
        result = capability(report, "ingestion_feasibility")
        assert result["satisfied"] is False
        observed = {
            check["requirement"]: check["observed"]
            for check in result["checks"]
        }
        detail = observed["every quoting instrument has a resolved definition"]
        assert detail.startswith("1 quoted instruments without a definition")
        assert "es-futures" in detail

    def test_the_synchronization_universe_includes_contracts_that_never_traded(
        self, tmp_path: Path
    ) -> None:
        build_archive(tmp_path / "archive", put_minutes=range(0, 0))
        write_config(tmp_path)
        sync = run(tmp_path)["per_date"]["2026-07-30"]["synchronization"]
        primary = next(item for item in sync if item["role"] == "primary")
        assert primary["contracts_measured"] == 2
        assert primary["contracts_with_no_tradable_minute"] == 1
        assert "including contracts with no tradable minute" in (
            primary["contracts_measured_basis"]
        )


class TestSameMinutePairing:
    """MEDIUM 2: a paired strike must be observable in one minute."""

    def test_disjoint_call_and_put_minutes_yield_no_paired_strike(
        self, tmp_path: Path
    ) -> None:
        build_archive(
            tmp_path / "archive", call_minutes=range(1, 3), put_minutes=range(3, 6)
        )
        write_config(tmp_path)
        report = run(tmp_path)
        coverage = report["per_date"]["2026-07-30"]["option_coverage"]["spy-options"]
        assert coverage["paired_strikes_total"] == 0
        assert coverage["expiries_with_minimum_paired_strikes"] == 0

    def test_overlapping_minutes_yield_a_paired_strike(self, tmp_path: Path) -> None:
        build_archive(
            tmp_path / "archive", call_minutes=range(1, 4), put_minutes=range(3, 6)
        )
        write_config(tmp_path)
        coverage = run(tmp_path)["per_date"]["2026-07-30"]["option_coverage"]["spy-options"]
        assert coverage["paired_strikes_total"] == 1
        assert coverage["common_tradable_minutes_per_paired_strike"][
            "maximum"
        ] == pytest.approx(1.0)

    def test_surface_readiness_follows_the_corrected_pairing(
        self, tmp_path: Path
    ) -> None:
        build_archive(
            tmp_path / "archive", call_minutes=range(1, 3), put_minutes=range(3, 6)
        )
        write_config(tmp_path)
        result = capability(run(tmp_path), "xsp_european_surface_readiness")
        paired = next(
            check
            for check in result["checks"]
            if "tradable call/put strike pairs" in check["requirement"]
        )
        assert paired["satisfied"] is False


class TestReferenceAttribution:
    """H4/H5: no product silently inherits another product's spot."""

    def test_the_american_chain_uses_its_own_declared_underlying(
        self, project: Path
    ) -> None:
        coverage = run(project)["per_date"]["2026-07-30"]["option_coverage"]["spy-options"]
        spot = coverage["spot_reference"]
        assert spot["basis"] == "direct"
        assert spot["source_product"] == "spy-underlying"
        assert coverage["moneyness_is_proxy_derived"] is False

    def test_the_control_chain_is_labelled_proxy_derived_with_its_caveat(
        self, project: Path
    ) -> None:
        coverage = run(project)["per_date"]["2026-07-30"]["option_coverage"]["xsp-options"]
        spot = coverage["spot_reference"]
        assert spot["basis"] == "proxy"
        assert spot["source_product"] == "spy-underlying"
        assert "basis is unmeasured" in spot["caveat"]
        assert coverage["moneyness_is_proxy_derived"] is True

    def test_the_underlying_feed_is_reported_as_partial_venue(self, project: Path) -> None:
        spot = run(project)["per_date"]["2026-07-30"]["option_coverage"]["spy-options"][
            "spot_reference"
        ]
        assert spot["feed_class"] == "partial_venue_consolidated"
        assert spot["is_nbbo_grade"] is False

    def test_the_spot_uncertainty_carries_the_reference_spread_distribution(
        self, project: Path
    ) -> None:
        uncertainty = run(project)["per_date"]["2026-07-30"]["option_coverage"][
            "spy-options"
        ]["spot_reference"]["spot_reference_uncertainty"]
        assert uncertainty["is_nbbo_grade"] is False
        distribution = uncertainty["quoted_spread_distribution"]
        assert distribution["count"] == SESSION_MINUTES
        assert distribution["maximum"] == pytest.approx(0.02)

    def test_forward_moneyness_is_not_claimed(self, project: Path) -> None:
        coverage = run(project)["per_date"]["2026-07-30"]["option_coverage"]["spy-options"]
        assert "not forward moneyness" in coverage["moneyness_definition"]
        assert "task 9B" in coverage["moneyness_definition"]


class TestDeclaredSourceProbes:
    """C1: the absent-input answer must come from the archive, not from the source."""

    def test_the_three_states_are_distinguished(self, project: Path) -> None:
        sources = {item["key"]: item for item in run(project)["declared_sources"]}
        assert sources["discrete_dividend_schedule"]["state"] == "present_but_empty"
        assert sources["corporate_action_and_contract_adjustment_history"]["state"] == "absent"
        assert sources["securities_lending_borrow_rate"]["state"] == "present"
        assert sources["securities_lending_borrow_rate"]["satisfied"] is True

    def test_populating_a_source_changes_the_answer(self, project: Path) -> None:
        """The decisive anti-hardcoding test: the verdict must move with the archive."""
        before = capability(run(project), "spy_american_calibration_readiness")
        assert any(
            "discrete_dividend_schedule" in item for item in before["unsatisfied_requirements"]
        )

        issuer = project / "archive" / "raw" / "issuer"
        (issuer / "distributions.csv").write_text(
            "ex_date,amount\n2026-06-19,1.76\n", encoding="utf-8"
        )
        (project / "archive" / "raw" / "occ").mkdir()
        (project / "archive" / "raw" / "occ" / "memo.txt").write_text("none", encoding="utf-8")
        write_config(project)
        write_manifest(
            project / "archive",
            [
                path.relative_to(project / "archive").as_posix()
                for path in sorted((project / "archive").rglob("*"))
                if path.is_file() and "sha256sums" not in path.name
            ],
        )

        after = run(project)
        sources = {item["key"]: item for item in after["declared_sources"]}
        assert sources["discrete_dividend_schedule"]["state"] == "present"
        assert sources["corporate_action_and_contract_adjustment_history"]["state"] == "present"
        result = capability(after, "spy_american_calibration_readiness")
        assert not any(
            "declared source" in item for item in result["unsatisfied_requirements"]
        )

    def test_an_empty_declared_source_is_a_finding(self, project: Path) -> None:
        kinds = {finding["kind"] for finding in run(project)["findings"]}
        assert "declared_source_present_but_empty" in kinds
        assert "declared_source_absent" in kinds


class TestVendorConditions:
    def test_available_conditions_are_recorded(self, project: Path) -> None:
        conditions = run(project)["vendor_conditions"]
        assert conditions["all_declared_dates_available"] is True
        assert len(conditions["entries"]) == 9

    def test_a_degraded_date_blocks_archive_integrity(self, tmp_path: Path) -> None:
        build_archive(tmp_path / "archive", condition="degraded")
        write_config(tmp_path)
        report = run(tmp_path)
        assert capability(report, "archive_integrity")["satisfied"] is False
        kinds = {finding["kind"] for finding in report["findings"]}
        assert "vendor_condition_not_available" in kinds


class TestStatisticsDeduplication:
    def test_a_republished_settlement_is_counted_once(self, project: Path) -> None:
        block = run(project)["per_date"]["2026-07-30"]["products"]["spy-options"]
        settlement = block["statistics"]["by_stat_type"]["SETTLEMENT_PRICE"]
        assert settlement["records"] == 3
        assert settlement["distinct_records"] == 1
        assert block["statistics"]["repeat_records"] == 2

    def test_the_parquet_marks_the_republished_rows(self, project: Path) -> None:
        run(project)
        stored = pq.read_table(
            project / "processed" / "product=spy-options" / "schema=statistics"
            / "date=2026-07-30" / "part-00000.parquet"
        )
        assert stored.column("is_repeat_of_earlier_record").to_pylist() == [False, True, True]


class TestCapabilities:
    def test_every_scoped_capability_is_reported(self, project: Path) -> None:
        results = run(project)["capabilities"]["results"]
        assert list(results) == [
            "archive_integrity",
            "ingestion_feasibility",
            "xsp_european_surface_readiness",
            "spy_american_calibration_readiness",
            "replication_readiness",
        ]

    def test_every_check_records_what_it_was_decided_by(self, project: Path) -> None:
        for result in run(project)["capabilities"]["results"].values():
            assert result["checks"]
            for check in result["checks"]:
                assert check["requirement"]
                assert check["observed"]
                assert isinstance(check["satisfied"], bool)

    def test_a_clean_archive_satisfies_integrity_and_ingestion(self, project: Path) -> None:
        report = run(project)
        assert capability(report, "archive_integrity")["satisfied"] is True
        assert capability(report, "ingestion_feasibility")["satisfied"] is True

    def test_three_dates_prevent_replication_readiness(self, project: Path) -> None:
        result = capability(run(project), "replication_readiness")
        assert result["satisfied"] is False
        assert any("60 distinct sessions" in item for item in result["unsatisfied_requirements"])
        observed = [
            check["observed"]
            for check in result["checks"]
            if "distinct sessions" in check["requirement"]
        ]
        assert observed == ["1 sessions declared and ingested"]

    def test_american_readiness_fails_on_observed_evidence_not_a_constant(
        self, project: Path
    ) -> None:
        result = capability(run(project), "spy_american_calibration_readiness")
        assert result["satisfied"] is False
        reasons = result["unsatisfied_requirements"]
        assert any("consolidated NBBO" in item for item in reasons)
        assert any("exercise style" in item for item in reasons)
        observed = {check["requirement"]: check["observed"] for check in result["checks"]}
        assert "partial_venue_consolidated" in observed[
            "the underlying reference feed is a consolidated NBBO"
        ]
        assert "0/2 contracts" in observed[
            "the option definitions evidence an exercise style"
        ]

    def test_a_tampered_archive_fails_the_integrity_capability(self, project: Path) -> None:
        """A genuine mutation: the capability must flip, not merely stay true."""
        config = load_market_config(project / "audit.toml")
        auditor = Auditor(config, project_root=project)
        clean = verify_archive(
            project / "archive",
            config.sha256_manifest,
            expected_entry_count=config.expected_manifest_entries,
            expected_total_files=config.expected_total_files,
        )
        coverage = {
            "expected_request_dates": 1,
            "request_dates_with_an_available_statement": 1,
            "request_dates_without_any_statement": [],
            "request_dates_not_available": [],
            "every_decoded_request_date_is_available": True,
        }
        before = auditor._archive_integrity(clean, coverage)
        assert before["satisfied"] is True

        target = project / "archive" / "raw" / "fred" / "SOFR.csv"
        target.write_text("observation_date,SOFR\n2026-07-30,9.99\n", encoding="utf-8")
        tampered = verify_archive(
            project / "archive",
            config.sha256_manifest,
            expected_entry_count=config.expected_manifest_entries,
            expected_total_files=config.expected_total_files,
        )
        after = auditor._archive_integrity(tampered, coverage)
        assert after["satisfied"] is False
        assert after["unsatisfied_requirements"] == [
            "the archive matches its SHA-256 manifest exactly"
        ]
        assert "1 altered" in after["checks"][0]["observed"]

    def test_surface_readiness_needs_paired_tradable_strikes(self, tmp_path: Path) -> None:
        build_archive(tmp_path / "archive")
        write_config(tmp_path)
        result = capability(run(tmp_path), "xsp_european_surface_readiness")
        paired = [
            check
            for check in result["checks"]
            if "tradable call/put strike pairs" in check["requirement"]
        ]
        assert len(paired) == 1
        # One strike carries both a call and a put, but the minimum is two pairs.
        assert paired[0]["satisfied"] is False
        assert "1 such expiries" in paired[0]["observed"] or "0 such" in paired[0]["observed"]

    def test_a_recommended_source_is_advisory_not_gating(self, project: Path) -> None:
        result = capability(run(project), "xsp_european_surface_readiness")
        gating = {check["requirement"] for check in result["checks"] if check["gating"]}
        assert not any("underlying_index_level" in item for item in gating)


class TestThresholdProvenance:
    """MEDIUM 3: the provenance of every minimum must be stated, not implied."""

    def test_capability_minima_are_serialized_with_their_provenance(
        self, project: Path
    ) -> None:
        minima = run(project)["config"]["capability_minima"]
        assert minima["session_minima_status"] == "provisional_pilot_target"
        assert minima["session_minima_are_frozen"] is False
        assert "after observing" in minima["session_minima_provenance"]
        assert minima["structural_minima_provenance"]

    def test_no_blanket_claim_that_no_threshold_followed_the_results(
        self, project: Path
    ) -> None:
        """The old rule text claimed this of every threshold. It was not true."""
        rule = run(project)["capabilities"]["rule"]
        assert "no threshold was chosen after the results were observed" not in rule
        assert "provisional pilot targets until frozen" in rule

    def test_exploratory_thresholds_cannot_become_capability_gates(
        self, tmp_path: Path
    ) -> None:
        """Behavioural: move every threshold and no capability outcome may move."""
        build_archive(tmp_path / "archive")
        write_config(tmp_path, thresholds="[0.02, 0.10, 0.50]")
        baseline = run(tmp_path)
        write_config(tmp_path, thresholds="[1e-09, 2e-09, 3e-09]")
        starved = run(tmp_path)

        # The thresholds really did bite: no contract-minute survives them now.
        original = baseline["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        mutated = starved["per_date"]["2026-07-30"]["products"]["spy-options"]["quotes"]
        assert original["tradable_coverage_by_relative_spread_threshold"][-1][
            "contract_minutes_present"
        ] > 0
        assert all(
            block["contract_minutes_present"] == 0
            for block in mutated["tradable_coverage_by_relative_spread_threshold"]
        )
        # And no capability outcome moved.
        assert {
            name: result["outcome"]
            for name, result in baseline["capabilities"]["results"].items()
        } == {
            name: result["outcome"]
            for name, result in starved["capabilities"]["results"].items()
        }


class TestCapabilityScope:
    def test_every_capability_states_what_its_outcome_means(self, project: Path) -> None:
        for name, result in run(project)["capabilities"]["results"].items():
            assert result["scope"], name
            assert len(result["scope"].split()) >= 8, name

    def test_the_ingestion_scope_disclaims_fitness_for_a_study(
        self, project: Path
    ) -> None:
        scope = capability(run(project), "ingestion_feasibility")["scope"]
        assert "says nothing about whether the resulting data supports any study" in scope


class TestVendorConditionCoverage:
    def test_every_decoded_request_and_date_must_be_stated_available(
        self, tmp_path: Path
    ) -> None:
        """A statement about another date must not vouch for the decoded one."""
        build_archive(tmp_path / "archive", condition_dates=["2026-07-29"])
        write_config(tmp_path)
        report = run(tmp_path)
        coverage = report["vendor_conditions"]["coverage"]
        assert coverage["expected_request_dates"] == 9
        assert coverage["request_dates_with_an_available_statement"] == 0
        assert len(coverage["request_dates_without_any_statement"]) == 9
        assert coverage["every_decoded_request_date_is_available"] is False
        assert capability(report, "archive_integrity")["satisfied"] is False
        kinds = {finding["kind"] for finding in report["findings"]}
        assert "vendor_condition_absent_for_decoded_date" in kinds

    def test_a_complete_statement_satisfies_the_check(self, project: Path) -> None:
        coverage = run(project)["vendor_conditions"]["coverage"]
        assert coverage["expected_request_dates"] == 9
        assert coverage["request_dates_with_an_available_statement"] == 9
        assert coverage["every_decoded_request_date_is_available"] is True


class TestRateAvailabilityIsDerived:
    def test_a_rate_product_with_quotes_reports_intraday_availability(
        self, tmp_path: Path
    ) -> None:
        """The old text was a constant False; this must follow the schemas."""
        build_archive(tmp_path / "archive")
        write_config(tmp_path, es_role="rate_control")
        rates = run(tmp_path)["rates"]["rate_futures"]
        assert rates["quote_schemas_acquired"] == ["bbo-1m"]
        assert rates["quote_records_observed"] > 0
        assert rates["intraday_rate_quotes_available"] is True
        assert rates["why_no_intraday_rate_quotes"] is None
        assert "SETTLEMENT_PRICE" in rates["statistic_types_observed"]


class TestNoDividendClaims:
    def test_the_report_defers_parity_and_dividend_work(self, project: Path) -> None:
        report = run(project)
        items = {entry["item"] for entry in report["deferred_work"]}
        assert any("parity" in item for item in items)
        assert any("dividend" in item for item in items)
        assert all(entry["task"] == "9B" for entry in report["deferred_work"])

    def test_no_reported_section_claims_an_observed_or_inferred_dividend(
        self, project: Path
    ) -> None:
        report = run(project)
        # `deferred_work` is the disclaimer itself, so it is excluded from the
        # scan and asserted separately below.
        report.pop("deferred_work")
        text = json.dumps(report).lower()
        for claim in (
            "dividend is observed",
            "dividends are observed",
            "inferred dividend",
            "implied dividend",
            "dividend schedule recovered",
            "put-call parity",
        ):
            assert claim not in text

    def test_the_dividend_disclaimer_is_explicit(self, project: Path) -> None:
        notes = " ".join(entry["note"] for entry in run(project)["deferred_work"])
        assert "no independent dividend ground truth" in notes
        assert "nothing here asserts" in notes.lower()

    def test_deliberately_deferred_low_findings_are_documented(self, project: Path) -> None:
        deferred = run(project)["deferred_low_findings"]
        assert deferred
        for entry in deferred:
            assert entry["finding"]
            assert entry["why"]


class TestDeterminism:
    def test_two_runs_produce_identical_reports(self, project: Path) -> None:
        first = json.dumps(run(project), sort_keys=True, default=str)
        second = json.dumps(run(project), sort_keys=True, default=str)
        assert first == second

    def test_two_runs_produce_byte_identical_parquet(self, project: Path) -> None:
        run(project)
        digests = {
            path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((project / "processed").rglob("*.parquet"))
        }
        assert len(digests) == 9
        run(project)
        repeated = {
            path.relative_to(project).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted((project / "processed").rglob("*.parquet"))
        }
        assert digests == repeated

    def test_findings_are_ordered_by_severity_not_alphabetically(
        self, project: Path
    ) -> None:
        findings = run(project)["findings"]
        order = {"blocking": 0, "warning": 1, "informational": 2}
        ranks = [order[item["severity"]] for item in findings]
        assert ranks == sorted(ranks)


class TestStalePartitions:
    def test_a_partition_from_an_earlier_run_is_reported_and_excluded(
        self, project: Path
    ) -> None:
        stale = (
            project / "processed" / "product=obsolete" / "schema=cbbo-1m"
            / "date=2020-01-01" / "part-00000.parquet"
        )
        stale.parent.mkdir(parents=True)
        stale.write_bytes(b"x" * 4096)
        report = run(project)
        assert report["storage"]["stale_partitions_excluded"] == [
            "product=obsolete/schema=cbbo-1m/date=2020-01-01/part-00000.parquet"
        ]
        assert "obsolete" not in report["storage"]["processed_bytes_by_product"]
        assert capability(report, "ingestion_feasibility")["satisfied"] is False
        kinds = {finding["kind"] for finding in report["findings"]}
        assert "stale_processed_partition" in kinds


class TestFailureHandling:
    def test_a_tampered_archive_stops_the_audit_before_any_output(
        self, project: Path
    ) -> None:
        target = project / "archive" / "raw" / "fred" / "SOFR.csv"
        target.write_text("observation_date,SOFR\n2026-07-30,9.99\n", encoding="utf-8")
        with pytest.raises(ManifestError, match="altered"):
            run(project)
        assert not (project / "processed").exists()

    def test_a_missing_payload_is_a_blocking_finding(self, project: Path) -> None:
        removed = (
            project / "archive" / "raw" / "equs" / "spy" / "bbo-1m" / "2026-07-30"
            / "REQ-BBO" / "equs-20260730.bbo-1m.dbn"
        )
        removed.unlink()
        write_manifest(
            project / "archive",
            [
                path.relative_to(project / "archive").as_posix()
                for path in sorted((project / "archive").rglob("*"))
                if path.is_file() and "sha256sums" not in path.name
            ],
        )
        write_config(project)
        report = run(project)
        kinds = {finding["kind"] for finding in report["findings"]}
        assert "expected_payload_missing" in kinds
        assert capability(report, "ingestion_feasibility")["satisfied"] is False

    def test_a_non_numeric_rate_on_a_session_date_is_blocking(self, project: Path) -> None:
        fred = project / "archive" / "raw" / "fred" / "SOFR.csv"
        fred.write_text("observation_date,SOFR\n2026-07-30,.\n", encoding="utf-8")
        write_manifest(
            project / "archive",
            [
                path.relative_to(project / "archive").as_posix()
                for path in sorted((project / "archive").rglob("*"))
                if path.is_file() and "sha256sums" not in path.name
            ],
        )
        report = run(project)
        series = report["rates"]["fred"]["series"]["SOFR"]
        assert series["numeric_on_all_session_dates"] is False
        assert series["session_dates_without_a_numeric_value"] == ["2026-07-30"]
        kinds = {finding["kind"] for finding in report["findings"]}
        assert "fred_series_missing_session_observation" in kinds

    def test_an_empty_archive_raises(self, tmp_path: Path) -> None:
        root = tmp_path / "archive"
        (root / "raw").mkdir(parents=True)
        (root / "raw" / "note.txt").write_text("nothing here", encoding="utf-8")
        write_manifest(root, ["raw/note.txt"])
        write_config(tmp_path, entries=1, total=2)
        config = load_market_config(tmp_path / "audit.toml")
        with pytest.raises(IngestError, match="no declared payloads"):
            Auditor(config, project_root=tmp_path).run()

    def test_discovery_ignores_dates_that_are_not_declared(self, project: Path) -> None:
        config = load_market_config(project / "audit.toml")
        stray = (
            project / "archive" / "raw" / "opra" / "spy" / "cbbo-1m" / "2026-08-03"
            / "REQ-OTHER" / "opra-20260803.cbbo-1m.dbn"
        )
        stray.parent.mkdir(parents=True)
        stray.write_bytes(b"DBN\x03ignored")
        inputs, findings = discover_inputs(config, project / "archive")
        assert all(item.date == SESSION_DATE for item in inputs)
        assert any(item["kind"] == "payload_for_undeclared_date" for item in findings)
