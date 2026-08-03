"""Build a tiny synthetic Task 9A output tree for the Task 9B tests.

Nothing here touches the private archive. The processed Parquet, the 9A report
and the raw manifest are all generated, and the option quotes are generated from
a chosen ``(D, F)`` so the reconstruction has a known right answer.

The synthetic market is deliberately small but structurally complete: two
sessions, two snapshots each, a European chain that satisfies parity exactly, an
American chain carrying a known carry offset plus a known strike-dependent
early-exercise contamination, a single underlying quote, and an SR3 settlement.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo

from differentiable_pricing.market.writer import (
    FUTURE_DEFINITION_SCHEMA,
    OPTION_DEFINITION_SCHEMA,
    QUOTE_SCHEMA,
    STATISTIC_SCHEMA,
    partition_path,
    write_table_atomically,
)

NEW_YORK: Final = ZoneInfo("America/New_York")
PUBLISHER: Final = 30

SESSIONS: Final = (dt.date(2026, 6, 17), dt.date(2026, 6, 18))
SNAPSHOTS_LOCAL: Final = (dt.time(10, 30), dt.time(15, 45))

EXPIRIES: Final = (dt.date(2026, 9, 18), dt.date(2026, 12, 18))
"""Both are matched between the two option products."""

EUROPEAN_ONLY_EXPIRY: Final = dt.date(2027, 3, 19)
"""Listed on the European product only, so it can never be matched."""

ZERO_DTE_EXPIRY: Final = SESSIONS[0]
"""A same-session expiry, listed on both products on the first session.

It exists so the zero-DTE exclusions are exercised: its tau is hours, so its
discount factor is pinned near one and its rate is meaningless. It must be
fitted and published, and excluded from rate interpretation and from every
stability aggregate.
"""

STRIKES: Final = tuple(700.0 + 10.0 * index for index in range(11))

SPOT_BID: Final = 749.98
SPOT_ASK: Final = 750.02
SPOT_MID: Final = 0.5 * (SPOT_BID + SPOT_ASK)

RATE: Final = 0.04
"""Continuously compounded rate the European chain is generated from."""

CARRY: Final = 0.015
"""Continuous dividend yield the American chain's forward is generated from."""

CONTAMINATION_PER_UNIT_STRIKE: Final = 0.01
"""Early-exercise premium added to the American put, growing with the strike."""

HALF_WIDTH: Final = 0.02


def discount_factor(tau: float) -> float:
    """The generated discount factor at ``tau``."""
    return math.exp(-RATE * tau)


def european_forward(tau: float) -> float:
    """The generated European forward at ``tau``. No carry: an index forward."""
    return SPOT_MID * math.exp(RATE * tau)


def american_forward(tau: float) -> float:
    """The generated American underlying's forward, carrying a dividend yield."""
    return SPOT_MID * math.exp((RATE - CARRY) * tau)


def year_fraction(snapshot: dt.datetime, expiry: dt.date) -> float:
    """ACT/365F to a 16:00 America/New_York expiry, matching the study's convention."""
    instant = dt.datetime.combine(expiry, dt.time(16, 0), tzinfo=NEW_YORK).astimezone(dt.UTC)
    return (instant - snapshot).total_seconds() / (365.0 * 86_400.0)


def snapshot_instants(date: dt.date) -> tuple[dt.datetime, ...]:
    return tuple(
        dt.datetime.combine(date, moment, tzinfo=NEW_YORK).astimezone(dt.UTC)
        for moment in SNAPSHOTS_LOCAL
    )


def _osi(root: str, expiry: dt.date, right: str, strike: float) -> str:
    return (
        f"{root:<6}{expiry.strftime('%y%m%d')}{right}{round(strike * 1000):08d}"
    )


def _instrument_id(root: str, expiry: dt.date, right: str, strike: float) -> int:
    digest = hashlib.sha256(_osi(root, expiry, right, strike).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % 1_000_000_000 + 1


def _definition_row(root: str, expiry: dt.date, right: str, strike: float) -> dict[str, Any]:
    return {
        "instrument_id": _instrument_id(root, expiry, right, strike),
        "raw_symbol": _osi(root, expiry, right, strike),
        "publisher_id": PUBLISHER,
        "root": root,
        "is_standard_root": True,
        "underlying": root,
        "option_type": right,
        "strike": strike,
        # Midnight UTC, exactly as the real archive stamps it: a date and no
        # settlement instant.
        "expiration_ns": dt.datetime.combine(expiry, dt.time(0, 0), tzinfo=dt.UTC),
        "expiration_date": expiry,
        "currency": "USD",
        "exchange": "OPRA",
        "underlying_id": None,
        # Unpopulated, exactly as in the real archive. The identifiability
        # matrix must classify these from the observation, not from a default.
        "contract_multiplier": None,
        "unit_of_measure_quantity": None,
        "original_contract_size": None,
        "exercise_style": None,
        "security_update_action": "A",
        "definition_ts_recv_ns": dt.datetime.combine(expiry, dt.time(0, 0), tzinfo=dt.UTC),
    }


def _quote_row(
    instrument_id: int, instant: dt.datetime, bid: float, ask: float, *, tradable: bool = True
) -> dict[str, Any]:
    mid = 0.5 * (bid + ask)
    return {
        "ts_recv_ns": instant,
        "ts_event_ns": instant,
        "instrument_id": instrument_id,
        "publisher_id": PUBLISHER,
        "bid_price": bid,
        "ask_price": ask,
        "bid_size": 10,
        "ask_size": 10,
        "mid_price": mid,
        "quoted_spread": ask - bid,
        "relative_spread": (ask - bid) / mid if mid else None,
        "bid_venue": PUBLISHER,
        "ask_venue": PUBLISHER,
        "in_regular_session": True,
        "quality": "ok",
        "vendor_flags": 0,
        "vendor_flag_names": "",
        "bad_ts_recv": False,
        "maybe_bad_book": False,
        "tradable_core": tradable,
    }


def _leg_quotes(combination: float, half_width: float) -> tuple[float, float, float, float]:
    """Split a combination value into call and put quotes around a fixed put level."""
    put_mid = 20.0
    call_mid = combination + put_mid
    return (
        call_mid - half_width,
        call_mid + half_width,
        put_mid - half_width,
        put_mid + half_width,
    )


def build_processed_tree(root: Path) -> Path:
    """Write the synthetic normalized Parquet tree and return its path."""
    for date in SESSIONS:
        european_expiries = (*EXPIRIES, EUROPEAN_ONLY_EXPIRY)
        american_expiries = EXPIRIES
        if date == ZERO_DTE_EXPIRY:
            european_expiries = (ZERO_DTE_EXPIRY, *european_expiries)
            american_expiries = (ZERO_DTE_EXPIRY, *american_expiries)
        european_definitions = [
            _definition_row("XSP", expiry, right, strike)
            for expiry in european_expiries
            for right in ("C", "P")
            for strike in STRIKES
        ]
        american_definitions = [
            _definition_row("SPY", expiry, right, strike)
            for expiry in american_expiries
            for right in ("C", "P")
            for strike in STRIKES
        ]
        write_table_atomically(
            partition_path(root, product_key="xsp-options", schema="definition",
                           date=date.isoformat()),
            european_definitions,
            OPTION_DEFINITION_SCHEMA,
        )
        write_table_atomically(
            partition_path(root, product_key="spy-options", schema="definition",
                           date=date.isoformat()),
            american_definitions,
            OPTION_DEFINITION_SCHEMA,
        )

        european_quotes: list[dict[str, Any]] = []
        american_quotes: list[dict[str, Any]] = []
        underlying_quotes: list[dict[str, Any]] = []
        for instant in snapshot_instants(date):
            underlying_quotes.append(
                _quote_row(_instrument_id("SPY", date, "C", 0.0), instant, SPOT_BID, SPOT_ASK)
            )
            for expiry in european_expiries:
                tau = year_fraction(instant, expiry)
                discount = discount_factor(tau)
                forward = european_forward(tau)
                for strike in STRIKES:
                    legs = _leg_quotes(discount * (forward - strike), HALF_WIDTH)
                    european_quotes.append(
                        _quote_row(_instrument_id("XSP", expiry, "C", strike), instant,
                                   legs[0], legs[1])
                    )
                    european_quotes.append(
                        _quote_row(_instrument_id("XSP", expiry, "P", strike), instant,
                                   legs[2], legs[3])
                    )
            for expiry in american_expiries:
                tau = year_fraction(instant, expiry)
                discount = discount_factor(tau)
                forward = american_forward(tau)
                for strike in STRIKES:
                    legs = _leg_quotes(discount * (forward - strike), HALF_WIDTH)
                    # Early exercise adds a premium to the in-the-money put,
                    # growing with the strike. It is added to the put alone, so
                    # the combination tilts and F_tilde acquires a slope.
                    premium = CONTAMINATION_PER_UNIT_STRIKE * max(0.0, strike - forward)
                    american_quotes.append(
                        _quote_row(_instrument_id("SPY", expiry, "C", strike), instant,
                                   legs[0], legs[1])
                    )
                    american_quotes.append(
                        _quote_row(_instrument_id("SPY", expiry, "P", strike), instant,
                                   legs[2] + premium, legs[3] + premium)
                    )
        write_table_atomically(
            partition_path(root, product_key="xsp-options", schema="cbbo-1m",
                           date=date.isoformat()),
            european_quotes,
            QUOTE_SCHEMA,
        )
        write_table_atomically(
            partition_path(root, product_key="spy-options", schema="cbbo-1m",
                           date=date.isoformat()),
            american_quotes,
            QUOTE_SCHEMA,
        )
        write_table_atomically(
            partition_path(root, product_key="spy-underlying", schema="bbo-1m",
                           date=date.isoformat()),
            underlying_quotes,
            QUOTE_SCHEMA,
        )
        write_table_atomically(
            partition_path(root, product_key="sr3-futures", schema="definition",
                           date=date.isoformat()),
            [
                {
                    "instrument_id": 27,
                    "raw_symbol": "SR3U6",
                    "publisher_id": 1,
                    "asset": "SR3",
                    "instrument_class": "F",
                    "is_outright": True,
                    "expiration_ns": dt.datetime(2026, 9, 15, tzinfo=dt.UTC),
                    "expiration_date": dt.date(2026, 9, 15),
                    "currency": "USD",
                    "exchange": "CME",
                    "min_price_increment": 0.0025,
                    "unit_of_measure": "USD",
                    "unit_of_measure_quantity": 2500.0,
                    "security_update_action": "A",
                }
            ],
            FUTURE_DEFINITION_SCHEMA,
        )
        write_table_atomically(
            partition_path(root, product_key="sr3-futures", schema="statistics",
                           date=date.isoformat()),
            [
                {
                    "ts_recv_ns": dt.datetime(2026, 6, 17, 20, 0, tzinfo=dt.UTC),
                    "ts_ref_ns": None,
                    "instrument_id": 27,
                    "publisher_id": 1,
                    "stat_type": "SETTLEMENT_PRICE",
                    "price": 96.25,
                    "quantity": None,
                    "update_action": "A",
                    "stat_flags": 0,
                    "is_outright": True,
                    "is_repeat_of_earlier_record": False,
                }
            ],
            STATISTIC_SCHEMA,
        )
    return root


def build_task_9a_outputs(base: Path) -> dict[str, Any]:
    """Write a synthetic 9A config, report, archive and processed tree.

    Returns the paths and digests the Task 9B configuration must pin.
    """
    archive_root = base / "archive"
    (archive_root / "manifests").mkdir(parents=True, exist_ok=True)
    # Probed declared sources, in the two states the real archive is in: an
    # empty directory for dividends and corporate actions, nothing at all for
    # borrow. Both must classify as unavailable and they are different facts.
    (archive_root / "state-street").mkdir(parents=True, exist_ok=True)
    (archive_root / "occ").mkdir(parents=True, exist_ok=True)

    manifest_path = archive_root / "manifests" / "sha256sums.txt"
    manifest_path.write_text("0" * 64 + "  raw/synthetic.dbn.zst\n", encoding="utf-8")
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()

    config_path = base / "feasibility.toml"
    config_path.write_text('schema_version = "market-feasibility/2"\n', encoding="utf-8")
    config_digest = hashlib.sha256(config_path.read_bytes()).hexdigest()

    processed_root = build_processed_tree(base / "processed")

    report = {
        "schema_version": "market-feasibility-report/2",
        "audit_version": "2.0.0",
        "config": {
            "source_sha256": config_digest,
            "processed_root": processed_root.relative_to(base).as_posix(),
        },
        "manifest": {"sha256": manifest_digest},
        "rates": {
            "assessment": "daily controls only",
            "date_alignment": "value date",
            "fred": {
                "assessment": "daily controls only",
                "date_alignment": "value date",
                "series": {
                    series_id: {
                        "role": "daily_treasury_control",
                        "session_date_values": {
                            date.isoformat(): "3.95" for date in SESSIONS
                        },
                    }
                    for series_id in ("SOFR", "DGS1MO", "DGS3MO", "DGS6MO", "DGS1")
                },
            },
        },
        "per_date": {
            date.isoformat(): {
                "products": {
                    "spy-underlying": {"feed_class": "partial_venue_consolidated"}
                }
            }
            for date in SESSIONS
        },
    }
    report_path = base / "feasibility-report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    report_digest = hashlib.sha256(report_path.read_bytes()).hexdigest()

    return {
        "base": base,
        "archive_root": archive_root.relative_to(base).as_posix(),
        "manifest_relative": "manifests/sha256sums.txt",
        "manifest_sha256": manifest_digest,
        "config_relative": config_path.relative_to(base).as_posix(),
        "config_sha256": config_digest,
        "report_relative": report_path.relative_to(base).as_posix(),
        "report_sha256": report_digest,
        "report_path": report_path,
        "processed_relative": processed_root.relative_to(base).as_posix(),
    }


def build_state_config_document(outputs: dict[str, Any]) -> dict[str, Any]:
    """Return a Task 9B configuration document pinned to the synthetic outputs."""
    import tomllib

    checked_in = Path(__file__).resolve().parents[3] / "configs/market_state_reconstruction_v1.toml"
    with checked_in.open("rb") as stream:
        document = tomllib.load(stream)

    document["provenance"].update(
        {
            "feasibility_config": outputs["config_relative"],
            "feasibility_config_sha256": outputs["config_sha256"],
            "feasibility_report": outputs["report_relative"],
            "feasibility_report_sha256": outputs["report_sha256"],
            "archive_root": outputs["archive_root"],
            "source_manifest": outputs["manifest_relative"],
            "source_manifest_sha256": outputs["manifest_sha256"],
            "processed_root": outputs["processed_relative"],
            "dividend_source_path": "state-street",
            "borrow_source_path": "borrow",
            "corporate_action_source_path": "occ",
        }
    )
    document["dates"] = [
        {"date": date.isoformat(), "role": "development"} for date in SESSIONS
    ]
    document["snapshots"]["times_local"] = [
        moment.isoformat(timespec="minutes") for moment in SNAPSHOTS_LOCAL
    ]
    return document
