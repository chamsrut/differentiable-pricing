"""Synthetic fixture builders for the market ingestion tests.

No test in this directory may read the private market archive. Everything here
is either a plain Python stand-in for a decoded record or a DBN file constructed
programmatically with the vendor encoder, so the whole suite runs in the
lightweight CI job and produces identical results on any machine.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FIXED_PRICE_SCALE = 1_000_000_000
UNDEF_PRICE = 2**63 - 1
UNDEF_TIMESTAMP = 2**64 - 1
UNDEF_INT32 = 2**31 - 1

FLAG_LAST = 128
FLAG_TOB = 64
FLAG_SNAPSHOT = 32
FLAG_BAD_TS_RECV = 8
FLAG_MAYBE_BAD_BOOK = 4


@dataclass
class FakeDefinition:
    """A duck-typed stand-in for a decoded ``InstrumentDefMsg``.

    The resolution code reads attributes, never the concrete vendor type, so a
    plain object exercises it end to end without the decoder installed.
    """

    instrument_id: int
    raw_symbol: str
    instrument_class: str
    publisher_id: int = 30
    strike_price: int = UNDEF_PRICE
    expiration: int = 0
    underlying: str = ""
    asset: str = ""
    currency: str = "USD"
    exchange: str = "OPRA"
    cfi: str = ""
    underlying_id: int = 0
    contract_multiplier: int = UNDEF_INT32
    unit_of_measure_qty: int = UNDEF_PRICE
    unit_of_measure: str = ""
    original_contract_size: int = UNDEF_INT32
    security_update_action: str = "A"
    security_type: str = "OPT"
    min_price_increment: int = UNDEF_PRICE
    ts_recv: int = 0


@dataclass
class FakeLevel:
    """One book level of a synthetic quote record."""

    bid_px: int
    ask_px: int
    bid_sz: int
    ask_sz: int
    bid_pb: int = 0
    ask_pb: int = 0


@dataclass
class FakeQuote:
    """A duck-typed stand-in for a decoded ``BBOMsg``/``CBBOMsg``."""

    ts_recv: int
    instrument_id: int
    publisher_id: int = 30
    levels: list[FakeLevel] = field(default_factory=list)
    flags: int = FLAG_LAST | FLAG_TOB
    ts_event: int = UNDEF_TIMESTAMP


def scaled(price: float) -> int:
    """Return a DBN fixed-point integer for a decimal price."""
    return round(price * FIXED_PRICE_SCALE)


def osi(root: str, expiry: dt.date, right: str, strike: float) -> str:
    """Build a 21-character OSI option symbol."""
    return (
        f"{root:<6}{expiry:%y%m%d}{right}{round(strike * 1000):08d}"
    )


def option_definition(
    *,
    instrument_id: int,
    root: str = "SPY",
    expiry: dt.date = dt.date(2026, 9, 18),
    right: str = "C",
    strike: float = 600.0,
    **overrides: Any,
) -> FakeDefinition:
    """Build a self-consistent synthetic option definition record."""
    expiration_ns = int(
        dt.datetime.combine(expiry, dt.time(20, 0), tzinfo=dt.UTC).timestamp()
    ) * FIXED_PRICE_SCALE
    record = FakeDefinition(
        instrument_id=instrument_id,
        raw_symbol=osi(root, expiry, right, strike),
        instrument_class=right,
        strike_price=scaled(strike),
        expiration=expiration_ns,
        underlying=root,
        asset=root,
    )
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


def future_definition(
    *,
    instrument_id: int,
    raw_symbol: str,
    instrument_class: str,
    expiry: dt.date | None = dt.date(2026, 9, 18),
    **overrides: Any,
) -> FakeDefinition:
    """Build a synthetic futures or futures-spread definition record."""
    expiration_ns = (
        0
        if expiry is None
        else int(
            dt.datetime.combine(expiry, dt.time(13, 30), tzinfo=dt.UTC).timestamp()
        )
        * FIXED_PRICE_SCALE
    )
    record = FakeDefinition(
        instrument_id=instrument_id,
        raw_symbol=raw_symbol,
        instrument_class=instrument_class,
        publisher_id=1,
        expiration=expiration_ns,
        asset="ES",
        exchange="XCME",
        security_type="FUT",
        min_price_increment=scaled(0.25),
        unit_of_measure="USD",
    )
    for name, value in overrides.items():
        setattr(record, name, value)
    return record


def quote(
    *,
    ts_recv_ns: int,
    instrument_id: int = 1,
    bid: float | None = 1.00,
    ask: float | None = 1.10,
    bid_size: int = 10,
    ask_size: int = 12,
    bid_venue: int = 0,
    ask_venue: int = 0,
    publisher_id: int = 30,
    flags: int = FLAG_LAST | FLAG_TOB,
    ts_event_ns: int = UNDEF_TIMESTAMP,
) -> FakeQuote:
    """Build a synthetic one-minute quote record."""
    return FakeQuote(
        ts_recv=ts_recv_ns,
        instrument_id=instrument_id,
        publisher_id=publisher_id,
        flags=flags,
        ts_event=ts_event_ns,
        levels=[
            FakeLevel(
                bid_px=UNDEF_PRICE if bid is None else scaled(bid),
                ask_px=UNDEF_PRICE if ask is None else scaled(ask),
                bid_sz=bid_size,
                ask_sz=ask_size,
                bid_pb=bid_venue,
                ask_pb=ask_venue,
            )
        ],
    )


def write_manifest(root: Path, relative_paths: list[str]) -> Path:
    """Write a valid ``sha256sums.txt`` covering ``relative_paths``."""
    manifest = root / "manifests" / "sha256sums.txt"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for relative in sorted(relative_paths):
        digest = hashlib.sha256((root / relative).read_bytes()).hexdigest()
        lines.append(f"{digest}  {relative}")
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def write_condition(directory: Path, dates: list[str], condition: str = "available") -> Path:
    """Write a vendor ``condition.json`` covering ``dates``."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "condition.json"
    path.write_text(
        json.dumps(
            [
                {"date": date, "condition": condition, "last_modified_date": date}
                for date in dates
            ],
            indent=4,
        ),
        encoding="utf-8",
    )
    return path


def build_synthetic_archive(root: Path) -> Path:
    """Create a tiny archive at ``root`` that verifies against its own manifest."""
    (root / "raw").mkdir(parents=True)
    (root / "raw" / "alpha.bin").write_bytes(b"alpha payload")
    (root / "raw" / "beta.bin").write_bytes(b"beta payload")
    write_manifest(root, ["raw/alpha.bin", "raw/beta.bin"])
    return root
