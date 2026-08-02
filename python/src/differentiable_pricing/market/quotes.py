"""Normalize one-minute quote records and classify their quality and tradability.

A quote is normalized into explicit UTC nanoseconds plus decimal prices and
sizes. Nothing is repaired: a crossed book stays crossed, an absent side stays
absent, and the classification says which it was. Downstream filtering is a
research decision, not an ingestion one, so every decoded record is written out
with its classification attached rather than being dropped here.

Session membership is likewise a flag, not a filter. Out-of-window records are
real observations and their count is itself a feasibility statistic.

Two orthogonal judgements
    :class:`QuoteQuality` grades the *book*: is it two-sided, finite, ordered,
    changed. Tradability asks a different question — could this observation have
    been traded against — and it needs sizes, the vendor's own record flags, and
    a spread that is not absurd. Conflating them is how a two-sided book with no
    size on either side and a spread wider than its mid ends up counted as
    usable market data.

Staleness semantics
    ``ts_event`` is populated on some ``bbo-1m``/``cbbo-1m`` records and is the
    undefined sentinel on others, so the age of the top of book is observable
    for part of a partition and not for the rest. A time-based staleness measure
    would therefore be defined for some records and undefined for others in the
    same statistic, which is worse than not having one. Staleness here is
    exactly one thing: the graded price/size tuple is identical to this
    instrument's previous graded record. :data:`STALENESS_BASIS` states that in
    the report so no reader has to infer it, and
    ``records_with_event_timestamp`` counts how many records carried a real
    ``ts_event`` so the size of the gap is visible rather than assumed.
"""

from __future__ import annotations

import datetime as dt
import math
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from differentiable_pricing.market.dbn import (
    FLAG_BAD_TS_RECV,
    FLAG_MAYBE_BAD_BOOK,
    decode_price,
    decode_timestamp,
    describe_flags,
)
from differentiable_pricing.market.instruments import InstrumentKey

NANOSECONDS_PER_SECOND: Final = 1_000_000_000

STALENESS_BASIS: Final = (
    "value equality with this instrument's previous graded record. A time-based "
    "measure of top-of-book age is not used because ts_event is present on only "
    "some records of these one-minute schemas and is the undefined sentinel on the "
    "rest, so an age would exist for part of a partition and not for the rest; "
    "records_with_event_timestamp reports how many records carried one"
)
"""Exactly what 'stale' means here, stated in the report rather than inferred."""


class QuoteQuality(StrEnum):
    """Mutually exclusive quality verdicts for a normalized quote's book.

    The order below is the precedence order used by :func:`classify_quote`. It
    runs from structurally unusable to merely suspicious, so a record that is
    both non-finite and crossed is reported under the more fundamental defect.
    """

    NON_FINITE = "non_finite"
    """A decoded bid or ask price was NaN or infinite."""

    MISSING = "missing"
    """At least one side of the book has no price."""

    CROSSED = "crossed"
    """Bid strictly exceeds ask."""

    ZERO_BID = "zero_bid"
    """A two-sided quote whose bid is not strictly positive, so no mid is meaningful."""

    LOCKED = "locked"
    """Bid equals ask at a positive price."""

    STALE = "stale"
    """Prices and sizes identical to this instrument's previous graded minute."""

    OK = "ok"
    """Two-sided, finite, positive bid, strictly positive spread, and changed."""


QUALITY_ORDER: Final = tuple(QuoteQuality)
"""Precedence order, most fundamental defect first."""


@dataclass(frozen=True, slots=True)
class TradabilityPolicy:
    """The versioned predicate separating tradable observations from records.

    ``core`` tradability is the structural part: both sides priced and sized,
    a positive bid, a strictly positive spread, no vendor suspicion flag, and
    optionally not stale. The relative-spread cut is deliberately absent from
    the core predicate — several exploratory thresholds are reported instead of
    one being promoted to an acceptance gate after the results were seen.

    One requirement is implicit in the code and therefore stated explicitly
    here and in :meth:`describe`: the final clause requires the graded quality
    to be ``OK`` or ``STALE``. That is what excludes crossed, locked, zero-bid,
    one-sided and non-finite books, and it is a condition of the predicate even
    though the price and size tests above would not all reject them on their own.
    """

    require_positive_sizes: bool = True
    require_positive_bid: bool = True
    require_positive_spread: bool = True
    exclude_vendor_flagged: bool = True
    exclude_stale: bool = False

    def describe(self) -> dict[str, Any]:
        """Return a JSON-ready statement of the predicate in force."""
        return {
            "require_positive_sizes": self.require_positive_sizes,
            "require_positive_bid": self.require_positive_bid,
            "require_positive_spread": self.require_positive_spread,
            "exclude_vendor_flagged": self.exclude_vendor_flagged,
            "exclude_stale": self.exclude_stale,
            "required_quality_grades": [QuoteQuality.OK.value, QuoteQuality.STALE.value],
            "quality_grade_note": (
                "an observation is tradable only if its book graded 'ok' or 'stale'; this "
                "is what excludes crossed, locked, zero-bid, one-sided and non-finite books"
            ),
            "relative_spread_cut": (
                "none; exploratory thresholds are reported separately and no capability "
                "check consults any of them"
            ),
        }

    def is_tradable_core(
        self,
        *,
        bid_price: float | None,
        ask_price: float | None,
        bid_size: int | None,
        ask_size: int | None,
        quality: QuoteQuality,
        vendor_suspect: bool,
    ) -> bool:
        """Whether one graded observation could have been traded against."""
        if bid_price is None or ask_price is None:
            return False
        if not (math.isfinite(bid_price) and math.isfinite(ask_price)):
            return False
        if self.require_positive_bid and bid_price <= 0.0:
            return False
        if self.require_positive_spread and not ask_price > bid_price:
            return False
        if self.require_positive_sizes and not _positive_size(bid_size):
            return False
        if self.require_positive_sizes and not _positive_size(ask_size):
            return False
        if self.exclude_vendor_flagged and vendor_suspect:
            return False
        if self.exclude_stale and quality is QuoteQuality.STALE:
            return False
        return quality in (QuoteQuality.OK, QuoteQuality.STALE)


def _positive_size(size: int | None) -> bool:
    """Whether a decoded size is a finite, strictly positive quantity."""
    if size is None:
        return False
    if isinstance(size, float) and not math.isfinite(size):
        return False
    return size > 0


@dataclass(frozen=True, slots=True)
class NormalizedQuote:
    """One minute of top-of-book for one instrument, in explicit units."""

    ts_recv_ns: int
    ts_event_ns: int | None
    instrument_id: int
    publisher_id: int
    bid_price: float | None
    ask_price: float | None
    bid_size: int | None
    ask_size: int | None
    mid_price: float | None
    quoted_spread: float | None
    relative_spread: float | None
    bid_venue: int | None
    ask_venue: int | None
    in_regular_session: bool
    quality: QuoteQuality
    vendor_flags: int
    bad_ts_recv: bool
    maybe_bad_book: bool
    tradable_core: bool

    @property
    def key(self) -> InstrumentKey:
        """The publisher-scoped identity this quote belongs to."""
        return InstrumentKey(self.publisher_id, self.instrument_id)

    @property
    def is_two_sided(self) -> bool:
        """Whether both sides carry a price."""
        return self.bid_price is not None and self.ask_price is not None

    @property
    def vendor_suspect(self) -> bool:
        """Whether the vendor flagged the timestamp or the book itself."""
        return self.bad_ts_recv or self.maybe_bad_book


@dataclass(frozen=True, slots=True)
class SessionWindow:
    """The UTC half-open trading window and grid for one calendar session."""

    date: dt.date
    open_ns: int
    close_ns: int
    frequency_ns: int

    @classmethod
    def from_utc(
        cls,
        date: dt.date,
        opened: dt.datetime,
        closed: dt.datetime,
        *,
        frequency_seconds: int,
    ) -> SessionWindow:
        """Build a window from UTC datetimes and a grid frequency."""
        if opened.tzinfo is None or closed.tzinfo is None:
            raise ValueError("session window boundaries must be timezone-aware")
        return cls(
            date=date,
            open_ns=int(opened.timestamp()) * NANOSECONDS_PER_SECOND,
            close_ns=int(closed.timestamp()) * NANOSECONDS_PER_SECOND,
            frequency_ns=frequency_seconds * NANOSECONDS_PER_SECOND,
        )

    def contains(self, ts_ns: int) -> bool:
        """Whether an interval-close timestamp lies on this session's grid window.

        One-minute records are stamped at the close of the interval they cover,
        so the first in-session stamp is ``open + frequency`` and the last is
        ``close``. The window is therefore half-open at the open and closed at
        the close.
        """
        return self.open_ns < ts_ns <= self.close_ns

    def grid_index(self, ts_ns: int) -> int | None:
        """Return the zero-based grid slot of ``ts_ns``, or ``None`` if off-grid.

        A timestamp inside the window but not exactly on the grid is off-grid.
        Silently rounding it would manufacture alignment that the data does not
        have, which is precisely the claim this audit must not make.
        """
        if not self.contains(ts_ns):
            return None
        offset = ts_ns - self.open_ns
        if offset % self.frequency_ns != 0:
            return None
        return offset // self.frequency_ns - 1

    @property
    def slot_count(self) -> int:
        """Number of grid slots in the session."""
        return (self.close_ns - self.open_ns) // self.frequency_ns


def _finite(value: float | None) -> bool:
    return value is None or math.isfinite(value)


def classify_quote(
    *,
    bid_price: float | None,
    ask_price: float | None,
    bid_size: int | None,
    ask_size: int | None,
    previous: tuple[float | None, float | None, int | None, int | None] | None,
) -> QuoteQuality:
    """Grade one quote's book against the precedence order in :class:`QuoteQuality`.

    ``previous`` is the last graded ``(bid, ask, bid size, ask size)`` seen for
    the same instrument, or ``None`` when this is its first record. Staleness is
    defined against the previous *graded* record rather than the previous
    wall-clock minute so a gap in coverage is not silently counted as a change;
    see :data:`STALENESS_BASIS` for why no time-based alternative exists here.

    Sizes take part only in the staleness comparison. Whether a size makes the
    observation tradable is :class:`TradabilityPolicy`'s question, not this one's.
    """
    if not (_finite(bid_price) and _finite(ask_price)):
        return QuoteQuality.NON_FINITE
    if bid_price is None or ask_price is None:
        return QuoteQuality.MISSING
    if bid_price > ask_price:
        return QuoteQuality.CROSSED
    if bid_price <= 0.0:
        # Checked before the locked test so a two-sided book at zero is reported
        # as having no bid rather than as a locked market at zero.
        return QuoteQuality.ZERO_BID
    if bid_price == ask_price:
        return QuoteQuality.LOCKED
    if previous is not None and previous == (bid_price, ask_price, bid_size, ask_size):
        return QuoteQuality.STALE
    return QuoteQuality.OK


def normalize_quote(
    record: Any,
    *,
    window: SessionWindow,
    previous: tuple[float | None, float | None, int | None, int | None] | None,
    policy: TradabilityPolicy | None = None,
) -> NormalizedQuote:
    """Normalize one ``bbo-1m`` or ``cbbo-1m`` record.

    Both schemas expose a single book level. ``cbbo-1m`` additionally names the
    publisher behind each side; ``bbo-1m`` does not, and those fields stay
    ``None`` rather than being filled with the record's own publisher, which
    would assert a venue attribution the data never made.
    """
    tradability = policy or TradabilityPolicy()
    ts_recv_ns = decode_timestamp(int(record.ts_recv))
    if ts_recv_ns is None:
        raise ValueError("quote record has no receive timestamp")

    levels = record.levels
    if not levels:
        raise ValueError("quote record carries no book level")
    level = levels[0]

    bid_price = decode_price(int(level.bid_px))
    ask_price = decode_price(int(level.ask_px))
    bid_size = int(level.bid_sz)
    ask_size = int(level.ask_sz)
    bid_venue = _optional_publisher(level, "bid_pb")
    ask_venue = _optional_publisher(level, "ask_pb")
    flags = int(getattr(record, "flags", 0) or 0)
    bad_ts_recv = bool(flags & FLAG_BAD_TS_RECV)
    maybe_bad_book = bool(flags & FLAG_MAYBE_BAD_BOOK)

    quality = classify_quote(
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        previous=previous,
    )

    mid_price: float | None = None
    quoted_spread: float | None = None
    relative_spread: float | None = None
    if bid_price is not None and ask_price is not None and math.isfinite(bid_price + ask_price):
        mid_price = 0.5 * (bid_price + ask_price)
        quoted_spread = ask_price - bid_price
        # A mid of zero cannot normalize a spread. Leave it undefined instead of
        # emitting an infinity that would poison every downstream summary.
        if mid_price > 0.0:
            relative_spread = quoted_spread / mid_price

    return NormalizedQuote(
        ts_recv_ns=ts_recv_ns,
        ts_event_ns=decode_timestamp(_maybe_int(record, "ts_event")),
        instrument_id=int(record.instrument_id),
        publisher_id=int(record.publisher_id),
        bid_price=bid_price,
        ask_price=ask_price,
        bid_size=bid_size,
        ask_size=ask_size,
        mid_price=mid_price,
        quoted_spread=quoted_spread,
        relative_spread=relative_spread,
        bid_venue=bid_venue,
        ask_venue=ask_venue,
        in_regular_session=window.contains(ts_recv_ns),
        quality=quality,
        vendor_flags=flags,
        bad_ts_recv=bad_ts_recv,
        maybe_bad_book=maybe_bad_book,
        tradable_core=tradability.is_tradable_core(
            bid_price=bid_price,
            ask_price=ask_price,
            bid_size=bid_size,
            ask_size=ask_size,
            quality=quality,
            vendor_suspect=bad_ts_recv or maybe_bad_book,
        ),
    )


def _maybe_int(record: Any, attribute: str) -> int | None:
    value = getattr(record, attribute, None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_publisher(level: Any, attribute: str) -> int | None:
    """Return a consolidated book level's side publisher, if the schema has one."""
    value = getattr(level, attribute, None)
    if value is None:
        return None
    code = int(value)
    return code if code != 0 else None


def quote_state_key(
    quote: NormalizedQuote,
) -> tuple[float | None, float | None, int | None, int | None]:
    """Return the staleness comparison key of a normalized quote."""
    return (quote.bid_price, quote.ask_price, quote.bid_size, quote.ask_size)


def quote_rows(quotes: Iterable[NormalizedQuote]) -> Iterable[dict[str, Any]]:
    """Yield normalized quote rows in the order they were produced."""
    for quote in quotes:
        yield quote_row(quote)


def quote_row(quote: NormalizedQuote) -> dict[str, Any]:
    """Return the persisted row for one normalized quote."""
    return {
        "ts_recv_ns": quote.ts_recv_ns,
        "ts_event_ns": quote.ts_event_ns,
        "instrument_id": quote.instrument_id,
        "publisher_id": quote.publisher_id,
        "bid_price": quote.bid_price,
        "ask_price": quote.ask_price,
        "bid_size": quote.bid_size,
        "ask_size": quote.ask_size,
        "mid_price": quote.mid_price,
        "quoted_spread": quote.quoted_spread,
        "relative_spread": quote.relative_spread,
        "bid_venue": quote.bid_venue,
        "ask_venue": quote.ask_venue,
        "in_regular_session": quote.in_regular_session,
        "quality": quote.quality.value,
        "vendor_flags": quote.vendor_flags,
        "vendor_flag_names": ",".join(describe_flags(quote.vendor_flags)),
        "bad_ts_recv": quote.bad_ts_recv,
        "maybe_bad_book": quote.maybe_bad_book,
        "tradable_core": quote.tradable_core,
    }
