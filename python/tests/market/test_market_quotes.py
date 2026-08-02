"""Session filtering, quality grading and tradability must be exact and separate.

Quality grades the book. Tradability asks whether the observation could have
been traded against, which needs sizes and the vendor's own record flags. The
two are asserted independently here so neither can quietly absorb the other.
"""

from __future__ import annotations

import datetime as dt

import pytest
from differentiable_pricing.market.dbn import FLAG_BAD_TS_RECV, FLAG_MAYBE_BAD_BOOK
from differentiable_pricing.market.instruments import InstrumentKey
from differentiable_pricing.market.quotes import (
    STALENESS_BASIS,
    QuoteQuality,
    SessionWindow,
    TradabilityPolicy,
    classify_quote,
    normalize_quote,
    quote_row,
    quote_state_key,
)
from market_fixtures import FLAG_LAST, FLAG_TOB, FakeLevel, FakeQuote, quote

SESSION_DATE = dt.date(2026, 7, 30)
OPEN_UTC = dt.datetime(2026, 7, 30, 13, 30, tzinfo=dt.UTC)
CLOSE_UTC = dt.datetime(2026, 7, 30, 20, 0, tzinfo=dt.UTC)
MINUTE_NS = 60 * 1_000_000_000


@pytest.fixture
def window() -> SessionWindow:
    return SessionWindow.from_utc(SESSION_DATE, OPEN_UTC, CLOSE_UTC, frequency_seconds=60)


def at(minutes_after_open: int) -> int:
    """Return the epoch-nanosecond stamp ``minutes_after_open`` past the open."""
    return int(OPEN_UTC.timestamp()) * 1_000_000_000 + minutes_after_open * MINUTE_NS


class TestSessionWindow:
    def test_grid_has_one_slot_per_regular_minute(self, window: SessionWindow) -> None:
        assert window.slot_count == 390

    def test_interval_close_labelling_excludes_the_open_and_includes_the_close(
        self, window: SessionWindow
    ) -> None:
        assert window.contains(at(0)) is False
        assert window.contains(at(1)) is True
        assert window.contains(at(390)) is True
        assert window.contains(at(391)) is False

    def test_slot_indices_run_from_zero_to_the_last_minute(
        self, window: SessionWindow
    ) -> None:
        assert window.grid_index(at(1)) == 0
        assert window.grid_index(at(390)) == 389
        assert window.grid_index(at(0)) is None
        assert window.grid_index(at(500)) is None

    def test_an_off_grid_stamp_is_not_rounded_onto_the_grid(
        self, window: SessionWindow
    ) -> None:
        assert window.contains(at(5) + 1) is True
        assert window.grid_index(at(5) + 1) is None

    def test_pre_and_post_market_stamps_are_outside(self, window: SessionWindow) -> None:
        assert window.contains(at(-30)) is False
        assert window.contains(at(400)) is False

    def test_boundaries_are_exact_integers_not_float_products(
        self, window: SessionWindow
    ) -> None:
        assert window.open_ns == int(OPEN_UTC.timestamp()) * 1_000_000_000
        assert window.open_ns % 1_000_000_000 == 0
        assert (window.close_ns - window.open_ns) % window.frequency_ns == 0

    def test_naive_boundaries_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            SessionWindow.from_utc(
                SESSION_DATE,
                dt.datetime(2026, 7, 30, 13, 30),
                CLOSE_UTC,
                frequency_seconds=60,
            )


class TestQuoteQuality:
    @pytest.mark.parametrize(
        ("bid", "ask", "expected"),
        [
            (1.00, 1.10, QuoteQuality.OK),
            (1.20, 1.10, QuoteQuality.CROSSED),
            (1.10, 1.10, QuoteQuality.LOCKED),
            (0.00, 1.10, QuoteQuality.ZERO_BID),
            (None, 1.10, QuoteQuality.MISSING),
            (1.00, None, QuoteQuality.MISSING),
            (None, None, QuoteQuality.MISSING),
        ],
    )
    def test_grades_each_defect(
        self, bid: float | None, ask: float | None, expected: QuoteQuality
    ) -> None:
        assert (
            classify_quote(
                bid_price=bid, ask_price=ask, bid_size=1, ask_size=1, previous=None
            )
            is expected
        )

    def test_a_two_sided_book_at_zero_is_zero_bid_not_locked(self) -> None:
        """Both sides at zero is an absent market, not a locked one."""
        assert (
            classify_quote(
                bid_price=0.0, ask_price=0.0, bid_size=1, ask_size=1, previous=None
            )
            is QuoteQuality.ZERO_BID
        )

    def test_non_finite_outranks_every_other_defect(self) -> None:
        assert (
            classify_quote(
                bid_price=float("nan"), ask_price=1.1, bid_size=1, ask_size=1, previous=None
            )
            is QuoteQuality.NON_FINITE
        )
        assert (
            classify_quote(
                bid_price=float("inf"),
                ask_price=1.0,
                bid_size=1,
                ask_size=1,
                previous=None,
            )
            is QuoteQuality.NON_FINITE
        )

    def test_an_unchanged_quote_is_stale(self) -> None:
        previous = (1.00, 1.10, 10, 12)
        assert (
            classify_quote(
                bid_price=1.00, ask_price=1.10, bid_size=10, ask_size=12, previous=previous
            )
            is QuoteQuality.STALE
        )

    def test_a_size_change_alone_is_not_stale(self) -> None:
        previous = (1.00, 1.10, 10, 12)
        assert (
            classify_quote(
                bid_price=1.00, ask_price=1.10, bid_size=11, ask_size=12, previous=previous
            )
            is QuoteQuality.OK
        )

    def test_the_first_record_of_an_instrument_is_never_stale(self) -> None:
        assert (
            classify_quote(
                bid_price=1.00, ask_price=1.10, bid_size=10, ask_size=12, previous=None
            )
            is QuoteQuality.OK
        )

    def test_the_staleness_basis_is_stated_rather_than_left_to_inference(self) -> None:
        assert "value equality" in STALENESS_BASIS
        # ts_event is populated on some records and undefined on others, so the
        # basis must say why no time-based measure is used rather than claiming
        # the field is always absent.
        assert "only" in STALENESS_BASIS
        assert "records_with_event_timestamp" in STALENESS_BASIS


class TestTradability:
    def policy(self, **overrides: bool) -> TradabilityPolicy:
        return TradabilityPolicy(**overrides)

    def test_a_normal_two_sided_book_is_tradable(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1)), window=window, previous=None, policy=self.policy()
        )
        assert normalized.tradable_core is True

    @pytest.mark.parametrize(("bid_size", "ask_size"), [(0, 12), (10, 0), (0, 0)])
    def test_a_side_with_no_size_is_not_tradable(
        self, window: SessionWindow, bid_size: int, ask_size: int
    ) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), bid_size=bid_size, ask_size=ask_size),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        assert normalized.quality is QuoteQuality.OK
        assert normalized.tradable_core is False

    def test_size_requirements_can_be_switched_off_by_configuration(
        self, window: SessionWindow
    ) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), bid_size=0),
            window=window,
            previous=None,
            policy=self.policy(require_positive_sizes=False),
        )
        assert normalized.tradable_core is True

    def test_a_zero_bid_is_not_tradable(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), bid=0.0, ask=1.10),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        assert normalized.tradable_core is False

    def test_a_crossed_or_locked_book_is_not_tradable(self, window: SessionWindow) -> None:
        crossed = normalize_quote(
            quote(ts_recv_ns=at(1), bid=1.20, ask=1.10),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        locked = normalize_quote(
            quote(ts_recv_ns=at(2), bid=1.10, ask=1.10),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        assert crossed.tradable_core is False
        assert locked.tradable_core is False

    def test_a_one_sided_book_is_not_tradable(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), bid=None),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        assert normalized.tradable_core is False

    def test_a_vendor_flagged_record_is_not_tradable(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), flags=FLAG_LAST | FLAG_BAD_TS_RECV),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        assert normalized.bad_ts_recv is True
        assert normalized.vendor_suspect is True
        assert normalized.tradable_core is False

    def test_a_maybe_bad_book_is_not_tradable(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), flags=FLAG_LAST | FLAG_MAYBE_BAD_BOOK),
            window=window,
            previous=None,
            policy=self.policy(),
        )
        assert normalized.maybe_bad_book is True
        assert normalized.tradable_core is False

    def test_stale_is_tradable_unless_configuration_says_otherwise(
        self, window: SessionWindow
    ) -> None:
        first = normalize_quote(
            quote(ts_recv_ns=at(1)), window=window, previous=None, policy=self.policy()
        )
        state = quote_state_key(first)
        kept = normalize_quote(
            quote(ts_recv_ns=at(2)), window=window, previous=state, policy=self.policy()
        )
        dropped = normalize_quote(
            quote(ts_recv_ns=at(2)),
            window=window,
            previous=state,
            policy=self.policy(exclude_stale=True),
        )
        assert kept.quality is QuoteQuality.STALE
        assert kept.tradable_core is True
        assert dropped.tradable_core is False

    def test_the_policy_states_that_no_spread_cut_is_applied(self) -> None:
        described = self.policy().describe()
        assert "no capability check consults" in described["relative_spread_cut"]
        assert described["require_positive_sizes"] is True

    def test_the_policy_discloses_the_implicit_quality_requirement(self) -> None:
        """The OK/STALE clause is a condition of the predicate, so it must be stated."""
        described = self.policy().describe()
        assert described["required_quality_grades"] == ["ok", "stale"]
        assert "crossed" in described["quality_grade_note"]


class TestNormalization:
    def test_derived_fields_are_exact(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(10), bid=1.00, ask=1.10), window=window, previous=None
        )
        assert normalized.mid_price == pytest.approx(1.05)
        assert normalized.quoted_spread == pytest.approx(0.10)
        assert normalized.relative_spread == pytest.approx(0.10 / 1.05)
        assert normalized.in_regular_session is True
        assert normalized.quality is QuoteQuality.OK

    def test_identity_carries_the_publisher(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), instrument_id=7, publisher_id=95),
            window=window,
            previous=None,
        )
        assert normalized.key == InstrumentKey(95, 7)

    def test_a_one_sided_book_has_no_mid_or_spread(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(10), bid=None), window=window, previous=None
        )
        assert normalized.mid_price is None
        assert normalized.quoted_spread is None
        assert normalized.relative_spread is None
        assert normalized.is_two_sided is False
        assert normalized.quality is QuoteQuality.MISSING

    def test_a_zero_mid_leaves_the_relative_spread_undefined(
        self, window: SessionWindow
    ) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(10), bid=0.0, ask=0.0), window=window, previous=None
        )
        assert normalized.quoted_spread == pytest.approx(0.0)
        assert normalized.relative_spread is None

    def test_out_of_session_records_are_flagged_not_dropped(
        self, window: SessionWindow
    ) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(-15)), window=window, previous=None
        )
        assert normalized.in_regular_session is False
        assert normalized.quality is QuoteQuality.OK

    def test_consolidated_venue_attribution_is_carried(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), bid_venue=24, ask_venue=23),
            window=window,
            previous=None,
        )
        assert normalized.bid_venue == 24
        assert normalized.ask_venue == 23

    def test_a_plain_bbo_gets_no_invented_venue(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), publisher_id=95), window=window, previous=None
        )
        assert normalized.bid_venue is None
        assert normalized.ask_venue is None
        assert normalized.publisher_id == 95

    def test_an_undefined_event_timestamp_becomes_none(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1)), window=window, previous=None
        )
        assert normalized.ts_event_ns is None

    def test_a_real_event_timestamp_is_carried(self, window: SessionWindow) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), ts_event_ns=at(1) - 5), window=window, previous=None
        )
        assert normalized.ts_event_ns == at(1) - 5

    def test_a_record_without_a_book_level_is_rejected(self, window: SessionWindow) -> None:
        with pytest.raises(ValueError, match="no book level"):
            normalize_quote(
                FakeQuote(ts_recv=at(1), instrument_id=1, levels=[]),
                window=window,
                previous=None,
            )

    def test_a_record_without_a_timestamp_is_rejected(self, window: SessionWindow) -> None:
        broken = FakeQuote(
            ts_recv=2**64 - 1,
            instrument_id=1,
            levels=[FakeLevel(bid_px=1, ask_px=2, bid_sz=1, ask_sz=1)],
        )
        with pytest.raises(ValueError, match="no receive timestamp"):
            normalize_quote(broken, window=window, previous=None)

    def test_state_key_round_trips_into_the_stale_check(
        self, window: SessionWindow
    ) -> None:
        first = normalize_quote(quote(ts_recv_ns=at(1)), window=window, previous=None)
        second = normalize_quote(
            quote(ts_recv_ns=at(2)), window=window, previous=quote_state_key(first)
        )
        assert second.quality is QuoteQuality.STALE


class TestPersistedRow:
    def test_the_row_carries_the_vendor_flags_and_tradability(
        self, window: SessionWindow
    ) -> None:
        normalized = normalize_quote(
            quote(ts_recv_ns=at(1), flags=FLAG_LAST | FLAG_TOB | FLAG_BAD_TS_RECV),
            window=window,
            previous=None,
        )
        row = quote_row(normalized)
        assert row["vendor_flags"] == FLAG_LAST | FLAG_TOB | FLAG_BAD_TS_RECV
        assert "bad_ts_recv" in row["vendor_flag_names"]
        assert row["bad_ts_recv"] is True
        assert row["tradable_core"] is False
        assert row["quality"] == "ok"
