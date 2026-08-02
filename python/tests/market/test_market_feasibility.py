"""The streaming accumulators must be exact, bounded, and honest about denominators.

The defects these tests exist to catch are all forms of the same mistake:
reporting a number whose denominator flatters it. Coverage measured over
contracts that happened to quote, synchronization measured as a union over a
whole chain, and statistics counted with republications included all read as
better data than the archive holds.
"""

from __future__ import annotations

import datetime as dt

import pytest
from differentiable_pricing.market.feasibility import (
    MULTI_LEG_POPULATION,
    PRIMARY_POPULATION,
    UNCLASSIFIED_POPULATION,
    OptionCoverageAudit,
    QuantizedDistribution,
    QuoteAudit,
    SpotReference,
    StatisticsAudit,
    SynchronizationAudit,
    extrapolate_storage,
)
from differentiable_pricing.market.instruments import InstrumentKey, resolve_option_definition
from differentiable_pricing.market.quotes import (
    SessionWindow,
    TradabilityPolicy,
    normalize_quote,
    quote_state_key,
)
from market_fixtures import option_definition, quote


def _all_day(contracts) -> dict:
    """Every contract tradable in the same single minute."""
    return {key: 0b1 for key in contracts}

OPEN_UTC = dt.datetime(2026, 7, 30, 13, 30, tzinfo=dt.UTC)
CLOSE_UTC = dt.datetime(2026, 7, 30, 13, 40, tzinfo=dt.UTC)
MINUTE_NS = 60 * 1_000_000_000
SESSION = dt.date(2026, 7, 30)


@pytest.fixture
def window() -> SessionWindow:
    return SessionWindow.from_utc(SESSION, OPEN_UTC, CLOSE_UTC, frequency_seconds=60)


def at(minute: int) -> int:
    return int(OPEN_UTC.timestamp()) * 1_000_000_000 + minute * MINUTE_NS


def audit_with(window: SessionWindow, thresholds: tuple[float, ...] = ()) -> QuoteAudit:
    return QuoteAudit(slot_count=window.slot_count, relative_spread_thresholds=thresholds)


def observe(audit: QuoteAudit, window: SessionWindow, record, previous=None) -> None:
    audit.observe(
        normalize_quote(
            record, window=window, previous=previous, policy=TradabilityPolicy()
        ),
        window,
    )


class TestQuantizedDistribution:
    def test_quantiles_are_exact_on_the_grid(self) -> None:
        distribution = QuantizedDistribution(grid=0.01, cap=10_000)
        for value in [0.01] * 50 + [0.05] * 40 + [0.20] * 10:
            distribution.add(value)
        assert distribution.total == 100
        assert distribution.quantile(0.50) == pytest.approx(0.01)
        assert distribution.quantile(0.90) == pytest.approx(0.05)
        assert distribution.quantile(0.99) == pytest.approx(0.20)

    def test_extremes_are_ungridded(self) -> None:
        distribution = QuantizedDistribution(grid=0.01, cap=10_000)
        distribution.add(0.0123)
        distribution.add(0.0456)
        assert distribution.minimum == pytest.approx(0.0123)
        assert distribution.maximum == pytest.approx(0.0456)
        assert distribution.mean == pytest.approx((0.0123 + 0.0456) / 2)

    def test_high_and_low_overflow_are_counted_separately(self) -> None:
        distribution = QuantizedDistribution(grid=0.01, cap=100)
        distribution.add(0.50)
        distribution.add(500.0)
        distribution.add(-500.0)
        summary = distribution.summary()
        assert distribution.total == 3
        assert summary["overflow_above_grid_cap"] == 1
        assert summary["overflow_below_grid_cap"] == 1
        assert distribution.minimum == pytest.approx(-500.0)

    def test_a_low_outlier_does_not_shift_a_low_quantile_onto_the_grid(self) -> None:
        distribution = QuantizedDistribution(grid=0.01, cap=100)
        for _ in range(9):
            distribution.add(-500.0)
        distribution.add(0.5)
        assert distribution.quantile(0.5) is None
        assert distribution.quantile(0.99) == pytest.approx(0.5)

    def test_non_finite_values_are_ignored(self) -> None:
        distribution = QuantizedDistribution(grid=0.01, cap=100)
        distribution.add(float("nan"))
        distribution.add(float("inf"))
        assert distribution.total == 0
        assert distribution.mean is None

    def test_an_empty_distribution_reports_none_not_zero(self) -> None:
        summary = QuantizedDistribution(grid=0.01, cap=100).summary()
        assert summary["count"] == 0
        assert summary["mean"] is None
        assert summary["quantiles"]["p50"] is None

    def test_every_required_tail_is_reported(self) -> None:
        quantiles = QuantizedDistribution(grid=1.0, cap=10).summary()["quantiles"]
        assert set(quantiles) == {"p01", "p05", "p25", "p50", "p75", "p90", "p95", "p99"}


class TestContractMinuteCoverage:
    def test_the_denominator_is_the_resolved_universe_not_the_quoting_subset(
        self, window: SessionWindow
    ) -> None:
        """One contract quoting every minute must not read as full coverage."""
        audit = audit_with(window)
        previous = None
        for minute in range(1, 11):
            record = quote(ts_recv_ns=at(minute), instrument_id=1, ask=1.10 + minute)
            normalized = normalize_quote(record, window=window, previous=previous)
            previous = quote_state_key(normalized)
            audit.observe(normalized, window)

        optimistic = audit.summary()["tradable_coverage"]
        honest = audit.summary(universe_size=100)["tradable_coverage"]
        assert optimistic["contract_minute_fill_ratio"] == pytest.approx(1.0)
        assert honest["denominator_contracts"] == 100
        assert honest["contract_minutes_present"] == 10
        assert honest["contract_minutes_expected"] == 1000
        assert honest["contract_minute_fill_ratio"] == pytest.approx(0.01)
        assert honest["denominator_basis"] == "resolved primary universe"

    def test_descriptive_liveness_saturates_and_is_labelled_as_such(
        self, window: SessionWindow
    ) -> None:
        audit = audit_with(window)
        for minute in range(1, 11):
            observe(audit, window, quote(ts_recv_ns=at(minute), instrument_id=minute))
        summary = audit.summary(universe_size=1000)
        liveness = summary["descriptive_liveness"]
        assert liveness["session_slots_with_any_quote"] == 10
        assert "never used as an acceptance gate" in liveness["note"]
        assert summary["tradable_coverage"]["contract_minute_fill_ratio"] == pytest.approx(
            10 / 10_000
        )

    def test_per_contract_coverage_reports_the_wings_not_just_the_mean(
        self, window: SessionWindow
    ) -> None:
        audit = audit_with(window)
        # One contract quotes every minute; nine quote once.
        previous = None
        for minute in range(1, 11):
            record = quote(ts_recv_ns=at(minute), instrument_id=1, ask=1.10 + minute)
            normalized = normalize_quote(record, window=window, previous=previous)
            previous = quote_state_key(normalized)
            audit.observe(normalized, window)
        for index in range(2, 11):
            observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=index))
        distribution = audit.summary()["tradable_coverage"]["per_contract_minutes"]
        assert distribution["count"] == 10
        assert distribution["maximum"] == pytest.approx(10.0)
        assert distribution["minimum"] == pytest.approx(1.0)
        assert distribution["quantiles"]["p50"] == pytest.approx(1.0)

    def test_raw_and_tradable_coverage_are_reported_separately(
        self, window: SessionWindow
    ) -> None:
        audit = audit_with(window)
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=1))
        observe(audit, window, quote(ts_recv_ns=at(2), instrument_id=1, bid_size=0))
        summary = audit.summary(universe_size=1)
        assert summary["raw_quality_coverage"]["contract_minutes_present"] == 2
        assert summary["tradable_coverage"]["contract_minutes_present"] == 1

    def test_every_declared_threshold_is_reported(self, window: SessionWindow) -> None:
        audit = audit_with(window, thresholds=(0.02, 0.10, 0.50))
        # relative spread 0.10/1.05 = 9.5%: inside 10% and 50%, outside 2%.
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=1, bid=1.00, ask=1.10))
        buckets = audit.summary(universe_size=1)[
            "tradable_coverage_by_relative_spread_threshold"
        ]
        assert [item["relative_spread_at_most"] for item in buckets] == [0.02, 0.10, 0.50]
        assert [item["contract_minutes_present"] for item in buckets] == [0, 1, 1]

    def test_no_threshold_is_promoted_to_a_gate(self, window: SessionWindow) -> None:
        audit = audit_with(window, thresholds=(0.02,))
        summary = audit.summary()
        assert "consulted by any capability check" in summary["threshold_note"]

    def test_vendor_flagged_records_are_counted_and_excluded_from_tradable(
        self, window: SessionWindow
    ) -> None:
        from differentiable_pricing.market.dbn import FLAG_BAD_TS_RECV

        audit = audit_with(window)
        observe(
            audit,
            window,
            quote(ts_recv_ns=at(1), instrument_id=1, flags=128 | FLAG_BAD_TS_RECV),
        )
        summary = audit.summary(universe_size=1)
        assert summary["bad_ts_recv_records"] == 1
        assert summary["vendor_flagged_records"] == 1
        assert summary["tradable_coverage"]["contract_minutes_present"] == 0
        assert summary["raw_quality_coverage"]["contract_minutes_present"] == 1

    def test_out_of_session_records_are_separated(self, window: SessionWindow) -> None:
        audit = audit_with(window)
        for minute in (-5, 1, 50):
            observe(audit, window, quote(ts_recv_ns=at(minute)))
        summary = audit.summary()
        assert summary["records"] == 3
        assert summary["in_session_records"] == 1
        assert summary["out_of_session_records"] == 2

    def test_off_grid_records_are_counted_not_rounded(self, window: SessionWindow) -> None:
        audit = audit_with(window)
        observe(audit, window, quote(ts_recv_ns=at(2) + 1))
        summary = audit.summary()
        assert summary["in_session_records"] == 1
        assert summary["off_grid_in_session_records"] == 1
        assert summary["descriptive_liveness"]["session_slots_with_any_quote"] == 0

    def test_unusable_books_are_excluded_from_spread_statistics(
        self, window: SessionWindow
    ) -> None:
        audit = audit_with(window)
        cases = [
            (1, 1.00, 1.10, 10),  # tradable
            (2, 1.20, 1.10, 10),  # crossed
            (3, 1.10, 1.10, 10),  # locked
            (4, 0.00, 1.10, 10),  # zero bid
            (5, 1.00, 1.10, 0),  # no size on the bid
        ]
        for minute, bid, ask, size in cases:
            observe(
                audit,
                window,
                quote(ts_recv_ns=at(minute), instrument_id=minute, bid=bid, ask=ask,
                      bid_size=size),
            )
        summary = audit.summary()
        assert summary["quoted_spread_absolute"]["count"] == 1
        assert summary["in_session_quality_counts"] == {
            "crossed": 1,
            "locked": 1,
            "ok": 2,
            "zero_bid": 1,
        }
        assert summary["tradable_core_records"] == 1

    def test_zero_sizes_are_counted_separately(self, window: SessionWindow) -> None:
        audit = audit_with(window)
        observe(audit, window, quote(ts_recv_ns=at(1), bid_size=0, ask_size=0))
        summary = audit.summary()
        assert summary["zero_bid_size_records"] == 1
        assert summary["zero_ask_size_records"] == 1

    def test_masks_are_addressable_per_instrument(self, window: SessionWindow) -> None:
        audit = audit_with(window)
        observe(audit, window, quote(ts_recv_ns=at(3), instrument_id=7, publisher_id=30))
        assert audit.mask_for(InstrumentKey(30, 7)) == 1 << 2
        assert audit.mask_for(InstrumentKey(30, 8)) == 0


class TestPopulationConsistency:
    """A coverage ratio must draw numerator and denominator from one population."""

    def classifier(self, outrights: set[int], spreads: set[int]):
        def classify(key: InstrumentKey) -> str:
            if key.instrument_id in outrights:
                return PRIMARY_POPULATION
            if key.instrument_id in spreads:
                return MULTI_LEG_POPULATION
            return UNCLASSIFIED_POPULATION

        return classify

    def test_a_spread_only_minute_does_not_contribute_to_primary_coverage(
        self, window: SessionWindow
    ) -> None:
        """The reviewer's scenario: the spread quotes every minute, the outright does not."""
        audit = QuoteAudit(
            slot_count=window.slot_count,
            classifier=self.classifier(outrights={1}, spreads={2}),
        )
        previous_spread = None
        for minute in range(1, 11):
            spread = quote(ts_recv_ns=at(minute), instrument_id=2, bid=60.0,
                           ask=60.25 + 0.01 * minute)
            normalized = normalize_quote(spread, window=window, previous=previous_spread)
            previous_spread = quote_state_key(normalized)
            audit.observe(normalized, window)
        # the outright quotes in exactly two of the ten minutes
        for minute in (3, 7):
            observe(audit, window, quote(ts_recv_ns=at(minute), instrument_id=1,
                                         bid=6100.0, ask=6100.25))

        summary = audit.summary(universe_size=1)
        primary = summary["tradable_coverage"]
        assert primary["population"] == PRIMARY_POPULATION
        assert primary["contract_minutes_present"] == 2
        assert primary["contract_minutes_expected"] == 10
        assert primary["contract_minute_fill_ratio"] == pytest.approx(0.2)
        # The union metric still sees the spread, and says it is descriptive.
        assert summary["descriptive_liveness"]["session_slots_with_any_quote"] == 10
        assert summary["descriptive_liveness"]["session_slots_with_any_primary_quote"] == 2

    def test_the_spread_population_is_reported_beside_the_primary_one(
        self, window: SessionWindow
    ) -> None:
        audit = QuoteAudit(
            slot_count=window.slot_count,
            classifier=self.classifier(outrights={1}, spreads={2}),
        )
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=1, bid=6100.0,
                                     ask=6100.25))
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=2, bid=60.0, ask=60.25))
        populations = audit.summary(universe_size=1)["populations"]
        assert set(populations) == {PRIMARY_POPULATION, MULTI_LEG_POPULATION}
        assert populations[MULTI_LEG_POPULATION]["tradable_coverage"][
            "contract_minutes_present"
        ] == 1
        assert populations[MULTI_LEG_POPULATION]["tradable_coverage"][
            "denominator_basis"
        ].startswith("multi_leg")

    def test_spread_distributions_are_not_blended_across_populations(
        self, window: SessionWindow
    ) -> None:
        """An outright's quarter-point book and a spread's must not share a quantile."""
        audit = QuoteAudit(
            slot_count=window.slot_count,
            classifier=self.classifier(outrights={1}, spreads={2}),
        )
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=1, bid=6100.0,
                                     ask=6100.25))
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=2, bid=60.0, ask=63.0))
        summary = audit.summary(universe_size=1)
        assert summary["quoted_spread_absolute"]["count"] == 1
        assert summary["quoted_spread_absolute"]["maximum"] == pytest.approx(0.25)
        multi = summary["populations"][MULTI_LEG_POPULATION]["quoted_spread_absolute"]
        assert multi["maximum"] == pytest.approx(3.0)

    def test_an_instrument_without_a_definition_is_unclassified_not_primary(
        self, window: SessionWindow
    ) -> None:
        audit = QuoteAudit(
            slot_count=window.slot_count,
            classifier=self.classifier(outrights={1}, spreads=set()),
        )
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=99))
        assert audit.instruments_in(UNCLASSIFIED_POPULATION) == 1
        assert audit.summary(universe_size=1)["tradable_coverage"][
            "contract_minutes_present"
        ] == 0

    def test_masks_for_universe_includes_contracts_that_never_quoted(
        self, window: SessionWindow
    ) -> None:
        audit = audit_with(window)
        observe(audit, window, quote(ts_recv_ns=at(1), instrument_id=1))
        masks = audit.masks_for_universe([InstrumentKey(30, 1), InstrumentKey(30, 2)])
        assert masks[InstrumentKey(30, 1)] == 0b1
        assert masks[InstrumentKey(30, 2)] == 0


class TestOptionCoverage:
    def spot(self) -> SpotReference:
        return SpotReference(value=600.0, basis="direct", source_product="underlying")

    def test_buckets_are_labelled_and_complete(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.0, 1.1),
            maturity_edges_days=(0, 30, 365),
            spot=self.spot(),
        )
        contracts = {
            InstrumentKey(30, index): resolve_option_definition(
                option_definition(instrument_id=index, strike=strike, right=right)
            )
            for index, (strike, right) in enumerate(
                [(540.0, "C"), (600.0, "P"), (660.0, "C")], start=1
            )
        }
        audit.observe_chain(contracts, list(contracts), _all_day(contracts))
        summary = audit.summary()
        assert summary["contracts_with_in_session_quotes"] == 3
        assert summary["contracts_with_tradable_quotes"] == 3
        assert summary["by_option_type"] == {"C": 2, "P": 1}
        assert summary["by_spot_moneyness"]["[0.9, 1)"] == 1
        assert summary["by_spot_moneyness"]["[1, 1.1)"] == 1
        assert summary["by_spot_moneyness"]["[1.1, inf)"] == 1
        assert summary["by_time_to_maturity_days"]["[30, 365)"] == 3

    def test_paired_strikes_need_a_call_and_a_put_that_are_both_tradable(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=self.spot(),
        )
        call = resolve_option_definition(option_definition(instrument_id=1, strike=600.0,
                                                          right="C"))
        put = resolve_option_definition(option_definition(instrument_id=2, strike=600.0,
                                                         right="P"))
        lonely = resolve_option_definition(option_definition(instrument_id=3, strike=610.0,
                                                            right="C"))
        contracts = {contract.key: contract for contract in (call, put, lonely)}
        audit.observe_chain(contracts, list(contracts), _all_day(contracts))
        summary = audit.summary()
        assert summary["tradable_paired_strikes_by_expiration"] == {"2026-09-18": 1}
        assert audit.expiries_with_paired_strikes(1) == 1
        assert audit.expiries_with_paired_strikes(2) == 0

    def test_a_quoted_but_untradable_pair_does_not_count(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=self.spot(),
        )
        call = resolve_option_definition(option_definition(instrument_id=1, strike=600.0,
                                                          right="C"))
        put = resolve_option_definition(option_definition(instrument_id=2, strike=600.0,
                                                         right="P"))
        contracts = {contract.key: contract for contract in (call, put)}
        audit.observe_chain(contracts, list(contracts), {call.key: 0b1})
        assert audit.expiries_with_paired_strikes(1) == 0

    def test_a_quoted_instrument_without_a_definition_is_counted(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=self.spot(),
        )
        audit.observe_chain({}, [InstrumentKey(30, 42)])
        summary = audit.summary()
        assert summary["quoted_instruments_without_a_definition"] == 1
        assert summary["contracts_with_in_session_quotes"] == 0

    def test_without_a_spot_reference_no_moneyness_is_invented(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=SpotReference(basis="unavailable", caveat="not acquired"),
        )
        contract = resolve_option_definition(option_definition(instrument_id=1))
        audit.observe_chain({contract.key: contract}, [contract.key])
        summary = audit.summary()
        assert summary["spot_reference"]["value"] is None
        assert summary["spot_reference"]["basis"] == "unavailable"
        assert sum(summary["by_spot_moneyness"].values()) == 0

    def test_a_proxy_anchor_is_labelled_on_every_derived_bucket(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=SpotReference(
                value=600.0,
                basis="proxy",
                source_product="spy-underlying",
                caveat="the index level was not acquired",
            ),
        )
        contract = resolve_option_definition(option_definition(instrument_id=1))
        audit.observe_chain({contract.key: contract}, [contract.key])
        summary = audit.summary()
        assert summary["moneyness_is_proxy_derived"] is True
        assert summary["spot_reference"]["basis"] == "proxy"
        assert summary["spot_reference"]["caveat"] == "the index level was not acquired"

    def test_standard_and_non_standard_roots_are_counted_separately(self) -> None:
        audit = OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=self.spot(),
        )
        standard = resolve_option_definition(
            option_definition(instrument_id=1, root="SPY"), product_root="SPY"
        )
        adjusted = resolve_option_definition(
            option_definition(instrument_id=2, root="SPY1"), product_root="SPY"
        )
        contracts = {item.key: item for item in (standard, adjusted)}
        audit.observe_chain(contracts, list(contracts))
        summary = audit.summary()
        assert summary["standard_root_contracts"] == 1
        assert summary["non_standard_root_contracts"] == 1


class TestSameMinutePairedStrikes:
    """A pair must be observable at once, not merely present on the same day."""

    def chain(self):
        call = resolve_option_definition(
            option_definition(instrument_id=1, strike=600.0, right="C")
        )
        put = resolve_option_definition(
            option_definition(instrument_id=2, strike=600.0, right="P")
        )
        return call, put, {call.key: call, put.key: put}

    def audit(self) -> OptionCoverageAudit:
        return OptionCoverageAudit(
            session_date=SESSION,
            moneyness_edges=(0.9, 1.1),
            maturity_edges_days=(0, 365),
            spot=SpotReference(value=600.0, basis="direct"),
        )

    def test_disjoint_minutes_yield_no_paired_strike(self) -> None:
        call, put, contracts = self.chain()
        audit = self.audit()
        # The call is tradable in minutes 0-2, the put in minutes 3-5. They are
        # never quotable together, so no simultaneous pair exists.
        audit.observe_chain(
            contracts, list(contracts), {call.key: 0b000111, put.key: 0b111000}
        )
        assert audit.paired_strikes_by_expiration == {"2026-09-18": 0}
        assert audit.expiries_with_paired_strikes(1) == 0
        assert audit.summary()["paired_strikes_total"] == 0

    def test_overlapping_minutes_yield_one_paired_strike(self) -> None:
        call, put, contracts = self.chain()
        audit = self.audit()
        audit.observe_chain(
            contracts, list(contracts), {call.key: 0b001111, put.key: 0b111100}
        )
        assert audit.paired_strikes_by_expiration == {"2026-09-18": 1}
        assert audit.expiries_with_paired_strikes(1) == 1
        summary = audit.summary()
        assert summary["paired_strikes_total"] == 1
        # 0b001100 is the intersection: two common minutes.
        assert summary["common_tradable_minutes_per_paired_strike"]["count"] == 1
        assert summary["common_tradable_minutes_per_paired_strike"][
            "minimum"
        ] == pytest.approx(2.0)

    def test_a_single_shared_minute_is_enough(self) -> None:
        call, put, contracts = self.chain()
        audit = self.audit()
        audit.observe_chain(contracts, list(contracts), {call.key: 0b0110, put.key: 0b0100})
        assert audit.expiries_with_paired_strikes(1) == 1
        assert audit.summary()["common_tradable_minutes_per_paired_strike"][
            "maximum"
        ] == pytest.approx(1.0)

    def test_a_put_that_is_never_tradable_yields_no_pair(self) -> None:
        call, put, contracts = self.chain()
        audit = self.audit()
        audit.observe_chain(contracts, list(contracts), {call.key: 0b1111, put.key: 0})
        assert audit.expiries_with_paired_strikes(1) == 0

    def test_the_pairing_rule_is_stated_in_the_summary(self) -> None:
        call, _put, contracts = self.chain()
        audit = self.audit()
        audit.observe_chain(contracts, list(contracts), {call.key: 0b1})
        assert "same minute" in audit.summary()["paired_strike_definition"]


class TestSynchronization:
    def test_per_contract_overlap_is_reported_not_just_the_union(self) -> None:
        """One always-aligned contract must not make a chain look synchronized."""
        aligned = InstrumentKey(30, 1)
        never = InstrumentKey(30, 2)
        audit = SynchronizationAudit(
            name="pair",
            option_product="options",
            reference_product="underlying",
            role="primary",
            slot_count=4,
            reference_selection_rule="single_instrument",
            reference_instrument="SPY",
            reference_mask=0b1111,
            option_masks={aligned: 0b1111, never: 0b0000},
        )
        summary = audit.summary()
        assert summary["descriptive_liveness"]["slots_with_both"] == 4
        assert summary["contracts_measured"] == 2
        assert summary["contracts_with_no_synchronized_minute"] == 1
        assert summary["contracts_synchronized_in_every_own_minute"] == 1
        assert summary["per_contract_synchronized_minutes"]["minimum"] == pytest.approx(0.0)
        assert summary["per_contract_synchronized_minutes"]["maximum"] == pytest.approx(4.0)

    def test_a_reference_that_never_quotes_yields_no_overlap(self) -> None:
        audit = SynchronizationAudit(
            name="pair",
            option_product="options",
            reference_product="underlying",
            role="primary",
            slot_count=4,
            reference_mask=0,
            option_masks={InstrumentKey(30, 1): 0b1111},
        )
        summary = audit.summary()
        assert summary["reference_tradable_minutes"] == 0
        assert summary["contracts_with_no_synchronized_minute"] == 1
        assert summary["per_contract_reference_overlap_fraction"]["maximum"] == pytest.approx(
            0.0
        )

    def test_contracts_with_no_tradable_minute_are_in_the_distribution(self) -> None:
        """A contract that never traded is part of the universe and counts as zero."""
        audit = SynchronizationAudit(
            name="pair",
            option_product="options",
            reference_product="underlying",
            role="primary",
            slot_count=4,
            reference_mask=0b1111,
            option_masks={
                InstrumentKey(30, 1): 0b1111,
                InstrumentKey(30, 2): 0,
                InstrumentKey(30, 3): 0,
            },
        )
        summary = audit.summary()
        assert summary["contracts_measured"] == 3
        assert summary["contracts_with_no_tradable_minute"] == 2
        assert summary["per_contract_synchronized_minutes"]["count"] == 3
        assert summary["per_contract_synchronized_minutes"]["minimum"] == pytest.approx(0.0)
        # The fraction of a contract's own minutes is undefined when it has none.
        assert summary["per_contract_reference_overlap_fraction"]["count"] == 1
        assert "undefined fraction" in summary["overlap_fraction_denominator"]

    def test_the_reference_selection_rule_is_reported(self) -> None:
        audit = SynchronizationAudit(
            name="pair",
            option_product="options",
            reference_product="futures",
            role="control",
            slot_count=4,
            reference_selection_rule="nearest_unexpired_outright",
            reference_instrument="ESM6",
            reference_selection_detail="earliest expiry among 3 unexpired outrights",
            reference_mask=0b0011,
            option_masks={InstrumentKey(30, 1): 0b0111},
        )
        summary = audit.summary()
        assert summary["reference_selection_rule"] == "nearest_unexpired_outright"
        assert summary["reference_instrument"] == "ESM6"
        assert "earliest expiry" in summary["reference_selection_detail"]
        assert summary["per_contract_synchronized_minutes"]["maximum"] == pytest.approx(2.0)


class TestStatisticsAudit:
    def key(self, instrument: int) -> InstrumentKey:
        return InstrumentKey(1, instrument)

    def observe(self, audit: StatisticsAudit, **overrides) -> bool:
        payload = {
            "stat_type": "SETTLEMENT_PRICE",
            "key": self.key(1),
            "price": 96.5,
            "quantity": None,
            "ts_ref_ns": 1000,
            "update_action": "NEW",
            "is_outright": True,
        }
        payload.update(overrides)
        return audit.observe(**payload)

    def test_a_republished_settlement_is_counted_once(self) -> None:
        audit = StatisticsAudit()
        assert self.observe(audit) is False
        assert self.observe(audit) is True
        assert self.observe(audit) is True
        summary = audit.summary()
        assert summary["records"] == 3
        assert summary["repeat_records"] == 2
        assert summary["by_stat_type"]["SETTLEMENT_PRICE"]["records"] == 3
        assert summary["by_stat_type"]["SETTLEMENT_PRICE"]["distinct_records"] == 1

    def test_a_different_reference_time_is_a_distinct_observation(self) -> None:
        audit = StatisticsAudit()
        self.observe(audit, ts_ref_ns=1000)
        self.observe(audit, ts_ref_ns=2000)
        assert audit.summary()["by_stat_type"]["SETTLEMENT_PRICE"]["distinct_records"] == 2

    def test_a_revised_price_is_a_distinct_observation(self) -> None:
        audit = StatisticsAudit()
        self.observe(audit, price=96.5)
        self.observe(audit, price=96.6)
        assert audit.summary()["by_stat_type"]["SETTLEMENT_PRICE"]["distinct_records"] == 2

    def test_outright_and_multi_leg_statistics_are_split(self) -> None:
        audit = StatisticsAudit()
        self.observe(audit, key=self.key(1), is_outright=True)
        self.observe(audit, key=self.key(2), is_outright=False)
        self.observe(audit, key=self.key(3), is_outright=None)
        block = audit.summary()["by_stat_type"]["SETTLEMENT_PRICE"]
        assert block["distinct_outright_records"] == 1
        assert block["distinct_multi_leg_records"] == 1
        assert block["distinct_unclassified_records"] == 1
        assert block["outright_instruments"] == 1
        assert audit.outright_instruments("SETTLEMENT_PRICE") == 1

    def test_delete_actions_are_counted(self) -> None:
        audit = StatisticsAudit()
        self.observe(audit, update_action="NEW")
        self.observe(audit, update_action="DELETE")
        block = audit.summary()["by_stat_type"]["SETTLEMENT_PRICE"]
        assert block["delete_actions"] == 1
        assert block["distinct_records"] == 2

    def test_priced_and_unpriced_statistics_are_distinguished(self) -> None:
        audit = StatisticsAudit()
        self.observe(audit, stat_type="OPEN_INTEREST", price=None, quantity=1234)
        block = audit.summary()["by_stat_type"]["OPEN_INTEREST"]
        assert block["distinct_records_with_a_price"] == 0
        assert block["distinct_reference_times"] == 1


class TestExtrapolation:
    def test_scaling_is_linear_and_labelled(self) -> None:
        projection = extrapolate_storage(
            observed_compressed_bytes=300,
            observed_processed_bytes=600,
            observed_dates=3,
            target_dates=200,
        )
        assert projection["compressed_bytes_per_date"] == pytest.approx(100.0)
        assert projection["projected_compressed_bytes"] == pytest.approx(20_000.0)
        assert projection["projected_processed_bytes"] == pytest.approx(40_000.0)
        assert "not stationary" in projection["caveat"]

    def test_zero_observed_dates_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="observed_dates must be positive"):
            extrapolate_storage(
                observed_compressed_bytes=1,
                observed_processed_bytes=1,
                observed_dates=0,
                target_dates=200,
            )
