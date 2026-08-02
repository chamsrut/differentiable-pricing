"""Bounded-memory accumulators for the feasibility statistics.

Every statistic here is computed in one streaming pass and stores counters, not
records. That constraint shapes the design:

* Distributions are exact on a declared quantization grid rather than
  approximate over unbounded storage. A quoted spread is graded to the cent and
  a relative spread to the basis point, and both grids are stated in the report
  so a reader knows exactly what "median spread" means.
* Per-instrument state is a single small integer used as a 390-bit set, so a
  hundred thousand instruments cost megabytes, not gigabytes.

Nothing in this module fills a gap. Where an input is absent the accumulator
records the absence and the report states it as a finding.

Coverage is reported at three levels, and they are not interchangeable:

``session_slots_with_any_quote``
    Descriptive only. A minute counts if *any* instrument quoted in it. With a
    fourteen-thousand-contract chain this saturates at 100% and says nothing
    about whether the chain is usable. It is retained because it is a genuine
    liveness statistic, and it is never used as an acceptance gate.

``contract_minutes``
    The unit a study actually consumes: one (contract, minute) cell. The
    denominator is the *resolved universe*, not the set of contracts that
    happened to quote, so contracts that never quoted count against coverage.

``per-contract distributions``
    Because a mean over fourteen thousand contracts hides the wings, coverage
    and synchronization are reported as quantized distributions with tails.

Raw quality and tradability are likewise reported side by side rather than
merged, and the relative-spread cut is never a single number: several
exploratory thresholds are declared in configuration and every one of them is
reported. Those thresholds are descriptive and no capability check consults
them. That is a statement about the spread thresholds specifically; the
structural capability minima have their own declared provenance, which is not
the same claim.

Populations
    A coverage ratio is only meaningful when its numerator and denominator
    describe the same set of instruments. Products whose payload mixes
    instrument kinds — a futures file carrying outrights and calendar spreads —
    are therefore accumulated per population, and the headline block is the
    primary population alone.
"""

from __future__ import annotations

import datetime as dt
import itertools
import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

from differentiable_pricing.market.instruments import InstrumentKey, OptionContract
from differentiable_pricing.market.quotes import NormalizedQuote, SessionWindow

SPREAD_CENTS_GRID: Final = 0.01
"""Quantization of the absolute quoted-spread distribution, in price units."""

SPREAD_BASIS_POINT_GRID: Final = 1.0e-4
"""Quantization of the relative quoted-spread distribution."""

SPREAD_CENTS_CAP: Final = 100_000
"""Absolute spreads above this many cents ($1,000) fall into one overflow bin."""

SPREAD_BASIS_POINT_CAP: Final = 200_000
"""Relative spreads above this many basis points (2,000%) share one bin."""

FRACTION_GRID: Final = 0.01
"""Quantization of reported fraction distributions."""

REPORTED_QUANTILES: Final = (0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99)
"""Quantiles reported for every distribution, tails included as required."""

STATISTIC_DEDUP_KEY: Final = "(publisher, instrument, stat type, reference time, price, quantity)"
"""What counts as the same statistic republished."""


@dataclass(slots=True)
class QuantizedDistribution:
    """An exact distribution over a declared integer grid, with overflow bins.

    Exactness is on the grid: the reported quantiles are the grid values whose
    cumulative count first reaches the requested fraction. Values outside the
    cap are counted in *separate* low and high overflow bins, so a negative
    outlier is never reported as a high-side one, and quantiles that fall in
    overflow are reported as ``None`` rather than as a fabricated number.
    """

    grid: float
    cap: int
    counts: Counter[int] = field(default_factory=Counter)
    overflow_high: int = 0
    overflow_low: int = 0
    total: int = 0
    minimum: float | None = None
    maximum: float | None = None
    _sum: float = 0.0

    def add(self, value: float) -> None:
        """Record one observation."""
        if not math.isfinite(value):
            return
        self.total += 1
        self._sum += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        key = round(value / self.grid)
        if key > self.cap:
            self.overflow_high += 1
            return
        if key < -self.cap:
            self.overflow_low += 1
            return
        self.counts[key] += 1

    @property
    def overflow(self) -> int:
        """Observations outside the grid on either side."""
        return self.overflow_high + self.overflow_low

    @property
    def mean(self) -> float | None:
        """Arithmetic mean of every observation, ungridded."""
        return None if self.total == 0 else self._sum / self.total

    def quantile(self, fraction: float) -> float | None:
        """Return the grid value at ``fraction``, or ``None`` if it is in overflow."""
        if self.total == 0:
            return None
        target = fraction * self.total
        cumulative = self.overflow_low
        if cumulative >= target:
            return None
        for key in sorted(self.counts):
            cumulative += self.counts[key]
            if cumulative >= target:
                return key * self.grid
        return None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready summary including tails and the overflow counts."""
        return {
            "count": self.total,
            "grid": self.grid,
            "minimum": self.minimum,
            "maximum": self.maximum,
            "mean": self.mean,
            "overflow_above_grid_cap": self.overflow_high,
            "overflow_below_grid_cap": self.overflow_low,
            "quantiles": {
                f"p{round(fraction * 100):02d}": self.quantile(fraction)
                for fraction in REPORTED_QUANTILES
            },
        }


def _bitset_distribution(masks: Iterable[int], slot_count: int) -> QuantizedDistribution:
    """Return the distribution of per-instrument covered-minute counts."""
    distribution = QuantizedDistribution(grid=1.0, cap=max(slot_count, 1))
    for mask in masks:
        distribution.add(float(mask.bit_count()))
    return distribution


PRIMARY_POPULATION: Final = "primary"
"""The instrument population a product's headline coverage is measured over."""

MULTI_LEG_POPULATION: Final = "multi_leg"
"""Spreads and other multi-leg instruments, reported but never blended in."""

UNCLASSIFIED_POPULATION: Final = "unclassified"
"""Quoting instruments with no resolved definition. Orphans, counted and gated."""


@dataclass(slots=True)
class PopulationCoverage:
    """Coverage, spread and quality statistics for one instrument population.

    A population is a set of instruments whose prices mean the same thing. An
    outright future and a calendar spread do not: one is an index level and the
    other is a few points of carry. Keeping a separate accumulator per
    population is what makes a coverage ratio internally consistent, because the
    numerator (contract-minutes observed), the denominator (contracts in the
    resolved universe) and the spread distribution are then all drawn from the
    same set of instruments.
    """

    label: str
    slot_count: int
    relative_spread_thresholds: tuple[float, ...] = ()
    records: int = 0
    in_session_records: int = 0
    tradable_core_records: int = 0
    quality_counts: Counter[str] = field(default_factory=Counter)
    in_session_quality_counts: Counter[str] = field(default_factory=Counter)
    raw_slots: dict[InstrumentKey, int] = field(default_factory=dict)
    core_slots: dict[InstrumentKey, int] = field(default_factory=dict)
    threshold_slots: list[dict[InstrumentKey, int]] = field(default_factory=list)
    raw_contract_minutes: int = 0
    core_contract_minutes: int = 0
    threshold_contract_minutes: list[int] = field(default_factory=list)
    absolute_spread: QuantizedDistribution = field(
        default_factory=lambda: QuantizedDistribution(SPREAD_CENTS_GRID, SPREAD_CENTS_CAP)
    )
    relative_spread: QuantizedDistribution = field(
        default_factory=lambda: QuantizedDistribution(
            SPREAD_BASIS_POINT_GRID, SPREAD_BASIS_POINT_CAP
        )
    )

    def __post_init__(self) -> None:
        if not self.threshold_slots:
            self.threshold_slots = [{} for _ in self.relative_spread_thresholds]
        if not self.threshold_contract_minutes:
            self.threshold_contract_minutes = [0 for _ in self.relative_spread_thresholds]

    def observe_record(self, quote: NormalizedQuote) -> None:
        """Fold one record's quality into this population, session or not."""
        self.records += 1
        self.quality_counts[quote.quality.value] += 1
        if quote.tradable_core:
            self.tradable_core_records += 1

    def observe_on_grid(self, quote: NormalizedQuote, slot: int) -> None:
        """Fold one in-session, on-grid record into this population's coverage."""
        self.in_session_records += 1
        self.in_session_quality_counts[quote.quality.value] += 1
        key = quote.key
        bit = 1 << slot
        previous = self.raw_slots.get(key, 0)
        if not previous & bit:
            self.raw_slots[key] = previous | bit
            self.raw_contract_minutes += 1

        if not quote.tradable_core:
            return

        previous_core = self.core_slots.get(key, 0)
        if not previous_core & bit:
            self.core_slots[key] = previous_core | bit
            self.core_contract_minutes += 1
        # Spread statistics describe tradable two-sided books of this population
        # only. Blending a calendar spread's quarter-point book with an
        # outright's would make both distributions meaningless.
        if quote.quoted_spread is not None:
            self.absolute_spread.add(quote.quoted_spread)
        if quote.relative_spread is not None:
            self.relative_spread.add(quote.relative_spread)
            for index, threshold in enumerate(self.relative_spread_thresholds):
                if quote.relative_spread > threshold:
                    continue
                bucket = self.threshold_slots[index]
                seen = bucket.get(key, 0)
                if not seen & bit:
                    bucket[key] = seen | bit
                    self.threshold_contract_minutes[index] += 1

    def union_mask(self, *, tradable: bool = False) -> int:
        """Return the OR of every instrument's bitset in this population."""
        source = self.core_slots if tradable else self.raw_slots
        union = 0
        for mask in source.values():
            union |= mask
        return union

    def coverage_block(
        self, slots: Mapping[InstrumentKey, int], contract_minutes: int,
        universe_size: int | None
    ) -> dict[str, Any]:
        """Return one coverage block over a consistent numerator and denominator."""
        quoting = len(slots)
        denominator_contracts = universe_size if universe_size is not None else quoting
        expected = denominator_contracts * self.slot_count
        return {
            "population": self.label,
            "contracts_with_at_least_one_minute": quoting,
            "denominator_contracts": denominator_contracts,
            "denominator_basis": (
                f"resolved {self.label} universe"
                if universe_size is not None
                else f"{self.label} contracts that quoted"
            ),
            "contract_minutes_present": contract_minutes,
            "contract_minutes_expected": expected,
            "contract_minutes_missing": max(expected - contract_minutes, 0),
            "contract_minute_fill_ratio": (
                None if expected == 0 else contract_minutes / expected
            ),
            "per_contract_minutes": _bitset_distribution(
                slots.values(), self.slot_count
            ).summary(),
        }

    def summary(self, *, universe_size: int | None = None) -> dict[str, Any]:
        """Return this population's coverage, quality and spread statistics."""
        return {
            "population": self.label,
            "records": self.records,
            "in_session_on_grid_records": self.in_session_records,
            "tradable_core_records": self.tradable_core_records,
            "quality_counts": dict(sorted(self.quality_counts.items())),
            "in_session_quality_counts": dict(sorted(self.in_session_quality_counts.items())),
            "raw_quality_coverage": self.coverage_block(
                self.raw_slots, self.raw_contract_minutes, universe_size
            ),
            "tradable_coverage": self.coverage_block(
                self.core_slots, self.core_contract_minutes, universe_size
            ),
            "tradable_coverage_by_relative_spread_threshold": [
                {
                    "relative_spread_at_most": threshold,
                    **self.coverage_block(
                        self.threshold_slots[index],
                        self.threshold_contract_minutes[index],
                        universe_size,
                    ),
                }
                for index, threshold in enumerate(self.relative_spread_thresholds)
            ],
            "quoted_spread_absolute": self.absolute_spread.summary(),
            "quoted_spread_relative": self.relative_spread.summary(),
        }


@dataclass(slots=True)
class QuoteAudit:
    """Streaming quote-quality, coverage and spread statistics for one partition.

    Records are routed to a population by ``classifier`` before anything is
    accumulated, so a product whose payload mixes instrument kinds — a futures
    file carrying outrights and calendar spreads in the same minute — never
    reports a ratio whose numerator and denominator come from different sets.
    The headline block is the primary population; every other population is
    reported beside it, never inside it.

    Three parallel per-instrument bitsets are kept per population: every on-grid
    record, every tradable-core record, and one per exploratory relative-spread
    threshold. Each is one Python integer used as a ``slot_count``-bit set.
    """

    slot_count: int
    relative_spread_thresholds: tuple[float, ...] = ()
    classifier: Callable[[InstrumentKey], str] | None = None
    records: int = 0
    in_session_records: int = 0
    out_of_session_records: int = 0
    off_grid_records: int = 0
    quality_counts: Counter[str] = field(default_factory=Counter)
    in_session_quality_counts: Counter[str] = field(default_factory=Counter)
    zero_bid_size_records: int = 0
    zero_ask_size_records: int = 0
    vendor_flagged_records: int = 0
    bad_ts_recv_records: int = 0
    maybe_bad_book_records: int = 0
    records_with_event_timestamp: int = 0
    tradable_core_records: int = 0
    populations: dict[str, PopulationCoverage] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.population(PRIMARY_POPULATION)

    def population(self, label: str) -> PopulationCoverage:
        """Return (creating if needed) the accumulator for one population."""
        existing = self.populations.get(label)
        if existing is None:
            existing = PopulationCoverage(
                label=label,
                slot_count=self.slot_count,
                relative_spread_thresholds=self.relative_spread_thresholds,
            )
            self.populations[label] = existing
        return existing

    @property
    def primary(self) -> PopulationCoverage:
        """The population whose coverage is this product's headline statistic."""
        return self.population(PRIMARY_POPULATION)

    @property
    def raw_slots(self) -> dict[InstrumentKey, int]:
        """Primary-population raw bitsets."""
        return self.primary.raw_slots

    @property
    def core_slots(self) -> dict[InstrumentKey, int]:
        """Primary-population tradable bitsets."""
        return self.primary.core_slots

    def observe(self, quote: NormalizedQuote, window: SessionWindow) -> None:
        """Fold one normalized quote into the statistics."""
        self.records += 1
        self.quality_counts[quote.quality.value] += 1
        if quote.bid_size == 0:
            self.zero_bid_size_records += 1
        if quote.ask_size == 0:
            self.zero_ask_size_records += 1
        if quote.vendor_suspect:
            self.vendor_flagged_records += 1
        if quote.bad_ts_recv:
            self.bad_ts_recv_records += 1
        if quote.maybe_bad_book:
            self.maybe_bad_book_records += 1
        if quote.ts_event_ns is not None:
            self.records_with_event_timestamp += 1
        if quote.tradable_core:
            self.tradable_core_records += 1

        label = PRIMARY_POPULATION if self.classifier is None else self.classifier(quote.key)
        population = self.population(label)
        population.observe_record(quote)

        if not quote.in_regular_session:
            self.out_of_session_records += 1
            return

        self.in_session_records += 1
        self.in_session_quality_counts[quote.quality.value] += 1
        slot = window.grid_index(quote.ts_recv_ns)
        if slot is None:
            self.off_grid_records += 1
            return
        population.observe_on_grid(quote, slot)

    def instruments_in(self, label: str) -> int:
        """Number of instruments of ``label`` that quoted on the grid."""
        population = self.populations.get(label)
        return 0 if population is None else len(population.raw_slots)

    def occupied_slots(self, *, population: str | None = None) -> set[int]:
        """Return grid slots in which any instrument quoted (descriptive)."""
        union = 0
        for label, coverage in self.populations.items():
            if population is not None and label != population:
                continue
            union |= coverage.union_mask()
        return {slot for slot in range(self.slot_count) if union & (1 << slot)}

    def tradable_slots(self, *, population: str | None = None) -> set[int]:
        """Return grid slots in which any instrument was tradable (descriptive)."""
        union = 0
        for label, coverage in self.populations.items():
            if population is not None and label != population:
                continue
            union |= coverage.union_mask(tradable=True)
        return {slot for slot in range(self.slot_count) if union & (1 << slot)}

    def mask_for(
        self, key: InstrumentKey, *, tradable: bool = True, population: str | None = None
    ) -> int:
        """Return one instrument's bitset, tradable by default."""
        for label, coverage in self.populations.items():
            if population is not None and label != population:
                continue
            source = coverage.core_slots if tradable else coverage.raw_slots
            if key in source:
                return source[key]
        return 0

    def masks_for_universe(
        self, keys: Iterable[InstrumentKey], *, tradable: bool = True
    ) -> dict[InstrumentKey, int]:
        """Return one bitset per resolved contract, zero for those that never quoted.

        Contracts that never produced a tradable minute are part of the universe
        and must appear in every distribution drawn over it; dropping them would
        report the coverage of the contracts that happened to trade.
        """
        source = self.primary.core_slots if tradable else self.primary.raw_slots
        return {key: source.get(key, 0) for key in keys}

    def summary(self, *, universe_size: int | None = None) -> dict[str, Any]:
        """Return a JSON-ready summary of this partition.

        The headline coverage blocks are the primary population's. Every other
        population is reported under ``populations`` with its own consistent
        numerator, denominator and spread distribution.
        """
        primary = self.primary
        primary_summary = primary.summary(universe_size=universe_size)
        return {
            "records": self.records,
            "in_session_records": self.in_session_records,
            "out_of_session_records": self.out_of_session_records,
            "off_grid_in_session_records": self.off_grid_records,
            "session_slots": self.slot_count,
            "primary_population": PRIMARY_POPULATION,
            "population_note": (
                "coverage ratios are computed within one population, so numerator, "
                "denominator and spread distribution always describe the same set of "
                "instruments; other populations are reported separately and never blended"
            ),
            "descriptive_liveness": {
                "note": (
                    "a slot counts if ANY instrument quoted in it; with a large chain this "
                    "saturates and is never used as an acceptance gate"
                ),
                "session_slots_with_any_quote": len(self.occupied_slots()),
                "session_slots_with_any_tradable_quote": len(self.tradable_slots()),
                "session_slots_with_any_primary_quote": len(
                    self.occupied_slots(population=PRIMARY_POPULATION)
                ),
                "session_slots_with_any_primary_tradable_quote": len(
                    self.tradable_slots(population=PRIMARY_POPULATION)
                ),
            },
            "raw_quality_coverage": primary_summary["raw_quality_coverage"],
            "tradable_coverage": primary_summary["tradable_coverage"],
            "tradable_coverage_by_relative_spread_threshold": primary_summary[
                "tradable_coverage_by_relative_spread_threshold"
            ],
            "threshold_note": (
                "every declared exploratory threshold is reported for description only; "
                "none of them is consulted by any capability check"
            ),
            "quality_counts": dict(sorted(self.quality_counts.items())),
            "in_session_quality_counts": dict(sorted(self.in_session_quality_counts.items())),
            "tradable_core_records": self.tradable_core_records,
            "zero_bid_size_records": self.zero_bid_size_records,
            "zero_ask_size_records": self.zero_ask_size_records,
            "vendor_flagged_records": self.vendor_flagged_records,
            "bad_ts_recv_records": self.bad_ts_recv_records,
            "maybe_bad_book_records": self.maybe_bad_book_records,
            "records_with_event_timestamp": self.records_with_event_timestamp,
            "quoted_spread_absolute": primary_summary["quoted_spread_absolute"],
            "quoted_spread_relative": primary_summary["quoted_spread_relative"],
            "populations": {
                label: coverage.summary(
                    universe_size=universe_size if label == PRIMARY_POPULATION else None
                )
                for label, coverage in sorted(self.populations.items())
            },
        }


@dataclass(slots=True)
class SpotReference:
    """The spot anchor used for moneyness, with its provenance and uncertainty.

    ``basis`` is one of ``direct`` (the product's own declared underlying),
    ``proxy`` (a declared stand-in, with the caveat carried alongside), or
    ``unavailable``. There is no fourth case in which a number appears without
    saying which of the three it is.
    """

    value: float | None = None
    basis: str = "unavailable"
    source_product: str | None = None
    source_instrument: str | None = None
    selection_rule: str | None = None
    feed_class: str | None = None
    is_nbbo_grade: bool = False
    caveat: str | None = None
    uncertainty: dict[str, Any] | None = None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statement of the anchor and everything qualifying it."""
        return {
            "value": self.value,
            "basis": self.basis,
            "source_product": self.source_product,
            "source_instrument": self.source_instrument,
            "selection_rule": self.selection_rule,
            "feed_class": self.feed_class,
            "is_nbbo_grade": self.is_nbbo_grade,
            "caveat": self.caveat,
            "spot_reference_uncertainty": self.uncertainty,
        }


@dataclass(slots=True)
class OptionCoverageAudit:
    """Coverage of the resolved option chain that actually quoted.

    Moneyness bucketing happens only when the product has a declared spot
    reference. When the reference is a proxy for an underlying that was not
    acquired, every bucket is labelled as proxy-derived and the caveat travels
    with the numbers.

    A strike counts as *paired* only when its call and its put are tradable in
    the **same minute**. A call that is tradable in the morning and a put that is
    tradable in the afternoon can never be observed together, so counting them as
    a pair would assert a simultaneous two-sided observation that the archive
    does not contain. The intersection of the two bitsets is the whole test, and
    the number of common minutes per pair is reported as a distribution so the
    thinness of a pair is visible rather than binary.
    """

    session_date: dt.date
    moneyness_edges: tuple[float, ...]
    maturity_edges_days: tuple[int, ...]
    spot: SpotReference = field(default_factory=SpotReference)
    by_option_type: Counter[str] = field(default_factory=Counter)
    by_expiration: Counter[str] = field(default_factory=Counter)
    by_maturity_bucket: Counter[str] = field(default_factory=Counter)
    by_moneyness_bucket: Counter[str] = field(default_factory=Counter)
    strike_distribution: QuantizedDistribution = field(
        default_factory=lambda: QuantizedDistribution(grid=0.01, cap=10_000_000)
    )
    resolved_contracts: int = 0
    standard_contracts: int = 0
    non_standard_contracts: int = 0
    quoted_contracts: int = 0
    tradable_contracts: int = 0
    quoted_but_unresolved_instruments: int = 0
    paired_strikes_by_expiration: dict[str, int] = field(default_factory=dict)
    common_minutes: QuantizedDistribution = field(
        default_factory=lambda: QuantizedDistribution(grid=1.0, cap=100_000)
    )

    def observe_chain(
        self,
        contracts: Mapping[InstrumentKey, OptionContract],
        quoted_keys: Iterable[InstrumentKey],
        tradable_masks: Mapping[InstrumentKey, int] | None = None,
    ) -> None:
        """Fold the quoting subset of a resolved chain into the coverage counts.

        ``tradable_masks`` maps each contract to the bitset of minutes in which
        it was tradable. Pairing is the intersection of a call's and a put's
        bitsets, so a pair exists only where both were quotable at once.
        """
        masks: Mapping[InstrumentKey, int] = tradable_masks or {}
        self.resolved_contracts = len(contracts)
        self.standard_contracts = sum(1 for item in contracts.values() if item.is_standard_root)
        self.non_standard_contracts = self.resolved_contracts - self.standard_contracts
        per_expiration: dict[str, dict[float, dict[str, int]]] = {}
        for key in quoted_keys:
            contract = contracts.get(key)
            if contract is None:
                self.quoted_but_unresolved_instruments += 1
                continue
            self.quoted_contracts += 1
            mask = masks.get(key, 0)
            if mask:
                self.tradable_contracts += 1
            self.by_option_type[contract.option_type] += 1
            expiry = contract.expiration_date.isoformat()
            self.by_expiration[expiry] += 1
            if mask:
                strikes = per_expiration.setdefault(expiry, {})
                rights = strikes.setdefault(contract.strike, {})
                rights[contract.option_type] = rights.get(contract.option_type, 0) | mask
            self.strike_distribution.add(contract.strike)
            self.by_maturity_bucket[
                _bucket_label_int(
                    contract.time_to_maturity_days(self.session_date), self.maturity_edges_days
                )
            ] += 1
            if self.spot.value is not None and self.spot.value > 0.0:
                self.by_moneyness_bucket[
                    _bucket_label_float(contract.strike / self.spot.value, self.moneyness_edges)
                ] += 1

        paired: dict[str, int] = {}
        for expiry, strikes in per_expiration.items():
            count = 0
            for rights in strikes.values():
                overlap = rights.get("C", 0) & rights.get("P", 0)
                if overlap:
                    count += 1
                    self.common_minutes.add(float(overlap.bit_count()))
            paired[expiry] = count
        self.paired_strikes_by_expiration = paired

    def expiries_with_paired_strikes(self, minimum_strikes: int) -> int:
        """Count expiries carrying at least ``minimum_strikes`` same-minute pairs."""
        return sum(
            1 for count in self.paired_strikes_by_expiration.values() if count >= minimum_strikes
        )

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready coverage summary."""
        paired = sorted(self.paired_strikes_by_expiration.items())
        return {
            "resolved_contracts": self.resolved_contracts,
            "standard_root_contracts": self.standard_contracts,
            "non_standard_root_contracts": self.non_standard_contracts,
            "contracts_with_in_session_quotes": self.quoted_contracts,
            "contracts_with_tradable_quotes": self.tradable_contracts,
            "quoted_instruments_without_a_definition": self.quoted_but_unresolved_instruments,
            "spot_reference": self.spot.summary(),
            "moneyness_definition": (
                "strike / spot reference; no dividend, borrow or carry adjustment is "
                "applied, so this is spot moneyness and not forward moneyness. Forward "
                "moneyness requires an implied forward and is deferred to task 9B."
            ),
            "moneyness_is_proxy_derived": self.spot.basis == "proxy",
            "by_option_type": dict(sorted(self.by_option_type.items())),
            "distinct_expirations": len(self.by_expiration),
            "by_expiration": dict(sorted(self.by_expiration.items())),
            "paired_strike_definition": (
                "a strike is paired when its call and its put are both tradable in the "
                "same minute; the intersection of their tradable-minute bitsets must be "
                "non-empty"
            ),
            "tradable_paired_strikes_by_expiration": dict(paired),
            "expirations_with_any_tradable_pair": sum(1 for _, count in paired if count >= 1),
            "paired_strikes_total": sum(count for _, count in paired),
            "common_tradable_minutes_per_paired_strike": self.common_minutes.summary(),
            "by_time_to_maturity_days": _ordered_buckets(
                self.by_maturity_bucket, self.maturity_edges_days
            ),
            "by_spot_moneyness": _ordered_buckets(
                self.by_moneyness_bucket, self.moneyness_edges
            ),
            "strike_distribution": self.strike_distribution.summary(),
        }


@dataclass(slots=True)
class SynchronizationAudit:
    """Minute-by-minute alignment of an option chain against one reference instrument.

    The reference is a single named instrument chosen by a declared rule, never
    "whatever the decoder yielded first". Overlap is reported per contract, so a
    chain in which one contract is always aligned and ten thousand are never
    aligned cannot read as fully synchronized.

    ``option_masks`` covers the whole resolved universe, including contracts that
    never produced a tradable minute; those carry an empty bitset and count as
    zero synchronized minutes. Measuring only the contracts that traded would
    describe the alignment of the subset that was easy to align.
    """

    name: str
    option_product: str
    reference_product: str
    role: str
    slot_count: int
    reference_selection_rule: str | None = None
    reference_instrument: str | None = None
    reference_selection_detail: str | None = None
    reference_mask: int = 0
    option_masks: dict[InstrumentKey, int] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready synchronization summary."""
        reference_minutes = self.reference_mask.bit_count()
        union = 0
        overlap_counts = QuantizedDistribution(grid=1.0, cap=max(self.slot_count, 1))
        fraction_counts = QuantizedDistribution(grid=FRACTION_GRID, cap=200)
        contracts_fully_covered = 0
        contracts_without_overlap = 0
        contracts_without_own_minutes = 0
        for mask in self.option_masks.values():
            union |= mask
            overlap = (mask & self.reference_mask).bit_count()
            own = mask.bit_count()
            # Contracts with no tradable minute of their own are in the universe
            # and contribute a zero here; their *fraction* of own minutes is
            # undefined, so they are excluded from the fraction distribution and
            # counted explicitly instead.
            overlap_counts.add(float(overlap))
            if own:
                fraction_counts.add(overlap / own)
                if overlap == own:
                    contracts_fully_covered += 1
            else:
                contracts_without_own_minutes += 1
            if overlap == 0:
                contracts_without_overlap += 1
        union_minutes = union.bit_count()
        both_union = (union & self.reference_mask).bit_count()
        return {
            "name": self.name,
            "option_product": self.option_product,
            "reference_product": self.reference_product,
            "role": self.role,
            "session_slots": self.slot_count,
            "reference_selection_rule": self.reference_selection_rule,
            "reference_instrument": self.reference_instrument,
            "reference_selection_detail": self.reference_selection_detail,
            "reference_tradable_minutes": reference_minutes,
            "descriptive_liveness": {
                "note": (
                    "union over all option contracts; saturates on a large chain and is "
                    "never an acceptance gate"
                ),
                "option_slots_present": union_minutes,
                "slots_with_both": both_union,
                "option_slots_without_reference": (union & ~self.reference_mask).bit_count(),
                "reference_slots_without_option": (self.reference_mask & ~union).bit_count(),
            },
            "contracts_measured": len(self.option_masks),
            "contracts_measured_basis": (
                "the resolved universe, including contracts with no tradable minute"
            ),
            "contracts_with_no_tradable_minute": contracts_without_own_minutes,
            "contracts_with_no_synchronized_minute": contracts_without_overlap,
            "contracts_synchronized_in_every_own_minute": contracts_fully_covered,
            "per_contract_synchronized_minutes": overlap_counts.summary(),
            "per_contract_reference_overlap_fraction": fraction_counts.summary(),
            "overlap_fraction_denominator": (
                "contracts with at least one tradable minute of their own; contracts "
                "with none have an undefined fraction and are counted separately"
            ),
        }


@dataclass(slots=True)
class StatisticsAudit:
    """What a ``statistics`` schema actually delivered, by stat type.

    Republished statistics are counted separately from distinct ones. CME emits
    a settlement several times per session with the same reference time and
    price; treating those as three observations triples every count and hides
    whether the strip is actually complete.

    Outright and multi-leg instruments are tracked separately: a settlement
    strip built from calendar spreads is not a curve.
    """

    records: int = 0
    repeat_records: int = 0
    by_stat_type: Counter[str] = field(default_factory=Counter)
    distinct_by_stat_type: Counter[str] = field(default_factory=Counter)
    outright_distinct_by_stat_type: Counter[str] = field(default_factory=Counter)
    multi_leg_distinct_by_stat_type: Counter[str] = field(default_factory=Counter)
    unclassified_distinct_by_stat_type: Counter[str] = field(default_factory=Counter)
    delete_actions_by_stat_type: Counter[str] = field(default_factory=Counter)
    instruments_by_stat_type: dict[str, set[InstrumentKey]] = field(default_factory=dict)
    outright_instruments_by_stat_type: dict[str, set[InstrumentKey]] = field(default_factory=dict)
    priced_distinct_by_stat_type: Counter[str] = field(default_factory=Counter)
    reference_times_by_stat_type: dict[str, set[int | None]] = field(default_factory=dict)
    _seen: set[tuple[Any, ...]] = field(default_factory=set)

    def observe(
        self,
        *,
        stat_type: str,
        key: InstrumentKey,
        price: float | None,
        quantity: int | None,
        ts_ref_ns: int | None,
        update_action: str,
        is_outright: bool | None,
    ) -> bool:
        """Fold one statistics record in; return whether it repeats an earlier one."""
        self.records += 1
        self.by_stat_type[stat_type] += 1
        if update_action == "DELETE":
            self.delete_actions_by_stat_type[stat_type] += 1
        identity = (key, stat_type, ts_ref_ns, price, quantity, update_action)
        if identity in self._seen:
            self.repeat_records += 1
            return True
        self._seen.add(identity)
        self.distinct_by_stat_type[stat_type] += 1
        self.instruments_by_stat_type.setdefault(stat_type, set()).add(key)
        if price is not None:
            self.priced_distinct_by_stat_type[stat_type] += 1
        if is_outright is True:
            self.outright_distinct_by_stat_type[stat_type] += 1
            self.outright_instruments_by_stat_type.setdefault(stat_type, set()).add(key)
        elif is_outright is False:
            self.multi_leg_distinct_by_stat_type[stat_type] += 1
        else:
            self.unclassified_distinct_by_stat_type[stat_type] += 1
        self.reference_times_by_stat_type.setdefault(stat_type, set()).add(ts_ref_ns)
        return False

    def outright_instruments(self, stat_type: str) -> int:
        """Number of distinct outright instruments carrying ``stat_type``."""
        return len(self.outright_instruments_by_stat_type.get(stat_type, set()))

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statistics summary."""
        return {
            "records": self.records,
            "repeat_records": self.repeat_records,
            "deduplicated_by": STATISTIC_DEDUP_KEY,
            "by_stat_type": {
                stat_type: {
                    "records": count,
                    "distinct_records": self.distinct_by_stat_type.get(stat_type, 0),
                    "distinct_outright_records": self.outright_distinct_by_stat_type.get(
                        stat_type, 0
                    ),
                    "distinct_multi_leg_records": self.multi_leg_distinct_by_stat_type.get(
                        stat_type, 0
                    ),
                    "distinct_unclassified_records": (
                        self.unclassified_distinct_by_stat_type.get(stat_type, 0)
                    ),
                    "delete_actions": self.delete_actions_by_stat_type.get(stat_type, 0),
                    "instruments": len(self.instruments_by_stat_type.get(stat_type, set())),
                    "outright_instruments": self.outright_instruments(stat_type),
                    "distinct_records_with_a_price": self.priced_distinct_by_stat_type.get(
                        stat_type, 0
                    ),
                    "distinct_reference_times": len(
                        self.reference_times_by_stat_type.get(stat_type, set())
                    ),
                }
                for stat_type, count in sorted(self.by_stat_type.items())
            },
        }


def _bucket_label_float(value: float, edges: Sequence[float]) -> str:
    if value < edges[0]:
        return f"(-inf, {edges[0]:g})"
    for lower, upper in itertools.pairwise(edges):
        if lower <= value < upper:
            return f"[{lower:g}, {upper:g})"
    return f"[{edges[-1]:g}, inf)"


def _bucket_label_int(value: int, edges: Sequence[int]) -> str:
    if value < edges[0]:
        return f"(-inf, {edges[0]})"
    for lower, upper in itertools.pairwise(edges):
        if lower <= value < upper:
            return f"[{lower}, {upper})"
    return f"[{edges[-1]}, inf)"


def _ordered_buckets(counts: Counter[str], edges: Sequence[float | int]) -> dict[str, int]:
    """Return bucket counts in increasing bucket order, including empty buckets."""
    if all(isinstance(edge, int) for edge in edges):
        labels = [f"(-inf, {edges[0]})"]
        labels += [
            f"[{lower}, {upper})" for lower, upper in itertools.pairwise(edges)
        ]
        labels.append(f"[{edges[-1]}, inf)")
    else:
        labels = [f"(-inf, {edges[0]:g})"]
        labels += [
            f"[{lower:g}, {upper:g})" for lower, upper in itertools.pairwise(edges)
        ]
        labels.append(f"[{edges[-1]:g}, inf)")
    return {label: counts.get(label, 0) for label in labels}


def extrapolate_storage(
    *,
    observed_compressed_bytes: int,
    observed_processed_bytes: int,
    observed_dates: int,
    target_dates: int,
) -> dict[str, Any]:
    """Project archive and processed size to ``target_dates`` trading dates.

    The projection is deliberately naive and labelled as such: it scales the
    observed per-date mean linearly. Option chains grow with listings and quote
    volume varies with volatility, so this is an order-of-magnitude planning
    number and not a forecast.
    """
    if observed_dates <= 0:
        raise ValueError("observed_dates must be positive")
    compressed_per_date = observed_compressed_bytes / observed_dates
    processed_per_date = observed_processed_bytes / observed_dates
    return {
        "method": "linear scaling of the observed per-date mean",
        "caveat": (
            "Order-of-magnitude planning only. Per-date size varies with listed "
            "chain size, quote volume and volatility; it is not stationary."
        ),
        "observed_dates": observed_dates,
        "target_dates": target_dates,
        "observed_compressed_bytes": observed_compressed_bytes,
        "observed_processed_bytes": observed_processed_bytes,
        "compressed_bytes_per_date": compressed_per_date,
        "processed_bytes_per_date": processed_per_date,
        "projected_compressed_bytes": compressed_per_date * target_dates,
        "projected_processed_bytes": processed_per_date * target_dates,
        "projected_compressed_gib": compressed_per_date * target_dates / (1 << 30),
        "projected_processed_gib": processed_per_date * target_dates / (1 << 30),
    }
