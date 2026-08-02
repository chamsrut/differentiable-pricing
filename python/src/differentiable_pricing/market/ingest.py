"""Deterministic, memory-bounded ingestion and feasibility audit of a raw archive.

This is the entry point for the whole task. It verifies the archive against its
manifest, consults the vendor's own per-date condition statements, decodes every
declared product and schema in one streaming pass per file, writes normalized
Parquet atomically under an ignored processed-data root, and emits a feasibility
report.

It answers whether the acquired data supports a credible American-option and
volatility-surface study. It does not price, fit a surface, compute Greeks or
train anything, and it never substitutes a missing market input with a plausible
default.

What this audit deliberately does not do
    It does not fit put-call parity, infer a forward, infer a dividend strip, or
    build forward moneyness. Those are task 9B. The archive contains no
    independent dividend ground truth, so nothing here may describe a dividend
    as observed or inferred; the report says only what was and was not acquired.

Scoped capabilities instead of one verdict
    A single pass/fail verdict over an archive this heterogeneous is not
    informative: archive integrity, ingestion, a European surface study, an
    American calibration and a replication protocol have different inputs and
    fail for different reasons. Each is reported separately, and every one of
    them is a rule applied to recorded checks rather than a constant.

Usage
    python -m differentiable_pricing.market.ingest \
        --config configs/market_feasibility_v1.toml
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from differentiable_pricing.market.config import (
    CAPABILITIES,
    QUOTE_SCHEMAS,
    ConfigError,
    MarketFeasibilityConfig,
    ProductSpec,
    load_market_config,
)
from differentiable_pricing.market.dbn import (
    DbnError,
    DbnFileSummary,
    decode_price,
    decode_quantity,
    decode_timestamp,
    open_dbn,
    require_dbn_module,
)
from differentiable_pricing.market.feasibility import (
    MULTI_LEG_POPULATION,
    PRIMARY_POPULATION,
    UNCLASSIFIED_POPULATION,
    QuantizedDistribution,
    QuoteAudit,
    SpotReference,
    StatisticsAudit,
    SynchronizationAudit,
    extrapolate_storage,
)
from differentiable_pricing.market.instruments import (
    REFERENCE_SELECTION_SINGLE,
    FutureContract,
    FutureUniverse,
    InstrumentKey,
    OptionUniverse,
    build_future_universe,
    build_option_universe,
    future_definition_rows,
    option_definition_rows,
)
from differentiable_pricing.market.manifest import ManifestError, require_verified_archive
from differentiable_pricing.market.provenance import (
    CONDITION_AVAILABLE,
    ConditionError,
    SourceObservation,
    VendorCondition,
    probe_declared_source,
    read_vendor_conditions,
)
from differentiable_pricing.market.quotes import (
    STALENESS_BASIS,
    NormalizedQuote,
    SessionWindow,
    TradabilityPolicy,
    normalize_quote,
    quote_row,
    quote_state_key,
)
from differentiable_pricing.market.writer import (
    FUTURE_DEFINITION_SCHEMA,
    OPTION_DEFINITION_SCHEMA,
    QUOTE_SCHEMA,
    STATISTIC_SCHEMA,
    AtomicParquetWriter,
    WriterError,
    partition_path,
)

REPORT_SCHEMA_VERSION: Final = "market-feasibility-report/2"
PROJECT_ROOT: Final = Path(__file__).resolve().parents[4]

_DATE_IN_FILENAME: Final = re.compile(r"-(\d{8})\.")
_SEVERITY_ORDER: Final = {"blocking": 0, "warning": 1, "informational": 2}

DEFERRED_WORK: Final = (
    {
        "task": "9B",
        "item": "XSP put-call parity fitting and implied discount/forward extraction",
        "status": "deferred",
        "note": (
            "No parity fit, forward, discount factor or dividend quantity is produced by "
            "this audit."
        ),
    },
    {
        "task": "9B",
        "item": "SPY discrete dividend inference",
        "status": "deferred",
        "note": (
            "The archive carries no independent dividend ground truth, so this audit "
            "records only whether a dividend source was acquired. Nothing here asserts "
            "that SPY dividends are observed or inferred."
        ),
    },
    {
        "task": "9B",
        "item": "futures reference roll policy near expiry",
        "status": "deferred",
        "note": (
            "The declared rule picks the nearest unexpired outright, which on a session "
            "one day before expiry selects a contract whose remaining life is a single "
            "day and whose basis behaviour is atypical. Choosing between an expiry-buffer "
            "roll, an open-interest roll and a volume roll is a curve-construction "
            "decision that belongs with the forward work, not with this audit; the rule "
            "in force and the instrument it selected are reported per date so the effect "
            "is visible."
        ),
    },
    {
        "task": "9B",
        "item": "forward moneyness coordinates",
        "status": "deferred",
        "note": (
            "Moneyness reported here is spot moneyness against a declared reference, "
            "labelled direct, proxy or unavailable."
        ),
    },
)
"""Work this audit explicitly does not do, recorded so no reader infers it did."""

DEFERRED_LOW_FINDINGS: Final = (
    {
        "finding": "L7 parquet row-buffer memory dominates the read chunk",
        "status": "documented_not_changed",
        "why": (
            "The batch size is a declared configuration knob and the writer docstring now "
            "states that the row buffer, not the read chunk, is the dominant memory term. "
            "Lowering it trades throughput for memory and is a tuning decision, not a defect."
        ),
    },
    {
        "finding": "L9 no trading-calendar awareness (early closes, holidays)",
        "status": "deferred",
        "why": (
            "All three declared sessions are full trading days, verified by observing 390 "
            "on-grid minutes each. A calendar dependency is a new third-party input whose "
            "reproducibility cost is not justified until the date set grows; the session "
            "window remains explicitly declared per configuration."
        ),
    },
    {
        "finding": "spot anchor is one minute's mid rather than a session-wide estimator",
        "status": "deferred",
        "why": (
            "The anchor exists to bucket moneyness descriptively, and its quoted-spread "
            "distribution is reported as its uncertainty. A smoothed or VWAP anchor would "
            "be a modelling choice, and forward moneyness supersedes the question in "
            "task 9B."
        ),
    },
    {
        "finding": "option coverage counts the whole chain, not a liquid subset",
        "status": "deferred",
        "why": (
            "Selecting a liquid subset requires a liquidity threshold, and this audit "
            "deliberately reports several exploratory thresholds rather than adopting "
            "one. The per-threshold coverage blocks let a reader apply their own."
        ),
    },
    {
        "finding": "durability fsync after os.replace",
        "status": "deferred",
        "why": (
            "os.replace makes publication atomic against a crashed run, which is the "
            "property the audit needs. Surviving a power loss additionally requires "
            "fsync on the file and its directory; the outputs are reproducible by re-running, "
            "so the cost is not justified."
        ),
    },
)
"""Low-severity findings deliberately not changed, with the reason."""


class IngestError(RuntimeError):
    """Raised when the audit cannot be completed."""


def _stat_name(value: Any) -> str:
    return str(getattr(value, "name", value))


@dataclass(frozen=True, slots=True)
class InputFile:
    """One decoded-eligible DBN payload located in the archive."""

    product_key: str
    schema: str
    date: dt.date
    path: Path
    relative_path: str
    request_id: str
    request_directory: Path
    compressed_bytes: int


@dataclass(slots=True)
class ReferenceSelection:
    """The single instrument chosen to represent a reference product."""

    product_key: str
    rule: str | None = None
    key: InstrumentKey | None = None
    symbol: str | None = None
    detail: str = "no reference selection rule declared"
    mid: float | None = None
    mid_slot: int | None = None
    spread: QuantizedDistribution | None = None

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statement of the rule and its outcome."""
        return {
            "product": self.product_key,
            "selection_rule": self.rule,
            "selected_instrument": self.symbol,
            "selected_instrument_id": None if self.key is None else self.key.instrument_id,
            "selected_publisher_id": None if self.key is None else self.key.publisher_id,
            "detail": self.detail,
            "reference_mid": self.mid,
            "reference_mid_slot": self.mid_slot,
            "reference_spread_distribution": (
                None if self.spread is None else self.spread.summary()
            ),
        }


def discover_inputs(
    config: MarketFeasibilityConfig, archive_root: Path
) -> tuple[tuple[InputFile, ...], list[dict[str, Any]]]:
    """Locate every declared product/schema/date payload, reporting what is absent.

    Only the calendar dates declared in the configuration are ingested. Files for
    other dates in the same request are counted and skipped rather than silently
    folded into a session's statistics.
    """
    declared_dates = {entry.date for entry in config.dates}
    found: list[InputFile] = []
    findings: list[dict[str, Any]] = []
    seen: set[tuple[str, str, dt.date]] = set()

    for product in config.products:
        product_root = archive_root / product.archive_subpath
        if not product_root.is_dir():
            findings.append(
                {
                    "severity": "blocking",
                    "kind": "product_directory_missing",
                    "product": product.key,
                    "detail": f"'{product.archive_subpath}' is not present in the archive",
                }
            )
            continue
        for payload in sorted(product_root.rglob("*.dbn.zst")) + sorted(
            product_root.rglob("*.dbn")
        ):
            relative = payload.relative_to(product_root).parts
            if len(relative) != 4:
                findings.append(
                    {
                        "severity": "warning",
                        "kind": "unexpected_archive_layout",
                        "product": product.key,
                        "detail": f"'{payload.relative_to(archive_root).as_posix()}'",
                    }
                )
                continue
            schema, _date_range, request_id, filename = relative
            if schema not in product.schemas:
                findings.append(
                    {
                        "severity": "warning",
                        "kind": "undeclared_schema_in_archive",
                        "product": product.key,
                        "detail": f"schema '{schema}' is present but not declared",
                    }
                )
                continue
            match = _DATE_IN_FILENAME.search(filename)
            if match is None:
                findings.append(
                    {
                        "severity": "warning",
                        "kind": "undated_payload",
                        "product": product.key,
                        "detail": f"'{filename}' carries no YYYYMMDD stamp",
                    }
                )
                continue
            date = dt.datetime.strptime(match.group(1), "%Y%m%d").date()
            if date not in declared_dates:
                findings.append(
                    {
                        "severity": "informational",
                        "kind": "payload_for_undeclared_date",
                        "product": product.key,
                        "detail": f"'{filename}' covers {date.isoformat()}, which is not declared",
                    }
                )
                continue
            key = (product.key, schema, date)
            if key in seen:
                findings.append(
                    {
                        "severity": "blocking",
                        "kind": "duplicate_payload",
                        "product": product.key,
                        "detail": (
                            f"a second payload covers schema '{schema}' on "
                            f"{date.isoformat()}: '{filename}'"
                        ),
                    }
                )
                continue
            seen.add(key)
            found.append(
                InputFile(
                    product_key=product.key,
                    schema=schema,
                    date=date,
                    path=payload,
                    relative_path=payload.relative_to(archive_root).as_posix(),
                    request_id=request_id,
                    request_directory=payload.parent,
                    compressed_bytes=payload.stat().st_size,
                )
            )

        for schema in product.schemas:
            for date in sorted(declared_dates):
                if (product.key, schema, date) not in seen:
                    findings.append(
                        {
                            "severity": "blocking",
                            "kind": "expected_payload_missing",
                            "product": product.key,
                            "detail": (
                                f"no '{schema}' payload for {date.isoformat()}"
                            ),
                        }
                    )
    ordered = tuple(sorted(found, key=lambda item: (item.product_key, item.schema, item.date)))
    return ordered, findings


@dataclass(slots=True)
class ProductDateResult:
    """Everything one product/date pass produced."""

    product_key: str
    date: dt.date
    quote_audit: QuoteAudit | None = None
    statistics_audit: StatisticsAudit | None = None
    option_universe: OptionUniverse | None = None
    future_universe: FutureUniverse | None = None
    reference: ReferenceSelection | None = None
    reference_mids: dict[InstrumentKey, dict[int, float]] = field(default_factory=dict)
    reference_spreads: dict[InstrumentKey, QuantizedDistribution] = field(default_factory=dict)
    unnormalizable_records: int = 0


class Auditor:
    """Runs the whole audit for one configuration against one archive."""

    def __init__(
        self,
        config: MarketFeasibilityConfig,
        *,
        project_root: Path,
        max_records_per_file: int | None = None,
    ) -> None:
        self._config = config
        self._root = project_root
        self._archive = project_root / config.archive_root
        self._processed = project_root / config.processed_root
        self._max_records = max_records_per_file
        self._policy = TradabilityPolicy(
            require_positive_sizes=config.tradability.require_positive_sizes,
            require_positive_bid=config.tradability.require_positive_bid,
            require_positive_spread=config.tradability.require_positive_spread,
            exclude_vendor_flagged=config.tradability.exclude_vendor_flagged,
            exclude_stale=config.tradability.exclude_stale,
        )
        self._results: dict[tuple[str, dt.date], ProductDateResult] = {}
        self._findings: list[dict[str, Any]] = []
        self._file_summaries: list[dict[str, Any]] = []
        self._published: set[Path] = set()
        self._decode_failures: list[str] = []

    # -- decoding -----------------------------------------------------------

    def _records(self, source: InputFile) -> Iterator[Any]:
        handle = open_dbn(source.path, chunk_bytes=self._config.audit.read_chunk_bytes)
        for count, record in enumerate(handle, start=1):
            yield record
            if self._max_records is not None and count >= self._max_records:
                break
        self._file_summaries.append(_summary_row(source, handle.summary))

    def _window(self, date: dt.date) -> SessionWindow:
        opened, closed = self._config.session.window_utc(date)
        return SessionWindow.from_utc(
            date,
            opened,
            closed,
            frequency_seconds=self._config.synchronization_frequency_seconds,
        )

    def _result(self, product_key: str, date: dt.date) -> ProductDateResult:
        key = (product_key, date)
        if key not in self._results:
            self._results[key] = ProductDateResult(product_key=product_key, date=date)
        return self._results[key]

    def _destination(self, product_key: str, schema: str, date: dt.date) -> Path:
        return partition_path(
            self._processed, product_key=product_key, schema=schema, date=date.isoformat()
        )

    def _writer(self, destination: Path, schema: Any) -> AtomicParquetWriter:
        return AtomicParquetWriter(
            destination,
            schema,
            batch_rows=self._config.audit.parquet_batch_rows,
            compression=self._config.audit.parquet_compression,
        )

    # -- per-schema passes --------------------------------------------------

    def ingest_definitions(self, product: ProductSpec, source: InputFile) -> None:
        """Resolve and persist one definition payload."""
        module = require_dbn_module()
        definition_type = module.InstrumentDefMsg
        result = self._result(product.key, source.date)
        records = (
            record for record in self._records(source) if isinstance(record, definition_type)
        )
        destination = self._destination(product.key, source.schema, source.date)

        if product.instrument_kind == "option":
            universe = build_option_universe(records, product_root=product.product)
            result.option_universe = universe
            with self._writer(destination, OPTION_DEFINITION_SCHEMA) as writer:
                writer.write_rows(option_definition_rows(universe.contracts.values()))
            self._published.add(destination)
            if universe.non_standard():
                self._findings.append(
                    {
                        "severity": "informational",
                        "kind": "non_standard_option_root",
                        "product": product.key,
                        "detail": (
                            f"{len(universe.non_standard())} contracts on "
                            f"{source.date.isoformat()} carry an OSI root other than "
                            f"'{product.product}', which is how OPRA marks a non-standard "
                            f"deliverable after a corporate action; they are excluded from "
                            f"the standard universe and counted separately"
                        ),
                    }
                )
            return

        future_universe = build_future_universe(records)
        result.future_universe = future_universe
        with self._writer(destination, FUTURE_DEFINITION_SCHEMA) as writer:
            writer.write_rows(future_definition_rows(future_universe.contracts.values()))
        self._published.add(destination)

    def _population_classifier(
        self, result: ProductDateResult
    ) -> Callable[[InstrumentKey], str] | None:
        """Return the rule assigning each quoting instrument to a population.

        A futures payload mixes outrights with calendar spreads whose prices mean
        entirely different things, so the primary population is the outrights and
        the spreads are accumulated beside them. An option payload's primary
        population is its resolved chain; anything quoting without a definition is
        an orphan and is kept out of the headline numbers rather than inflating
        them.
        """
        if result.future_universe is not None:
            classes = {
                key: contract.is_outright
                for key, contract in result.future_universe.contracts.items()
            }

            def classify_future(key: InstrumentKey) -> str:
                outright = classes.get(key)
                if outright is None:
                    return UNCLASSIFIED_POPULATION
                return PRIMARY_POPULATION if outright else MULTI_LEG_POPULATION

            return classify_future
        if result.option_universe is not None:
            resolved = set(result.option_universe.contracts)

            def classify_option(key: InstrumentKey) -> str:
                return PRIMARY_POPULATION if key in resolved else UNCLASSIFIED_POPULATION

            return classify_option
        return None

    def ingest_quotes(self, product: ProductSpec, source: InputFile) -> None:
        """Normalize, grade and persist one quote payload in a single pass."""
        window = self._window(source.date)
        result = self._result(product.key, source.date)
        audit = QuoteAudit(
            slot_count=window.slot_count,
            relative_spread_thresholds=(
                self._config.tradability.exploratory_relative_spread_thresholds
            ),
            classifier=self._population_classifier(result),
        )
        result.quote_audit = audit
        # Only a product that can act as a reference needs a per-instrument mid
        # series, and those products hold tens of instruments rather than tens of
        # thousands, so this stays bounded.
        track_reference = product.reference_selection is not None

        previous: dict[InstrumentKey, tuple[float | None, float | None, int | None, int | None]]
        previous = {}
        destination = self._destination(product.key, source.schema, source.date)
        skipped = 0

        with self._writer(destination, QUOTE_SCHEMA) as writer:
            for record in self._records(source):
                if not hasattr(record, "levels"):
                    skipped += 1
                    continue
                try:
                    identity = InstrumentKey(
                        int(record.publisher_id), int(record.instrument_id)
                    )
                    quote = normalize_quote(
                        record,
                        window=window,
                        previous=previous.get(identity),
                        policy=self._policy,
                    )
                except (AttributeError, ValueError, TypeError):
                    skipped += 1
                    continue
                previous[quote.key] = quote_state_key(quote)
                audit.observe(quote, window)
                if track_reference and quote.tradable_core and quote.in_regular_session:
                    self._track_reference_quote(result, quote, window)
                writer.write_row(quote_row(quote))
        self._published.add(destination)

        result.unnormalizable_records = skipped
        if skipped:
            self._findings.append(
                {
                    "severity": "warning",
                    "kind": "unnormalizable_quote_records",
                    "product": product.key,
                    "detail": (
                        f"{skipped} records on {source.date.isoformat()} carried no usable "
                        f"book level and were counted but not normalized"
                    ),
                }
            )

    def _track_reference_quote(
        self, result: ProductDateResult, quote: NormalizedQuote, window: SessionWindow
    ) -> None:
        slot = window.grid_index(quote.ts_recv_ns)
        if slot is None or quote.mid_price is None:
            return
        result.reference_mids.setdefault(quote.key, {})[slot] = quote.mid_price
        if quote.quoted_spread is not None:
            distribution = result.reference_spreads.get(quote.key)
            if distribution is None:
                distribution = QuantizedDistribution(grid=0.01, cap=1_000_000)
                result.reference_spreads[quote.key] = distribution
            distribution.add(quote.quoted_spread)

    def ingest_statistics(self, product: ProductSpec, source: InputFile) -> None:
        """Persist and summarize one statistics payload, deduplicated."""
        module = require_dbn_module()
        stat_type = module.StatMsg
        result = self._result(product.key, source.date)
        audit = result.statistics_audit or StatisticsAudit()
        result.statistics_audit = audit
        destination = self._destination(product.key, source.schema, source.date)
        outright_by_key = self._outright_lookup(product.key, source.date)

        with self._writer(destination, STATISTIC_SCHEMA) as writer:
            for record in self._records(source):
                if not isinstance(record, stat_type):
                    continue
                ts_recv = decode_timestamp(int(record.ts_recv))
                if ts_recv is None:
                    continue
                identity = InstrumentKey(int(record.publisher_id), int(record.instrument_id))
                name = _stat_name(record.stat_type)
                price = decode_price(int(record.price))
                quantity = decode_quantity(int(record.quantity))
                ts_ref = decode_timestamp(int(record.ts_ref))
                action = _stat_name(record.update_action)
                is_outright = outright_by_key.get(identity)
                repeat = audit.observe(
                    stat_type=name,
                    key=identity,
                    price=price,
                    quantity=quantity,
                    ts_ref_ns=ts_ref,
                    update_action=action,
                    is_outright=is_outright,
                )
                writer.write_row(
                    {
                        "ts_recv_ns": ts_recv,
                        "ts_ref_ns": ts_ref,
                        "instrument_id": identity.instrument_id,
                        "publisher_id": identity.publisher_id,
                        "stat_type": name,
                        "price": price,
                        "quantity": quantity,
                        "update_action": action,
                        "stat_flags": int(getattr(record, "stat_flags", 0) or 0),
                        "is_outright": is_outright,
                        "is_repeat_of_earlier_record": repeat,
                    }
                )
        self._published.add(destination)

    def _outright_lookup(self, product_key: str, date: dt.date) -> dict[InstrumentKey, bool]:
        result = self._results.get((product_key, date))
        if result is None or result.future_universe is None:
            return {}
        return {
            key: contract.is_outright
            for key, contract in result.future_universe.contracts.items()
        }

    # -- orchestration ------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Verify, ingest and audit the whole archive; return the report document."""
        verification = require_verified_archive(
            self._archive,
            self._config.sha256_manifest,
            expected_entry_count=self._config.expected_manifest_entries,
            expected_total_files=self._config.expected_total_files,
        )
        inputs, discovery_findings = discover_inputs(self._config, self._archive)
        self._findings.extend(discovery_findings)
        if not inputs:
            raise IngestError("no declared payloads were found in the archive")

        conditions, condition_coverage = self._read_conditions(inputs)

        by_key = {product.key: product for product in self._config.products}
        # Definitions first so a quote pass can be reconciled against a resolved
        # chain, then quotes, then statistics.
        order = {"definition": 0, "bbo-1m": 1, "cbbo-1m": 1, "statistics": 2}
        for source in sorted(
            inputs, key=lambda item: (order[item.schema], item.product_key, item.date)
        ):
            product = by_key[source.product_key]
            try:
                if source.schema == "definition":
                    self.ingest_definitions(product, source)
                elif source.schema in {"bbo-1m", "cbbo-1m"}:
                    self.ingest_quotes(product, source)
                elif source.schema == "statistics":
                    self.ingest_statistics(product, source)
            except DbnError as error:
                self._decode_failures.append(f"{source.relative_path}: {error}")
                self._findings.append(
                    {
                        "severity": "blocking",
                        "kind": "payload_decode_failed",
                        "product": product.key,
                        "detail": f"{source.relative_path}: {error}",
                    }
                )

        self._resolve_references()
        return self._build_report(
            verification, inputs, conditions, condition_coverage
        )

    def _read_conditions(
        self, inputs: Sequence[InputFile]
    ) -> tuple[tuple[VendorCondition, ...], dict[str, Any]]:
        """Consume the vendor's per-date condition statement for every request.

        Coverage is checked pair by pair: every (request, declared date) the audit
        actually decoded must have its own ``available`` statement. Asking only
        whether *some* relevant condition entry exists would let one available
        date in a two-date request vouch for a second date the vendor never
        described.
        """
        filename = self._config.vendor_condition_filename
        declared = {entry.date for entry in self._config.dates}
        expected = {
            (item.request_directory.relative_to(self._archive).as_posix(), item.date)
            for item in inputs
        }
        entries: list[VendorCondition] = []
        for directory in sorted({item.request_directory for item in inputs}):
            candidate = directory / filename
            relative = directory.relative_to(self._archive).as_posix()
            if not candidate.is_file():
                self._findings.append(
                    {
                        "severity": "blocking",
                        "kind": "vendor_condition_missing",
                        "product": "",
                        "detail": (
                            f"'{relative}' carries no '{filename}'; the vendor's own "
                            f"completeness statement for this request is unavailable"
                        ),
                    }
                )
                continue
            try:
                entries.extend(read_vendor_conditions(candidate, relative_to=self._archive))
            except ConditionError as error:
                self._findings.append(
                    {
                        "severity": "blocking",
                        "kind": "vendor_condition_unreadable",
                        "product": "",
                        "detail": str(error),
                    }
                )

        available_pairs = {
            (entry.request_path, entry.date) for entry in entries if entry.is_available
        }
        stated_pairs = {(entry.request_path, entry.date) for entry in entries}
        uncovered = sorted(
            f"{request}@{date.isoformat()}"
            for request, date in expected - stated_pairs
        )
        unavailable = sorted(
            f"{request}@{date.isoformat()}"
            for request, date in (expected & stated_pairs) - available_pairs
        )
        for label in uncovered:
            self._findings.append(
                {
                    "severity": "blocking",
                    "kind": "vendor_condition_absent_for_decoded_date",
                    "product": "",
                    "detail": (
                        f"'{label}' was decoded but the vendor states no condition for that "
                        f"request and date"
                    ),
                }
            )
        for entry in entries:
            if entry.date in declared and not entry.is_available:
                self._findings.append(
                    {
                        "severity": "blocking",
                        "kind": "vendor_condition_not_available",
                        "product": "",
                        "detail": (
                            f"'{entry.request_path}' reports condition '{entry.condition}' "
                            f"for {entry.raw_date}, not '{CONDITION_AVAILABLE}'"
                        ),
                    }
                )
        coverage = {
            "expected_request_dates": len(expected),
            "request_dates_with_an_available_statement": len(expected & available_pairs),
            "request_dates_without_any_statement": uncovered,
            "request_dates_not_available": unavailable,
            "every_decoded_request_date_is_available": (
                bool(expected) and expected <= available_pairs
            ),
        }
        return tuple(entries), coverage

    def _resolve_references(self) -> None:
        """Apply each reference product's declared selection rule, per date."""
        for product in self._config.products:
            if product.reference_selection is None:
                continue
            for entry in self._config.dates:
                result = self._results.get((product.key, entry.date))
                if result is None:
                    continue
                selection = ReferenceSelection(
                    product_key=product.key, rule=product.reference_selection
                )
                chosen: FutureContract | None = None
                if result.future_universe is not None:
                    chosen, detail = result.future_universe.select_reference(
                        product.reference_selection, entry.date
                    )
                    selection.detail = detail
                    if chosen is not None:
                        selection.key = chosen.key
                        selection.symbol = chosen.raw_symbol
                elif product.reference_selection == REFERENCE_SELECTION_SINGLE:
                    quoting = sorted(result.reference_mids)
                    if len(quoting) == 1:
                        selection.key = quoting[0]
                        selection.symbol = product.product
                        selection.detail = (
                            f"rule '{REFERENCE_SELECTION_SINGLE}': the product quoted exactly "
                            f"one instrument ({quoting[0].publisher_id}/"
                            f"{quoting[0].instrument_id})"
                        )
                    else:
                        selection.detail = (
                            f"rule '{REFERENCE_SELECTION_SINGLE}': {len(quoting)} instruments "
                            f"quoted, so a single reference is not defined"
                        )
                else:
                    selection.detail = (
                        f"rule '{product.reference_selection}' needs resolved definitions, "
                        f"which this product does not declare"
                    )

                if selection.key is not None:
                    mids = result.reference_mids.get(selection.key, {})
                    if mids:
                        slot = max(mids)
                        selection.mid = mids[slot]
                        selection.mid_slot = slot
                    selection.spread = result.reference_spreads.get(selection.key)
                    if not mids:
                        selection.detail += "; the selected instrument had no tradable minute"
                else:
                    self._findings.append(
                        {
                            "severity": "warning",
                            "kind": "reference_instrument_unresolved",
                            "product": product.key,
                            "detail": f"{entry.date.isoformat()}: {selection.detail}",
                        }
                    )
                result.reference = selection

    def _spot_reference(self, product: ProductSpec, date: dt.date) -> SpotReference:
        """Return the declared spot anchor for ``product`` on ``date``.

        Attribution is per product. A product whose underlying was not acquired
        gets ``unavailable`` unless the configuration explicitly declares a proxy
        together with the caveat that travels with every number derived from it.
        """
        if product.underlying_product is not None:
            return self._reference_spot(product.underlying_product, date, basis="direct")
        if product.proxy_underlying_product is not None:
            spot = self._reference_spot(
                product.proxy_underlying_product, date, basis="proxy"
            )
            spot.caveat = product.proxy_caveat
            return spot
        return SpotReference(
            basis="unavailable",
            caveat=(
                f"product '{product.key}' settles on {product.settles_on}, which was not "
                f"acquired and for which no proxy is declared; spot moneyness is not "
                f"computed"
            ),
        )

    def _reference_spot(self, product_key: str, date: dt.date, *, basis: str) -> SpotReference:
        product = self._config.product(product_key)
        result = self._results.get((product_key, date))
        selection = None if result is None else result.reference
        if selection is None or selection.mid is None:
            return SpotReference(
                basis="unavailable",
                source_product=product_key,
                selection_rule=product.reference_selection,
                feed_class=product.feed_class,
                is_nbbo_grade=product.is_nbbo_grade,
                caveat=(
                    None if selection is None else f"no reference mid: {selection.detail}"
                ),
            )
        return SpotReference(
            value=selection.mid,
            basis=basis,
            source_product=product_key,
            source_instrument=selection.symbol,
            selection_rule=selection.rule,
            feed_class=product.feed_class,
            is_nbbo_grade=product.is_nbbo_grade,
            uncertainty={
                "note": (
                    "the anchor is one feed's mid at one minute; its quoted-spread "
                    "distribution bounds how precisely that feed locates the price"
                ),
                "feed_class": product.feed_class,
                "is_nbbo_grade": product.is_nbbo_grade,
                "reference_instrument": selection.symbol,
                "mid_slot": selection.mid_slot,
                "quoted_spread_distribution": (
                    None if selection.spread is None else selection.spread.summary()
                ),
            },
        )

    def _option_coverage(self, date: dt.date) -> dict[str, Any]:
        from differentiable_pricing.market.feasibility import OptionCoverageAudit

        coverage: dict[str, Any] = {}
        for product in self._config.products:
            if product.instrument_kind != "option":
                continue
            result = self._results.get((product.key, date))
            if result is None or result.option_universe is None:
                continue
            audit = OptionCoverageAudit(
                session_date=date,
                moneyness_edges=self._config.audit.moneyness_bucket_edges,
                maturity_edges_days=self._config.audit.maturity_bucket_edges_days,
                spot=self._spot_reference(product, date),
            )
            quoted = (
                sorted(result.quote_audit.raw_slots)
                if result.quote_audit is not None
                else []
            )
            masks = (
                result.quote_audit.masks_for_universe(quoted)
                if result.quote_audit is not None
                else {}
            )
            audit.observe_chain(result.option_universe.contracts, quoted, masks)
            coverage[product.key] = audit.summary()
            coverage[product.key]["expiries_with_minimum_paired_strikes"] = (
                audit.expiries_with_paired_strikes(
                    self._config.capabilities.minimum_paired_strikes_per_expiry
                )
            )
        return coverage

    def _synchronization(self, date: dt.date) -> list[dict[str, Any]]:
        summaries: list[dict[str, Any]] = []
        for pair in self._config.synchronization_pairs:
            option = self._results.get((pair.option_product, date))
            reference = self._results.get((pair.reference_product, date))
            if option is None or reference is None:
                continue
            if option.quote_audit is None or reference.quote_audit is None:
                continue
            selection = reference.reference
            reference_mask = 0
            if selection is not None and selection.key is not None:
                # The reference is one named outright instrument, so a
                # spread-only minute cannot make the reference look present.
                reference_mask = reference.quote_audit.mask_for(
                    selection.key, tradable=True, population=PRIMARY_POPULATION
                )
            universe = self._primary_universe_keys(option)
            audit = SynchronizationAudit(
                name=pair.name,
                option_product=pair.option_product,
                reference_product=pair.reference_product,
                role=pair.role,
                slot_count=option.quote_audit.slot_count,
                reference_selection_rule=None if selection is None else selection.rule,
                reference_instrument=None if selection is None else selection.symbol,
                reference_selection_detail=None if selection is None else selection.detail,
                reference_mask=reference_mask,
                option_masks=option.quote_audit.masks_for_universe(universe),
            )
            summaries.append(audit.summary())
        return summaries

    def _primary_universe_keys(self, result: ProductDateResult) -> list[InstrumentKey]:
        """Return the resolved primary instruments of a product/date.

        For options that is the resolved chain; for futures, the outrights only.
        Contracts that never quoted are included, because they are part of the
        universe a study would have to cover.
        """
        if result.option_universe is not None:
            return sorted(result.option_universe.contracts)
        if result.future_universe is not None:
            return sorted(result.future_universe.outrights())
        if result.quote_audit is not None:
            return sorted(result.quote_audit.raw_slots)
        return []

    def _fred_audit(self) -> dict[str, Any]:
        series: dict[str, Any] = {}
        declared_dates = [entry.date for entry in self._config.dates]
        all_numeric = True
        for spec in self._config.fred_series:
            path = self._archive / spec.path
            observations: dict[dt.date, str] = {}
            if path.is_file():
                for number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(), start=1
                ):
                    if number == 1 or not line.strip():
                        continue
                    date_text, _, value = line.partition(",")
                    try:
                        observations[dt.date.fromisoformat(date_text.strip())] = value.strip()
                    except ValueError:
                        continue
            numeric = sum(1 for value in observations.values() if _is_float(value))
            session_values = {
                date.isoformat(): observations.get(date) for date in sorted(declared_dates)
            }
            missing_numeric = sorted(
                date.isoformat()
                for date in declared_dates
                if not _is_float(observations.get(date, ""))
            )
            if missing_numeric:
                all_numeric = False
                self._findings.append(
                    {
                        "severity": "blocking",
                        "kind": "fred_series_missing_session_observation",
                        "product": spec.series_id,
                        "detail": (
                            f"series '{spec.series_id}' has no numeric observation on "
                            f"{', '.join(missing_numeric)}"
                        ),
                    }
                )
            series[spec.series_id] = {
                "role": spec.role,
                "path": spec.path,
                "present": path.is_file(),
                "observations": len(observations),
                "numeric_observations": numeric,
                "non_numeric_observations": len(observations) - numeric,
                "first_date": min(observations).isoformat() if observations else None,
                "last_date": max(observations).isoformat() if observations else None,
                "covers_all_session_dates": set(declared_dates) <= set(observations),
                "numeric_on_all_session_dates": not missing_numeric,
                "session_dates_without_a_numeric_value": missing_numeric,
                "session_date_values": session_values,
                "observation_frequency": "daily",
            }
        return {
            "series": series,
            "date_alignment": (
                "a FRED row dated D carries the value for D. SOFR for D is published on the "
                "following business morning, so a same-day intraday consumer would not have "
                "had it; the audit aligns on the value date and states the lag rather than "
                "shifting the series."
            ),
            "assessment": (
                "Every FRED series here is a daily published rate. SOFR is a "
                "backward-looking overnight benchmark fixed once per day and the DGS "
                "series are daily constant-maturity Treasury yields. None is a "
                "tradable instrument and none carries an intraday timestamp: they are "
                "daily controls only."
            ),
            "numeric_on_every_session_date": all_numeric,
            "sufficient_as_tradable_ois_curve": False,
        }

    def _rate_future_assessment(self) -> dict[str, Any]:
        product = next(
            (spec for spec in self._config.products if spec.role == "rate_control"), None
        )
        if product is None:
            return {"available": False}
        per_date: dict[str, Any] = {}
        for entry in self._config.dates:
            result = self._results.get((product.key, entry.date))
            if result is None:
                continue
            statistics = (
                result.statistics_audit.summary()
                if result.statistics_audit is not None
                else {"records": 0, "by_stat_type": {}}
            )
            universe = result.future_universe
            outrights = {} if universe is None else universe.outrights()
            unexpired = (
                {} if universe is None else universe.unexpired_outrights(entry.date)
            )
            settlement_outrights = (
                0
                if result.statistics_audit is None
                else result.statistics_audit.outright_instruments("SETTLEMENT_PRICE")
            )
            per_date[entry.date.isoformat()] = {
                "definition_records": 0 if universe is None else universe.definition_records,
                "outright_contracts": len(outrights),
                "unexpired_outright_contracts": len(unexpired),
                "multi_leg_contracts": 0 if universe is None else len(universe.multi_leg()),
                "outright_contracts_with_a_settlement_price": settlement_outrights,
                "statistics": statistics,
            }
        # Every statement below is derived from what the configuration declares
        # and what the decode observed, never from a constant. If a quote schema
        # is added to this product later, these fields change on their own.
        quote_schemas = sorted(set(product.schemas) & QUOTE_SCHEMAS)
        observed_quote_records = sum(
            result.quote_audit.records
            for entry in self._config.dates
            if (result := self._results.get((product.key, entry.date))) is not None
            and result.quote_audit is not None
        )
        intraday_available = bool(quote_schemas) and observed_quote_records > 0
        observed_stat_types = sorted(
            {
                stat_type
                for entry in self._config.dates
                if (result := self._results.get((product.key, entry.date))) is not None
                and result.statistics_audit is not None
                for stat_type in result.statistics_audit.by_stat_type
            }
        )
        if intraday_available:
            why = (
                f"quote schemas {quote_schemas} were acquired and produced "
                f"{observed_quote_records} records"
            )
        elif quote_schemas:
            why = (
                f"quote schemas {quote_schemas} are declared for this product but no quote "
                f"record was decoded"
            )
        else:
            why = (
                "no quote schema is declared for this product, so no minute-aligned rate "
                "observation exists in the archive. Whether a daily settlement strip is "
                "adequate for a given study is a modelling decision this audit does not make."
            )
        return {
            "product": product.key,
            "schemas_acquired": list(product.schemas),
            "quote_schemas_acquired": quote_schemas,
            "quote_records_observed": observed_quote_records,
            "statistic_types_observed": observed_stat_types,
            "per_date": per_date,
            "what_the_statistics_provide": (
                "Per-contract session statistics, reported by the stat types actually "
                "observed rather than an assumed list. Counts are deduplicated by "
                "(publisher, instrument, stat type, reference time, price, quantity), and "
                "outright contracts are counted separately from calendar spreads."
            ),
            "intraday_rate_quotes_available": intraday_available,
            "why_no_intraday_rate_quotes": None if intraday_available else why,
            "intraday_rate_quotes_basis": why,
        }

    def _declared_sources(self) -> tuple[SourceObservation, ...]:
        observations = tuple(
            probe_declared_source(
                self._archive,
                key=source.key,
                path=source.path,
                kind=source.kind,
                requirement=source.requirement,
                capability=source.capability,
                why=source.why,
                candidate_source=source.candidate_source,
            )
            for source in self._config.declared_sources
        )
        for observation in observations:
            if observation.satisfied:
                continue
            self._findings.append(
                {
                    "severity": (
                        "blocking" if observation.requirement == "required" else "warning"
                    ),
                    "kind": f"declared_source_{observation.state}",
                    "product": observation.key,
                    "detail": observation.detail,
                }
            )
        return observations

    def _processed_bytes(self) -> tuple[int, dict[str, int], list[str]]:
        """Return published bytes, per-product bytes and any stale partitions.

        Only partitions this run published are counted. A partition left behind
        by an earlier run with a different configuration is reported and excluded,
        so the storage projection cannot be inflated by a dirty output tree.
        """
        total = 0
        by_product: dict[str, int] = {}
        stale: list[str] = []
        if not self._processed.is_dir():
            return 0, {}, []
        for path in sorted(self._processed.rglob("*.parquet")):
            relative = path.relative_to(self._processed)
            if path not in self._published:
                stale.append(relative.as_posix())
                continue
            size = path.stat().st_size
            total += size
            product = relative.parts[0].removeprefix("product=")
            by_product[product] = by_product.get(product, 0) + size
        return total, dict(sorted(by_product.items())), stale

    def _build_report(
        self,
        verification: Any,
        inputs: Sequence[InputFile],
        conditions: Sequence[VendorCondition],
        condition_coverage: Mapping[str, Any],
    ) -> dict[str, Any]:
        config = self._config
        declared_sources = self._declared_sources()
        per_date: dict[str, Any] = {}
        for entry in config.dates:
            products: dict[str, Any] = {}
            for product in config.products:
                result = self._results.get((product.key, entry.date))
                if result is None:
                    continue
                products[product.key] = self._product_block(product, result)
            per_date[entry.date.isoformat()] = {
                "role": entry.role,
                "session_window_utc": [
                    stamp.isoformat() for stamp in config.session.window_utc(entry.date)
                ],
                "products": products,
                "option_coverage": self._option_coverage(entry.date),
                "synchronization": self._synchronization(entry.date),
            }

        compressed = sum(item.compressed_bytes for item in inputs)
        processed_total, processed_by_product, stale = self._processed_bytes()
        if stale:
            self._findings.append(
                {
                    "severity": "warning",
                    "kind": "stale_processed_partition",
                    "product": "",
                    "detail": (
                        f"{len(stale)} parquet files under the processed root were not "
                        f"published by this run and are excluded from the storage "
                        f"projection: {', '.join(stale[:5])}"
                    ),
                }
            )
        storage = extrapolate_storage(
            observed_compressed_bytes=compressed,
            observed_processed_bytes=processed_total,
            observed_dates=len(config.dates),
            target_dates=config.audit.extrapolation_trading_dates,
        )
        storage["processed_bytes_by_product"] = processed_by_product
        storage["archive_total_bytes_including_unprocessed"] = verification.total_bytes
        storage["stale_partitions_excluded"] = stale

        fred = self._fred_audit()
        rate_futures = self._rate_future_assessment()
        capabilities = self._capabilities(
            verification=verification,
            condition_coverage=condition_coverage,
            declared_sources=declared_sources,
            per_date=per_date,
            fred=fred,
            rate_futures=rate_futures,
            stale=stale,
        )

        return {
            "schema_version": REPORT_SCHEMA_VERSION,
            "audit_version": config.audit_version,
            "config": {
                "source_name": config.source_name,
                "source_sha256": config.source_sha256,
                "archive_root": config.archive_root,
                "processed_root": config.processed_root,
                "session": {
                    "name": config.session.name,
                    "time_zone": config.session.time_zone,
                    "open_local": config.session.open_local.isoformat(),
                    "close_local": config.session.close_local.isoformat(),
                    "expected_minutes": config.session.expected_minutes,
                },
                "synchronization_frequency_seconds": (
                    config.synchronization_frequency_seconds
                ),
                "synchronization_label": config.synchronization_label,
                "tradability": {
                    **self._policy.describe(),
                    "exploratory_relative_spread_thresholds": list(
                        config.tradability.exploratory_relative_spread_thresholds
                    ),
                },
                "capability_minima": config.capabilities.summary(),
                "staleness_basis": STALENESS_BASIS,
            },
            "manifest": {
                "path": verification.manifest_path,
                "sha256": verification.manifest_sha256,
                "entries": verification.entry_count,
                "verified": verification.verified_count,
                "bytes_hashed": verification.total_bytes,
                "missing": list(verification.missing),
                "altered": [item.relative_path for item in verification.altered],
                "duplicated": list(verification.duplicated),
                "unexpected": list(verification.unexpected),
                "unreadable": list(verification.unreadable),
                "note": (
                    "A SHA-256 manifest covers files. An empty directory is invisible to it, "
                    "which is why declared sources are probed separately."
                ),
            },
            "vendor_conditions": {
                "filename": config.vendor_condition_filename,
                "entries": [entry.summary() for entry in conditions],
                "coverage": dict(condition_coverage),
                "all_declared_dates_available": all(
                    entry.is_available
                    for entry in conditions
                    if entry.date in {item.date for item in config.dates}
                ),
            },
            "declared_sources": [item.summary() for item in declared_sources],
            "inputs": {
                "payloads_decoded": len(inputs),
                "compressed_bytes_decoded": compressed,
                "decode_failures": list(self._decode_failures),
                "files": [_input_row(item) for item in inputs],
                "dbn_headers": sorted(
                    self._file_summaries, key=lambda row: row["relative_path"]
                ),
            },
            "per_date": per_date,
            "rates": {"fred": fred, "rate_futures": rate_futures},
            "storage": storage,
            "findings": sorted(
                self._findings,
                key=lambda item: (
                    _SEVERITY_ORDER.get(item["severity"], 99),
                    item["kind"],
                    item["detail"],
                ),
            ),
            "capabilities": capabilities,
            "deferred_work": list(DEFERRED_WORK),
            "deferred_low_findings": list(DEFERRED_LOW_FINDINGS),
        }

    def _product_block(self, product: ProductSpec, result: ProductDateResult) -> dict[str, Any]:
        block: dict[str, Any] = {
            "role": product.role,
            "instrument_kind": product.instrument_kind,
            "settles_on": product.settles_on,
            "feed_class": product.feed_class,
            "is_nbbo_grade": product.is_nbbo_grade,
        }
        universe_size: int | None = None
        if result.option_universe is not None:
            universe = result.option_universe
            universe_size = len(universe.contracts)
            block["definition_lifecycle"] = universe.counters.as_dict()
            block["definition_resolution"] = {
                "records": universe.definition_records,
                "resolved": len(universe.contracts),
                "standard_root": len(universe.standard()),
                "non_standard_root": len(universe.non_standard()),
                "failed": len(universe.failures),
                "failure_examples": [
                    {
                        "instrument_id": failure.instrument_id,
                        "publisher_id": failure.publisher_id,
                        "raw_symbol": failure.raw_symbol,
                        "reason": failure.reason,
                        "detail": failure.detail,
                    }
                    for failure in universe.failures[:5]
                ],
            }
            block["contract_metadata_evidence"] = _metadata_evidence(universe)
        if result.future_universe is not None:
            universe = result.future_universe
            outrights = universe.outrights()
            universe_size = len(outrights)
            block["definition_lifecycle"] = universe.counters.as_dict()
            block["definition_resolution"] = {
                "records": universe.definition_records,
                "resolved": len(universe.contracts),
                "failed": len(universe.failures),
                "failure_examples": [
                    {
                        "instrument_id": failure.instrument_id,
                        "publisher_id": failure.publisher_id,
                        "raw_symbol": failure.raw_symbol,
                        "reason": failure.reason,
                        "detail": failure.detail,
                    }
                    for failure in universe.failures[:5]
                ],
            }
            block["futures"] = {
                "outright_contracts": len(outrights),
                "unexpired_outright_contracts": len(
                    universe.unexpired_outrights(result.date)
                ),
                "multi_leg_contracts": len(universe.multi_leg()),
                "outright_symbols": sorted(item.raw_symbol for item in outrights.values()),
                "multi_leg_symbols": sorted(
                    item.raw_symbol for item in universe.multi_leg().values()
                ),
            }
        if result.reference is not None:
            block["reference_selection"] = result.reference.summary()
        if result.quote_audit is not None:
            block["quotes"] = result.quote_audit.summary(universe_size=universe_size)
            if result.future_universe is not None:
                block["quotes_by_instrument_class"] = self._futures_quote_split(result)
        if result.statistics_audit is not None:
            block["statistics"] = result.statistics_audit.summary()
        block["unnormalizable_records"] = result.unnormalizable_records
        return block

    def _futures_quote_split(self, result: ProductDateResult) -> dict[str, Any]:
        """Report each futures instrument population separately.

        The primary population — the outrights — already carries the headline
        coverage. This block names the instruments behind every population so a
        reader can see what was and was not folded into that number.
        """
        assert result.future_universe is not None
        assert result.quote_audit is not None
        audit = result.quote_audit
        contracts = result.future_universe.contracts
        groups: dict[str, dict[str, Any]] = {}
        for label, coverage in sorted(audit.populations.items()):
            symbols = sorted(
                contracts[key].raw_symbol for key in coverage.raw_slots if key in contracts
            )
            groups[label] = {
                "instruments_with_quotes": len(coverage.raw_slots),
                "instruments_with_tradable_quotes": len(coverage.core_slots),
                "contract_minutes_present": coverage.raw_contract_minutes,
                "tradable_contract_minutes_present": coverage.core_contract_minutes,
                "quoted_spread_absolute": coverage.absolute_spread.summary(),
                "symbols": symbols,
            }
        return groups

    # -- scoped capabilities ------------------------------------------------

    def _capabilities(
        self,
        *,
        verification: Any,
        condition_coverage: Mapping[str, Any],
        declared_sources: Sequence[SourceObservation],
        per_date: Mapping[str, Any],
        fred: Mapping[str, Any],
        rate_futures: Mapping[str, Any],
        stale: Sequence[str],
    ) -> dict[str, Any]:
        """Evaluate every scoped capability from recorded checks only."""
        results = {
            "archive_integrity": self._archive_integrity(verification, condition_coverage),
            "ingestion_feasibility": self._ingestion_feasibility(per_date, stale),
            "xsp_european_surface_readiness": self._surface_readiness(
                per_date, fred, rate_futures, declared_sources
            ),
            "spy_american_calibration_readiness": self._american_readiness(
                per_date, declared_sources
            ),
        }
        results["replication_readiness"] = self._replication_readiness(results)
        ordered = {name: results[name] for name in CAPABILITIES}
        return {
            "rule": (
                "Each capability is satisfied when every one of its gating checks is "
                "satisfied. Every check states the observation it was decided by and no "
                "capability outcome is a constant. The exploratory relative-spread "
                "thresholds are descriptive and no check consults them. The capability "
                "minima are a different matter and carry their own provenance: the "
                "structural surface minima are properties of the method, while the "
                "session-count minima were chosen after this feasibility archive was "
                "observed and are provisional pilot targets until frozen. See "
                "config.capability_minima."
            ),
            "results": ordered,
        }

    def _archive_integrity(
        self, verification: Any, condition_coverage: Mapping[str, Any]
    ) -> dict[str, Any]:
        expected = condition_coverage["expected_request_dates"]
        available = condition_coverage["request_dates_with_an_available_statement"]
        checks = [
            _check(
                "the archive matches its SHA-256 manifest exactly",
                verification.ok,
                f"{verification.verified_count}/{verification.entry_count} entries verified, "
                f"{len(verification.missing)} missing, {len(verification.altered)} altered, "
                f"{len(verification.duplicated)} duplicated, "
                f"{len(verification.unexpected)} unexpected",
            ),
            _check(
                "no manifest-covered file is unreadable",
                not verification.unreadable,
                f"{len(verification.unreadable)} unreadable files",
            ),
            _check(
                f"the vendor states '{CONDITION_AVAILABLE}' for every decoded request and date",
                bool(condition_coverage["every_decoded_request_date_is_available"]),
                (
                    f"{available}/{expected} decoded (request, date) pairs carry an "
                    f"'{CONDITION_AVAILABLE}' statement; "
                    f"{len(condition_coverage['request_dates_without_any_statement'])} have no "
                    f"statement and "
                    f"{len(condition_coverage['request_dates_not_available'])} are not available"
                ),
            ),
        ]
        return _capability("archive_integrity", checks)

    def _ingestion_feasibility(
        self, per_date: Mapping[str, Any], stale: Sequence[str]
    ) -> dict[str, Any]:
        discovery_kinds = {
            "expected_payload_missing",
            "duplicate_payload",
            "product_directory_missing",
        }
        # Scoped to discovery: a blocking finding about a declared source belongs
        # to the capability that source gates, not to ingestion.
        blocking_kinds = {
            finding["kind"]
            for finding in self._findings
            if finding["severity"] == "blocking" and finding["kind"] in discovery_kinds
        }
        resolution_failures = 0
        resolution_records = 0
        off_grid = 0
        unnormalizable = 0
        for date_block in per_date.values():
            for block in date_block["products"].values():
                resolution = block.get("definition_resolution")
                if resolution is not None:
                    resolution_failures += resolution["failed"]
                    resolution_records += resolution["records"]
                quotes = block.get("quotes")
                if quotes is not None:
                    off_grid += quotes["off_grid_in_session_records"]
                unnormalizable += block.get("unnormalizable_records", 0)

        # Orphans are counted from the unclassified population of every product
        # that resolves definitions, so a futures instrument quoting without a
        # definition is gated exactly like an option one.
        orphans_by_product: dict[str, int] = {}
        for (product_key, _), result in self._results.items():
            audit = result.quote_audit
            if audit is None:
                continue
            if result.option_universe is None and result.future_universe is None:
                continue
            count = audit.instruments_in(UNCLASSIFIED_POPULATION)
            if count:
                orphans_by_product[product_key] = orphans_by_product.get(product_key, 0) + count
        orphan_instruments = sum(orphans_by_product.values())

        checks = [
            _check(
                "every declared payload is present exactly once",
                not blocking_kinds,
                f"blocking discovery findings: {sorted(blocking_kinds) or 'none'}",
            ),
            _check(
                "every payload decoded without truncation",
                not self._decode_failures,
                f"{len(self._decode_failures)} decode failures",
            ),
            _check(
                "every definition record resolves to a usable contract",
                resolution_failures == 0,
                f"{resolution_failures} failures across {resolution_records} records",
            ),
            _check(
                "every quoting instrument has a resolved definition",
                orphan_instruments == 0,
                (
                    f"{orphan_instruments} quoted instruments without a definition"
                    + (f" ({sorted(orphans_by_product.items())})" if orphans_by_product else "")
                ),
            ),
            _check(
                "every in-session record lands exactly on the synchronization grid",
                off_grid == 0,
                f"{off_grid} in-session records off grid",
            ),
            _check(
                "every quote record carried a book level",
                unnormalizable == 0,
                f"{unnormalizable} records could not be normalized",
            ),
            _check(
                "the processed root holds only partitions this run published",
                not stale,
                f"{len(stale)} stale partitions",
            ),
        ]
        return _capability("ingestion_feasibility", checks)

    def _surface_readiness(
        self,
        per_date: Mapping[str, Any],
        fred: Mapping[str, Any],
        rate_futures: Mapping[str, Any],
        declared_sources: Sequence[SourceObservation],
    ) -> dict[str, Any]:
        capability = "xsp_european_surface_readiness"
        product_key = self._european_product_key()
        minimum_expiries = self._config.capabilities.minimum_expiries_with_paired_strikes
        minimum_strikes = self._config.capabilities.minimum_paired_strikes_per_expiry
        worst_expiries: int | None = None
        attribution: set[str] = set()
        failures = 0
        for date_block in per_date.values():
            coverage = date_block["option_coverage"].get(product_key)
            if coverage is None:
                continue
            count = coverage["expiries_with_minimum_paired_strikes"]
            worst_expiries = count if worst_expiries is None else min(worst_expiries, count)
            attribution.add(coverage["spot_reference"]["basis"])
            block = date_block["products"].get(product_key, {})
            resolution = block.get("definition_resolution", {})
            failures += resolution.get("failed", 0)

        settlement_dates = [
            entry
            for entry in rate_futures.get("per_date", {}).values()
            if entry.get("outright_contracts_with_a_settlement_price", 0) > 0
        ]
        checks = [
            _check(
                f"the European product '{product_key}' resolves every definition record",
                failures == 0,
                f"{failures} unresolvable definition records",
            ),
            _check(
                f"every session carries at least {minimum_expiries} expiries with at least "
                f"{minimum_strikes} tradable call/put strike pairs",
                worst_expiries is not None and worst_expiries >= minimum_expiries,
                (
                    "no coverage"
                    if worst_expiries is None
                    else f"worst session has {worst_expiries} such expiries"
                ),
            ),
            _check(
                "the European product's spot attribution is declared, not inherited",
                bool(attribution) and attribution <= {"direct", "proxy", "unavailable"},
                f"observed spot attribution bases: {sorted(attribution) or 'none'}",
            ),
            _check(
                "a daily rate strip is available from outright rate-future settlements "
                "on every session",
                len(settlement_dates) == len(self._config.dates),
                f"{len(settlement_dates)}/{len(self._config.dates)} sessions carry outright "
                f"settlement prices",
            ),
            _check(
                "every declared daily rate control is numeric on every session date",
                bool(fred.get("numeric_on_every_session_date")),
                f"numeric on all session dates: {fred.get('numeric_on_every_session_date')}",
            ),
        ]
        checks.extend(_source_checks(declared_sources, capability))
        return _capability(capability, checks)

    def _american_readiness(
        self, per_date: Mapping[str, Any], declared_sources: Sequence[SourceObservation]
    ) -> dict[str, Any]:
        capability = "spy_american_calibration_readiness"
        product_key = self._american_product_key()
        nbbo = True
        observed_feeds: set[str] = set()
        multiplier_evidence = 0
        exercise_evidence = 0
        contracts = 0
        for date_block in per_date.values():
            coverage = date_block["option_coverage"].get(product_key)
            if coverage is not None:
                spot = coverage["spot_reference"]
                nbbo = nbbo and bool(spot["is_nbbo_grade"])
                if spot["feed_class"]:
                    observed_feeds.add(str(spot["feed_class"]))
            block = date_block["products"].get(product_key, {})
            evidence = block.get("contract_metadata_evidence")
            if evidence is not None:
                multiplier_evidence += evidence["contracts_with_a_contract_multiplier"]
                exercise_evidence += evidence["contracts_with_an_exercise_style"]
                contracts += evidence["contracts"]

        checks = [
            _check(
                "the underlying reference feed is a consolidated NBBO",
                nbbo and bool(observed_feeds),
                f"observed underlying feed classes: {sorted(observed_feeds) or 'none'}",
            ),
            _check(
                "the option definitions evidence a contract multiplier",
                contracts > 0 and multiplier_evidence == contracts,
                f"{multiplier_evidence}/{contracts} contracts carry a multiplier",
            ),
            _check(
                "the option definitions evidence an exercise style",
                contracts > 0 and exercise_evidence == contracts,
                f"{exercise_evidence}/{contracts} contracts carry an exercise style",
            ),
        ]
        checks.extend(_source_checks(declared_sources, capability))
        return _capability(capability, checks)

    def _replication_readiness(self, others: Mapping[str, Any]) -> dict[str, Any]:
        minimum = self._config.capabilities.minimum_sessions
        minimum_held_out = self._config.capabilities.minimum_held_out_sessions
        sessions = len(self._config.dates)
        held_out = len(self._config.held_out_dates())
        prerequisites = sorted(others)
        unsatisfied = [name for name in prerequisites if not others[name]["satisfied"]]
        checks = [
            _check(
                f"at least {minimum} distinct sessions are ingested",
                sessions >= minimum,
                f"{sessions} sessions declared and ingested",
            ),
            _check(
                f"at least {minimum_held_out} held-out sessions are reserved",
                held_out >= minimum_held_out,
                f"{held_out} held-out sessions declared",
            ),
            _check(
                "every prerequisite capability is satisfied",
                not unsatisfied,
                f"unsatisfied prerequisites: {unsatisfied or 'none'}",
            ),
        ]
        return _capability("replication_readiness", checks)

    def _european_product_key(self) -> str:
        for spec in self._config.products:
            if spec.role == "cash_settled_control":
                return spec.key
        return ""

    def _american_product_key(self) -> str:
        for spec in self._config.products:
            if spec.role == "primary_option_chain":
                return spec.key
        return ""


def _metadata_evidence(universe: OptionUniverse) -> dict[str, Any]:
    """Count how many contracts actually evidence their own conventions."""
    contracts = list(universe.contracts.values())
    return {
        "contracts": len(contracts),
        "contracts_with_a_contract_multiplier": sum(
            1 for item in contracts if item.contract_multiplier is not None
        ),
        "contracts_with_an_exercise_style": sum(
            1 for item in contracts if item.exercise_style is not None
        ),
        "contracts_with_a_unit_of_measure_quantity": sum(
            1 for item in contracts if item.unit_of_measure_quantity is not None
        ),
        "note": (
            "these are counts of what the definition records evidence, not assertions "
            "about market convention"
        ),
    }


def _check(
    requirement: str, satisfied: bool, observed: str, *, gating: bool = True
) -> dict[str, Any]:
    return {
        "requirement": requirement,
        "satisfied": bool(satisfied),
        "observed": observed,
        "gating": gating,
    }


def _source_checks(
    sources: Sequence[SourceObservation], capability: str
) -> list[dict[str, Any]]:
    """Turn declared-source observations for one capability into checks.

    A ``required`` source gates the capability; a ``recommended`` one is
    reported as advisory. Both carry the observed state, so ``absent`` and
    ``present_but_empty`` remain distinguishable in the check itself.
    """
    return [
        _check(
            f"declared source '{source.key}' supplies content",
            source.satisfied,
            f"state '{source.state}': {source.detail}",
            gating=source.requirement == "required",
        )
        for source in sources
        if source.capability == capability
    ]


CAPABILITY_SCOPES: Final = {
    "archive_integrity": (
        "whether the bytes on disk are the bytes that were acquired, and the vendor "
        "calls every requested date complete"
    ),
    "ingestion_feasibility": (
        "whether every declared payload decodes, resolves and normalizes without loss; "
        "it says nothing about whether the resulting data supports any study"
    ),
    "xsp_european_surface_readiness": (
        "whether the cash-settled European chain carries enough same-minute call/put "
        "structure, and enough daily rate control, to attempt a surface fit; it does not "
        "assert that any fit was performed or would succeed"
    ),
    "spy_american_calibration_readiness": (
        "whether the inputs an American exercise decision depends on — an NBBO-grade "
        "underlying, evidenced contract terms, dividends, borrow and adjustment history — "
        "are present in the archive"
    ),
    "replication_readiness": (
        "whether enough independent sessions and untouched held-out sessions exist to "
        "support a replicated performance estimate; it is about the protocol, not the data "
        "quality of any single session"
    ),
}
"""One line per capability saying exactly what its outcome does and does not mean."""


def _capability(name: str, checks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    unsatisfied = [
        item["requirement"]
        for item in checks
        if item.get("gating", True) and not item["satisfied"]
    ]
    advisory = [
        item["requirement"]
        for item in checks
        if not item.get("gating", True) and not item["satisfied"]
    ]
    return {
        "capability": name,
        "scope": CAPABILITY_SCOPES.get(name, ""),
        "satisfied": not unsatisfied,
        "outcome": "satisfied" if not unsatisfied else "not_satisfied",
        "checks": list(checks),
        "unsatisfied_requirements": unsatisfied,
        "unsatisfied_advisory_requirements": advisory,
    }


def _is_float(value: str) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _input_row(item: InputFile) -> dict[str, Any]:
    return {
        "product": item.product_key,
        "schema": item.schema,
        "date": item.date.isoformat(),
        "relative_path": item.relative_path,
        "request_id": item.request_id,
        "compressed_bytes": item.compressed_bytes,
    }


def _summary_row(source: InputFile, summary: DbnFileSummary) -> dict[str, Any]:
    return {
        "relative_path": source.relative_path,
        "product": source.product_key,
        "schema": source.schema,
        "date": source.date.isoformat(),
        "compression": summary.compression,
        "compressed_bytes": summary.compressed_bytes,
        "dbn_version": summary.dbn_version,
        "dataset": summary.dataset,
        "declared_schema": summary.schema,
        "stype_in": summary.stype_in,
        "stype_out": summary.stype_out,
        "start_ns": summary.start_ns,
        "end_ns": summary.end_ns,
        "symbols": list(summary.symbols),
        "symbol_mappings": summary.symbol_mapping_count,
        "records_decoded": summary.record_count,
    }


def write_json_atomically(destination: Path, document: Any) -> None:
    """Serialize ``document`` to ``destination`` atomically and deterministically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        dir=destination.parent, prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(document, stream, indent=2, sort_keys=True, default=str)
            stream.write("\n")
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def build_parser() -> argparse.ArgumentParser:
    """Return the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="python -m differentiable_pricing.market.ingest",
        description="Verify, ingest and audit a raw market archive for feasibility.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/market_feasibility_v1.toml"),
        help="frozen audit configuration to run",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="report path; defaults to <report_root>/feasibility-report-v1.json",
    )
    parser.add_argument(
        "--max-records-per-file",
        type=int,
        default=None,
        help="stop after this many records per payload; for smoke tests only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the audit and write its report."""
    arguments = build_parser().parse_args(argv)
    try:
        config = load_market_config(arguments.config)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    auditor = Auditor(
        config,
        project_root=PROJECT_ROOT,
        max_records_per_file=arguments.max_records_per_file,
    )
    try:
        report = auditor.run()
    except (ManifestError, DbnError, IngestError, WriterError, ConfigError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    destination = arguments.output or (
        PROJECT_ROOT / config.report_root / "feasibility-report-v1.json"
    )
    try:
        write_json_atomically(Path(destination), report)
    except OSError as error:
        print(f"error: cannot write report '{destination}': {error}", file=sys.stderr)
        return 1
    print(f"feasibility report written to {destination}")
    for name, result in report["capabilities"]["results"].items():
        print(f"  {name}: {result['outcome']}")
        for requirement in result["unsatisfied_requirements"]:
            print(f"      unsatisfied: {requirement}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
