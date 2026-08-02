"""Validated, versioned configuration for the market feasibility audit.

The configuration file is the single declaration of which archive is audited,
which dates carry which role, which products and schemas are expected, what the
trading session is, which external sources are *supposed* to be present, what
"tradable" means, and where output goes. It is hashed verbatim so a report can
pin the exact assumptions it was produced under.

Unknown keys are rejected. A silently ignored key is a silently wrong audit.

Three declarations exist here specifically so the audit cannot bake a conclusion
into its own source:

``[[declared_sources]]``
    Every external input the study needs is declared as a *probe*: a path, a
    kind, and the capability it gates. The audit observes whether that path is
    absent, present but empty, or present with content. Nothing about dividends,
    borrow, curves or corporate actions is a constant in the code.

``[tradability]``
    The predicate separating a quote that could have been traded from one that
    merely exists. Structural requirements (positive finite sizes, a positive
    bid, a strictly positive spread) are fixed; the relative-spread cut is
    deliberately *not* a single number. Several exploratory thresholds are
    declared, all of them are reported, and no capability check consults any of
    them. That claim is about the spread thresholds and nothing else.

``[capabilities]``
    Minima for each scoped capability, each carrying its own declared
    provenance. Two of them are structural properties of the method: a surface
    fit needs several expiries carrying several same-minute call/put pairs, and
    that is true before any archive is opened. The two session-count minima are
    not: they were chosen after this three-session feasibility archive was
    observed, they are prospective pilot targets rather than desk-grade
    requirements, and they must be frozen before any pilot partition is
    consumed. ``session_minima_status`` records which of those two states they
    are in and the report serializes it beside the numbers.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import itertools
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

SCHEMA_VERSION: Final = "market-feasibility/2"

KNOWN_DATE_ROLES: Final = frozenset({"development", "held_out"})
"""Roles a session date may carry.

``held_out`` dates exist so a later experiment has an untouched partition. This
audit only measures them; it selects nothing and tunes nothing.
"""

KNOWN_INSTRUMENT_KINDS: Final = frozenset({"option", "equity", "future"})
KNOWN_SCHEMAS: Final = frozenset({"definition", "cbbo-1m", "bbo-1m", "statistics"})
QUOTE_SCHEMAS: Final = frozenset({"cbbo-1m", "bbo-1m"})
KNOWN_PARQUET_COMPRESSION: Final = frozenset({"zstd", "snappy", "none"})

KNOWN_FEED_CLASSES: Final = frozenset(
    {
        "consolidated_nbbo",
        "partial_venue_consolidated",
        "single_venue",
        "exchange_direct",
    }
)
"""How much of the market a quote feed actually sees.

This is provenance, not quality: an audit that calls a partial-venue
consolidation "the spot" has asserted an NBBO it never observed.
"""

NBBO_GRADE_FEED_CLASSES: Final = frozenset({"consolidated_nbbo"})
"""Feed classes whose top of book may be described as a national best bid/offer."""

KNOWN_REFERENCE_SELECTIONS: Final = frozenset(
    {"single_instrument", "nearest_unexpired_outright"}
)
"""Declared rules for choosing which instrument of a product is *the* reference.

``single_instrument``
    The product must resolve to exactly one quoting instrument (an equity).
``nearest_unexpired_outright``
    Among outright contracts whose expiration is on or after the session date,
    the earliest expiry wins. Multi-leg instruments are never eligible.
"""

KNOWN_SOURCE_KINDS: Final = frozenset({"file", "directory"})
KNOWN_SOURCE_REQUIREMENTS: Final = frozenset({"required", "recommended"})

CAPABILITIES: Final = (
    "archive_integrity",
    "ingestion_feasibility",
    "xsp_european_surface_readiness",
    "spy_american_calibration_readiness",
    "replication_readiness",
)
"""The scoped capability questions this audit answers, in reporting order."""

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "audit_version",
        "archive",
        "output",
        "session",
        "synchronization",
        "dates",
        "products",
        "synchronization_pairs",
        "fred_series",
        "declared_sources",
        "tradability",
        "capabilities",
        "audit",
    }
)
_ARCHIVE_KEYS: Final = frozenset(
    {
        "root",
        "sha256_manifest",
        "request_inventory",
        "expected_manifest_entries",
        "expected_total_files",
        "vendor_condition_filename",
    }
)
_OUTPUT_KEYS: Final = frozenset({"processed_root", "report_root"})
_SESSION_KEYS: Final = frozenset(
    {"name", "time_zone", "open_local", "close_local", "expected_minutes"}
)
_SYNC_KEYS: Final = frozenset({"frequency_seconds", "label"})
_DATE_KEYS: Final = frozenset({"date", "role"})
_PRODUCT_KEYS: Final = frozenset(
    {
        "key",
        "dataset",
        "product",
        "instrument_kind",
        "role",
        "archive_subpath",
        "definition_schema",
        "quote_schema",
        "schemas",
        "settles_on",
        "underlying_product",
        "proxy_underlying_product",
        "proxy_caveat",
        "feed_class",
        "reference_selection",
    }
)
_PAIR_KEYS: Final = frozenset({"name", "option_product", "reference_product", "role"})
_FRED_KEYS: Final = frozenset({"series_id", "path", "role"})
_SOURCE_KEYS: Final = frozenset(
    {"key", "path", "kind", "requirement", "capability", "why", "candidate_source"}
)
_TRADABILITY_KEYS: Final = frozenset(
    {
        "require_positive_sizes",
        "require_positive_bid",
        "require_positive_spread",
        "exclude_vendor_flagged",
        "exclude_stale",
        "exploratory_relative_spread_thresholds",
    }
)
_CAPABILITY_KEYS: Final = frozenset(
    {
        "minimum_sessions",
        "minimum_held_out_sessions",
        "minimum_expiries_with_paired_strikes",
        "minimum_paired_strikes_per_expiry",
        "session_minima_provenance",
        "session_minima_status",
        "structural_minima_provenance",
    }
)

KNOWN_MINIMA_STATUS: Final = frozenset({"provisional_pilot_target", "frozen"})
"""Whether a minimum is still provisional or has been frozen before a pilot run.

``provisional_pilot_target`` is an honest label for a number chosen after
looking at a feasibility archive: useful for planning, not an acceptance
criterion, and not yet binding on anything.
"""
_AUDIT_KEYS: Final = frozenset(
    {
        "read_chunk_bytes",
        "parquet_batch_rows",
        "parquet_compression",
        "moneyness_bucket_edges",
        "maturity_bucket_edges_days",
        "extrapolation_trading_dates",
    }
)


class ConfigError(ValueError):
    """Raised when the audit configuration is missing, malformed, or unusable."""


@dataclass(frozen=True, slots=True)
class SessionDate:
    """One calendar session and the role it plays in the research protocol."""

    date: dt.date
    role: str


@dataclass(frozen=True, slots=True)
class ProductSpec:
    """One acquired product, its schemas, and its reference attribution.

    ``underlying_product`` is the declared product that carries this product's
    underlying. It is deliberately per-product: an audit that reaches for
    whichever product happens to be labelled "the underlying" will silently
    price one instrument's chain against another instrument's spot.

    ``proxy_underlying_product`` is the escape hatch for a product whose real
    underlying was not acquired. Using it is allowed; using it silently is not,
    so ``proxy_caveat`` is mandatory alongside it and both are echoed into the
    report next to every number they touch.
    """

    key: str
    dataset: str
    product: str
    instrument_kind: str
    role: str
    archive_subpath: str
    schemas: tuple[str, ...]
    definition_schema: str | None
    quote_schema: str | None
    settles_on: str
    underlying_product: str | None
    proxy_underlying_product: str | None
    proxy_caveat: str | None
    feed_class: str | None
    reference_selection: str | None

    @property
    def is_nbbo_grade(self) -> bool:
        """Whether this product's quote feed may be described as an NBBO."""
        return self.feed_class in NBBO_GRADE_FEED_CLASSES


@dataclass(frozen=True, slots=True)
class SynchronizationPair:
    """A declared option/reference pair whose minute alignment is audited."""

    name: str
    option_product: str
    reference_product: str
    role: str


@dataclass(frozen=True, slots=True)
class FredSeriesSpec:
    """One FRED series and the role it is allowed to play."""

    series_id: str
    path: str
    role: str


@dataclass(frozen=True, slots=True)
class DeclaredSource:
    """An external input the study needs, declared as an observable probe.

    The audit reports one of three states for each: ``absent`` (the path does
    not exist), ``present_but_empty`` (it exists and holds nothing), or
    ``present`` (it exists and holds content). Those are distinct facts and a
    verdict that cannot tell them apart is not an observation.
    """

    key: str
    path: str
    kind: str
    requirement: str
    capability: str
    why: str
    candidate_source: str


@dataclass(frozen=True, slots=True)
class TradabilitySpec:
    """The versioned predicate separating tradable quotes from mere records.

    The structural requirements are fixed because they are definitional: a book
    with no size on a side is not a book you could have hit. The relative-spread
    cut is *not* fixed, because any single value would be an acceptance
    threshold chosen after the results were observed. Several are declared and
    every one of them is reported.
    """

    require_positive_sizes: bool
    require_positive_bid: bool
    require_positive_spread: bool
    exclude_vendor_flagged: bool
    exclude_stale: bool
    exploratory_relative_spread_thresholds: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class CapabilitySpec:
    """Minima each scoped capability requires, with their declared provenance.

    The surface minima are structural: a fit needs several expiries carrying
    several same-minute call/put pairs, which is true of the method rather than
    of any archive. The session minima are not structural and are not claimed to
    be: they were picked after this feasibility archive was observed and are
    prospective pilot targets until frozen.
    """

    minimum_sessions: int
    minimum_held_out_sessions: int
    minimum_expiries_with_paired_strikes: int
    minimum_paired_strikes_per_expiry: int
    session_minima_provenance: str
    session_minima_status: str
    structural_minima_provenance: str

    @property
    def session_minima_are_frozen(self) -> bool:
        """Whether the session minima have been frozen ahead of a pilot partition."""
        return self.session_minima_status == "frozen"

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statement of every minimum and where it came from."""
        return {
            "minimum_sessions": self.minimum_sessions,
            "minimum_held_out_sessions": self.minimum_held_out_sessions,
            "minimum_expiries_with_paired_strikes": self.minimum_expiries_with_paired_strikes,
            "minimum_paired_strikes_per_expiry": self.minimum_paired_strikes_per_expiry,
            "session_minima_status": self.session_minima_status,
            "session_minima_provenance": self.session_minima_provenance,
            "session_minima_are_frozen": self.session_minima_are_frozen,
            "structural_minima_provenance": self.structural_minima_provenance,
        }


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Regular trading hours declared in exchange-local wall-clock time."""

    name: str
    time_zone: str
    open_local: dt.time
    close_local: dt.time
    expected_minutes: int

    def window_utc(self, date: dt.date) -> tuple[dt.datetime, dt.datetime]:
        """Return the UTC ``(open, close)`` instants for ``date``.

        The conversion is done per date through a real time zone so that
        daylight-saving transitions shift the UTC window instead of silently
        misaligning every subsequent minute.
        """
        zone = ZoneInfo(self.time_zone)
        opened = dt.datetime.combine(date, self.open_local, tzinfo=zone)
        closed = dt.datetime.combine(date, self.close_local, tzinfo=zone)
        if closed <= opened:
            raise ConfigError(
                f"session '{self.name}' closes at or before it opens on {date.isoformat()}"
            )
        return opened.astimezone(dt.UTC), closed.astimezone(dt.UTC)


@dataclass(frozen=True, slots=True)
class AuditSpec:
    """Bounded-memory ingestion knobs and predeclared bucket edges."""

    read_chunk_bytes: int
    parquet_batch_rows: int
    parquet_compression: str
    moneyness_bucket_edges: tuple[float, ...]
    maturity_bucket_edges_days: tuple[int, ...]
    extrapolation_trading_dates: int


@dataclass(frozen=True, slots=True)
class MarketFeasibilityConfig:
    """A fully validated audit configuration plus its provenance hash."""

    schema_version: str
    audit_version: str
    archive_root: str
    sha256_manifest: str
    request_inventory: str
    vendor_condition_filename: str
    expected_manifest_entries: int
    expected_total_files: int
    processed_root: str
    report_root: str
    session: SessionSpec
    synchronization_frequency_seconds: int
    synchronization_label: str
    dates: tuple[SessionDate, ...]
    products: tuple[ProductSpec, ...]
    synchronization_pairs: tuple[SynchronizationPair, ...]
    fred_series: tuple[FredSeriesSpec, ...]
    declared_sources: tuple[DeclaredSource, ...]
    tradability: TradabilitySpec
    capabilities: CapabilitySpec
    audit: AuditSpec
    source_name: str
    source_sha256: str

    def product(self, key: str) -> ProductSpec:
        """Return the product declared under ``key``."""
        for spec in self.products:
            if spec.key == key:
                return spec
        raise KeyError(key)

    def held_out_dates(self) -> tuple[SessionDate, ...]:
        """Return the declared held-out sessions."""
        return tuple(entry for entry in self.dates if entry.role == "held_out")

    def session_minutes_utc(self, date: dt.date) -> tuple[dt.datetime, ...]:
        """Return every synchronization instant of ``date``, in UTC.

        Databento's one-minute schemas stamp each interval at its close, so the
        grid runs from ``open + frequency`` through ``close`` inclusive.
        """
        opened, closed = self.session.window_utc(date)
        step = dt.timedelta(seconds=self.synchronization_frequency_seconds)
        span = closed - opened
        if span % step != dt.timedelta(0):
            raise ConfigError(
                f"session '{self.session.name}' on {date.isoformat()} is not an exact "
                f"multiple of the {self.synchronization_frequency_seconds}s grid"
            )
        count = span // step
        return tuple(opened + step * (index + 1) for index in range(count))


def load_market_config(path: Path | str) -> MarketFeasibilityConfig:
    """Read, hash, and validate the audit configuration at ``path``."""
    config_path = Path(path)
    try:
        raw = config_path.read_bytes()
    except OSError as error:
        raise ConfigError(f"cannot read configuration '{config_path}': {error}") from error

    digest = hashlib.sha256(raw).hexdigest()
    try:
        document = tomllib.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(
            f"configuration '{config_path}' is not valid UTF-8 TOML: {error}"
        ) from error

    return parse_market_config(document, source_name=config_path.name, source_sha256=digest)


def parse_market_config(
    document: Mapping[str, Any],
    *,
    source_name: str,
    source_sha256: str,
) -> MarketFeasibilityConfig:
    """Validate an already-parsed TOML document. Unknown keys are rejected."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")

    schema_version = _require_str(document, "schema_version", "top-level")
    if schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schema_version '{schema_version}'; this audit only "
            f"accepts '{SCHEMA_VERSION}'"
        )
    audit_version = _require_str(document, "audit_version", "top-level")

    archive = _require_table(document, "archive")
    _reject_unknown(archive, _ARCHIVE_KEYS, "[archive]")
    output = _require_table(document, "output")
    _reject_unknown(output, _OUTPUT_KEYS, "[output]")

    session = _parse_session(_require_table(document, "session"))
    sync = _require_table(document, "synchronization")
    _reject_unknown(sync, _SYNC_KEYS, "[synchronization]")
    frequency = _require_positive_int(sync, "frequency_seconds", "[synchronization]")
    label = _require_str(sync, "label", "[synchronization]")

    dates = _parse_dates(document)
    products = _parse_products(document)
    pairs = _parse_pairs(document, products)
    fred = _parse_fred(document)
    sources = _parse_declared_sources(document)
    tradability = _parse_tradability(_require_table(document, "tradability"))
    capabilities = _parse_capabilities(_require_table(document, "capabilities"))
    audit = _parse_audit(_require_table(document, "audit"))

    config = MarketFeasibilityConfig(
        schema_version=schema_version,
        audit_version=audit_version,
        archive_root=_require_relative_path(archive, "root", "[archive]"),
        sha256_manifest=_require_relative_path(archive, "sha256_manifest", "[archive]"),
        request_inventory=_require_relative_path(archive, "request_inventory", "[archive]"),
        vendor_condition_filename=_require_str(
            archive, "vendor_condition_filename", "[archive]"
        ),
        expected_manifest_entries=_require_positive_int(
            archive, "expected_manifest_entries", "[archive]"
        ),
        expected_total_files=_require_positive_int(archive, "expected_total_files", "[archive]"),
        processed_root=_require_relative_path(output, "processed_root", "[output]"),
        report_root=_require_relative_path(output, "report_root", "[output]"),
        session=session,
        synchronization_frequency_seconds=frequency,
        synchronization_label=label,
        dates=dates,
        products=products,
        synchronization_pairs=pairs,
        fred_series=fred,
        declared_sources=sources,
        tradability=tradability,
        capabilities=capabilities,
        audit=audit,
        source_name=source_name,
        source_sha256=source_sha256,
    )

    if config.expected_manifest_entries > config.expected_total_files:
        raise ConfigError(
            "[archive] expected_manifest_entries exceeds expected_total_files; the "
            "manifest cannot cover more files than the archive holds"
        )
    unknown_capabilities = sorted(
        {source.capability for source in sources} - set(CAPABILITIES)
    )
    if unknown_capabilities:
        raise ConfigError(
            f"[[declared_sources]] reference unknown capabilities {unknown_capabilities}; "
            f"expected one of {list(CAPABILITIES)}"
        )
    # Every declared date must produce a whole number of grid minutes, and the
    # declared minute count must match. Checking here means a bad session
    # declaration fails before any file is opened.
    for entry in config.dates:
        minutes = config.session_minutes_utc(entry.date)
        if len(minutes) != session.expected_minutes:
            raise ConfigError(
                f"[session] expected_minutes is {session.expected_minutes} but "
                f"{entry.date.isoformat()} yields {len(minutes)} grid minutes"
            )
    return config


def _parse_session(table: Mapping[str, Any]) -> SessionSpec:
    _reject_unknown(table, _SESSION_KEYS, "[session]")
    time_zone = _require_str(table, "time_zone", "[session]")
    try:
        ZoneInfo(time_zone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ConfigError(f"[session] time_zone '{time_zone}' is not available: {error}") from error
    return SessionSpec(
        name=_require_str(table, "name", "[session]"),
        time_zone=time_zone,
        open_local=_require_time(table, "open_local", "[session]"),
        close_local=_require_time(table, "close_local", "[session]"),
        expected_minutes=_require_positive_int(table, "expected_minutes", "[session]"),
    )


def _parse_dates(document: Mapping[str, Any]) -> tuple[SessionDate, ...]:
    entries = _require_array_of_tables(document, "dates")
    parsed: list[SessionDate] = []
    seen: set[dt.date] = set()
    for index, entry in enumerate(entries):
        where = f"[[dates]][{index}]"
        _reject_unknown(entry, _DATE_KEYS, where)
        date = _require_date(entry, "date", where)
        role = _require_str(entry, "role", where)
        if role not in KNOWN_DATE_ROLES:
            raise ConfigError(
                f"{where} role '{role}' is unknown; expected one of "
                f"{sorted(KNOWN_DATE_ROLES)}"
            )
        if date in seen:
            raise ConfigError(f"{where} repeats date {date.isoformat()}")
        seen.add(date)
        parsed.append(SessionDate(date=date, role=role))
    if not any(entry.role == "held_out" for entry in parsed):
        raise ConfigError("[[dates]] declares no held_out date; the audit needs an untouched one")
    return tuple(sorted(parsed, key=lambda entry: entry.date))


def _parse_products(document: Mapping[str, Any]) -> tuple[ProductSpec, ...]:
    entries = _require_array_of_tables(document, "products")
    parsed: list[ProductSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"[[products]][{index}]"
        _reject_unknown(entry, _PRODUCT_KEYS, where)
        key = _require_str(entry, "key", where)
        if key in seen:
            raise ConfigError(f"{where} repeats product key '{key}'")
        seen.add(key)

        kind = _require_str(entry, "instrument_kind", where)
        if kind not in KNOWN_INSTRUMENT_KINDS:
            raise ConfigError(
                f"{where} instrument_kind '{kind}' is unknown; expected one of "
                f"{sorted(KNOWN_INSTRUMENT_KINDS)}"
            )

        schemas = _require_str_tuple(entry, "schemas", where)
        unknown = sorted(set(schemas) - KNOWN_SCHEMAS)
        if unknown:
            raise ConfigError(f"{where} declares unsupported schemas {unknown}")
        if len(set(schemas)) != len(schemas):
            raise ConfigError(f"{where} repeats a schema")

        definition_schema = _optional_str(entry, "definition_schema", where)
        if definition_schema is not None and definition_schema not in schemas:
            raise ConfigError(
                f"{where} definition_schema '{definition_schema}' is not in its schemas list"
            )
        quote_schema = _optional_str(entry, "quote_schema", where)
        if quote_schema is not None:
            if quote_schema not in schemas:
                raise ConfigError(
                    f"{where} quote_schema '{quote_schema}' is not in its schemas list"
                )
            if quote_schema not in QUOTE_SCHEMAS:
                raise ConfigError(
                    f"{where} quote_schema '{quote_schema}' is not a quote schema; expected "
                    f"one of {sorted(QUOTE_SCHEMAS)}"
                )
        if kind == "option" and definition_schema is None:
            raise ConfigError(
                f"{where} is an option product without a definition_schema; option "
                f"contracts must be resolved from definitions, not from file names"
            )

        feed_class = _optional_str(entry, "feed_class", where)
        if quote_schema is not None and feed_class is None:
            raise ConfigError(
                f"{where} declares a quote_schema but no feed_class; a quote feed whose "
                f"venue coverage is undeclared cannot be described as a market price"
            )
        if feed_class is not None and feed_class not in KNOWN_FEED_CLASSES:
            raise ConfigError(
                f"{where} feed_class '{feed_class}' is unknown; expected one of "
                f"{sorted(KNOWN_FEED_CLASSES)}"
            )

        selection = _optional_str(entry, "reference_selection", where)
        if selection is not None and selection not in KNOWN_REFERENCE_SELECTIONS:
            raise ConfigError(
                f"{where} reference_selection '{selection}' is unknown; expected one of "
                f"{sorted(KNOWN_REFERENCE_SELECTIONS)}"
            )

        underlying = _optional_str(entry, "underlying_product", where)
        proxy = _optional_str(entry, "proxy_underlying_product", where)
        caveat = _optional_str(entry, "proxy_caveat", where)
        if underlying is not None and proxy is not None:
            raise ConfigError(
                f"{where} declares both underlying_product and proxy_underlying_product; "
                f"a product has one reference attribution, not two"
            )
        if proxy is not None and not caveat:
            raise ConfigError(
                f"{where} declares proxy_underlying_product without proxy_caveat; a proxy "
                f"underlying may be used but never silently"
            )
        if caveat is not None and proxy is None:
            raise ConfigError(f"{where} declares proxy_caveat without proxy_underlying_product")

        parsed.append(
            ProductSpec(
                key=key,
                dataset=_require_str(entry, "dataset", where),
                product=_require_str(entry, "product", where),
                instrument_kind=kind,
                role=_require_str(entry, "role", where),
                archive_subpath=_require_relative_path(entry, "archive_subpath", where),
                schemas=schemas,
                definition_schema=definition_schema,
                quote_schema=quote_schema,
                settles_on=_require_str(entry, "settles_on", where),
                underlying_product=underlying,
                proxy_underlying_product=proxy,
                proxy_caveat=caveat,
                feed_class=feed_class,
                reference_selection=selection,
            )
        )

    by_key = {spec.key: spec for spec in parsed}
    for spec in parsed:
        for field_name, referenced in (
            ("underlying_product", spec.underlying_product),
            ("proxy_underlying_product", spec.proxy_underlying_product),
        ):
            if referenced is None:
                continue
            if referenced not in by_key:
                raise ConfigError(
                    f"[[products]] '{spec.key}' {field_name} '{referenced}' is not a "
                    f"declared product"
                )
            target = by_key[referenced]
            if target.quote_schema is None:
                raise ConfigError(
                    f"[[products]] '{spec.key}' {field_name} '{referenced}' has no quote "
                    f"schema, so it cannot supply a reference price"
                )
            if target.reference_selection is None:
                raise ConfigError(
                    f"[[products]] '{referenced}' is used as a reference by '{spec.key}' but "
                    f"declares no reference_selection rule"
                )
    return tuple(parsed)


def _parse_pairs(
    document: Mapping[str, Any], products: Sequence[ProductSpec]
) -> tuple[SynchronizationPair, ...]:
    entries = _require_array_of_tables(document, "synchronization_pairs")
    by_key = {spec.key: spec for spec in products}
    parsed: list[SynchronizationPair] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"[[synchronization_pairs]][{index}]"
        _reject_unknown(entry, _PAIR_KEYS, where)
        name = _require_str(entry, "name", where)
        if name in seen:
            raise ConfigError(f"{where} repeats pair name '{name}'")
        seen.add(name)
        option_key = _require_str(entry, "option_product", where)
        reference_key = _require_str(entry, "reference_product", where)
        referenced = (("option_product", option_key), ("reference_product", reference_key))
        for role_name, key in referenced:
            if key not in by_key:
                raise ConfigError(f"{where} {role_name} '{key}' is not a declared product")
            if by_key[key].quote_schema is None:
                raise ConfigError(
                    f"{where} {role_name} '{key}' has no quote schema, so minute "
                    f"synchronization against it cannot be measured"
                )
        if by_key[reference_key].reference_selection is None:
            raise ConfigError(
                f"{where} reference_product '{reference_key}' declares no "
                f"reference_selection rule"
            )
        parsed.append(
            SynchronizationPair(
                name=name,
                option_product=option_key,
                reference_product=reference_key,
                role=_require_str(entry, "role", where),
            )
        )
    return tuple(parsed)


def _parse_fred(document: Mapping[str, Any]) -> tuple[FredSeriesSpec, ...]:
    entries = _require_array_of_tables(document, "fred_series")
    parsed: list[FredSeriesSpec] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"[[fred_series]][{index}]"
        _reject_unknown(entry, _FRED_KEYS, where)
        series_id = _require_str(entry, "series_id", where)
        if series_id in seen:
            raise ConfigError(f"{where} repeats series_id '{series_id}'")
        seen.add(series_id)
        parsed.append(
            FredSeriesSpec(
                series_id=series_id,
                path=_require_relative_path(entry, "path", where),
                role=_require_str(entry, "role", where),
            )
        )
    return tuple(parsed)


def _parse_declared_sources(document: Mapping[str, Any]) -> tuple[DeclaredSource, ...]:
    entries = _require_array_of_tables(document, "declared_sources")
    parsed: list[DeclaredSource] = []
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        where = f"[[declared_sources]][{index}]"
        _reject_unknown(entry, _SOURCE_KEYS, where)
        key = _require_str(entry, "key", where)
        if key in seen:
            raise ConfigError(f"{where} repeats source key '{key}'")
        seen.add(key)
        kind = _require_str(entry, "kind", where)
        if kind not in KNOWN_SOURCE_KINDS:
            raise ConfigError(
                f"{where} kind '{kind}' is unknown; expected one of {sorted(KNOWN_SOURCE_KINDS)}"
            )
        requirement = _require_str(entry, "requirement", where)
        if requirement not in KNOWN_SOURCE_REQUIREMENTS:
            raise ConfigError(
                f"{where} requirement '{requirement}' is unknown; expected one of "
                f"{sorted(KNOWN_SOURCE_REQUIREMENTS)}"
            )
        parsed.append(
            DeclaredSource(
                key=key,
                path=_require_relative_path(entry, "path", where),
                kind=kind,
                requirement=requirement,
                capability=_require_str(entry, "capability", where),
                why=_require_str(entry, "why", where),
                candidate_source=_require_str(entry, "candidate_source", where),
            )
        )
    return tuple(parsed)


def _parse_tradability(table: Mapping[str, Any]) -> TradabilitySpec:
    _reject_unknown(table, _TRADABILITY_KEYS, "[tradability]")
    thresholds = _require_increasing_floats(
        table, "exploratory_relative_spread_thresholds", "[tradability]"
    )
    if thresholds[0] <= 0.0:
        raise ConfigError(
            "[tradability] exploratory_relative_spread_thresholds must be strictly positive"
        )
    return TradabilitySpec(
        require_positive_sizes=_require_bool(table, "require_positive_sizes", "[tradability]"),
        require_positive_bid=_require_bool(table, "require_positive_bid", "[tradability]"),
        require_positive_spread=_require_bool(table, "require_positive_spread", "[tradability]"),
        exclude_vendor_flagged=_require_bool(table, "exclude_vendor_flagged", "[tradability]"),
        exclude_stale=_require_bool(table, "exclude_stale", "[tradability]"),
        exploratory_relative_spread_thresholds=thresholds,
    )


def _parse_capabilities(table: Mapping[str, Any]) -> CapabilitySpec:
    _reject_unknown(table, _CAPABILITY_KEYS, "[capabilities]")
    minimum_sessions = _require_positive_int(table, "minimum_sessions", "[capabilities]")
    minimum_held_out = _require_positive_int(
        table, "minimum_held_out_sessions", "[capabilities]"
    )
    if minimum_held_out > minimum_sessions:
        raise ConfigError(
            "[capabilities] minimum_held_out_sessions exceeds minimum_sessions"
        )
    status = _require_str(table, "session_minima_status", "[capabilities]")
    if status not in KNOWN_MINIMA_STATUS:
        raise ConfigError(
            f"[capabilities] session_minima_status '{status}' is unknown; expected one of "
            f"{sorted(KNOWN_MINIMA_STATUS)}"
        )
    return CapabilitySpec(
        minimum_sessions=minimum_sessions,
        minimum_held_out_sessions=minimum_held_out,
        minimum_expiries_with_paired_strikes=_require_positive_int(
            table, "minimum_expiries_with_paired_strikes", "[capabilities]"
        ),
        minimum_paired_strikes_per_expiry=_require_positive_int(
            table, "minimum_paired_strikes_per_expiry", "[capabilities]"
        ),
        session_minima_provenance=_require_str(
            table, "session_minima_provenance", "[capabilities]"
        ),
        session_minima_status=status,
        structural_minima_provenance=_require_str(
            table, "structural_minima_provenance", "[capabilities]"
        ),
    )


def _parse_audit(table: Mapping[str, Any]) -> AuditSpec:
    _reject_unknown(table, _AUDIT_KEYS, "[audit]")
    compression = _require_str(table, "parquet_compression", "[audit]")
    if compression not in KNOWN_PARQUET_COMPRESSION:
        raise ConfigError(
            f"[audit] parquet_compression '{compression}' is unsupported; expected one of "
            f"{sorted(KNOWN_PARQUET_COMPRESSION)}"
        )
    moneyness = _require_increasing_floats(table, "moneyness_bucket_edges", "[audit]")
    if moneyness[0] <= 0.0:
        raise ConfigError("[audit] moneyness_bucket_edges must be strictly positive")
    maturities = _require_increasing_ints(table, "maturity_bucket_edges_days", "[audit]")
    if maturities[0] < 0:
        raise ConfigError("[audit] maturity_bucket_edges_days must be non-negative")
    return AuditSpec(
        read_chunk_bytes=_require_positive_int(table, "read_chunk_bytes", "[audit]"),
        parquet_batch_rows=_require_positive_int(table, "parquet_batch_rows", "[audit]"),
        parquet_compression=compression,
        moneyness_bucket_edges=moneyness,
        maturity_bucket_edges_days=maturities,
        extrapolation_trading_dates=_require_positive_int(
            table, "extrapolation_trading_dates", "[audit]"
        ),
    )


def _reject_unknown(table: Mapping[str, Any], allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(table) - allowed)
    if unknown:
        raise ConfigError(f"{where} has unknown keys {unknown}; expected {sorted(allowed)}")


def _require_table(document: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{key}] is missing or is not a table")
    return value


def _require_array_of_tables(document: Mapping[str, Any], key: str) -> Sequence[Mapping[str, Any]]:
    value = document.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"[[{key}]] is missing or empty")
    for entry in value:
        if not isinstance(entry, Mapping):
            raise ConfigError(f"[[{key}]] contains an entry that is not a table")
    return value


def _require_str(table: Mapping[str, Any], key: str, where: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{where} key '{key}' must be a non-empty string")
    return value


def _optional_str(table: Mapping[str, Any], key: str, where: str) -> str | None:
    if key not in table:
        return None
    return _require_str(table, key, where)


def _require_bool(table: Mapping[str, Any], key: str, where: str) -> bool:
    value = table.get(key)
    if not isinstance(value, bool):
        raise ConfigError(f"{where} key '{key}' must be a boolean")
    return value


def _require_relative_path(table: Mapping[str, Any], key: str, where: str) -> str:
    value = _require_str(table, key, where)
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ConfigError(
            f"{where} key '{key}' must be a relative path without '..'; got '{value}'"
        )
    return value


def _require_positive_int(table: Mapping[str, Any], key: str, where: str) -> int:
    value = table.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ConfigError(f"{where} key '{key}' must be a positive integer")
    return value


def _require_time(table: Mapping[str, Any], key: str, where: str) -> dt.time:
    value = table.get(key)
    if isinstance(value, dt.time):
        parsed = value
    elif isinstance(value, str):
        try:
            parsed = dt.time.fromisoformat(value)
        except ValueError as error:
            raise ConfigError(f"{where} key '{key}' is not an ISO 8601 time: {error}") from error
    else:
        raise ConfigError(f"{where} key '{key}' must be an ISO 8601 local time")
    if parsed.tzinfo is not None:
        raise ConfigError(
            f"{where} key '{key}' must be a local wall-clock time without an offset; the "
            f"offset is derived from [session] time_zone per date"
        )
    return parsed


def _require_date(table: Mapping[str, Any], key: str, where: str) -> dt.date:
    value = table.get(key)
    if isinstance(value, dt.datetime):
        raise ConfigError(f"{where} key '{key}' must be a calendar date, not a datetime")
    if isinstance(value, dt.date):
        return value
    if isinstance(value, str):
        try:
            return dt.date.fromisoformat(value)
        except ValueError as error:
            raise ConfigError(f"{where} key '{key}' is not an ISO 8601 date: {error}") from error
    raise ConfigError(f"{where} key '{key}' must be an ISO 8601 date")


def _require_increasing_floats(
    table: Mapping[str, Any], key: str, where: str
) -> tuple[float, ...]:
    value = table.get(key)
    if not isinstance(value, list) or len(value) < 2:
        raise ConfigError(f"{where} key '{key}' must be a list of at least two numbers")
    edges: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ConfigError(f"{where} key '{key}' must contain only numbers")
        edges.append(float(item))
    if any(later <= earlier for earlier, later in itertools.pairwise(edges)):
        raise ConfigError(f"{where} key '{key}' must be strictly increasing")
    return tuple(edges)


def _require_increasing_ints(table: Mapping[str, Any], key: str, where: str) -> tuple[int, ...]:
    value = table.get(key)
    if not isinstance(value, list) or len(value) < 2:
        raise ConfigError(f"{where} key '{key}' must be a list of at least two integers")
    edges: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise ConfigError(f"{where} key '{key}' must contain only integers")
        edges.append(item)
    if any(later <= earlier for earlier, later in itertools.pairwise(edges)):
        raise ConfigError(f"{where} key '{key}' must be strictly increasing")
    return tuple(edges)


def _require_str_tuple(table: Mapping[str, Any], key: str, where: str) -> tuple[str, ...]:
    value = table.get(key)
    if not isinstance(value, list) or not value:
        raise ConfigError(f"{where} key '{key}' must be a non-empty list of strings")
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise ConfigError(f"{where} key '{key}' must contain only non-empty strings")
    return tuple(value)
