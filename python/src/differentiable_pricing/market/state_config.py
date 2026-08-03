"""Validated, versioned configuration for the Task 9B reconstruction study.

The configuration is the single declaration of which Task 9A outputs are
consumed and under which digests, which instants are sampled, which strike
windows and fit methods are compared, what the day-count and expiry-time
assumptions are, and where output goes. It is hashed verbatim so the report can
pin the exact assumptions it was produced under. Unknown keys are rejected.

Three declarations exist so this study cannot smuggle a conclusion into its own
source.

``[provenance]``
    Task 9A's configuration digest, report digest and raw-archive manifest
    digest are pinned here and verified before any arithmetic runs. A study that
    reads a processed tree without proving which run produced it has no
    provenance at all, only a directory name.

``settings_status`` / ``settings_provenance``
    Every snapshot time, strike window, minimum count and method here was chosen
    after the three-session archive was observed. The status field says so in
    the report rather than leaving a reader to assume these were predeclared.

``[american.ex_date_schedule]``
    The issuer's published distribution schedule, cited by title and URL. It
    fixes the scheduled dates and nothing else: the parser requires the amount
    and economic-effect statuses to stay ``not_verified``, because a calendar
    carries no cash amount and measures nothing about quoted option prices. The
    dates enter no arithmetic; they label which cross-session transition
    contains the scheduled event.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from differentiable_pricing.market.config import (
    KNOWN_DATE_ROLES,
    ConfigError,
    SessionDate,
    _reject_unknown,
    _require_array_of_tables,
    _require_bool,
    _require_date,
    _require_positive_int,
    _require_relative_path,
    _require_str,
    _require_str_tuple,
    _require_table,
    _require_time,
)
from differentiable_pricing.market.parity import KNOWN_METHODS

SCHEMA_VERSION: Final = "market-state-reconstruction/1"

SETTINGS_STATUS_EXPLORATORY: Final = "exploratory_pilot"
SETTINGS_STATUS_FROZEN: Final = "frozen"
KNOWN_SETTINGS_STATUS: Final = frozenset({SETTINGS_STATUS_EXPLORATORY, SETTINGS_STATUS_FROZEN})

EX_DATE_STATUS_SCHEDULED: Final = "officially_scheduled"
"""The only status the ex-date block may carry.

The dates come from the issuer's own published distribution schedule, which is a
calendar and not a measurement. The label is narrow on purpose: it fixes the
scheduled dates and nothing else. Anything stronger -- a cash amount, or a claim
about how quoted option prices responded -- would need evidence this study does
not have, so the two accompanying status fields are pinned negative.
"""

NOT_VERIFIED: Final = "not_verified"
"""Required value for the amount and economic-effect status fields."""

EXPIRY_TIME_STATUS_ASSUMED: Final = "assumed"
"""The archive carries no settlement metadata, so the expiry instant is assumed."""

KNOWN_SPOT_VARIANTS: Final = ("mid", "bid", "ask")

_TOP_LEVEL_KEYS: Final = frozenset(
    {
        "schema_version",
        "study_version",
        "settings_status",
        "settings_provenance",
        "provenance",
        "dates",
        "snapshots",
        "pairing",
        "strike_windows",
        "fit",
        "time",
        "sensitivity",
        "american",
        "rate_controls",
        "output",
    }
)
_PROVENANCE_KEYS: Final = frozenset(
    {
        "feasibility_config",
        "feasibility_config_sha256",
        "feasibility_report",
        "feasibility_report_sha256",
        "archive_root",
        "source_manifest",
        "source_manifest_sha256",
        "processed_root",
        "expected_feasibility_schema_version",
        "expected_feasibility_audit_version",
        "dividend_source_path",
        "borrow_source_path",
        "corporate_action_source_path",
    }
)
_DATE_KEYS: Final = frozenset({"date", "role"})
_SNAPSHOT_KEYS: Final = frozenset({"time_zone", "label", "times_local", "rationale"})
_PAIRING_KEYS: Final = frozenset(
    {
        "european_option_product",
        "american_option_product",
        "underlying_product",
        "require_standard_root",
        "require_tradable",
        "require_same_minute",
        "tradability_source",
    }
)
_WINDOW_KEYS: Final = frozenset({"anchor_rule", "relative_half_widths"})
_FIT_KEYS: Final = frozenset(
    {
        "methods",
        "weight_definition",
        "weight_caveat",
        "minimum_pairs_per_fit",
        "minimum_pairs_for_american_diagnostic",
        "condition_number_warning",
    }
)
_TIME_KEYS: Final = frozenset(
    {
        "day_count",
        "day_count_note",
        "expiry_time_local",
        "expiry_time_zone",
        "expiry_time_status",
        "expiry_time_provenance",
        "minimum_rate_interpretation_days",
        "zero_dte_rate_interpretation",
        "zero_dte_note",
    }
)
_SENSITIVITY_KEYS: Final = frozenset(
    {
        "across_strike_windows",
        "across_fit_methods",
        "across_snapshots",
        "across_dates",
        "underlying_spot_variants",
        "underlying_spot_caveat",
    }
)
_AMERICAN_KEYS: Final = frozenset(
    {
        "require_exact_expiry_match",
        "matched_expiry_rule",
        "residual_name",
        "forbidden_names",
        "residual_caveat",
        "report_strike_dependence",
        "ex_date_schedule",
    }
)
_EX_DATE_KEYS: Final = frozenset(
    {
        "status",
        "source_title",
        "source_publisher",
        "source_url",
        "applies_to",
        "ex_dates",
        "record_dates",
        "payable_dates",
        "amount_status",
        "economic_effect_status",
        "scope",
    }
)
_RATE_CONTROL_KEYS: Final = frozenset(
    {
        "fred_series",
        "sr3_product",
        "sr3_statistic",
        "sr3_rate_definition",
        "comparison_status",
        "comparison_caveat",
    }
)
_OUTPUT_KEYS: Final = frozenset(
    {
        "report_root",
        "report_filename",
        "table_filename",
        "american_table_filename",
        "figure_subdirectory",
    }
)

REQUIRED_FORBIDDEN_NAMES: Final = (
    "observed dividend",
    "exact dividend present value",
    "borrow rate",
    "exact SPY forward",
    "independent market input",
)
"""Names the American residual may never be given.

Declared in configuration and enforced by the parser so removing one is an
edit to a declaration that states why it is there, not a quiet deletion.
"""

DAY_COUNT_ACT365F: Final = "ACT/365F"


@dataclass(frozen=True, slots=True)
class ProvenanceSpec:
    """Pinned digests and roots of the Task 9A run this study consumes."""

    feasibility_config: str
    feasibility_config_sha256: str
    feasibility_report: str
    feasibility_report_sha256: str
    archive_root: str
    source_manifest: str
    source_manifest_sha256: str
    processed_root: str
    expected_feasibility_schema_version: str
    expected_feasibility_audit_version: str
    dividend_source_path: str
    borrow_source_path: str
    corporate_action_source_path: str


@dataclass(frozen=True, slots=True)
class SnapshotSpec:
    """Fixed intraday instants, declared in exchange-local wall-clock time."""

    time_zone: str
    label: str
    times_local: tuple[dt.time, ...]
    rationale: str

    def instants_utc(self, date: dt.date) -> tuple[dt.datetime, ...]:
        """Return this date's snapshot instants in UTC.

        The conversion runs per date through a real time zone so a
        daylight-saving transition moves the instant rather than misaligning it.
        """
        zone = ZoneInfo(self.time_zone)
        return tuple(
            dt.datetime.combine(date, moment, tzinfo=zone).astimezone(dt.UTC)
            for moment in self.times_local
        )


@dataclass(frozen=True, slots=True)
class PairingSpec:
    """Which products are paired and under which admissibility rules."""

    european_option_product: str
    american_option_product: str
    underlying_product: str
    require_standard_root: bool
    require_tradable: bool
    require_same_minute: bool
    tradability_source: str


@dataclass(frozen=True, slots=True)
class StrikeWindowSpec:
    """The forward anchor rule and the relative half-widths compared."""

    anchor_rule: str
    relative_half_widths: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class FitSpec:
    """Deterministic fit methods, minima and the conditioning warning level."""

    methods: tuple[str, ...]
    weight_definition: str
    weight_caveat: str
    minimum_pairs_per_fit: int
    minimum_pairs_for_american_diagnostic: int
    condition_number_warning: float


@dataclass(frozen=True, slots=True)
class TimeSpec:
    """Day count and the assumed expiry instant, with its provenance."""

    day_count: str
    day_count_note: str
    expiry_time_local: dt.time
    expiry_time_zone: str
    expiry_time_status: str
    expiry_time_provenance: str
    minimum_rate_interpretation_days: float
    zero_dte_rate_interpretation: bool
    zero_dte_note: str

    @property
    def minimum_rate_tau_years(self) -> float:
        """The rate-interpretation floor expressed in ACT/365 years."""
        return self.minimum_rate_interpretation_days / 365.0

    def expiry_instant(self, expiry: dt.date) -> dt.datetime:
        """Return the assumed expiry instant in UTC.

        This is an assumption. The archive's OPRA definitions carry a date
        stamped at midnight UTC and no settlement time, style or AM/PM flag.
        """
        zone = ZoneInfo(self.expiry_time_zone)
        return dt.datetime.combine(expiry, self.expiry_time_local, tzinfo=zone).astimezone(dt.UTC)


@dataclass(frozen=True, slots=True)
class SensitivitySpec:
    """Which axes are reported, and how the spot is perturbed."""

    across_strike_windows: bool
    across_fit_methods: bool
    across_snapshots: bool
    across_dates: bool
    underlying_spot_variants: tuple[str, ...]
    underlying_spot_caveat: str


@dataclass(frozen=True, slots=True)
class ExDateSchedule:
    """The issuer's published distribution schedule: dates, and nothing else.

    This is a calendar, not a measurement. It establishes when a distribution is
    scheduled. It does not carry the cash amount, and it says nothing about
    whether or how the options market reflected the event. The two negative
    status fields keep that boundary explicit, and the parser refuses to let
    either of them claim otherwise.

    The dates are never used to adjust, correct or decompose any number. They
    label which cross-session transition contains the scheduled event.
    """

    status: str
    source_title: str
    source_publisher: str
    source_url: str
    applies_to: tuple[str, ...]
    ex_dates: tuple[dt.date, ...]
    record_dates: tuple[dt.date, ...]
    payable_dates: tuple[dt.date, ...]
    amount_status: str
    economic_effect_status: str
    scope: str

    @property
    def dates_are_officially_scheduled(self) -> bool:
        """Whether the ex-dates come from the issuer's published schedule."""
        return self.status == EX_DATE_STATUS_SCHEDULED

    @property
    def amount_is_verified(self) -> bool:
        """Always false: a published schedule states dates, not cash amounts."""
        return False

    @property
    def economic_effect_is_verified(self) -> bool:
        """Always false: a calendar measures nothing about quoted option prices."""
        return False

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statement of what the schedule does and does not fix."""
        return {
            "status": self.status,
            "source_title": self.source_title,
            "source_publisher": self.source_publisher,
            "source_url": self.source_url,
            "applies_to": list(self.applies_to),
            "ex_dates": [value.isoformat() for value in self.ex_dates],
            "record_dates": [value.isoformat() for value in self.record_dates],
            "payable_dates": [value.isoformat() for value in self.payable_dates],
            "dates_are_officially_scheduled": self.dates_are_officially_scheduled,
            "amount_status": self.amount_status,
            "amount_is_verified": self.amount_is_verified,
            "economic_effect_status": self.economic_effect_status,
            "economic_effect_is_verified": self.economic_effect_is_verified,
            "scope": self.scope,
        }


@dataclass(frozen=True, slots=True)
class AmericanSpec:
    """The SPY diagnostic's scope, its mandatory name and its forbidden names."""

    require_exact_expiry_match: bool
    matched_expiry_rule: str
    residual_name: str
    forbidden_names: tuple[str, ...]
    residual_caveat: str
    report_strike_dependence: bool
    ex_date_schedule: ExDateSchedule


@dataclass(frozen=True, slots=True)
class RateControlSpec:
    """Descriptive rate comparators and the caveat that bounds their reading."""

    fred_series: tuple[str, ...]
    sr3_product: str
    sr3_statistic: str
    sr3_rate_definition: str
    comparison_status: str
    comparison_caveat: str


@dataclass(frozen=True, slots=True)
class OutputSpec:
    """Git-ignored destinations for the report, tables and figures."""

    report_root: str
    report_filename: str
    table_filename: str
    american_table_filename: str
    figure_subdirectory: str


@dataclass(frozen=True, slots=True)
class MarketStateConfig:
    """A fully validated Task 9B configuration plus its provenance hash."""

    schema_version: str
    study_version: str
    settings_status: str
    settings_provenance: str
    provenance: ProvenanceSpec
    dates: tuple[SessionDate, ...]
    snapshots: SnapshotSpec
    pairing: PairingSpec
    strike_windows: StrikeWindowSpec
    fit: FitSpec
    time: TimeSpec
    sensitivity: SensitivitySpec
    american: AmericanSpec
    rate_controls: RateControlSpec
    output: OutputSpec
    source_name: str
    source_sha256: str

    @property
    def settings_are_frozen(self) -> bool:
        """Whether these settings have been frozen ahead of a consumed partition."""
        return self.settings_status == SETTINGS_STATUS_FROZEN

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready statement of every assumption this study made."""
        return {
            "schema_version": self.schema_version,
            "study_version": self.study_version,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "settings_status": self.settings_status,
            "settings_are_frozen": self.settings_are_frozen,
            "settings_provenance": self.settings_provenance,
            "snapshot_time_zone": self.snapshots.time_zone,
            "snapshot_label": self.snapshots.label,
            "snapshot_times_local": [
                moment.isoformat() for moment in self.snapshots.times_local
            ],
            "snapshot_rationale": self.snapshots.rationale,
            "strike_window_anchor_rule": self.strike_windows.anchor_rule,
            "relative_half_widths": list(self.strike_windows.relative_half_widths),
            "fit_methods": list(self.fit.methods),
            "weight_definition": self.fit.weight_definition,
            "weight_caveat": self.fit.weight_caveat,
            "minimum_pairs_per_fit": self.fit.minimum_pairs_per_fit,
            "minimum_pairs_for_american_diagnostic": (
                self.fit.minimum_pairs_for_american_diagnostic
            ),
            "condition_number_warning": self.fit.condition_number_warning,
            "day_count": self.time.day_count,
            "day_count_note": self.time.day_count_note,
            "expiry_time_local": self.time.expiry_time_local.isoformat(),
            "expiry_time_zone": self.time.expiry_time_zone,
            "expiry_time_status": self.time.expiry_time_status,
            "expiry_time_provenance": self.time.expiry_time_provenance,
            "settlement_metadata": (
                "absent from the archive: OPRA definition records in this archive carry no "
                "settlement time, no AM/PM settlement flag, no exercise style and no contract "
                "multiplier"
            ),
            "minimum_rate_interpretation_days": self.time.minimum_rate_interpretation_days,
            "zero_dte_rate_interpretation": self.time.zero_dte_rate_interpretation,
            "zero_dte_note": self.time.zero_dte_note,
            "underlying_spot_variants": list(self.sensitivity.underlying_spot_variants),
            "underlying_spot_caveat": self.sensitivity.underlying_spot_caveat,
            "american_residual_name": self.american.residual_name,
            "american_forbidden_names": list(self.american.forbidden_names),
            "american_residual_caveat": self.american.residual_caveat,
            "ex_date_schedule": self.american.ex_date_schedule.summary(),
            "rate_control_status": self.rate_controls.comparison_status,
            "rate_control_caveat": self.rate_controls.comparison_caveat,
            "sr3_rate_definition": self.rate_controls.sr3_rate_definition,
        }


def load_state_config(path: Path | str) -> MarketStateConfig:
    """Read, hash and validate the Task 9B configuration at ``path``."""
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
    return parse_state_config(document, source_name=config_path.name, source_sha256=digest)


def parse_state_config(
    document: Mapping[str, Any], *, source_name: str, source_sha256: str
) -> MarketStateConfig:
    """Validate an already-parsed TOML document. Unknown keys are rejected."""
    _reject_unknown(document, _TOP_LEVEL_KEYS, "top-level")

    schema_version = _require_str(document, "schema_version", "top-level")
    if schema_version != SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schema_version '{schema_version}'; this study only "
            f"accepts '{SCHEMA_VERSION}'"
        )
    settings_status = _require_str(document, "settings_status", "top-level")
    if settings_status not in KNOWN_SETTINGS_STATUS:
        raise ConfigError(
            f"settings_status '{settings_status}' is not one of "
            f"{sorted(KNOWN_SETTINGS_STATUS)}"
        )

    config = MarketStateConfig(
        schema_version=schema_version,
        study_version=_require_str(document, "study_version", "top-level"),
        settings_status=settings_status,
        settings_provenance=_require_str(document, "settings_provenance", "top-level"),
        provenance=_parse_provenance(_require_table(document, "provenance")),
        dates=_parse_dates(document),
        snapshots=_parse_snapshots(_require_table(document, "snapshots")),
        pairing=_parse_pairing(_require_table(document, "pairing")),
        strike_windows=_parse_windows(_require_table(document, "strike_windows")),
        fit=_parse_fit(_require_table(document, "fit")),
        time=_parse_time(_require_table(document, "time")),
        sensitivity=_parse_sensitivity(_require_table(document, "sensitivity")),
        american=_parse_american(_require_table(document, "american")),
        rate_controls=_parse_rate_controls(_require_table(document, "rate_controls")),
        output=_parse_output(_require_table(document, "output")),
        source_name=source_name,
        source_sha256=source_sha256,
    )
    _validate_cross_cutting(config)
    return config


def _validate_cross_cutting(config: MarketStateConfig) -> None:
    """Reject combinations that are individually valid but jointly incoherent."""
    if config.pairing.european_option_product == config.pairing.american_option_product:
        raise ConfigError(
            "[pairing] european_option_product and american_option_product must differ; "
            "the European reconstruction and the American diagnostic are different studies"
        )
    if not config.pairing.require_same_minute:
        raise ConfigError(
            "[pairing] require_same_minute must be true: put-call parity is an identity "
            "between simultaneous prices, and pairing across minutes fits a different quantity"
        )
    if config.time.zero_dte_rate_interpretation:
        raise ConfigError(
            "[time] zero_dte_rate_interpretation must be false: dividing a discount factor "
            "within one tick of one by a tau of hours reports the quote grid, not a rate"
        )
    if config.time.expiry_time_status != EXPIRY_TIME_STATUS_ASSUMED:
        raise ConfigError(
            f"[time] expiry_time_status must be '{EXPIRY_TIME_STATUS_ASSUMED}': this archive "
            f"carries no settlement metadata, so the expiry instant cannot be observed"
        )
    if config.time.day_count != DAY_COUNT_ACT365F:
        raise ConfigError(
            f"[time] day_count must be '{DAY_COUNT_ACT365F}'; this study implements no other "
            f"day count and must not label its year fractions with one it does not compute"
        )
    declared = set(config.american.forbidden_names)
    missing = [name for name in REQUIRED_FORBIDDEN_NAMES if name not in declared]
    if missing:
        raise ConfigError(
            f"[american] forbidden_names is missing {missing}; the American parity carry "
            f"residual may never be described by any of them"
        )
    if config.american.residual_name in config.american.forbidden_names:
        raise ConfigError(
            "[american] residual_name may not be one of the forbidden names"
        )


def _parse_provenance(table: Mapping[str, Any]) -> ProvenanceSpec:
    _reject_unknown(table, _PROVENANCE_KEYS, "[provenance]")
    return ProvenanceSpec(
        feasibility_config=_require_relative_path(table, "feasibility_config", "[provenance]"),
        feasibility_config_sha256=_require_sha256(
            table, "feasibility_config_sha256", "[provenance]"
        ),
        feasibility_report=_require_relative_path(table, "feasibility_report", "[provenance]"),
        feasibility_report_sha256=_require_sha256(
            table, "feasibility_report_sha256", "[provenance]"
        ),
        archive_root=_require_relative_path(table, "archive_root", "[provenance]"),
        source_manifest=_require_relative_path(table, "source_manifest", "[provenance]"),
        source_manifest_sha256=_require_sha256(table, "source_manifest_sha256", "[provenance]"),
        processed_root=_require_relative_path(table, "processed_root", "[provenance]"),
        expected_feasibility_schema_version=_require_str(
            table, "expected_feasibility_schema_version", "[provenance]"
        ),
        expected_feasibility_audit_version=_require_str(
            table, "expected_feasibility_audit_version", "[provenance]"
        ),
        dividend_source_path=_require_relative_path(
            table, "dividend_source_path", "[provenance]"
        ),
        borrow_source_path=_require_relative_path(table, "borrow_source_path", "[provenance]"),
        corporate_action_source_path=_require_relative_path(
            table, "corporate_action_source_path", "[provenance]"
        ),
    )


def _parse_dates(document: Mapping[str, Any]) -> tuple[SessionDate, ...]:
    entries = _require_array_of_tables(document, "dates")
    dates: list[SessionDate] = []
    seen: set[dt.date] = set()
    for entry in entries:
        _reject_unknown(entry, _DATE_KEYS, "[[dates]]")
        value = _require_date(entry, "date", "[[dates]]")
        role = _require_str(entry, "role", "[[dates]]")
        if role not in KNOWN_DATE_ROLES:
            raise ConfigError(
                f"[[dates]] role '{role}' is not one of {sorted(KNOWN_DATE_ROLES)}"
            )
        if value in seen:
            raise ConfigError(f"[[dates]] lists {value.isoformat()} more than once")
        seen.add(value)
        dates.append(SessionDate(date=value, role=role))
    return tuple(sorted(dates, key=lambda entry: entry.date))


def _parse_snapshots(table: Mapping[str, Any]) -> SnapshotSpec:
    _reject_unknown(table, _SNAPSHOT_KEYS, "[snapshots]")
    zone = _require_str(table, "time_zone", "[snapshots]")
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ConfigError(f"[snapshots] time_zone '{zone}' is not a known zone") from error
    raw_times = table.get("times_local")
    if not isinstance(raw_times, list) or not raw_times:
        raise ConfigError("[snapshots] key 'times_local' must be a non-empty list of times")
    moments: list[dt.time] = []
    for index, _ in enumerate(raw_times):
        moments.append(_require_time({"value": raw_times[index]}, "value", "[snapshots]"))
    if len(set(moments)) != len(moments):
        raise ConfigError("[snapshots] times_local contains a repeated instant")
    return SnapshotSpec(
        time_zone=zone,
        label=_require_str(table, "label", "[snapshots]"),
        times_local=tuple(sorted(moments)),
        rationale=_require_str(table, "rationale", "[snapshots]"),
    )


def _parse_pairing(table: Mapping[str, Any]) -> PairingSpec:
    _reject_unknown(table, _PAIRING_KEYS, "[pairing]")
    return PairingSpec(
        european_option_product=_require_str(table, "european_option_product", "[pairing]"),
        american_option_product=_require_str(table, "american_option_product", "[pairing]"),
        underlying_product=_require_str(table, "underlying_product", "[pairing]"),
        require_standard_root=_require_bool(table, "require_standard_root", "[pairing]"),
        require_tradable=_require_bool(table, "require_tradable", "[pairing]"),
        require_same_minute=_require_bool(table, "require_same_minute", "[pairing]"),
        tradability_source=_require_str(table, "tradability_source", "[pairing]"),
    )


def _parse_windows(table: Mapping[str, Any]) -> StrikeWindowSpec:
    _reject_unknown(table, _WINDOW_KEYS, "[strike_windows]")
    raw = table.get("relative_half_widths")
    if not isinstance(raw, list) or not raw:
        raise ConfigError(
            "[strike_windows] key 'relative_half_widths' must be a non-empty list of numbers"
        )
    widths: list[float] = []
    for item in raw:
        if isinstance(item, bool) or not isinstance(item, int | float):
            raise ConfigError("[strike_windows] relative_half_widths must contain only numbers")
        value = float(item)
        if not 0.0 < value < 1.0:
            raise ConfigError(
                f"[strike_windows] relative_half_width {value} must lie strictly between 0 and 1"
            )
        widths.append(value)
    if len(set(widths)) != len(widths):
        raise ConfigError("[strike_windows] relative_half_widths contains a repeated width")
    return StrikeWindowSpec(
        anchor_rule=_require_str(table, "anchor_rule", "[strike_windows]"),
        relative_half_widths=tuple(sorted(widths)),
    )


def _parse_fit(table: Mapping[str, Any]) -> FitSpec:
    _reject_unknown(table, _FIT_KEYS, "[fit]")
    methods = _require_str_tuple(table, "methods", "[fit]")
    unknown = sorted(set(methods) - set(KNOWN_METHODS))
    if unknown:
        raise ConfigError(
            f"[fit] methods {unknown} are not implemented; expected {list(KNOWN_METHODS)}"
        )
    if len(set(methods)) != len(methods):
        raise ConfigError("[fit] methods contains a repeated method")
    minimum_pairs = _require_positive_int(table, "minimum_pairs_per_fit", "[fit]")
    if minimum_pairs < 3:
        raise ConfigError(
            "[fit] minimum_pairs_per_fit must be at least 3: two points determine the line "
            "exactly, so a two-point fit has no residual and its diagnostics are vacuous"
        )
    warning = table.get("condition_number_warning")
    if isinstance(warning, bool) or not isinstance(warning, int | float) or warning <= 1.0:
        raise ConfigError(
            "[fit] condition_number_warning must be a number greater than 1"
        )
    return FitSpec(
        methods=methods,
        weight_definition=_require_str(table, "weight_definition", "[fit]"),
        weight_caveat=_require_str(table, "weight_caveat", "[fit]"),
        minimum_pairs_per_fit=minimum_pairs,
        minimum_pairs_for_american_diagnostic=_require_positive_int(
            table, "minimum_pairs_for_american_diagnostic", "[fit]"
        ),
        condition_number_warning=float(warning),
    )


def _parse_time(table: Mapping[str, Any]) -> TimeSpec:
    _reject_unknown(table, _TIME_KEYS, "[time]")
    zone = _require_str(table, "expiry_time_zone", "[time]")
    try:
        ZoneInfo(zone)
    except (ZoneInfoNotFoundError, ValueError) as error:
        raise ConfigError(f"[time] expiry_time_zone '{zone}' is not a known zone") from error
    floor = table.get("minimum_rate_interpretation_days")
    if isinstance(floor, bool) or not isinstance(floor, int | float) or floor <= 0.0:
        raise ConfigError("[time] minimum_rate_interpretation_days must be a positive number")
    return TimeSpec(
        day_count=_require_str(table, "day_count", "[time]"),
        day_count_note=_require_str(table, "day_count_note", "[time]"),
        expiry_time_local=_require_time(table, "expiry_time_local", "[time]"),
        expiry_time_zone=zone,
        expiry_time_status=_require_str(table, "expiry_time_status", "[time]"),
        expiry_time_provenance=_require_str(table, "expiry_time_provenance", "[time]"),
        minimum_rate_interpretation_days=float(floor),
        zero_dte_rate_interpretation=_require_bool(table, "zero_dte_rate_interpretation", "[time]"),
        zero_dte_note=_require_str(table, "zero_dte_note", "[time]"),
    )


def _parse_sensitivity(table: Mapping[str, Any]) -> SensitivitySpec:
    _reject_unknown(table, _SENSITIVITY_KEYS, "[sensitivity]")
    variants = _require_str_tuple(table, "underlying_spot_variants", "[sensitivity]")
    unknown = sorted(set(variants) - set(KNOWN_SPOT_VARIANTS))
    if unknown:
        raise ConfigError(
            f"[sensitivity] underlying_spot_variants {unknown} are not implemented; "
            f"expected {list(KNOWN_SPOT_VARIANTS)}"
        )
    return SensitivitySpec(
        across_strike_windows=_require_bool(table, "across_strike_windows", "[sensitivity]"),
        across_fit_methods=_require_bool(table, "across_fit_methods", "[sensitivity]"),
        across_snapshots=_require_bool(table, "across_snapshots", "[sensitivity]"),
        across_dates=_require_bool(table, "across_dates", "[sensitivity]"),
        underlying_spot_variants=variants,
        underlying_spot_caveat=_require_str(table, "underlying_spot_caveat", "[sensitivity]"),
    )


def _parse_american(table: Mapping[str, Any]) -> AmericanSpec:
    _reject_unknown(table, _AMERICAN_KEYS, "[american]")
    where = "[american.ex_date_schedule]"
    schedule_table = _require_table(table, "ex_date_schedule")
    _reject_unknown(schedule_table, _EX_DATE_KEYS, where)
    status = _require_str(schedule_table, "status", where)
    if status != EX_DATE_STATUS_SCHEDULED:
        raise ConfigError(
            f"{where} status must be '{EX_DATE_STATUS_SCHEDULED}': the dates come from the "
            f"issuer's published distribution schedule, which fixes when a distribution is "
            f"scheduled and nothing else"
        )
    for field in ("amount_status", "economic_effect_status"):
        value = _require_str(schedule_table, field, where)
        if value != NOT_VERIFIED:
            raise ConfigError(
                f"{where} {field} must be '{NOT_VERIFIED}': a published calendar states dates. "
                f"It carries no cash amount, and it measures nothing about how quoted option "
                f"prices responded, so neither may be described as verified here"
            )
    ex_dates = _require_dates(schedule_table, "ex_dates", where)
    record_dates = _require_dates(schedule_table, "record_dates", where)
    payable_dates = _require_dates(schedule_table, "payable_dates", where)
    if len(record_dates) != len(ex_dates) or len(payable_dates) != len(ex_dates):
        raise ConfigError(
            f"{where} ex_dates, record_dates and payable_dates must have equal length; each "
            f"scheduled distribution has all three"
        )
    for ex_date, payable in zip(ex_dates, payable_dates, strict=True):
        if payable < ex_date:
            raise ConfigError(
                f"{where} payable date {payable.isoformat()} precedes its ex-date "
                f"{ex_date.isoformat()}"
            )
    require_match = _require_bool(table, "require_exact_expiry_match", "[american]")
    if not require_match:
        raise ConfigError(
            "[american] require_exact_expiry_match must be true: a discount factor from a "
            "neighbouring expiry is a different maturity, and interpolating one here would "
            "make the residual absorb the interpolation error"
        )
    return AmericanSpec(
        require_exact_expiry_match=require_match,
        matched_expiry_rule=_require_str(table, "matched_expiry_rule", "[american]"),
        residual_name=_require_str(table, "residual_name", "[american]"),
        forbidden_names=_require_str_tuple(table, "forbidden_names", "[american]"),
        residual_caveat=_require_str(table, "residual_caveat", "[american]"),
        report_strike_dependence=_require_bool(table, "report_strike_dependence", "[american]"),
        ex_date_schedule=ExDateSchedule(
            status=status,
            source_title=_require_str(schedule_table, "source_title", where),
            source_publisher=_require_str(schedule_table, "source_publisher", where),
            source_url=_require_str(schedule_table, "source_url", where),
            applies_to=_require_str_tuple(schedule_table, "applies_to", where),
            ex_dates=ex_dates,
            record_dates=record_dates,
            payable_dates=payable_dates,
            amount_status=_require_str(schedule_table, "amount_status", where),
            economic_effect_status=_require_str(schedule_table, "economic_effect_status", where),
            scope=_require_str(schedule_table, "scope", where),
        ),
    )


def _require_dates(table: Mapping[str, Any], key: str, where: str) -> tuple[dt.date, ...]:
    """Return a non-empty, ascending tuple of calendar dates."""
    raw = table.get(key)
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{where} key '{key}' must be a non-empty list of dates")
    values = tuple(_require_date({"value": item}, "value", where) for item in raw)
    if list(values) != sorted(values):
        raise ConfigError(f"{where} key '{key}' must be in ascending date order")
    return values


def _parse_rate_controls(table: Mapping[str, Any]) -> RateControlSpec:
    _reject_unknown(table, _RATE_CONTROL_KEYS, "[rate_controls]")
    return RateControlSpec(
        fred_series=_require_str_tuple(table, "fred_series", "[rate_controls]"),
        sr3_product=_require_str(table, "sr3_product", "[rate_controls]"),
        sr3_statistic=_require_str(table, "sr3_statistic", "[rate_controls]"),
        sr3_rate_definition=_require_str(table, "sr3_rate_definition", "[rate_controls]"),
        comparison_status=_require_str(table, "comparison_status", "[rate_controls]"),
        comparison_caveat=_require_str(table, "comparison_caveat", "[rate_controls]"),
    )


def _parse_output(table: Mapping[str, Any]) -> OutputSpec:
    _reject_unknown(table, _OUTPUT_KEYS, "[output]")
    return OutputSpec(
        report_root=_require_relative_path(table, "report_root", "[output]"),
        report_filename=_require_filename(table, "report_filename", "[output]"),
        table_filename=_require_filename(table, "table_filename", "[output]"),
        american_table_filename=_require_filename(table, "american_table_filename", "[output]"),
        figure_subdirectory=_require_relative_path(table, "figure_subdirectory", "[output]"),
    )


def _require_sha256(table: Mapping[str, Any], key: str, where: str) -> str:
    value = _require_str(table, key, where).strip().lower()
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ConfigError(f"{where} key '{key}' must be a 64-character hex SHA-256 digest")
    return value


def _require_filename(table: Mapping[str, Any], key: str, where: str) -> str:
    value = _require_str(table, key, where)
    if Path(value).name != value:
        raise ConfigError(f"{where} key '{key}' must be a bare file name, got '{value}'")
    return value


def snapshot_label(instant: dt.datetime, zone: str) -> str:
    """Return the stable ``HH:MM`` local label used to key a snapshot."""
    return instant.astimezone(ZoneInfo(zone)).strftime("%H:%M")


def sequence_of_floats(values: Sequence[float]) -> tuple[float, ...]:
    """Return a defensive float tuple. Small helper kept for report assembly."""
    return tuple(float(value) for value in values)
