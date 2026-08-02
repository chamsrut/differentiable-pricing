"""Resolve tradable contracts from historical definition records.

Contracts are never inferred from a file name. Every instrument the audit
reports comes from a decoded ``definition`` record for the session in question,
and the symbol embedded in that record is cross-checked against the record's own
structured fields. Where the two disagree the instrument is recorded as a
resolution failure rather than being silently repaired: an option whose strike
is ambiguous is worse than an option that is known to be unusable.

Identity, effective time and lifecycle
    An instrument is identified by ``(publisher_id, instrument_id)``. Instrument
    ids are assigned per publisher, so an id alone is not an identity even when
    a single archive happens never to collide.

    Definitions are applied in *effective-time* order (``ts_recv``, then arrival
    order within the file), not in file order, and ``security_update_action``
    is honoured: a ``DELETE`` retires the instrument instead of installing it.
    Both the superseded and the retired counts are reported, because "the last
    line in the file won" is a different statement from "the latest correction
    won".

Standard versus adjusted contracts
    OPRA encodes a non-standard deliverable in the OSI root: ``SPY1``, ``SPY2``
    and so on after a corporate action. A well-formed symbol with an unexpected
    root is therefore an *observation about the contract*, not a parse failure,
    and it is counted separately. Lumping the two together would let one
    adjusted listing look like a decoding defect.

Futures
    Futures are classified by ``instrument_class``. The leg count of a CME
    spread definition is not populated in this dataset, so leg-count filtering
    would silently keep every spread; the instrument class is the field that
    actually distinguishes them. Outrights and multi-leg instruments are kept in
    separate views everywhere downstream.
"""

from __future__ import annotations

import datetime as dt
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Final, NamedTuple

from differentiable_pricing.market.dbn import (
    UNDEF_INT32,
    decode_optional_int,
    decode_price,
    decode_timestamp,
)

INSTRUMENT_CLASS_CALL: Final = "C"
INSTRUMENT_CLASS_PUT: Final = "P"
INSTRUMENT_CLASS_FUTURE: Final = "F"
INSTRUMENT_CLASS_FUTURE_SPREAD: Final = "S"
INSTRUMENT_CLASS_OPTION_SPREAD: Final = "T"
INSTRUMENT_CLASS_MIXED_SPREAD: Final = "M"
INSTRUMENT_CLASS_FX_SPOT: Final = "X"
INSTRUMENT_CLASS_BOND: Final = "B"

OPTION_CLASSES: Final = frozenset({INSTRUMENT_CLASS_CALL, INSTRUMENT_CLASS_PUT})
"""Definition classes that denote an outright option."""

MULTI_LEG_CLASSES: Final = frozenset(
    {INSTRUMENT_CLASS_FUTURE_SPREAD, INSTRUMENT_CLASS_OPTION_SPREAD, INSTRUMENT_CLASS_MIXED_SPREAD}
)
"""Definition classes that denote a spread or other multi-leg instrument."""

SECURITY_UPDATE_DELETE: Final = "D"
"""``security_update_action`` value that retires an instrument."""

OSI_STRIKE_SCALE: Final = 1000
"""OSI encodes the strike as an eight-digit integer of thousandths of a unit."""

REFERENCE_SELECTION_SINGLE: Final = "single_instrument"
REFERENCE_SELECTION_NEAREST_OUTRIGHT: Final = "nearest_unexpired_outright"

_OSI_PATTERN: Final = re.compile(r"^(?P<root>[A-Z0-9 .]{1,6}) *(?P<yy>\d{2})(?P<mm>\d{2})"
                                r"(?P<dd>\d{2})(?P<right>[CP])(?P<strike>\d{8})$")

# OSI two-digit years are unambiguous in practice: listed options do not run
# eighty years out. Anchoring the century here keeps the parse total.
_OSI_CENTURY: Final = 2000


class InstrumentError(ValueError):
    """Raised when a definition record cannot be interpreted at all."""


class InstrumentKey(NamedTuple):
    """The identity of a listed instrument: publisher plus instrument id."""

    publisher_id: int
    instrument_id: int


@dataclass(frozen=True, slots=True)
class OsiSymbol:
    """The contract identity encoded in an OSI option symbol."""

    root: str
    expiration_date: dt.date
    option_type: str
    strike: float


def parse_osi_symbol(raw_symbol: str) -> OsiSymbol:
    """Parse a 21-character OSI option symbol.

    The OSI layout is a six-character space-padded root, ``YYMMDD``, ``C`` or
    ``P``, then an eight-digit strike in thousandths.
    """
    if not isinstance(raw_symbol, str):
        raise InstrumentError(f"OSI symbol must be a string, got {type(raw_symbol).__name__}")
    match = _OSI_PATTERN.match(raw_symbol.rstrip())
    if match is None:
        raise InstrumentError(f"'{raw_symbol}' is not an OSI option symbol")
    year = _OSI_CENTURY + int(match["yy"])
    try:
        expiration = dt.date(year, int(match["mm"]), int(match["dd"]))
    except ValueError as error:
        raise InstrumentError(f"'{raw_symbol}' encodes an impossible date: {error}") from error
    return OsiSymbol(
        root=match["root"].strip(),
        expiration_date=expiration,
        option_type=match["right"],
        strike=int(match["strike"]) / OSI_STRIKE_SCALE,
    )


@dataclass(frozen=True, slots=True)
class OptionContract:
    """A resolved listed option contract.

    ``contract_multiplier`` and ``exercise_style`` are optional because OPRA
    definition records in this archive leave them undefined. They are carried as
    ``None`` rather than defaulted to 100 and 'American': a plausible default
    that is never checked is exactly the kind of silent assumption this audit
    exists to catch.

    ``is_standard_root`` is false when the OSI root differs from the product's
    root, which is how OPRA marks a contract carrying a non-standard deliverable
    after a corporate action.
    """

    instrument_id: int
    raw_symbol: str
    publisher_id: int
    root: str
    is_standard_root: bool
    underlying: str
    option_type: str
    strike: float
    expiration_ns: int
    expiration_date: dt.date
    currency: str
    exchange: str
    underlying_id: int | None
    contract_multiplier: int | None
    unit_of_measure_quantity: float | None
    original_contract_size: int | None
    exercise_style: str | None
    security_update_action: str
    ts_recv_ns: int | None

    @property
    def key(self) -> InstrumentKey:
        """The publisher-scoped identity of this contract."""
        return InstrumentKey(self.publisher_id, self.instrument_id)

    @property
    def is_call(self) -> bool:
        """Whether the contract is a call."""
        return self.option_type == INSTRUMENT_CLASS_CALL

    def time_to_maturity_days(self, session_date: dt.date) -> int:
        """Calendar days from ``session_date`` to expiration.

        Calendar days, not a year fraction: the audit measures data coverage and
        must not silently pick a day-count convention that a later pricing task
        is entitled to choose for itself.
        """
        return (self.expiration_date - session_date).days


@dataclass(frozen=True, slots=True)
class FutureContract:
    """A resolved futures instrument, outright or multi-leg."""

    instrument_id: int
    raw_symbol: str
    publisher_id: int
    asset: str
    instrument_class: str
    expiration_ns: int | None
    expiration_date: dt.date | None
    currency: str
    exchange: str
    min_price_increment: float | None
    unit_of_measure: str
    unit_of_measure_quantity: float | None
    security_update_action: str
    is_outright: bool

    @property
    def key(self) -> InstrumentKey:
        """The publisher-scoped identity of this contract."""
        return InstrumentKey(self.publisher_id, self.instrument_id)

    def is_unexpired_on(self, session_date: dt.date) -> bool:
        """Whether the contract still expires on or after ``session_date``."""
        return self.expiration_date is not None and self.expiration_date >= session_date


@dataclass(frozen=True, slots=True)
class ResolutionFailure:
    """One definition record that could not be turned into a usable contract."""

    instrument_id: int | None
    publisher_id: int | None
    raw_symbol: str
    reason: str
    detail: str


@dataclass(slots=True)
class UniverseCounters:
    """Lifecycle bookkeeping shared by every resolved universe."""

    definition_records: int = 0
    superseded_definitions: int = 0
    deleted_definitions: int = 0
    out_of_order_definitions: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-ready view of the lifecycle counters."""
        return {
            "definition_records": self.definition_records,
            "superseded_by_later_effective_time": self.superseded_definitions,
            "retired_by_delete_action": self.deleted_definitions,
            "arrived_out_of_effective_time_order": self.out_of_order_definitions,
        }


@dataclass(slots=True)
class OptionUniverse:
    """Resolved option contracts for one product and session, keyed by identity."""

    contracts: dict[InstrumentKey, OptionContract] = field(default_factory=dict)
    failures: list[ResolutionFailure] = field(default_factory=list)
    counters: UniverseCounters = field(default_factory=UniverseCounters)

    @classmethod
    def empty(cls) -> OptionUniverse:
        """Return an empty universe."""
        return cls()

    def standard(self) -> dict[InstrumentKey, OptionContract]:
        """Return only contracts whose OSI root matches the declared product."""
        return {key: item for key, item in self.contracts.items() if item.is_standard_root}

    def non_standard(self) -> dict[InstrumentKey, OptionContract]:
        """Return only contracts carrying a non-standard (adjusted) OSI root."""
        return {key: item for key, item in self.contracts.items() if not item.is_standard_root}

    @property
    def definition_records(self) -> int:
        """Number of definition records seen."""
        return self.counters.definition_records


@dataclass(slots=True)
class FutureUniverse:
    """Resolved futures instruments for one product and session, keyed by identity."""

    contracts: dict[InstrumentKey, FutureContract] = field(default_factory=dict)
    failures: list[ResolutionFailure] = field(default_factory=list)
    counters: UniverseCounters = field(default_factory=UniverseCounters)

    @classmethod
    def empty(cls) -> FutureUniverse:
        """Return an empty universe."""
        return cls()

    def outrights(self) -> dict[InstrumentKey, FutureContract]:
        """Return only the outright futures, excluding spreads and other legs."""
        return {key: item for key, item in self.contracts.items() if item.is_outright}

    def multi_leg(self) -> dict[InstrumentKey, FutureContract]:
        """Return only the multi-leg instruments."""
        return {key: item for key, item in self.contracts.items() if not item.is_outright}

    def unexpired_outrights(self, session_date: dt.date) -> dict[InstrumentKey, FutureContract]:
        """Return outrights still expiring on or after ``session_date``."""
        return {
            key: item
            for key, item in self.outrights().items()
            if item.is_unexpired_on(session_date)
        }

    def select_reference(
        self, rule: str, session_date: dt.date
    ) -> tuple[FutureContract | None, str]:
        """Apply a declared reference-selection rule and explain the outcome.

        Returns the chosen contract and a human-readable statement of how it was
        chosen, so the report can name the rule and the instrument rather than
        implying that "the futures price" is a well-defined single number.
        """
        if rule == REFERENCE_SELECTION_NEAREST_OUTRIGHT:
            eligible = self.unexpired_outrights(session_date)
            if not eligible:
                return None, (
                    f"rule '{rule}': no outright contract expires on or after "
                    f"{session_date.isoformat()}"
                )
            chosen = min(
                eligible.values(),
                key=lambda item: (item.expiration_date, item.raw_symbol),
            )
            return chosen, (
                f"rule '{rule}': earliest expiry among {len(eligible)} unexpired outrights "
                f"is {chosen.raw_symbol} expiring {chosen.expiration_date}"
            )
        if rule == REFERENCE_SELECTION_SINGLE:
            if len(self.contracts) == 1:
                only = next(iter(self.contracts.values()))
                return only, f"rule '{rule}': the product's only instrument is {only.raw_symbol}"
            return None, (
                f"rule '{rule}': product resolves {len(self.contracts)} instruments, so a "
                f"single reference is not defined"
            )
        raise InstrumentError(f"unknown reference selection rule '{rule}'")

    @property
    def definition_records(self) -> int:
        """Number of definition records seen."""
        return self.counters.definition_records


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _class_code(record: Any) -> str:
    """Return the single-character instrument class code of a definition record."""
    return str(record.instrument_class).strip()


def _update_action(record: Any) -> str:
    """Return the record's ``security_update_action`` as a single-character code."""
    return _text(getattr(record, "security_update_action", ""))[:1].upper()


def resolve_option_definition(record: Any, *, product_root: str | None = None) -> OptionContract:
    """Turn one OPRA definition record into an :class:`OptionContract`.

    Raises :class:`InstrumentError` when the record's structured fields and its
    OSI symbol disagree, or when a field the audit depends on is absent. A
    well-formed symbol whose root differs from ``product_root`` resolves
    successfully and is marked ``is_standard_root=False``.
    """
    class_code = _class_code(record)
    if class_code not in OPTION_CLASSES:
        raise InstrumentError(
            f"instrument class '{class_code}' is not an outright option "
            f"(expected one of {sorted(OPTION_CLASSES)})"
        )

    raw_symbol = _text(record.raw_symbol)
    osi = parse_osi_symbol(raw_symbol)
    if osi.option_type != class_code:
        raise InstrumentError(
            f"symbol '{raw_symbol}' encodes a {osi.option_type} but the definition "
            f"class is {class_code}"
        )

    strike = decode_price(int(record.strike_price))
    if strike is None:
        raise InstrumentError(f"symbol '{raw_symbol}' has no strike price in its definition")
    # OSI carries three strike decimals, so tolerate exactly that quantum.
    if abs(strike - osi.strike) > 5e-4:
        raise InstrumentError(
            f"symbol '{raw_symbol}' encodes strike {osi.strike} but the definition "
            f"records {strike}"
        )

    expiration_ns = decode_timestamp(int(record.expiration))
    if expiration_ns is None:
        raise InstrumentError(f"symbol '{raw_symbol}' has no expiration in its definition")
    expiration_utc = dt.datetime.fromtimestamp(expiration_ns / 1e9, tz=dt.UTC)
    if expiration_utc.date() != osi.expiration_date:
        raise InstrumentError(
            f"symbol '{raw_symbol}' encodes expiry {osi.expiration_date.isoformat()} but the "
            f"definition expires {expiration_utc.date().isoformat()}"
        )

    underlying = _text(record.underlying) or _text(record.asset)
    root = osi.root
    return OptionContract(
        instrument_id=int(record.instrument_id),
        raw_symbol=raw_symbol,
        publisher_id=int(record.publisher_id),
        root=root,
        is_standard_root=product_root is None or root == product_root,
        underlying=underlying,
        option_type=class_code,
        strike=strike,
        expiration_ns=expiration_ns,
        expiration_date=osi.expiration_date,
        currency=_text(record.currency),
        exchange=_text(record.exchange),
        underlying_id=decode_optional_int(int(record.underlying_id), 0),
        contract_multiplier=decode_optional_int(int(record.contract_multiplier), UNDEF_INT32),
        unit_of_measure_quantity=decode_price(int(record.unit_of_measure_qty)),
        original_contract_size=decode_optional_int(
            int(record.original_contract_size), UNDEF_INT32
        ),
        # OPRA definitions in this archive carry an empty CFI code, which is the
        # only place exercise style would appear. Absent means absent.
        exercise_style=_exercise_style_from_cfi(_text(record.cfi)),
        security_update_action=_update_action(record),
        ts_recv_ns=decode_timestamp(int(record.ts_recv)),
    )


def _exercise_style_from_cfi(cfi: str) -> str | None:
    """Map an ISO 10962 CFI option code to an exercise style, if it carries one.

    For category ``O`` (options) the code is ``O<group><exercise><underlying>
    <delivery><standardisation>``, so the third character is the exercise style:
    ``A`` American, ``E`` European, ``B`` Bermudan. A shorter or unrecognised
    code yields ``None`` rather than a guess.
    """
    if len(cfi) < 3 or cfi[0].upper() != "O":
        return None
    return {"A": "american", "E": "european", "B": "bermudan"}.get(cfi[2].upper())


def resolve_future_definition(record: Any) -> FutureContract:
    """Turn one CME definition record into a :class:`FutureContract`."""
    class_code = _class_code(record)
    if not class_code:
        raise InstrumentError("definition record has no instrument class")
    expiration_ns = decode_timestamp(int(record.expiration))
    expiration_date = (
        None
        if expiration_ns is None
        else dt.datetime.fromtimestamp(expiration_ns / 1e9, tz=dt.UTC).date()
    )
    return FutureContract(
        instrument_id=int(record.instrument_id),
        raw_symbol=_text(record.raw_symbol),
        publisher_id=int(record.publisher_id),
        asset=_text(record.asset),
        instrument_class=class_code,
        expiration_ns=expiration_ns,
        expiration_date=expiration_date,
        currency=_text(record.currency),
        exchange=_text(record.exchange),
        min_price_increment=decode_price(int(record.min_price_increment)),
        unit_of_measure=_text(record.unit_of_measure),
        unit_of_measure_quantity=decode_price(int(record.unit_of_measure_qty)),
        security_update_action=_update_action(record),
        is_outright=class_code == INSTRUMENT_CLASS_FUTURE,
    )


def is_outright_future(record: Any) -> bool:
    """Whether a definition record is an outright future rather than a spread."""
    return _class_code(record) == INSTRUMENT_CLASS_FUTURE


class _EffectiveTimeLedger:
    """Applies definition records in effective-time order with delete handling."""

    __slots__ = ("_counters", "_state")

    def __init__(self, counters: UniverseCounters) -> None:
        self._counters = counters
        self._state: dict[InstrumentKey, tuple[tuple[int, int], str, Any]] = {}

    def offer(
        self, key: InstrumentKey, effective_ns: int | None, order: int, action: str, payload: Any
    ) -> None:
        """Record one definition, keeping the latest effective statement per key."""
        stamp = (effective_ns if effective_ns is not None else 0, order)
        previous = self._state.get(key)
        if previous is not None:
            if previous[0] > stamp:
                # A record that arrived later but is effective earlier does not
                # win. Silently letting file order decide would make a corrected
                # definition depend on how the vendor happened to order the file.
                self._counters.out_of_order_definitions += 1
                return
            if action == SECURITY_UPDATE_DELETE:
                self._counters.deleted_definitions += 1
            else:
                self._counters.superseded_definitions += 1
        elif action == SECURITY_UPDATE_DELETE:
            self._counters.deleted_definitions += 1
        self._state[key] = (stamp, action, payload)

    def resolved(self) -> dict[InstrumentKey, Any]:
        """Return the live instruments: latest statement, delete actions removed."""
        return {
            key: payload
            for key, (_, action, payload) in self._state.items()
            if action != SECURITY_UPDATE_DELETE and payload is not None
        }


def _record_key(record: Any) -> InstrumentKey | None:
    try:
        return InstrumentKey(int(record.publisher_id), int(record.instrument_id))
    except (AttributeError, TypeError, ValueError):
        return None


def build_option_universe(
    records: Iterable[Any],
    *,
    product_root: str | None = None,
) -> OptionUniverse:
    """Resolve every option definition in ``records`` into a keyed universe.

    Records are applied in effective-time order and ``DELETE`` actions retire the
    instrument. A well-formed symbol with an unexpected root is kept and marked
    non-standard; only a record the audit cannot interpret is a failure.
    """
    universe = OptionUniverse.empty()
    ledger = _EffectiveTimeLedger(universe.counters)
    for order, record in enumerate(records):
        universe.counters.definition_records += 1
        key = _record_key(record)
        action = _update_action(record)
        try:
            contract = resolve_option_definition(record, product_root=product_root)
        except InstrumentError as error:
            if action == SECURITY_UPDATE_DELETE and key is not None:
                # A retirement only needs an identity to be actionable.
                ledger.offer(key, decode_timestamp(_maybe_int(record, "ts_recv")), order, action,
                             None)
                continue
            universe.failures.append(
                ResolutionFailure(
                    instrument_id=None if key is None else key.instrument_id,
                    publisher_id=None if key is None else key.publisher_id,
                    raw_symbol=_text(getattr(record, "raw_symbol", "")),
                    reason="option_definition_unresolvable",
                    detail=str(error),
                )
            )
            continue
        ledger.offer(contract.key, contract.ts_recv_ns, order, action, contract)
    universe.contracts = ledger.resolved()
    return universe


def build_future_universe(records: Iterable[Any]) -> FutureUniverse:
    """Resolve every futures definition in ``records`` into a keyed universe."""
    universe = FutureUniverse.empty()
    ledger = _EffectiveTimeLedger(universe.counters)
    for order, record in enumerate(records):
        universe.counters.definition_records += 1
        key = _record_key(record)
        action = _update_action(record)
        try:
            contract = resolve_future_definition(record)
        except (InstrumentError, TypeError, ValueError) as error:
            if action == SECURITY_UPDATE_DELETE and key is not None:
                ledger.offer(key, decode_timestamp(_maybe_int(record, "ts_recv")), order, action,
                             None)
                continue
            universe.failures.append(
                ResolutionFailure(
                    instrument_id=None if key is None else key.instrument_id,
                    publisher_id=None if key is None else key.publisher_id,
                    raw_symbol=_text(getattr(record, "raw_symbol", "")),
                    reason="future_definition_unresolvable",
                    detail=str(error),
                )
            )
            continue
        ledger.offer(
            contract.key, decode_timestamp(_maybe_int(record, "ts_recv")), order, action, contract
        )
    universe.contracts = ledger.resolved()
    return universe


def _maybe_int(record: Any, attribute: str) -> int | None:
    try:
        return int(getattr(record, attribute))
    except (AttributeError, TypeError, ValueError):
        return None


def option_definition_rows(contracts: Iterable[OptionContract]) -> Iterator[dict[str, Any]]:
    """Yield normalized definition rows, in ascending identity order."""
    for contract in sorted(contracts, key=lambda item: (item.publisher_id, item.instrument_id)):
        yield {
            "instrument_id": contract.instrument_id,
            "raw_symbol": contract.raw_symbol,
            "publisher_id": contract.publisher_id,
            "root": contract.root,
            "is_standard_root": contract.is_standard_root,
            "underlying": contract.underlying,
            "option_type": contract.option_type,
            "strike": contract.strike,
            "expiration_ns": contract.expiration_ns,
            "expiration_date": contract.expiration_date,
            "currency": contract.currency,
            "exchange": contract.exchange,
            "underlying_id": contract.underlying_id,
            "contract_multiplier": contract.contract_multiplier,
            "unit_of_measure_quantity": contract.unit_of_measure_quantity,
            "original_contract_size": contract.original_contract_size,
            "exercise_style": contract.exercise_style,
            "security_update_action": contract.security_update_action,
            "definition_ts_recv_ns": contract.ts_recv_ns,
        }


def future_definition_rows(contracts: Iterable[FutureContract]) -> Iterator[dict[str, Any]]:
    """Yield normalized futures definition rows, in ascending identity order."""
    for contract in sorted(contracts, key=lambda item: (item.publisher_id, item.instrument_id)):
        yield {
            "instrument_id": contract.instrument_id,
            "raw_symbol": contract.raw_symbol,
            "publisher_id": contract.publisher_id,
            "asset": contract.asset,
            "instrument_class": contract.instrument_class,
            "is_outright": contract.is_outright,
            "expiration_ns": contract.expiration_ns,
            "expiration_date": contract.expiration_date,
            "currency": contract.currency,
            "exchange": contract.exchange,
            "min_price_increment": contract.min_price_increment,
            "unit_of_measure": contract.unit_of_measure,
            "unit_of_measure_quantity": contract.unit_of_measure_quantity,
            "security_update_action": contract.security_update_action,
        }
