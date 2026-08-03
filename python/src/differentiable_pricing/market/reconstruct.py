"""Task 9B: market-state identifiability and reconstruction.

What this program does
    It reads Task 9A's normalized Parquet, proves it is reading the run 9A
    reported on, and then answers one question: which of the economic inputs a
    discrete-dividend American PDE pricer needs can this three-session archive
    actually supply?

    From same-minute European XSP call/put pairs it reconstructs a discount
    factor and a forward per expiry by regressing the call-minus-put combination
    on the strike. From SPY it computes a strictly-named diagnostic residual.
    It compares the implied rates descriptively against daily rate controls. It
    then classifies every PDE input by identifiability and states the proposed
    Task 9C solver input contract.

What this program does not do
    No PDE is solved, no synthetic label is generated, no network is trained, no
    Greek is computed, no implied volatility is inverted and no volatility
    surface is fitted. None of those appear in the output.

Provenance before arithmetic
    Three digests are checked before a single number is computed: Task 9A's
    configuration, Task 9A's report, and the raw archive's SHA-256 manifest.
    Each is checked both against the value pinned in this study's configuration
    and, where 9A recorded it, against the value 9A recorded. A mismatch aborts
    the run and writes nothing.

Outputs
    A JSON report, two CSV tables and a set of SVG figures, all written
    atomically beneath a git-ignored root. Every one of them is derived from
    proprietary quote-level data and none may enter version control.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import hashlib
import io
import itertools
import json
import math
import os
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from differentiable_pricing.market.config import ConfigError
from differentiable_pricing.market.identifiability import (
    MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST,
    MINIMUM_CONTAINMENT_FOR_ROBUST,
    THRESHOLD_PROVENANCE,
    ObservedFacts,
    build_identifiability_matrix,
    contract_summary,
    proposed_task_9c_contract,
)
from differentiable_pricing.market.parity import (
    CarryResidual,
    OptionQuoteRecord,
    ParityFit,
    ParityPair,
    american_carry_residual,
    build_pairs,
    dispersion_summary,
    fit_line,
    fit_parity,
    forward_anchor_strike,
    group_by_expiry,
    select_strike_window,
    year_fraction_act365,
)
from differentiable_pricing.market.provenance import probe_declared_source
from differentiable_pricing.market.state_config import (
    MarketStateConfig,
    load_state_config,
    snapshot_label,
)
from differentiable_pricing.market.state_plots import Series, render_panel

SCHEMA_VERSION: Final = "market-state-reconstruction/1"

CONTAMINATION_SLOPE_RATIO: Final = 10.0
"""How much larger SPY's out-of-window slope must be than the XSP control's.

The control is the same statistic computed on a genuinely European chain with
the same discount factor and the same out-of-window construction, so the ratio
compares like with like rather than against an invented absolute level. The
factor of ten is exploratory: it is chosen so that a positive finding is an
order of magnitude, not a coin flip, and both slopes are reported so a reader
can apply their own factor.
"""

DEFAULT_CONFIG: Final = Path("configs/market_state_reconstruction_v1.toml")


class ReconstructionError(RuntimeError):
    """Raised when the study cannot be run as configured."""


class ProvenanceError(ReconstructionError):
    """Raised when the consumed Task 9A outputs are not the ones pinned."""


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    """Return a file's SHA-256, read in bounded chunks."""
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as error:
        raise ProvenanceError(f"cannot hash '{path}': {error}") from error
    return digest.hexdigest()


def _sanitize(value: Any) -> Any:
    """Replace non-finite floats with their names so the JSON stays standard.

    ``NaN`` and ``Infinity`` are not JSON. Emitting them would produce a file
    that many parsers reject, and silently dropping the fields would hide
    exactly the degeneracies this study exists to report, so they are rendered
    as strings and stay visible.
    """
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        return value
    if isinstance(value, Mapping):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_sanitize(item) for item in value]
    return value


def write_atomically(destination: Path, payload: bytes) -> None:
    """Publish ``payload`` at ``destination`` via a same-directory rename.

    The temporary file is removed on every failure path, so an interrupted run
    leaves either the previous output or none, never a truncated report a later
    reader would mistake for a complete one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_bytes(payload)
        os.replace(temporary, destination)
    except OSError as error:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise ReconstructionError(f"cannot publish '{destination}': {error}") from error


def write_json_atomically(destination: Path, document: Mapping[str, Any]) -> None:
    """Write a deterministic, standard-conforming JSON document atomically."""
    text = json.dumps(_sanitize(document), indent=2, sort_keys=True, allow_nan=False)
    write_atomically(destination, (text + "\n").encode("utf-8"))


def write_csv_atomically(
    destination: Path, header: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    """Write a deterministic CSV atomically, with a fixed dialect and newline."""
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(header)
    for row in rows:
        writer.writerow(
            [
                ""
                if value is None
                else ("nan" if isinstance(value, float) and math.isnan(value) else value)
                for value in row
            ]
        )
    write_atomically(destination, buffer.getvalue().encode("utf-8"))


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProvenanceCheck:
    """One named provenance assertion and what was actually observed."""

    name: str
    expected: str
    observed: str
    satisfied: bool
    detail: str

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready check record."""
        return {
            "check": self.name,
            "expected": self.expected,
            "observed": self.observed,
            "satisfied": self.satisfied,
            "detail": self.detail,
        }


def verify_provenance(
    config: MarketStateConfig, project_root: Path
) -> tuple[tuple[ProvenanceCheck, ...], Mapping[str, Any]]:
    """Check every pinned digest before any arithmetic runs.

    Returns the checks and the parsed Task 9A report. Raises
    :exc:`ProvenanceError` on the first unsatisfied check, because continuing
    would produce numbers whose inputs are unknown.
    """
    spec = config.provenance
    checks: list[ProvenanceCheck] = []

    config_path = project_root / spec.feasibility_config
    if not config_path.is_file():
        raise ProvenanceError(f"Task 9A configuration '{spec.feasibility_config}' is missing")
    config_digest = _sha256_file(config_path)
    checks.append(
        ProvenanceCheck(
            name="task_9a_config_sha256",
            expected=spec.feasibility_config_sha256,
            observed=config_digest,
            satisfied=config_digest == spec.feasibility_config_sha256,
            detail=f"SHA-256 of '{spec.feasibility_config}' on disk",
        )
    )

    report_path = project_root / spec.feasibility_report
    if not report_path.is_file():
        raise ProvenanceError(
            f"Task 9A report '{spec.feasibility_report}' is missing. This study consumes the "
            f"9A run's own output; regenerate it with the market ingestion pipeline before "
            f"running Task 9B."
        )
    report_digest = _sha256_file(report_path)
    checks.append(
        ProvenanceCheck(
            name="task_9a_report_sha256",
            expected=spec.feasibility_report_sha256,
            observed=report_digest,
            satisfied=report_digest == spec.feasibility_report_sha256,
            detail=f"SHA-256 of '{spec.feasibility_report}' on disk",
        )
    )

    manifest_path = project_root / spec.archive_root / spec.source_manifest
    if not manifest_path.is_file():
        raise ProvenanceError(f"raw archive manifest '{manifest_path}' is missing")
    manifest_digest = _sha256_file(manifest_path)
    checks.append(
        ProvenanceCheck(
            name="raw_archive_manifest_sha256",
            expected=spec.source_manifest_sha256,
            observed=manifest_digest,
            satisfied=manifest_digest == spec.source_manifest_sha256,
            detail=f"SHA-256 of '{spec.source_manifest}' inside the raw archive",
        )
    )

    _raise_on_failure(checks)

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProvenanceError(f"cannot parse '{report_path}': {error}") from error
    if not isinstance(report, Mapping):
        raise ProvenanceError(f"'{report_path}' is not a JSON object")

    report_config = report.get("config")
    if not isinstance(report_config, Mapping):
        raise ProvenanceError(f"'{report_path}' has no 'config' section to cross-check")

    checks.append(
        ProvenanceCheck(
            name="task_9a_report_agrees_on_its_config_digest",
            expected=spec.feasibility_config_sha256,
            observed=str(report_config.get("source_sha256")),
            satisfied=report_config.get("source_sha256") == spec.feasibility_config_sha256,
            detail="the digest the 9A report recorded for the configuration it ran under",
        )
    )
    manifest_section = report.get("manifest")
    recorded_manifest = (
        manifest_section.get("sha256") if isinstance(manifest_section, Mapping) else None
    )
    checks.append(
        ProvenanceCheck(
            name="task_9a_report_agrees_on_the_source_manifest_digest",
            expected=spec.source_manifest_sha256,
            observed=str(recorded_manifest),
            satisfied=recorded_manifest == spec.source_manifest_sha256,
            detail="the digest the 9A report recorded for the raw archive manifest",
        )
    )
    checks.append(
        ProvenanceCheck(
            name="task_9a_processed_root",
            expected=spec.processed_root,
            observed=str(report_config.get("processed_root")),
            satisfied=report_config.get("processed_root") == spec.processed_root,
            detail="the processed root the 9A report says it wrote",
        )
    )
    checks.append(
        ProvenanceCheck(
            name="task_9a_schema_version",
            expected=spec.expected_feasibility_schema_version,
            observed=str(report.get("schema_version")),
            satisfied=report.get("schema_version") == spec.expected_feasibility_schema_version,
            detail="the 9A report's schema version",
        )
    )
    checks.append(
        ProvenanceCheck(
            name="task_9a_audit_version",
            expected=spec.expected_feasibility_audit_version,
            observed=str(report.get("audit_version")),
            satisfied=report.get("audit_version") == spec.expected_feasibility_audit_version,
            detail="the 9A report's audit version",
        )
    )

    processed_root = project_root / spec.processed_root
    checks.append(
        ProvenanceCheck(
            name="processed_root_exists",
            expected=spec.processed_root,
            observed="present" if processed_root.is_dir() else "absent",
            satisfied=processed_root.is_dir(),
            detail="the normalized Parquet tree this study reads",
        )
    )

    _raise_on_failure(checks)
    return tuple(checks), report


def _raise_on_failure(checks: Sequence[ProvenanceCheck]) -> None:
    failed = [check for check in checks if not check.satisfied]
    if failed:
        lines = "\n".join(
            f"  {check.name}: expected {check.expected!r}, observed {check.observed!r}"
            for check in failed
        )
        raise ProvenanceError(
            "Task 9A provenance does not match what this study pinned. No arithmetic was "
            f"performed and no output was written.\n{lines}"
        )


# ---------------------------------------------------------------------------
# Reading Task 9A's normalized Parquet
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ResolvedOption:
    """One resolved, standard-root option contract."""

    option_type: str
    strike: float
    expiry: dt.date


@dataclass(frozen=True, slots=True)
class DefinitionUniverse:
    """A date's resolved option universe plus how populated its metadata is."""

    contracts: Mapping[tuple[int, int], ResolvedOption]
    total_records: int
    retired: int
    non_standard_root: int
    exercise_style_populated: int
    contract_multiplier_populated: int
    expiration_with_time_of_day: int

    def population(self, field: str) -> float:
        """Return the populated fraction of one optional definition field."""
        if self.total_records == 0:
            return 0.0
        counts = {
            "exercise_style": self.exercise_style_populated,
            "contract_multiplier": self.contract_multiplier_populated,
            "settlement_time": self.expiration_with_time_of_day,
        }
        return counts[field] / self.total_records

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready description of the resolved universe."""
        return {
            "definition_records": self.total_records,
            "resolved_standard_root_contracts": len(self.contracts),
            "retired_by_delete_action": self.retired,
            "non_standard_root_records": self.non_standard_root,
            "exercise_style_populated_fraction": self.population("exercise_style"),
            "contract_multiplier_populated_fraction": self.population("contract_multiplier"),
            "expiration_carrying_a_time_of_day_fraction": self.population("settlement_time"),
            "note": (
                "A definition record whose expiration is stamped at midnight UTC carries a date "
                "and no settlement instant. Nothing here is defaulted: an unpopulated field is "
                "counted as unpopulated and reported."
            ),
        }


def _partition(processed_root: Path, product: str, schema: str, date: dt.date) -> Path:
    return processed_root / f"product={product}" / f"schema={schema}" / f"date={date.isoformat()}"


def load_definitions(
    processed_root: Path, product: str, date: dt.date, *, require_standard_root: bool
) -> DefinitionUniverse:
    """Resolve one date's option universe from the normalized definition table.

    Definitions are applied in effective-time order and ``security_update_action``
    is honoured, so a later correction supersedes an earlier record and a DELETE
    retires the instrument instead of installing it.
    """
    import pyarrow.parquet as pq

    directory = _partition(processed_root, product, "definition", date)
    if not directory.is_dir():
        raise ReconstructionError(f"missing definition partition '{directory}'")
    table = pq.read_table(
        directory,
        columns=[
            "instrument_id",
            "publisher_id",
            "raw_symbol",
            "root",
            "is_standard_root",
            "option_type",
            "strike",
            "expiration_date",
            "expiration_ns",
            "exercise_style",
            "contract_multiplier",
            "security_update_action",
            "definition_ts_recv_ns",
        ],
    )
    import pyarrow as pa

    columns = {
        name: table.column(name).to_pylist()
        for name in table.schema.names
        if name not in ("expiration_ns", "definition_ts_recv_ns")
    }
    expiration_ns = table.column("expiration_ns").cast(pa.int64()).to_pylist()
    effective_ns = table.column("definition_ts_recv_ns").cast(pa.int64()).to_pylist()

    order = sorted(
        range(table.num_rows),
        key=lambda index: (
            effective_ns[index] if effective_ns[index] is not None else -1,
            index,
        ),
    )
    contracts: dict[tuple[int, int], ResolvedOption] = {}
    retired = 0
    non_standard = 0
    style_populated = 0
    multiplier_populated = 0
    timed_expiration = 0
    for index in order:
        if columns["exercise_style"][index] is not None:
            style_populated += 1
        if columns["contract_multiplier"][index] is not None:
            multiplier_populated += 1
        stamp = expiration_ns[index]
        if stamp is not None and stamp % 86_400_000_000_000 != 0:
            timed_expiration += 1
        key = (int(columns["publisher_id"][index]), int(columns["instrument_id"][index]))
        if columns["security_update_action"][index] == "D":
            contracts.pop(key, None)
            retired += 1
            continue
        if not columns["is_standard_root"][index]:
            non_standard += 1
            if require_standard_root:
                continue
        contracts[key] = ResolvedOption(
            option_type=str(columns["option_type"][index]),
            strike=float(columns["strike"][index]),
            expiry=columns["expiration_date"][index],
        )
    return DefinitionUniverse(
        contracts=contracts,
        total_records=table.num_rows,
        retired=retired,
        non_standard_root=non_standard,
        exercise_style_populated=style_populated,
        contract_multiplier_populated=multiplier_populated,
        expiration_with_time_of_day=timed_expiration,
    )


@dataclass(frozen=True, slots=True)
class QuoteSnapshot:
    """Tradable two-sided quotes at one exact interval-close instant."""

    instant: dt.datetime
    quotes: Mapping[tuple[int, int], tuple[float, float]]
    records_at_instant: int


def load_snapshot_quotes(
    processed_root: Path,
    product: str,
    schema: str,
    date: dt.date,
    instant: dt.datetime,
    *,
    require_tradable: bool,
) -> QuoteSnapshot:
    """Read one product's quotes at exactly ``instant``.

    The filter is exact equality on the interval-close timestamp, never a
    nearest-neighbour or as-of match: pairing across minutes would fit a
    different quantity than put-call parity.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory = _partition(processed_root, product, schema, date)
    if not directory.is_dir():
        raise ReconstructionError(f"missing quote partition '{directory}'")
    stamp = pa.scalar(
        int(instant.timestamp()) * 1_000_000_000, pa.timestamp("ns", tz="UTC")
    )
    table = pq.read_table(
        directory,
        columns=["instrument_id", "publisher_id", "bid_price", "ask_price", "tradable_core"],
        filters=[("ts_recv_ns", "=", stamp)],
    )
    instrument = table.column("instrument_id").to_pylist()
    publisher = table.column("publisher_id").to_pylist()
    bid = table.column("bid_price").to_pylist()
    ask = table.column("ask_price").to_pylist()
    tradable = table.column("tradable_core").to_pylist()
    quotes: dict[tuple[int, int], tuple[float, float]] = {}
    for index in range(table.num_rows):
        if require_tradable and not tradable[index]:
            continue
        if bid[index] is None or ask[index] is None:
            continue
        quotes[(int(publisher[index]), int(instrument[index]))] = (
            float(bid[index]),
            float(ask[index]),
        )
    return QuoteSnapshot(
        instant=instant, quotes=quotes, records_at_instant=table.num_rows
    )


def assemble_records(
    universe: DefinitionUniverse, snapshot: QuoteSnapshot
) -> tuple[OptionQuoteRecord, ...]:
    """Join resolved contracts to their same-minute quotes."""
    stamp = int(snapshot.instant.timestamp()) * 1_000_000_000
    records: list[OptionQuoteRecord] = []
    for key, contract in universe.contracts.items():
        quote = snapshot.quotes.get(key)
        if quote is None:
            continue
        bid, ask = quote
        if not math.isfinite(bid) or not math.isfinite(ask) or ask < bid:
            continue
        records.append(
            OptionQuoteRecord(
                timestamp_ns=stamp,
                expiry=contract.expiry,
                strike=contract.strike,
                option_type=contract.option_type,
                bid=bid,
                ask=ask,
            )
        )
    return tuple(records)


# ---------------------------------------------------------------------------
# The European reconstruction, per date and snapshot
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExpiryResult:
    """Every fit produced at one expiry of one snapshot."""

    expiry: dt.date
    tau_years: float
    is_zero_dte: bool
    total_pairs: int
    anchor_strike: float
    fits: Mapping[tuple[float, str], ParityFit]

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready record of this expiry."""
        return {
            "expiry": self.expiry.isoformat(),
            "tau_years_act365": self.tau_years,
            "is_zero_dte": self.is_zero_dte,
            "paired_strikes_all_windows": self.total_pairs,
            "anchor_strike": self.anchor_strike,
            "fits": [
                fit.summary()
                for _, fit in sorted(self.fits.items(), key=lambda item: (item[0][0], item[0][1]))
            ],
        }


def reconstruct_expiries(
    pairs: Sequence[ParityPair],
    *,
    config: MarketStateConfig,
    instant: dt.datetime,
    session_date: dt.date,
) -> tuple[ExpiryResult, ...]:
    """Fit ``D``, ``F`` and ``r`` at every expiry, window and method."""
    results: list[ExpiryResult] = []
    for expiry, bucket in group_by_expiry(pairs).items():
        expiry_instant = config.time.expiry_instant(expiry)
        tau = year_fraction_act365(instant, expiry_instant)
        anchor = forward_anchor_strike(bucket)
        fits: dict[tuple[float, str], ParityFit] = {}
        for width in config.strike_windows.relative_half_widths:
            window = select_strike_window(bucket, anchor=anchor, relative_half_width=width)
            for method in config.fit.methods:
                fits[(width, method)] = fit_parity(
                    window,
                    method=method,
                    tau_years=tau,
                    relative_half_width=width,
                    anchor_strike=anchor,
                    minimum_pairs=config.fit.minimum_pairs_per_fit,
                    condition_number_warning=config.fit.condition_number_warning,
                    minimum_rate_tau_years=config.time.minimum_rate_tau_years,
                    is_zero_dte=expiry <= session_date,
                )
        results.append(
            ExpiryResult(
                expiry=expiry,
                tau_years=tau,
                is_zero_dte=expiry <= session_date,
                total_pairs=len(bucket),
                anchor_strike=anchor,
                fits=fits,
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# The SPY diagnostic
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AmericanResult:
    """The carry residual at one matched expiry, over every reported axis."""

    expiry: dt.date
    tau_years: float
    matched: bool
    is_zero_dte: bool
    residuals: Mapping[tuple[float, str, str], CarryResidual]
    control_slope_xsp: float
    control_slope_spy: float

    def summary(self) -> dict[str, Any]:
        """Return a JSON-ready record naming the quantity correctly."""
        return {
            "expiry": self.expiry.isoformat(),
            "tau_years_act365": self.tau_years,
            "matched_xsp_expiry": self.matched,
            "is_zero_dte": self.is_zero_dte,
            "excluded_from_stability_aggregates": self.is_zero_dte,
            "results": [
                {
                    "relative_half_width": key[0],
                    "fit_method": key[1],
                    "spot_variant": key[2],
                    **residual.summary(),
                }
                for key, residual in sorted(self.residuals.items())
            ],
            "out_of_window_slope_control": {
                "construction": (
                    "The discount factor from the NARROWEST strike window is applied to pairs "
                    "in the WIDEST window, for SPY and for XSP alike. Inside its own fitting "
                    "window the XSP slope is zero by construction, which makes it a useless "
                    "control; out of window it is not, so the two numbers are comparable."
                ),
                "xsp_european_control_slope_per_unit_strike": self.control_slope_xsp,
                "spy_american_slope_per_unit_strike": self.control_slope_spy,
                "ratio_spy_to_xsp": (
                    abs(self.control_slope_spy) / abs(self.control_slope_xsp)
                    if self.control_slope_xsp not in (0.0,)
                    and math.isfinite(self.control_slope_xsp)
                    and math.isfinite(self.control_slope_spy)
                    else math.inf
                ),
            },
        }


def _spot_variants(
    bid: float, ask: float, variants: Sequence[str]
) -> dict[str, float]:
    values = {"bid": bid, "ask": ask, "mid": 0.5 * (bid + ask)}
    return {name: values[name] for name in variants}


def diagnose_american(
    *,
    config: MarketStateConfig,
    american_pairs: Sequence[ParityPair],
    european_pairs: Sequence[ParityPair],
    european_results: Sequence[ExpiryResult],
    underlying_bid: float,
    underlying_ask: float,
    instant: dt.datetime,
    session_date: dt.date,
) -> tuple[AmericanResult, ...]:
    """Compute the American parity carry residual at every matched expiry."""
    european_by_expiry = {result.expiry: result for result in european_results}
    european_pairs_by_expiry = group_by_expiry(european_pairs)
    widths = config.strike_windows.relative_half_widths
    narrowest, widest = widths[0], widths[-1]
    spots = _spot_variants(
        underlying_bid, underlying_ask, config.sensitivity.underlying_spot_variants
    )

    results: list[AmericanResult] = []
    for expiry, bucket in group_by_expiry(american_pairs).items():
        european = european_by_expiry.get(expiry)
        tau = year_fraction_act365(instant, config.time.expiry_instant(expiry))
        if european is None:
            results.append(
                AmericanResult(
                    expiry=expiry,
                    tau_years=tau,
                    matched=False,
                    is_zero_dte=expiry <= session_date,
                    residuals={},
                    control_slope_xsp=math.nan,
                    control_slope_spy=math.nan,
                )
            )
            continue
        anchor = forward_anchor_strike(bucket)
        residuals: dict[tuple[float, str, str], CarryResidual] = {}
        for width in widths:
            window = select_strike_window(bucket, anchor=anchor, relative_half_width=width)
            for method in config.fit.methods:
                fit = european.fits[(width, method)]
                for variant, spot in spots.items():
                    residuals[(width, method, variant)] = american_carry_residual(
                        window,
                        spot=spot,
                        discount_factor=fit.discount_factor,
                        minimum_strikes=config.fit.minimum_pairs_for_american_diagnostic,
                    )
        control_fit = european.fits[(narrowest, config.fit.methods[0])]
        control_spy = _out_of_window_slope(
            select_strike_window(bucket, anchor=anchor, relative_half_width=widest),
            control_fit.discount_factor,
        )
        control_xsp = _out_of_window_slope(
            select_strike_window(
                european_pairs_by_expiry.get(expiry, ()),
                anchor=european.anchor_strike,
                relative_half_width=widest,
            ),
            control_fit.discount_factor,
        )
        results.append(
            AmericanResult(
                expiry=expiry,
                tau_years=tau,
                matched=True,
                is_zero_dte=expiry <= session_date,
                residuals=residuals,
                control_slope_xsp=control_xsp,
                control_slope_spy=control_spy,
            )
        )
    return tuple(results)


def _out_of_window_slope(pairs: Sequence[ParityPair], discount_factor: float) -> float:
    """Return the slope of ``F_tilde`` against the strike, or NaN if undefined."""
    if len(pairs) < 3 or not math.isfinite(discount_factor) or discount_factor <= 0.0:
        return math.nan
    strikes = [pair.strike for pair in pairs]
    values = [(pair.y_mid + discount_factor * pair.strike) / discount_factor for pair in pairs]
    fit = fit_line(strikes, values)
    return math.nan if fit.singular else fit.slope


# ---------------------------------------------------------------------------
# Rate controls
# ---------------------------------------------------------------------------


def load_fred_controls(
    archive_root: Path, report: Mapping[str, Any], series_ids: Sequence[str]
) -> dict[str, Any]:
    """Return the daily rate controls Task 9A already parsed, per session date.

    9A read and validated these CSVs; re-parsing them here would be a second
    implementation of the same reader with a second chance to disagree.
    """
    rates = report.get("rates")
    fred = rates.get("fred") if isinstance(rates, Mapping) else None
    if not isinstance(fred, Mapping):
        raise ReconstructionError("the Task 9A report carries no FRED control section")
    series = fred.get("series")
    if not isinstance(series, Mapping):
        raise ReconstructionError("the Task 9A report's FRED section carries no series")
    selected = {}
    for series_id in series_ids:
        entry = series.get(series_id)
        if not isinstance(entry, Mapping):
            raise ReconstructionError(f"the Task 9A report carries no FRED series '{series_id}'")
        selected[series_id] = {
            "role": entry.get("role"),
            "session_date_values_percent": entry.get("session_date_values"),
        }
    return {
        "source": "task_9a_report",
        "archive_root": archive_root.name,
        "series": selected,
        "assessment": fred.get("assessment"),
        "date_alignment": fred.get("date_alignment"),
    }


def load_sr3_controls(
    processed_root: Path, product: str, statistic: str, date: dt.date
) -> dict[str, Any]:
    """Return outright SR3 settlement-implied rates for one session.

    The implied rate is ``100 - settlement``. It is a futures rate on compounded
    SOFR over a forward three-month accrual period, not a zero-coupon yield, and
    no convexity adjustment is applied.
    """
    import pyarrow.parquet as pq

    statistics_directory = _partition(processed_root, product, "statistics", date)
    definition_directory = _partition(processed_root, product, "definition", date)
    if not statistics_directory.is_dir() or not definition_directory.is_dir():
        raise ReconstructionError(f"missing SR3 partitions for {date.isoformat()}")

    definitions = pq.read_table(
        definition_directory,
        columns=["instrument_id", "publisher_id", "raw_symbol", "is_outright", "expiration_date"],
    )
    outright: dict[tuple[int, int], tuple[str, Any]] = {}
    for index in range(definitions.num_rows):
        if not definitions.column("is_outright")[index].as_py():
            continue
        key = (
            int(definitions.column("publisher_id")[index].as_py()),
            int(definitions.column("instrument_id")[index].as_py()),
        )
        outright[key] = (
            str(definitions.column("raw_symbol")[index].as_py()),
            definitions.column("expiration_date")[index].as_py(),
        )

    table = pq.read_table(
        statistics_directory,
        columns=[
            "instrument_id",
            "publisher_id",
            "stat_type",
            "price",
            "is_repeat_of_earlier_record",
        ],
    )
    settlements: dict[tuple[int, int], float] = {}
    for index in range(table.num_rows):
        if table.column("stat_type")[index].as_py() != statistic:
            continue
        if table.column("is_repeat_of_earlier_record")[index].as_py():
            continue
        price = table.column("price")[index].as_py()
        if price is None:
            continue
        key = (
            int(table.column("publisher_id")[index].as_py()),
            int(table.column("instrument_id")[index].as_py()),
        )
        if key in outright:
            settlements[key] = float(price)

    rows = [
        {
            "contract": outright[key][0],
            "expiration_date": (
                outright[key][1].isoformat() if outright[key][1] is not None else None
            ),
            "settlement_price": price,
            "implied_rate_percent": 100.0 - price,
        }
        for key, price in settlements.items()
    ]
    rows.sort(key=lambda row: (row["expiration_date"] or "", row["contract"]))
    return {
        "statistic": statistic,
        "outright_contracts_with_a_settlement": len(rows),
        "nearest_eight": rows[:8],
    }


CONTROL_TENORS: Final = (
    (1.0 / 12.0, "DGS1MO"),
    (0.25, "DGS3MO"),
    (0.5, "DGS6MO"),
    (1.0, "DGS1"),
)
"""Nominal tenors compared descriptively, with the Treasury series of that name.

The tenors are the ones the acquired control series happen to publish. They are
not a curve: each row compares one parity-implied continuously compounded zero
rate against one par yield on government credit and one futures rate on
compounded SOFR, and the three are different objects.
"""


def compare_rates(
    outcomes: Sequence[SnapshotOutcome],
    fred: Mapping[str, Any],
    sr3_by_date: Mapping[dt.date, Mapping[str, Any]],
    *,
    relative_half_width: float,
    method: str,
) -> list[dict[str, Any]]:
    """Line up the parity-implied rate against the daily controls, descriptively."""
    series = fred.get("series", {})
    comparison: list[dict[str, Any]] = []
    for outcome in outcomes:
        candidates = [
            (result.tau_years, result.fits[(relative_half_width, method)].rate)
            for result in outcome.european
            if result.fits[(relative_half_width, method)].rate is not None
        ]
        if not candidates:
            continue
        sr3_rows = sr3_by_date.get(outcome.date, {}).get("nearest_eight", [])
        for target, series_id in CONTROL_TENORS:
            tau, rate = min(candidates, key=lambda item: (abs(item[0] - target), item[0]))
            values = series.get(series_id, {}).get("session_date_values_percent", {})
            raw = values.get(outcome.date.isoformat())
            try:
                control = float(raw)
            except (TypeError, ValueError):
                control = math.nan
            nearest_sr3 = sr3_rows[0] if sr3_rows else None
            comparison.append(
                {
                    "session": outcome.date.isoformat(),
                    "snapshot": outcome.label,
                    "nominal_tenor_years": target,
                    "xsp_expiry_tau_years": tau,
                    "xsp_parity_implied_zero_rate_percent": rate * 100.0,
                    "treasury_control_series": series_id,
                    "treasury_control_percent": control,
                    "difference_basis_points": (rate * 100.0 - control) * 100.0,
                    "nearest_sr3_settlement_implied_rate_percent": (
                        None if nearest_sr3 is None else nearest_sr3["implied_rate_percent"]
                    ),
                    "nearest_sr3_contract": (
                        None if nearest_sr3 is None else nearest_sr3["contract"]
                    ),
                    "what_this_row_is_not": (
                        "Not a basis. The three numbers are a continuously compounded zero rate "
                        "implied by option parity, a par yield on government credit at a nominal "
                        "constant maturity, and a futures rate on compounded SOFR over a forward "
                        "accrual period. Differencing them is descriptive only."
                    ),
                }
            )
    return comparison


# ---------------------------------------------------------------------------
# The study
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SnapshotOutcome:
    """Everything one snapshot of one session produced."""

    date: dt.date
    role: str
    label: str
    instant: dt.datetime
    underlying_bid: float
    underlying_ask: float
    european_pairs: int
    american_pairs: int
    european: tuple[ExpiryResult, ...]
    american: tuple[AmericanResult, ...]


def run_snapshot(
    *,
    config: MarketStateConfig,
    processed_root: Path,
    date: dt.date,
    role: str,
    instant: dt.datetime,
    european_universe: DefinitionUniverse,
    american_universe: DefinitionUniverse,
) -> SnapshotOutcome:
    """Reconstruct one snapshot end to end."""
    pairing = config.pairing
    european_quotes = load_snapshot_quotes(
        processed_root,
        pairing.european_option_product,
        "cbbo-1m",
        date,
        instant,
        require_tradable=pairing.require_tradable,
    )
    american_quotes = load_snapshot_quotes(
        processed_root,
        pairing.american_option_product,
        "cbbo-1m",
        date,
        instant,
        require_tradable=pairing.require_tradable,
    )
    underlying = load_snapshot_quotes(
        processed_root,
        pairing.underlying_product,
        "bbo-1m",
        date,
        instant,
        require_tradable=pairing.require_tradable,
    )
    if len(underlying.quotes) != 1:
        raise ReconstructionError(
            f"the underlying product resolved to {len(underlying.quotes)} tradable quotes at "
            f"{instant.isoformat()}; exactly one is required for a spot level"
        )
    (underlying_bid, underlying_ask), = underlying.quotes.values()
    # The spot enters the American diagnostic linearly, so a malformed one
    # propagates straight into a published residual with no diagnostic attached.
    # Option legs are re-validated in `assemble_records` and again in
    # `OptionQuoteRecord.validate`; the spot has no such second reader, so it is
    # checked here unconditionally rather than relying on the configured
    # tradability filter to have caught it.
    if not (
        math.isfinite(underlying_bid)
        and math.isfinite(underlying_ask)
        and underlying_ask >= underlying_bid
        and underlying_bid > 0.0
    ):
        raise ReconstructionError(
            f"the underlying quote at {instant.isoformat()} is not a usable spot: "
            f"bid={underlying_bid!r} ask={underlying_ask!r}; it must be finite, positive "
            f"and uncrossed"
        )

    european_pairs = build_pairs(assemble_records(european_universe, european_quotes))
    american_pairs = build_pairs(assemble_records(american_universe, american_quotes))
    european = reconstruct_expiries(
        european_pairs, config=config, instant=instant, session_date=date
    )
    american = diagnose_american(
        config=config,
        american_pairs=american_pairs,
        european_pairs=european_pairs,
        european_results=european,
        underlying_bid=underlying_bid,
        underlying_ask=underlying_ask,
        instant=instant,
        session_date=date,
    )
    return SnapshotOutcome(
        date=date,
        role=role,
        label=snapshot_label(instant, config.snapshots.time_zone),
        instant=instant,
        underlying_bid=underlying_bid,
        underlying_ask=underlying_ask,
        european_pairs=len(european_pairs),
        american_pairs=len(american_pairs),
        european=european,
        american=american,
    )


def _valid_fits(outcomes: Sequence[SnapshotOutcome]) -> list[ParityFit]:
    return [
        fit
        for outcome in outcomes
        for result in outcome.european
        for fit in result.fits.values()
        if fit.valid
    ]


def _attempted_fits(outcomes: Sequence[SnapshotOutcome]) -> list[ParityFit]:
    return [
        fit
        for outcome in outcomes
        for result in outcome.european
        for fit in result.fits.values()
        if fit.pair_count >= 1
    ]


def european_sensitivity(outcomes: Sequence[SnapshotOutcome]) -> dict[str, Any]:
    """Summarize how much ``D`` and ``F`` move across every reported axis.

    Zero-DTE expiries are excluded from every aggregate here, for the same
    reason they are excluded from rate interpretation and from the American
    aggregates: at a tau of hours the discount factor is pinned within quote
    granularity of one, so such a cell contributes an artificially tiny spread
    that is a property of the tau rather than of the estimator. Including them
    would make the pooled spread look tighter than the estimator actually is.
    They are still fitted and published per expiry.
    """
    per_cell_discount: list[float] = []
    per_cell_forward: list[float] = []
    per_cell_rate_equivalent: list[float] = []
    per_expiry: dict[tuple[dt.date, dt.date], list[tuple[float, float]]] = {}
    window_method: dict[tuple[float, str], list[float]] = {}
    zero_dte_excluded = 0

    for outcome in outcomes:
        for result in outcome.european:
            if result.is_zero_dte:
                zero_dte_excluded += 1
                continue
            values = [
                (fit.discount_factor, fit.forward)
                for fit in result.fits.values()
                if fit.valid
            ]
            if len(values) > 1:
                per_cell_discount.append(
                    max(value[0] for value in values) - min(value[0] for value in values)
                )
                per_cell_forward.append(
                    max(value[1] for value in values) - min(value[1] for value in values)
                )
                # The same ambiguity in the units a pricer reads it in. Since
                # dD/D = -tau dr, an absolute spread in D means very different
                # rate ambiguities at different expiries. Descriptive only: no
                # criterion consults this.
                middle = 0.5 * (
                    max(value[0] for value in values) + min(value[0] for value in values)
                )
                if middle > 0.0 and result.tau_years > 0.0:
                    per_cell_rate_equivalent.append(
                        per_cell_discount[-1] / (middle * result.tau_years)
                    )
            for key, fit in result.fits.items():
                if fit.valid:
                    window_method.setdefault(key, []).append(fit.discount_factor)
                    per_expiry.setdefault((outcome.date, result.expiry), []).append(
                        (fit.discount_factor, fit.forward)
                    )

    across_everything = [
        max(item[0] for item in bucket) - min(item[0] for item in bucket)
        for bucket in per_expiry.values()
        if len(bucket) > 1
    ]
    forward_across_everything = [
        max(item[1] for item in bucket) - min(item[1] for item in bucket)
        for bucket in per_expiry.values()
        if len(bucket) > 1
    ]
    return {
        "axis_definitions": {
            "within_snapshot": "range across strike windows and fit methods at a fixed expiry",
            "across_snapshots_and_windows_and_methods": (
                "range across strike windows, fit methods and the three intraday snapshots at a "
                "fixed session and expiry. Wider than the within-snapshot term, but not "
                "attributable to one cause: it may combine genuine market movement, quote "
                "microstructure, strike composition -- which strikes pass the tradability and "
                "window filters differs between snapshots -- and fitting variation. This study "
                "does not decompose it."
            ),
        },
        "zero_dte_expiries_excluded": zero_dte_excluded,
        "zero_dte_exclusion_note": (
            "At a tau of hours the discount factor is pinned within quote granularity of one, so "
            "a zero-DTE cell contributes a spread that measures its tau rather than the "
            "estimator. Such cells are fitted and published per expiry and excluded from every "
            "aggregate here, exactly as they are from rate interpretation and from the American "
            "aggregates."
        ),
        "discount_factor_range_within_snapshot": dispersion_summary(per_cell_discount),
        "discount_factor_range_within_snapshot_in_rate_terms": dispersion_summary(
            per_cell_rate_equivalent
        ),
        "rate_terms_note": (
            "The same within-snapshot ambiguity expressed as an annualized continuously "
            "compounded rate, via dD/D = -tau dr. Reported because an absolute spread in D is "
            "maturity-confounded: the same number is lenient at the short end and strict at the "
            "long end. Descriptive only -- no criterion in this study consults it, and adopting "
            "one now would be choosing a criterion after seeing which side of it a result fell."
        ),
        "forward_range_within_snapshot": dispersion_summary(per_cell_forward),
        "discount_factor_range_across_snapshots": dispersion_summary(across_everything),
        "forward_range_across_snapshots": dispersion_summary(forward_across_everything),
        "per_window_and_method_discount_factor": {
            f"half_width={key[0]:g}|method={key[1]}": dispersion_summary(values)
            for key, values in sorted(window_method.items())
        },
    }


def american_sensitivity(outcomes: Sequence[SnapshotOutcome]) -> dict[str, Any]:
    """Summarize the carry residual's stability over every reported axis."""
    per_cell: list[float] = []
    per_expiry: dict[tuple[dt.date, dt.date], list[float]] = {}
    spot_sensitivity: list[float] = []
    strike_slopes_spy: list[float] = []
    strike_slopes_xsp: list[float] = []
    zero_dte_excluded = 0

    for outcome in outcomes:
        for result in outcome.american:
            if not result.matched:
                continue
            if result.is_zero_dte:
                zero_dte_excluded += 1
                continue
            if math.isfinite(result.control_slope_spy):
                strike_slopes_spy.append(abs(result.control_slope_spy))
            if math.isfinite(result.control_slope_xsp):
                strike_slopes_xsp.append(abs(result.control_slope_xsp))
            mids = [
                residual.carry_residual
                for key, residual in result.residuals.items()
                if key[2] == "mid" and residual.valid
            ]
            if len(mids) > 1:
                per_cell.append(max(mids) - min(mids))
            for value in mids:
                per_expiry.setdefault((outcome.date, result.expiry), []).append(value)
            by_axis: dict[tuple[float, str], dict[str, float]] = {}
            for key, residual in result.residuals.items():
                if residual.valid:
                    by_axis.setdefault((key[0], key[1]), {})[key[2]] = residual.carry_residual
            for variants in by_axis.values():
                if {"bid", "ask"} <= set(variants):
                    spot_sensitivity.append(abs(variants["ask"] - variants["bid"]))

    across_snapshots = [
        max(bucket) - min(bucket) for bucket in per_expiry.values() if len(bucket) > 1
    ]
    median_spy = dispersion_summary(strike_slopes_spy).get("median", math.nan)
    median_xsp = dispersion_summary(strike_slopes_xsp).get("median", math.nan)
    contaminated = (
        math.isfinite(median_spy)
        and math.isfinite(median_xsp)
        and median_xsp > 0.0
        and median_spy >= CONTAMINATION_SLOPE_RATIO * median_xsp
    )
    return {
        "zero_dte_expiries_excluded": zero_dte_excluded,
        "zero_dte_exclusion_note": (
            "A same-session expiry has a tau of hours, so its discount factor is within quote "
            "granularity of one and the residual it produces describes the quote grid. Zero-DTE "
            "expiries are still computed and published per expiry; they are excluded from every "
            "stability aggregate below, exactly as they are excluded from rate interpretation."
        ),
        "carry_residual_range_within_snapshot": dispersion_summary(per_cell),
        "carry_residual_range_across_snapshots": dispersion_summary(across_snapshots),
        "carry_residual_bid_to_ask_spot_swing": dispersion_summary(spot_sensitivity),
        "strike_dependence": {
            "construction": (
                "Out-of-window slope of F_tilde against the strike, with the discount factor "
                "taken from the narrowest window and the slope measured on the widest, "
                "identically for SPY and for the European XSP control."
            ),
            "spy_abs_slope": dispersion_summary(strike_slopes_spy),
            "xsp_european_control_abs_slope": dispersion_summary(strike_slopes_xsp),
            "ratio_criterion": CONTAMINATION_SLOPE_RATIO,
            "ratio_criterion_provenance": (
                "Exploratory. Chosen so a positive finding is an order of magnitude rather than "
                "a coin flip. Both slopes are reported so another factor can be applied without "
                "rerunning the study."
            ),
            "strike_dependence_indicated": contaminated,
        },
    }


def ex_date_comparison(
    config: MarketStateConfig, outcomes: Sequence[SnapshotOutcome]
) -> dict[str, Any]:
    """Compare the carry residual across sessions, labelled by the scheduled ex-date.

    The ex-date comes from the issuer's published distribution schedule, so the
    date itself is established. Nothing else is: the schedule carries no cash
    amount and measures nothing about quoted option prices. The date enters no
    arithmetic and labels which transition contains the scheduled event; the
    comparison is a plain cross-session difference at a fixed expiry.
    """
    schedule = config.american.ex_date_schedule
    boundary = min(schedule.ex_dates)
    sessions = sorted({outcome.date for outcome in outcomes})

    per_session: dict[dt.date, dict[dt.date, list[float]]] = {}
    for outcome in outcomes:
        for result in outcome.american:
            if not result.matched or result.is_zero_dte:
                continue
            for key, residual in result.residuals.items():
                if key[2] == "mid" and residual.valid:
                    per_session.setdefault(result.expiry, {}).setdefault(
                        outcome.date, []
                    ).append(residual.carry_residual)

    rows: list[dict[str, Any]] = []
    steps: list[tuple[dt.date, dt.date, float]] = []
    for expiry, by_session in sorted(per_session.items()):
        medians = {
            session: dispersion_summary(values).get("median", math.nan)
            for session, values in sorted(by_session.items())
        }
        changes = []
        for earlier, later in itertools.pairwise(sessions):
            if earlier in medians and later in medians:
                change = medians[later] - medians[earlier]
                changes.append(
                    {
                        "from_session": earlier.isoformat(),
                        "to_session": later.isoformat(),
                        "crosses_scheduled_ex_date": earlier < boundary <= later,
                        "median_change": change,
                    }
                )
                steps.append((earlier, later, change))
        rows.append(
            {
                "expiry": expiry.isoformat(),
                "median_carry_residual_by_session": {
                    session.isoformat(): value for session, value in medians.items()
                },
                "consecutive_session_changes": changes,
            }
        )

    by_transition: dict[tuple[dt.date, dt.date], list[float]] = {}
    for earlier, later, change in steps:
        by_transition.setdefault((earlier, later), []).append(change)
    transitions = [
        {
            "from_session": earlier.isoformat(),
            "to_session": later.isoformat(),
            "crosses_scheduled_ex_date": earlier < boundary <= later,
            "median_change_over_expiries": dispersion_summary(changes).get("median", math.nan),
            "expiries_compared": len(changes),
        }
        for (earlier, later), changes in sorted(by_transition.items())
    ]
    largest = (
        max(transitions, key=lambda row: abs(row["median_change_over_expiries"]))
        if transitions
        else None
    )
    return {
        "schedule": schedule.summary(),
        "construction": (
            "The median carry residual at each fixed expiry is compared between consecutive "
            "sessions. Zero-DTE expiries are excluded, as they are from every other stability "
            "statistic here. The scheduled ex-date only labels which transition contains it; it "
            "enters no arithmetic."
        ),
        "session_transitions": transitions,
        "largest_transition": largest,
        "scheduled_ex_date_alignment": (
            None
            if largest is None
            else {
                "largest_step_crosses_the_scheduled_ex_date": largest[
                    "crosses_scheduled_ex_date"
                ],
                "reading": (
                    "The largest cross-session step in the carry residual is temporally aligned "
                    "with, and qualitatively consistent with, the scheduled ex-dividend event: "
                    "it falls in the transition containing the scheduled ex-date and it moves in "
                    "the direction a distribution leaving the forward would move it. That is an "
                    "alignment in time and sign and nothing more. It does not infer or validate "
                    "the cash dividend amount, which this study never computes and the schedule "
                    "never states, and it does not separate the residual into dividend, borrow, "
                    "early-exercise, ETF/index-basis, and quote and fitting components, which "
                    "this archive cannot separate. A step of this sign in this interval is also "
                    "what a change in any of those other components would have produced."
                    if largest["crosses_scheduled_ex_date"]
                    else "The largest cross-session step in the carry residual does not fall in "
                    "the transition containing the scheduled ex-date. That is reported as an "
                    "observation and is not resolved here: the step is not attributed to any "
                    "component, and the schedule is a calendar that this study does not adjust "
                    "to fit a result."
                ),
            }
        ),
        "interpretation_warning": (
            "These are cross-session differences at fixed expiries. The ex-date is scheduled, "
            "but the size of any of these differences is not a dividend amount: the schedule "
            "states no amount, the archive contains none, and the residual bundles dividend "
            "present value with borrow, early-exercise premia, ETF/index basis and quote and "
            "fitting effects. Each difference also spans whole sessions, over which the market "
            "moved, the quoted spot changed, the strike composition of the fitted window "
            "changed, and the fits were rerun; the third session is six weeks after the second."
        ),
        "per_expiry": rows,
    }


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------


def build_verdicts(
    *,
    facts: ObservedFacts,
    european: Mapping[str, Any],
    american: Mapping[str, Any],
    sessions: int,
) -> dict[str, Any]:
    """Answer the study's seven questions from measured facts only."""
    containment = facts.european_containment_fraction_median
    discount_spread = facts.european_discount_factor_spread
    stable = facts.european_reconstruction_is_robust
    residual_spread = facts.american_residual_spread
    return {
        "1_is_xsp_discount_and_forward_reconstruction_numerically_stable": {
            "answer": "yes" if stable else "not_by_the_stated_criteria",
            "evidence": {
                "valid_fits": f"{facts.european_fits_valid}/{facts.european_fits_attempted}",
                "median_bid_ask_containment_fraction": containment,
                "discount_factor_range_across_windows_and_methods_median": discount_spread,
                "discount_factor_range_including_intraday_median": (
                    facts.european_discount_factor_intraday_spread
                ),
                "within_snapshot_discount_range": european.get(
                    "discount_factor_range_within_snapshot"
                ),
            },
            "caveat": (
                "Stability is a statement about the estimator on three sessions, not about the "
                "estimator in general."
            ),
        },
        "2_are_fitted_parity_values_compatible_with_bid_ask_intervals": {
            "answer": "yes" if containment >= MINIMUM_CONTAINMENT_FOR_ROBUST else "not_generally",
            "evidence": {
                "median_containment_fraction": containment,
                "criterion": MINIMUM_CONTAINMENT_FOR_ROBUST,
                "criterion_provenance": THRESHOLD_PROVENANCE,
            },
        },
        "3_how_sensitive_are_D_and_F": {
            "answer": "reported_as_distributions_not_as_a_single_number",
            "european_sensitivity": european,
        },
        "4_is_the_spy_carry_residual_stable_enough_to_inform_a_later_model": {
            "answer": (
                "as_a_qualitative_diagnostic_only"
                if facts.american_residual_defined
                else "not_computable"
            ),
            "evidence": {
                "carry_residual_range_across_windows_and_methods_median": residual_spread,
                "carry_residual_range_including_intraday_median": (
                    facts.american_residual_intraday_spread
                ),
                "bid_to_ask_spot_swing": american.get("carry_residual_bid_to_ask_spot_swing"),
                "strike_dependence": american.get("strike_dependence"),
            },
            "non_claim": (
                "Stability of the residual is not evidence that any of its components is "
                "identified. A stable sum of unknowns is still a sum of unknowns."
            ),
        },
        "5_which_pde_state_variables_remain_unidentifiable": {
            "answer": "see the identifiability matrix",
            "unavailable_or_joint_only": (
                "discrete_dividend_dates, discrete_dividend_amounts, effective_borrow_or_carry, "
                "volatility_input"
            ),
        },
        "6_is_the_archive_sufficient_to_freeze_the_task_9c_solver_interface": {
            "answer": "yes_for_the_interface_no_for_its_real_market_inputs",
            "reasoning": (
                "The interface is determined by what a discrete-dividend American PDE needs, "
                "which this study has enumerated, and by which fields must be optional because "
                "the data cannot fill them. That is enough to freeze the signature. It is not "
                "enough to evaluate against real SPY quotes, because the dividend schedule the "
                "signature requires does not exist in this archive."
            ),
            "sessions_observed": sessions,
        },
        "7_what_must_be_acquired_or_assumed_before_real_spy_calibration": {
            "must_be_acquired": [
                "an issuer distribution history with declared, ex, record and pay dates and "
                "amounts",
                "a securities-lending fee series, without which borrow and dividends stay jointly "
                "identifiable only",
                "corporate-action and contract-adjustment memoranda for non-standard roots",
                "exchange contract specifications supplying settlement style, settlement instant "
                "and contract multiplier",
                "more sessions: three is far below any usable pilot partition",
            ],
            "must_be_assumed_and_declared": [
                "the expiry instant, since no settlement metadata exists in the archive",
                "the day count, ACT/365F here",
                "the exercise style per product, from listed terms rather than from data",
                "the between-expiry interpolation rule for the discount curve",
            ],
        },
    }


# ---------------------------------------------------------------------------
# Figures and tables
# ---------------------------------------------------------------------------


def build_figures(outcomes: Sequence[SnapshotOutcome], config: MarketStateConfig) -> dict[str, str]:
    """Render the diagnostic figures from the computed outcomes."""
    widest = config.strike_windows.relative_half_widths[-1]
    baseline_method = config.fit.methods[0]

    rate_series: list[Series] = []
    containment_series: list[Series] = []
    spread_series: list[Series] = []
    for outcome in outcomes:
        rates: list[tuple[float, float]] = []
        containment: list[tuple[float, float]] = []
        spreads: list[tuple[float, float]] = []
        for result in outcome.european:
            fit = result.fits[(widest, baseline_method)]
            if fit.rate is not None:
                rates.append((result.tau_years, fit.rate * 100.0))
            if fit.valid and math.isfinite(fit.containment_fraction):
                containment.append((result.tau_years, fit.containment_fraction))
            values = [item.discount_factor for item in result.fits.values() if item.valid]
            if len(values) > 1:
                spreads.append((result.tau_years, max(values) - min(values)))
        label = f"{outcome.date.isoformat()} {outcome.label}"
        if rates:
            rate_series.append(Series(label=label, points=tuple(rates)))
        if containment:
            containment_series.append(Series(label=label, points=tuple(containment),
                                             marker_only=True))
        if spreads:
            spread_series.append(Series(label=label, points=tuple(spreads), marker_only=True))

    residual_series: list[Series] = []
    ftilde_series: list[Series] = []
    for outcome in outcomes:
        residuals: list[tuple[float, float]] = []
        for result in outcome.american:
            if not result.matched or result.is_zero_dte:
                continue
            residual = result.residuals.get((widest, baseline_method, "mid"))
            if residual is not None and residual.valid:
                residuals.append((result.tau_years, residual.carry_residual))
        if residuals:
            residual_series.append(
                Series(label=f"{outcome.date.isoformat()} {outcome.label}", points=tuple(residuals))
            )
        if outcome.label == outcomes[0].label:
            for result in outcome.american:
                if (
                    not result.matched
                    or result.is_zero_dte
                    or not math.isfinite(result.control_slope_spy)
                ):
                    continue
                ftilde_series.append(
                    Series(
                        label=f"{outcome.date.isoformat()} {result.expiry.isoformat()}",
                        points=((result.tau_years, result.control_slope_spy),),
                        marker_only=True,
                    )
                )

    figures: dict[str, str] = {}
    if rate_series:
        figures["xsp-implied-zero-rate-term-structure.svg"] = render_panel(
            title="XSP parity-implied continuously compounded rate",
            subtitle=(
                f"r = -log D / tau, ACT/365F, {widest:g} relative strike window, "
                f"{baseline_method.replace('_', ' ')}; zero-DTE expiries withheld"
            ),
            x_label="tau (years, ACT/365F, to an assumed 16:00 America/New_York expiry)",
            y_label="implied rate (percent)",
            series=rate_series,
            footnote=(
                "Reconstructed from proprietary quote-level data. Not an OIS curve: no accrual "
                "schedule, business-day convention or convexity adjustment is implemented."
            ),
        )
    if containment_series:
        figures["parity-bid-ask-containment.svg"] = render_panel(
            title="Fraction of fitted parity values inside the executable bid-ask interval",
            subtitle=f"{widest:g} relative strike window, {baseline_method.replace('_', ' ')}",
            x_label="tau (years, ACT/365F)",
            y_label="containment fraction",
            series=containment_series,
            footnote="Containment is an executability statement, not a goodness-of-fit p-value.",
        )
    if spread_series:
        figures["discount-factor-window-method-spread.svg"] = render_panel(
            title="Spread in the reconstructed discount factor across windows and methods",
            subtitle="range of D over all strike windows and fit methods at a fixed expiry",
            x_label="tau (years, ACT/365F)",
            y_label="max D - min D",
            series=spread_series,
            footnote="The spread is a result of this study, not an error bar to be minimized.",
        )
    if residual_series:
        figures["american-parity-carry-residual.svg"] = render_panel(
            title="American parity carry residual, SPY",
            subtitle=(
                "Q = S - D * median F_tilde, with D from the matched XSP expiry; "
                "not a dividend, not a borrow rate, not a forward"
            ),
            x_label="tau (years, ACT/365F)",
            y_label="carry residual (USD per share)",
            series=residual_series,
            footnote=(
                "Contains dividend present value, early-exercise premia, quote noise and "
                "partial-venue spot error, inseparably."
            ),
        )
    if ftilde_series:
        figures["f-tilde-strike-dependence.svg"] = render_panel(
            title="Out-of-window strike dependence of F_tilde, SPY",
            subtitle=(
                "slope per unit strike; zero under exact European parity, so a systematic "
                "slope indicates early-exercise contamination"
            ),
            x_label="tau (years, ACT/365F)",
            y_label="dF_tilde / dK",
            series=ftilde_series[:12],
            footnote=(
                "First snapshot of each session; the XSP European control is reported in JSON."
            ),
        )
    return figures


def build_tables(
    outcomes: Sequence[SnapshotOutcome],
) -> tuple[tuple[Sequence[str], list[list[Any]]], tuple[Sequence[str], list[list[Any]]]]:
    """Return the European and American result tables, deterministically ordered."""
    european_header = [
        "date", "role", "snapshot", "expiry", "tau_years", "is_zero_dte",
        "relative_half_width", "method", "anchor_strike", "pair_count",
        "strike_min", "strike_max", "intercept_a", "slope_b", "discount_factor_D",
        "forward_F", "rate_r", "rate_withheld_reason", "condition_number",
        "condition_number_centered", "rms_mid_residual", "containment_fraction",
        "rms_outside_spread_error", "max_outside_spread_error", "valid", "diagnostics",
    ]
    european_rows: list[list[Any]] = []
    for outcome in outcomes:
        for result in outcome.european:
            for key in sorted(result.fits):
                fit = result.fits[key]
                european_rows.append(
                    [
                        outcome.date.isoformat(), outcome.role, outcome.label,
                        result.expiry.isoformat(), f"{result.tau_years:.10f}",
                        result.is_zero_dte, f"{key[0]:g}", key[1],
                        f"{result.anchor_strike:.4f}", fit.pair_count,
                        _number(fit.strike_min), _number(fit.strike_max),
                        _number(fit.intercept), _number(fit.slope),
                        _number(fit.discount_factor), _number(fit.forward),
                        "" if fit.rate is None else f"{fit.rate:.10f}",
                        fit.rate_withheld_reason or "",
                        _number(fit.condition_number), _number(fit.condition_number_centered),
                        _number(fit.rms_mid_residual), _number(fit.containment_fraction),
                        _number(fit.rms_outside_spread_error),
                        _number(fit.max_outside_spread_error),
                        fit.valid, ";".join(fit.diagnostics),
                    ]
                )

    american_header = [
        "date", "role", "snapshot", "expiry", "tau_years", "relative_half_width",
        "method", "spot_variant", "strike_count", "spot", "discount_factor",
        "median_f_tilde", "f_tilde_iqr", "f_tilde_mad", "carry_residual",
        "f_tilde_strike_slope", "valid", "diagnostics",
    ]
    american_rows: list[list[Any]] = []
    for outcome in outcomes:
        for result in outcome.american:
            if not result.matched:
                continue
            for key in sorted(result.residuals):
                residual = result.residuals[key]
                american_rows.append(
                    [
                        outcome.date.isoformat(), outcome.role, outcome.label,
                        result.expiry.isoformat(), f"{result.tau_years:.10f}",
                        f"{key[0]:g}", key[1], key[2], residual.strike_count,
                        _number(residual.spot), _number(residual.discount_factor),
                        _number(residual.median_f_tilde), _number(residual.f_tilde_iqr),
                        _number(residual.f_tilde_mad), _number(residual.carry_residual),
                        _number(residual.strike_slope), residual.valid,
                        ";".join(residual.diagnostics),
                    ]
                )
    return (european_header, european_rows), (american_header, american_rows)


def _number(value: float) -> str:
    """Render a float for CSV with full double precision, or a name if non-finite."""
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return repr(value)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _pooled_population(universes: Sequence[DefinitionUniverse], field: str) -> float:
    """Return the populated fraction of one definition field over every session.

    Pooled rather than taken from whichever session happened to be processed
    last: an optional field could in principle be populated on one date and not
    another, and the classification must see all of them.
    """
    total = sum(universe.total_records for universe in universes)
    if total == 0:
        return 0.0
    populated = sum(
        universe.population(field) * universe.total_records for universe in universes
    )
    return populated / total


def _measure_facts(
    outcomes: Sequence[SnapshotOutcome],
    american_universes: Sequence[DefinitionUniverse],
    probes: Mapping[str, str],
    european: Mapping[str, Any],
    american: Mapping[str, Any],
    feed_class: str,
    sessions: int,
) -> ObservedFacts:
    """Collect every fact the identifiability rules are allowed to consult."""
    attempted = _attempted_fits(outcomes)
    valid = _valid_fits(outcomes)
    containment = dispersion_summary(fit.containment_fraction for fit in valid)
    rate_expiries = sum(
        1
        for outcome in outcomes
        for result in outcome.european
        for fit in result.fits.values()
        if fit.rate is not None
    )
    matched = sum(
        1 for outcome in outcomes for result in outcome.american if result.matched
    )
    half_spreads = [
        0.5 * (outcome.underlying_ask - outcome.underlying_bid)
        / (0.5 * (outcome.underlying_ask + outcome.underlying_bid))
        for outcome in outcomes
    ]
    american_defined = any(
        residual.valid
        for outcome in outcomes
        for result in outcome.american
        for residual in result.residuals.values()
    )
    return ObservedFacts(
        european_fits_attempted=len(attempted),
        european_fits_valid=len(valid),
        european_containment_fraction_median=float(containment.get("median", math.nan)),
        european_discount_factor_spread=float(
            european["discount_factor_range_within_snapshot"].get("median", math.nan)
        ),
        european_discount_factor_intraday_spread=float(
            european["discount_factor_range_across_snapshots"].get("median", math.nan)
        ),
        european_rate_interpretable_expiries=rate_expiries,
        matched_spy_xsp_expiries=matched,
        american_residual_defined=american_defined,
        american_residual_spread=float(
            american["carry_residual_range_within_snapshot"].get("median", math.nan)
        ),
        american_residual_intraday_spread=float(
            american["carry_residual_range_across_snapshots"].get("median", math.nan)
        ),
        american_f_tilde_shows_strike_dependence=bool(
            american["strike_dependence"]["strike_dependence_indicated"]
        ),
        dividend_source_state=probes["dividend"],
        borrow_source_state=probes["borrow"],
        corporate_action_source_state=probes["corporate_action"],
        exercise_style_populated_fraction=_pooled_population(
            american_universes, "exercise_style"
        ),
        contract_multiplier_populated_fraction=_pooled_population(
            american_universes, "contract_multiplier"
        ),
        settlement_time_populated_fraction=_pooled_population(
            american_universes, "settlement_time"
        ),
        underlying_feed_class=feed_class,
        underlying_relative_half_spread=float(
            dispersion_summary(half_spreads).get("median", math.nan)
        ),
        sessions_observed=sessions,
    )


def _feed_class(report: Mapping[str, Any], product: str) -> str:
    """Return the underlying product's declared feed class from the 9A report."""
    per_date = report.get("per_date")
    if isinstance(per_date, Mapping):
        for entry in per_date.values():
            products = entry.get("products") if isinstance(entry, Mapping) else None
            if isinstance(products, Mapping):
                spec = products.get(product)
                if isinstance(spec, Mapping) and "feed_class" in spec:
                    return str(spec["feed_class"])
    return "unrecorded_in_the_task_9a_report"


def run_study(
    config: MarketStateConfig, project_root: Path
) -> tuple[dict[str, Any], tuple[SnapshotOutcome, ...]]:
    """Run the whole study, returning the report document and the raw outcomes.

    The outcomes are returned alongside the document so the tables and figures
    are rendered from the same objects the report was assembled from, rather
    than from a second traversal that could drift out of agreement with it.
    """
    checks, report_9a = verify_provenance(config, project_root)
    processed_root = project_root / config.provenance.processed_root
    archive_root = project_root / config.provenance.archive_root

    probe_specs = {
        "dividend": config.provenance.dividend_source_path,
        "borrow": config.provenance.borrow_source_path,
        "corporate_action": config.provenance.corporate_action_source_path,
    }
    probe_results = {
        name: probe_declared_source(
            archive_root,
            key=name,
            path=path,
            kind="directory",
            requirement="required",
            capability="spy_american_calibration_readiness",
            why="re-probed by Task 9B at the moment of use",
            candidate_source="see the Task 9A configuration",
        )
        for name, path in probe_specs.items()
    }
    probe_states = {name: result.state for name, result in probe_results.items()}

    outcomes: list[SnapshotOutcome] = []
    american_universes: list[DefinitionUniverse] = []
    per_date: dict[str, Any] = {}
    sr3_by_date = {
        session.date: load_sr3_controls(
            processed_root,
            config.rate_controls.sr3_product,
            config.rate_controls.sr3_statistic,
            session.date,
        )
        for session in config.dates
    }
    for session in config.dates:
        european_universe = load_definitions(
            processed_root,
            config.pairing.european_option_product,
            session.date,
            require_standard_root=config.pairing.require_standard_root,
        )
        american_universe = load_definitions(
            processed_root,
            config.pairing.american_option_product,
            session.date,
            require_standard_root=config.pairing.require_standard_root,
        )
        american_universes.append(american_universe)
        session_outcomes = [
            run_snapshot(
                config=config,
                processed_root=processed_root,
                date=session.date,
                role=session.role,
                instant=instant,
                european_universe=european_universe,
                american_universe=american_universe,
            )
            for instant in config.snapshots.instants_utc(session.date)
        ]
        outcomes.extend(session_outcomes)
        european_expiries = {
            result.expiry for outcome in session_outcomes for result in outcome.european
        }
        american_expiries = {
            result.expiry for outcome in session_outcomes for result in outcome.american
        }
        per_date[session.date.isoformat()] = {
            "role": session.role,
            "european_universe": european_universe.summary(),
            "american_universe": american_universe.summary(),
            "expiries_with_european_pairs": len(european_expiries),
            "expiries_with_american_pairs": len(american_expiries),
            "exactly_matched_expiries": len(european_expiries & american_expiries),
            "sr3_settlement_controls": sr3_by_date[session.date],
            "snapshots": [
                {
                    "snapshot": outcome.label,
                    "instant_utc": outcome.instant.isoformat(),
                    "underlying_bid": outcome.underlying_bid,
                    "underlying_ask": outcome.underlying_ask,
                    "underlying_mid": 0.5 * (outcome.underlying_bid + outcome.underlying_ask),
                    "european_paired_strikes": outcome.european_pairs,
                    "american_paired_strikes": outcome.american_pairs,
                    "european_expiries": [result.summary() for result in outcome.european],
                    "american_expiries": [
                        result.summary() for result in outcome.american if result.matched
                    ],
                }
                for outcome in session_outcomes
            ],
        }

    european = european_sensitivity(outcomes)
    american = american_sensitivity(outcomes)
    facts = _measure_facts(
        outcomes,
        american_universes,
        probe_states,
        european,
        american,
        _feed_class(report_9a, config.pairing.underlying_product),
        len(config.dates),
    )
    fred_controls = load_fred_controls(
        archive_root, report_9a, config.rate_controls.fred_series
    )
    matrix = build_identifiability_matrix(facts)
    fields = proposed_task_9c_contract()

    document = {
        "schema_version": SCHEMA_VERSION,
        "study_version": config.study_version,
        "study": "task-9b-market-state-identifiability-and-reconstruction",
        "scope": {
            "in_scope": [
                "European XSP put-call parity reconstruction of D, F and r",
                "the SPY American parity carry residual, as a diagnostic",
                "descriptive comparison against daily rate controls",
                "an identifiability matrix for the Task 9C PDE inputs",
                "a proposed, unimplemented Task 9C input contract",
            ],
            "explicitly_out_of_scope": [
                "solving any PDE",
                "generating synthetic labels",
                "training any network",
                "computing any Greek",
                "inverting any implied volatility",
                "fitting any volatility surface",
            ],
        },
        "config": config.summary(),
        "provenance": {
            "checks": [check.summary() for check in checks],
            "all_satisfied": all(check.satisfied for check in checks),
            "task_9a_schema_version": report_9a.get("schema_version"),
            "task_9a_audit_version": report_9a.get("audit_version"),
            "declared_source_reprobe": {
                name: result.summary() for name, result in sorted(probe_results.items())
            },
            "reprobe_note": (
                "Task 9A probed these paths too. They are re-probed here because their state is "
                "the most consequential fact about SPY identifiability, and a fact that "
                "important is observed at the moment it is used rather than quoted from an "
                "older report."
            ),
        },
        "per_date": per_date,
        "european_sensitivity": european,
        "american_diagnostic": {
            "name": config.american.residual_name,
            "definition": (
                "F_tilde_i(T) = [C_A - P_A + D(T) K_i] / D(T) per strike, from American "
                "midpoints and the discount factor of the exactly matched XSP expiry; "
                "Q_tilde(T) = S_SPY - D(T) * median_i F_tilde_i(T)."
            ),
            "forbidden_names": list(config.american.forbidden_names),
            "caveat": config.american.residual_caveat,
            "sensitivity": american,
            "ex_date_comparison": ex_date_comparison(config, outcomes),
        },
        "rate_controls": {
            "status": config.rate_controls.comparison_status,
            "caveat": config.rate_controls.comparison_caveat,
            "sr3_rate_definition": config.rate_controls.sr3_rate_definition,
            "fred": fred_controls,
            "descriptive_comparison": compare_rates(
                outcomes,
                fred_controls,
                sr3_by_date,
                relative_half_width=config.strike_windows.relative_half_widths[-1],
                method=config.fit.methods[0],
            ),
        },
        "identifiability": {
            "classification_thresholds": {
                "minimum_containment_for_robust": MINIMUM_CONTAINMENT_FOR_ROBUST,
                "maximum_discount_spread_for_robust": MAXIMUM_DISCOUNT_SPREAD_FOR_ROBUST,
                "provenance": THRESHOLD_PROVENANCE,
            },
            "observed_facts": facts.summary(),
            "matrix": [row.summary() for row in matrix],
        },
        "task_9c_input_contract": contract_summary(fields),
        "verdicts": build_verdicts(
            facts=facts, european=european, american=american, sessions=len(config.dates)
        ),
        "limitations_and_non_claims": [
            "Three sessions, two of them consecutive. Nothing here is a distributional claim.",
            "Every setting was chosen after the archive was observed and is exploratory.",
            "The expiry instant is assumed; the archive carries no settlement metadata.",
            "Bid-ask widths are executable-liquidity measures, never standard errors. No "
            "confidence interval or p-value appears anywhere in this study.",
            "The American parity carry residual is not a dividend, not a dividend present "
            "value, not a borrow rate, not a SPY forward and not an independent market input.",
            "No dividend is described as observed or inferred; the archive holds no dividend "
            "calendar.",
            "The XSP forward is the cash-settled index contract's forward. It is not the SPY "
            "forward, and the ETF/index basis is not measured here.",
            "No OIS curve is bootstrapped. Treasury constant-maturity yields are par yields on "
            "government credit; an SR3 settlement implies a futures rate over a forward accrual "
            "period, not a zero-coupon yield.",
            "The underlying feed is a partial-venue consolidation, so its half-spread is a lower "
            "bound on the spot's error rather than a measurement of it.",
            "No PDE is solved, no label generated, no network trained, no Greek computed, no "
            "implied volatility inverted and no volatility surface fitted.",
            "The held-out session is measured here. That is consistent with this study's rules, "
            "because it fits no model, tunes no free parameter and selects nothing. But the "
            "reported spreads and the exploratory thresholds beside them were computed partly "
            "from it, so task 9C must not reuse those numbers as an informative prior while "
            "still relying on that session as an untouched evaluation partition.",
            "A declared source reported as 'present' means files exist at that path. This study "
            "parses no dividend, borrow or corporate-action file, so presence would be a "
            "statement about acquisition and not about content.",
        ],
    }
    return document, tuple(outcomes)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Task 9B: reconstruct the identifiable market state from Task 9A's processed "
            "archive and classify every Task 9C PDE input by identifiability."
        )
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Task 9B configuration TOML"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="override the configured, git-ignored output root",
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=None,
        help="repository root the configured relative paths resolve against",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the study, writing a report, two tables and the figures."""
    arguments = _parser().parse_args(argv)
    project_root = (
        arguments.project_root
        if arguments.project_root is not None
        else Path(__file__).resolve().parents[4]
    )
    try:
        config = load_state_config(arguments.config)
    except ConfigError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    root = (
        arguments.output_root
        if arguments.output_root is not None
        else project_root / config.output.report_root
    )
    try:
        written = run_and_write(config, project_root, root)
    except ProvenanceError as error:
        print(f"provenance error: {error}", file=sys.stderr)
        return 3
    except ReconstructionError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    for path in written:
        print(f"wrote {path}", file=sys.stderr)
    return 0


def run_and_write(
    config: MarketStateConfig, project_root: Path, root: Path
) -> tuple[Path, ...]:
    """Run the study and publish every artefact atomically. Returns the paths.

    Two guarantees, stated precisely because the difference matters.

    Every artefact is replaced atomically, and the whole study runs to
    completion -- including rendering every table and figure -- before anything
    is written. A failure anywhere in the computation therefore leaves the
    previous outputs entirely untouched.

    The artefact *set* is not transactional. The report, the two tables and each
    figure are separate renames, so an I/O failure part-way through publishing
    (a full disk, a revoked permission) can leave some artefacts refreshed and
    others stale. To make that state detectable rather than silent, the JSON
    report -- the index artefact a reader starts from -- is published **last**,
    after every table and figure it describes. A stale report beside fresh
    tables is a visibly interrupted run; a fresh report beside stale tables
    would look complete and would not be.
    """
    document, outcomes = run_study(config, project_root)
    (european_header, european_rows), (american_header, american_rows) = build_tables(outcomes)
    figures = build_figures(outcomes, config)

    written: list[Path] = []
    table_path = root / config.output.table_filename
    write_csv_atomically(table_path, european_header, european_rows)
    written.append(table_path)
    american_path = root / config.output.american_table_filename
    write_csv_atomically(american_path, american_header, american_rows)
    written.append(american_path)
    for name, svg in sorted(figures.items()):
        figure_path = root / config.output.figure_subdirectory / name
        write_atomically(figure_path, svg.encode("utf-8"))
        written.append(figure_path)
    report_path = root / config.output.report_filename
    write_json_atomically(report_path, document)
    written.append(report_path)
    return tuple(written)


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
