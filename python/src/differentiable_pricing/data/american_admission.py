"""Integrity and leakage gates for ``american-option-dataset/1`` tables.

Task 9E's mechanically checkable half. Each gate answers one question about a
dataset's bytes, and every one of them fails closed: a gate raises
:class:`AdmissionGateError` rather than returning a warning, because a dataset
that half-passes is a dataset nobody can describe.

The gates are deliberately about the **CRR mapping itself** — schema, manifest
reconciliation, label identities, partition disjointness. They run no pricing
study and open no other engine.

Nothing here admits a dataset. Passing every gate is a precondition for the
admission decision recorded in ``docs/american-crr-dataset-admission.md``, not
the decision itself.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from .american_schema import (
    DOMAIN_FIELDS,
    NO_EARLY_EXERCISE_STEP,
    OPTION_TYPES,
    SPLIT_NAMES,
    TABLE_SCHEMA,
    AmericanManifestError,
    load_american_manifest,
)

STATE_COLUMNS: Final = (
    "option_type",
    "spot",
    "strike",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
)
"""Columns that jointly identify a contract state, for the leakage gates."""

_READ_CHUNK_BYTES: Final = 1 << 22


class AdmissionGateError(RuntimeError):
    """Raised when a dataset fails one of the task 9E admission gates."""


@dataclass(frozen=True)
class GateReport:
    """What a gate measured, so a caller can record it rather than restate it."""

    name: str
    measurements: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_READ_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_schema(schema: pa.Schema, *, where: str) -> GateReport:
    """The file's schema must equal the pinned contract exactly.

    Exact equality, not a subset or a superset: a missing column, an extra
    column, a reordered column, a retyped column, or a column that became
    nullable all fail. There is no permissive fallback.
    """
    if not schema.equals(TABLE_SCHEMA, check_metadata=False):
        expected = [f"{f.name}:{f.type}" for f in TABLE_SCHEMA]
        actual = [f"{f.name}:{f.type}" for f in schema]
        missing = sorted(set(TABLE_SCHEMA.names) - set(schema.names))
        extra = sorted(set(schema.names) - set(TABLE_SCHEMA.names))
        detail = (
            f"missing={missing} extra={extra}"
            if (missing or extra)
            else (f"expected {expected}, got {actual}")
        )
        raise AdmissionGateError(
            f"{where} schema does not match american-option-dataset/1: {detail}"
        )
    for field in schema:
        if field.nullable:
            raise AdmissionGateError(
                f"{where} column '{field.name}' is nullable; the contract requires "
                "every column non-nullable"
            )
    return GateReport("schema", {"columns": len(schema.names)})


def check_manifest_reconciliation(dataset: Path) -> GateReport:
    """Manifest, files on disk, digests and row counts must agree exactly.

    Runs to completion before a single row is read, and rejects a declared file
    that is missing, a digest that does not match the bytes, a row count that
    does not match the footer, and a stray Parquet partition the manifest never
    declared.
    """
    dataset = Path(dataset)
    try:
        manifest = load_american_manifest(dataset)
    except AmericanManifestError as error:
        raise AdmissionGateError(str(error)) from error

    declared: dict[str, str] = {}
    digests: dict[str, str] = {}
    rows: dict[str, int] = {}
    for entry in manifest["files"]:
        split = str(entry["split"])
        path = dataset / str(entry["file"])
        declared[split] = path.name
        if not path.is_file():
            raise AdmissionGateError(
                f"manifest declares split '{split}' at '{path.name}', which is missing"
            )
        digest = sha256_file(path)
        if digest != entry["sha256"]:
            raise AdmissionGateError(
                f"sha256 mismatch for split '{split}': manifest declares "
                f"{entry['sha256']}, file is {digest}"
            )
        digests[split] = digest
        footer_rows = pq.ParquetFile(path).metadata.num_rows
        if footer_rows != int(entry["rows"]):
            raise AdmissionGateError(
                f"split '{split}' holds {footer_rows} rows, manifest declares {entry['rows']}"
            )
        rows[split] = footer_rows

    present = {path.name for path in dataset.glob("*.parquet")}
    undeclared = sorted(present - set(declared.values()))
    if undeclared:
        raise AdmissionGateError(
            f"dataset directory holds Parquet file(s) the manifest never declares: {undeclared}"
        )
    if sorted(declared) != sorted(SPLIT_NAMES):
        raise AdmissionGateError(
            f"manifest declares partitions {sorted(declared)}, expected {sorted(SPLIT_NAMES)}"
        )
    return GateReport(
        "manifest_reconciliation",
        {"rows": rows, "digests": digests, "total_rows": sum(rows.values())},
    )


def check_row_identities(table: pa.Table, *, where: str) -> GateReport:
    """Every label identity, bound and finiteness property the schema asserts.

    The identities that hold bitwise are checked bitwise, with no tolerance.
    That is not strictness for its own sake: they are exact consequences of how
    the generator built the columns, so any tolerance would hide a real defect.
    The one inequality that is only exact up to rounding — the intrinsic lower
    bound — is checked against an explicit, named allowance.
    """
    check_schema(table.schema, where=where)

    def column(name: str) -> np.ndarray:
        return np.asarray(table[name])

    strings = {
        name: np.asarray(table[name].to_pylist(), dtype=object)
        for name in ("option_type", "split", "label_policy")
    }

    unknown = sorted(set(strings["option_type"]) - set(OPTION_TYPES))
    if unknown:
        raise AdmissionGateError(f"{where} has unknown option_type value(s): {unknown}")

    for name in table.schema.names:
        field = table.schema.field(name)
        if pa.types.is_floating(field.type):
            values = column(name)
            if not bool(np.isfinite(values).all()):
                count = int((~np.isfinite(values)).sum())
                raise AdmissionGateError(
                    f"{where} column '{name}' holds {count} non-finite value(s)"
                )
        if table[name].null_count:
            raise AdmissionGateError(
                f"{where} column '{name}' holds {table[name].null_count} null(s)"
            )

    american = column("american_price")
    european = column("european_crr_price")
    premium = column("early_exercise_premium")
    intrinsic = column("intrinsic_value")
    spot = column("spot")
    strike = column("strike")
    gap = column("adjacent_step_gap")
    a_n = column("american_price_at_steps")
    a_n1 = column("american_price_at_steps_plus_one")
    e_n = column("european_crr_price_at_steps")
    e_n1 = column("european_crr_price_at_steps_plus_one")
    probability = column("risk_neutral_probability")

    exact = {
        "american_price == 0.5 * (at_steps + at_steps_plus_one)": (
            american,
            0.5 * (a_n + a_n1),
        ),
        "european_crr_price == 0.5 * (at_steps + at_steps_plus_one)": (
            european,
            0.5 * (e_n + e_n1),
        ),
        "adjacent_step_gap == |at_steps - at_steps_plus_one|": (
            gap,
            np.abs(a_n - a_n1),
        ),
        "early_exercise_premium == american_price - european_crr_price": (
            premium,
            american - european,
        ),
    }
    for identity, (left, right) in exact.items():
        if not bool(np.array_equal(left, right)):
            residual = float(np.max(np.abs(left - right)))
            raise AdmissionGateError(
                f"{where} violates the exact identity '{identity}': maximum residual "
                f"{residual:.6e}, expected 0"
            )

    if bool((american < european).any()):
        worst = float(np.min(american - european))
        raise AdmissionGateError(
            f"{where} violates American dominance: minimum "
            f"american_price - european_crr_price is {worst:.6e}, expected >= 0"
        )
    if bool((premium < 0.0).any()):
        raise AdmissionGateError(f"{where} holds a negative early_exercise_premium")

    # The intrinsic lower bound is exact in exact arithmetic but the columns are
    # float64, so it is checked against a spot-scaled rounding allowance rather
    # than at zero. The allowance is a rounding budget, never an error bound.
    intrinsic_slack = american - intrinsic
    allowance = 1.0e-9 * spot
    if bool((intrinsic_slack < -allowance).any()):
        worst = float(np.min(intrinsic_slack))
        raise AdmissionGateError(
            f"{where} violates the intrinsic lower bound beyond the rounding "
            f"allowance: minimum american_price - intrinsic_value is {worst:.6e}"
        )

    log_moneyness = column("log_moneyness")
    residual = float(np.max(np.abs(np.log(spot / strike) - log_moneyness)))
    if not residual <= 1.0e-12:
        raise AdmissionGateError(
            f"{where} violates log_moneyness == log(spot / strike): maximum residual {residual:.6e}"
        )

    if bool(((probability <= 0.0) | (probability >= 1.0)).any()):
        raise AdmissionGateError(
            f"{where} holds a risk_neutral_probability outside the open interval (0, 1)"
        )

    steps = column("label_steps")
    if bool((steps <= 0).any()):
        raise AdmissionGateError(f"{where} holds a non-positive label_steps")
    earliest = column("earliest_exercise_step")
    invalid = (earliest < NO_EARLY_EXERCISE_STEP) | (earliest >= steps)
    if bool(invalid.any()):
        raise AdmissionGateError(
            f"{where} holds an earliest_exercise_step outside "
            f"[{NO_EARLY_EXERCISE_STEP}, label_steps)"
        )

    return GateReport(
        "row_identities",
        {
            "rows": table.num_rows,
            "worst_intrinsic_slack": float(np.min(intrinsic_slack)),
            "worst_log_moneyness_residual": residual,
            "zero_premium_rows": int((premium == 0.0).sum()),
        },
    )


def check_partition_labels(table: pa.Table, split: str, *, where: str) -> GateReport:
    """Every row of a partition must declare that partition."""
    values = sorted(set(table["split"].to_pylist()))
    if values != [split]:
        raise AdmissionGateError(f"{where} declares split value(s) {values}, expected ['{split}']")
    return GateReport("partition_labels", {"split": split})


def check_label_policy(
    table: pa.Table,
    manifest: Mapping[str, Any],
    *,
    where: str,
) -> GateReport:
    """Require every row to use the manifest's exact label policy."""
    policy = manifest.get("label_policy")
    if not isinstance(policy, Mapping):
        raise AdmissionGateError("manifest.label_policy must be an object")
    expected_name = policy.get("name")
    expected_steps = policy.get("steps")
    if not isinstance(expected_name, str) or not expected_name:
        raise AdmissionGateError("manifest.label_policy.name must be a non-empty string")
    if (
        isinstance(expected_steps, bool)
        or not isinstance(expected_steps, int)
        or expected_steps <= 0
    ):
        raise AdmissionGateError("manifest.label_policy.steps must be a positive integer")

    names = np.asarray(table["label_policy"].to_pylist(), dtype=object)
    steps = np.asarray(table["label_steps"], dtype=np.int64)
    bad_names = int((names != expected_name).sum())
    bad_steps = int((steps != expected_steps).sum())
    if bad_names or bad_steps:
        details: list[str] = []
        if bad_names:
            details.append(
                f"{bad_names} row label_policy value(s) differ from "
                f"manifest.label_policy.name={expected_name!r}"
            )
        if bad_steps:
            details.append(
                f"{bad_steps} row label_steps value(s) differ from "
                f"manifest.label_policy.steps={expected_steps}"
            )
        raise AdmissionGateError(f"{where} label-policy mismatch: " + "; ".join(details))
    return GateReport(
        "label_policy",
        {"name": expected_name, "steps": expected_steps, "rows": table.num_rows},
    )


def check_domain(table: pa.Table, manifest: Mapping[str, Any], *, where: str) -> GateReport:
    """Every sampled field must lie inside the manifest's declared domain."""
    observed: dict[str, tuple[float, float]] = {}
    for field in DOMAIN_FIELDS:
        low, high = (float(bound) for bound in manifest["domain"][field])
        values = np.asarray(table[field])
        seen = (float(values.min()), float(values.max()))
        observed[field] = seen
        if seen[0] < low - 1.0e-12 or seen[1] > high + 1.0e-12:
            raise AdmissionGateError(
                f"{where} field '{field}' spans {seen}, outside the declared domain [{low}, {high}]"
            )
    return GateReport("domain", {"observed": observed})


def _state_keys(table: pa.Table) -> np.ndarray:
    """A hashable exact key per row over the contract-state columns.

    Float bit patterns are used rather than rounded values: two rows are the
    same state only if every state column is bit-for-bit identical. A tolerance
    here would silently merge distinct states.
    """
    parts: list[np.ndarray] = []
    for name in STATE_COLUMNS:
        if name == "option_type":
            parts.append(np.asarray(table[name].to_pylist(), dtype="U8"))
        else:
            parts.append(np.asarray(table[name], dtype=np.float64).view(np.int64).astype("U20"))
    return np.array(["|".join(row) for row in zip(*parts, strict=True)], dtype=object)


def check_identifier_uniqueness(table: pa.Table, *, where: str) -> GateReport:
    """``sample_id`` must be unique within a partition."""
    identifiers = table["sample_id"].to_pylist()
    unique = len(set(identifiers))
    if unique != len(identifiers):
        raise AdmissionGateError(
            f"{where} holds {len(identifiers) - unique} duplicate sample_id value(s)"
        )
    return GateReport("identifier_uniqueness", {"rows": len(identifiers)})


def check_state_uniqueness(table: pa.Table, *, where: str) -> GateReport:
    """No two rows of a partition may share an exact contract state."""
    keys = _state_keys(table)
    unique = len(set(keys.tolist()))
    if unique != len(keys):
        raise AdmissionGateError(f"{where} holds {len(keys) - unique} duplicate contract state(s)")
    return GateReport("state_uniqueness", {"rows": len(keys)})


def check_cross_partition_disjointness(
    tables: Mapping[str, pa.Table],
) -> GateReport:
    """No identifier and no exact contract state may appear in two partitions.

    This is the leakage gate the data protocol requires. It is exact: it says
    nothing about near-duplicate states, which remain a descriptive diagnostic
    with no predeclared threshold.
    """
    identifiers = {name: set(table["sample_id"].to_pylist()) for name, table in tables.items()}
    states = {name: set(_state_keys(table).tolist()) for name, table in tables.items()}
    names: Sequence[str] = sorted(tables)
    overlaps: dict[str, int] = {}
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            shared_ids = identifiers[left] & identifiers[right]
            if shared_ids:
                raise AdmissionGateError(
                    f"partitions '{left}' and '{right}' share {len(shared_ids)} sample_id value(s)"
                )
            shared_states = states[left] & states[right]
            if shared_states:
                raise AdmissionGateError(
                    f"partitions '{left}' and '{right}' share {len(shared_states)} "
                    "exact contract state(s)"
                )
            overlaps[f"{left}|{right}"] = 0
    return GateReport("cross_partition_disjointness", {"pairs": overlaps})


def run_all_gates(dataset: Path) -> list[GateReport]:
    """Run every gate over a dataset directory, in order, failing on the first.

    Reads whole partitions, so it is a deliberate, explicit call rather than
    something a test suite runs incidentally: the tests exercise these gates on
    small fixtures instead.
    """
    dataset = Path(dataset)
    reports = [check_manifest_reconciliation(dataset)]
    manifest = load_american_manifest(dataset)
    tables: dict[str, pa.Table] = {}
    for entry in manifest["files"]:
        split = str(entry["split"])
        table = pq.read_table(dataset / str(entry["file"]))
        tables[split] = table
        where = f"split '{split}'"
        reports.append(check_partition_labels(table, split, where=where))
        reports.append(check_label_policy(table, manifest, where=where))
        reports.append(check_row_identities(table, where=where))
        reports.append(check_domain(table, manifest, where=where))
        reports.append(check_identifier_uniqueness(table, where=where))
        reports.append(check_state_uniqueness(table, where=where))
    reports.append(check_cross_partition_disjointness(tables))
    return reports
