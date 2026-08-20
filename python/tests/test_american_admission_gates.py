"""Task 9E integrity and leakage gates.

Each test breaks exactly one property of a valid fixture and asserts that the
corresponding gate fires. Fixtures are a handful of rows; nothing here reads the
Git-ignored candidate dataset.

Nearest-neighbour distance is deliberately absent. Task 9D measured it *after*
the fact, so there is no predeclared threshold to gate on, and inventing one
here would be a post-hoc criterion. It stays a descriptive diagnostic.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from american_admission_fixtures import (
    LABEL_STEPS,
    rewrite_split,
    rows_for_split,
    table_from_rows,
    write_dataset,
    write_manifest,
)
from differentiable_pricing.data.american_admission import (
    AdmissionGateError,
    check_cross_partition_disjointness,
    check_label_policy,
    check_manifest_reconciliation,
    check_row_identities,
    check_schema,
    run_all_gates,
)
from differentiable_pricing.data.american_schema import LABEL_POLICY_NAME, TABLE_SCHEMA


def _table(count: int = 4, *, offset: int = 0, split: str = "train") -> pa.Table:
    return table_from_rows(rows_for_split(split, count, offset=offset))


def _with_column(table: pa.Table, name: str, values: list) -> pa.Table:
    index = table.schema.get_field_index(name)
    field = table.schema.field(index)
    return table.set_column(index, field, pa.array(values, type=field.type))


def test_valid_fixture_passes_every_gate(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    names = [report.name for report in run_all_gates(directory)]
    assert "manifest_reconciliation" in names
    assert names.count("row_identities") == 3
    assert names[-1] == "cross_partition_disjointness"


# --- schema selection and rejection -----------------------------------------


def test_schema_gate_accepts_the_contract() -> None:
    assert check_schema(TABLE_SCHEMA, where="fixture").measurements["columns"] == 29


def test_schema_gate_rejects_a_missing_column() -> None:
    table = _table().drop_columns(["intrinsic_value"])
    with pytest.raises(AdmissionGateError, match="missing="):
        check_schema(table.schema, where="fixture")


def test_schema_gate_rejects_an_extra_column() -> None:
    table = _table().append_column("delta", pa.array([0.5] * 4, type=pa.float64()))
    with pytest.raises(AdmissionGateError, match="extra="):
        check_schema(table.schema, where="fixture")


def test_schema_gate_rejects_a_retyped_column() -> None:
    fields = [
        field.with_type(pa.float32()) if field.name == "spot" else field for field in TABLE_SCHEMA
    ]
    with pytest.raises(AdmissionGateError, match="does not match"):
        check_schema(pa.schema(fields), where="fixture")


def test_schema_gate_rejects_a_reordered_column() -> None:
    fields = list(TABLE_SCHEMA)
    fields[6], fields[7] = fields[7], fields[6]
    with pytest.raises(AdmissionGateError, match="does not match"):
        check_schema(pa.schema(fields), where="fixture")


def test_schema_gate_rejects_a_nullable_column() -> None:
    fields = [
        field.with_nullable(True) if field.name == "spot" else field for field in TABLE_SCHEMA
    ]
    with pytest.raises(AdmissionGateError, match=r"does not match|nullable"):
        check_schema(pa.schema(fields), where="fixture")


# --- manifest and file reconciliation ---------------------------------------


def test_manifest_gate_rejects_a_digest_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    rewrite_split(directory, "train", _table(6, offset=100))
    with pytest.raises(AdmissionGateError, match="sha256 mismatch"):
        check_manifest_reconciliation(directory)


def test_manifest_gate_rejects_a_row_count_mismatch(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    shorter = _table(5)
    rewrite_split(directory, "train", shorter)
    import hashlib

    payload = (directory / "train.parquet").read_bytes()
    for entry in manifest["files"]:
        if entry["split"] == "train":
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
    write_manifest(directory, manifest)
    with pytest.raises(AdmissionGateError, match="rows, manifest declares"):
        check_manifest_reconciliation(directory)


def test_manifest_gate_rejects_a_missing_file(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    (directory / "validation.parquet").unlink()
    with pytest.raises(AdmissionGateError, match="which is missing"):
        check_manifest_reconciliation(directory)


def test_manifest_gate_rejects_an_undeclared_partition(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    pq.write_table(_table(2), directory / "stray.parquet")
    with pytest.raises(AdmissionGateError, match="never declares"):
        check_manifest_reconciliation(directory)


# --- label identities and bounds --------------------------------------------


def test_identity_gate_rejects_a_broken_average() -> None:
    table = _with_column(_table(), "american_price", [1.0, 2.0, 3.0, 4.0])
    with pytest.raises(AdmissionGateError, match=r"0\.5 \* \(at_steps"):
        check_row_identities(table, where="fixture")


def test_identity_gate_rejects_a_broken_premium() -> None:
    rows = rows_for_split("train", 4)
    rows[2]["early_exercise_premium"] += 1e-9
    with pytest.raises(AdmissionGateError, match="early_exercise_premium =="):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_a_broken_adjacent_gap() -> None:
    rows = rows_for_split("train", 4)
    rows[1]["adjacent_step_gap"] = 0.0
    with pytest.raises(AdmissionGateError, match="adjacent_step_gap =="):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_broken_american_dominance() -> None:
    rows = rows_for_split("train", 4)
    row = rows[0]
    row["european_crr_price_at_steps"] = row["american_price_at_steps"] + 1.0
    row["european_crr_price_at_steps_plus_one"] = row["american_price_at_steps"] + 1.0
    row["european_crr_price"] = 0.5 * (
        row["european_crr_price_at_steps"] + row["european_crr_price_at_steps_plus_one"]
    )
    row["early_exercise_premium"] = row["american_price"] - row["european_crr_price"]
    with pytest.raises(AdmissionGateError, match="American dominance"):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_a_price_below_intrinsic() -> None:
    rows = rows_for_split("train", 4)
    row = rows[0]
    row["intrinsic_value"] = row["american_price"] + 1.0
    with pytest.raises(AdmissionGateError, match="intrinsic lower bound"):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_broken_log_moneyness() -> None:
    rows = rows_for_split("train", 4)
    rows[3]["log_moneyness"] += 1e-6
    with pytest.raises(AdmissionGateError, match="log_moneyness =="):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_a_non_finite_value() -> None:
    rows = rows_for_split("train", 4)
    rows[1]["exercise_activity_fraction"] = float("nan")
    with pytest.raises(AdmissionGateError, match="non-finite"):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_an_unknown_option_type() -> None:
    rows = rows_for_split("train", 4)
    rows[0]["option_type"] = "straddle"
    with pytest.raises(AdmissionGateError, match="unknown option_type"):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_a_degenerate_probability() -> None:
    rows = rows_for_split("train", 4)
    rows[2]["risk_neutral_probability"] = 1.0
    with pytest.raises(AdmissionGateError, match="open interval"):
        check_row_identities(table_from_rows(rows), where="fixture")


def test_identity_gate_rejects_an_out_of_range_exercise_step() -> None:
    rows = rows_for_split("train", 4)
    rows[1]["earliest_exercise_step"] = -2
    with pytest.raises(AdmissionGateError, match="earliest_exercise_step"):
        check_row_identities(table_from_rows(rows), where="fixture")


# --- leakage ----------------------------------------------------------------


def test_duplicate_sample_ids_are_rejected(tmp_path: Path) -> None:
    rows = rows_for_split("train", 4)
    rows[3]["sample_id"] = rows[0]["sample_id"]
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    _rewrite_with_manifest(directory, manifest, "train", table_from_rows(rows))
    with pytest.raises(AdmissionGateError, match="duplicate sample_id"):
        run_all_gates(directory)


def test_duplicate_contract_states_are_rejected(tmp_path: Path) -> None:
    rows = rows_for_split("train", 4)
    duplicate = dict(rows[0])
    duplicate["sample_id"] = "train-99"
    rows.append(duplicate)
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    _rewrite_with_manifest(directory, manifest, "train", table_from_rows(rows))
    with pytest.raises(AdmissionGateError, match="duplicate contract state"):
        run_all_gates(directory)


def test_cross_partition_identifier_overlap_is_rejected() -> None:
    train = _table(4, split="train")
    other = table_from_rows(rows_for_split("train", 2, offset=50))
    with pytest.raises(AdmissionGateError, match="sample_id value"):
        check_cross_partition_disjointness({"train": train, "validation": other})


def test_cross_partition_state_overlap_is_rejected() -> None:
    train_rows = rows_for_split("train", 4)
    shared = dict(train_rows[1])
    shared["sample_id"] = "validation-0"
    shared["split"] = "validation"
    with pytest.raises(AdmissionGateError, match="exact contract state"):
        check_cross_partition_disjointness(
            {
                "train": table_from_rows(train_rows),
                "validation": table_from_rows([shared]),
            }
        )


def test_partition_label_mismatch_is_rejected(tmp_path: Path) -> None:
    rows = rows_for_split("train", 4)
    rows[0]["split"] = "validation"
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    _rewrite_with_manifest(directory, manifest, "train", table_from_rows(rows))
    with pytest.raises(AdmissionGateError, match="declares split value"):
        run_all_gates(directory)


def test_label_policy_gate_accepts_matching_columns() -> None:
    table = _table()
    manifest = {"label_policy": {"name": LABEL_POLICY_NAME, "steps": LABEL_STEPS}}
    report = check_label_policy(table, manifest, where="fixture")
    assert report.measurements == {
        "name": LABEL_POLICY_NAME,
        "steps": LABEL_STEPS,
        "rows": table.num_rows,
    }


def test_label_policy_gate_rejects_name_mismatch() -> None:
    rows = rows_for_split("train", 4)
    rows[1]["label_policy"] = "other-policy/1"
    manifest = {"label_policy": {"name": LABEL_POLICY_NAME, "steps": LABEL_STEPS}}
    with pytest.raises(AdmissionGateError, match=r"1 row label_policy"):
        check_label_policy(table_from_rows(rows), manifest, where="fixture")


def test_label_policy_gate_rejects_steps_mismatch() -> None:
    rows = rows_for_split("train", 4)
    rows[2]["label_steps"] = LABEL_STEPS + 1
    manifest = {"label_policy": {"name": LABEL_POLICY_NAME, "steps": LABEL_STEPS}}
    with pytest.raises(AdmissionGateError, match=r"1 row label_steps"):
        check_label_policy(table_from_rows(rows), manifest, where="fixture")


def test_label_policy_gate_reports_mixed_column_mismatches() -> None:
    rows = rows_for_split("train", 4)
    rows[0]["label_policy"] = "other-policy/1"
    rows[1]["label_policy"] = "other-policy/1"
    rows[2]["label_steps"] = LABEL_STEPS + 1
    manifest = {"label_policy": {"name": LABEL_POLICY_NAME, "steps": LABEL_STEPS}}
    with pytest.raises(
        AdmissionGateError,
        match=r"2 row label_policy.*1 row label_steps",
    ):
        check_label_policy(table_from_rows(rows), manifest, where="fixture")


def test_domain_violation_is_rejected(tmp_path: Path) -> None:
    rows = rows_for_split("train", 4)
    rows[0]["volatility"] = 5.0
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    _rewrite_with_manifest(directory, manifest, "train", table_from_rows(rows))
    with pytest.raises(AdmissionGateError, match="outside the declared domain"):
        run_all_gates(directory)


def _rewrite_with_manifest(directory: Path, manifest: dict, split: str, table: pa.Table) -> None:
    """Replace one partition and re-pin its digest and row count."""
    import hashlib

    rewrite_split(directory, split, table)
    payload = (directory / f"{split}.parquet").read_bytes()
    for entry in manifest["files"]:
        if entry["split"] == split:
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["rows"] = table.num_rows
    manifest["row_counts"][split] = table.num_rows
    manifest["label_diagnostics"]["splits"][split]["rows"] = table.num_rows
    write_manifest(directory, manifest)
