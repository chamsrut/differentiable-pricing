"""Tests for the deterministic European-option dataset diagnostics.

Every test builds a real dataset in a temporary directory with the generator
under test, then, where a failure mode is being exercised, corrupts it
deliberately. Nothing here reimplements a pricing formula: the expected values
are recomputed from the stored columns with NumPy, which is the point of a
diagnostic tool.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from differentiable_pricing.data.config import (
    DOMAIN_FIELDS,
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    load_dataset_config,
)
from differentiable_pricing.data.diagnose import (
    ARBITRAGE_RELATIVE_TOLERANCE,
    DIAGNOSTICS_SCHEMA_VERSION,
    FLOAT_COLUMNS,
    MONEYNESS_BAND_EDGES,
    QUANTILE_KEYS,
    QUANTILE_LEVELS,
    QUANTILE_METHOD,
    STANDARD_DEVIATION_DDOF,
    DiagnosticsError,
    diagnose_dataset,
    main,
    render_report,
)
from differentiable_pricing.data.generate import (
    LABEL_FIELDS,
    MANIFEST_NAME,
    PARQUET_COMPRESSION,
    PARQUET_VERSION,
    generate_dataset,
)

CONFIG_TEMPLATE = f"""
schema_version = "{SCHEMA_VERSION}"
generator_version = "{GENERATOR_VERSION}"

[dataset]
name = "european-option-diagnostics-test"
oracle = "dp::black_scholes"
base_seed = 909090

[dataset.rows]
train = 1024
validation = 128
interpolation_test = 128

[domain]
spot = [50.0, 150.0]
log_forward_moneyness = [-0.4, 0.4]
maturity = [0.019178082191780823, 2.0]
rate = [-0.01, 0.08]
dividend_yield = [0.0, 0.05]
volatility = [0.05, 0.8]
"""


# --- fixtures and corruption helpers ---------------------------------------


def build_dataset(root: Path, name: str = "out") -> Path:
    """Generate a valid dataset under ``root/name`` and return its directory.

    The configuration always has the same base name, because the manifest
    records that name: two datasets generated from identical bytes must be
    indistinguishable for the regeneration test to mean anything.
    """
    config_path = root / f"{name}-config" / "dataset.toml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(CONFIG_TEMPLATE, encoding="utf-8")
    directory = root / name
    generate_dataset(load_dataset_config(config_path), directory)
    return directory


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return build_dataset(tmp_path)


def read_manifest(directory: Path) -> dict[str, Any]:
    return json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))


def write_manifest(directory: Path, manifest: Mapping[str, Any]) -> None:
    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n"
    )


def read_table(directory: Path, split: str) -> pa.Table:
    return pq.read_table(directory / f"{split}.parquet")


def rewrite_split(directory: Path, split: str, table: pa.Table, *, rehash: bool = True) -> None:
    """Replace one partition's Parquet file, refreshing the manifest digest.

    ``rehash`` is what separates a byte-corruption test from a semantic one: a
    caller wanting the tool to reach the data must leave the manifest honest
    about the bytes.
    """
    path = directory / f"{split}.parquet"
    pq.write_table(
        table,
        path,
        compression=PARQUET_COMPRESSION,
        version=PARQUET_VERSION,
        write_statistics=True,
    )
    if not rehash:
        return
    manifest = read_manifest(directory)
    for entry in manifest["files"]:
        if entry["split"] == split:
            entry["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
            entry["rows"] = table.num_rows
    manifest["row_counts"][split] = table.num_rows
    write_manifest(directory, manifest)


def with_value(table: pa.Table, column: str, index: int, value: Any) -> pa.Table:
    """Return ``table`` with a single cell replaced, preserving the schema."""
    field_index = table.schema.get_field_index(column)
    field = table.schema.field(field_index)
    values = table.column(column).to_numpy(zero_copy_only=False).copy()
    values[index] = value
    return table.set_column(field_index, field, pa.array(values, type=field.type))


def columns_of(table: pa.Table) -> dict[str, np.ndarray]:
    return {
        name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names
    }


def copy_row(table: pa.Table, source: pa.Table, source_index: int, target_index: int) -> pa.Table:
    """Overwrite one row with another's contract, keeping its own identity.

    ``sample_id`` and ``split`` are preserved, so the result is a genuinely
    duplicated economic state wearing a different, legitimate-looking label --
    the leakage an identifier-only check cannot see.
    """
    result = table
    for name in table.schema.names:
        if name in ("sample_id", "split"):
            continue
        result = with_value(result, name, target_index, source.column(name)[source_index].as_py())
    return result


def discounted(columns: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(S * exp(-q * T), K * exp(-r * T))`` for every row."""
    spot = columns["spot"] * np.exp(-columns["dividend_yield"] * columns["maturity"])
    strike = columns["strike"] * np.exp(-columns["rate"] * columns["maturity"])
    return spot, strike


# --- happy path ------------------------------------------------------------


def test_valid_dataset_reports_no_findings(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    assert report["status"] == "ok"
    assert report["findings"] == []
    assert report["diagnostics_schema_version"] == DIAGNOSTICS_SCHEMA_VERSION
    assert sorted(report["splits"]) == sorted(SPLIT_NAMES)
    for entry in report["files"]:
        assert entry["sha256_verified"] is True
        assert entry["bytes"] > 0


def test_zero_arbitrage_violations_on_a_valid_dataset(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    for split in SPLIT_NAMES:
        bounds = report["splits"][split]["no_arbitrage"]
        assert bounds["violations"] == 0
        assert bounds["lower_bound_violations"] == 0
        assert bounds["upper_bound_violations"] == 0
        assert bounds["rows_not_comparable"] == 0
        assert bounds["max_relative_violation"] <= ARBITRAGE_RELATIVE_TOLERANCE


def test_no_duplicate_or_shared_sample_ids_on_a_valid_dataset(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    for split in SPLIT_NAMES:
        identifiers = report["splits"][split]["sample_ids"]
        assert identifiers["duplicate_rows"] == 0
        assert identifiers["unique"] == identifiers["rows"]
    intersections = report["cross_split"]["sample_id_intersections"]
    assert len(intersections) == 3
    assert all(entry["count"] == 0 for entry in intersections)
    assert [entry["splits"] for entry in intersections] == sorted(
        entry["splits"] for entry in intersections
    )


def test_row_and_option_type_counts_match_the_tables(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    manifest = read_manifest(dataset)
    for split in SPLIT_NAMES:
        table = read_table(dataset, split)
        block = report["splits"][split]
        option_type = columns_of(table)["option_type"]
        assert block["rows"] == table.num_rows == manifest["row_counts"][split]
        assert block["manifest_rows"] == manifest["row_counts"][split]
        assert block["option_types"]["call"] == int((option_type == "call").sum())
        assert block["option_types"]["put"] == int((option_type == "put").sum())
        assert sum(block["option_types"].values()) == table.num_rows


# --- statistics ------------------------------------------------------------


def test_statistics_cover_every_input_price_and_greek(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    expected_columns = set(FLOAT_COLUMNS)
    assert expected_columns >= set(LABEL_FIELDS)
    for split in SPLIT_NAMES:
        statistics = report["splits"][split]["statistics"]
        assert set(statistics) == expected_columns
        for name, block in statistics.items():
            assert set(block) == {
                "count",
                "finite",
                "non_finite",
                "min",
                "max",
                "mean",
                "standard_deviation",
                *QUANTILE_KEYS,
            }, name


def test_statistics_match_a_direct_numpy_computation(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    columns = columns_of(read_table(dataset, "train"))
    statistics = report["splits"]["train"]["statistics"]
    for name in FLOAT_COLUMNS:
        values = columns[name]
        block = statistics[name]
        assert block["count"] == values.size
        assert block["finite"] == values.size
        assert block["non_finite"] == 0
        assert block["min"] == float(values.min())
        assert block["max"] == float(values.max())
        assert block["mean"] == float(values.mean())
        assert block["standard_deviation"] == float(values.std(ddof=STANDARD_DEVIATION_DDOF))
        quantiles = np.quantile(values, QUANTILE_LEVELS, method=QUANTILE_METHOD)
        for key, quantile in zip(QUANTILE_KEYS, quantiles, strict=True):
            assert block[key] == float(quantile), f"{name}.{key}"


def test_moneyness_bands_match_a_direct_computation(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    core_max = MONEYNESS_BAND_EDGES["core_max"]
    tail_max = MONEYNESS_BAND_EDGES["tail_max"]
    for split in SPLIT_NAMES:
        columns = columns_of(read_table(dataset, split))
        absolute_z = np.abs(columns["log_forward_moneyness"]) / (
            columns["volatility"] * np.sqrt(columns["maturity"])
        )
        bands = report["splits"][split]["moneyness_bands"]
        assert bands["core"]["count"] == int((absolute_z <= core_max).sum())
        assert bands["tail"]["count"] == int(
            ((absolute_z > core_max) & (absolute_z <= tail_max)).sum()
        )
        assert bands["extreme"]["count"] == int((absolute_z > tail_max).sum())
        assert bands["non_finite"]["count"] == 0

        rows = absolute_z.size
        assert sum(band["count"] for band in bands.values()) == rows
        assert sum(band["proportion"] for band in bands.values()) == pytest.approx(1.0)
        assert report["splits"][split]["derived_statistics"]["absolute_z"]["max"] == float(
            absolute_z.max()
        )


def test_the_sampling_box_actually_reaches_the_extreme_band(dataset: Path) -> None:
    """The independent-uniform box is known to reach very large |z|."""
    report = diagnose_dataset(dataset)
    assert report["splits"]["train"]["moneyness_bands"]["extreme"]["count"] > 0


# --- determinism -----------------------------------------------------------


def test_diagnostics_are_deterministic(tmp_path: Path) -> None:
    directory = build_dataset(tmp_path)
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main(["--dataset", str(directory), "--output", str(first)]) == 0
    assert main(["--dataset", str(directory), "--output", str(second)]) == 0
    assert first.read_bytes() == second.read_bytes()
    assert diagnose_dataset(directory) == diagnose_dataset(directory)


def test_diagnostics_of_an_identically_regenerated_dataset_are_identical(tmp_path: Path) -> None:
    left = build_dataset(tmp_path, "left")
    right = build_dataset(tmp_path, "right")
    assert (left / "train.parquet").read_bytes() == (right / "train.parquet").read_bytes()
    assert render_report(diagnose_dataset(left)) == render_report(diagnose_dataset(right))


def _keys(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [key for name, value in node.items() for key in [name, *_keys(value)]]
    if isinstance(node, list):
        return [key for item in node for key in _keys(item)]
    return []


def test_report_holds_no_wall_clock_keys(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    banned = {"timestamp", "generated_at", "created_at", "date", "time", "wall_clock", "run_id"}
    assert banned.isdisjoint({key.lower() for key in _keys(report)})


def test_report_is_strict_json_with_sorted_keys(dataset: Path) -> None:
    text = render_report(diagnose_dataset(dataset))
    assert "NaN" not in text and "Infinity" not in text
    parsed = json.loads(text)
    assert list(parsed) == sorted(parsed)


# --- thresholds come from the manifest -------------------------------------


def test_thresholds_are_read_from_the_manifest(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    report = diagnose_dataset(dataset)
    recorded = manifest["label_diagnostics"]["thresholds"]
    assert report["thresholds"]["near_zero_price"] == recorded["near_zero_price"]
    assert report["thresholds"]["saturated_delta"] == recorded["saturated_delta"]
    assert report["thresholds"]["threshold_source"].startswith(MANIFEST_NAME)
    for split in SPLIT_NAMES:
        block = report["splits"][split]["label_diagnostics"]
        assert block["matches_manifest"] is True
        assert block["recomputed"] == manifest["label_diagnostics"]["splits"][split]


def test_a_changed_manifest_threshold_changes_the_recomputed_counts(dataset: Path) -> None:
    """A published threshold, not the installed constant, defines the counts."""
    manifest = read_manifest(dataset)
    manifest["label_diagnostics"]["thresholds"]["near_zero_price"] = 1.0
    write_manifest(dataset, manifest)

    report = diagnose_dataset(dataset)
    assert report["thresholds"]["near_zero_price"] == 1.0
    columns = columns_of(read_table(dataset, "train"))
    block = report["splits"]["train"]["label_diagnostics"]
    assert block["recomputed"]["near_zero_price_rows"] == int((columns["price"] <= 1.0).sum())
    assert block["matches_manifest"] is False
    assert any(finding["check"] == "label_diagnostics" for finding in report["findings"])


@pytest.mark.parametrize("value", [0.0, -1.0, "small"])
def test_an_unusable_manifest_threshold_is_rejected(dataset: Path, value: Any) -> None:
    manifest = read_manifest(dataset)
    manifest["label_diagnostics"]["thresholds"]["saturated_delta"] = value
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="saturated_delta"):
        diagnose_dataset(dataset)


# --- integrity failures ----------------------------------------------------


def test_corrupted_parquet_hash_is_rejected(dataset: Path) -> None:
    path = dataset / "validation.parquet"
    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF
    path.write_bytes(bytes(data))
    with pytest.raises(DiagnosticsError, match="sha256 mismatch"):
        diagnose_dataset(dataset)


def test_hashes_are_verified_before_any_data_is_read(dataset: Path, monkeypatch) -> None:
    """A hash failure must abort before a single Parquet row is opened."""
    opened: list[str] = []
    original = pq.read_table

    def spy(*args: Any, **kwargs: Any) -> pa.Table:
        opened.append(str(args[0]))
        return original(*args, **kwargs)

    monkeypatch.setattr(pq, "read_table", spy)
    path = dataset / "interpolation_test.parquet"
    data = bytearray(path.read_bytes())
    data[0] ^= 0xFF
    path.write_bytes(bytes(data))

    with pytest.raises(DiagnosticsError, match="sha256 mismatch"):
        diagnose_dataset(dataset)
    assert opened == []


def test_missing_file_is_rejected(dataset: Path) -> None:
    (dataset / "train.parquet").unlink()
    with pytest.raises(DiagnosticsError, match="missing from"):
        diagnose_dataset(dataset)


def test_undeclared_parquet_file_is_rejected(dataset: Path) -> None:
    table = read_table(dataset, "train")
    pq.write_table(table, dataset / "extra.parquet")
    with pytest.raises(DiagnosticsError, match="undeclared Parquet file"):
        diagnose_dataset(dataset)


def test_writing_diagnostics_into_the_dataset_directory_is_allowed(dataset: Path) -> None:
    output = dataset / "diagnostics.json"
    assert main(["--dataset", str(dataset), "--output", str(output)]) == 0
    assert main(["--dataset", str(dataset), "--output", str(output)]) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_missing_manifest_is_rejected(tmp_path: Path) -> None:
    directory = build_dataset(tmp_path)
    (directory / MANIFEST_NAME).unlink()
    with pytest.raises(DiagnosticsError, match="cannot read manifest"):
        diagnose_dataset(directory)


def test_malformed_manifest_is_rejected(dataset: Path) -> None:
    (dataset / MANIFEST_NAME).write_text("{not json", encoding="utf-8")
    with pytest.raises(DiagnosticsError, match="not valid JSON"):
        diagnose_dataset(dataset)


def test_unknown_manifest_schema_version_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    manifest["schema_version"] = "european-option-dataset/9"
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="only understands"):
        diagnose_dataset(dataset)


def test_missing_dataset_directory_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(DiagnosticsError, match="does not exist"):
        diagnose_dataset(tmp_path / "absent")


def test_manifest_row_count_mismatch_is_rejected(dataset: Path) -> None:
    """``manifest.json`` is not hashed, so only this check catches a lone edit."""
    manifest = read_manifest(dataset)
    manifest["row_counts"]["train"] += 1
    for entry in manifest["files"]:
        if entry["split"] == "train":
            entry["rows"] += 1
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="the manifest declares"):
        diagnose_dataset(dataset)


def test_internally_inconsistent_manifest_row_counts_are_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    for entry in manifest["files"]:
        if entry["split"] == "train":
            entry["rows"] = 999999
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match=r"manifest\.row_counts\.train"):
        diagnose_dataset(dataset)


def test_one_file_declared_for_two_partitions_is_rejected(dataset: Path) -> None:
    """Aliasing would collapse two hash checks into one and drop a partition."""
    manifest = read_manifest(dataset)
    train = next(entry for entry in manifest["files"] if entry["split"] == "train")
    for entry in manifest["files"]:
        if entry["split"] == "validation":
            entry["file"] = train["file"]
            entry["sha256"] = train["sha256"]
            entry["rows"] = train["rows"]
    manifest["row_counts"]["validation"] = manifest["row_counts"]["train"]
    write_manifest(dataset, manifest)
    (dataset / "validation.parquet").unlink()

    with pytest.raises(DiagnosticsError, match="more than one partition"):
        diagnose_dataset(dataset)


def test_a_malformed_manifest_file_entry_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    manifest["files"][0] = "train.parquet"
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="must be a JSON object"):
        diagnose_dataset(dataset)


def test_a_partition_missing_from_the_manifest_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    manifest["files"] = [entry for entry in manifest["files"] if entry["split"] != "validation"]
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="does not declare partition"):
        diagnose_dataset(dataset)


# --- malformed data --------------------------------------------------------


def test_malformed_schema_is_rejected(dataset: Path) -> None:
    table = read_table(dataset, "train")
    rewrite_split(dataset, "train", table.drop_columns(["vega"]))
    with pytest.raises(DiagnosticsError, match="has columns"):
        diagnose_dataset(dataset)


def test_wrong_column_type_is_rejected(dataset: Path) -> None:
    table = read_table(dataset, "train")
    index = table.schema.get_field_index("gamma")
    demoted = table.set_column(
        index,
        pa.field("gamma", pa.float32(), nullable=False),
        table.column("gamma").cast(pa.float32()),
    )
    rewrite_split(dataset, "train", demoted)
    with pytest.raises(DiagnosticsError, match="expected double"):
        diagnose_dataset(dataset)


def test_unknown_option_type_is_rejected(dataset: Path) -> None:
    table = with_value(read_table(dataset, "train"), "option_type", 0, "straddle")
    rewrite_split(dataset, "train", table)
    with pytest.raises(DiagnosticsError, match="unknown option_type value"):
        diagnose_dataset(dataset)


# --- reported findings -----------------------------------------------------


def test_duplicate_sample_ids_are_reported(dataset: Path) -> None:
    table = read_table(dataset, "train")
    identifiers = table.column("sample_id").to_numpy(zero_copy_only=False)
    duplicated = with_value(table, "sample_id", 5, str(identifiers[0]))
    rewrite_split(dataset, "train", duplicated)

    report = diagnose_dataset(dataset)
    block = report["splits"]["train"]["sample_ids"]
    assert block["duplicate_rows"] == 1
    assert block["unique"] == block["rows"] - 1
    assert block["duplicate_examples"] == [str(identifiers[0])]
    assert report["status"] == "findings"
    assert [finding["check"] for finding in report["findings"]] == ["duplicate_sample_ids"]


def test_sample_id_intersection_across_splits_is_reported(dataset: Path) -> None:
    train_id = str(read_table(dataset, "train").column("sample_id")[0])
    table = with_value(read_table(dataset, "validation"), "sample_id", 3, train_id)
    rewrite_split(dataset, "validation", table)

    report = diagnose_dataset(dataset)
    shared = [
        entry
        for entry in report["cross_split"]["sample_id_intersections"]
        if entry["count"] > 0
    ]
    assert len(shared) == 1
    assert shared[0]["splits"] == ["train", "validation"]
    assert shared[0]["examples"] == [train_id]
    assert any(finding["check"] == "sample_id_intersection" for finding in report["findings"])


def test_a_price_above_the_upper_bound_is_reported(dataset: Path) -> None:
    table = read_table(dataset, "train")
    columns = columns_of(table)
    index = int(np.argmax(columns["option_type"] == "call"))
    upper = float(discounted(columns)[0][index])
    rewrite_split(dataset, "train", with_value(table, "price", index, upper + 1.0))

    report = diagnose_dataset(dataset)
    bounds = report["splits"]["train"]["no_arbitrage"]
    assert bounds["upper_bound_violations"] == 1
    assert bounds["lower_bound_violations"] == 0
    assert bounds["violations"] == 1
    assert bounds["max_upper_violation"] == pytest.approx(1.0, abs=1e-9)
    assert report["status"] == "findings"
    assert any(finding["check"] == "no_arbitrage" for finding in report["findings"])


def test_a_price_below_the_lower_bound_is_reported(dataset: Path) -> None:
    table = read_table(dataset, "train")
    columns = columns_of(table)
    discounted_spot, discounted_strike = discounted(columns)
    intrinsic = np.where(
        columns["option_type"] == "call",
        discounted_spot - discounted_strike,
        discounted_strike - discounted_spot,
    )
    index = int(np.argmax(intrinsic))
    assert intrinsic[index] > 1.0
    rewrite_split(dataset, "train", with_value(table, "price", index, 0.0))

    report = diagnose_dataset(dataset)
    bounds = report["splits"]["train"]["no_arbitrage"]
    assert bounds["lower_bound_violations"] == 1
    assert bounds["upper_bound_violations"] == 0
    assert bounds["max_lower_violation"] == pytest.approx(float(intrinsic[index]))


def test_a_violation_inside_the_tolerance_is_not_counted(dataset: Path) -> None:
    """The tolerance must absorb rounding-scale excess without hiding it."""
    table = read_table(dataset, "train")
    columns = columns_of(table)
    index = int(np.argmax(columns["option_type"] == "call"))
    upper = float(discounted(columns)[0][index])
    nudge = 0.1 * ARBITRAGE_RELATIVE_TOLERANCE * upper
    rewrite_split(dataset, "train", with_value(table, "price", index, upper + nudge))

    report = diagnose_dataset(dataset)
    bounds = report["splits"]["train"]["no_arbitrage"]
    assert bounds["violations"] == 0
    assert bounds["max_upper_violation"] > 0.0
    assert report["status"] == "ok"


def test_domain_coverage_matches_the_declared_manifest_bounds(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    manifest = read_manifest(dataset)
    assert report["dataset"]["domain"] == manifest["domain"]
    for split in SPLIT_NAMES:
        columns = columns_of(read_table(dataset, split))
        coverage = report["splits"][split]["domain_coverage"]
        assert sorted(coverage) == sorted(DOMAIN_FIELDS)
        for field, block in coverage.items():
            assert block["declared"] == manifest["domain"][field]
            assert block["observed_min"] == float(columns[field].min())
            assert block["observed_max"] == float(columns[field].max())
            assert block["rows_below"] == 0
            assert block["rows_above"] == 0


def test_a_manifest_domain_narrower_than_the_data_is_reported(dataset: Path) -> None:
    """The manifest is not hashed, so a lone edit to its domain must be caught."""
    columns = columns_of(read_table(dataset, "train"))
    cut = float(np.median(columns["spot"]))
    manifest = read_manifest(dataset)
    manifest["domain"]["spot"] = [manifest["domain"]["spot"][0], cut]
    write_manifest(dataset, manifest)

    report = diagnose_dataset(dataset)
    coverage = report["splits"]["train"]["domain_coverage"]["spot"]
    assert coverage["rows_above"] == int((columns["spot"] > cut).sum())
    assert coverage["rows_above"] > 0
    assert coverage["rows_below"] == 0
    assert report["status"] == "findings"
    findings = [finding for finding in report["findings"] if finding["check"] == "domain_coverage"]
    assert findings and all(finding["scope"].endswith("/spot") for finding in findings)


def test_a_malformed_manifest_domain_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    manifest["domain"]["volatility"] = [0.8, 0.05]
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="low < high"):
        diagnose_dataset(dataset)


def test_a_manifest_missing_a_domain_field_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    del manifest["domain"]["rate"]
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="manifest 'domain' covers"):
        diagnose_dataset(dataset)


def test_a_missing_label_diagnostics_entry_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    del manifest["label_diagnostics"]["splits"]["validation"]
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match=r"label_diagnostics\.splits covers"):
        diagnose_dataset(dataset)


def test_a_malformed_label_diagnostics_entry_is_rejected(dataset: Path) -> None:
    manifest = read_manifest(dataset)
    manifest["label_diagnostics"]["splits"]["train"]["zero_price_rows"] = -1
    write_manifest(dataset, manifest)
    with pytest.raises(DiagnosticsError, match="is negative"):
        diagnose_dataset(dataset)


def test_a_non_finite_price_is_excluded_from_the_bound_check(dataset: Path) -> None:
    table = read_table(dataset, "train")
    rewrite_split(dataset, "train", with_value(table, "price", 11, np.nan))

    report = diagnose_dataset(dataset)
    bounds = report["splits"]["train"]["no_arbitrage"]
    assert bounds["rows_not_comparable"] == 1
    assert bounds["rows_checked"] == table.num_rows - 1
    assert bounds["violations"] == 0
    assert report["splits"]["train"]["statistics"]["price"]["non_finite"] == 1


def test_a_non_finite_moneyness_lands_in_the_non_finite_band(dataset: Path) -> None:
    table = read_table(dataset, "train")
    rewrite_split(dataset, "train", with_value(table, "log_forward_moneyness", 13, np.nan))

    report = diagnose_dataset(dataset)
    bands = report["splits"]["train"]["moneyness_bands"]
    assert bands["non_finite"]["count"] == 1
    assert sum(band["count"] for band in bands.values()) == table.num_rows
    assert report["splits"]["train"]["derived_statistics"]["absolute_z"]["non_finite"] == 1


def test_non_finite_values_are_counted_and_reported(dataset: Path) -> None:
    table = read_table(dataset, "train")
    rewrite_split(dataset, "train", with_value(table, "theta", 7, np.nan))

    report = diagnose_dataset(dataset)
    block = report["splits"]["train"]
    assert block["statistics"]["theta"]["non_finite"] == 1
    assert block["statistics"]["theta"]["finite"] == block["rows"] - 1
    assert block["statistics"]["theta"]["mean"] is not None
    assert block["non_finite_values"] == 1
    assert any(finding["check"] == "non_finite_values" for finding in report["findings"])
    assert "NaN" not in render_report(report)


def test_duplicate_economic_states_within_a_split_are_reported(dataset: Path) -> None:
    """Identifiers stay unique; the contract does not. Only the state check sees it."""
    table = read_table(dataset, "train")
    rewrite_split(dataset, "train", copy_row(table, table, 0, 1))

    report = diagnose_dataset(dataset)
    block = report["splits"]["train"]
    assert block["sample_ids"]["duplicate_rows"] == 0
    assert block["states"]["duplicate_rows"] == 1
    assert block["states"]["unique"] == block["rows"] - 1
    assert sorted(block["states"]["duplicate_examples"]) == sorted(
        str(table.column("sample_id")[index]) for index in (0, 1)
    )
    assert [finding["check"] for finding in report["findings"]] == ["duplicate_states"]


def test_shared_economic_states_across_splits_are_reported(dataset: Path) -> None:
    train = read_table(dataset, "train")
    validation = read_table(dataset, "validation")
    rewrite_split(dataset, "validation", copy_row(validation, train, 0, 3))

    report = diagnose_dataset(dataset)
    shared = [
        entry for entry in report["cross_split"]["state_intersections"] if entry["count"] > 0
    ]
    assert len(shared) == 1
    assert shared[0]["splits"] == ["train", "validation"]
    assert shared[0]["count"] == 1
    assert shared[0]["examples"] == [str(train.column("sample_id")[0])]
    assert all(
        entry["count"] == 0 for entry in report["cross_split"]["sample_id_intersections"]
    )
    assert any(finding["check"] == "state_intersection" for finding in report["findings"])


def test_a_valid_dataset_shares_no_states_between_splits(dataset: Path) -> None:
    report = diagnose_dataset(dataset)
    intersections = report["cross_split"]["state_intersections"]
    assert len(intersections) == 3
    assert all(entry["count"] == 0 for entry in intersections)
    for split in SPLIT_NAMES:
        block = report["splits"][split]["states"]
        assert block["duplicate_rows"] == 0
        assert block["unique"] == report["splits"][split]["rows"]


def test_a_mislabelled_split_column_is_reported(dataset: Path) -> None:
    table = with_value(read_table(dataset, "validation"), "split", 0, "train")
    rewrite_split(dataset, "validation", table)

    report = diagnose_dataset(dataset)
    assert report["splits"]["validation"]["split_column_mismatches"] == 1
    assert any(finding["check"] == "split_column" for finding in report["findings"])


# --- command line ----------------------------------------------------------


def test_cli_writes_the_report_and_exits_zero(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "nested" / "diagnostics.json"
    status = main(["--dataset", str(dataset), "--output", str(output)])
    captured = capsys.readouterr()

    assert status == 0
    assert "no findings" in captured.out
    assert "diagnosed 1280 rows across 3 splits" in captured.out
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "ok"


def test_cli_reports_findings_with_exit_code_three(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    table = read_table(dataset, "train")
    identifiers = table.column("sample_id").to_numpy(zero_copy_only=False)
    rewrite_split(dataset, "train", with_value(table, "sample_id", 2, str(identifiers[1])))

    output = tmp_path / "diagnostics.json"
    status = main(["--dataset", str(dataset), "--output", str(output)])
    assert status == 3
    assert "duplicate_sample_ids" in capsys.readouterr().out
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "findings"


def test_cli_reports_an_integrity_failure_with_exit_code_two(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (dataset / "train.parquet").unlink()
    output = tmp_path / "diagnostics.json"
    status = main(["--dataset", str(dataset), "--output", str(output)])

    assert status == 2
    assert "missing from" in capsys.readouterr().err
    assert not output.exists()


def test_cli_reports_an_unwritable_output_path(
    dataset: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory", encoding="utf-8")
    status = main(["--dataset", str(dataset), "--output", str(blocked / "diagnostics.json")])

    assert status == 2
    assert "cannot write diagnostics" in capsys.readouterr().err
