"""The recovered ``american-option-dataset/1`` schema contract.

These tests pin what task 9E ported from the unmerged generator branch. They
run on small fixtures; none of them touches the Git-ignored 47 MB candidate
dataset, which is absent in CI.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

import pyarrow as pa
import pytest
from american_admission_fixtures import write_dataset, write_manifest
from differentiable_pricing.data import american_schema
from differentiable_pricing.data.american_schema import (
    CAVEATS,
    FEATURE_ORDER,
    REPRESENTATION,
    REPRESENTATIONS_REJECTED_FOR_AMERICAN,
    SCHEMA_VERSION,
    TABLE_SCHEMA,
    UNITS_AND_CONVENTIONS,
    AmericanManifestError,
    load_american_manifest,
)


def test_schema_version_and_column_layout_are_pinned() -> None:
    assert SCHEMA_VERSION == "american-option-dataset/1"
    assert TABLE_SCHEMA.names == [
        "sample_id",
        "split",
        "stratum",
        "exercise_regime",
        "option_type",
        "label_policy",
        "spot",
        "strike",
        "log_moneyness",
        "maturity",
        "rate",
        "dividend_yield",
        "volatility",
        "american_price",
        "american_price_at_steps",
        "american_price_at_steps_plus_one",
        "adjacent_step_gap",
        "european_crr_price",
        "european_crr_price_at_steps",
        "european_crr_price_at_steps_plus_one",
        "early_exercise_premium",
        "european_black_scholes_price",
        "intrinsic_value",
        "risk_neutral_probability",
        "exercise_activity_fraction",
        "label_steps",
        "early_exercise_nodes",
        "earliest_exercise_step",
        "exercise_boundary_layers",
    ]


def test_every_column_is_typed_and_non_nullable() -> None:
    for field in TABLE_SCHEMA:
        assert not field.nullable, field.name
        if field.name in american_schema.STRING_COLUMNS:
            assert field.type == pa.string()
        elif field.name in american_schema.INTEGER_COLUMNS:
            assert field.type == pa.int64()
        else:
            assert field.type == pa.float64()


def test_every_column_carries_a_recovered_definition() -> None:
    documented = set(UNITS_AND_CONVENTIONS)
    missing = sorted(set(TABLE_SCHEMA.names) - documented)
    assert missing == []


def test_schema_declares_no_greek_and_no_discrete_dividend() -> None:
    names = set(TABLE_SCHEMA.names)
    assert names.isdisjoint({"delta", "gamma", "vega", "theta", "rho"})
    assert names.isdisjoint({"dividend_amount", "ex_date", "dividend_count"})
    assert "continuous" in UNITS_AND_CONVENTIONS["dividend_yield"]
    assert "NOT a discrete" in UNITS_AND_CONVENTIONS["dividend_yield"]
    assert any("discrete cash-dividend schedule" in caveat for caveat in CAVEATS)


def test_target_and_comparator_are_named() -> None:
    assert american_schema.TARGET_COLUMN == "american_price"
    assert american_schema.EUROPEAN_COMPARATOR_COLUMN == "european_crr_price"
    assert american_schema.TARGET_COLUMN in TABLE_SCHEMA.names
    assert american_schema.EUROPEAN_COMPARATOR_COLUMN in TABLE_SCHEMA.names


def test_representation_carries_every_state_variable() -> None:
    assert REPRESENTATION == "american_raw_physical_v1"
    assert set(FEATURE_ORDER) == {
        "option_type",
        "spot",
        "strike",
        "maturity",
        "rate",
        "dividend_yield",
        "volatility",
    }
    assert "forward_normalized_v1" in REPRESENTATIONS_REJECTED_FOR_AMERICAN


PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]

RECOVERED_CONFIGS: Final = {
    "american_option_dataset_v1.toml": (
        "d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98"
    ),
    "american_label_policy_pilot_v1.toml": (
        "7858e1d6cd283f6e35cbd843a0b5cf3aaf02628dcb34e1cd727a606fff7f248b"
    ),
    "american_label_policy_pilot_v2.toml": (
        "c5eafa09b19c41dc22044bbee8a43469c077e25c95ce17a24cca028b1910944c"
    ),
}
"""Digests the recovered configurations must keep.

The first is the digest ``data/american-option-v1/manifest.json`` records for
the configuration it was generated under; the other two are the digests the
label-policy pilot reports record. Pinning them here is what makes the recovery
verifiable: if one of these files drifts, the dataset's provenance link to it
silently breaks, and this test is where that is caught.
"""


@pytest.mark.parametrize(("name", "digest"), sorted(RECOVERED_CONFIGS.items()))
def test_recovered_configuration_digest_is_unchanged(name: str, digest: str) -> None:
    path = PROJECT_ROOT / "configs" / name
    assert path.is_file(), f"recovered configuration '{name}' is missing"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_valid_fixture_manifest_loads(tmp_path: Path) -> None:
    write_dataset(tmp_path / "dataset")
    manifest = load_american_manifest(tmp_path / "dataset")
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert sorted(manifest["row_counts"]) == [
        "interpolation_test",
        "train",
        "validation",
    ]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: m.__setitem__("schema_version", "european-option-dataset/1"),
         "only understands"),
        (lambda m: m.__setitem__("columns", ["spot"]), "does not match the pinned"),
        (lambda m: m["label_policy"].__setitem__("name", "other/1"), "label_policy.name"),
        (lambda m: m["label_policy"].__setitem__("steps", 0), "steps must be positive"),
        (lambda m: m["oracle"].__setitem__("name", "dp::not_a_tree"), "not a recognised"),
        (lambda m: m["oracle"].__setitem__("crr_source_sha256", "abc"), "SHA-256"),
        (lambda m: m["config"].__setitem__("sha256", "nope"), "SHA-256"),
        (lambda m: m.__setitem__("generator_version", ""), "non-empty"),
        (lambda m: m["row_counts"].pop("validation"), "row_counts"),
        (lambda m: m["domain"].pop("volatility"), "domain"),
        (lambda m: m["domain"].__setitem__("rate", [0.5, 0.1]), "requires low < high"),
        (lambda m: m["files"].pop(), "does not declare partition"),
        (lambda m: m["label_diagnostics"]["splits"]["train"].pop("rows"), "expected"),
        (lambda m: m.pop("oracle"), "missing required key 'oracle'"),
    ],
)
def test_manifest_rejections(tmp_path: Path, mutate, expected: str) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    mutate(manifest)
    write_manifest(directory, manifest)
    with pytest.raises(AmericanManifestError, match=expected):
        load_american_manifest(directory)


def test_manifest_rejects_two_splits_sharing_one_file(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    manifest["files"][1]["file"] = manifest["files"][0]["file"]
    write_manifest(directory, manifest)
    with pytest.raises(AmericanManifestError, match="more than one partition"):
        load_american_manifest(directory)


def test_manifest_rejects_row_count_disagreement(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    manifest["files"][0]["rows"] += 1
    write_manifest(directory, manifest)
    with pytest.raises(AmericanManifestError, match=r"manifest\.row_counts"):
        load_american_manifest(directory)


def test_manifest_must_be_a_json_object(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(AmericanManifestError, match="must be a JSON object"):
        load_american_manifest(directory)
