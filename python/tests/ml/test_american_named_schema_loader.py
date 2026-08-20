"""Named-schema loading: the American schema is admitted, nothing else is.

The European path is exercised alongside the American one in the same file so
that a change which quietly alters European behaviour fails here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from american_admission_fixtures import (
    rows_for_split,
    table_from_rows,
    write_dataset,
    write_manifest,
)
from differentiable_pricing.data.american_schema import (
    SCHEMA_VERSION as AMERICAN_SCHEMA_VERSION,
)
from differentiable_pricing.ml.dataset import (
    SCHEMA_REGISTRY,
    DatasetLoadError,
    SplitData,
    load_dataset_manifest,
    load_split,
    registered_schema,
)


def test_only_two_schemas_are_registered() -> None:
    assert sorted(SCHEMA_REGISTRY) == [
        "american-option-dataset/1",
        "european-option-dataset/1",
    ]


def test_unregistered_schema_fails_closed() -> None:
    with pytest.raises(DatasetLoadError, match="unregistered schema_version"):
        registered_schema("american-option-dataset/2")
    with pytest.raises(DatasetLoadError, match="unregistered schema_version"):
        registered_schema("")


def test_american_manifest_selects_the_american_schema(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    manifest, identity = load_dataset_manifest(directory)
    assert identity.schema_version == AMERICAN_SCHEMA_VERSION
    assert manifest["label_policy"]["name"] == "american-crr-adjacent-average/1"
    assert len(identity.manifest_sha256) == 64


def test_unknown_schema_in_a_manifest_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    manifest["schema_version"] = "american-option-dataset/99"
    write_manifest(directory, manifest)
    with pytest.raises(DatasetLoadError, match="unregistered schema_version"):
        load_dataset_manifest(directory)


def test_manifest_without_a_schema_version_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    directory.mkdir()
    (directory / "manifest.json").write_text(json.dumps({"files": []}), encoding="utf-8")
    with pytest.raises(DatasetLoadError, match="does not declare a string"):
        load_dataset_manifest(directory)


def test_american_split_exposes_target_and_comparator(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    manifest, _ = load_dataset_manifest(directory)
    split = load_split(directory, manifest, "train", evaluation=False)

    assert split.schema_version == AMERICAN_SCHEMA_VERSION
    assert split.target_column == "american_price"
    assert split.comparator_column == "european_crr_price"
    assert split.features.shape == (split.rows, 7)
    assert split.target.shape == (split.rows,)

    comparator = split.european_comparator
    assert comparator is not None
    assert np.all(split.target >= comparator)


def test_american_evaluation_columns_include_the_premium(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    manifest, _ = load_dataset_manifest(directory)
    split = load_split(directory, manifest, "interpolation_test", evaluation=True)
    for name in (
        "american_price",
        "european_crr_price",
        "early_exercise_premium",
        "adjacent_step_gap",
        "intrinsic_value",
        "european_black_scholes_price",
    ):
        assert name in split.columns
    premium = split.columns["early_exercise_premium"]
    assert np.allclose(premium, split.target - split.european_comparator, atol=0.0)


def test_option_type_is_encoded_and_ordered_first(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    manifest, _ = load_dataset_manifest(directory)
    split = load_split(directory, manifest, "train", evaluation=False)
    encoded = split.features[:, 0]
    assert set(np.unique(encoded)) <= {-1.0, 1.0}
    assert np.all((split.option_types == "call") == (encoded == 1.0))


def test_unknown_option_type_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    rows = rows_for_split("train", 6)
    rows[0]["option_type"] = "binary"
    _rewrite(directory, manifest, "train", rows)
    manifest, _ = load_dataset_manifest(directory)
    with pytest.raises(DatasetLoadError, match="unknown option_type"):
        load_split(directory, manifest, "train", evaluation=False)


def test_non_finite_data_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    rows = rows_for_split("train", 6)
    rows[2]["american_price"] = float("inf")
    _rewrite(directory, manifest, "train", rows)
    manifest, _ = load_dataset_manifest(directory)
    with pytest.raises(DatasetLoadError, match="non-finite"):
        load_split(directory, manifest, "train", evaluation=False)


def test_digest_mismatch_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    manifest, _ = load_dataset_manifest(directory)
    (directory / "train.parquet").write_bytes(
        (directory / "validation.parquet").read_bytes()
    )
    with pytest.raises(DatasetLoadError, match="sha256 mismatch"):
        load_split(directory, manifest, "train", evaluation=False)


def test_unknown_split_name_is_rejected(tmp_path: Path) -> None:
    directory = tmp_path / "dataset"
    write_dataset(directory)
    manifest, _ = load_dataset_manifest(directory)
    with pytest.raises(DatasetLoadError, match="is not a partition of schema"):
        load_split(directory, manifest, "boundary_test", evaluation=False)


def test_a_foreign_schema_file_under_an_american_manifest_is_rejected(
    tmp_path: Path,
) -> None:
    """A manifest that says American over a file that is not fails closed."""
    directory = tmp_path / "dataset"
    manifest = write_dataset(directory)
    import hashlib

    import pyarrow as pa
    import pyarrow.parquet as pq

    foreign = pa.table({"spot": pa.array([1.0, 2.0], type=pa.float64())})
    pq.write_table(foreign, directory / "train.parquet", compression="snappy", version="2.6")
    payload = (directory / "train.parquet").read_bytes()
    for entry in manifest["files"]:
        if entry["split"] == "train":
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["rows"] = 2
    manifest["row_counts"]["train"] = 2
    manifest["label_diagnostics"]["splits"]["train"]["rows"] = 2
    write_manifest(directory, manifest)
    manifest, _ = load_dataset_manifest(directory)
    with pytest.raises(DatasetLoadError, match="does not match the 'american-option"):
        load_split(directory, manifest, "train", evaluation=False)


def test_split_data_defaults_stay_european() -> None:
    """Existing European constructions keep working without new arguments."""
    split = SplitData(
        split="train",
        features=np.zeros((3, 7)),
        option_types=np.array(["call", "put", "call"]),
        columns={"price": np.array([1.0, 2.0, 3.0])},
    )
    assert split.schema_version == "european-option-dataset/1"
    assert split.target_column == "price"
    assert split.european_comparator is None
    assert split.target.tolist() == [1.0, 2.0, 3.0]


def _rewrite(directory: Path, manifest: dict, split: str, rows: list) -> None:
    import hashlib

    import pyarrow.parquet as pq

    table = table_from_rows(rows)
    pq.write_table(
        table, directory / f"{split}.parquet", compression="snappy", version="2.6"
    )
    payload = (directory / f"{split}.parquet").read_bytes()
    for entry in manifest["files"]:
        if entry["split"] == split:
            entry["sha256"] = hashlib.sha256(payload).hexdigest()
            entry["rows"] = table.num_rows
    manifest["row_counts"][split] = table.num_rows
    manifest["label_diagnostics"]["splits"][split]["rows"] = table.num_rows
    write_manifest(directory, manifest)
