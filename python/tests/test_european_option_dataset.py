"""Tests for the deterministic European-option dataset generator.

All tests use temporary directories and small row counts. The oracle is the
compiled C++ Black-Scholes binding; nothing here reimplements it.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
import math
import platform
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from differentiable_pricing._core import black_scholes
from differentiable_pricing.data import generate as generate_module
from differentiable_pricing.data.config import (
    DOMAIN_FIELDS,
    GENERATOR_VERSION,
    SCHEMA_VERSION,
    SPLIT_NAMES,
    ConfigError,
    DatasetConfig,
    load_dataset_config,
    parse_dataset_config,
)
from differentiable_pricing.data.generate import (
    LABEL_DIAGNOSTICS_THRESHOLDS,
    LABEL_FIELDS,
    MANIFEST_NAME,
    GenerationError,
    build_table,
    generate_dataset,
    main,
    split_generator,
)

CONFIG_TEMPLATE = f"""
schema_version = "{SCHEMA_VERSION}"
generator_version = "{GENERATOR_VERSION}"

[dataset]
name = "european-option-test"
oracle = "dp::black_scholes"
base_seed = 424242

[dataset.rows]
train = 96
validation = 48
interpolation_test = 48

[domain]
spot = [50.0, 150.0]
log_forward_moneyness = [-0.4, 0.4]
maturity = [0.019178082191780823, 2.0]
rate = [-0.01, 0.08]
dividend_yield = [0.0, 0.05]
volatility = [0.05, 0.8]
"""

FLOAT_COLUMNS = (
    "spot",
    "strike",
    "log_forward_moneyness",
    "maturity",
    "rate",
    "dividend_yield",
    "volatility",
    *LABEL_FIELDS,
)


def base_document() -> dict[str, Any]:
    """A valid parsed configuration document, safe for callers to mutate."""
    return tomllib.loads(CONFIG_TEMPLATE)


def write_config(directory: Path, text: str = CONFIG_TEMPLATE) -> Path:
    path = directory / "dataset.toml"
    path.write_text(text, encoding="utf-8")
    return path


def load_config(directory: Path) -> DatasetConfig:
    return load_dataset_config(write_config(directory))


def read_split(directory: Path, split: str) -> dict[str, np.ndarray]:
    table = pq.read_table(directory / f"{split}.parquet")
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.column_names}


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict[str, Any]]:
    directory = tmp_path_factory.mktemp("dataset")
    config = load_config(directory)
    manifest = generate_dataset(config, directory / "out")
    return directory / "out", manifest


# --- structure -------------------------------------------------------------


def test_generates_one_parquet_file_per_split_plus_manifest(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    directory, _ = dataset
    written = sorted(path.name for path in directory.iterdir())
    assert written == sorted([f"{split}.parquet" for split in SPLIT_NAMES] + [MANIFEST_NAME])


def test_columns_and_dtypes_match_the_contract(dataset: tuple[Path, dict[str, Any]]) -> None:
    directory, _ = dataset
    schema = pq.read_schema(directory / "train.parquet")
    assert schema.names == [
        "sample_id",
        "split",
        "option_type",
        "spot",
        "strike",
        "log_forward_moneyness",
        "maturity",
        "rate",
        "dividend_yield",
        "volatility",
        "price",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]
    for name in FLOAT_COLUMNS:
        assert schema.field(name).type == pa.float64(), name
    for name in ("sample_id", "split", "option_type"):
        assert schema.field(name).type == pa.string(), name


def test_row_counts_match_the_configuration(dataset: tuple[Path, dict[str, Any]]) -> None:
    directory, manifest = dataset
    document = base_document()
    for split in SPLIT_NAMES:
        expected = document["dataset"]["rows"][split]
        assert manifest["row_counts"][split] == expected
        assert pq.read_metadata(directory / f"{split}.parquet").num_rows == expected


# --- determinism and partitioning -----------------------------------------


def test_regeneration_is_byte_identical(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    first = generate_dataset(config, tmp_path / "first")
    second = generate_dataset(config, tmp_path / "second")

    assert first == second
    for split in SPLIT_NAMES:
        name = f"{split}.parquet"
        left = (tmp_path / "first" / name).read_bytes()
        right = (tmp_path / "second" / name).read_bytes()
        assert hashlib.sha256(left).hexdigest() == hashlib.sha256(right).hexdigest()
        assert pq.read_table(tmp_path / "first" / name).equals(
            pq.read_table(tmp_path / "second" / name)
        )
    assert (tmp_path / "first" / MANIFEST_NAME).read_bytes() == (
        tmp_path / "second" / MANIFEST_NAME
    ).read_bytes()


def test_a_different_seed_changes_the_samples(tmp_path: Path) -> None:
    baseline = load_config(tmp_path)
    document = base_document()
    document["dataset"]["base_seed"] = baseline.base_seed + 1
    altered = parse_dataset_config(document, source_name="dataset.toml", source_sha256="0" * 64)

    generate_dataset(baseline, tmp_path / "a")
    generate_dataset(altered, tmp_path / "b")
    assert not np.array_equal(
        read_split(tmp_path / "a", "train")["spot"], read_split(tmp_path / "b", "train")["spot"]
    )


def test_split_streams_are_independent(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    streams = {
        split: split_generator(config.base_seed, split).uniform(size=64) for split in SPLIT_NAMES
    }
    for left, right in itertools.combinations(SPLIT_NAMES, 2):
        assert not np.array_equal(streams[left], streams[right])


def test_sample_ids_are_unique_within_and_across_splits(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    directory, manifest = dataset
    seen: list[str] = []
    for split in SPLIT_NAMES:
        columns = read_split(directory, split)
        ids = list(columns["sample_id"])
        assert len(set(ids)) == len(ids)
        assert set(columns["split"]) == {split}
        seen.extend(ids)
    assert len(set(seen)) == len(seen) == sum(manifest["row_counts"].values())


def test_partitions_do_not_share_sampled_states(dataset: tuple[Path, dict[str, Any]]) -> None:
    directory, _ = dataset
    states = {
        split: {
            tuple(float(read_split(directory, split)[field][index]) for field in DOMAIN_FIELDS)
            for index in range(read_split(directory, split)["spot"].size)
        }
        for split in SPLIT_NAMES
    }
    for left, right in itertools.combinations(SPLIT_NAMES, 2):
        assert states[left].isdisjoint(states[right])


# --- values ----------------------------------------------------------------


def test_values_are_finite_and_inside_the_configured_domain(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    directory, _ = dataset
    document = base_document()["domain"]
    for split in SPLIT_NAMES:
        columns = read_split(directory, split)
        for name in FLOAT_COLUMNS:
            assert np.isfinite(columns[name]).all(), f"{split}.{name}"
        for field in DOMAIN_FIELDS:
            low, high = document[field]
            values = columns[field]
            assert values.min() >= low, f"{split}.{field}"
            assert values.max() <= high, f"{split}.{field}"
        assert (columns["strike"] > 0.0).all()
        assert set(columns["option_type"]) <= {"call", "put"}


def test_option_type_is_balanced(dataset: tuple[Path, dict[str, Any]]) -> None:
    directory, _ = dataset
    for split in SPLIT_NAMES:
        option_type = read_split(directory, split)["option_type"]
        calls = int((option_type == "call").sum())
        assert abs(2 * calls - option_type.size) <= 1


def test_strike_is_derived_from_the_forward_relation(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    directory, _ = dataset
    for split in SPLIT_NAMES:
        columns = read_split(directory, split)
        forward = columns["spot"] * np.exp(
            (columns["rate"] - columns["dividend_yield"]) * columns["maturity"]
        )
        expected = forward * np.exp(-columns["log_forward_moneyness"])
        np.testing.assert_allclose(columns["strike"], expected, rtol=1e-15, atol=0.0)
        recovered = np.log(forward / columns["strike"])
        np.testing.assert_allclose(
            recovered, columns["log_forward_moneyness"], rtol=0.0, atol=1e-12
        )


def test_labels_match_direct_calls_to_the_cpp_oracle(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    directory, _ = dataset
    for split in SPLIT_NAMES:
        columns = read_split(directory, split)
        for index in range(columns["spot"].size):
            expected = black_scholes(
                str(columns["option_type"][index]),
                float(columns["spot"][index]),
                float(columns["strike"][index]),
                float(columns["maturity"][index]),
                float(columns["rate"][index]),
                float(columns["dividend_yield"][index]),
                float(columns["volatility"][index]),
            )
            for name in LABEL_FIELDS:
                assert columns[name][index] == expected[name], f"{split}[{index}].{name}"


def test_prices_respect_european_no_arbitrage_bounds(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    for split in SPLIT_NAMES:
        columns = read_split(dataset[0], split)
        discounted_spot = columns["spot"] * np.exp(
            -columns["dividend_yield"] * columns["maturity"]
        )
        discounted_strike = columns["strike"] * np.exp(-columns["rate"] * columns["maturity"])
        is_call = columns["option_type"] == "call"
        lower = np.where(
            is_call,
            np.maximum(discounted_spot - discounted_strike, 0.0),
            np.maximum(discounted_strike - discounted_spot, 0.0),
        )
        upper = np.where(is_call, discounted_spot, discounted_strike)
        tolerance = 1e-12 * np.maximum(discounted_spot, discounted_strike)
        assert (columns["price"] >= lower - tolerance).all()
        assert (columns["price"] <= upper + tolerance).all()
        assert (columns["gamma"] >= 0.0).all()
        assert (columns["vega"] >= 0.0).all()
        assert np.all(columns["delta"][is_call] >= 0.0)
        assert np.all(columns["delta"][~is_call] <= 0.0)


def test_put_call_parity_holds_on_generated_states(dataset: tuple[Path, dict[str, Any]]) -> None:
    columns = read_split(dataset[0], "interpolation_test")
    for index in range(min(24, columns["spot"].size)):
        arguments = (
            float(columns["spot"][index]),
            float(columns["strike"][index]),
            float(columns["maturity"][index]),
            float(columns["rate"][index]),
            float(columns["dividend_yield"][index]),
            float(columns["volatility"][index]),
        )
        call = black_scholes("call", *arguments)["price"]
        put = black_scholes("put", *arguments)["price"]
        spot, strike, maturity, rate, dividend_yield, _ = arguments
        parity = spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity)
        assert call - put == pytest.approx(parity, abs=1e-12 * max(spot, strike))


# --- manifest --------------------------------------------------------------


def test_manifest_records_valid_provenance_and_hashes(tmp_path: Path) -> None:
    config_path = write_config(tmp_path)
    config = load_dataset_config(config_path)
    output = tmp_path / "out"
    returned = generate_dataset(config, output)

    manifest = json.loads((output / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest == returned
    assert manifest["schema_version"] == SCHEMA_VERSION
    assert manifest["generator_version"] == GENERATOR_VERSION
    assert manifest["seed"]["base_seed"] == config.base_seed
    assert manifest["config"]["file"] == config_path.name
    assert (
        manifest["config"]["sha256"]
        == hashlib.sha256(config_path.read_bytes()).hexdigest()
        == config.source_sha256
    )
    assert manifest["oracle"]["name"] == "dp::black_scholes"
    assert manifest["oracle"]["language"] == "C++"
    assert set(manifest["units_and_conventions"]) >= set(DOMAIN_FIELDS) | set(LABEL_FIELDS)
    assert manifest["units_and_conventions"]["vega"].startswith("d price / d volatility")

    recorded = {entry["split"]: entry for entry in manifest["files"]}
    assert set(recorded) == set(SPLIT_NAMES)
    for split, entry in recorded.items():
        path = output / entry["file"]
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert entry["rows"] == manifest["row_counts"][split] == pq.read_metadata(path).num_rows


def _keys(node: Any) -> list[str]:
    if isinstance(node, dict):
        return [key for name, value in node.items() for key in [name, *_keys(value)]]
    if isinstance(node, list):
        return [key for item in node for key in _keys(item)]
    return []


def test_manifest_holds_no_wall_clock_keys(dataset: tuple[Path, dict[str, Any]]) -> None:
    manifest = json.loads((dataset[0] / MANIFEST_NAME).read_text(encoding="utf-8"))
    banned = {"timestamp", "generated_at", "created_at", "date", "time", "wall_clock", "run_id"}
    assert banned.isdisjoint({key.lower() for key in _keys(manifest)})


def test_manifest_records_the_generating_toolchain(dataset: tuple[Path, dict[str, Any]]) -> None:
    _, manifest = dataset
    assert manifest["runtime"] == {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
    }
    assert "float64 rounding" in manifest["oracle"]["numerical_error"]


def test_manifest_quantifies_the_degenerate_label_tail(
    dataset: tuple[Path, dict[str, Any]],
) -> None:
    directory, manifest = dataset
    diagnostics = manifest["label_diagnostics"]
    assert diagnostics["thresholds"] == dict(LABEL_DIAGNOSTICS_THRESHOLDS)
    for split in SPLIT_NAMES:
        columns = read_split(directory, split)
        recorded = diagnostics["splits"][split]
        assert recorded["rows"] == columns["price"].size
        assert recorded["zero_price_rows"] == int((columns["price"] == 0.0).sum())
        assert (
            recorded["near_zero_price_rows"]
            == int((columns["price"] <= LABEL_DIAGNOSTICS_THRESHOLDS["near_zero_price"]).sum())
        )
        delta = np.abs(columns["delta"])
        bound = np.exp(-columns["dividend_yield"] * columns["maturity"])
        tolerance = LABEL_DIAGNOSTICS_THRESHOLDS["saturated_delta"]
        expected = int(((delta <= tolerance) | (bound - delta <= tolerance)).sum())
        assert recorded["saturated_delta_rows"] == expected


def test_config_hash_changes_when_the_configuration_changes(tmp_path: Path) -> None:
    first = load_dataset_config(write_config(tmp_path))
    altered = CONFIG_TEMPLATE.replace("base_seed = 424242", "base_seed = 424243")
    second = load_dataset_config(write_config(tmp_path, altered))
    assert first.source_sha256 != second.source_sha256


# --- configuration validation ---------------------------------------------


def _without(document: Mapping[str, Any], *path: str) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(document))
    cursor: Any = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    del cursor[path[-1]]
    return mutated


def _with(document: Mapping[str, Any], value: Any, *path: str) -> dict[str, Any]:
    mutated = copy.deepcopy(dict(document))
    cursor: Any = mutated
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = value
    return mutated


def invalid_documents() -> list[tuple[str, dict[str, Any], str]]:
    document = base_document()
    return [
        ("unknown top-level key", _with(document, 1, "extra"), "unknown keys"),
        ("unknown dataset key", _with(document, 1, "dataset", "extra"), "unknown keys"),
        ("unknown split", _with(document, 8, "dataset", "rows", "holdout"), "unknown keys"),
        ("unknown domain field", _with(document, [0.0, 1.0], "domain", "beta"), "unknown keys"),
        ("missing split", _without(document, "dataset", "rows", "validation"), "validation"),
        ("missing domain field", _without(document, "domain", "volatility"), "volatility"),
        ("missing dataset table", _without(document, "dataset"), "dataset"),
        ("wrong schema version", _with(document, "european-option-dataset/9", "schema_version"),
         "unsupported schema_version"),
        ("wrong generator version", _with(document, "9.9.9", "generator_version"),
         "generator_version"),
        ("zero rows", _with(document, 0, "dataset", "rows", "train"), "must be positive"),
        ("negative rows", _with(document, -5, "dataset", "rows", "train"), "must be positive"),
        ("boolean rows", _with(document, True, "dataset", "rows", "train"), "must be an integer"),
        ("float rows", _with(document, 12.0, "dataset", "rows", "train"), "must be an integer"),
        ("negative seed", _with(document, -1, "dataset", "base_seed"), "non-negative"),
        ("unknown oracle", _with(document, "dp::black_sholes", "dataset", "oracle"),
         "is not a known oracle"),
        ("empty dataset name", _with(document, "", "dataset", "name"), "non-empty string"),
        ("inverted bounds", _with(document, [1.0, -1.0], "domain", "rate"), "low < high"),
        ("degenerate bounds", _with(document, [0.2, 0.2], "domain", "volatility"), "low < high"),
        ("non-positive spot", _with(document, [0.0, 150.0], "domain", "spot"), "must be positive"),
        ("non-positive maturity", _with(document, [-1.0, 2.0], "domain", "maturity"),
         "must be positive"),
        ("non-positive volatility", _with(document, [0.0, 0.8], "domain", "volatility"),
         "must be positive"),
        ("non-finite bound", _with(document, [0.05, math.inf], "domain", "volatility"),
         "must be finite"),
        ("bounds not a pair", _with(document, [1.0, 2.0, 3.0], "domain", "rate"), "two-element"),
        ("bounds not an array", _with(document, 0.5, "domain", "rate"), "two-element"),
        ("bound not a number", _with(document, ["a", "b"], "domain", "rate"), "must be a number"),
        ("rows not a table", _with(document, 96, "dataset", "rows"), "must be a table"),
    ]


@pytest.mark.parametrize(
    ("document", "message"),
    [pytest.param(doc, msg, id=name) for name, doc, msg in invalid_documents()],
)
def test_invalid_configuration_is_rejected(document: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        parse_dataset_config(document, source_name="dataset.toml", source_sha256="0" * 64)


def test_valid_configuration_is_accepted() -> None:
    config = parse_dataset_config(
        base_document(), source_name="dataset.toml", source_sha256="0" * 64
    )
    assert config.rows["train"] == 96
    assert config.domain["spot"].contains(50.0)
    assert not config.domain["spot"].contains(49.0)


def test_missing_configuration_file_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="cannot read configuration"):
        load_dataset_config(tmp_path / "absent.toml")


def test_malformed_toml_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("this is not = = toml", encoding="utf-8")
    with pytest.raises(ConfigError, match="not valid UTF-8 TOML"):
        load_dataset_config(path)


# --- output safety and CLI -------------------------------------------------


def test_existing_output_is_not_overwritten_by_default(tmp_path: Path) -> None:
    config = load_config(tmp_path)
    output = tmp_path / "out"
    generate_dataset(config, output)
    with pytest.raises(GenerationError, match="refusing to overwrite"):
        generate_dataset(config, output)
    generate_dataset(config, output, overwrite=True)


@pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")
def test_oracle_rejection_is_reported_as_a_generation_error(tmp_path: Path) -> None:
    """Fields can be individually valid yet jointly overflow the forward."""
    text = CONFIG_TEMPLATE.replace("rate = [-0.01, 0.08]", "rate = [600.0, 700.0]").replace(
        "maturity = [0.019178082191780823, 2.0]", "maturity = [1.5, 2.0]"
    )
    config = load_dataset_config(write_config(tmp_path, text))
    with pytest.raises(GenerationError, match="oracle rejected split 'train' row"):
        generate_dataset(config, tmp_path / "out")


def test_out_of_domain_samples_are_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = load_config(tmp_path)
    original = generate_module.sample_inputs

    def leaking_sample(inner: DatasetConfig, split: str) -> dict[str, np.ndarray]:
        columns = original(inner, split)
        columns["volatility"] = columns["volatility"] + 10.0
        return columns

    monkeypatch.setattr(generate_module, "sample_inputs", leaking_sample)
    with pytest.raises(GenerationError, match="leaves the configured domain"):
        build_table(config, "train")


def test_unwritable_output_path_is_reported(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocking_file = tmp_path / "out"
    blocking_file.write_text("not a directory", encoding="utf-8")
    config_path = write_config(tmp_path)
    with pytest.raises(GenerationError, match="cannot write dataset"):
        generate_dataset(load_dataset_config(config_path), blocking_file)

    status = main(["--config", str(config_path), "--output", str(blocking_file)])
    assert status == 2
    assert "cannot write dataset" in capsys.readouterr().err


def test_cli_generates_the_dataset(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config_path = write_config(tmp_path)
    output = tmp_path / "european-option-test"
    status = main(["--config", str(config_path), "--output", str(output)])
    captured = capsys.readouterr()

    assert status == 0
    assert "wrote 192 rows" in captured.out
    assert (output / MANIFEST_NAME).is_file()
    for split in SPLIT_NAMES:
        assert (output / f"{split}.parquet").is_file()


def test_cli_rejects_an_invalid_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = write_config(tmp_path, CONFIG_TEMPLATE.replace("train = 96", "train = 0"))
    status = main(["--config", str(path), "--output", str(tmp_path / "out")])

    assert status == 2
    assert "must be positive" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()
