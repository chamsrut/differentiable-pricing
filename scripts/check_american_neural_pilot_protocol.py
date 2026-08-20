#!/usr/bin/env python3
"""Strict offline validator for the locked Task 9G pilot protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL: Final = PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml"
SCHEMA: Final = "american-neural-pilot-protocol/1"
EXPECTED_PATH_KEYS: Final = frozenset(
    {
        "protocol",
        "training_config",
        "acceptance_config",
        "latency_config",
        "iv_config",
        "dataset",
        "dataset_manifest",
        "source_artifact",
        "output_directory",
        "execution_ledger",
        "raw_report",
        "final_report",
        "final_failure_report",
        "result_snapshot",
    }
)
EXPECTED_TOP_KEYS: Final = frozenset(
    {
        "schema_version",
        "name",
        "paths",
        "dataset",
        "source_artifact",
        "model",
        "lifecycle",
        "output_schemas",
        "environment_metadata",
        "limitations",
        "independent_pde_check",
        "tracked_inputs",
    }
)
MANDATED_DATASET = {
    "schema_version": "american-option-dataset/1",
    "generator_version": "1.0.0",
    "config_sha256": "d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98",
    "manifest_sha256": "252858929f3a981717dafc54c1639e1bd8159c5ad000aec7a0cc0cf24e9630d2",
    "train_sha256": "ff7a114f5bee260dba04c99afc78b0c9098fd9dfdd18c7a9dc952ce249ef50a9",
    "validation_sha256": "6f53a72e26a917f44324d3df86bffc4924eaf4739655f0563e63a0ad91f839c9",
    "locked_final_sha256": "f006a17cf5818f5950b0402e8942e6738726f2547e2e3c309038a836a1ed1058",
}
MANDATED_SOURCE = {
    "manifest_sha256": "054ca945de851b4e5b2a10455e93be44546770410bcf698e0051175fe8d19f51",
    "weights_sha256": "42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8",
}


class ProtocolError(RuntimeError):
    """Raised when any locked protocol field is absent, mutable, or inconsistent."""


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], where: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        raise ProtocolError(f"{where} keys differ: missing={missing}, unknown={unknown}")


def _table(value: Any, where: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{where} must be a table")
    return value


def _sequence(value: Any, where: str) -> Sequence[Any]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise ProtocolError(f"{where} must be an array")
    return value


def _digest(value: Any, where: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ProtocolError(f"{where} must be a lowercase SHA-256 digest")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_toml(path: Path, where: str) -> dict[str, Any]:
    try:
        with path.open("rb") as stream:
            value = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ProtocolError(f"cannot load {where} '{path}': {error}") from error
    if not isinstance(value, dict):
        raise ProtocolError(f"{where} must be a TOML table")
    return value


def _resolve(root: Path, relative: Any, where: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ProtocolError(f"{where} must be a non-empty repository-relative path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ProtocolError(f"{where} escapes the repository") from error
    return path


def _validate_leaf_configs(protocol: Mapping[str, Any], root: Path) -> dict[str, Any]:
    paths = protocol["paths"]
    training = _load_toml(
        _resolve(root, paths["training_config"], "training path"), "training config"
    )
    acceptance = _load_toml(
        _resolve(root, paths["acceptance_config"], "acceptance path"), "acceptance config"
    )
    latency = _load_toml(_resolve(root, paths["latency_config"], "latency path"), "latency config")
    iv = _load_toml(_resolve(root, paths["iv_config"], "IV path"), "IV config")
    _exact_keys(
        training,
        frozenset(
            {
                "schema_version",
                "experiment_name",
                "representation",
                "physical_feature_order",
                "network_feature_order",
                "target",
                "physical_reconstruction",
                "dtype",
                "device",
                "architecture",
                "arms",
                "row_selection",
                "standardization",
                "optimizer",
                "training",
                "checkpoint",
                "exact_lift",
            }
        ),
        "training config",
    )
    for name, keys in {
        "architecture": {"hidden_dimensions", "activation", "output_dimension"},
        "arms": {
            "names",
            "scratch_seed_label",
            "scratch_seed",
            "transfer_seed_label",
            "transfer_seed",
            "seed_derivation",
        },
        "row_selection": {"partition", "row_budget", "rule", "salt"},
        "standardization": {
            "fit_partition",
            "population_standard_deviation_ddof",
            "shared_between_arms",
        },
        "optimizer": {
            "name",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "amsgrad",
            "maximize",
            "foreach",
            "capturable",
            "differentiable",
            "fused",
            "schedule",
            "minimum_learning_rate",
            "schedule_period_epochs",
            "schedule_last_epoch",
        },
        "training": {
            "objective",
            "batch_size",
            "epochs",
            "num_threads",
            "deterministic_algorithms",
            "shuffle",
        },
        "checkpoint": {
            "selection_partition",
            "metric",
            "rule",
            "early_stopping",
        },
        "exact_lift": {
            "source_representation",
            "source_output",
            "relative_tolerance",
            "absolute_tolerance",
            "probes",
        },
    }.items():
        _exact_keys(_table(training[name], f"training.{name}"), frozenset(keys), f"training.{name}")
    if training.get("schema_version") != "american-neural-pilot-training/1":
        raise ProtocolError("training config schema differs")
    if training.get("representation") != "american_forward_carry_v1":
        raise ProtocolError("training representation differs")
    if training.get("network_feature_order") != [
        "option_type",
        "log_forward_moneyness",
        "total_volatility",
        "rate_time",
        "yield_time",
    ]:
        raise ProtocolError("training feature order differs")
    if training.get("dtype") != "float64" or training.get("device") != "cpu":
        raise ProtocolError("training must be float64 CPU")
    if training.get("architecture") != {
        "hidden_dimensions": [64, 64, 64],
        "activation": "tanh",
        "output_dimension": 1,
    }:
        raise ProtocolError("training architecture differs")
    if {
        key: training[key]
        for key in (
            "experiment_name",
            "physical_feature_order",
            "target",
            "physical_reconstruction",
        )
    } != {
        "experiment_name": "task-9g-american-neural-pilot-v1",
        "physical_feature_order": [
            "option_type",
            "spot",
            "strike",
            "maturity",
            "rate",
            "dividend_yield",
            "volatility",
        ],
        "target": "u = V / (S * exp(-q * T))",
        "physical_reconstruction": "V = S * exp(-q * T) * u",
    }:
        raise ProtocolError("training representation text or physical feature order differs")
    arms = training.get("arms", {})
    if arms.get("names") != ["scratch", "transfer"]:
        raise ProtocolError("training must contain exactly scratch and transfer arms")
    for arm in ("scratch", "transfer"):
        label = arms.get(f"{arm}_seed_label")
        expected = (
            int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "big")
            if isinstance(label, str)
            else None
        )
        if arms.get(f"{arm}_seed") != expected:
            raise ProtocolError(f"{arm} seed does not match its public SHA-256 derivation")
    if arms != {
        "names": ["scratch", "transfer"],
        "scratch_seed_label": "task-9g-american-neural-pilot-v1/scratch",
        "scratch_seed": 2909056561,
        "transfer_seed_label": "task-9g-american-neural-pilot-v1/transfer",
        "transfer_seed": 2009073353,
        "seed_derivation": "first four bytes of SHA-256(label), unsigned big-endian",
    }:
        raise ProtocolError("arm labels, seeds, or derivation differ")
    if training["row_selection"] != {
        "partition": "train",
        "row_budget": 32768,
        "rule": "lowest SHA-256(salt + NUL + sample_id), sample_id tie-break",
        "salt": "task-9g-train-subset-v1",
    }:
        raise ProtocolError("train-row selection contract differs")
    if training["standardization"] != {
        "fit_partition": "selected train rows only",
        "population_standard_deviation_ddof": 0,
        "shared_between_arms": True,
    }:
        raise ProtocolError("standardization contract differs")
    if training.get("row_selection", {}).get("row_budget") != 32768:
        raise ProtocolError("training row budget differs")
    if training.get("training", {}).get("epochs") != 120:
        raise ProtocolError("training epoch budget differs")
    if training["optimizer"] != {
        "name": "adamw",
        "learning_rate": 0.001,
        "weight_decay": 0.000001,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1.0e-8,
        "amsgrad": False,
        "maximize": False,
        "foreach": False,
        "capturable": False,
        "differentiable": False,
        "fused": False,
        "schedule": "cosine_annealing",
        "minimum_learning_rate": 0.0,
        "schedule_period_epochs": 120,
        "schedule_last_epoch": -1,
    }:
        raise ProtocolError("optimizer or schedule contract differs")
    if training["training"] != {
        "objective": "standardized target mean squared error",
        "batch_size": 2048,
        "epochs": 120,
        "num_threads": 4,
        "deterministic_algorithms": True,
        "shuffle": "torch.randperm from the arm seed once per epoch",
    }:
        raise ProtocolError("training loop contract differs")
    if training["checkpoint"] != {
        "selection_partition": "validation",
        "metric": "standardized_target_mse",
        "rule": "minimum metric; earliest epoch wins exact ties",
        "early_stopping": False,
    }:
        raise ProtocolError("checkpoint-selection contract differs")
    lift = training.get("exact_lift", {})
    if lift.get("relative_tolerance") > 1.0e-12 or lift.get("absolute_tolerance") > 1.0e-12:
        raise ProtocolError("exact-lift tolerance is looser than 1e-12")
    if len(lift.get("probes", [])) < 8:
        raise ProtocolError("exact-lift probe set is incomplete")
    expected_probes = [
        ("call", 100.0, 100.0, 0.019178082191780823, -0.02, 0.0, 0.05),
        ("put", 100.0, 120.0, 0.08, 0.12, 0.12, 0.8),
        ("call", 150.0, 75.0, 3.0, 0.08, 0.0, 0.2),
        ("put", 50.0, 100.0, 3.0, -0.01, 0.1, 0.6),
        ("call", 90.0, 110.0, 1.5, 0.03, 0.08, 0.35),
        ("put", 125.0, 95.0, 0.5, 0.0, 0.02, 0.15),
        ("call", 70.0, 90.0, 2.25, 0.1, 0.12, 0.75),
        ("put", 140.0, 130.0, 0.25, -0.015, 0.0, 0.4),
    ]
    actual_probes = [
        tuple(
            probe[key]
            for key in (
                "option_type",
                "spot",
                "strike",
                "maturity",
                "rate",
                "dividend_yield",
                "volatility",
            )
        )
        for probe in lift["probes"]
    ]
    if actual_probes != expected_probes:
        raise ProtocolError("exact-lift physical probe set differs")
    if acceptance.get("schema_version") != "american-neural-pilot-acceptance/1":
        raise ProtocolError("acceptance config schema differs")
    _exact_keys(
        acceptance,
        frozenset(
            {
                "schema_version",
                "name",
                "selection_partition",
                "final_partition",
                "reported_error_statistics",
                "quantile_method",
                "moneyness_coordinate",
                "bin_interval_convention",
                "reported_units",
                "required_slices",
                "baseline",
                "bins",
                "validation_final_entry",
                "final_accuracy",
                "diagnostics",
                "interpretation",
                "latency",
                "implied_volatility",
            }
        ),
        "acceptance config",
    )
    for name, keys in {
        "bins": {
            "expiry_edges_years",
            "log_moneyness_edges",
            "volatility_edges",
            "premium_status",
            "exercise_status",
        },
        "validation_final_entry": {
            "applies_to",
            "normalized_rmse_max",
            "normalized_p99_absolute_error_max",
            "normalized_maximum_absolute_error_max",
            "maximum_material_bound_violations",
            "maximum_material_shape_violations",
            "all_required_evidence_complete",
        },
        "final_accuracy": {
            "normalized_rmse_max",
            "normalized_p99_absolute_error_max",
            "normalized_maximum_absolute_error_max",
            "maximum_material_bound_violations",
            "maximum_material_shape_violations",
        },
        "diagnostics": {
            "material_normalized_tolerance",
            "spot_bump_fraction",
            "volatility_bump",
            "spot_domain",
            "volatility_domain",
            "log_moneyness_domain",
            "checks",
            "american_put_upper_bound",
            "american_call_upper_bound",
            "output_projection",
        },
        "interpretation": {
            "promising_requires",
            "negative_transfer",
            "failure_to_learn",
            "mixed_accuracy_outcome",
            "required_evidence_failed",
            "pilot_claim",
        },
        "latency": {
            "reference_depth_for_interpretation",
            "minimum_median_end_to_end_speedup",
            "confidence_interval",
            "confidence_level",
        },
        "implied_volatility": {
            "maximum_absolute_volatility_error",
            "failures_max",
            "terminology",
        },
    }.items():
        _exact_keys(
            _table(acceptance[name], f"acceptance.{name}"),
            frozenset(keys),
            f"acceptance.{name}",
        )
    if acceptance.get("required_slices") != [
        "overall",
        "option_type",
        "expiry",
        "moneyness",
        "volatility",
        "premium_status",
        "exercise_status",
    ]:
        raise ProtocolError("acceptance slices differ")
    if acceptance.get("quantile_method") != "linear":
        raise ProtocolError("metric quantile method differs")
    if acceptance.get("moneyness_coordinate") != "log(spot/strike)":
        raise ProtocolError("metric moneyness coordinate differs")
    if acceptance.get("bin_interval_convention") != "[low, high)":
        raise ProtocolError("metric bin interval convention differs")
    if acceptance.get("diagnostics", {}).get("output_projection") != "none":
        raise ProtocolError("American acceptance must not project model output")
    if acceptance.get("diagnostics", {}).get("spot_domain") != [50.0, 150.0]:
        raise ProtocolError("shape-diagnostic spot domain differs")
    if acceptance.get("diagnostics", {}).get("volatility_domain") != [0.05, 0.8]:
        raise ProtocolError("shape-diagnostic volatility domain differs")
    if acceptance.get("diagnostics", {}).get("log_moneyness_domain") != [-0.7, 0.7]:
        raise ProtocolError("shape-diagnostic moneyness domain differs")
    if acceptance["bins"] != {
        "expiry_edges_years": [0.25, 1.0, 2.0],
        "log_moneyness_edges": [-0.2, 0.2],
        "volatility_edges": [0.15, 0.5],
        "premium_status": ["zero", "positive"],
        "exercise_status": ["no_exercise", "exercise_observed"],
    }:
        raise ProtocolError("acceptance bins differ")
    for section in ("validation_final_entry", "final_accuracy"):
        expected_gate = {
            "normalized_rmse_max": 0.003,
            "normalized_p99_absolute_error_max": 0.015,
            "normalized_maximum_absolute_error_max": 0.08,
            "maximum_material_bound_violations": 0,
            "maximum_material_shape_violations": 0,
        }
        actual_gate = dict(acceptance[section])
        actual_gate.pop("applies_to", None)
        actual_gate.pop("all_required_evidence_complete", None)
        if actual_gate != expected_gate:
            raise ProtocolError(f"acceptance {section} thresholds differ")
    if acceptance["validation_final_entry"]["applies_to"] != ["scratch", "transfer"] or (
        acceptance["validation_final_entry"]["all_required_evidence_complete"] is not True
    ):
        raise ProtocolError("validation final-entry scope differs")
    if acceptance["diagnostics"] != {
        "material_normalized_tolerance": 1.0e-6,
        "spot_bump_fraction": 0.01,
        "volatility_bump": 0.01,
        "spot_domain": [50.0, 150.0],
        "volatility_domain": [0.05, 0.8],
        "log_moneyness_domain": [-0.7, 0.7],
        "checks": [
            "intrinsic_lower_bound",
            "european_comparator_lower_bound",
            "american_call_upper_bound",
            "american_put_rate_aware_upper_bound",
            "spot_monotonicity",
            "spot_convexity",
            "volatility_monotonicity",
        ],
        "american_put_upper_bound": "strike * exp(max(-rate * maturity, 0))",
        "american_call_upper_bound": "spot",
        "output_projection": "none",
    }:
        raise ProtocolError("American diagnostic contract differs")
    if {
        "name": acceptance["name"],
        "selection_partition": acceptance["selection_partition"],
        "final_partition": acceptance["final_partition"],
        "reported_error_statistics": acceptance["reported_error_statistics"],
        "reported_units": acceptance["reported_units"],
        "baseline": acceptance["baseline"],
    } != {
        "name": "task-9g-american-neural-pilot-acceptance-v1",
        "selection_partition": "validation",
        "final_partition": "interpolation_test",
        "reported_error_statistics": [
            "mae",
            "rmse",
            "p95_absolute_error",
            "p99_absolute_error",
            "maximum_absolute_error",
        ],
        "reported_units": ["physical_price", "normalized_by_discounted_spot"],
        "baseline": "stored european_crr_price",
    }:
        raise ProtocolError("acceptance identity or reported metrics differ")
    if acceptance["interpretation"] != {
        "promising_requires": (
            "both final accuracy gates, median N=1024 end-to-end speedup >= 10 for both "
            "arms at every request shape, IV maximum absolute volatility error <= 0.01 for "
            "both arms, and transfer validation RMSE < scratch validation RMSE"
        ),
        "negative_transfer": (
            "transfer validation normalized RMSE >= scratch validation normalized RMSE; "
            "failure_to_learn and mixed_accuracy_outcome take precedence when their accuracy "
            "conditions hold"
        ),
        "failure_to_learn": "neither arm reaches the locked final accuracy gates",
        "mixed_accuracy_outcome": (
            "exactly one arm reaches the locked final accuracy gates; report the passing arm "
            "without calling the pilot promising or a failure to learn"
        ),
        "required_evidence_failed": (
            "both arm accuracy gates pass but a required PDE, latency, or synthetic-IV evidence "
            "gate fails; final evaluation remains forbidden"
        ),
        "pilot_claim": "feasibility only; one seed and one budget cannot establish H2",
    }:
        raise ProtocolError("acceptance outcome interpretation differs")
    if acceptance["latency"] != {
        "reference_depth_for_interpretation": 1024,
        "minimum_median_end_to_end_speedup": 10.0,
        "confidence_interval": (
            "two-sided distribution-free order-statistic interval for the population median; "
            "with 7 repetitions use [minimum, maximum], actual coverage 0.984375"
        ),
        "confidence_level": 0.95,
    } or acceptance["implied_volatility"] != {
        "maximum_absolute_volatility_error": 0.01,
        "failures_max": 0,
        "terminology": "surface",
    }:
        raise ProtocolError("latency or IV acceptance interpretation differs")
    if latency.get("schema_version") != "american-neural-pilot-latency/1":
        raise ProtocolError("latency config schema differs")
    _exact_keys(
        latency,
        frozenset(
            {
                "schema_version",
                "name",
                "clock",
                "crr_depths",
                "crr_operation",
                "neural_end_to_end",
                "request_shapes",
                "batch_sizes",
                "thread_budgets",
                "torch_interop_threads",
                "warmups",
                "repetitions",
                "measurement_order",
                "speedup_statistic",
                "median_confidence_interval",
                "request_case_names",
                "cases",
            }
        ),
        "latency config",
    )
    if latency.get("crr_depths") != [256, 512, 1024, 2048, 4096]:
        raise ProtocolError("latency CRR ladder differs")
    if {
        "name": latency["name"],
        "clock": latency["clock"],
        "neural_end_to_end": latency["neural_end_to_end"],
    } != {
        "name": "task-9g-american-neural-pilot-latency-v1",
        "clock": "time.perf_counter_ns",
        "neural_end_to_end": (
            "feature construction + standardization + inference + inverse target transform + "
            "physical reconstruction"
        ),
    }:
        raise ProtocolError("latency identity or neural timing boundary differs")
    if latency.get("crr_operation") != "0.5 * (CRR(N) + CRR(N+1))":
        raise ProtocolError("latency CRR comparator differs")
    if (
        len(latency.get("cases", [])) != 8
        or latency.get("warmups") != 2
        or latency.get("repetitions") != 7
        or latency.get("torch_interop_threads") != 1
        or latency.get("measurement_order") != "deterministic cyclic rotation by repetition"
    ):
        raise ProtocolError("latency cases or repetitions differ")
    latency_case_names = [case["name"] for case in latency["cases"]]
    if latency["request_case_names"] != {
        "single": latency_case_names[:1],
        "batch8": latency_case_names,
    }:
        raise ProtocolError("latency request-set mapping differs")
    if (
        latency["request_shapes"] != ["single", "batch8"]
        or latency["batch_sizes"] != [1, 8]
        or latency["thread_budgets"] != [1, 4]
        or latency["speedup_statistic"]
        != (
            "median of paired CRR-adjacent-average nanoseconds divided by neural end-to-end "
            "nanoseconds"
        )
        or latency["median_confidence_interval"]
        != (
            "distribution-free [minimum, maximum] paired-speedup order-statistic interval; "
            "actual coverage 0.984375 for n=7"
        )
    ):
        raise ProtocolError("latency shapes, budgets, or statistic differs")
    expected_latency_cases = [
        ("call_short_negative_rate", "call", 100.0, 105.0, 0.08, -0.015, 0.0, 0.25),
        ("put_short_high_yield", "put", 95.0, 100.0, 0.1, 0.02, 0.12, 0.4),
        ("call_core", "call", 105.0, 100.0, 1.0, 0.04, 0.02, 0.2),
        ("put_core", "put", 95.0, 100.0, 1.0, 0.04, 0.02, 0.2),
        ("call_long_high_yield", "call", 120.0, 100.0, 3.0, 0.06, 0.1, 0.35),
        ("put_long_high_rate", "put", 80.0, 100.0, 3.0, 0.12, 0.0, 0.3),
        ("call_deep_otm_high_vol", "call", 60.0, 110.0, 1.5, 0.01, 0.04, 0.75),
        ("put_deep_otm_low_vol", "put", 140.0, 90.0, 2.0, -0.01, 0.01, 0.1),
    ]
    if [
        tuple(
            case[key]
            for key in (
                "name",
                "option_type",
                "spot",
                "strike",
                "maturity",
                "rate",
                "dividend_yield",
                "volatility",
            )
        )
        for case in latency["cases"]
    ] != expected_latency_cases:
        raise ProtocolError("latency economic cases differ")
    if (
        iv.get("schema_version") != "american-neural-pilot-iv/1"
        or iv.get("solver") != "safeguarded_bisection"
        or iv.get("crr_steps") != 1024
    ):
        raise ProtocolError("IV solver contract differs")
    if {
        "name": iv["name"],
        "price_source": iv["price_source"],
        "comparison": iv["comparison"],
    } != {
        "name": "task-9g-american-neural-pilot-iv-v1",
        "price_source": "CRR adjacent average at N=1024 and N+1=1025",
        "comparison": (
            "invert label price and each neural price through the same CRR adjacent-average "
            "function"
        ),
    }:
        raise ProtocolError("IV identity or comparison boundary differs")
    _exact_keys(
        iv,
        frozenset(
            {
                "schema_version",
                "name",
                "terminology",
                "price_source",
                "crr_steps",
                "comparison",
                "solver",
                "volatility_bracket",
                "price_absolute_tolerance_over_spot",
                "volatility_absolute_tolerance",
                "maximum_iterations",
                "failure_rule",
                "cases",
            }
        ),
        "IV config",
    )
    case_keys = frozenset(
        {
            "name",
            "option_type",
            "spot",
            "strike",
            "maturity",
            "rate",
            "dividend_yield",
            "volatility",
        }
    )
    for index, case in enumerate(latency.get("cases", [])):
        _exact_keys(_table(case, f"latency.cases[{index}]"), case_keys, f"latency.cases[{index}]")
    for index, case in enumerate(iv.get("cases", [])):
        _exact_keys(
            _table(case, f"iv.cases[{index}]"),
            frozenset(case_keys - {"volatility"} | {"true_volatility"}),
            f"iv.cases[{index}]",
        )
    maturities = {case["maturity"] for case in iv.get("cases", [])}
    if len(maturities) < 2 or iv.get("terminology") != "surface":
        raise ProtocolError("IV evidence may be called a surface only with multiple maturities")
    if len({(case["spot"], case["rate"], case["dividend_yield"]) for case in iv["cases"]}) != 1:
        raise ProtocolError("IV surface cases must share spot, rate, and yield")
    if any(
        len({case["strike"] for case in iv["cases"] if case["maturity"] == maturity}) < 2
        for maturity in maturities
    ):
        raise ProtocolError("every IV smile slice must contain multiple strikes")
    if (
        iv["volatility_bracket"] != [0.05, 0.8]
        or iv["price_absolute_tolerance_over_spot"] != 1.0e-10
        or iv["volatility_absolute_tolerance"] != 1.0e-8
        or iv["maximum_iterations"] != 80
        or iv["failure_rule"]
        != (
            "unbracketed root, non-finite price, or iteration exhaustion is a reported failure "
            "and is never dropped"
        )
    ):
        raise ProtocolError("IV bracket, tolerances, iteration budget, or failure rule differs")
    expected_iv_cases = [
        ("call_t025_k90", "call", 100.0, 90.0, 0.25, 0.03, 0.02, 0.18),
        ("call_t025_k100", "call", 100.0, 100.0, 0.25, 0.03, 0.02, 0.2),
        ("put_t025_k110", "put", 100.0, 110.0, 0.25, 0.03, 0.02, 0.24),
        ("put_t2_k90", "put", 100.0, 90.0, 2.0, 0.03, 0.02, 0.3),
        ("call_t2_k100", "call", 100.0, 100.0, 2.0, 0.03, 0.02, 0.28),
        ("put_t2_k110", "put", 100.0, 110.0, 2.0, 0.03, 0.02, 0.32),
    ]
    if [
        tuple(
            case[key]
            for key in (
                "name",
                "option_type",
                "spot",
                "strike",
                "maturity",
                "rate",
                "dividend_yield",
                "true_volatility",
            )
        )
        for case in iv["cases"]
    ] != expected_iv_cases:
        raise ProtocolError("synthetic IV surface cases differ")
    return {"training": training, "acceptance": acceptance, "latency": latency, "iv": iv}


def validate_protocol(
    protocol_path: Path = DEFAULT_PROTOCOL,
    project_root: Path = PROJECT_ROOT,
) -> dict[str, Any]:
    protocol = _load_toml(protocol_path, "Task 9G protocol")
    _exact_keys(protocol, EXPECTED_TOP_KEYS, "protocol")
    if (
        protocol["schema_version"] != SCHEMA
        or protocol["name"] != "task-9g-american-neural-pilot-v1"
    ):
        raise ProtocolError("protocol identity differs")
    paths = _table(protocol["paths"], "protocol.paths")
    _exact_keys(paths, EXPECTED_PATH_KEYS, "protocol.paths")
    if paths["protocol"] != str(protocol_path.resolve().relative_to(project_root.resolve())):
        raise ProtocolError("protocol.paths.protocol does not name this file")
    if paths["dataset"] != "data/american-option-v1":
        raise ProtocolError("protocol dataset path differs")
    if dict(paths) != {
        "protocol": "configs/american_neural_pilot_protocol_v1.toml",
        "training_config": "configs/american_neural_pilot_training_v1.toml",
        "acceptance_config": "configs/american_neural_pilot_acceptance_v1.toml",
        "latency_config": "configs/american_neural_pilot_latency_cases_v1.toml",
        "iv_config": "configs/american_neural_pilot_iv_cases_v1.toml",
        "dataset": "data/american-option-v1",
        "dataset_manifest": "data/american-option-v1/manifest.json",
        "source_artifact": ("artifacts/european-neural-forward-differential-replication-v1"),
        "output_directory": "artifacts/task-9g-american-neural-pilot-v1",
        "execution_ledger": "runs/task-9g-american-neural-pilot-v1/execution.json",
        "raw_report": "artifacts/task-9g-american-neural-pilot-v1/validation-report.json",
        "final_report": "artifacts/task-9g-american-neural-pilot-v1/final-report.json",
        "final_failure_report": (
            "artifacts/task-9g-american-neural-pilot-v1/final-attempt-failure-report.json"
        ),
        "result_snapshot": "docs/results/american_neural_pilot_results_v1.json",
    }:
        raise ProtocolError("protocol paths differ")
    dataset = _table(protocol["dataset"], "protocol.dataset")
    _exact_keys(dataset, frozenset(MANDATED_DATASET), "protocol.dataset")
    if dict(dataset) != MANDATED_DATASET:
        raise ProtocolError("protocol dataset identity differs from the mandated pins")
    source = _table(protocol["source_artifact"], "protocol.source_artifact")
    _exact_keys(
        source,
        frozenset({"schema_version", "representation", "architecture", *MANDATED_SOURCE}),
        "protocol.source_artifact",
    )
    if (
        source["schema_version"] != "european-neural-artifact/v1"
        or source["representation"] != "forward_normalized_v1"
    ):
        raise ProtocolError("source artifact contract differs")
    if source["architecture"] != [3, 64, 64, 64, 1]:
        raise ProtocolError("source artifact architecture differs")
    for name, digest in MANDATED_SOURCE.items():
        if source[name] != digest:
            raise ProtocolError(f"source artifact {name} differs")
    model = _table(protocol["model"], "protocol.model")
    if model != {
        "representation": "american_forward_carry_v1",
        "network_feature_order": [
            "option_type",
            "log_forward_moneyness",
            "total_volatility",
            "rate_time",
            "yield_time",
        ],
        "target": "u = V / (S * exp(-q * T))",
        "physical_reconstruction": "V = S * exp(-q * T) * u",
        "architecture": [5, 64, 64, 64, 1],
        "activation": "tanh",
        "dtype": "float64",
        "device": "cpu",
        "output_constraint": "none",
    }:
        raise ProtocolError("master model contract differs")
    lifecycle = _table(protocol["lifecycle"], "protocol.lifecycle")
    if lifecycle != {
        "public_modes": [
            "run-to-validation",
            "status",
            "final-evaluate --confirm-locked-final-evaluation",
        ],
        "selection_partition": "validation",
        "final_partition": "interpolation_test",
        "maximum_final_attempts": 1,
        "reserve_before_final_access": True,
        "interruption_consumes_attempt": True,
        "second_attempt_allowed": False,
        "failure_action": "record_failure; new protocol and fresh final partition required",
    }:
        raise ProtocolError("partition lifecycle differs")
    if protocol["output_schemas"] != {
        "execution_ledger": "american-neural-pilot-execution/1",
        "validation_raw_report": "american-neural-pilot-raw-report/1",
        "entry_failure_raw_report": "american-neural-pilot-entry-failure-report/1",
        "final_raw_report": "american-neural-pilot-final-report/1",
        "final_failure_raw_report": "american-neural-pilot-final-attempt-failure-report/1",
        "model_artifact": "american-neural-artifact/1",
        "result_snapshot": "american-neural-pilot-result/1",
    }:
        raise ProtocolError("output schema identities differ")
    if protocol["environment_metadata"] != {
        "required": [
            "python",
            "numpy",
            "torch",
            "platform",
            "machine",
            "processor",
            "logical_cpu_count",
            "cpu_affinity",
            "core_version",
            "build_configuration",
            "cxx_compiler",
        ],
        "timing_clock": "time.perf_counter_ns",
        "timestamps_are_evidence_only": True,
        "portable_latency_claim_allowed": False,
    }:
        raise ProtocolError("environment metadata contract differs")
    if protocol["limitations"] != {
        "arm_shuffle_confound": (
            "scratch and transfer use different arm-seeded epoch permutations as well as "
            "different initialization; this one-seed pilot cannot attribute an observed arm "
            "difference solely to transfer initialization"
        ),
        "pde_domain_truncation": (
            "the 400x200 versus 800x400 PDE refinement comparison changes resolution while "
            "holding spot_maximum fixed and therefore does not independently bound "
            "domain-truncation error"
        ),
        "snapshot_authentication": (
            "offline --check detects internal inconsistency and tracked-input drift but cannot "
            "authenticate a fully coordinated fabricated raw report and snapshot"
        ),
    }:
        raise ProtocolError("pilot limitations differ")
    pde = _table(protocol["independent_pde_check"], "protocol.independent_pde_check")
    _exact_keys(
        pde,
        frozenset(
            {
                "engine",
                "exercise_style",
                "continuous_carry_mapping",
                "cash_dividends",
                "selection_partition",
                "selection_columns",
                "selection_rule",
                "selection_salt",
                "grids",
                "maximum_normalized_refinement_difference",
                "maximum_normalized_pde_crr_difference",
                "normalization",
                "interpretation",
                "requested_spot_maximum_rule",
                "cases",
                "valuation_time",
                "settlement",
                "contract_multiplier",
                "rannacher_steps",
                "psor_tolerance",
                "psor_relaxation",
                "psor_maximum_iterations",
            }
        ),
        "protocol.independent_pde_check",
    )
    if (
        pde["engine"] != "dp::finite_difference_price"
        or pde["continuous_carry_mapping"] != "continuous_carry = dividend_yield"
        or pde["cash_dividends"] != "declared empty schedule"
    ):
        raise ProtocolError("independent PDE mapping differs")
    if {
        "valuation_time": pde["valuation_time"],
        "settlement": pde["settlement"],
        "contract_multiplier": pde["contract_multiplier"],
        "rannacher_steps": pde["rannacher_steps"],
        "psor_tolerance": pde["psor_tolerance"],
        "psor_relaxation": pde["psor_relaxation"],
        "psor_maximum_iterations": pde["psor_maximum_iterations"],
    } != {
        "valuation_time": 0.0,
        "settlement": "cash",
        "contract_multiplier": 1.0,
        "rannacher_steps": 2,
        "psor_tolerance": 1.0e-11,
        "psor_relaxation": 1.2,
        "psor_maximum_iterations": 50_000,
    }:
        raise ProtocolError("independent PDE solver settings differ")
    if pde["selection_partition"] != "validation" or "american_price" in pde["selection_columns"]:
        raise ProtocolError("PDE selection must use validation identifiers and inputs only")
    grids = _sequence(pde["grids"], "PDE grids")
    if grids != [
        {
            "name": "coarse",
            "spot_intervals": 400,
            "time_steps": 200,
            "spot_maximum_factor": 4.0,
        },
        {
            "name": "fine",
            "spot_intervals": 800,
            "time_steps": 400,
            "spot_maximum_factor": 4.0,
        },
    ]:
        raise ProtocolError("PDE refinement grids differ")
    cases = _sequence(pde["cases"], "PDE cases")
    if len(cases) != 21 or len({case["sample_id"] for case in cases}) != 21:
        raise ProtocolError("PDE cases must pin 21 distinct validation samples")
    for case in cases:
        expected = hashlib.sha256(
            f"{pde['selection_salt']}\0{case['sample_id']}".encode()
        ).hexdigest()
        if case.get("selection_sha256") != expected:
            raise ProtocolError(f"PDE case {case['sample_id']} selection digest differs")
        for key in ("spot", "strike", "maturity", "rate", "dividend_yield", "volatility"):
            if (
                isinstance(case.get(key), bool)
                or not isinstance(case.get(key), int | float)
                or not math.isfinite(float(case[key]))
            ):
                raise ProtocolError(f"PDE case {case['sample_id']} has invalid {key}")
    tracked = _sequence(protocol["tracked_inputs"], "protocol.tracked_inputs")
    required = {
        paths["training_config"],
        paths["acceptance_config"],
        paths["latency_config"],
        paths["iv_config"],
        "configs/american_option_dataset_v1.toml",
        "docs/architecture.md",
        "docs/research-contract.md",
        "docs/american-crr-contract.md",
        "docs/american-crr-dataset-admission.md",
        "docs/pde-numerical-contract.md",
        "docs/results/european_replication_results_v1.json",
        "pyproject.toml",
        "python/src/differentiable_pricing/__init__.py",
        "python/src/differentiable_pricing/data/american_schema.py",
        "python/src/differentiable_pricing/ml/artifact.py",
        "python/src/differentiable_pricing/ml/config.py",
        "python/src/differentiable_pricing/ml/model.py",
        "python/src/differentiable_pricing/ml/american.py",
        "python/src/differentiable_pricing/ml/american_artifact.py",
        "python/src/differentiable_pricing/ml/american_pilot.py",
        "scripts/run_american_neural_pilot.py",
        "scripts/check_american_neural_pilot_protocol.py",
        "scripts/freeze_american_neural_pilot_results.py",
        "cpp/include/dp/finite_difference_pde.hpp",
        "cpp/src/finite_difference_pde.cpp",
        "bindings/python/pde_module.cpp",
        "bindings/python/module.cpp",
        "cpp/include/dp/binomial_tree.hpp",
        "cpp/src/binomial_tree.cpp",
        "cpp/include/dp/black_scholes.hpp",
        "cpp/src/black_scholes.cpp",
        "cpp/include/dp/option.hpp",
        "cpp/src/option.cpp",
        "CMakeLists.txt",
    }
    seen = set()
    for index, entry_value in enumerate(tracked):
        entry = _table(entry_value, f"tracked_inputs[{index}]")
        _exact_keys(entry, frozenset({"path", "sha256"}), f"tracked_inputs[{index}]")
        path_text = entry["path"]
        if path_text in seen:
            raise ProtocolError(f"duplicate tracked input {path_text!r}")
        seen.add(path_text)
        path = _resolve(project_root, path_text, f"tracked_inputs[{index}].path")
        actual = _sha256(path)
        if _digest(entry["sha256"], f"tracked_inputs[{index}].sha256") != actual:
            raise ProtocolError(f"tracked input digest drift: {path_text}")
    missing = sorted(required - seen)
    if missing:
        raise ProtocolError(f"protocol does not pin authoritative input(s): {missing}")
    leaves = _validate_leaf_configs(protocol, project_root)
    try:
        snapshot = json.loads(
            (project_root / "docs/results/european_replication_results_v1.json").read_text()
        )
    except (OSError, json.JSONDecodeError) as error:
        raise ProtocolError(f"cannot load European source snapshot: {error}") from error
    unconstrained = snapshot["model"]["unconstrained_artifact"]
    if (
        unconstrained["manifest_sha256"] != source["manifest_sha256"]
        or unconstrained["weights_sha256"] != source["weights_sha256"]
    ):
        raise ProtocolError("European source snapshot identity differs")
    return {
        "name": protocol["name"],
        "tracked_inputs": len(tracked),
        "pde_cases": len(cases),
        "scratch_seed": leaves["training"]["arms"]["scratch_seed"],
        "transfer_seed": leaves["training"]["arms"]["transfer_seed"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    arguments = parser.parse_args(argv)
    try:
        summary = validate_protocol(arguments.protocol.resolve(), PROJECT_ROOT)
    except ProtocolError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
