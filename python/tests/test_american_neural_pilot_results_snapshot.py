"""Preservation tests for the frozen task 9G American neural-pricer snapshot.

`docs/results/american_neural_pilot_results_v1.json` is the terminal evidence of
task 9G: the human operator invoked the locked `run-to-validation` exactly once,
the predeclared feasibility gates were not met, and the pilot closed with
`status = validation_gates_failed`, `outcome = failure_to_learn`. This module
pins that file so a later change cannot quietly reinterpret, regenerate, or
loosen it.

It follows the same three rules as the other frozen-result preservation modules
in this directory (`test_pde_label_policy_v2_results_snapshot.py`,
`test_american_lsm_results_snapshot.py`):

* it reads only checked-in files — never `artifacts/`, never the ignored raw
  validation report, never a dataset partition, never a compiled extension;
* it re-runs the designated offline `--check` validator in-process, so the
  semantic recomputation `scripts/check.sh` and CI perform is exercised here
  too, not merely the byte digest;
* it never edits data to satisfy an assertion. Every expected value below was
  read out of the reviewed snapshot, and the snapshot's own SHA-256 is pinned
  first, so a mutation fails here before any derived assertion is reached.

**This is a negative result, and it stays one.** `transfer` beat `scratch` on
validation RMSE, but the two locked arm seeds also produce different epoch
shuffle permutations, so that comparison is confounded and is not evidence that
transfer initialization helps. Neither arm passed its gate. One seed and one
budget do not establish H2.

**The final partition is unconsumed and stays that way.** The snapshot records
`final_evaluation_attempts = 0` and `final_partition_consumed = false`, and
`validation_final_entry_passed = false` makes `final-evaluate` forbidden under
the task 9G protocol — now and later. No test here opens, hashes, imports, or
counts `interpolation_test`; the only final-partition value that appears is the
digest the protocol pinned in advance, which is compared as a string.
"""

from __future__ import annotations

import importlib.util
import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Final

import pytest

PROJECT_ROOT: Final = Path(__file__).resolve().parents[2]
SNAPSHOT: Final = PROJECT_ROOT / "docs/results/american_neural_pilot_results_v1.json"
PROTOCOL: Final = PROJECT_ROOT / "configs/american_neural_pilot_protocol_v1.toml"
FREEZE_SCRIPT: Final = PROJECT_ROOT / "scripts/freeze_american_neural_pilot_results.py"

# The reviewed snapshot, and the raw validation report it was distilled from.
# Neither may ever be regenerated, reformatted, or hand-edited to make a test
# pass. The raw report stays beneath ignored paths; only its digest lives here.
EXPECTED_SNAPSHOT_SHA256: Final = (
    "8a125c81d469c320bc3a2cea9709f695e11c0a3a4ed59b985b88b93e3f3cdedc"
)
EXPECTED_VALIDATION_REPORT_SHA256: Final = (
    "9a4acdfd3b96eee3d29ef9c12c467ea304297dc44d561f56fdaeb2292ba999a9"
)
EXPECTED_PROTOCOL_SHA256: Final = (
    "6600a46ea2132bd3c834645ddb704ccc71069a0cca84d3762b4c4ca753aac591"
)

# Dataset and source-artifact identities the locked run verified before it
# opened anything. `locked_final_sha256` is compared as a pinned string; the
# file it names is never resolved, stat-ed, hashed, or read here.
EXPECTED_DATASET_CONFIG_SHA256: Final = (
    "d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98"
)
EXPECTED_SOURCE_WEIGHTS_SHA256: Final = (
    "42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8"
)
EXPECTED_SOURCE_MANIFEST_SHA256: Final = (
    "054ca945de851b4e5b2a10455e93be44546770410bcf698e0051175fe8d19f51"
)

# The five predeclared validation checks, and exactly how each arm fared.
# `transfer` converted one of five; a single converted check is not a pass.
EXPECTED_GATE_CHECKS: Final = {
    "scratch": {
        "normalized_rmse": False,
        "normalized_p99_absolute_error": False,
        "normalized_maximum_absolute_error": False,
        "material_bound_violations": False,
        "material_shape_violations": False,
    },
    "transfer": {
        "normalized_rmse": False,
        "normalized_p99_absolute_error": True,
        "normalized_maximum_absolute_error": False,
        "material_bound_violations": False,
        "material_shape_violations": False,
    },
}

EXPECTED_EXACT_LIFT_PROBES: Final = 8
EXPECTED_EXACT_LIFT_MAX_ABSOLUTE: Final = 1.4210854715202004e-14
EXPECTED_EXACT_LIFT_MAX_RELATIVE: Final = 3.6214823824845716e-13
EXPECTED_PDE_CHECK_ROWS: Final = 21
EXPECTED_IV_CASES: Final = 6
EXPECTED_LATENCY_REFERENCE_DEPTH: Final = 1024


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def freeze_tool() -> Any:
    specification = importlib.util.spec_from_file_location("freeze_task9g", FREEZE_SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Identity of the frozen file
# ---------------------------------------------------------------------------


def test_snapshot_digest_is_frozen() -> None:
    """The reviewed bytes. Never regenerate this file to fix a failing test."""
    assert _digest(SNAPSHOT) == EXPECTED_SNAPSHOT_SHA256


def test_snapshot_schema_and_consumed_report_are_pinned(snapshot: dict[str, Any]) -> None:
    """The raw validation report is git-ignored; only its digest is preserved."""
    assert snapshot["schema_version"] == "american-neural-pilot-result/1"
    assert snapshot["source_reports"]["validation"] == {
        "file": "validation-report.json",
        "sha256": EXPECTED_VALIDATION_REPORT_SHA256,
    }
    # No final evaluation ran, so there is no final report to distil.
    assert snapshot["source_reports"]["final"] is None
    assert snapshot["entry_failure"] is None


def test_snapshot_pins_the_executed_protocol(snapshot: dict[str, Any]) -> None:
    assert snapshot["protocol"]["path"] == "configs/american_neural_pilot_protocol_v1.toml"
    assert snapshot["protocol"]["sha256"] == EXPECTED_PROTOCOL_SHA256
    assert snapshot["protocol"]["sha256"] == _digest(PROTOCOL)


def test_snapshot_pins_dataset_and_source_artifact_identities(
    snapshot: dict[str, Any],
) -> None:
    dataset = snapshot["dataset"]
    assert dataset["schema_version"] == "american-option-dataset/1"
    assert dataset["generator_version"] == "1.0.0"
    assert dataset["config_sha256"] == EXPECTED_DATASET_CONFIG_SHA256

    source = snapshot["source_artifact"]
    assert source["schema_version"] == "european-neural-artifact/v1"
    assert source["representation"] == "forward_normalized_v1"
    assert source["architecture"] == [3, 64, 64, 64, 1]
    assert source["manifest_sha256"] == EXPECTED_SOURCE_MANIFEST_SHA256
    assert source["weights_sha256"] == EXPECTED_SOURCE_WEIGHTS_SHA256


# ---------------------------------------------------------------------------
# Terminal lifecycle: the final partition is unconsumed and stays unconsumed
# ---------------------------------------------------------------------------


def test_lifecycle_is_the_terminal_validation_state(snapshot: dict[str, Any]) -> None:
    lifecycle = snapshot["lifecycle"]
    assert lifecycle["state"] == "validation_terminal"
    assert lifecycle["validation_final_entry_passed"] is False
    assert lifecycle["second_attempt_allowed"] is False


def test_final_partition_was_never_consumed(snapshot: dict[str, Any]) -> None:
    """`final-evaluate` is forbidden under this protocol, now and later.

    A failed pilot is never rescued by opening the partition it did not earn.
    """
    lifecycle = snapshot["lifecycle"]
    assert lifecycle["final_evaluation_attempts"] == 0
    assert lifecycle["final_partition_consumed"] is False
    assert snapshot["outcome"]["phase"] == "validation"
    for arm in ("scratch", "transfer"):
        assert snapshot["arms"][arm]["final"] is None
    assert snapshot["baseline"]["final"] is None
    assert snapshot["audit"]["final"] is None


# ---------------------------------------------------------------------------
# The recorded negative outcome
# ---------------------------------------------------------------------------


def test_outcome_is_the_predeclared_failure_to_learn(snapshot: dict[str, Any]) -> None:
    outcome = snapshot["outcome"]
    assert outcome["outcome"] == "failure_to_learn"
    assert outcome["arm_accuracy_passed"] == {"scratch": False, "transfer": False}
    assert outcome["all_iv_errors_passed"] is False
    assert outcome["all_reference_speedups_passed"] is False


def test_each_arm_failed_its_own_predeclared_checks(snapshot: dict[str, Any]) -> None:
    """`transfer` converted exactly one of five checks. That is not a pass."""
    for arm, expected in EXPECTED_GATE_CHECKS.items():
        gate = snapshot["arms"][arm]["validation"]["gate"]
        assert gate["checks"] == expected
        assert gate["passed"] is False
    transfer = EXPECTED_GATE_CHECKS["transfer"]
    assert sum(transfer.values()) == 1


def test_transfer_rmse_advantage_is_recorded_as_a_confounded_observation(
    snapshot: dict[str, Any],
) -> None:
    """True, and deliberately not evidence that transfer initialization helps."""
    assert snapshot["outcome"]["transfer_validation_rmse_strictly_better"] is True
    scratch = snapshot["arms"]["scratch"]["validation"]["slices"]["overall"]["normalized"]
    transfer = snapshot["arms"]["transfer"]["validation"]["slices"]["overall"]["normalized"]
    assert transfer["rmse"] < scratch["rmse"]
    assert "arm-seeded epoch permutations" in snapshot["interpretation"]["limitations"][
        "arm_shuffle_confound"
    ]


def test_claim_scope_and_non_claims_survive_into_the_snapshot(
    snapshot: dict[str, Any],
) -> None:
    interpretation = snapshot["interpretation"]
    assert interpretation["claim_scope"] == (
        "one-seed one-budget mapping-only feasibility pilot; not H2"
    )
    assert interpretation["no_american_greek_claim"] is True
    assert interpretation["no_market_or_bid_ask_claim"] is True
    limitations = interpretation["limitations"]
    assert set(limitations) == {
        "arm_shuffle_confound",
        "pde_domain_truncation",
        "snapshot_authentication",
    }
    assert "does not independently bound domain-truncation error" in limitations[
        "pde_domain_truncation"
    ]
    assert "cannot authenticate a fully coordinated fabricated raw report" in limitations[
        "snapshot_authentication"
    ]


# ---------------------------------------------------------------------------
# Entry gates that did pass, recorded at their true scope
# ---------------------------------------------------------------------------


def test_exact_transfer_lift_passed_at_every_pinned_probe(snapshot: dict[str, Any]) -> None:
    lift = snapshot["exact_lift"]
    assert lift["passed"] is True
    assert lift["probes"] == EXPECTED_EXACT_LIFT_PROBES
    assert len(lift["probe_prices"]) == EXPECTED_EXACT_LIFT_PROBES
    assert lift["maximum_absolute_difference"] == EXPECTED_EXACT_LIFT_MAX_ABSOLUTE
    assert lift["maximum_relative_difference"] == EXPECTED_EXACT_LIFT_MAX_RELATIVE
    assert lift["absolute_tolerance"] == 1.0e-12
    assert lift["relative_tolerance"] == 1.0e-12


def test_independent_pde_check_passed_as_mapping_consistency_only(
    snapshot: dict[str, Any],
) -> None:
    """21 of 21 rows, before optimization. Not converged American-price truth."""
    check = snapshot["independent_pde_check"]
    assert check["passed"] is True
    assert len(check["rows"]) == EXPECTED_PDE_CHECK_ROWS
    assert all(row["passed"] for row in check["rows"])
    assert check["interpretation"] == (
        "mapping-consistency only; not converged truth or semantic-coverage evidence"
    )


def test_implied_volatility_evidence_spans_maturities_and_is_termed_a_surface(
    snapshot: dict[str, Any],
) -> None:
    """Terminology is structural: one maturity is a slice, several are a surface."""
    implied = snapshot["implied_volatility"]
    assert implied["terminology"] == "surface"
    assert implied["failures"] == 0
    assert len(implied["rows"]) == EXPECTED_IV_CASES
    assert snapshot["outcome"]["all_iv_errors_passed"] is False


def test_latency_evidence_is_present_and_failed_its_reference_gate(
    snapshot: dict[str, Any],
) -> None:
    assert snapshot["outcome"]["reference_depth"] == EXPECTED_LATENCY_REFERENCE_DEPTH
    speedups = snapshot["outcome"]["reference_median_speedups"]
    assert set(speedups) == {
        "single:scratch",
        "single:transfer",
        "batch8:scratch",
        "batch8:transfer",
    }
    # The gate required >= 10 at every recorded shape; the single-request shape
    # did not reach it, so the aggregate verdict is False.
    assert min(speedups.values()) < 10.0
    assert snapshot["outcome"]["all_reference_speedups_passed"] is False
    assert snapshot["latency"]["measurements"]


# ---------------------------------------------------------------------------
# The designated offline validator
# ---------------------------------------------------------------------------


def test_designated_check_mode_validates_the_checked_in_snapshot(freeze_tool: Any) -> None:
    """The same validator `scripts/check.sh` and CI run, invoked in-process.

    It reads only tracked files, reruns no pricing, training, latency
    measurement or IV inversion, and opens no dataset partition.
    """
    assert freeze_tool.main(["--check"]) == 0
