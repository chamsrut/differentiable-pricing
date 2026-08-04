import json
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "docs/results/american_pde_label_policy_results_v1.json"
CONFIG = ROOT / "configs/pde_label_policy_pilot_v1.toml"
RUNNER = ROOT / "python/src/differentiable_pricing/american/pde_label_policy.py"

EXPECTED_REPORT_SHA256 = "7871e52e52149171c16fb665f2114ccf300ac4e33babee2242d99e35488736c6"
EXPECTED_CRITERIA = {
    "delta_absolute_error": 0.001,
    "delta_bump_variation": 0.001,
    "gamma_absolute_error": 0.0002,
    "gamma_bump_variation": 0.0002,
    "price_absolute_error": 0.0005,
    "shape_tolerance": 1e-8,
    "vega_absolute_error_per_unit_volatility": 0.05,
    "vega_bump_variation_per_unit_volatility": 0.05,
}


def _digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def test_pde_label_policy_result_digest_is_frozen() -> None:
    assert _digest(RESULT) == EXPECTED_REPORT_SHA256


def test_pde_label_policy_result_records_negative_selection() -> None:
    report = json.loads(RESULT.read_text())

    assert report["schema_version"] == "pde-label-policy-report/1"
    assert report["predeclared_criteria"] == EXPECTED_CRITERIA
    assert report["recommendation"]["criteria_were_not_loosened"] is True
    assert report["recommendation"]["regular_cases_only_decide_selection"] is True
    assert report["recommendation"]["selected_accuracy_policy"] == "no_policy_selected"


def test_pde_label_policy_result_matches_sources() -> None:
    report = json.loads(RESULT.read_text())
    source = report["source"]

    assert source["config_sha256"] == _digest(CONFIG)
    assert source["runner_sha256"] == _digest(RUNNER)
