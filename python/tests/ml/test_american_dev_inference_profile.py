"""Task 9H inference profile and bounded optimization: the decision rule.

Timing is measured on a synthetic model with small repetition counts: these
tests check the *contracts* -- that the variant list is closed, that every
variant computes the same function, and that the capacity branch follows the
corrected four-quantity rule -- not the machine's speed.
"""

from __future__ import annotations

from typing import Final

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.frozen import FrozenCheckpointError
from differentiable_pricing.ml.american_dev.inference_profile import (
    BACKBONE_SHARE_DESCRIPTIVE_HIGH,
    BACKBONE_SHARE_DESCRIPTIVE_LOW,
    COMPONENTS,
    COUNTERFACTUAL_BACKBONE_REDUCTIONS,
    HISTORICAL_SPEEDUP_REFERENCE,
    INFERENCE_VARIANTS,
    InferenceProfileError,
    assert_profiled_head,
    assert_variant_declared,
    build_variant,
    component_profile,
    decide_capacity_branch,
    measure_variants,
    time_callable,
)
from differentiable_pricing.ml.american_dev.models import build_network
from differentiable_pricing.ml.american_dev.representation import AmericanDevPriceModel
from differentiable_pricing.ml.model import Scaling

CRR: Final = 6_000_000.0


def _model(head: str = "smooth_lower_floor_margin_raw_loss") -> AmericanDevPriceModel:
    torch.manual_seed(5)
    network = build_network(
        {"name": "smooth_residual", "width": 16, "blocks": 2, "activation": "tanh"}, 5
    )
    scaling = Scaling(
        feature_mean=np.zeros(5, dtype=np.float64),
        feature_scale=np.ones(5, dtype=np.float64),
        price_mean=0.05,
        price_scale=0.05,
    )
    model = AmericanDevPriceModel(network, scaling, head=head)
    model.eval()
    return model


def _requests() -> dict[int, torch.Tensor]:
    return {
        size: torch.tensor([[-1.0, 100.0, 100.0, 1.0, 0.05, 0.02, 0.2]] * size, dtype=torch.float64)
        for size in (1, 8)
    }


# ---------------------------------------------------------------------------
# Section D is bounded
# ---------------------------------------------------------------------------


def test_the_variant_list_is_closed_and_predeclared() -> None:
    names = [entry["name"] for entry in INFERENCE_VARIANTS]
    assert len(names) == len(set(names))
    assert names[0] == "frozen_float64_reference"
    for entry in INFERENCE_VARIANTS:
        assert set(entry) == {"name", "changes_the_function", "description"}


def test_a_variant_outside_the_list_is_refused() -> None:
    with pytest.raises(InferenceProfileError, match="not predeclared"):
        assert_variant_declared("some_clever_idea")


def test_reduced_precision_is_a_deployment_variant_not_an_optimization() -> None:
    entry = assert_variant_declared("float32_deployment")
    assert entry["changes_the_function"] is True
    assert "DISTINCT DEPLOYMENT VARIANT" in entry["description"]
    _, record = build_variant(_model(), "float32_deployment")
    assert record["available"] is False
    assert "equivalence" in record["unavailable_reason"] or "battery" in (
        record["unavailable_reason"]
    )


def test_every_available_variant_computes_the_same_function() -> None:
    """"Faster" is never reported without "and it computed the same thing"."""
    measurements = measure_variants(_model(), _requests(), warmups=3, repetitions=8)
    for entry in measurements.values():
        if not isinstance(entry, dict) or "declaration" not in entry:
            continue
        if not entry["declaration"]["available"]:
            continue
        for shape in entry["shapes"].values():
            assert shape["maximum_absolute_difference_from_reference"] == 0.0
            assert shape["bitwise_identical_to_reference"] is True


def test_an_unavailable_variant_is_reported_rather_than_silently_repeated() -> None:
    measurements = measure_variants(_model(), _requests(), warmups=2, repetitions=4)
    unavailable = [
        name
        for name, entry in measurements.items()
        if isinstance(entry, dict)
        and "declaration" in entry
        and not entry["declaration"]["available"]
    ]
    for name in unavailable:
        assert measurements[name]["shapes"] == {}
        assert measurements[name]["declaration"]["unavailable_reason"]


# ---------------------------------------------------------------------------
# The component profile
# ---------------------------------------------------------------------------


def test_the_component_profile_names_every_declared_component() -> None:
    profile = component_profile(_model(), _requests()[1], warmups=3, repetitions=12)
    assert set(profile["components"]) == set(COMPONENTS)
    assert profile["batch_size"] == 1
    assert profile["backbone_share"] > 0.0


def test_the_dispatch_residual_is_reported_rather_than_distributed() -> None:
    profile = component_profile(_model(), _requests()[1], warmups=3, repetitions=12)
    wrapper = profile["median_nanoseconds"]["complete_wrapper"]
    parts = sum(
        value for name, value in profile["median_nanoseconds"].items() if name != "complete_wrapper"
    )
    assert profile["residual_dispatch_nanoseconds"] == pytest.approx(wrapper - parts)
    assert "distributed" in profile["residual_note"]


def test_the_descriptive_thresholds_are_marked_as_not_deciding() -> None:
    profile = component_profile(_model(), _requests()[1], warmups=2, repetitions=8)
    thresholds = profile["descriptive_thresholds"]
    assert thresholds["low"] == BACKBONE_SHARE_DESCRIPTIVE_LOW
    assert thresholds["high"] == BACKBONE_SHARE_DESCRIPTIVE_HIGH
    assert "do not decide" in thresholds["status"]


def test_a_head_without_a_floor_cannot_be_profiled() -> None:
    with pytest.raises(FrozenCheckpointError):
        assert_profiled_head(_model("direct"))


def test_timing_needs_a_positive_repetition_count() -> None:
    with pytest.raises(InferenceProfileError):
        time_callable(lambda: None, warmups=0, repetitions=0)


# ---------------------------------------------------------------------------
# The corrected capacity branch
# ---------------------------------------------------------------------------


def test_the_branch_is_undecided_before_section_d_runs() -> None:
    """The whole correction: the pre-optimization profile does not decide it."""
    decision = decide_capacity_branch(1_085_000.0, 285_000.0, None, CRR)
    assert decision["branch"] == "undecided_until_section_d_runs"
    assert decision["c_optimized_speedup"] is None
    assert decision["d_counterfactuals"] is None
    assert "standalone" in decision["reason"]


def test_a_zero_backbone_ceiling_below_the_reference_closes_only_the_standalone_case() -> None:
    decision = decide_capacity_branch(1_085_000.0, 285_000.0, None, 6_000_000.0)
    assert decision["b_zero_backbone_speedup"] < HISTORICAL_SPEEDUP_REFERENCE
    assert decision["standalone_capacity_closed"] is True
    assert decision["branch"] == "undecided_until_section_d_runs"


def test_an_optimized_path_that_clears_the_reference_needs_no_capacity_reduction() -> None:
    decision = decide_capacity_branch(1_085_000.0, 285_000.0, 400_000.0, CRR)
    assert decision["branch"] == "no_capacity_reduction_needed"
    assert decision["c_optimized_speedup"] >= HISTORICAL_SPEEDUP_REFERENCE


def test_capacity_stays_live_when_a_counterfactual_clears_the_reference() -> None:
    """c below the reference, d above it: the branch section H must weigh."""
    decision = decide_capacity_branch(2_000_000.0, 1_500_000.0, 900_000.0, CRR)
    assert decision["c_optimized_speedup"] < HISTORICAL_SPEEDUP_REFERENCE
    assert any(entry["clears_reference"] for entry in decision["d_counterfactuals"].values())
    assert decision["branch"] == "capacity_reduction_remains_live"
    assert "one remaining neural attempt" in decision["reason"]


def test_capacity_closes_when_neither_optimization_nor_reduction_clears_it() -> None:
    decision = decide_capacity_branch(9_000_000.0, 400_000.0, 8_000_000.0, CRR)
    assert decision["c_optimized_speedup"] < HISTORICAL_SPEEDUP_REFERENCE
    assert not any(entry["clears_reference"] for entry in decision["d_counterfactuals"].values())
    assert decision["branch"] == "capacity_reduction_closed_as_a_practical_remedy"


def test_both_declared_counterfactuals_are_reported() -> None:
    decision = decide_capacity_branch(2_000_000.0, 1_500_000.0, 900_000.0, CRR)
    assert set(decision["d_counterfactuals"]) == {
        f"{factor:g}x" for factor in COUNTERFACTUAL_BACKBONE_REDUCTIONS
    }


def test_the_reference_is_recorded_as_a_reference_not_a_gate() -> None:
    decision = decide_capacity_branch(1_085_000.0, 285_000.0, 400_000.0, CRR)
    assert decision["reference"] == HISTORICAL_SPEEDUP_REFERENCE
    assert "not a Task 9H gate" in decision["reference_status"]
    assert decision["descriptive_thresholds_do_not_decide"] is True


@pytest.mark.parametrize("crr", [0.0, -1.0])
def test_a_non_positive_reference_duration_is_refused(crr: float) -> None:
    with pytest.raises(InferenceProfileError):
        decide_capacity_branch(1.0, 0.5, 0.5, crr)
