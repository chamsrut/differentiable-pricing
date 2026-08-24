"""Task 9H assessability: the ordering, and the four ways it can be defeated.

No dataset partition is opened and no model is built.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

import numpy as np
import pytest
import torch
from differentiable_pricing.ml.american_dev.attempts import FinalPartitionAccessError
from differentiable_pricing.ml.american_dev.eligibility import (
    ASSESSABLE_ROW_SETS,
    ELIGIBILITY_SCHEMA,
    EligibilityError,
    assert_matches,
    assert_row_set,
    assess_rows,
    compact,
    conditional_subset,
    load,
    row_set_identity,
)
from differentiable_pricing.ml.american_dev.frozen import declaration_state
from differentiable_pricing.ml.american_dev.representation import european_price
from differentiable_pricing.ml.american_dev.tolerance import (
    ASSESSABILITY_DEGENERATE_LIMIT,
    NOT_ASSESSABLE,
    VEGA_DEGENERACY_NORMALIZED,
    assessability,
    eligible_vega_diagnostics,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
ROWS: Final = 64


def _columns(rows: int = ROWS, *, prefix: str = "row") -> dict[str, np.ndarray]:
    """Rows spanning the money, plus deliberately degenerate deep-ITM contracts."""
    generator = np.random.default_rng(99)
    spot = generator.uniform(80.0, 120.0, rows)
    strike = spot * np.exp(-generator.uniform(-0.5, 0.5, rows))
    maturity = generator.uniform(0.05, 2.0, rows)
    volatility = generator.uniform(0.1, 0.6, rows)
    # A handful of near-zero-vega states, so degeneracy is exercised rather than
    # assumed absent.
    spot[:4] = 145.0
    strike[:4] = 55.0
    maturity[:4] = 0.03
    volatility[:4] = 0.06
    option_type = np.where(np.arange(rows) % 2 == 0, "call", "put")
    physical = torch.tensor(
        np.column_stack(
            [
                np.where(option_type == "call", 1.0, -1.0),
                spot,
                strike,
                maturity,
                np.zeros(rows),
                np.zeros(rows),
                volatility,
            ]
        ),
        dtype=torch.float64,
    )
    with torch.no_grad():
        european = european_price(physical).numpy()
    return {
        "sample_id": np.array([f"{prefix}-{index:05d}" for index in range(rows)], dtype=object),
        "option_type": option_type,
        "spot": spot,
        "strike": strike,
        "maturity": maturity,
        "rate": np.zeros(rows),
        "dividend_yield": np.zeros(rows),
        "volatility": volatility,
        "european_crr_price": european,
    }


@pytest.fixture(scope="module")
def columns() -> dict[str, np.ndarray]:
    return _columns()


@pytest.fixture(scope="module")
def record(columns: dict[str, np.ndarray]) -> dict[str, Any]:
    return {
        **assess_rows("validation", columns),
        "provenance": declaration_state(PROJECT_ROOT),
    }


# ---------------------------------------------------------------------------
# What may be assessed
# ---------------------------------------------------------------------------


def test_only_validation_and_h1_are_assessable() -> None:
    assert ASSESSABLE_ROW_SETS == ("validation", "H1")
    assert assert_row_set("validation") == "validation"
    assert assert_row_set("H1") == "H1"


@pytest.mark.parametrize("row_set", ["H2", "interpolation_test", "train", "test"])
def test_every_other_row_set_is_refused(row_set: str) -> None:
    with pytest.raises((EligibilityError, FinalPartitionAccessError)):
        assert_row_set(row_set)


def test_a_row_set_identity_names_the_set_not_an_ordering(
    columns: dict[str, np.ndarray]
) -> None:
    straight = row_set_identity("validation", columns)
    reversed_columns = {key: value[::-1] for key, value in columns.items()}
    assert row_set_identity("validation", reversed_columns) == straight
    assert straight["rows"] == ROWS
    assert len(straight["sample_id_sha256"]) == 64


# ---------------------------------------------------------------------------
# The measurement reads no model
# ---------------------------------------------------------------------------


def test_assessability_reads_no_model_and_no_prediction(record: dict[str, Any]) -> None:
    assert record["schema_version"] == ELIGIBILITY_SCHEMA
    assert record["status"]["reads_a_model"] is False
    assert record["status"]["reads_a_prediction"] is False
    assert record["status"]["computed_before_scoring"] is True
    assert record["status"]["creates_a_model_gate"] is False


def test_the_degeneracy_threshold_is_derived_from_the_declared_tolerances() -> None:
    """5.5e-4 / 0.01 = 0.055, not a third asserted number."""
    assert pytest.approx(0.055) == VEGA_DEGENERACY_NORMALIZED
    diagnostics = eligible_vega_diagnostics(
        torch.tensor([[1.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.2]], dtype=torch.float64),
        np.array([10.0]),
    )
    assert diagnostics["degeneracy"]["threshold_normalized_vega"] == pytest.approx(0.055)
    assert "SECONDARY_NORMALIZED_TOLERANCE" in (
        diagnostics["degeneracy"]["threshold_derivation"]
    )


def test_the_verdict_follows_the_predeclared_one_percent_rule(
    record: dict[str, Any]
) -> None:
    verdict = record["assessability"]
    fraction = verdict["degenerate_eligible_fraction"]
    assert verdict["limit"] == ASSESSABILITY_DEGENERATE_LIMIT
    assert verdict["predeclared"] is True
    if fraction is not None and fraction > ASSESSABILITY_DEGENERATE_LIMIT:
        assert verdict["verdict"] == NOT_ASSESSABLE
        assert verdict["primary_statistic_assessable"] is False
    else:
        assert verdict["verdict"] == "assessable"


def test_a_fully_resolvable_set_is_assessable() -> None:
    """The rule must be capable of returning 'assessable', or it decides nothing."""
    physical = torch.tensor(
        [[1.0, 100.0, 100.0, 1.0, 0.0, 0.0, 0.2]] * 8, dtype=torch.float64
    )
    with torch.no_grad():
        european = european_price(physical).numpy()
    diagnostics = eligible_vega_diagnostics(physical, european)
    verdict = assessability(diagnostics)
    assert verdict["primary_statistic_assessable"] is True
    assert verdict["verdict"] == "assessable"


def test_the_secondary_view_is_marked_descriptive_and_discloses_its_origin(
    record: dict[str, Any]
) -> None:
    view = record["vega_diagnostics"]["secondary_view"]
    assert view["descriptive_only"] is True
    assert view["never_replaces_the_declared_rule"] is True
    disclosure = view["disclosure"]
    assert "synthetic smoke exercise" in disclosure
    assert "not the performance of any trained checkpoint" in disclosure
    assert "no E2c result was observed" in disclosure


def test_the_conditional_subset_excludes_degenerate_eligible_rows(
    columns: dict[str, np.ndarray]
) -> None:
    physical = torch.as_tensor(
        np.column_stack(
            [
                np.where(columns["option_type"] == "call", 1.0, -1.0),
                columns["spot"],
                columns["strike"],
                columns["maturity"],
                columns["rate"],
                columns["dividend_yield"],
                columns["volatility"],
            ]
        ),
        dtype=torch.float64,
    )
    diagnostics = eligible_vega_diagnostics(physical, columns["european_crr_price"])
    subset = conditional_subset(diagnostics)
    eligible = np.asarray(diagnostics["_eligible"], dtype=bool)
    degenerate = np.asarray(diagnostics["_degenerate"], dtype=bool)
    assert bool((subset <= eligible).all())
    assert not bool((subset & degenerate).any())


def test_the_report_carries_no_per_row_arrays(record: dict[str, Any]) -> None:
    assert not [key for key in record["vega_diagnostics"] if key.startswith("_")]


def test_the_compact_view_answers_the_only_question_that_matters(
    record: dict[str, Any]
) -> None:
    view = compact(record)
    assert view["row_set"] == "validation"
    assert view["verdict"] in {"assessable", NOT_ASSESSABLE}
    assert "degenerate_eligible_fraction" in view
    assert "eligible_fraction_below_1e-4_normalized_vega" in view


# ---------------------------------------------------------------------------
# The four ways the ordering could be defeated
# ---------------------------------------------------------------------------


def test_matching_accepts_the_rows_it_was_computed_on(
    record: dict[str, Any], columns: dict[str, np.ndarray]
) -> None:
    assert assert_matches(record, PROJECT_ROOT, "validation", columns)["row_set"] == (
        record["row_set"]
    )


def test_a_different_row_set_is_refused(
    record: dict[str, Any], columns: dict[str, np.ndarray]
) -> None:
    with pytest.raises(EligibilityError, match="describes exactly"):
        assert_matches(record, PROJECT_ROOT, "H1", columns)


def test_different_rows_of_the_same_partition_are_refused(
    record: dict[str, Any], columns: dict[str, np.ndarray]
) -> None:
    other = _columns(prefix="other")
    with pytest.raises(EligibilityError, match="describes exactly"):
        assert_matches(record, PROJECT_ROOT, "validation", other)


def test_an_edited_declaration_invalidates_the_artifact(
    record: dict[str, Any], columns: dict[str, np.ndarray]
) -> None:
    stale = {
        **record,
        "provenance": {**record["provenance"], "declaration_digests": {"tolerance.py": "0" * 64}},
    }
    with pytest.raises(EligibilityError, match="declarations changed"):
        assert_matches(stale, PROJECT_ROOT, "validation", columns)


def test_an_artifact_from_another_protocol_commit_is_refused(
    record: dict[str, Any], columns: dict[str, np.ndarray]
) -> None:
    stale = {**record, "provenance": {**record["provenance"], "protocol_commit": "0" * 40}}
    with pytest.raises(EligibilityError, match="another source state"):
        assert_matches(stale, PROJECT_ROOT, "validation", columns)


def test_an_unknown_schema_is_refused(
    record: dict[str, Any], columns: dict[str, np.ndarray]
) -> None:
    with pytest.raises(EligibilityError, match="schema"):
        assert_matches({**record, "schema_version": "other/1"}, PROJECT_ROOT, "validation", columns)


def test_a_missing_artifact_explains_the_ordering(tmp_path: Path) -> None:
    with pytest.raises(EligibilityError, match="before"):
        load(tmp_path, "validation")
