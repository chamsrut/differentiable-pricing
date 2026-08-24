"""Task 9H H1/H2 carve: determinism, disjointness, and H2's unreachability.

Pure arithmetic over synthetic sample identifiers. No partition is opened, no
model is built and nothing is priced.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Final

import numpy as np
import pytest
from differentiable_pricing.ml.american_dev.attempts import (
    FORBIDDEN_PARTITION_TOKENS,
    select_rows_by_hash,
)
from differentiable_pricing.ml.american_dev.heldout import (
    CARVE_PARTITION,
    CARVE_SALT,
    CARVE_SIZE,
    EVALUABLE_HALVES,
    HALF_SIZE,
    HALVES,
    TRAIN_SUBSET_BUDGET,
    TRAIN_SUBSET_SALT,
    HeldoutAccessError,
    assert_half_evaluable,
    carve,
    declare,
    evaluable_columns,
    half_index,
)

PROJECT_ROOT: Final = Path(__file__).resolve().parents[3]
MODULE: Final = (
    PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev/heldout.py"
)
#: Big enough to exercise the real budgets, small enough to stay fast.
ROWS: Final = 100_000


def _ids(rows: int = ROWS, *, prefix: str = "row") -> list[str]:
    return [f"{prefix}-{index:07d}" for index in range(rows)]


# ---------------------------------------------------------------------------
# The rule
# ---------------------------------------------------------------------------


def test_the_carve_reuses_the_attempts_own_training_subset_rule() -> None:
    """The complement must be the complement of the rows that really trained."""
    ids = _ids()
    carved = carve(ids)
    expected = np.sort(select_rows_by_hash(ids, TRAIN_SUBSET_BUDGET, TRAIN_SUBSET_SALT))
    assert np.array_equal(carved.training_subset, expected)
    assert carved.complement.size == ROWS - TRAIN_SUBSET_BUDGET


def test_the_carve_is_deterministic_and_sized_as_declared() -> None:
    ids = _ids()
    first, second = carve(ids), carve(ids)
    assert np.array_equal(first.carve, second.carve)
    assert first.carve.size == CARVE_SIZE
    assert first.h1.size == HALF_SIZE
    assert first.h2.size == HALF_SIZE


def test_the_carve_identifies_a_set_not_an_ordering() -> None:
    """Presenting the same identifiers in another order selects the same rows.

    The declaration's digests are over sorted sample identifiers on purpose: two
    derivations that select the same rows have to agree even if they enumerated
    the partition differently.
    """
    ids = _ids()
    shuffled = list(reversed(ids))
    straight = declare(ids)
    reversed_declaration = declare(shuffled)
    for half in HALVES:
        assert (
            straight["halves"][half]["sample_id_sha256"]
            == reversed_declaration["halves"][half]["sample_id_sha256"]
        )


def test_the_declaration_verifies_disjointness_rather_than_describing_it() -> None:
    declaration = declare(_ids())
    assert declaration["disjointness_verified"] == {
        "H1_and_H2": True,
        "H1_and_training_subset": True,
        "H2_and_training_subset": True,
    }
    assert declaration["counts"]["training_subset"] == TRAIN_SUBSET_BUDGET
    assert declaration["counts"]["carve"] == CARVE_SIZE
    for half in HALVES:
        assert declaration["halves"][half]["rows"] == HALF_SIZE
        assert len(declaration["halves"][half]["sample_id_sha256"]) == 64


def test_the_two_halves_are_disjoint_as_row_sets() -> None:
    carved = carve(_ids())
    assert not (set(carved.h1.tolist()) & set(carved.h2.tolist()))
    assert not (set(carved.h1.tolist()) & set(carved.training_subset.tolist()))
    assert not (set(carved.h2.tolist()) & set(carved.training_subset.tolist()))


def test_the_declaration_states_what_h1_is_not() -> None:
    declaration = declare(_ids())
    assert "not an independently generated partition" in (
        declaration["not_an_independent_partition"]
    )
    assert "No threshold, gate or model-selection" in declaration["no_selection_gate"]
    assert declaration["partition"] == CARVE_PARTITION


def test_a_partition_too_small_to_carve_fails_closed() -> None:
    with pytest.raises(HeldoutAccessError):
        carve(_ids(TRAIN_SUBSET_BUDGET + CARVE_SIZE))


def test_duplicate_identifiers_fail_closed() -> None:
    ids = _ids(TRAIN_SUBSET_BUDGET + CARVE_SIZE + 10)
    ids[0] = ids[1]
    with pytest.raises(HeldoutAccessError):
        carve(ids)


# ---------------------------------------------------------------------------
# H2 is unreachable
# ---------------------------------------------------------------------------


def test_only_h1_is_evaluable() -> None:
    assert EVALUABLE_HALVES == ("H1",)
    assert assert_half_evaluable("H1") == "H1"


@pytest.mark.parametrize("half", ["H2", "h2", "H3", ""])
def test_every_other_half_is_refused(half: str) -> None:
    with pytest.raises(HeldoutAccessError):
        assert_half_evaluable(half)


def test_the_reserved_half_cannot_be_turned_into_row_positions() -> None:
    carved = carve(_ids())
    with pytest.raises(HeldoutAccessError, match="reserved"):
        half_index(carved, "H2")
    assert half_index(carved, "H1").size == HALF_SIZE


def test_the_reserved_half_cannot_be_turned_into_columns() -> None:
    ids = np.asarray(_ids(), dtype=object)
    columns = {"sample_id": ids, "spot": np.arange(ids.size, dtype=np.float64)}
    with pytest.raises(HeldoutAccessError, match="reserved"):
        evaluable_columns(columns, "H2")
    restricted = evaluable_columns(columns, "H1")
    assert restricted["spot"].size == HALF_SIZE
    assert restricted["sample_id"].size == HALF_SIZE


def test_the_columns_door_needs_sample_identifiers() -> None:
    with pytest.raises(HeldoutAccessError, match="sample_id"):
        evaluable_columns({"spot": np.zeros(3)}, "H1")


def test_no_function_returns_reserved_rows_without_passing_the_gate() -> None:
    """Every path from the carve to rows must call ``assert_half_evaluable``.

    Read out of the source rather than asserted in prose: a new accessor added
    later without the gate is exactly the failure this guards against.
    """
    tree = ast.parse(MODULE.read_text(encoding="utf-8"), filename=str(MODULE))
    gated = {"half_index", "evaluable_columns"}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in gated:
            called = {
                child.func.id
                for child in ast.walk(node)
                if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
            }
            assert "assert_half_evaluable" in called, node.name


# ---------------------------------------------------------------------------
# The carve never names a partition Task 9H may not reach
# ---------------------------------------------------------------------------


def test_the_carve_salt_names_no_forbidden_partition() -> None:
    """``holdout`` is reserved for final partitions; ``heldout`` is a different word.

    The two are one letter apart, so the relationship is checked rather than
    trusted: these reserved rows live inside ``train``, which Task 9H is
    explicitly permitted to read.
    """
    for token in FORBIDDEN_PARTITION_TOKENS:
        assert token not in CARVE_SALT.lower()
        assert token not in TRAIN_SUBSET_SALT.lower()
        assert token not in CARVE_PARTITION.lower()


def test_the_module_names_no_partition_but_train() -> None:
    """The only split literal the carve module spells is ``train``."""
    text = MODULE.read_text(encoding="utf-8")
    for token in FORBIDDEN_PARTITION_TOKENS:
        assert f'"{token}"' not in text
        assert f"'{token}'" not in text
