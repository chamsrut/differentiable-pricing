"""Task 9H diagnostic publication: the artifact parent is created, once, safely.

``artifacts/`` is gitignored, so in a fresh clone -- and in every ``tmp_path``
here -- none of the per-diagnostic output directories exists. A writer that
assumed its parent was already there failed in ``tempfile.mkstemp`` before any
measurement could be published. These tests start from a root holding *no*
``artifacts`` tree at all and prove each writer creates only its own parent,
strictly after the path guards have accepted the request.

Nothing here trains, prices, opens a dataset partition or reads a real
checkpoint. Every measurement collaborator is stubbed: what is under test is the
publication layer, not the numerics it publishes.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Final

import pytest
from differentiable_pricing.ml.american_dev import (
    eligibility,
    frozen,
    greek_fidelity,
    greek_reference,
    heldout,
    inference_profile,
    price_fidelity,
)
from differentiable_pricing.ml.american_dev.attempts import (
    AttemptError,
    FinalPartitionAccessError,
)

PACKAGE: Final = Path(frozen.__file__).resolve().parent
PROVENANCE: Final = {"protocol_commit": "0" * 40, "repository": {"commit": "0" * 40}}

#: Enough rows for a 32768-row training subset and a disjoint 50000-row carve.
CARVE_ROWS: Final = 90_000


@pytest.fixture(autouse=True)
def _stub_provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    """Attest nothing: the clean-worktree attestation is not what is under test.

    ``diagnostic_provenance`` is bound at module scope by some writers and
    imported lazily from :mod:`.frozen` by others, so every binding is replaced.
    """
    for module in (frozen, eligibility, price_fidelity):
        monkeypatch.setattr(module, "diagnostic_provenance", lambda _root: dict(PROVENANCE))


def _absent(root: Path, relative: str) -> Path:
    """Assert the artifact's whole parent chain is missing, and return its path."""
    assert not (root / "artifacts").exists()
    return root / relative


def _published(path: Path) -> dict[str, Any]:
    """Assert exactly one complete artifact was published, and return it."""
    assert path.is_file()
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# The two writers the failure was first observed in
# ---------------------------------------------------------------------------


def _stub_freeze(monkeypatch: pytest.MonkeyPatch, *, present: bool = False) -> None:
    """``present`` varies the manifest's content, so a replacement is observable."""
    monkeypatch.setattr(
        frozen,
        "load_frozen_model",
        lambda _root, attempt_id: (None, {"attempt_id": attempt_id, "label": attempt_id}),
    )
    monkeypatch.setattr(frozen, "_ledger_facts", lambda _root, _attempt: {"present": present})


def test_the_freeze_manifest_creates_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_freeze(monkeypatch)
    path = _absent(tmp_path, frozen.DEFAULT_OUTPUT)
    report = frozen.freeze_checkpoints(tmp_path)
    assert report["schema_version"] == frozen.FROZEN_SCHEMA
    assert set(_published(path)["checkpoints"]) == set(frozen.FROZEN_ATTEMPTS)


def _stub_carve(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    from differentiable_pricing.ml.american_dev import workbench

    manifest = root / eligibility.DATASET_MANIFEST
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(workbench, "restricted_manifest", dict)
    monkeypatch.setattr(
        workbench,
        "load_partition",
        lambda _dataset, _manifest, _split: {
            "sample_id": [f"row-{index:07d}" for index in range(CARVE_ROWS)]
        },
    )


def test_the_carve_declaration_creates_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_carve(monkeypatch, tmp_path)
    path = _absent(tmp_path, heldout.DEFAULT_OUTPUT)
    report = heldout.declare_carve(tmp_path)
    assert report["counts"]["train_rows"] == CARVE_ROWS
    published = _published(path)
    assert published["halves"]["H1"]["rows"] == heldout.HALF_SIZE
    assert published["halves"]["H2"]["evaluable"] is False


# ---------------------------------------------------------------------------
# Every other diagnostic writer introduced in this phase
# ---------------------------------------------------------------------------


def test_the_eligibility_artifact_creates_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        eligibility, "assess", lambda _root, row_set, _columns: {"row_set": {"row_set": row_set}}
    )
    relative = eligibility.DEFAULT_OUTPUT_TEMPLATE.format(row_set="validation")
    path = _absent(tmp_path, relative)
    report, returned = eligibility.write(tmp_path, "validation", {})
    assert returned == path
    assert _published(path)["row_set"]["row_set"] == report["row_set"]["row_set"] == "validation"


def test_the_price_fidelity_report_creates_its_parent(tmp_path: Path) -> None:
    relative = price_fidelity.DEFAULT_OUTPUT_TEMPLATE.format(row_set="validation")
    path = _absent(tmp_path, relative)
    assert price_fidelity.write_report(tmp_path, relative, {"row_set": "validation"}) == path
    published = _published(path)
    assert published["schema_version"] == price_fidelity.PRICE_FIDELITY_SCHEMA
    assert published["provenance"] == PROVENANCE


def test_the_greek_reference_contract_creates_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(greek_reference, "declared_states", lambda _count: {})
    monkeypatch.setattr(
        greek_reference, "bump_convergence", lambda _states, thread_count: {"selected": True}
    )
    monkeypatch.setattr(
        greek_reference,
        "depth_convergence",
        lambda _states, _selection, thread_count: {"converged": True},
    )
    monkeypatch.setattr(
        greek_reference, "contract", lambda selection, convergence: {"operator": "stub"}
    )
    path = _absent(tmp_path, greek_reference.DEFAULT_OUTPUT)
    assert greek_reference.analyze(tmp_path)["operator"] == "stub"
    assert _published(path)["states"]["seed"] == greek_reference.STATE_SEED


def test_the_greek_grid_creates_its_parent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Grid 2 is the label-free scan: it needs no row set and no reference file."""
    monkeypatch.setattr(
        frozen, "load_frozen_model", lambda _root, attempt_id: (None, {"attempt_id": attempt_id})
    )
    monkeypatch.setattr(greek_fidelity, "scope_declaration", dict)
    monkeypatch.setattr(greek_fidelity, "scan_all", lambda _model, axis: {"axis": axis})
    relative = greek_fidelity.DEFAULT_OUTPUT_TEMPLATE.format(grid="2", suffix="")
    path = _absent(tmp_path, relative)
    assert greek_fidelity.analyze(tmp_path, "2")["axis"] == "spot"
    assert _published(path)["checkpoint"]["attempt_id"] == greek_fidelity.STUDIED_ATTEMPT


def test_the_inference_profile_creates_both_of_its_parents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The profile and the variants comparison are two separately guarded paths."""
    monkeypatch.setattr(
        frozen, "load_frozen_model", lambda _root, attempt_id: (None, {"attempt_id": attempt_id})
    )
    monkeypatch.setattr(inference_profile, "assert_profiled_head", lambda _model: None)
    monkeypatch.setattr(
        inference_profile,
        "component_profile",
        lambda _model, _request, warmups, repetitions: {"median_nanoseconds": {}},
    )
    monkeypatch.setattr(
        inference_profile,
        "measure_variants",
        lambda _model, _requests, warmups, repetitions: {"baseline": {}},
    )
    profile_output = "artifacts/task-9h/profile-only/inference-profile-v1.json"
    variants_output = "artifacts/task-9h/variants-only/inference-variants-v1.json"
    profile_path = _absent(tmp_path, profile_output)
    variants_path = tmp_path / variants_output
    report = inference_profile.profile(
        tmp_path, output=profile_output, variants_output=variants_output
    )
    assert set(report["component_profiles"]) == {"1", "8"}
    assert set(_published(profile_path)["component_profiles"]) == {"1", "8"}
    assert _published(variants_path)["measurements"] == {"baseline": {}}


# ---------------------------------------------------------------------------
# Creating the parent weakens nothing
# ---------------------------------------------------------------------------


def test_an_existing_artifact_is_still_refused_without_overwrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _stub_freeze(monkeypatch)
    _stub_carve(monkeypatch, tmp_path)
    frozen.freeze_checkpoints(tmp_path)
    heldout.declare_carve(tmp_path)
    manifest = (tmp_path / frozen.DEFAULT_OUTPUT).read_bytes()
    declaration = (tmp_path / heldout.DEFAULT_OUTPUT).read_bytes()
    with pytest.raises(frozen.FrozenCheckpointError, match="already exists"):
        frozen.freeze_checkpoints(tmp_path)
    with pytest.raises(heldout.HeldoutAccessError, match="already exists"):
        heldout.declare_carve(tmp_path)
    assert (tmp_path / frozen.DEFAULT_OUTPUT).read_bytes() == manifest
    assert (tmp_path / heldout.DEFAULT_OUTPUT).read_bytes() == declaration


def test_the_freeze_manifest_is_replaced_only_when_overwrite_is_asked_for(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``freeze --overwrite`` is documented, so the flag has to reach the writer.

    The exclusive publish and the replacing publish are different syscalls, and
    forwarding ``overwrite`` is the only thing that chooses between them: a
    ``freeze_checkpoints`` that guarded on the flag but published without it
    passed its own guard and then died in :func:`os.link`.
    """
    _stub_freeze(monkeypatch)
    path = tmp_path / frozen.DEFAULT_OUTPUT
    first = frozen.freeze_checkpoints(tmp_path)
    original = path.read_bytes()

    with pytest.raises(frozen.FrozenCheckpointError, match="already exists"):
        frozen.freeze_checkpoints(tmp_path)
    assert path.read_bytes() == original

    _stub_freeze(monkeypatch, present=True)
    replaced = frozen.freeze_checkpoints(tmp_path, overwrite=True)
    assert replaced != first
    published = _published(path)
    assert published == replaced
    assert all(entry["run_ledger"]["present"] for entry in published["checkpoints"].values())


def test_every_overwritable_publisher_forwards_its_flag() -> None:
    """A guard on ``overwrite`` that the writer never sees is not an option.

    Read out of the source: the flag is checked in one place and applied in
    another, so the two drifting apart is invisible until a rewrite is attempted
    -- which, for a once-per-phase artifact, is long after it was written.
    """
    checked = 0
    for module in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            arguments = node.args
            declared = {
                argument.arg
                for argument in (
                    *arguments.posonlyargs,
                    *arguments.args,
                    *arguments.kwonlyargs,
                )
            }
            if "overwrite" not in declared:
                continue
            for child in ast.walk(node):
                if (
                    not isinstance(child, ast.Call)
                    or not isinstance(child.func, ast.Name)
                    or child.func.id != "write_json_atomic"
                ):
                    continue
                forwarded = {
                    keyword.arg: ast.unparse(keyword.value) for keyword in child.keywords
                }
                assert forwarded.get("overwrite") == "overwrite", (
                    f"{module.name}:{node.name} accepts overwrite but publishes with "
                    f"overwrite={forwarded.get('overwrite', 'the default')}"
                )
                checked += 1
    assert checked >= 11


def test_an_existing_artifact_survives_a_refused_republication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``overwrite=False`` must leave the first artifact byte-for-byte intact."""
    relative = price_fidelity.DEFAULT_OUTPUT_TEMPLATE.format(row_set="validation")
    path = price_fidelity.write_report(tmp_path, relative, {"row_set": "validation"})
    first = path.read_bytes()
    with pytest.raises(FileExistsError):
        price_fidelity.write_report(tmp_path, relative, {"row_set": "replacement"})
    assert path.read_bytes() == first
    assert [entry.name for entry in path.parent.iterdir()] == [path.name]


def test_a_failed_write_publishes_nothing_into_the_created_parent(tmp_path: Path) -> None:
    """The parent may exist afterwards; a partial or temporary artifact may not."""
    relative = price_fidelity.DEFAULT_OUTPUT_TEMPLATE.format(row_set="validation")
    path = tmp_path / relative
    with pytest.raises(ValueError):
        price_fidelity.write_report(tmp_path, relative, {"nan": float("nan")})
    assert not path.exists()
    assert list(path.parent.iterdir()) == []


@pytest.mark.parametrize(
    ("relative", "error"),
    [
        ("docs/results/frozen-checkpoints-v1.json", frozen.FrozenCheckpointError),
        ("artifacts/../configs/frozen-checkpoints-v1.json", AttemptError),
        ("artifacts/task-9h/interpolation_test/report.json", FinalPartitionAccessError),
    ],
)
def test_a_guarded_path_is_refused_before_any_directory_is_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, relative: str, error: type[Exception]
) -> None:
    """Nothing is created for a request the guards reject -- not even the parent."""
    _stub_freeze(monkeypatch)
    with pytest.raises(error):
        frozen.freeze_checkpoints(tmp_path, relative)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# The assumption cannot quietly return in a writer added later
# ---------------------------------------------------------------------------


def _required_parent(argument: ast.expr) -> str | None:
    """The directory a ``write_json_atomic`` destination needs to already exist."""
    if isinstance(argument, ast.Name):
        return f"{argument.id}.parent"
    if isinstance(argument, ast.BinOp) and isinstance(argument.op, ast.Div):
        return ast.unparse(argument.left)
    return None


def test_every_publication_in_the_package_creates_its_own_parent() -> None:
    """Read out of the source, so a writer added at a later step is covered too.

    A destination named directly needs *its parent* made; one built as
    ``directory / "name.json"`` needs ``directory`` itself made.
    """
    checked = 0
    for module in sorted(PACKAGE.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"), filename=str(module))
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            made = {
                ast.unparse(child.func.value)
                for child in ast.walk(node)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Attribute)
                and child.func.attr == "mkdir"
            }
            for child in ast.walk(node):
                if (
                    not isinstance(child, ast.Call)
                    or not isinstance(child.func, ast.Name)
                    or child.func.id != "write_json_atomic"
                    or not child.args
                ):
                    continue
                required = _required_parent(child.args[0])
                assert required is not None, f"{module.name}:{node.name} destination is opaque"
                assert required in made, (
                    f"{module.name}:{node.name} publishes to {required[: -len('.parent')]} "
                    f"without creating {required}"
                )
                checked += 1
    assert checked >= 12
