"""Task 9H frozen checkpoints: immutable identity, and the deployed function.

**Diagnostic phase only.** Nothing here trains, retrains, fine-tunes or writes a
weight. It rebuilds two already-completed attempts from their tracked immutable
configurations plus the checkpoints those runs wrote, records exactly what they
are, and exposes the pieces of the deployed head that the price-fidelity and
Greek analyses need to address separately.

Two checkpoints are frozen:

* **E2b** ``scratch_residual_smooth_floor_raw_loss_v1`` -- zero-margin floor;
* **E2c** ``scratch_residual_smooth_floor_margin_v1`` -- ``delta = 1e-4`` floor.

**Why a registry rather than a path argument.** A diagnostic that will be quoted
against a decision has to name which weights produced each number. A closed
registry makes "which checkpoint is this?" answerable from the code rather than
from whichever directory the caller happened to pass.

**The margin is a deployment-time transformation, not a weight.** Both frozen
weight sets can therefore be evaluated under both floor definitions, which is
what :func:`normalized_with_margin` exists for. It is an *evaluation* override:
it never mutates a model, and at the head's own margin it reproduces
:meth:`~.representation.AmericanDevPriceModel.normalized_target` bitwise, which
is asserted in the tests rather than assumed.

**Checkpoint identity: what the evidence actually is.** The strongest evidence
that a frozen checkpoint is the one an attempt produced is **functional**: under
its native deployment head it reproduces that attempt's digest-verified
25,000-row report metrics to float64 agreement. A wrong weight set does not
reproduce five error statistics and seven diagnostic counts by coincidence.

Structural evidence supports it: the state dict loads ``strict=True`` into the
architecture rebuilt from the immutable configuration, and the parameter count
matches the 199,041 the attempt recorded.

**The separate provenance limitation** is that ``checkpoint.pt`` had no
run-time-recorded digest in the historical ledger -- the ledger digests the
attempt report and the compact summary only. The checkpoint's current file
digest therefore cannot be compared against an original run-time checkpoint
digest. It is recorded as observed at freeze time, and the functional check
above is what stands in for the missing comparison.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

from ..artifact import write_json_atomic
from ..model import Scaling
from .attempts import (
    ATTEMPT_LOG_PATH,
    _git,
    assert_contained_relative_path,
    assert_path_allowed,
    attempt_log_entries,
    load_toml,
    repository_identity,
    sha256_file,
    source_digests,
    validate_attempt_config,
)
from .representation import (
    AmericanDevPriceModel,
    discounted_spot,
    european_price,
    intrinsic_value,
    normalized_lower_floor,
    smooth_lower_bound,
)
from .workbench import build_model

FROZEN_SCHEMA: Final = "american-dev-frozen-checkpoints/1"

#: The closed set of frozen checkpoints, and the ignored directory each
#: completed attempt wrote. A checkpoint outside this mapping is not freezable
#: and not addressable by any diagnostic in this phase.
FROZEN_ATTEMPTS: Final = {
    "scratch_residual_smooth_floor_raw_loss_v1": {
        "label": "E2b",
        "directory": "artifacts/task-9h/scratch_residual_smooth_floor_raw_loss_v1",
    },
    "scratch_residual_smooth_floor_margin_v1": {
        "label": "E2c",
        "directory": "artifacts/task-9h/scratch_residual_smooth_floor_margin_v1",
    },
}

ATTEMPT_REPORT_NAME: Final = "attempt-report.json"
CHECKPOINT_NAME: Final = "checkpoint.pt"
LEDGER_DIRECTORY: Final = "runs/task-9h"
LEDGER_NAME: Final = "attempt.json"

#: Where the freeze report is written: beneath the ignored ``artifacts/`` tree.
#: A development record, never frozen evidence under ``docs/results/``.
DEFAULT_OUTPUT: Final = "artifacts/task-9h/frozen/frozen-checkpoints-v1.json"

#: Heads whose deployed output is a smooth one-sided projection onto a floor.
#: Only these carry a temperature and a margin, and only these can be evaluated
#: under an overridden margin.
FLOOR_HEADS: Final = (
    "smooth_lower_floor",
    "smooth_lower_floor_raw_loss",
    "smooth_lower_floor_margin_raw_loss",
)

#: The exact deployed function, written out rather than described. Every later
#: report cites this string beside the checkpoint that produced its numbers, so
#: "which deployment floor was this?" is answerable from the artifact alone.
DEPLOYMENT_FUNCTION: Final = (
    "A       = S * exp(-q * T)\n"
    "raw     = network(standardize(base_features(x)))\n"
    "u_dir   = raw * price_scale + price_mean\n"
    "E       = analytic Black-Scholes European price at x, continuous yield\n"
    "I       = max(omega * (S - K), 0)\n"
    "e_leg   = E / A + delta\n"
    "i_leg   = I / A\n"
    "floor   = max(e_leg, i_leg) + tau * log1p(exp(-|e_leg - i_leg| / tau))\n"
    "u_dep   = floor + tau * softplus((u_dir - floor) / tau)\n"
    "V_hat   = A * u_dep\n"
    "with omega = +1 for a call and -1 for a put, tau the head temperature and\n"
    "delta the additive European-leg margin. The margin is added to the European\n"
    "leg only; the intrinsic leg carries none."
)

#: The projection weight, named once so no report invents a threshold for it.
PROJECTION_WEIGHT_DEFINITION: Final = (
    "s = sigmoid((u_dir - floor) / tau), the exact derivative of the deployed "
    "output with respect to the pre-projection direct normalized price; s -> 0 "
    "is floor-dominated and s -> 1 is network-dominated"
)

#: The floor's own leg weight, constructed the same way for the same reason.
FLOOR_LEG_WEIGHT_DEFINITION: Final = (
    "w = sigmoid((E / A + delta - I / A) / tau), the exact derivative of the "
    "floor's smooth maximum with respect to its European leg; w -> 1 is "
    "European-leg dominance and w -> 0 is intrinsic-leg dominance"
)


class FrozenCheckpointError(RuntimeError):
    """Raised when a frozen checkpoint cannot be identified or rebuilt safely."""


# ---------------------------------------------------------------------------
# Guards
# ---------------------------------------------------------------------------


def _guarded_artifact_path(project_root: Path, relative: str, *, where: str) -> Path:
    assert_path_allowed(relative, where=where)
    assert_contained_relative_path(relative, where=where)
    if not str(relative).replace("\\", "/").startswith("artifacts/"):
        raise FrozenCheckpointError(
            f"{where} lives beneath the ignored artifacts/ tree; a Task 9H checkpoint and "
            "its diagnostic reports are development records, not frozen evidence"
        )
    return Path(project_root) / relative


def assert_frozen(attempt_id: str) -> dict[str, str]:
    """Resolve one attempt ID against the closed frozen registry."""
    entry = FROZEN_ATTEMPTS.get(str(attempt_id))
    if entry is None:
        raise FrozenCheckpointError(
            f"attempt {attempt_id!r} is not a frozen Task 9H checkpoint; the frozen set is "
            f"{sorted(FROZEN_ATTEMPTS)}"
        )
    return dict(entry)


# ---------------------------------------------------------------------------
# Rebuilding a frozen model from committed configuration plus its checkpoint
# ---------------------------------------------------------------------------


def recorded_scaling(report: Mapping[str, Any]) -> Scaling:
    """The train-fitted scaling the attempt recorded, rebuilt exactly.

    The checkpoint stores the network's weights only, so the affine input and
    target transforms come from the attempt report. They are read, never
    refitted: refitting would require opening a partition, and a scaling that
    differs from the one training used is a different model.
    """
    section = report.get("scaling")
    if not isinstance(section, Mapping):
        raise FrozenCheckpointError("the attempt report records no scaling")
    try:
        scaling = Scaling(
            feature_mean=np.asarray(section["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(section["feature_scale"], dtype=np.float64),
            price_mean=float(section["target_mean"]),
            price_scale=float(section["target_scale"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise FrozenCheckpointError(f"the recorded scaling is unusable: {error}") from error
    if (
        not bool(np.isfinite(scaling.feature_mean).all())
        or not bool(np.isfinite(scaling.feature_scale).all())
        or bool((scaling.feature_scale <= 0.0).any())
        or not np.isfinite(scaling.price_scale)
        or scaling.price_scale <= 0.0
    ):
        raise FrozenCheckpointError("the recorded scaling is not finite with positive scales")
    return scaling


def load_frozen_model(
    project_root: Path, attempt_id: str, directory: str | None = None
) -> tuple[AmericanDevPriceModel, dict[str, Any]]:
    """Rebuild one frozen attempt's deployed model, with its full provenance.

    The architecture, head and conditioning features come from the **tracked,
    immutable attempt configuration**, not from the report, and the report's
    recorded configuration digest is required to match the tracked file's -- the
    same immutability property ``scripts/american_dev_attempts.py check``
    enforces for the log. The attempt is also required to be recorded in the
    append-only attempt log under that same digest, so a checkpoint from an
    unrecorded run cannot be frozen as if it were part of the search.

    The model is returned in ``eval()`` mode with gradients still enabled on its
    parameters: the Greek study differentiates through it, so disabling them
    here would silently break section F.
    """
    project_root = Path(project_root)
    entry = assert_frozen(attempt_id)
    relative_directory = str(directory) if directory is not None else entry["directory"]
    attempt_directory = _guarded_artifact_path(
        project_root, relative_directory, where="frozen attempt directory"
    )
    report_path = attempt_directory / ATTEMPT_REPORT_NAME
    checkpoint_path = attempt_directory / CHECKPOINT_NAME
    for path in (report_path, checkpoint_path):
        if not path.is_file():
            raise FrozenCheckpointError(
                f"'{path}' does not exist; a frozen checkpoint comes from a completed attempt "
                "and this phase trains nothing"
            )
    if report_path.stat().st_size == 0:
        raise FrozenCheckpointError(
            f"'{report_path}' is empty; this attempt's evidence did not survive and it cannot "
            "be frozen -- see attempts.UNRECORDABLE_ATTEMPTS"
        )
    report = json.loads(report_path.read_text(encoding="utf-8"))
    recorded_id = str(report.get("attempt_id"))
    if recorded_id != attempt_id:
        raise FrozenCheckpointError(
            f"'{relative_directory}' records attempt {recorded_id!r}, not {attempt_id!r}"
        )
    relative_config = str(report.get("config_path"))
    assert_path_allowed(relative_config, where="frozen attempt configuration")
    assert_contained_relative_path(relative_config, where="frozen attempt configuration")
    config_path = project_root / relative_config
    if not config_path.is_file():
        raise FrozenCheckpointError(f"'{relative_config}' does not exist")
    config_digest = sha256_file(config_path)
    if config_digest != str(report.get("config_sha256")):
        raise FrozenCheckpointError(
            f"'{relative_config}' hashes to {config_digest[:12]}… but the attempt recorded "
            f"{str(report.get('config_sha256'))[:12]}…; a used configuration is immutable"
        )
    logged = [
        record
        for record in attempt_log_entries(project_root / ATTEMPT_LOG_PATH)
        if str(record.get("attempt_id")) == attempt_id
    ]
    if len(logged) != 1 or str(logged[0].get("config_sha256")) != config_digest:
        raise FrozenCheckpointError(
            f"attempt {attempt_id!r} is not recorded exactly once in '{ATTEMPT_LOG_PATH}' under "
            "the configuration digest it ran; an unrecorded checkpoint is not frozen"
        )

    config = load_toml(config_path)
    validate_attempt_config(config)
    model = build_model(config, recorded_scaling(report))
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state, strict=True)
    model.eval()
    parameters = int(sum(int(tensor.numel()) for tensor in model.network.parameters()))
    recorded_parameters = report.get("architecture", {}).get("parameters")
    if recorded_parameters is not None and int(recorded_parameters) != parameters:
        raise FrozenCheckpointError(
            f"the rebuilt network has {parameters} parameters but the attempt recorded "
            f"{int(recorded_parameters)}; the checkpoint does not match its configuration"
        )
    provenance = {
        "attempt_id": attempt_id,
        "label": entry["label"],
        "attempt_directory": relative_directory,
        "config_path": relative_config,
        "config_sha256": config_digest,
        "checkpoint_path": f"{relative_directory}/{CHECKPOINT_NAME}",
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_digest_provenance": (
            "observed at freeze time. checkpoint.pt had no run-time-recorded digest in the "
            "historical ledger -- the ledger digests the attempt report and the compact "
            "summary only -- so this digest cannot be compared against an original run-time "
            "checkpoint digest."
        ),
        "identity_evidence": {
            "functional": (
                "the strongest evidence: under its native deployment head this checkpoint "
                "reproduces the diagonal cell's digest-verified 25000-row attempt-report "
                "metrics to float64 agreement. Recorded by the price-fidelity decomposition, "
                "which evaluates that cell; a wrong weight set does not reproduce five error "
                "statistics and seven diagnostic counts by coincidence."
            ),
            "structural_state_dict_strict": True,
            "structural_architecture_source": "the tracked immutable attempt configuration",
            "structural_parameter_count": parameters,
        },
        "attempt_report_path": f"{relative_directory}/{ATTEMPT_REPORT_NAME}",
        "attempt_report_sha256": sha256_file(report_path),
        "summary_path": f"{relative_directory}/summary.json",
        "recorded_in": ATTEMPT_LOG_PATH,
        "git_commit": report.get("repository", {}).get("commit"),
        "head": str(config["head"]),
        "head_temperature": model.temperature,
        "head_european_margin": model.european_margin,
        "conditioning_features": list(config.get("conditioning_features", ())),
        "architecture": dict(config["architecture"]),
        "parameters": parameters,
        "representation": model.representation,
        "seeds": report.get("seeds"),
        "best_epoch": report.get("training", {}).get("best_epoch"),
        "loss_prediction": report.get("training", {}).get("loss_prediction"),
        "selection_prediction": report.get("training", {}).get("selection_prediction"),
        "scaling_source": f"{relative_directory}/{ATTEMPT_REPORT_NAME} :: scaling",
        "deployment_function": DEPLOYMENT_FUNCTION,
    }
    return model, provenance


# ---------------------------------------------------------------------------
# Addressing the deployed head's pieces separately
# ---------------------------------------------------------------------------


def assert_floor_head(model: AmericanDevPriceModel) -> float:
    """Require a floor head and return its temperature."""
    if model.head not in FLOOR_HEADS or model.temperature is None:
        raise FrozenCheckpointError(
            f"head {model.head!r} carries no smooth floor; the projection weight, the margin "
            "override and the leg decomposition are defined only for a floor head"
        )
    return float(model.temperature)


def direct_normalized(
    model: AmericanDevPriceModel, physical_features: torch.Tensor
) -> torch.Tensor:
    """The **pre-projection** direct normalized price ``u_dir``.

    This is the raw learned function -- the object section F contrasts against
    the deployed output. It calls the model's own two helpers rather than
    recomputing the standardization and the affine target transform, so the
    "raw network" this phase measures cannot drift from the one that trained.
    """
    return model._direct_from_raw(model._network_output(physical_features))


def deployed_floor(
    model: AmericanDevPriceModel,
    physical_features: torch.Tensor,
    european_margin: float | None = None,
) -> torch.Tensor:
    """The normalized lower floor, optionally under an overridden margin."""
    temperature = assert_floor_head(model)
    margin = model.european_margin if european_margin is None else float(european_margin)
    return normalized_lower_floor(physical_features, temperature, margin)


def normalized_with_margin(
    model: AmericanDevPriceModel,
    physical_features: torch.Tensor,
    european_margin: float | None = None,
) -> torch.Tensor:
    """The deployed normalized output with the European-leg margin overridden.

    **The whole point of B.0.** The margin is a deployment-time transformation
    rather than a weight, so a frozen weight set can be evaluated under a floor
    it was not selected under. Fixing the weights and varying the margin
    isolates the margin's own effect; fixing the margin and varying the weights
    isolates attempt-to-attempt variation.

    At ``european_margin = None`` -- or at the head's own margin -- this
    reproduces :meth:`~.representation.AmericanDevPriceModel.normalized_target`
    bitwise, because it composes the same two functions in the same order with
    the same arguments. That is asserted in the tests, not assumed here.

    Nothing is mutated: the model's ``european_margin`` attribute is read, never
    written.
    """
    temperature = assert_floor_head(model)
    return smooth_lower_bound(
        direct_normalized(model, physical_features),
        deployed_floor(model, physical_features, european_margin),
        temperature,
    )


def price_with_margin(
    model: AmericanDevPriceModel,
    physical_features: torch.Tensor,
    european_margin: float | None = None,
) -> torch.Tensor:
    """``V = A * u_dep`` under an overridden margin: the physical deployed price."""
    return discounted_spot(physical_features) * normalized_with_margin(
        model, physical_features, european_margin
    )


def raw_price(model: AmericanDevPriceModel, physical_features: torch.Tensor) -> torch.Tensor:
    """``V = A * u_dir``: the physical price of the **unprojected** network."""
    return discounted_spot(physical_features) * direct_normalized(model, physical_features)


def projection_weight(
    model: AmericanDevPriceModel,
    physical_features: torch.Tensor,
    european_margin: float | None = None,
) -> torch.Tensor:
    """``s = sigmoid((u_dir - floor) / tau)``: the deployment behaviour partition.

    It is the exact derivative of the deployed output with respect to the direct
    normalized price, so it needs no invented threshold: ``s`` is how much of the
    answer the network is supplying and ``1 - s`` is how much the floor is.
    """
    temperature = assert_floor_head(model)
    direct = direct_normalized(model, physical_features)
    floor = deployed_floor(model, physical_features, european_margin)
    return torch.sigmoid((direct - floor) / temperature)


def floor_leg_weight(
    model: AmericanDevPriceModel,
    physical_features: torch.Tensor,
    european_margin: float | None = None,
) -> torch.Tensor:
    """``w = sigmoid((e_leg - i_leg) / tau)``: which leg of the floor dominates.

    Constructed the same way and for the same reason as
    :func:`projection_weight` -- it is the exact derivative of the floor's smooth
    maximum with respect to its European leg. The two legs behave completely
    differently under differentiation: the intrinsic leg is independent of
    volatility, so wherever it dominates the deployed Vega and Gamma are zero by
    construction. Section F needs that distinction, so it is computed here from
    the same primitives the floor itself uses.
    """
    temperature = assert_floor_head(model)
    margin = model.european_margin if european_margin is None else float(european_margin)
    scale = discounted_spot(physical_features)
    european = european_price(physical_features) / scale + margin
    intrinsic = intrinsic_value(physical_features) / scale
    return torch.sigmoid((european - intrinsic) / temperature)


# ---------------------------------------------------------------------------
# Evaluation-only views of one frozen model
# ---------------------------------------------------------------------------


class _FrozenView(torch.nn.Module):
    """Base for a read-only view that presents a frozen model as a pricer.

    Deliberately an ``nn.Module`` with a ``forward`` that returns a physical
    price, because that is the **entire** interface
    :func:`..american_pilot.predict_prices` needs. Wrapping rather than
    reimplementing means the Task 9G price metrics, the bound diagnostics and
    the shape diagnostics all run unchanged over a view -- including their
    bumped-state predictions, which is the part a "just recompute the centre
    prediction" shortcut would silently get wrong.

    The wrapped model is held by reference and never mutated.
    """

    def __init__(self, model: AmericanDevPriceModel) -> None:
        super().__init__()
        self.model = model
        self.eval()

    def _check(self, physical_features: torch.Tensor) -> None:
        if (
            physical_features.dtype != torch.float64
            or physical_features.device.type != "cpu"
        ):
            raise FrozenCheckpointError(
                "Task 9H American pricing requires a float64 CPU tensor"
            )


class MarginOverrideModel(_FrozenView):
    """One frozen weight set deployed under a **stated** European-leg margin.

    The object the B.0 2x2 is built from. Each of the four cells is a
    ``(weights, margin)`` pair, and each pair is a view like this one, so a cell
    is evaluated by exactly the code that evaluated the attempt it came from.
    """

    def __init__(self, model: AmericanDevPriceModel, european_margin: float) -> None:
        super().__init__(model)
        assert_floor_head(model)
        if not math.isfinite(european_margin) or european_margin < 0.0:
            raise FrozenCheckpointError(
                "the European-leg margin must be a finite non-negative number"
            )
        self.european_margin = float(european_margin)

    def forward(self, physical_features: torch.Tensor) -> torch.Tensor:
        self._check(physical_features)
        return price_with_margin(self.model, physical_features, self.european_margin)


class RawNetworkModel(_FrozenView):
    """The **unprojected** learned function, as a pricer.

    ``V = A * u_dir``: no floor, no margin, no projection. Section B.3 asks what
    fraction of states the learned network itself reproduces within tolerance,
    and section F asks whether its derivatives are faithful; both need the raw
    function driven through the same evaluation code as the deployed one.

    Its output carries **no lower bound**, so it may be negative and the bound
    diagnostics will report violations against it. That is the measurement, not
    a defect: the floor is exactly what those violations are about.
    """

    def forward(self, physical_features: torch.Tensor) -> torch.Tensor:
        self._check(physical_features)
        return raw_price(self.model, physical_features)


# ---------------------------------------------------------------------------
# Provenance shared by every diagnostic artifact
# ---------------------------------------------------------------------------

#: The declaration files whose digests every diagnostic artifact pins by name.
#: They are already inside :func:`..attempts.source_digests`, but a reader
#: checking "was this tolerance declared before this measurement?" should not
#: have to know which of forty digests to look at.
DECLARATION_FILES: Final = (
    "python/src/differentiable_pricing/ml/american_dev/tolerance.py",
    "python/src/differentiable_pricing/ml/american_dev/heldout.py",
    "python/src/differentiable_pricing/ml/american_dev/eligibility.py",
    "python/src/differentiable_pricing/ml/american_dev/frozen.py",
    "configs/american_neural_pilot_acceptance_v1.toml",
)


def declaration_state(project_root: Path) -> dict[str, Any]:
    """The commit and declaration digests, **without** requiring a clean tree.

    Used for *comparison*: checking that an existing artifact was written under
    the current protocol does not itself need the worktree to be clean, and
    demanding it would make the check unusable in the very situation it exists
    to detect. Writing an artifact is different, and
    :func:`diagnostic_provenance` is strict.
    """
    project_root = Path(project_root)
    digests = source_digests(project_root)
    return {
        "protocol_commit": _git(project_root, "rev-parse", "HEAD"),
        "declaration_digests": {
            name: digests[name] for name in DECLARATION_FILES if name in digests
        },
        "source_digests": digests,
    }


def diagnostic_provenance(project_root: Path) -> dict[str, Any]:
    """The provenance block every post-commit diagnostic artifact carries.

    Predeclaration has to be demonstrable from an artifact rather than asserted
    afterwards in prose, so every artifact records the commit it ran under and
    the digests of the declarations that bound it. A reader can then check that
    the tolerance file's digest matches the one in the protocol commit, without
    taking anyone's word for when it was written.

    **This requires a clean tracked worktree**, through
    :func:`..attempts.repository_identity`. That is deliberate and is the
    mechanism by which no decision-bearing diagnostic can be produced before the
    protocol is committed: a measurement whose source state cannot be named is
    not evidence of a predeclaration.
    """
    project_root = Path(project_root)
    state = declaration_state(project_root)
    repository = repository_identity(project_root)
    return {
        "protocol_commit": repository["commit"],
        "repository": repository,
        "declaration_digests": state["declaration_digests"],
        "source_digests": state["source_digests"],
    }


# ---------------------------------------------------------------------------
# The freeze report
# ---------------------------------------------------------------------------


def _ledger_facts(project_root: Path, attempt_id: str) -> dict[str, Any]:
    """The run ledger's own record, read for cross-reference only."""
    path = Path(project_root) / LEDGER_DIRECTORY / attempt_id / LEDGER_NAME
    if not path.is_file():
        return {"present": False}
    ledger = json.loads(path.read_text(encoding="utf-8"))
    return {
        "present": True,
        "path": f"{LEDGER_DIRECTORY}/{attempt_id}/{LEDGER_NAME}",
        "status": ledger.get("status"),
        "commit": ledger.get("repository", {}).get("commit"),
        "tracked_worktree_clean": ledger.get("repository", {}).get("tracked_worktree_clean"),
        "recorded_report_sha256": ledger.get("report", {}).get("sha256"),
        "recorded_summary_sha256": ledger.get("summary", {}).get("sha256"),
        "records_a_checkpoint_digest": False,
    }


def freeze_checkpoints(
    project_root: Path,
    output: str = DEFAULT_OUTPUT,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Record the immutable identity of every frozen checkpoint.

    Opens no dataset partition, prices nothing and trains nothing. It loads each
    model only to verify that the checkpoint really does rebuild into the
    architecture its immutable configuration declares.
    """
    project_root = Path(project_root)
    output_path = _guarded_artifact_path(project_root, output, where="freeze report path")
    if output_path.exists() and not overwrite:
        raise FrozenCheckpointError(
            f"'{output}' already exists; pass overwrite to replace this development record"
        )
    checkpoints: dict[str, Any] = {}
    for attempt_id in FROZEN_ATTEMPTS:
        _, provenance = load_frozen_model(project_root, attempt_id)
        provenance["run_ledger"] = _ledger_facts(project_root, attempt_id)
        checkpoints[attempt_id] = provenance
    report = {
        "schema_version": FROZEN_SCHEMA,
        "task": "task-9h-american-pricer-development",
        "status": {
            "phase": "diagnostic",
            "trains_a_model": False,
            "modifies_a_weight": False,
            "modifies_a_configuration": False,
            "opens_a_dataset_partition": False,
            "consumes_a_neural_attempt": False,
            "selection_bias": (
                "both checkpoints were selected against validation across a nine-attempt "
                "search; neither is a project result"
            ),
        },
        "deployment_function": DEPLOYMENT_FUNCTION,
        "projection_weight": PROJECTION_WEIGHT_DEFINITION,
        "floor_leg_weight": FLOOR_LEG_WEIGHT_DEFINITION,
        "checkpoints": checkpoints,
        "provenance": diagnostic_provenance(project_root),
    }
    write_json_atomic(output_path, report)
    return report


def compact_frozen(report: Mapping[str, Any]) -> dict[str, Any]:
    """The small, agent-readable view of the freeze."""
    return {
        "schema_version": report.get("schema_version"),
        "checkpoints": {
            attempt_id: {
                "label": entry["label"],
                "checkpoint_sha256": entry["checkpoint_sha256"][:16],
                "config_sha256": entry["config_sha256"][:16],
                "git_commit": entry["git_commit"],
                "parameters": entry["parameters"],
                "best_epoch": entry["best_epoch"],
                "head": entry["head"],
                "tau": entry["head_temperature"],
                "delta": entry["head_european_margin"],
                "initialization_seed": (entry.get("seeds") or {}).get("initialization"),
                "shuffle_seed": (entry.get("seeds") or {}).get("shuffle"),
            }
            for attempt_id, entry in report["checkpoints"].items()
        },
    }
