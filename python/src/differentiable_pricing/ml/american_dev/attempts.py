"""Task 9H attempt identity, immutability, partition guard and attempt log.

Deliberately free of any PyTorch import: the offline attempt-log tool validates
every tracked attempt configuration in the lightweight environment, where
neither PyTorch nor the compiled pricing extensions are installed. This module
also owns the small name lists the model modules validate against, so the two
cannot drift apart.

Four rules are enforced mechanically rather than by good intentions:

* **only ``train`` and ``validation`` are reachable.** Any split name or
  resolved path that looks like a final or held-out partition fails closed,
  before anything is opened, hashed, stat-ed, imported or counted. The token
  list below is the single definition used by both the runtime guard and the
  offline static scan, so the two cannot disagree about what is forbidden.
* **an attempt runs only from committed source.** The configuration it names,
  and every source file whose digest it records, must be tracked at ``HEAD``
  and byte-identical to the ``HEAD`` blob. Ignored artifacts and runs are
  unaffected: the untracked workspace is not required to be empty.
* **an attempt is written once.** Its output directory must not already exist,
  its ledger must not already exist, and the append-only attempt log refuses a
  duplicate attempt ID.
* **a used configuration is immutable.** Each logged attempt records the SHA-256
  of the configuration file it ran, and the offline checker re-verifies that the
  tracked file still hashes to it.

Every declared configuration field is validated here against behavior that is
actually implemented and dispatched, and an unsupported or unknown field is
refused, so a recorded declaration is never one that execution silently ignored.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, Final

import numpy as np

ATTEMPT_SCHEMA: Final = "american-dev-attempt/1"
ATTEMPT_LOG_SCHEMA: Final = "american-dev-attempt-log/1"
ATTEMPT_REPORT_SCHEMA: Final = "american-dev-attempt-report/1"
ATTEMPT_LOG_PATH: Final = "docs/attempts/task-9h-attempt-log.jsonl"

ALLOWED_SPLITS: Final = ("train", "validation")
#: Substrings that may never appear in a Task 9H split name or resolved path.
#: Checked on split and path fields only, so an ordinary word elsewhere in a
#: configuration is unaffected.
#:
#: This is the **single** definition. The runtime guards below match a token as
#: a substring of a lowered split name or path; the offline static scan in
#: ``scripts/american_dev_attempts.py`` imports this same tuple and matches a
#: token against a path component's stem, which is what lets prose state the
#: prohibition while a path may never name it. Both spellings of a partition are
#: listed so neither matching rule is weaker than the other.
FORBIDDEN_PARTITION_TOKENS: Final = (
    "interpolation_test",
    "boundary_test",
    "extrapolation",
    "extrapolation_test",
    "ood_test",
    "scenario_test",
    "holdout",
    "final",
    "final_test",
)

#: Network families an attempt configuration may name.
ARCHITECTURES: Final = ("smooth_mlp", "smooth_residual")
#: Output heads, and what each one reconstructs.
#:
#: ``smooth_lower_floor`` and ``smooth_lower_floor_raw_loss`` produce the
#: **same output**, bit for bit: the same floor, the same temperature, the same
#: projection. They differ only in which prediction the training loss is
#: computed against -- see :data:`RAW_LOSS_HEADS`.
#:
#: ``smooth_lower_floor_margin_raw_loss`` is ``smooth_lower_floor_raw_loss``
#: with :data:`EUROPEAN_FLOOR_MARGIN` added to the analytic European leg before
#: the smooth maximum with intrinsic value. Same temperature, same projection,
#: same training loss; one additive constant, on one leg.
HEADS: Final = (
    "direct",
    "premium_over_european",
    "smooth_lower_floor",
    "smooth_lower_floor_raw_loss",
    "smooth_lower_floor_margin_raw_loss",
)

#: Heads whose training loss is computed against the **pre-projection** direct
#: normalized price rather than against the deployed output.
#:
#: ``scratch_residual_smooth_floor_v1`` (E2) trained its loss through the
#: projection and stopped learning at epoch 1. The projection's derivative is
#: ``sigmoid((direct - floor) / tau)``, which at ``tau = 1e-4`` is of order
#: ``1e-44`` once the prediction sits a hundredth of a discounted spot below the
#: floor -- where a scratch model starts. The gradient reaching the network is
#: multiplied by that factor, so the loss cannot pull the prediction back up and
#: the model is pinned to the floor.
#:
#: A head listed here keeps the floor at evaluation and deployment and computes
#: the loss on the value that precedes it, so the latent network solves exactly
#: the direct-price problem the residual architecture already solved. Evaluation,
#: checkpoint selection, bound diagnostics and shape diagnostics are unaffected:
#: all of them read the projected output.
RAW_LOSS_HEADS: Final = (
    "smooth_lower_floor_raw_loss",
    "smooth_lower_floor_margin_raw_loss",
)

#: The **predeclared** normalized temperature of the ``smooth_lower_floor``
#: head, fixed here rather than in a configuration file.
#:
#: It is a code constant on purpose. Every attempt configuration must declare
#: exactly the same top-level keys, so a new per-attempt temperature field would
#: have to be added to the six immutable configurations that already ran — which
#: is precisely what "immutable after use" forbids. Pinning it here keeps it in
#: ``source_digests``, so the attempt report records which temperature ran.
#:
#: ``1e-4`` is stated in **normalized** units, the units of
#: ``u = V / (S*exp(-q*T))``, the same units the acceptance criterion is stated
#: in. :func:`assert_temperature_consistent` re-derives that this is the right
#: order of magnitude from the acceptance file rather than asserting it in prose:
#: the temperature must sit strictly above the material violation tolerance
#: (``1e-6``, so the smoothing is resolvable at all) and strictly below the
#: normalized RMSE limit (``3e-3``, so the smoothing bias cannot consume the
#: accuracy budget it is supposed to preserve).
SMOOTH_FLOOR_TEMPERATURE: Final = 1.0e-4

#: Heads that carry a temperature, and the one each carries. A head absent from
#: this mapping has none, and no temperature is recorded for it.
HEAD_TEMPERATURES: Final = {
    "smooth_lower_floor": SMOOTH_FLOOR_TEMPERATURE,
    "smooth_lower_floor_raw_loss": SMOOTH_FLOOR_TEMPERATURE,
    "smooth_lower_floor_margin_raw_loss": SMOOTH_FLOOR_TEMPERATURE,
}

#: The **predeclared** additive margin, in normalized units, applied to the
#: analytic European leg of the floor before the smooth maximum with intrinsic
#: value is taken. A code constant for the same reason
#: :data:`SMOOTH_FLOOR_TEMPERATURE` is one: every attempt configuration declares
#: exactly the same top-level keys, so a per-attempt field would have to be added
#: to the configurations that already ran, which "immutable after use" forbids.
#: Pinning it here keeps it in ``source_digests``, so the attempt report records
#: which margin ran.
#:
#: **Why the floor needs one.** The floor enforces the analytic Black-Scholes
#: European value; the ``european_comparator_lower_bound`` diagnostic compares
#: against the dataset's stored CRR European leg. The two differ by the lattice's
#: discretization error, so a prediction resting on the analytic floor is counted
#: as violating the stored comparator wherever that difference exceeds the
#: material tolerance. A zero-margin analytic floor therefore cannot satisfy that
#: gate, whatever the model does.
#:
#: **Its value is derived, not chosen.** The label-free characterization of
#: ``(E_CRR - E_BS) / A`` over the declared domain
#: (``american_dev.domain``) measured a supremum of
#: ``4.192769575172157e-05``. Its rule, predeclared in code before that run, is
#: ``delta = max(1e-4, ceil_to_1e-5(2 * supremum))``, which gives
#: ``ceil_to_1e-5(8.385539150344314e-05) = 9e-5`` and therefore
#: ``delta = 1e-4``: the rule's declared minimum binds, at a realized safety
#: factor of ``2.385`` over the measured supremum. The derivation used no
#: partition row and no partition-derived sampling location.
#:
#: :func:`assert_margin_consistent` re-derives from the digest-pinned acceptance
#: file that the value sits strictly above the material violation tolerance
#: (``1e-6``, below which the margin would be unresolvable) and strictly below
#: the normalized RMSE limit (``3e-3``, at or above which the margin's own bias
#: would consume the accuracy budget).
EUROPEAN_FLOOR_MARGIN: Final = 1.0e-4

#: Heads that add a margin to the European leg of their floor, and the margin
#: each adds. A head absent from this mapping adds none: its floor is the
#: zero-margin analytic value, and ``0.0`` is what the model applies.
HEAD_EUROPEAN_MARGINS: Final = {
    "smooth_lower_floor_margin_raw_loss": EUROPEAN_FLOOR_MARGIN,
}
#: Deterministic conditioning features, computed from the physical inputs.
CONDITIONING_FEATURES: Final = ("european_price_ratio", "intrinsic_ratio", "european_gap")

#: The representation, target and reconstruction Task 9H shares with Task 9G.
#: Declared here rather than in ``representation.py`` so the PyTorch-free
#: validator can reject a configuration that declares a different one, and so
#: the two modules cannot drift apart.
REPRESENTATION: Final = "american_forward_carry_v1"
NORMALIZED_TARGET: Final = "u = V / (S * exp(-q * T))"
PHYSICAL_RECONSTRUCTION: Final = "V = S * exp(-q * T) * u"

#: Declared values that correspond to behavior this workbench implements. A
#: configuration naming anything else is refused rather than recorded and
#: silently ignored.
OPTIMIZERS: Final = ("adamw",)
SCHEDULES: Final = ("cosine_annealing",)
CHECKPOINT_METRICS: Final = ("standardized_target_mse",)
CHECKPOINT_RULES: Final = ("minimum metric; earliest epoch wins exact ties",)
SHUFFLE_RULES: Final = ("torch.randperm from the shuffle seed once per epoch",)
ROW_SELECTION_RULES: Final = ("lowest SHA-256(salt + NUL + sample_id), sample_id tie-break",)
SEED_DERIVATIONS: Final = ("first four bytes of SHA-256(label), unsigned big-endian",)

#: The Task 9G acceptance configuration and section Task 9H reuses verbatim as
#: its fixed "works" criterion. Both are canonical: an attempt may not point at
#: another file or another section, so the criterion cannot be quietly loosened
#: after an attempt fails.
ACCEPTANCE_CONFIG_PATH: Final = "configs/american_neural_pilot_acceptance_v1.toml"
CRITERION_SECTION: Final = "validation_final_entry"
#: The locked Task 9G protocol, read only for the dataset identities Task 9H is
#: allowed to know.
PROTOCOL_CONFIG_PATH: Final = "configs/american_neural_pilot_protocol_v1.toml"

OUTCOMES: Final = (
    "criterion_met",
    "criterion_not_met",
    "abandoned",
    "infrastructure_failure",
)
NEXT_ACTIONS: Final = ("continue", "stop", "change-direction")

#: Attempts that completed but whose ``attempt-report.json`` no longer exists in
#: a recordable form, declared here so the append-only log stays checkable
#: without being rewritten.
#:
#: **Why this exists.** ``record`` is the only writer of the attempt log and it
#: requires a valid attempt report. ``scratch_residual_smooth_floor_v1`` (E2)
#: ran to completion -- its ledger says ``status="complete"`` and its compact
#: summary survives -- but the report file it wrote was afterwards truncated to
#: zero bytes, so it hashes to neither the digest the ledger recorded nor
#: anything ``record`` will accept. It therefore can never be recorded from its
#: own evidence, and rerunning it is forbidden: the attempt ID is spent.
#:
#: Its child ``scratch_residual_smooth_floor_raw_loss_v1`` (E2b) was recorded
#: naming it as parent, so :func:`validate_attempt_log` failed on a parent that
#: is not an earlier entry -- which made the offline check, ``scripts/check.sh``
#: and CI all fail at that commit. Appending E2 now cannot fix it, because an
#: append lands *after* E2b.
#:
#: **What this declaration does and does not do.** It makes the parent reference
#: resolvable and records exactly what was lost. It does **not** reconstruct the
#: attempt, and nothing here may be cited as that attempt's result: the compact
#: summary is the only surviving evidence and it is not an attempt report. The
#: log itself is untouched -- no entry is rewritten, reordered or removed.
#:
#: A declaration is verified only against **tracked** state: the configuration
#: must still exist and hash to the recorded digest, and the ID must genuinely
#: be absent from the log. The artifact digests below sit under the ignored
#: ``artifacts/`` and ``runs/`` trees, so they are recorded as declaration facts
#: and are not re-verified -- a fresh clone does not have those files at all.
UNRECORDABLE_ATTEMPTS: Final = {
    "scratch_residual_smooth_floor_v1": {
        "parent_attempt": "scratch_residual_architecture_v1",
        "config_path": "configs/american_dev_attempt_scratch_residual_smooth_floor_v1.toml",
        "config_sha256": (
            "f22e62aff228180590ff3417a49a8b91502f8e293c1303d5aa0618e087055820"
        ),
        "ledger_commit": "88e8300951762bb161a5a889d7d92faabdeb3833",
        "ledger_status": "complete",
        "reason": (
            "the attempt report was truncated to zero bytes after the run completed, so it "
            "hashes to neither the digest the run ledger recorded nor anything `record` "
            "accepts; the attempt ID is spent and rerunning it is forbidden"
        ),
        "expected_report_sha256": (
            "0980b38e06338a4cee221984bc32f58b316d593464004a42bd09f8f1d13ec447"
        ),
        "observed_report_bytes": 0,
        "surviving_evidence": "artifacts/task-9h/scratch_residual_smooth_floor_v1/summary.json",
        "surviving_evidence_sha256": (
            "19210af4e02784fa833199ddd03e79b32f05aac09b9e264cca2e765cd2afdfae"
        ),
        "not_a_result": (
            "the compact summary is not an attempt report; nothing here may be cited as this "
            "attempt's recorded result"
        ),
    },
}

REQUIRED_LOG_FIELDS: Final = (
    "attempt_id",
    "parent_attempt",
    "hypothesis",
    "git_commit",
    "config_path",
    "config_sha256",
    "source_digests",
    "architecture",
    "features",
    "target_and_reconstruction",
    "seeds",
    "selected_rows",
    "optimizer_and_budget",
    "price_metrics",
    "bound_and_shape_diagnostics",
    "criterion",
    "outcome",
    "interpretation",
    "next_action",
)


class AttemptError(RuntimeError):
    """Raised on any fail-closed identity, lifecycle or partition condition."""


class FinalPartitionAccessError(AttemptError):
    """Raised when anything would reach a final or held-out partition.

    Task 9H exposes no final-evaluation command and no flag that adds one. A
    candidate that meets the development criterion needs a separately
    predeclared confirmation on a fresh final partition, as its own task.
    """


# ---------------------------------------------------------------------------
# Partition guard
# ---------------------------------------------------------------------------


def assert_split_allowed(split: str) -> str:
    lowered = str(split).lower()
    if any(token in lowered for token in FORBIDDEN_PARTITION_TOKENS):
        raise FinalPartitionAccessError(
            f"Task 9H may not touch partition {split!r}; only {list(ALLOWED_SPLITS)} are reachable"
        )
    if lowered not in ALLOWED_SPLITS:
        raise FinalPartitionAccessError(
            f"unknown Task 9H partition {split!r}; only {list(ALLOWED_SPLITS)} are reachable"
        )
    return lowered


def assert_path_allowed(path: Path | str, *, where: str) -> Path:
    text = str(path).lower().replace("\\", "/")
    for token in FORBIDDEN_PARTITION_TOKENS:
        if token in text:
            raise FinalPartitionAccessError(
                f"{where} resolves to {path!r}, which names a final or held-out partition"
            )
    return Path(path)


def assert_contained_relative_path(path: str, *, where: str) -> PurePosixPath:
    """A supplied name must stay inside the tree it is resolved against.

    Applied to every declared path and to every retained dataset manifest entry
    before it is opened, hashed, stat-ed or counted: an absolute name or one
    containing ``..`` could otherwise reach outside the two reachable
    partitions without ever spelling a forbidden token.
    """
    text = str(path).replace("\\", "/")
    if not text.strip():
        raise AttemptError(f"{where} must be a non-empty relative path")
    candidate = PurePosixPath(text)
    if candidate.is_absolute() or any(part == ".." for part in candidate.parts):
        raise AttemptError(f"{where} must be a relative path inside the repository, got {path!r}")
    return candidate


# ---------------------------------------------------------------------------
# Deterministic, paired row and seed selection
# ---------------------------------------------------------------------------


def derive_seed(label: str) -> int:
    """First four bytes of SHA-256(label), unsigned big-endian.

    The convention Task 9G used, so a paired comparison can reuse a label and
    get a reproducible seed without a magic number in a configuration.
    """
    if not label:
        raise AttemptError("a seed label must be a non-empty string")
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "big")


def select_rows_by_hash(sample_ids: Sequence[Any], budget: int, salt: str) -> np.ndarray:
    """Lowest SHA-256(salt + NUL + sample_id) ranks, sample_id as tie-break.

    Order-independent and count-independent: two attempts sharing a salt and a
    budget select exactly the same rows, which is what makes the candidates
    comparable to each other and to the Task 9G control.
    """
    ids = list(sample_ids)
    if budget <= 0 or budget > len(ids) or not salt:
        raise AttemptError("row budget must be in [1, rows] and salt must be non-empty")
    ranked = sorted(
        (hashlib.sha256(f"{salt}\0{sample_id}".encode()).digest(), str(sample_id), index)
        for index, sample_id in enumerate(ids)
    )
    return np.asarray([entry[2] for entry in ranked[:budget]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Digests and repository identity
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_digests(project_root: Path) -> dict[str, str]:
    """Digest every tracked file that can change a Task 9H result.

    The Task 9H package and scripts, plus the two Task 9G configurations the
    verdict and the dataset identity are read from: a change to any of them
    changes what an attempt means, so all of them are pinned in the report and
    all of them are verified against ``HEAD`` before an attempt starts.
    """
    package = Path("python/src/differentiable_pricing/ml/american_dev")
    scripts = (
        Path("scripts/run_american_dev_attempt.py"),
        Path("scripts/american_dev_attempts.py"),
    )
    digests: dict[str, str] = {}
    for path in sorted((project_root / package).glob("*.py")):
        digests[str(path.relative_to(project_root))] = sha256_file(path)
    for relative in (*scripts, Path(ACCEPTANCE_CONFIG_PATH), Path(PROTOCOL_CONFIG_PATH)):
        candidate = project_root / relative
        if candidate.is_file():
            digests[str(relative)] = sha256_file(candidate)
    return digests


def _git(project_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AttemptError(f"git {' '.join(arguments)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def head_blob(project_root: Path, relative: str) -> str | None:
    """The blob name of ``relative`` at ``HEAD``, or ``None`` when untracked."""
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"HEAD:{str(relative).replace(chr(92), '/')}"],
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
    )
    name = result.stdout.strip()
    return name if result.returncode == 0 and name else None


def worktree_blob(project_root: Path, relative: str) -> str:
    """The blob name the working-tree file would hash to.

    ``--path`` makes the hash filter-aware, so a checkout with ``core.autocrlf``
    or a ``.gitattributes`` text rule compares like with like instead of
    reporting a spurious difference from ``HEAD``.
    """
    text = str(relative).replace("\\", "/")
    return _git(project_root, "hash-object", "--path", text, "--", text)


def verify_committed_source(project_root: Path, relatives: Iterable[str]) -> dict[str, str]:
    """Require every named file to be tracked at ``HEAD`` and unmodified.

    This is the ``git status`` check's missing half. A clean *tracked* worktree
    still allows a brand-new untracked attempt configuration or module to be
    picked up, in which case the recorded commit would not describe what
    actually ran. Ignored artifacts and runs are untouched by this: only the
    named files are required to be committed, never the whole workspace.
    """
    identities: dict[str, str] = {}
    for relative in sorted({str(entry).replace("\\", "/") for entry in relatives}):
        assert_contained_relative_path(relative, where=f"source file '{relative}'")
        committed = head_blob(project_root, relative)
        if committed is None:
            raise AttemptError(
                f"'{relative}' is not tracked at HEAD; a Task 9H attempt or analysis runs only "
                "from committed source — commit the exact source state first"
            )
        working = worktree_blob(project_root, relative)
        if working != committed:
            raise AttemptError(
                f"'{relative}' differs from its HEAD blob ({working[:12]}… vs "
                f"{committed[:12]}…); a Task 9H attempt or analysis runs only from committed "
                "source"
            )
        identities[relative] = committed
    return identities


def repository_identity(project_root: Path) -> dict[str, Any]:
    """Require a clean tracked worktree and record the exact commit.

    A real attempt runs from a committed source state, so the attempt log's
    commit is enough to reconstruct exactly what ran. Untracked and ignored
    files are deliberately not consulted here — checkpoints, reports and run
    ledgers live beneath ignored trees — which is why
    :func:`verify_committed_source` separately pins the files that matter.
    """
    dirty = _git(project_root, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise AttemptError(
            "a Task 9H attempt requires a clean tracked worktree; commit the exact "
            f"source state first (modified: {dirty.splitlines()[0].strip()!r} and possibly more)"
        )
    return {
        "commit": _git(project_root, "rev-parse", "HEAD"),
        "branch": _git(project_root, "branch", "--show-current"),
        "tracked_worktree_clean": True,
    }


# ---------------------------------------------------------------------------
# Attempt configuration
# ---------------------------------------------------------------------------


def load_toml(path: Path) -> dict[str, Any]:
    try:
        with Path(path).open("rb") as stream:
            payload = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise AttemptError(f"cannot load '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise AttemptError(f"'{path}' must contain a table")
    return payload


def _table(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = config.get(name)
    if not isinstance(section, Mapping):
        raise AttemptError(f"attempt config needs a [{name}] table")
    return section


def _exact_keys(section: Mapping[str, Any], name: str, allowed: Sequence[str]) -> None:
    """Every declared key must be one execution reads; every one must be there.

    An unknown key would be recorded in the attempt log while execution ignored
    it, which is exactly the failure this refuses: the log has to describe what
    ran.
    """
    keys = set(section)
    missing = sorted(set(allowed) - keys)
    if missing:
        raise AttemptError(f"{name} is missing key(s): {missing}")
    unknown = sorted(keys - set(allowed))
    if unknown:
        raise AttemptError(
            f"{name} declares unsupported key(s): {unknown}; Task 9H never records a "
            "declaration execution would silently ignore"
        )


def _one_of(section: Mapping[str, Any], key: str, allowed: Sequence[str], name: str) -> None:
    value = section.get(key)
    if not isinstance(value, str) or value not in allowed:
        raise AttemptError(f"{name}.{key} must be one of {list(allowed)}, got {value!r}")


def _positive_int(section: Mapping[str, Any], key: str, name: str) -> int:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AttemptError(f"{name}.{key} must be a positive integer")
    return int(value)


def _number(section: Mapping[str, Any], key: str, name: str, *, minimum: float) -> float:
    value = section.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise AttemptError(f"{name}.{key} must be a number")
    if float(value) < minimum:
        raise AttemptError(f"{name}.{key} must be at least {minimum}")
    return float(value)


_TOP_LEVEL_KEYS: Final = (
    "schema_version",
    "attempt_id",
    "parent_attempt",
    "hypothesis",
    "objective",
    "representation",
    "target",
    "physical_reconstruction",
    "head",
    "conditioning_features",
    "dtype",
    "device",
    "architecture",
    "seeds",
    "row_selection",
    "training",
    "optimizer",
    "checkpoint",
    "evaluation",
    "paths",
)
_PATH_KEYS: Final = (
    "dataset",
    "dataset_manifest",
    "acceptance_config",
    "output_directory",
    "ledger",
)


def _validate_architecture(config: Mapping[str, Any]) -> None:
    architecture = _table(config, "architecture")
    _one_of(architecture, "name", ARCHITECTURES, "architecture")
    if architecture["name"] == "smooth_mlp":
        _exact_keys(architecture, "[architecture]", ("name", "activation", "hidden_dimensions"))
        widths = architecture["hidden_dimensions"]
        if not isinstance(widths, list) or not widths:
            raise AttemptError("architecture.hidden_dimensions must be a non-empty list")
        for width in widths:
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise AttemptError("architecture.hidden_dimensions entries must be positive")
    else:
        _exact_keys(architecture, "[architecture]", ("name", "activation", "width", "blocks"))
        _positive_int(architecture, "width", "architecture")
        _positive_int(architecture, "blocks", "architecture")
    if not str(architecture.get("activation", "")):
        raise AttemptError("architecture.activation must be declared")


def validate_attempt_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable, price-only attempt configuration, fail-closed.

    Every declared field is checked against behavior that is implemented and
    dispatched, and unknown fields are refused, so an attempt never records a
    declaration that execution ignored.
    """
    _exact_keys(config, "attempt config", _TOP_LEVEL_KEYS)
    if config.get("schema_version") != ATTEMPT_SCHEMA:
        raise AttemptError(f"attempt config schema must be {ATTEMPT_SCHEMA!r}")
    attempt_id = config.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id:
        raise AttemptError("attempt_id must be a non-empty string without a path separator")
    if not isinstance(config.get("parent_attempt"), str):
        raise AttemptError("parent_attempt must be a string, empty when the attempt has no parent")
    if not str(config.get("hypothesis", "")).strip():
        raise AttemptError("every attempt must state the hypothesis it tests")
    if config.get("head") not in HEADS:
        raise AttemptError(f"head must be one of {list(HEADS)}")
    if config.get("dtype") != "float64" or config.get("device") != "cpu":
        raise AttemptError("Task 9H attempts are float64 and CPU")
    if config.get("objective") != "price_only":
        raise AttemptError(
            "every Task 9H attempt is price_only; Greeks are a separate follow-up stage"
        )
    _one_of(config, "representation", (REPRESENTATION,), "attempt config")
    _one_of(config, "target", (NORMALIZED_TARGET,), "attempt config")
    _one_of(config, "physical_reconstruction", (PHYSICAL_RECONSTRUCTION,), "attempt config")

    features = config.get("conditioning_features")
    if not isinstance(features, list):
        raise AttemptError("conditioning_features must be a list")
    conditioning = tuple(features)
    for name in conditioning:
        if name not in CONDITIONING_FEATURES:
            raise AttemptError(f"unknown conditioning feature {name!r}")
    if len(set(conditioning)) != len(conditioning):
        raise AttemptError("conditioning features must be distinct")

    _validate_architecture(config)

    seeds = _table(config, "seeds")
    _exact_keys(seeds, "[seeds]", ("initialization_label", "shuffle_label", "derivation"))
    for key in ("initialization_label", "shuffle_label"):
        if not str(seeds.get(key, "")):
            raise AttemptError(f"seeds.{key} must be a non-empty string")
    _one_of(seeds, "derivation", SEED_DERIVATIONS, "seeds")

    rows = _table(config, "row_selection")
    _exact_keys(rows, "[row_selection]", ("partition", "row_budget", "salt", "rule"))
    assert_split_allowed(str(rows.get("partition", "")))
    _positive_int(rows, "row_budget", "row_selection")
    if not str(rows.get("salt", "")):
        raise AttemptError("row_selection.salt must be a non-empty string")
    _one_of(rows, "rule", ROW_SELECTION_RULES, "row_selection")

    training = _table(config, "training")
    _exact_keys(
        training,
        "[training]",
        ("batch_size", "epochs", "num_threads", "deterministic_algorithms", "shuffle"),
    )
    for key in ("batch_size", "epochs", "num_threads"):
        _positive_int(training, key, "training")
    if not isinstance(training.get("deterministic_algorithms"), bool):
        raise AttemptError("training.deterministic_algorithms must be a boolean")
    _one_of(training, "shuffle", SHUFFLE_RULES, "training")

    checkpoint = _table(config, "checkpoint")
    _exact_keys(checkpoint, "[checkpoint]", ("selection_partition", "metric", "rule"))
    assert_split_allowed(str(checkpoint.get("selection_partition", "")))
    if checkpoint["selection_partition"] != "validation":
        raise AttemptError("checkpoints are selected on validation, and only on validation")
    _one_of(checkpoint, "metric", CHECKPOINT_METRICS, "checkpoint")
    _one_of(checkpoint, "rule", CHECKPOINT_RULES, "checkpoint")

    optimizer = _table(config, "optimizer")
    _exact_keys(
        optimizer,
        "[optimizer]",
        (
            "name",
            "learning_rate",
            "weight_decay",
            "beta1",
            "beta2",
            "epsilon",
            "schedule",
            "minimum_learning_rate",
            "schedule_period_epochs",
        ),
    )
    _one_of(optimizer, "name", OPTIMIZERS, "optimizer")
    _one_of(optimizer, "schedule", SCHEDULES, "optimizer")
    if not _number(optimizer, "learning_rate", "optimizer", minimum=0.0) > 0.0:
        raise AttemptError("optimizer.learning_rate must be positive")
    _number(optimizer, "weight_decay", "optimizer", minimum=0.0)
    _number(optimizer, "minimum_learning_rate", "optimizer", minimum=0.0)
    for key in ("beta1", "beta2"):
        value = _number(optimizer, key, "optimizer", minimum=0.0)
        if not value < 1.0:
            raise AttemptError(f"optimizer.{key} must lie in [0, 1)")
    if not _number(optimizer, "epsilon", "optimizer", minimum=0.0) > 0.0:
        raise AttemptError("optimizer.epsilon must be positive")
    _positive_int(optimizer, "schedule_period_epochs", "optimizer")

    evaluation = _table(config, "evaluation")
    _exact_keys(evaluation, "[evaluation]", ("batch_size",))
    _positive_int(evaluation, "batch_size", "evaluation")

    paths = _table(config, "paths")
    _exact_keys(paths, "[paths]", _PATH_KEYS)
    for key in _PATH_KEYS:
        value = str(paths.get(key, ""))
        assert_path_allowed(value, where=f"paths.{key}")
        assert_contained_relative_path(value, where=f"paths.{key}")
    if paths["acceptance_config"] != ACCEPTANCE_CONFIG_PATH:
        raise AttemptError(
            f"paths.acceptance_config must be {ACCEPTANCE_CONFIG_PATH!r}; the Task 9H criterion "
            "is Task 9G's, read from the digest-pinned acceptance configuration"
        )
    if not str(paths["output_directory"]).startswith("artifacts/"):
        raise AttemptError("attempt outputs must live beneath the ignored artifacts/ tree")
    if not str(paths["ledger"]).startswith("runs/"):
        raise AttemptError("the attempt ledger must live beneath the ignored runs/ tree")
    return dict(config)


def validate_acceptance_config(acceptance: Mapping[str, Any]) -> dict[str, Any]:
    """Check the reused Task 9G criterion is present and complete.

    Checked before an attempt directory is created, so a missing or reshaped
    acceptance file fails before anything is reserved rather than after a
    training budget has been spent.
    """
    if acceptance.get("schema_version") != "american-neural-pilot-acceptance/1":
        raise AttemptError(
            "the acceptance configuration must be Task 9G's "
            "'american-neural-pilot-acceptance/1' schema"
        )
    section = acceptance.get(CRITERION_SECTION)
    if not isinstance(section, Mapping):
        raise AttemptError(f"the acceptance configuration needs a [{CRITERION_SECTION}] table")
    required = (
        "normalized_rmse_max",
        "normalized_p99_absolute_error_max",
        "normalized_maximum_absolute_error_max",
        "maximum_material_bound_violations",
        "maximum_material_shape_violations",
    )
    missing = sorted(key for key in required if key not in section)
    if missing:
        raise AttemptError(f"[{CRITERION_SECTION}] is missing threshold(s): {missing}")
    return dict(section)


def head_temperature(head: str) -> float | None:
    """The predeclared normalized temperature of ``head``, or ``None``."""
    if head not in HEADS:
        raise AttemptError(f"unknown head {head!r}")
    return HEAD_TEMPERATURES.get(head)


def head_european_margin(head: str) -> float:
    """The predeclared normalized European-leg margin of ``head``, or ``0.0``."""
    if head not in HEADS:
        raise AttemptError(f"unknown head {head!r}")
    return float(HEAD_EUROPEAN_MARGINS.get(head, 0.0))


def assert_margin_consistent(head: str, acceptance: Mapping[str, Any]) -> float:
    """Check the head's European-leg margin against the units of the criterion.

    The margin is normalized, in the units of ``u = V / (S*exp(-q*T))`` -- the
    units the acceptance criterion is also stated in. Two bounds, both read from
    the digest-pinned acceptance file rather than restated:

    * it must exceed the **material violation tolerance**. A margin at or below
      it lifts the floor by less than the diagnostics can resolve, which is
      indistinguishable from no margin at all.
    * it must fall below the **normalized RMSE limit**. Wherever the projection
      binds, the margin is added to the prediction, so a margin at or above the
      accuracy limit would spend the entire error budget it exists inside.

    A head that declares no margin returns ``0.0`` and is not checked, because
    zero is the absence of the transformation rather than a value of it.

    Raises rather than silently substituting another value: if the repository's
    units ever contradict the predeclared margin, that is a fact to report, not a
    number to quietly change.
    """
    margin = head_european_margin(head)
    if margin == 0.0:
        return margin
    diagnostics = acceptance.get("diagnostics")
    criterion = acceptance.get(CRITERION_SECTION)
    tolerance = diagnostics.get("material_normalized_tolerance") if isinstance(
        diagnostics, Mapping
    ) else None
    accuracy = criterion.get("normalized_rmse_max") if isinstance(criterion, Mapping) else None
    if not isinstance(tolerance, int | float) or not isinstance(accuracy, int | float):
        raise AttemptError(
            "the head European-leg margin is checked against the acceptance file's "
            f"[diagnostics].material_normalized_tolerance and [{CRITERION_SECTION}]."
            "normalized_rmse_max; both must be numbers"
        )
    if not float(tolerance) < margin < float(accuracy):
        raise AttemptError(
            f"head {head!r} declares normalized European-leg margin {margin:g}, which is not "
            f"strictly between the material tolerance {float(tolerance):g} and the normalized "
            f"RMSE limit {float(accuracy):g}; the repository's units contradict the predeclared "
            "value and the discrepancy is reported, never silently resolved"
        )
    return margin


def assert_temperature_consistent(head: str, acceptance: Mapping[str, Any]) -> float | None:
    """Check the head's temperature is consistent with the units of the criterion.

    The temperature is normalized, in the units of ``u = V / (S*exp(-q*T))`` --
    the units the acceptance criterion is also stated in. Two bounds, both read
    from the digest-pinned acceptance file rather than restated:

    * it must exceed the **material violation tolerance**. A temperature at or
      below it would smooth over a scale the diagnostics cannot resolve, which
      is indistinguishable from no smoothing at all.
    * it must fall below the **normalized RMSE limit**. The smoothing raises the
      output by at most a small multiple of the temperature near the floor, so a
      temperature at or above the accuracy limit would spend the entire error
      budget the transformation exists to preserve.

    Raises rather than silently substituting another value: if the repository's
    units ever contradict the predeclared temperature, that is a fact to report,
    not a number to quietly change.
    """
    temperature = head_temperature(head)
    if temperature is None:
        return None
    diagnostics = acceptance.get("diagnostics")
    criterion = acceptance.get(CRITERION_SECTION)
    tolerance = diagnostics.get("material_normalized_tolerance") if isinstance(
        diagnostics, Mapping
    ) else None
    accuracy = criterion.get("normalized_rmse_max") if isinstance(criterion, Mapping) else None
    if not isinstance(tolerance, int | float) or not isinstance(accuracy, int | float):
        raise AttemptError(
            "the head temperature is checked against the acceptance file's "
            f"[diagnostics].material_normalized_tolerance and [{CRITERION_SECTION}]."
            "normalized_rmse_max; both must be numbers"
        )
    if not float(tolerance) < temperature < float(accuracy):
        raise AttemptError(
            f"head {head!r} declares normalized temperature {temperature:g}, which is not "
            f"strictly between the material tolerance {float(tolerance):g} and the normalized "
            f"RMSE limit {float(accuracy):g}; the repository's units contradict the predeclared "
            "value and the discrepancy is reported, never silently resolved"
        )
    return temperature


def attempt_seeds(config: Mapping[str, Any]) -> dict[str, int]:
    """Derive every seed the attempt uses from its declared labels."""
    return {
        "initialization": derive_seed(str(config["seeds"]["initialization_label"])),
        "shuffle": derive_seed(str(config["seeds"]["shuffle_label"])),
    }


# ---------------------------------------------------------------------------
# Append-only attempt log
# ---------------------------------------------------------------------------


def read_attempt_log(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    if not Path(path).is_file():
        return entries
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AttemptError(f"attempt log line {number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise AttemptError(f"attempt log line {number} must be an object")
        entries.append(record)
    return entries


def attempt_log_entries(path: Path) -> list[dict[str, Any]]:
    """Return the attempt records, excluding the schema header line."""
    return [record for record in read_attempt_log(path) if record.get("record") != "header"]


def validate_attempt_log(path: Path) -> dict[str, Any]:
    """Check header, field completeness, ID uniqueness and parent references."""
    records = read_attempt_log(path)
    if not records or records[0].get("record") != "header":
        raise AttemptError("the attempt log must begin with its schema header record")
    header = records[0]
    if header.get("schema_version") != ATTEMPT_LOG_SCHEMA:
        raise AttemptError(f"attempt log schema must be {ATTEMPT_LOG_SCHEMA!r}")
    seen: set[str] = set()
    for record in records[1:]:
        if record.get("record") != "attempt":
            raise AttemptError("every non-header attempt log line must be an attempt record")
        missing = sorted(set(REQUIRED_LOG_FIELDS) - set(record))
        if missing:
            raise AttemptError(
                f"attempt {record.get('attempt_id')!r} is missing log field(s): {missing}"
            )
        if record["outcome"] not in OUTCOMES:
            raise AttemptError(
                f"attempt {record.get('attempt_id')!r} has outcome {record['outcome']!r}, "
                f"which is not one of {list(OUTCOMES)}"
            )
        attempt_id = str(record["attempt_id"])
        if attempt_id in seen:
            raise AttemptError(
                f"attempt {attempt_id!r} appears twice; an existing entry is never rewritten"
            )
        parent = str(record.get("parent_attempt") or "")
        # A parent resolves either to an earlier entry or to an attempt declared
        # in UNRECORDABLE_ATTEMPTS. The second case exists because `record`
        # needs a valid attempt report and one completed attempt's report was
        # destroyed after its run; see that mapping for what was lost. The log
        # is append-only, so an attempt whose record can never be written cannot
        # be slotted in ahead of the child that names it.
        if parent and parent not in seen and parent not in UNRECORDABLE_ATTEMPTS:
            raise AttemptError(
                f"attempt {attempt_id!r} names parent {parent!r}, which is neither an earlier "
                f"entry nor a declared unrecordable attempt"
            )
        seen.add(attempt_id)
    return {"header": header, "attempts": len(seen), "attempt_ids": sorted(seen)}


def assert_canonical_log_path(path: Path, project_root: Path) -> Path:
    """The recorded search lives in exactly one tracked file.

    A second log would let an attempt be recorded somewhere the offline checker
    never reads, which is indistinguishable from not recording it at all.
    """
    canonical = (Path(project_root) / ATTEMPT_LOG_PATH).resolve()
    if Path(path).resolve() != canonical:
        raise AttemptError(
            f"the attempt log must be '{ATTEMPT_LOG_PATH}', not '{path}'; the recorded search "
            "lives in exactly one append-only file"
        )
    return canonical


def assert_report_is_recordable(report: Mapping[str, Any]) -> None:
    """Refuse a report of an unknown schema or a non-canonical criterion."""
    if report.get("schema_version") != ATTEMPT_REPORT_SCHEMA:
        raise AttemptError(
            f"attempt report schema must be {ATTEMPT_REPORT_SCHEMA!r}, got "
            f"{report.get('schema_version')!r}"
        )
    criterion = report.get("criterion")
    if not isinstance(criterion, Mapping):
        raise AttemptError("an attempt report must carry its criterion")
    if criterion.get("source") != ACCEPTANCE_CONFIG_PATH:
        raise AttemptError(
            f"the criterion must come from {ACCEPTANCE_CONFIG_PATH!r}, got "
            f"{criterion.get('source')!r}"
        )
    if criterion.get("section") != CRITERION_SECTION:
        raise AttemptError(
            f"the criterion must be section {CRITERION_SECTION!r}, got "
            f"{criterion.get('section')!r}"
        )


def append_attempt(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    """Append exactly one attempt record. Never rewrites an existing entry."""
    path = Path(path)
    existing = validate_attempt_log(path) if path.is_file() else None
    if existing is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = json.dumps(
            {
                "record": "header",
                "schema_version": ATTEMPT_LOG_SCHEMA,
                "task": "task-9h-american-pricer-development",
                "append_only": True,
                "selection_bias": (
                    "every entry is a development measurement selected against validation; "
                    "none is a project result"
                ),
            },
            sort_keys=True,
        )
        path.write_text(header + "\n", encoding="utf-8")
        existing = validate_attempt_log(path)
    attempt_id = str(record.get("attempt_id", ""))
    if not attempt_id:
        raise AttemptError("an attempt record must carry an attempt_id")
    if attempt_id in existing["attempt_ids"]:
        raise AttemptError(
            f"attempt {attempt_id!r} is already recorded; an existing entry is never rewritten"
        )
    missing = sorted(set(REQUIRED_LOG_FIELDS) - set(record))
    if missing:
        raise AttemptError(f"attempt record is missing field(s): {missing}")
    if record["outcome"] not in OUTCOMES:
        raise AttemptError(f"outcome must be one of {list(OUTCOMES)}")
    payload = dict(record)
    payload["record"] = "attempt"
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
    return validate_attempt_log(path)


def check_configuration_immutability(project_root: Path, log_path: Path) -> list[str]:
    """Every logged attempt's configuration must still hash to its recorded digest."""
    failures: list[str] = []
    for record in attempt_log_entries(log_path):
        relative = record.get("config_path")
        expected = record.get("config_sha256")
        if not relative or not expected:
            failures.append(
                f"attempt {record.get('attempt_id')!r} does not pin its configuration digest"
            )
            continue
        path = project_root / str(relative)
        if not path.is_file():
            failures.append(
                f"attempt {record['attempt_id']!r} names configuration '{relative}', which no "
                "longer exists; a used configuration is never removed"
            )
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                f"attempt {record['attempt_id']!r} ran configuration '{relative}' at "
                f"{expected[:12]}…, which now hashes to {actual[:12]}…; an attempt "
                "configuration is immutable after use"
            )
    return failures


DECLARATION_FIELDS: Final = (
    "parent_attempt",
    "config_path",
    "config_sha256",
    "ledger_commit",
    "ledger_status",
    "reason",
    "surviving_evidence",
    "not_a_result",
)


def check_unrecordable_declarations(project_root: Path, log_path: Path) -> list[str]:
    """Reconcile every :data:`UNRECORDABLE_ATTEMPTS` entry against tracked state.

    A declaration is an admission that evidence was lost, so it has to stay
    honest in both directions. It is checked against **tracked** state only --
    the configuration file -- because the artifacts it names live under ignored
    trees that a fresh clone does not have.

    Three ways a declaration goes wrong:

    * the attempt is in the log after all, so the declaration is stale and the
      relaxed parent rule is being kept alive for nothing;
    * the configuration it names is gone or has changed, so the attempt can no
      longer be identified with what ran;
    * its own parent resolves to nothing, which would let a declaration hide a
      second gap behind the first.
    """
    failures: list[str] = []
    recorded = {str(record.get("attempt_id")) for record in attempt_log_entries(log_path)}
    for attempt_id, declaration in UNRECORDABLE_ATTEMPTS.items():
        missing = sorted(set(DECLARATION_FIELDS) - set(declaration))
        if missing:
            failures.append(
                f"unrecordable attempt {attempt_id!r} declaration is missing field(s): {missing}"
            )
            continue
        if attempt_id in recorded:
            failures.append(
                f"attempt {attempt_id!r} is declared unrecordable but is recorded in "
                f"'{ATTEMPT_LOG_PATH}'; remove the stale declaration"
            )
        relative = str(declaration["config_path"])
        config_path = project_root / relative
        if not config_path.is_file():
            failures.append(
                f"unrecordable attempt {attempt_id!r} names configuration '{relative}', which "
                "does not exist; a used configuration is never removed"
            )
        else:
            actual = sha256_file(config_path)
            if actual != str(declaration["config_sha256"]):
                failures.append(
                    f"unrecordable attempt {attempt_id!r} declares configuration '{relative}' at "
                    f"{str(declaration['config_sha256'])[:12]}…, which now hashes to "
                    f"{actual[:12]}…; a used configuration is immutable after use"
                )
        parent = str(declaration["parent_attempt"] or "")
        if parent and parent not in recorded and parent not in UNRECORDABLE_ATTEMPTS:
            failures.append(
                f"unrecordable attempt {attempt_id!r} names parent {parent!r}, which is neither "
                "recorded nor itself declared"
            )
    return failures


def check_recorded_criterion(project_root: Path, log_path: Path) -> list[str]:
    """Every logged attempt must cite the canonical criterion, by file and digest.

    The offline half of "the criterion cannot be quietly loosened after an
    attempt fails": a recorded attempt that judged itself against another file,
    another section, or another version of the acceptance configuration is a
    check failure, not a result.
    """
    failures: list[str] = []
    acceptance = project_root / ACCEPTANCE_CONFIG_PATH
    expected_digest = sha256_file(acceptance) if acceptance.is_file() else None
    if expected_digest is None:
        return [f"'{ACCEPTANCE_CONFIG_PATH}' is missing; it is the Task 9H criterion"]
    for record in attempt_log_entries(log_path):
        attempt_id = record.get("attempt_id")
        criterion = record.get("criterion")
        if not isinstance(criterion, Mapping):
            failures.append(f"attempt {attempt_id!r} does not record its criterion")
            continue
        if criterion.get("source") != ACCEPTANCE_CONFIG_PATH:
            failures.append(
                f"attempt {attempt_id!r} names criterion source {criterion.get('source')!r}; "
                f"every attempt reuses {ACCEPTANCE_CONFIG_PATH!r}"
            )
        if criterion.get("section") != CRITERION_SECTION:
            failures.append(
                f"attempt {attempt_id!r} names criterion section {criterion.get('section')!r}; "
                f"every attempt reuses {CRITERION_SECTION!r}"
            )
        if criterion.get("sha256") != expected_digest:
            failures.append(
                f"attempt {attempt_id!r} judged itself against acceptance digest "
                f"{str(criterion.get('sha256'))[:12]}…, but the tracked file hashes to "
                f"{expected_digest[:12]}…; the criterion is fixed, not revised"
            )
    return failures
