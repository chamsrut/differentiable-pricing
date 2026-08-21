"""Task 9H attempt identity, immutability, partition guard and attempt log.

Deliberately free of any PyTorch import: the offline attempt-log tool validates
every tracked attempt configuration in the lightweight environment, where
neither PyTorch nor the compiled pricing extensions are installed. This module
also owns the small name lists the model modules validate against, so the two
cannot drift apart.

Three rules are enforced mechanically rather than by good intentions:

* **only ``train`` and ``validation`` are reachable.** Any split name or
  resolved path that looks like a final or held-out partition fails closed,
  before anything is opened, hashed, stat-ed, imported or counted.
* **an attempt is written once.** Its output directory must not already exist,
  its ledger must not already exist, and the append-only attempt log refuses a
  duplicate attempt ID.
* **a used configuration is immutable.** Each logged attempt records the SHA-256
  of the configuration file it ran, and the offline checker re-verifies that the
  tracked file still hashes to it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tomllib
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np

ATTEMPT_SCHEMA: Final = "american-dev-attempt/1"
ATTEMPT_LOG_SCHEMA: Final = "american-dev-attempt-log/1"
ATTEMPT_LOG_PATH: Final = "docs/attempts/task-9h-attempt-log.jsonl"

ALLOWED_SPLITS: Final = ("train", "validation")
#: Substrings that may never appear in a Task 9H split name or resolved path.
#: Checked on split and path fields only, so an ordinary word elsewhere in a
#: configuration is unaffected.
FORBIDDEN_PARTITION_TOKENS: Final = (
    "interpolation_test",
    "boundary_test",
    "extrapolation",
    "ood_test",
    "scenario_test",
    "holdout",
    "final",
)

#: Network families an attempt configuration may name.
ARCHITECTURES: Final = ("smooth_mlp", "smooth_residual")
#: Output heads, and what each one reconstructs.
HEADS: Final = ("direct", "premium_over_european")
#: Deterministic conditioning features, computed from the physical inputs.
CONDITIONING_FEATURES: Final = ("european_price_ratio", "intrinsic_ratio", "european_gap")

OUTCOMES: Final = (
    "criterion_met",
    "criterion_not_met",
    "abandoned",
    "infrastructure_failure",
)
NEXT_ACTIONS: Final = ("continue", "stop", "change-direction")

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
    """Digest every tracked Task 9H source file that can change a result."""
    package = Path("python/src/differentiable_pricing/ml/american_dev")
    scripts = (
        Path("scripts/run_american_dev_attempt.py"),
        Path("scripts/american_dev_attempts.py"),
    )
    digests: dict[str, str] = {}
    for path in sorted((project_root / package).glob("*.py")):
        digests[str(path.relative_to(project_root))] = sha256_file(path)
    for relative in scripts:
        candidate = project_root / relative
        if candidate.is_file():
            digests[str(relative)] = sha256_file(candidate)
    return digests


def repository_identity(project_root: Path) -> dict[str, Any]:
    """Require a clean tracked worktree and record the exact commit.

    A real attempt runs from a committed source state, so the attempt log's
    commit is enough to reconstruct exactly what ran.
    """

    def git(*arguments: str) -> str:
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

    dirty = git("status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise AttemptError(
            "a Task 9H attempt requires a clean tracked worktree; commit the exact "
            f"source state first (modified: {dirty.splitlines()[0].strip()!r} and possibly more)"
        )
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
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


def validate_attempt_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate one immutable, price-only attempt configuration, fail-closed."""
    if config.get("schema_version") != ATTEMPT_SCHEMA:
        raise AttemptError(f"attempt config schema must be {ATTEMPT_SCHEMA!r}")
    attempt_id = config.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id or "/" in attempt_id:
        raise AttemptError("attempt_id must be a non-empty string without a path separator")
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

    conditioning = tuple(config.get("conditioning_features", ()))
    for name in conditioning:
        if name not in CONDITIONING_FEATURES:
            raise AttemptError(f"unknown conditioning feature {name!r}")
    if len(set(conditioning)) != len(conditioning):
        raise AttemptError("conditioning features must be distinct")

    architecture = config.get("architecture")
    if not isinstance(architecture, Mapping):
        raise AttemptError("attempt config needs an [architecture] table")
    if architecture.get("name") not in ARCHITECTURES:
        raise AttemptError(f"architecture.name must be one of {list(ARCHITECTURES)}")
    if not str(architecture.get("activation", "")):
        raise AttemptError("architecture.activation must be declared")

    seeds = config.get("seeds")
    if not isinstance(seeds, Mapping):
        raise AttemptError("attempt config needs a [seeds] table")
    for key in ("initialization_label", "shuffle_label"):
        if not str(seeds.get(key, "")):
            raise AttemptError(f"seeds.{key} must be a non-empty string")

    rows = config.get("row_selection")
    if not isinstance(rows, Mapping):
        raise AttemptError("attempt config needs a [row_selection] table")
    assert_split_allowed(str(rows.get("partition", "")))
    if not isinstance(rows.get("row_budget"), int) or rows["row_budget"] <= 0:
        raise AttemptError("row_selection.row_budget must be a positive integer")
    if not str(rows.get("salt", "")):
        raise AttemptError("row_selection.salt must be a non-empty string")

    training = config.get("training")
    if not isinstance(training, Mapping):
        raise AttemptError("attempt config needs a [training] table")
    for key in ("batch_size", "epochs", "num_threads"):
        if not isinstance(training.get(key), int) or training[key] <= 0:
            raise AttemptError(f"training.{key} must be a positive integer")

    checkpoint = config.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise AttemptError("attempt config needs a [checkpoint] table")
    assert_split_allowed(str(checkpoint.get("selection_partition", "")))
    if checkpoint["selection_partition"] != "validation":
        raise AttemptError("checkpoints are selected on validation, and only on validation")

    optimizer = config.get("optimizer")
    if not isinstance(optimizer, Mapping):
        raise AttemptError("attempt config needs an [optimizer] table")
    learning_rate = optimizer.get("learning_rate")
    if isinstance(learning_rate, bool) or not isinstance(learning_rate, int | float):
        raise AttemptError("optimizer.learning_rate must be a number")
    if not float(learning_rate) > 0.0:
        raise AttemptError("optimizer.learning_rate must be positive")

    evaluation = config.get("evaluation")
    if not isinstance(evaluation, Mapping):
        raise AttemptError("attempt config needs an [evaluation] table")
    if not isinstance(evaluation.get("batch_size"), int) or evaluation["batch_size"] <= 0:
        raise AttemptError("evaluation.batch_size must be a positive integer")

    paths = config.get("paths")
    if not isinstance(paths, Mapping):
        raise AttemptError("attempt config needs a [paths] table")
    for key in ("dataset", "dataset_manifest", "acceptance_config", "output_directory", "ledger"):
        assert_path_allowed(str(paths.get(key, "")), where=f"paths.{key}")
    if not str(paths.get("output_directory", "")).startswith("artifacts/"):
        raise AttemptError("attempt outputs must live beneath the ignored artifacts/ tree")
    if not str(paths.get("ledger", "")).startswith("runs/"):
        raise AttemptError("the attempt ledger must live beneath the ignored runs/ tree")
    return dict(config)


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
        if parent and parent not in seen:
            raise AttemptError(
                f"attempt {attempt_id!r} names parent {parent!r}, which is not an earlier entry"
            )
        seen.add(attempt_id)
    return {"header": header, "attempts": len(seen), "attempt_ids": sorted(seen)}


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
