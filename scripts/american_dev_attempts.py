#!/usr/bin/env python3
"""Task 9H attempt-log tool: `record` one attempt, or `check` the log offline.

Both modes are offline. They read tracked files and the ignored reports a
completed run already wrote; they run no pricing, no training and no dataset
access, and they import nothing that needs PyTorch or a compiled extension — so
`check` runs in `scripts/check.sh` and in the lightweight CI job.

`record` is the **only** writer of `docs/attempts/task-9h-attempt-log.jsonl`. It
refuses a duplicate attempt ID, so an existing entry is never rewritten, and it
records the configuration digest the attempt actually ran — which is what makes
"attempt configurations are immutable after use" checkable rather than
aspirational.

`record` writes only the canonical tracked log and only accepts a report of the
schema this workbench emits, judged against the canonical Task 9G acceptance
configuration and section. A second log file, an unrecognized report, or an
attempt that judged itself against some other criterion is refused.

`check` enforces six properties:

1. no Task 9H source names a final or held-out partition as data, using the
   **same** forbidden-token definition the runtime guard uses;
2. the runner exposes no final-evaluation command;
3. the validation-geometry analysis is validation-only: its partition is a
   module constant, no other split name appears as a literal in its module or
   its script, and its script exposes no final-evaluation subcommand;
4. the attempt log is a valid append-only log, and every tracked attempt
   configuration is valid — which includes pointing at the canonical acceptance
   configuration;
5. every logged attempt's configuration still hashes to its recorded digest;
6. every logged attempt cites the canonical criterion file, section and digest.

Every recorded attempt is a development measurement selected against
`validation`. None of them is a project result.
"""

from __future__ import annotations

import argparse
import ast
import importlib.machinery
import importlib.util
import json
import sys
import tomllib
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Final

PROJECT_ROOT: Final = Path(__file__).resolve().parents[1]
PACKAGE: Final = PROJECT_ROOT / "python/src/differentiable_pricing/ml/american_dev"
ATTEMPTS_MODULE: Final = PACKAGE / "attempts.py"
ATTEMPT_LOG: Final = PROJECT_ROOT / "docs/attempts/task-9h-attempt-log.jsonl"
RUNNER: Final = PROJECT_ROOT / "scripts/run_american_dev_attempt.py"
GEOMETRY_MODULE: Final = PACKAGE / "geometry.py"
GEOMETRY_SCRIPT: Final = PROJECT_ROOT / "scripts/analyze_american_dev_geometry.py"
TASK_9H_SCRIPTS: Final = (
    "scripts/run_american_dev_attempt.py",
    "scripts/american_dev_attempts.py",
    "scripts/analyze_american_dev_geometry.py",
)
ATTEMPT_CONFIG_GLOB: Final = "configs/american_dev_attempt_*.toml"

EXPECTED_SUBCOMMANDS: Final = {"run", "status"}
EXPECTED_GEOMETRY_SUBCOMMANDS: Final = {"analyze", "show"}
#: The one partition the validation-geometry analysis may read as data. Checked
#: against the module constant, and against every split name either the module
#: or its script spells as a literal.
GEOMETRY_PARTITION: Final = "validation"


def _relative(path: Path) -> str:
    """Repository-relative when possible, absolute otherwise."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_attempt_rules() -> Any:
    """Load ``attempts.py`` by path, without importing the project package.

    ``attempts.py`` is deliberately PyTorch-free, so this tool runs in the
    lightweight environment. A synthetic parent package is registered so its
    relative imports resolve beside it, and nothing else is touched.
    """
    package_name = "task9h_attempt_rules"
    package = importlib.util.module_from_spec(
        importlib.machinery.ModuleSpec(package_name, None, is_package=True)
    )
    package.__path__ = [str(PACKAGE)]
    sys.modules[package_name] = package
    specification = importlib.util.spec_from_file_location(
        f"{package_name}.attempts", ATTEMPTS_MODULE
    )
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load '{ATTEMPTS_MODULE}'")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


#: The rules module, loaded once. Everything below reads its definitions rather
#: than restating them.
RULES: Final = _load_attempt_rules()

#: Partition names that may never be used as data by a Task 9H code path.
#: **The same tuple the runtime guard enforces**, imported rather than copied:
#: a token added for one and forgotten for the other is how a static check and
#: the code it guards quietly stop agreeing.
FORBIDDEN_PARTITIONS: Final = RULES.FORBIDDEN_PARTITION_TOKENS


# ---------------------------------------------------------------------------
# record
# ---------------------------------------------------------------------------


def _load_json(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"error: cannot load {description} '{path}': {error}") from error
    if not isinstance(payload, dict):
        raise SystemExit(f"error: {description} must be a JSON object")
    return payload


def build_record(report: dict[str, Any], arguments: Any) -> dict[str, Any]:
    """Assemble the attempt-log record from evidence the run already wrote."""
    metrics = report["price_metrics"]
    return {
        "attempt_id": report["attempt_id"],
        "parent_attempt": report["parent_attempt"],
        "hypothesis": report["hypothesis"],
        "git_commit": report["repository"]["commit"],
        "config_path": report["config_path"],
        "config_sha256": report["config_sha256"],
        "source_digests": report["source_digests"],
        "architecture": report["architecture"],
        "features": report["features"],
        "target_and_reconstruction": report["target_and_reconstruction"],
        "seeds": report["seeds"],
        "selected_rows": {
            key: report["selected_rows"][key]
            for key in ("partition", "salt", "rule", "count")
        },
        "optimizer_and_budget": report["optimizer_and_budget"],
        "price_metrics": {
            "best_epoch": report["training"]["best_epoch"],
            "overall_normalized": metrics["slices"]["overall"]["normalized"],
            "european_crr_baseline_normalized": metrics["european_crr_baseline"]["normalized"],
            "gate": metrics["gate"],
        },
        "bound_and_shape_diagnostics": {
            "material_bound_violations": metrics["diagnostics"]["material_bound_violations"],
            "material_shape_violations": metrics["diagnostics"]["material_shape_violations"],
            "checks": metrics["diagnostics"]["checks"],
        },
        "criterion": report["criterion"],
        "outcome": arguments.outcome,
        "interpretation": arguments.interpretation,
        "next_action": arguments.next_action,
        "selection_bias": (
            "measured against validation, which this loop selects on repeatedly; a "
            "development measurement, not a project result"
        ),
    }


def record(arguments: Any, rules: Any) -> int:
    if str(arguments.next_action).split(" ", 1)[0] not in rules.NEXT_ACTIONS:
        print(
            f"error: --next-action must start with one of {list(rules.NEXT_ACTIONS)}",
            file=sys.stderr,
        )
        return 2
    if not str(arguments.interpretation).strip():
        print("error: --interpretation must not be empty", file=sys.stderr)
        return 2
    report_path = arguments.report.resolve()
    report = _load_json(report_path, "attempt report")
    log_path = arguments.log.resolve() if arguments.log else PROJECT_ROOT / rules.ATTEMPT_LOG_PATH
    try:
        # The log is one tracked file, and the report has to be one this
        # workbench wrote against the canonical criterion. Both are checked
        # before the append, so a refusal leaves the log untouched.
        log_path = rules.assert_canonical_log_path(log_path, PROJECT_ROOT)
        rules.assert_report_is_recordable(report)
        state = rules.append_attempt(log_path, build_record(report, arguments))
    except rules.AttemptError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except KeyError as error:
        print(f"error: attempt report is missing {error}", file=sys.stderr)
        return 2
    print(json.dumps({"attempts": state["attempts"], "log": _relative(log_path)}, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def _task_9h_sources() -> Iterator[Path]:
    yield from sorted(PACKAGE.glob("*.py"))
    for relative in TASK_9H_SCRIPTS:
        candidate = PROJECT_ROOT / relative
        if candidate.is_file():
            yield candidate


def _docstring_nodes(tree: ast.AST) -> set[int]:
    marked: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                marked.add(id(body[0].value))
    return marked


def names_a_partition(value: str) -> bool:
    """True when a string uses a forbidden partition as a name or path component.

    Prose that merely mentions the partition — "interpolation_test was never
    opened" — has the token embedded in a sentence, so no path component equals
    it and the string is left alone. That distinction is deliberate: the
    prohibition has to be stated somewhere, and it is stated in prose.
    """
    for component in value.replace("\\", "/").split("/"):
        if component.split(".")[0].strip() in FORBIDDEN_PARTITIONS:
            return True
    return False


def check_no_final_partition_references() -> list[str]:
    failures: list[str] = []
    owners = {ATTEMPTS_MODULE.resolve(), Path(__file__).resolve()}
    for path in _task_9h_sources():
        if path.resolve() in owners:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings or not names_a_partition(node.value):
                continue
            failures.append(
                f"{_relative(path)}:{node.lineno} uses forbidden partition name "
                f"{node.value!r}; Task 9H reaches train and validation only"
            )
    return failures


def check_runner_has_no_final_evaluation() -> list[str]:
    text = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(text, filename=str(RUNNER))
    declared: set[str] = set()
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        if target != "COMMANDS" or node.value is None:
            continue
        declared = {
            element.value
            for element in getattr(node.value, "elts", [])
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    failures: list[str] = []
    if declared != EXPECTED_SUBCOMMANDS:
        failures.append(
            f"{_relative(RUNNER)} declares subcommands {sorted(declared)}; expected exactly "
            f"{sorted(EXPECTED_SUBCOMMANDS)}"
        )
    for forbidden in ("final-evaluate", "final_evaluate"):
        if f'"{forbidden}"' in text or f"'{forbidden}'" in text:
            failures.append(
                f"{_relative(RUNNER)} references {forbidden!r}; Task 9H exposes no "
                "final-evaluation command and no flag that adds one"
            )
    return failures


def _module_constant(tree: ast.AST, name: str) -> Any:
    """The value assigned to a module-level string constant, or ``None``."""
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        if target != name or node.value is None:
            continue
        if isinstance(node.value, ast.Constant):
            return node.value.value
    return None


def _declared_subcommands(tree: ast.AST) -> set[str]:
    for node in ast.walk(tree):
        target = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target = node.target.id
        elif (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            target = node.targets[0].id
        if target != "COMMANDS" or node.value is None:
            continue
        return {
            element.value
            for element in getattr(node.value, "elts", [])
            if isinstance(element, ast.Constant) and isinstance(element.value, str)
        }
    return set()


def check_geometry_is_validation_only(rules: Any) -> list[str]:
    """The validation-geometry analysis reads one partition, fixed in the source.

    Three separable properties, because the interesting failure is a later edit
    that quietly widens the analysis rather than an obviously wrong one:

    * the analysis partition is a **module constant** equal to ``validation`` —
      not an argument, a configuration key or a subcommand flag;
    * neither the module nor its script spells any *other* split name as a
      string literal outside a docstring, so nothing can be pointed at ``train``
      or anywhere else without this check noticing;
    * the script declares exactly ``analyze`` and ``show``, and mentions no
      final-evaluation command.

    The final- and held-out-partition prohibition is not restated here: both
    files are Task 9H sources, so :func:`check_no_final_partition_references`
    already scans them with the runtime guard's own token list.
    """
    failures: list[str] = []
    if not GEOMETRY_MODULE.is_file() or not GEOMETRY_SCRIPT.is_file():
        return [
            f"{_relative(GEOMETRY_MODULE)} and {_relative(GEOMETRY_SCRIPT)} must both exist; "
            "the validation-geometry analysis is part of the checked Task 9H surface"
        ]
    module_tree = ast.parse(
        GEOMETRY_MODULE.read_text(encoding="utf-8"), filename=str(GEOMETRY_MODULE)
    )
    declared_partition = _module_constant(module_tree, "ANALYSIS_PARTITION")
    if declared_partition != GEOMETRY_PARTITION:
        failures.append(
            f"{_relative(GEOMETRY_MODULE)} declares ANALYSIS_PARTITION "
            f"{declared_partition!r}; the geometry analysis reads {GEOMETRY_PARTITION!r} and "
            "nothing else"
        )
    other_splits = {name for name in rules.ALLOWED_SPLITS if name != GEOMETRY_PARTITION}
    for path, tree in (
        (GEOMETRY_MODULE, module_tree),
        (
            GEOMETRY_SCRIPT,
            ast.parse(GEOMETRY_SCRIPT.read_text(encoding="utf-8"), filename=str(GEOMETRY_SCRIPT)),
        ),
    ):
        docstrings = _docstring_nodes(tree)
        for node in ast.walk(tree):
            if not isinstance(node, ast.Constant) or not isinstance(node.value, str):
                continue
            if id(node) in docstrings or node.value not in other_splits:
                continue
            failures.append(
                f"{_relative(path)}:{node.lineno} names partition {node.value!r}; the "
                f"validation-geometry analysis reads {GEOMETRY_PARTITION!r} as data and derives "
                "every other partition name from attempts.ALLOWED_SPLITS"
            )
    script_text = GEOMETRY_SCRIPT.read_text(encoding="utf-8")
    script_tree = ast.parse(script_text, filename=str(GEOMETRY_SCRIPT))
    declared = _declared_subcommands(script_tree)
    if declared != EXPECTED_GEOMETRY_SUBCOMMANDS:
        failures.append(
            f"{_relative(GEOMETRY_SCRIPT)} declares subcommands {sorted(declared)}; expected "
            f"exactly {sorted(EXPECTED_GEOMETRY_SUBCOMMANDS)}"
        )
    for forbidden in ("final-evaluate", "final_evaluate"):
        if f'"{forbidden}"' in script_text or f"'{forbidden}'" in script_text:
            failures.append(
                f"{_relative(GEOMETRY_SCRIPT)} references {forbidden!r}; Task 9H exposes no "
                "final-evaluation command and no flag that adds one"
            )
    return failures


def check_attempt_configurations(rules: Any) -> list[str]:
    failures: list[str] = []
    acceptance_path = PROJECT_ROOT / rules.ACCEPTANCE_CONFIG_PATH
    acceptance: dict[str, Any] | None = None
    if acceptance_path.is_file():
        try:
            with acceptance_path.open("rb") as stream:
                acceptance = tomllib.load(stream)
        except Exception as error:  # an unloadable acceptance file is a check failure
            failures.append(f"{_relative(acceptance_path)} is unloadable: {error}")
    for path in sorted(PROJECT_ROOT.glob(ATTEMPT_CONFIG_GLOB)):
        try:
            with path.open("rb") as stream:
                payload = tomllib.load(stream)
            rules.validate_attempt_config(payload)
        except Exception as error:  # an unloadable or invalid config is a check failure
            failures.append(f"{_relative(path)} is invalid: {error}")
            continue
        if acceptance is None:
            continue
        # A head carrying a predeclared normalized temperature is re-checked
        # offline against the units the acceptance file states its thresholds in,
        # so an inconsistency surfaces in check.sh and CI rather than only at the
        # moment a human starts an expensive run.
        try:
            rules.assert_temperature_consistent(str(payload.get("head")), acceptance)
        except Exception as error:
            failures.append(f"{_relative(path)} declares an inconsistent head temperature: {error}")
    return failures


def check(rules: Any) -> int:
    failures: list[str] = []
    failures.extend(check_no_final_partition_references())
    failures.extend(check_runner_has_no_final_evaluation())
    failures.extend(check_geometry_is_validation_only(rules))
    attempts = 0
    if not ATTEMPT_LOG.is_file():
        failures.append(f"{_relative(ATTEMPT_LOG)} is missing")
    else:
        try:
            attempts = rules.validate_attempt_log(ATTEMPT_LOG)["attempts"]
        except Exception as error:  # every failure mode of the log is a check failure
            failures.append(f"attempt log is invalid: {error}")
        else:
            failures.extend(rules.check_configuration_immutability(PROJECT_ROOT, ATTEMPT_LOG))
            failures.extend(rules.check_recorded_criterion(PROJECT_ROOT, ATTEMPT_LOG))
    failures.extend(check_attempt_configurations(rules))
    if failures:
        print("error: Task 9H attempt-log checks failed", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    configurations = len(list(PROJECT_ROOT.glob(ATTEMPT_CONFIG_GLOB)))
    print(
        f"task 9H attempts ok: {configurations} attempt configuration(s) valid, "
        f"{attempts} logged attempt(s) with immutable configurations and the canonical "
        "criterion, no final-partition reference, no final-evaluation command, "
        f"geometry analysis restricted to {GEOMETRY_PARTITION!r}"
    )
    return 0


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    entry = commands.add_parser("record")
    entry.add_argument("--report", type=Path, required=True)
    entry.add_argument("--log", type=Path)
    entry.add_argument("--outcome", required=True)
    entry.add_argument("--interpretation", required=True)
    entry.add_argument(
        "--next-action",
        required=True,
        help="continue|stop|change-direction, followed by the reason",
    )
    commands.add_parser("check")
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    rules = RULES
    if arguments.command == "record":
        if arguments.outcome not in rules.OUTCOMES:
            print(f"error: --outcome must be one of {list(rules.OUTCOMES)}", file=sys.stderr)
            return 2
        return record(arguments, rules)
    return check(rules)


if __name__ == "__main__":
    raise SystemExit(main())
