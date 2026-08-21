# CLAUDE.md

@AGENTS.md

Read `AGENTS.md` first, in full — it is imported above and holds every
durable, cross-tool rule (mission, source-of-truth hierarchy, git/data
safety, the bounded workflow, the ban on agent-supervised expensive runs,
and the preliminary-vs-independent-review distinction). This file imports
`AGENTS.md` and primarily adds Claude Code-specific instructions. It also
repeats a small number of critical shared reading, workflow, and
git-safety guardrails in condensed form for local clarity. `AGENTS.md`
remains the canonical source for those shared rules.

Also read, before starting work:
[docs/project-state.md](docs/project-state.md) (current state and the exact
next task) and the active task spec
[docs/tasks/active/task-9h-american-pricer-development.md](docs/tasks/active/task-9h-american-pricer-development.md)
(task 9H — adaptive American **price**-model development on branch
`experiment/task-9h-american-pricer-development`). **Task 9H is development,
not confirmatory research**: it selects against `validation` repeatedly, so
nothing it produces is a project result, and a candidate that meets its
criterion needs a separately predeclared confirmation on a fresh final partition
as its own task. **No task 9H code path may open, hash, stat, import, count or
inspect `interpolation_test` or any other final partition**, and its runner
exposes no final-evaluation command. Its scope is **price only** — Greeks,
latency, IV and transfer are separate follow-up stages and none of their
machinery exists. Its "works" criterion is Task 9G's, read from
`configs/american_neural_pilot_acceptance_v1.toml` and applied through
`ml.american_pilot`; it is not restated in task 9H and not revised because an
attempt failed. Agents implement code and analyze compact summaries; **the human
invokes every training run**.

Task 9G is **terminal and approved**:
[docs/tasks/active/task-9g-american-neural-pricer-pilot.md](docs/tasks/active/task-9g-american-neural-pricer-pilot.md)
— implementation merged at `e930454`, the single human-invoked locked
`run-to-validation` completed and its validation gates failed
(`status=validation_gates_failed`, `outcome=failure_to_learn`,
`final_evaluation_attempts=0`, `final_partition_consumed=false`), and
`final-evaluate` is forbidden under that protocol. The frozen result is
[docs/results/american_neural_pilot_results_v1.json](docs/results/american_neural_pilot_results_v1.json)
(DEC-038), approved by a fresh top-level `APPROVE TASK 9G RESULT FOR MERGE`
(DEC-039). One seed and one budget do not establish H2. Do not rerun task 9G,
retune against it, or open `interpolation_test`; task 9E's conditional
admission alone authorizes no training.

**Task 9G's protocol pins 35 tracked files by SHA-256**, and
`scripts/check_american_neural_pilot_protocol.py` reconciles them in both
`scripts/check.sh` and CI. That set includes `docs/architecture.md`,
`docs/research-contract.md`, `docs/american-crr-contract.md`,
`docs/american-crr-dataset-admission.md`, `docs/pde-numerical-contract.md`,
`pyproject.toml`, `python/src/differentiable_pricing/__init__.py`, the
`ml/american*.py`, `ml/artifact.py`, `ml/config.py` and `ml/model.py` modules,
the task 9G scripts, `CMakeLists.txt` and the listed C++/binding sources.
**Editing any of them breaks a historical protocol check**; new work adds new
files instead, and records the resulting wording limitation.

Three other
specs stay in `docs/tasks/active/` and are **not** work to pick up: task 9C-C2b2 is `Deferred`
([docs/tasks/active/task-9c-c2b2-parallel-resumable-generation.md](docs/tasks/active/task-9c-c2b2-parallel-resumable-generation.md)),
to be resumed only when the XSP/SPY phase needs dataset-scale PDE generation;
task 9C-C3 and task 9D are **completed**
([docs/tasks/active/task-9c-c3-label-policy-v2.md](docs/tasks/active/task-9c-c3-label-policy-v2.md),
[docs/tasks/active/task-9d-data-holdings-audit.md](docs/tasks/active/task-9d-data-holdings-audit.md)),
terminal, marked-`Completed` history; and task 9F is `On hold`
([docs/tasks/active/task-9f-remote-data-access-plan.md](docs/tasks/active/task-9f-remote-data-access-plan.md)),
a plan only. Task 9D's catalogue of the local data holdings is
[docs/data-holdings-catalogue.md](docs/data-holdings-catalogue.md).

## Hooks

- `PostToolUse` on `Write`/`Edit` runs `.claude/hooks/check-edited-file.py`:
  validates JSON/TOML parse on the edited file; runs `ruff check` on an
  edited `.py` file **when `ruff` is importable, falling back to
  `python -m py_compile` otherwise** (syntax-only, not a lint, when Ruff is
  absent); and runs `clang-format --dry-run --Werror` on an edited C++ file
  **when `clang-format` is importable on this machine**. It never rewrites a
  file. Report a clean run as "syntax/lint/format-clean where checked, with
  whichever tools were actually available," not as a correctness or design
  approval — and not as "linted" if Ruff fell back to `py_compile`.
- `.githooks/pre-commit` runs `./scripts/check.sh --quick`. It only takes
  effect once `scripts/install-git-hooks.sh` has been run in this clone — it
  is not active by default. Do not assume it ran on a change you did not
  make in this session.

## Subagents

- `code-reviewer` (Sonnet, read-only, plan mode) — independent software
  review of a diff.
- `numerical-reviewer` (Sonnet, read-only, plan mode) — independent
  pricing/Greeks/sampling/calibration/ML review.

**Both agents perform separate-context preliminary review**, per
`AGENTS.md` — never call either one "independent final approval." They
inform the implementer in the same session; they are not the independent
approval gate. Do not describe a subagent run as final approval of a
numerical result or a milestone. Material approval requires a fresh
top-level session with no prior anchoring on this session's reasoning —
see [docs/agent-system.md](docs/agent-system.md).

No other subagent exists in this repository today. A cheap read-only triage
agent is proposed but not implemented; see `docs/agent-system.md`.

## Model-routing guidance

- Non-trivial architecture, numerical design, or cross-file synthesis:
  prefer the strongest available main-session model.
- Bounded, well-specified implementation once the design is fixed: Sonnet
  is sufficient.
- `code-reviewer` / `numerical-reviewer`: Sonnet, as configured.

Full rationale and the proposed (not yet implemented) routing for a future
triage subagent: [docs/agent-system.md](docs/agent-system.md).

## Command routing

- **Full local gate:** `./scripts/check.sh`. Non-quick (full) mode always
  configures and builds `build/check` fresh and always runs the C++ tests,
  validates configs/snapshots/figures, and finishes with the full Python
  suite — which needs `pip install -e '.[dev,train]'` first, or the gate
  fails rather than silently skipping it. `--quick` (used by the pre-commit
  hook) skips the Python suite **and runs the C++ tests only if `build/check`
  already exists** from a prior full run — on a fresh clone, `--quick` may
  perform **no C++ test execution at all**; that is not a failure, there is
  simply nothing built yet to test.
  **It runs `ruff check` only when `ruff` is importable, and runs
  `clang-format` only when `clang-format` is installed locally — neither is
  unconditional, and CI does not run `clang-format` at all.** The
  clang-format-18 baseline observed on at least one contributor machine is
  already **non-green** against the checked-in C++ sources, so even a
  successful local `clang-format` invocation is not evidence the tree is
  clang-format-clean by some other version's standard. Never report the
  gate, or CI, as having verified C++ formatting unless `clang-format`
  genuinely ran in that invocation and its baseline is known-green — see
  [docs/agent-system.md](docs/agent-system.md), "Clang-format baseline
  problem." **GitHub Actions CI remains the authoritative clean-environment
  gate**; the local script's behavior depends on machine state (what's
  installed, whether `build/check` exists) in ways CI does not.
- **Study runners, snapshot-freeze commands, and demo scripts:** these
  change as tasks complete. Read
  [docs/project-state.md](docs/project-state.md) for what is current and
  the relevant `docs/*-contract.md` for the exact invocation and
  load-bearing rules, rather than trusting a command copied from an older
  conversation or an out-of-date narrative document.
- **Active task:**
  [docs/tasks/active/task-9h-american-pricer-development.md](docs/tasks/active/task-9h-american-pricer-development.md)
  (task 9H — adaptive American price-model development), status
  `infrastructure implemented and hardened; one attempt recorded`
  (`scratch_direct_control_v1`, `criterion_not_met`; DEC-042). Its manual
  human command is
  `python3 scripts/run_american_dev_attempt.py run|status` — **an agent reports
  it and stops**. The offline, agent-safe tool is
  `python3 scripts/american_dev_attempts.py record|check`.
  Task 9G is terminal and approved (DEC-038, DEC-039); `final-evaluate` is
  **forbidden** under the 9G protocol, since its final-entry rule failed, so no
  final evaluation may be invoked now or later, and any future one needs a new
  protocol and a fresh final partition.
  Task 9C-C2b2 is deferred,
  tasks 9C-C3 and 9D are completed and terminal, and task 9F is on hold; do not
  treat any of those still-present specs as the active one.
- **Frozen-result checks:** `./scripts/check.sh` and CI both run
  `python scripts/freeze_pde_label_policy_v2_results.py --check` and
  `python scripts/freeze_american_neural_pilot_results.py --check`. Both are
  offline: they read only tracked files and rerun no pricing, training,
  latency measurement, or IV inversion. The task 9C-C3 and task 9G snapshots
  they enforce are frozen evidence — never regenerate, reformat, or hand-edit
  either to make a check pass. `python3 scripts/american_dev_attempts.py check`
  runs beside them and is also offline: it parses tracked task 9H sources,
  attempt configurations and the append-only attempt log, and imports nothing
  from the project package.

## Required workflow

1. Read `AGENTS.md`, the relevant contract(s), and the active task spec.
2. State the numerical or software assumption being changed.
3. Make the smallest coherent change.
4. Add or update tests before claiming completion.
5. Run the narrow test, then `./scripts/check.sh`.
6. For non-trivial changes, invoke `code-reviewer`.
7. For pricing, Greeks, sampling, calibration, or ML evaluation, also invoke
   `numerical-reviewer`.
8. Resolve findings in the main context and rerun the gates. Review agents
   do not edit files, and their findings remain preliminary until a fresh
   top-level session materially approves the result.
9. Update [docs/project-state.md](docs/project-state.md) and
   [docs/decision-log.md](docs/decision-log.md) whenever a milestone starts,
   changes scope, or completes.

## Review standard

Reviews are findings-first. Include severity, file and line, mechanism of
failure, proposed fix, and a regression test. Avoid style comments already
enforced mechanically. If no issues are found, say so and list residual
risks/testing gaps.

## Git safety

Do not commit or push without an explicit instruction to do so in the
*current* conversation — an earlier approval does not carry forward. Full
git/data/frozen-result safety rules are in `AGENTS.md` and are not repeated
here.
