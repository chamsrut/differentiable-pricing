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
next task) and, if it applies to the work at hand, the active task spec
[docs/tasks/active/task-9g-american-neural-pricer-pilot.md](docs/tasks/active/task-9g-american-neural-pricer-pilot.md)
(task 9G — **protocol and implementation built; `REQUEST CHANGES` findings are
addressed, awaiting fresh approval and merge**).
Execution remains blocked until that review and merge; task 9E's conditional
admission alone authorizes no training. Three other
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
  [docs/tasks/active/task-9g-american-neural-pricer-pilot.md](docs/tasks/active/task-9g-american-neural-pricer-pilot.md)
  (task 9G — continuous-yield American neural-pricer feasibility pilot), status
  `Protocol and implementation built; REQUEST CHANGES findings addressed;
  awaiting fresh approval; no locked run`. The outstanding step is a subsequent fresh approval of the
  corrected protocol/implementation PR, followed by a human-only manual run
  after merge. Task 9C-C2b2 is deferred,
  tasks 9C-C3 and 9D are completed and terminal, and task 9F is on hold; do not
  treat any of those still-present specs as the active one.
- **Frozen-result checks:** `./scripts/check.sh` and CI both run
  `python scripts/freeze_pde_label_policy_v2_results.py --check`. The task
  9C-C3 snapshot it enforces is frozen evidence — never regenerate,
  reformat, or hand-edit it to make a check pass.

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
