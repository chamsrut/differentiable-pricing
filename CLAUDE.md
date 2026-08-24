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

- [docs/project-state.md](docs/project-state.md) — current state, current
  evidence, limitations, and the **exact next task**. It is the authority on
  what the project is doing right now; this file deliberately does not
  duplicate it, so it cannot go stale here.
- [docs/american-neural-architecture-freeze-v2.3.md](docs/american-neural-architecture-freeze-v2.3.md)
  — the normative parent of the American neural-pricer roadmap, whenever the
  work touches the American surrogate.
- The active task spec, and [docs/tasks/README.md](docs/tasks/README.md) for
  the lifecycle grouping. **Task specs are marked in place and never
  relocated**, so `docs/tasks/active/` also holds completed, deferred, and
  on-hold specs. Presence in that directory is not evidence that a task is the
  work to pick up — read its `## Status` block and the index.

Two standing prohibitions that survive every roadmap change, and that you must
not relax on your own judgement:

- **You must never open, hash, stat, import, count, or inspect
  `interpolation_test` or any other final partition, and neither may any
  development or diagnostic code path.** This binds agents and development code
  absolutely: there is no task, review, or debugging reason that justifies
  reading a final partition, and "just checking the row count" is a violation.
  Task 9G's `final-evaluate` is **forbidden** under its own protocol, so no
  final evaluation may be invoked under it now or later (DEC-038).

  The one narrow exception is not yours to take: a **future predeclared
  one-shot final-evaluation protocol** may authorize its own designated,
  human-invoked evaluation path to open a **fresh** final partition **exactly
  once**. That access belongs to that protocol's named entry point, run by a
  human, under gates fixed before the partition exists — never to an agent, an
  interactive session, or any development code path, and never to a partition a
  protocol has already consumed.
- **Never rerun, retune against, regenerate, reformat, or hand-edit frozen
  evidence** under `docs/results/`, and never treat an admission or a label
  policy as training authorization.

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
- **Active task:** read [docs/project-state.md](docs/project-state.md),
  "Exact next task", and [docs/tasks/README.md](docs/tasks/README.md) for the
  lifecycle grouping. Do not infer the active task from directory membership:
  `docs/tasks/active/` holds completed, deferred, and on-hold specs too,
  because specs are marked in place and never relocated.
- **Frozen-result checks:** `./scripts/check.sh` and CI both run
  `python scripts/freeze_pde_label_policy_v2_results.py --check` and
  `python scripts/freeze_american_neural_pilot_results.py --check`. Both are
  offline: they read only tracked files and rerun no pricing, training,
  latency measurement, or IV inversion. The task 9C-C3 and task 9G snapshots
  they enforce are frozen evidence — never regenerate, reformat, or hand-edit
  either to make a check pass.

## Required workflow

1. Read `AGENTS.md`, the relevant contract(s) — including
   [docs/american-neural-architecture-freeze-v2.3.md](docs/american-neural-architecture-freeze-v2.3.md)
   for American neural work — and the active task spec.
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
