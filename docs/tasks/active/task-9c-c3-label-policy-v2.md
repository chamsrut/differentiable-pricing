# Task 9C-C3: PDE label-policy v2

## Status

`Active` — remediation run not yet started. No v2 criteria, remediation
case set, or config exists yet; this document specifies what must be
predeclared before that run, and flags every value it cannot responsibly fix
itself.

## Objective

Task 9C-B (v1) showed the 1600×800 candidate met every predeclared
absolute-error cap on all 22 regular cases, and still failed 5 of them on
checks that are not error caps — bump-ladder stability of the *reference*
delta/vega, and one American-dominance shape violation whose size was
comparable to the solver's own accumulated PSOR residual. Task 9C-C3 asks a
new, separately predeclared question: can a revised stability/shape
criterion (v2) — not a loosened error cap — pass those same five cases,
without breaking cases that already passed, and then hold up on the full
28-case set? This is a **one-shot, predeclared protocol**: the criteria are
fixed before execution and are never revised inside this task after seeing
a result. A positive answer produces a **selected candidate — for price,
and, where separately eligible, delta/vega — pending fresh top-level
independent approval**, not an automatically accepted policy. A negative
answer is a second, honest `no_policy_selected` termination, which may occur
either at the remediation stage or at the confirmation stage. Both are
valid, completed outcomes of this task; see "Acceptance gates" for the
exact state machine.

## Motivation

Task 9C-B froze a negative v1 label-policy result
(`selected_accuracy_policy = no_policy_selected`; see
[../../decision-log.md](../../decision-log.md) DEC-003). **9C-C3 is not the
literal next milestone after 9C-B** — three implemented tasks sit between
them. At the time 9C-B was frozen, the documented plan for what came
immediately next was to read a valuation-time slice out of a single solve
rather than buy more cores; that is a **historical statement of the plan as
of 9C-B**, not the current roadmap, per
[../../research-contract.md](../../research-contract.md) ("the pilot's
measured cost ... is why the next milestone (task 9C-C1) is to read a
valuation-time slice ... not to buy more cores"). That plan was then carried
out in full: task 9C-C1 implemented the valuation-time surface, 9C-C2a
implemented leakage-safe grouped harvesting, and 9C-C2b1 implemented
authoritative external-config verification and three-surface vega — see
[../../pde-numerical-contract.md](../../pde-numerical-contract.md) (Task
9C-C2b1: "`vega_numerically_available` is ... not a supervision-eligibility
claim; task 9C-C3 decides stability"). **Task 9C-C3 is now the current
label-policy milestone.** Tasks 9C-C1, 9C-C2a, and 9C-C2b1 built the
machinery (surface reuse, grouped partitioning, three-surface vega) that
this task's eligibility decision runs on top of, but none of them touched
v1's thresholds or outcome. See
[../../decision-log.md](../../decision-log.md) DEC-003, DEC-004, DEC-007,
DEC-010, DEC-012.

## Authoritative inputs

- [../../pde-numerical-contract.md](../../pde-numerical-contract.md) — PDE
  solver contract and the full task 9C-B/9C-C1/9C-C2a/9C-C2b1 record.
- [../../results/american_pde_label_policy_results_v1.json](../../results/american_pde_label_policy_results_v1.json)
  — frozen v1 evidence: per-case errors, pass/fail outcomes, and the exact
  22 regular / 6 stress case names, digest-pinned by
  `python/tests/test_pde_label_policy_results_snapshot.py`.
- `configs/pde_label_policy_pilot_v1.toml` — v1's frozen case/grid design.
  **Read-only for this task. Never edited.** v2 must define its own,
  separately versioned configuration.
- [../../decision-log.md](../../decision-log.md) DEC-003 (v1 is
  frozen-terminal), DEC-009 (gamma evaluation-only), DEC-010 (Richardson
  validation-only), DEC-012 (algorithmic reuse before clusters).

## In scope

- Predeclaring a v2 stability/shape criterion for the bump-ladder and
  American-dominance checks that failed v1 — before observing any v2
  result.
- Predeclaring a small **remediation case set** and running it once against
  the predeclared v2 criterion.
- Running the full **28-case confirmation set** exactly once, and only if
  the remediation run passes.
- Freezing whichever terminal outcome results — a selected candidate, or a
  `no_policy_selected` termination at either the remediation or the
  confirmation stage — the same way v1 was frozen: schema-validated,
  digest-pinned, recomputed from rows, never hand-edited. A
  remediation-stage termination is frozen too; it is not exempt because it
  is earlier or smaller than a full confirmation run.
- Deciding, as part of the predeclared criterion, whether the
  American-dominance check becomes residual/scale-aware (its tolerance
  scales with the solver's own certified PSOR residual) or is instead
  satisfied by tightening the solve (a finer grid or tighter PSOR
  convergence) so the existing fixed tolerance is no longer smaller than
  the solver's own residual. One of these two paths must be chosen and
  versioned before the remediation run; this document does not choose for
  it.

## Out of scope

- Any American dataset generation or neural training. Task 9C-C3 decides a
  label policy; it does not consume one.
- Parallel or resumable generation (task 9C-C2b2 and later) — this task
  reasons about criteria and a small case set, not throughput.
- Any change to Richardson's role: it stays a reference/validation
  technique, never a candidate label policy, regardless of v2's outcome
  ([../../decision-log.md](../../decision-log.md) DEC-010).
- Any change to gamma's evaluation-only status
  ([../../decision-log.md](../../decision-log.md) DEC-009). This task may
  decide delta/vega eligibility; it does not extend eligibility to gamma.
- Loosening, editing, rerunning, or reinterpreting the frozen v1 result in
  any way ([../../decision-log.md](../../decision-log.md) DEC-003).
- Treating the 6 stress cases as selection inputs. They remain
  price-evaluation cases only — retained in all output, descriptive, and
  never deciding the v2 gate, exactly as in v1.

## Predeclared conventions

The following must be fixed, in a new versioned configuration, **before**
the first remediation solve. None of them may be chosen, or adjusted, after
seeing a v2 result.

- **The v2 stability/shape rule itself** (residual/scale-aware tolerance,
  or tightened-solve target) — **OPEN, not yet decided by this document.**
- **The remediation case set**, ten cases total:
  - the five v1 grid-1600×800 regular failures, by name, from
    `docs/results/american_pde_label_policy_results_v1.json`:
    `regular_euro_negative_rate_call`, `regular_euro_high_rate_call`,
    `regular_american_put_early_exercise`,
    `regular_american_put_negative_rate_control`,
    `regular_american_one_dividend_call`;
  - three smooth controls — **OPEN, proposed candidates only, not yet
    confirmed**: `regular_euro_atm_put`,
    `regular_euro_deep_itm_call_long_high_vol`,
    `regular_euro_deep_otm_put_long`. These are plain European,
    non-dividend regular cases from v1's design that already passed at
    1600×800; the intent is a regression control showing v2 does not
    destabilize cases v1 already handled. Confirm or replace before the
    remediation run.
  - two anchors, from `configs/pde_label_policy_pilot_v1.toml`
    (`anchor_cases`): `regular_euro_atm_call`,
    `stress_american_put_exercise_boundary`.
- **The new configuration's name/version** — proposed
  `configs/pde_label_policy_pilot_v2.toml`, not yet created. Creating it is
  implementation work for this task, not part of this documentation
  restructuring.
- Every convention already fixed by the PDE solver contract (sign, units,
  discount interpolation, dividend-jump treatment) carries over unchanged;
  v2 does not reopen them.

## Acceptance gates

Task 9C-C3 is a **one-shot, predeclared protocol**, not an iterative search.
The v2 criteria and the ten-case remediation set are fixed, in a versioned
config, before the first remediation solve (see "Predeclared conventions").
Once fixed, they are **not revised inside this task**, regardless of what
either run observes. The exact state machine:

1. **Remediation runs once**, against the criteria fixed before execution.
2. **If remediation fails** (any of the ten cases fails the predeclared v2
   criteria):
   - C3 terminates. `selected_accuracy_policy = no_policy_selected`.
   - `confirmation_status = not_run`.
   - `confirmation_not_run_reason` records, precisely, that remediation
     failed and which cases failed on which check.
   - A terminal snapshot is frozen from the remediation run alone (see
     "Artifact policy") and reviewed. This is a complete, valid outcome of
     the task — not a paused, incomplete, or retriable one.
3. **If remediation passes** (all ten cases pass):
   - the unchanged 28-case confirmation set runs exactly once, against the
     same, still-unadjusted v2 criteria. `confirmation_status = run`.
   - if confirmation fails (no candidate clears the regular-case gate): C3
     terminates with `selected_accuracy_policy = no_policy_selected`, the
     same as a remediation-stage failure, and a terminal snapshot is frozen
     from the confirmation run.
   - if confirmation succeeds, the passing candidate **becomes the selected
     candidate pending fresh top-level independent approval** — restricted
     to **price**, on numerically valid regular cases (a case the solver
     actually returned a result for, not one it failed on). Delta and vega
     are not automatically included: each requires its own explicit
     stability-eligibility pass under the v2 criterion; a case can be
     price-selected and simultaneously delta/vega-ineligible. Gamma is not
     a candidate for eligibility in this task regardless of outcome (see
     "Out of scope").
4. **Every terminal outcome freezes a snapshot** — a remediation-stage
   termination, a confirmation-stage `no_policy_selected`, and a
   confirmation-stage selection all produce a schema-validated,
   digest-pinned snapshot under `docs/results/`.

**Criteria may never be loosened or redesigned within C3 after observing
any result**, from either the remediation or the confirmation run, in
either direction. There is no "adjust and try again" step inside this task.
Any later attempt to revise the stability/shape criterion is **not a
continuation of C3**: it requires a separately versioned task and config —
analogous to how 9C-C3 itself is a separately versioned study from 9C-B,
never a rerun of it (see [../../decision-log.md](../../decision-log.md)
DEC-003) — and that later task must explicitly acknowledge that the ten
C3 remediation cases are **already consumed evidence** from this task's
one-shot run, not untouched validation cases available for a second
attempt.

## Manual-run protocol

Both the remediation run and the confirmation run are manual, terminal-invoked
jobs, per `AGENTS.md` ("No agent-supervised expensive numerical runs"). No
hook, CI job, or agent may launch or schedule either run. The exact command
form follows the same pattern as the v1 pilot
([../../pde-numerical-contract.md](../../pde-numerical-contract.md), "Run
the controlled task 9C-B PDE label-policy pilot") once the v2 config exists;
it is not restated here to avoid a second copy that can drift.

## Artifact policy

Raw reports for the remediation run, and for the confirmation run if it
runs, go to ignored `artifacts/`, never committed. A frozen,
schema-validated, digest-pinned snapshot under `docs/results/` — generated
the same way `american_pde_label_policy_results_v1.json` was, by its own
dedicated freeze/check scripts, never hand-edited — is committed for
**every** terminal outcome, including a remediation-stage termination
(`confirmation_status = not_run`). Freezing is not conditional on reaching
confirmation.

## Review requirements

`code-reviewer` for the v2 runner/config code. `numerical-reviewer` for the
v2 stability/shape criterion, the remediation case selection, and the
eligibility logic for delta/vega. Both are separate-context preliminary
review ([../../agent-system.md](../../agent-system.md)), not independent
final approval. Accepting the selected candidate as an approved v2 policy —
or accepting any `no_policy_selected` termination, from either stage, as
final — as the basis for unblocking American dataset generation is a
material approval and requires a fresh top-level session.

## Stop conditions

- Any remediation-set case fails: stop before the confirmation run;
  terminate per the state machine in "Acceptance gates"
  (`confirmation_status = not_run`), do not attempt a second remediation
  run inside this task.
- Any of the "OPEN" values above is still unresolved when a run is about to
  start: stop and resolve it first; do not fill it in ad hoc during the run.
- Any evidence that a predeclared criterion was adjusted after observing a
  result, in either the remediation or confirmation run: that run's outcome
  is invalid as C3 evidence. It is **not** redone inside this task under an
  adjusted criterion; any further attempt requires a separately versioned
  follow-up task and config, per "Acceptance gates."
- Any attempt to reuse a v1 threshold as a v2 threshold without an
  independent, recorded justification in this file or the decision log.

## Implementation requirements (future — not built in this documentation change)

This document specifies the protocol; it does not implement it. When the
v2 runner is implemented, its test suite must include, at minimum:

- a validator test for a remediation-failure snapshot (`confirmation_status
  = not_run`, `confirmation_not_run_reason` populated,
  `selected_accuracy_policy = no_policy_selected`);
- a test that the confirmation run is **prohibited** — refuses to execute,
  not merely "not called" — when the remediation gate has failed;
- a test that rejects a post-result criterion change: replaying a
  remediation or confirmation run with an altered v2 criterion after a
  result has been observed must fail loudly, not silently produce a second
  attempt.

These tests are not created by this documentation-only change.

## Completion report

On completion, record, unconditionally:

- the exact remediation-set outcome, case by case;
- `confirmation_status` (`run` or `not_run`);
- the frozen snapshot path and its digest;
- an update to [../../project-state.md](../../project-state.md) reflecting
  the new current state and next task;
- a new entry in [../../decision-log.md](../../decision-log.md) recording
  the outcome, whichever it is.

Additionally, **only when `confirmation_status == run`**, record the exact
confirmation-set outcome, case by case, and which of price/delta/vega
eligibility (if any) was granted to the selected candidate. **When
`confirmation_status == not_run`**, record instead the
`confirmation_not_run_reason` (i.e., that remediation failed) in place of
any confirmation-set outcome — there is no confirmation result to report.
