# Task specification template

Copy this file to `docs/tasks/active/task-<id>-<short-name>.md` when a task
is ready to start. Fill every section; do not delete a section because it
feels inapplicable — write "None" and say why. Keep numerical thresholds,
formulas, and result tables in the linked contract or frozen snapshot, not
duplicated here.

---

## Status

One of: `Active` (in progress), `Blocked` (state on what), `Completed`
(state the outcome — positive or negative result, both are valid
completions). On completion, this file's outcome must be reflected in
[../project-state.md](../project-state.md) and recorded in
[../decision-log.md](../decision-log.md).

## Objective

One paragraph: the single predeclared question this task answers, and what
a positive versus a negative answer each looks like.

## Motivation

Why this task, why now — what prior result or gap makes it the right next
step. Link the prior task/decision it follows from.

## Authoritative inputs

The contract(s), prior frozen result(s), and versioned config(s) this task
must read and must not contradict without an explicit, recorded decision to
supersede them.

## In scope

What this task will do, stated concretely enough to check against later.

## Out of scope

What this task will explicitly not do, especially anything a reader might
otherwise assume follows automatically (a natural next step that is
deliberately deferred to a later task).

## Predeclared conventions

Any convention, threshold, case set, or criterion that must be fixed and
versioned *before* the first run this task depends on, and cannot be
changed after a result is observed. Name every value that is still open and
must be resolved before that run — do not invent a value to fill the gap.

## Acceptance gates

The exact, predeclared pass/fail conditions, stated as gates a later reader
can check mechanically against the frozen output. State what happens on
failure (does the task stop, or is there a next candidate to test) as
explicitly as what happens on success.

## Manual-run protocol

Exactly how this task's numerical work is invoked. Per `AGENTS.md`, long
numerical runs are manual terminal jobs — never CI-, hook-, or
agent-triggered. State the commands, their order, and any preconditions.

## Artifact policy

Where raw output goes (normally ignored `artifacts/`), what gets frozen and
by which script, and the exact schema/digest discipline that freeze must
satisfy.

## Review requirements

Which review(s) this task requires (`code-reviewer`, `numerical-reviewer`,
both) and what additionally requires a fresh top-level session for material
approval, per [../agent-system.md](../agent-system.md).

## Stop conditions

Conditions under which work must stop before completion — a failed gate
that blocks a later step, a predeclared value that turns out to be
unresolved, evidence that a convention from "Predeclared conventions" was
violated after the fact.

## Completion report

What must be recorded when this task finishes, whatever the outcome:
the exact result, which gates passed/failed, the update made to
[../project-state.md](../project-state.md), and the decision-log entry
recorded in [../decision-log.md](../decision-log.md).
