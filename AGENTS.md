# AGENTS.md

Durable, cross-tool operating rules for this repository. Read this file
first, in full, before making any change. Tool-specific behavior (hooks,
subagents, model routing) lives in `CLAUDE.md`; this file does not repeat it.

## Mission and research scope

Build a defensible research platform for classical and neural derivative
pricing. Prefer a small, correct experiment with hard validation gates over a
broad demo with unverified claims. The research thesis, hypotheses, and
acceptance standards are in
[docs/research-contract.md](docs/research-contract.md); do not restate or
reinterpret them here.

## Mandatory starting reads

Before any change:

1. This file, in full.
2. [docs/project-state.md](docs/project-state.md) — the current, living
   handoff: what is done, what is in progress, and the exact next task.
3. The active task spec under `docs/tasks/active/`, if one exists for the
   work at hand.
4. The relevant normative contract under `docs/*-contract.md` for anything
   touching pricing, Greeks, sampling, calibration, or data partitioning; and
   [docs/architecture.md](docs/architecture.md) — also a normative contract,
   not narrative — whenever the work touches language-boundary architecture,
   artifact identity/provenance, model structure, or stage-1 compatibility.
5. [docs/american-neural-architecture-freeze-v2.3.md](docs/american-neural-architecture-freeze-v2.3.md)
   — the normative parent of the American neural-pricer roadmap — whenever the
   work touches the American surrogate: its teacher, datasets, partitions,
   references, Greeks, artifacts, native inference, or acceptance sequencing.
   It is versioned, not amended in place.

## Source-of-truth hierarchy

When documents disagree, the higher tier wins. Full precedence rules and the
edit policy for each document class are in
[docs/documentation-map.md](docs/documentation-map.md):

1. code, versioned configuration, and frozen evidence (`docs/results/`,
   `docs/figures/`);
2. normative contracts — for the American neural-pricer roadmap,
   [docs/american-neural-architecture-freeze-v2.3.md](docs/american-neural-architecture-freeze-v2.3.md)
   first; then [docs/architecture.md](docs/architecture.md) and the
   applicable `docs/*-contract.md` files;
3. [docs/decision-log.md](docs/decision-log.md);
4. [docs/project-state.md](docs/project-state.md) and the active task spec;
5. README.md and other narrative documentation.

A narrative document (README, `docs/agentic-workflow.md`) can be stale.
Trust the higher tier when they disagree.

## Git, data, and frozen-result safety

- Inspect `git status` and `git diff` before any commit.
- **Never commit or push without an explicit instruction to do so in the
  current conversation.** Approval from an earlier turn does not carry
  forward to a later one.
- Never discard changes you did not create.
- Never commit secrets, credentials, market-data licenses, or proprietary
  data. The archive under `data/market-feasibility-v1/` and everything
  derived from it stay out of Git.
- Never hand-edit a file under `docs/results/`. Each snapshot family has one
  generator/validator script that recomputes it from a reviewed raw report;
  use that script, never a text edit.
- Never rerun or tune against a partition a protocol already consumed (for
  example the terminal European replication `interpolation_test`, or the
  frozen task 9C-B pilot).
- Keep commits scoped: pricing core, bindings, experiments, and
  documentation should be separable when practical.

## Task specifications are marked in place

Every task spec lives at a stable path under `docs/tasks/active/` for its whole
life, whatever its status. `active/` is a location, not a status claim: the
authoritative status of a task is its own `## Status` block, and the lifecycle
grouping — active, completed, deferred, and the archived exploratory line — is
[docs/tasks/README.md](docs/tasks/README.md). Specs are never relocated, because
the append-only decision log links them by path and two source files name one
spec path directly. Do not treat a spec's presence in `active/` as evidence that
it is the work to pick up.

## Bounded implementation workflow

1. Read the relevant source, test, config, contract, and active task spec.
2. State the numerical or software assumption being changed.
3. Make the smallest coherent change.
4. Add or update tests before claiming completion.
5. Run the narrowest relevant test, then the project's full local gate.
6. Resolve review findings, then rerun the gates.

## No agent-supervised expensive numerical runs

Long numerical studies (CRR convergence, LSM cross-check, PDE label-policy
pilots, market ingestion/reconstruction) are manual, terminal-invoked jobs
only. No hook, CI job, or agent may launch one, schedule one, or treat its
completion as a gate. An agent may report the documented command; a human
runs it, reviews the raw report, and only then does a designated script
freeze a snapshot.

## Verification requirements

- Run the narrowest test that exercises the change first.
- State exactly which checks were actually run and what their real result
  was; do not claim a gate is green from memory or from a previous run.
- A conditional check (one that only runs when a tool is installed) must be
  reported as conditional, not as passed, when that tool was not actually
  invoked.

## Preliminary versus independent review

Any single-session review — including a review subagent's findings — is
**preliminary**. It informs the implementer inside the same conversation. It
is not the independent approval gate. Material approval (accepting a
numerical result, a label policy, or a milestone as done) requires a fresh
top-level session with no prior anchoring on the implementer's reasoning.

## Keep project state and the decision log current

When a milestone starts, changes scope, or completes:

- update [docs/project-state.md](docs/project-state.md) so it states the
  true current state and the exact next task;
- add an entry to [docs/decision-log.md](docs/decision-log.md) for any
  decision that will matter to a later task, using a stable ID, not an
  invented date.

Do not let these drift; they are what the next session — human or agent —
reads to reconstruct where the project actually is.
