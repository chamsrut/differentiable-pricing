# Agent system architecture

The intended architecture for agent-assisted work in this repository, and an
explicit account of which parts of it exist today versus are only planned.
Nothing in this file is evidence that a planned piece is implemented; see
[documentation-map.md](documentation-map.md).

## `CLAUDE.md` versus `AGENTS.md`

`AGENTS.md` (repository root) is **automatically discovered and loaded by
Codex**. It contains the shared, durable, cross-tool coding-agent rules:
mission, source-of-truth hierarchy, git/data safety, the bounded
implementation workflow, the ban on agent-supervised expensive runs, and the
preliminary-vs-independent-review distinction. It is written to be read
directly by Codex, and by any other AGENTS.md-aware tool, not only Claude
Code.

`CLAUDE.md` is **automatically loaded by Claude Code** and imports
`AGENTS.md`: `@AGENTS.md` at the top of the file pulls it in in full.
`CLAUDE.md` **primarily** adds what is genuinely Claude-specific — which
hooks run, which subagents exist and when to invoke them, model-routing
guidance, and command routing — but it also **deliberately repeats a small
number of critical cross-tool reading, workflow, and git-safety guardrails**
in condensed, local form (for example: read `AGENTS.md` first; the numbered
required-workflow steps; the headline "do not commit or push without an
explicit instruction" rule). **Those repetitions do not create a separate
source of truth**: they exist only for a reader's local clarity inside
`CLAUDE.md`, and they must stay consistent with `AGENTS.md`'s fuller
statement of the same rules, which remains canonical. `CLAUDE.md` **should
not grow into a comprehensive duplicate of `AGENTS.md`**; it does not
restate the full git/data/frozen-result safety list or numerical contract
content, both of which belong in one place each. **Claude Code therefore
receives the shared, durable rules primarily through `CLAUDE.md`'s import
of `AGENTS.md`**, not by reading `AGENTS.md` as a separate automatic entry
point.

**Codex does not use `CLAUDE.md` as its automatic entry point.** Codex's
automatic entry point is `AGENTS.md` alone; `CLAUDE.md`'s Claude-specific
content (hooks, subagents, model routing) is not read by Codex through any
automatic mechanism.

Human readers begin with `README.md`, not with either agent entry point.

## Responsibilities by piece

| Piece | Responsibility | Status |
|---|---|---|
| `docs/project-state.md` | Current truth: what is done, what is next | Implemented |
| `docs/architecture.md`, `docs/*-contract.md` (tier 2, normative) | Load-bearing numerical/interface/architecture rules for the language boundary, one engine, or one study | Implemented |
| `docs/tasks/active/*.md` | Scope, gates, and stop conditions for one in-flight task | Implemented (one active task; completed specs stay in place marked `Completed`) |
| `.claude/skills/` | Packaged, invokable multi-step procedures | Not implemented — planned only, see "Future PR2/PR3 plan" |
| `.claude/hooks/check-edited-file.py` | Deterministic post-edit syntax/lint/format check | Implemented |
| `.githooks/pre-commit` | Deterministic local gate before commit | Implemented, but not activated by default (see "Clang-format baseline problem" and `CONTRIBUTING.md`) |
| `.claude/agents/code-reviewer.md`, `.claude/agents/numerical-reviewer.md` | Independent, read-only, single-session review | Implemented |
| A cheap read-only triage subagent (proposed name: `repo-status`) | Cheap situational-awareness lookups | Not implemented — proposed only, unvalidated |

## Preliminary subagent review versus fresh independent approval

`code-reviewer` and `numerical-reviewer` run inside a session that already
has the implementer's context available to it in the same conversation (the
subagent's own context is clean, but the human/agent reading its findings is
not). Their findings are **preliminary**: real, useful, and required by the
workflow for non-trivial and numerical changes — but not the independent
approval gate.

**Material approval** — accepting a numerical result, a label policy, or a
milestone as done — requires a fresh top-level session: a new conversation
with no prior anchoring on the implementer's reasoning, that reads
`AGENTS.md`, `docs/project-state.md`, the relevant contract, and the actual
diff/evidence cold. Do not describe a `code-reviewer` or `numerical-reviewer`
run as that approval. This distinction is stated in `AGENTS.md` and repeated
here because it is easy to elide under time pressure.

## Model-routing recommendation

| Role | Model | Rationale |
|---|---|---|
| Main session: numerical design and synthesis | Opus / frontier-class | Architecture and cross-file numerical decisions benefit from the strongest available reasoning; this is also the only role with Write/Edit |
| Main session: bounded implementation | Sonnet | Sufficient for a smallest-coherent-change once the design is fixed |
| `code-reviewer`, `numerical-reviewer` | Sonnet | Read-only, single-diff scope keeps cost bounded; needs real reasoning, not just pattern matching |
| `repo-status` (proposed) | Haiku | Cheap, high-frequency, low-stakes lookups only — **unvalidated**: whether Haiku is accurate enough on this repository's dense numerical vocabulary has not been tested |
| Material approval | A fresh top-level session (any sufficiently capable model, human-directed) | Independence from the implementing session matters more than which model runs it |

## Auto-memory is never authoritative project state

Some assistant tools offer a persistent "memory" feature that survives
across separate conversations. Whatever such a feature records is a
convenience cache, never a source of truth, for this repository. If a
memory entry and `docs/project-state.md` disagree, `project-state.md` is
correct and the memory entry should be treated as stale. This matters
specifically because a memory feature can silently keep asserting a fact
(a task's status, a threshold, a "current next step") long after the
repository has moved on, with no mechanism forcing it to reconcile against
the living document.

## Hooks enforce deterministic rules only

`.claude/hooks/check-edited-file.py` checks JSON/TOML parse validity; runs
`ruff check` on an edited Python file when `ruff` is importable, falling back
to `python -m py_compile` (syntax-only) when it is not; and runs
`clang-format --dry-run --Werror` on an edited C++ file when `clang-format`
is importable. It never rewrites a file and never evaluates numerical,
design, or research correctness. A clean hook result means "this file parses
and is syntactically/format-clean with whichever tools were actually
available," nothing more — and specifically not "linted" if Ruff was absent
and the fallback ran instead.

## Long jobs remain manual

No hook, skill, or subagent may launch, schedule, or gate on a long
numerical study (CRR convergence, LSM cross-check, PDE label-policy pilots,
market ingestion/reconstruction). See `AGENTS.md`, "No agent-supervised
expensive numerical runs," and `docs/project-state.md`, "Operational
constraints."

## Cross-tool skill sharing requires a prototype

Claude Code discovers skills from `.claude/skills/*/SKILL.md`. Whether
Codex, or any other AGENTS.md-reading tool, discovers or invokes that same
directory is **not established** — there is no documented evidence either
way as of this writing, and none has been tested in this repository. Do not
build a shared-skills workflow on the assumption that it works across tools
without first prototyping it (create one trivial skill, open a session in
the other tool, observe whether it is discovered). `AGENTS.md` is the
verified cross-tool mechanism today; a shared skills directory is not.

## Clang-format baseline problem

`clang-format` appears in exactly **two conditional local mechanisms** —
`.claude/hooks/check-edited-file.py` (per edited file) and `scripts/check.sh`
(repository-wide, in both `--quick` and full mode) — and is **absent from
GitHub CI**: `.github/workflows/ci.yml` does not run `clang-format` in any
job. Neither local mechanism is unconditional: each runs `clang-format` only
`if` the binary is importable on the machine running it, and silently does
nothing otherwise.

There is also a second, independent problem beyond availability: **the
current clang-format-18 baseline is already non-green**. Running
`clang-format --dry-run --Werror` over `cpp`/`bindings` with clang-format
18.1.3 flags real formatting violations against the checked-in sources
today, on a machine where the tool happens to be installed. That means
"clang-format ran and passed" cannot currently be true for a from-scratch
run against this baseline with that version — only a subset of files or an
older/pinned formatter version might pass, and neither should be assumed
without checking.

Consequences that must not be misstated:

- **`./scripts/check.sh` exiting zero is not evidence that C++ formatting
  was checked**, unless `clang-format` was actually installed and actually
  ran in that invocation — and even then, a genuinely clean run is not
  guaranteed given the baseline problem above. Report the gate's result
  honestly: state whether `clang-format` was present and, if it ran,
  whether it actually reported clean.
- CI currently provides no formatting backstop at all. A misformatted C++
  file can merge cleanly.
- Closing this gap (adding a CI formatting job, reformatting the baseline to
  a pinned clang-format version, or explicitly documenting formatting as
  best-effort/local-only everywhere) is deferred to a future PR; it is not
  done by this documentation restructuring, and no hook or script is changed
  here.

## Future PR2/PR3 plan (planned, not implemented)

A prior audit proposed closing the enforcement gaps above and adding a small
set of skills/subagents in two further PRs. Two originally-planned PR2 items
are **already done** as of this documentation change and are not repeated
here: documenting `.githooks/pre-commit` activation (see `CONTRIBUTING.md`,
"Enable local git hooks") and narrowing `.gitignore` for local-only Claude
settings (`.claude/*.local.json`). Neither one is CI or hook automation —
both are still doc-only fixes to a pre-existing gap, not evidence that the
hook itself was changed. What remains genuinely open:

- **PR2 (planned):**
  - decide, and then make true everywhere, whether `clang-format` is
    CI-enforced or explicitly local-only, and resolve the current
    clang-format-18 non-green baseline (reformat to a pinned version, or
    document the discrepancy explicitly) rather than leaving it ambiguous;
  - add tests for missing-Ruff behavior in
    `.claude/hooks/check-edited-file.py` (the `py_compile` fallback path
    actually triggers and behaves as documented when `ruff` is absent);
  - add tests for `scripts/check.sh --quick` behavior when `build/check`
    does not yet exist (it must exit 0 having run no C++ test, not silently
    claim success it didn't earn, and this must be asserted, not assumed);
  - a baseline-aware formatting policy: whatever CI/local split is chosen
    above, it must state which clang-format version is authoritative and
    how a contributor reconciles a local non-green baseline with it;
  - a documentation-map validator that verifies every document enumerated
    in [documentation-map.md](documentation-map.md)'s classes — in
    particular every normative contract, `docs/architecture.md` included —
    actually exists on disk, so a renamed or deleted file cannot silently
    fall out of the hierarchy. **Not implemented here**; this is an
    acceptance item for PR2, not a validator shipped by this change.
- **PR3 (planned):** add `.claude/skills/` (a small, bounded set — no more
  than four) wrapping existing `scripts/*` commands with no new logic, and
  only after the cross-tool prototype above, decide whether they can be
  shared with other tools or are Claude Code-only; add the `repo-status`
  subagent only after validating it is actually useful at Haiku's
  capability level.

Until those PRs land, treat all `.claude/skills/` content, CI clang-format
enforcement, the documentation-map validator, and the `repo-status`
subagent as **not existing**, regardless of what any narrative document
says. This documentation change modifies no hook and no script; every
behavior described above (Ruff/clang-format conditionality, `--quick`'s
`build/check` dependency) is a description of pre-existing behavior, not a
change made here.
