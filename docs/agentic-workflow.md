# Agentic development workflow

The repository demonstrates agent-assisted engineering through constrained,
inspectable components rather than autonomous changes with broad
permissions. This page is human-facing explanation; the operative rules it
describes live in [../AGENTS.md](../AGENTS.md) (cross-tool) and
[../CLAUDE.md](../CLAUDE.md) (Claude Code-specific), and the intended
architecture — including what is planned but not yet built — is in
[agent-system.md](agent-system.md). If this page and either of those
disagree, they win; see [documentation-map.md](documentation-map.md).

## Components

| Component | Role | Authority |
|---|---|---|
| `AGENTS.md` | Durable, cross-tool operating contract | Guidance |
| `CLAUDE.md` | Claude Code-specific layer: hooks, subagents, model routing, command routing | Guidance |
| post-edit hook | Fast syntax/config/format checks after edits | Read/check only |
| `code-reviewer` | Independent software review of a diff | Read-only, preliminary |
| `numerical-reviewer` | Independent quant/research-validity audit | Read-only, preliminary |
| pre-commit hook | Deterministic local checks (opt-in per clone) | Read/check only |
| GitHub Actions | Reproducible clean-environment gate | Read/check only |

The review agents use separate contexts to reduce anchoring on the
implementer's reasoning. They return findings to the main agent; they never
edit the files they review. Their findings are **preliminary** — see
"Guardrails" below and [agent-system.md](agent-system.md) for what
independent, material approval actually requires.

## The loop, and where its steps live

The step-by-step workflow (read the contract, state the assumption, make
the smallest change, test, gate, review, resolve, update project state) is
specified once, in [../CLAUDE.md](../CLAUDE.md), "Required workflow." It is
not repeated here to avoid a second copy that can drift.

Example prompts, illustrating the shape of the loop rather than its exact
steps:

```text
Implement the stage-1 dataset schema. Follow AGENTS.md and CLAUDE.md, write
tests, and stop after the local gate passes.
```

```text
Use code-reviewer to inspect the current diff. Return findings only and do not
modify files.
```

```text
Use numerical-reviewer to audit label generation, splitting, Greek scaling,
and acceptance metrics. Do not modify files.
```

## Guardrails

- Hooks do not auto-format or silently rewrite an edit.
- Review agents lack Write/Edit tools and use plan permission mode.
- Review output must explain the mechanism and regression test, not merely
  assign a severity.
- **A review subagent's findings are preliminary, not the independent
  approval gate.** Material approval of a numerical result, a label policy,
  or a milestone requires a fresh top-level session with no prior anchoring
  on the implementer's reasoning — see [agent-system.md](agent-system.md).
- CI remains authoritative because local agent environments differ — though
  note CI does not currently check C++ formatting at all
  ([agent-system.md](agent-system.md), "Clang-format baseline problem").
- Humans own model assumptions, data rights, risk limits, and claims made
  from experiments.
- Agent-generated code receives the same tests and review as human-generated
  code.
