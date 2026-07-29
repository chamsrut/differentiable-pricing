# Agentic development workflow

The repository demonstrates agent-assisted engineering through constrained,
inspectable components rather than autonomous changes with broad permissions.

## Components

| Component | Role | Authority |
|---|---|---|
| `CLAUDE.md` | Shared architecture, numerical, test, and Git contract | Guidance |
| post-edit hook | Fast syntax/config/format checks after edits | Read/check only |
| `code-reviewer` | Independent software review of a diff | Read-only |
| `numerical-reviewer` | Independent quant/research-validity audit | Read-only |
| pre-commit hook | Deterministic local checks | Read/check only |
| GitHub Actions | Reproducible clean-environment gate | Read/check only |

The review agents use separate contexts to reduce anchoring on the
implementer's reasoning. They return findings to the main agent; they never
edit the files they review.

## Suggested Claude Code loop

1. Ask the main agent for one bounded issue and an explicit acceptance test.
2. Let the deterministic post-edit hook catch immediate failures.
3. Run the focused test and `./scripts/check.sh`.
4. Invoke `code-reviewer` on the current diff.
5. For numerical or ML work, invoke `numerical-reviewer` independently.
6. Resolve findings in the main context.
7. Rerun checks, inspect the diff, and commit.

Example prompts:

```text
Implement the stage-1 dataset schema. Follow CLAUDE.md, write tests, and stop
after the local gate passes.
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
- CI remains authoritative because local agent environments differ.
- Humans own model assumptions, data rights, risk limits, and claims made from
  experiments.
- Agent-generated code receives the same tests and review as human-generated
  code.

## What to show in an interview

Demonstrate a small issue from specification to test, implementation, clean
review, and CI. The useful story is not “agents wrote the repository”; it is
that you designed a workflow where separate implementation, review, and
deterministic validation roles leave auditable evidence.
