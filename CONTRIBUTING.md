# Contributing

Start with [AGENTS.md](AGENTS.md), [CLAUDE.md](CLAUDE.md) (if you are using
Claude Code), [docs/project-state.md](docs/project-state.md), and
[docs/research-contract.md](docs/research-contract.md). Every change should
state its model or software assumption, add a regression test, and pass:

```bash
./scripts/check.sh
```

The full gate ends with `python3 -m pytest -q`, so install the editable
package first with `python -m pip install -e '.[dev,train]'`. A missing pytest
or missing extension fails the gate instead of skipping the Python suite.
`./scripts/check.sh` runs `ruff check` only when `ruff` is importable and
`clang-format --dry-run --Werror` only when `clang-format` is installed on
your machine — neither is unconditional, and CI does not run `clang-format`
at all today; do not report a passing gate as evidence that C++ formatting
was checked, or that linting ran, unless the relevant tool genuinely ran.
The script exits on its first failing command, so the C++ and Python test
stages are reached only once every preceding enabled check (config/snapshot
validation, Ruff, clang-format) has passed. `--quick`, used by the
pre-commit hook, skips the Python suite and runs the C++ tests **only if
`build/check` already exists** from a prior full run — on a fresh clone
without it, `--quick` may perform no C++ test execution at all. Run the full
gate before opening a pull request; **GitHub Actions CI remains the
authoritative clean-environment gate**, since local results depend on
machine state.

## Documentation hierarchy

When two documents disagree, the higher tier is correct: code/config/frozen
evidence, then normative contracts — [docs/architecture.md](docs/architecture.md)
and the applicable `docs/*-contract.md` files — then
[docs/decision-log.md](docs/decision-log.md), then
[docs/project-state.md](docs/project-state.md) and the active task under
`docs/tasks/active/`, then README and other narrative documentation. Full
rules and each document's edit policy:
[docs/documentation-map.md](docs/documentation-map.md). Update
`docs/project-state.md` and add a `docs/decision-log.md` entry in the same
change whenever a milestone starts, changes scope, or completes.

## Enable local git hooks

Git hooks are versioned under `.githooks/` but are **not** activated
automatically by cloning the repository — Git does not enable a non-default
`core.hooksPath` on its own. Run this once per clone to enable the local
pre-commit gate:

```bash
./scripts/install-git-hooks.sh
```

Without it, `.githooks/pre-commit` simply never runs, and CI is your only
gate.

Tests are split by directory so CI can keep one job PyTorch-free:
`python/tests/ml/` holds every PyTorch-dependent test, and everything else
under `python/tests/` must import only the `data` extra. Add new
PyTorch-dependent tests under `python/tests/ml/`;
`scripts/check_test_partition.py` fails the gate if a test outside that
directory reaches `torch` or `differentiable_pricing.ml`.

Expensive study reports stay ignored under `artifacts/`. What gets committed is
a compact result snapshot under `docs/results/` plus deterministic figures under
`docs/figures/`. Never commit a raw report. To refresh the American LSM
cross-check evidence after a reviewed rerun:

```bash
python scripts/freeze_american_lsm_results.py \
  --report artifacts/american-lsm-crosscheck-review-fixed-v1.json \
  --output docs/results/american_lsm_crosscheck_results_v1.json --update
python scripts/plot_american_lsm_results.py
```

The gate and CI then enforce both with checked-in files only:

```bash
python scripts/freeze_american_lsm_results.py --check
python scripts/plot_american_lsm_results.py --check
```

`docs/results/american_pde_label_policy_results_v1.json` is **immutable
historical evidence**: the task 9C-B report itself, checked in unchanged and
pinned by SHA-256 in `python/tests/test_pde_label_policy_results_snapshot.py`.
It must **never** be replaced, refreshed, rerun, or reinterpreted, by hand or
by script — not even to reflect a later engine change (see
[docs/decision-log.md](docs/decision-log.md) DEC-003). Its snapshot test
reconciles only the configuration and runner digests it records against
current files; that check validates **preservation of historical evidence**,
not currency with HEAD, and it deliberately does not reconcile the PDE
source digests it also records, because those are frozen provenance of the
run that produced this file, not a claim about today's engine. A new
numerical-policy question — including task 9C-C3 — requires its own,
separately versioned config, runner, and result snapshot; it is never
answered by editing or rerunning this one.

Task 9C-C3 is that separately versioned study, and it has its own family:
`configs/pde_label_policy_pilot_v2.toml`,
`python/src/differentiable_pricing/american/pde_label_policy_v2.py`, and
`scripts/freeze_pde_label_policy_v2_results.py`, whose snapshot would live at
`docs/results/american_pde_label_policy_v2_results_v1.json`. **That snapshot
does not exist yet** — neither v2 stage has been run — so
`freeze_pde_label_policy_v2_results.py --check` currently fails by design, and
the tool is deliberately **not** wired into `scripts/check.sh` or CI. Wire it
in only in the change that first commits a v2 snapshot.

That tool always names its mode: `--extract --report R --output S` to distil a
reviewed terminal report, `--check --output S` to enforce a checked-in one.
Neither mode trusts what the report or snapshot claims: both recompute every
per-case verdict, Greek eligibility, aggregate count and lifecycle field from
the raw per-solve numbers against the checked-in configuration, and reconcile
every executable-source digest against the repository. Neither re-solves, so
neither can authenticate a fully coordinated fabricated numerical report — a
limitation the snapshot states itself. Both v2 stages are manual,
terminal-invoked jobs; see
[docs/pde-numerical-contract.md](docs/pde-numerical-contract.md), "Task 9C-C3:
label-policy v2".

## Markdown and math

Documentation is read on GitHub, which renders math only with dollar
delimiters. Use `$...$` inline and `$$` fences on their own lines for display;
never `\(...\)` or `\[...\]`, which GitHub prints as literal backslashes. Keep
a blank line before and after every display block, keep an inline expression on
a single line — a `$...$` broken across a line break does not render — and keep
shell commands, file names, and code literals in backticks or fenced blocks so
no stray `$` is parsed as math. Do not indent a continuation line inside a `$$`
block, which Markdown can read as code. Stay inside core KaTeX: prefer
`\mathrm{...}` to `\operatorname{...}`, and never use `\DeclareMathOperator`,
`\newcommand`, `\require`, or `\label`, which GitHub does not define.

Pull requests should include:

- the problem and numerical assumptions;
- tests and commands run;
- price, Greek, convergence, or latency evidence as applicable;
- known limits and out-of-domain behaviour;
- independent code review and, for quant changes, numerical review.

Do not commit proprietary market data, licensed datasets, credentials, large
generated labels, or model checkpoints.
