# CLAUDE.md

## Mission

Build a defensible research platform for classical and neural derivative
pricing. Prefer a small, correct experiment with hard validation gates over a
broad demo with unverified claims.

## Architecture

- `cpp/include`, `cpp/src`: C++20 pricing and differentiable inference.
- `cpp/tests`: dependency-free numerical unit tests.
- `bindings/python`: narrow pybind11 boundary.
- `python/src`: data generation, training, evaluation, and export.
- `python/tests`: cross-language and pipeline tests.
- `configs`: versioned assumptions and experiment settings.
- `docs/research-contract.md`: hypotheses, controls, and acceptance criteria.

C++ owns production-style pricing and inference. Python owns research
orchestration and neural-network training. Do not duplicate pricing formulas
in Python merely to make a test pass.

## Commands

Run the full local gate. It compiles Python, script, and hook sources, validates
the TOML and JSON configuration, checks that the CI test partition is still a
partition, verifies the locked European replication protocol and its pinned
file hashes, validates both the development and completed-replication result
snapshots, validates the frozen American LSM cross-check snapshot and its
repository provenance, verifies that all checked-in figures match
deterministic renders,
runs Ruff and clang-format, builds and runs the C++ tests, and finishes with
the full Python suite (`python3 -m pytest -q`, both partitions). It therefore
needs the editable install below; a missing pytest fails the gate rather than
skipping it. `--quick` (used by the pre-commit hook) stops after the C++ tests
and does not run the Python suite.

```bash
./scripts/check.sh
```

The frozen European replication is executed only through
`scripts/run_european_replication.py`. `run-to-validation` must be run from a
clean, synchronized `main`; `final-evaluate` is a distinct, explicit, one-shot
transition and is never invoked by tests or CI. Protocol v1 is now terminal:
its final partition was consumed exactly once and its immutable outcome lives
in `docs/results/european_replication_results_v1.json`. Never rerun or tune
against that partition.

Build only the C++ core:

```bash
cmake -S . -B build/dev -DDP_BUILD_PYTHON_BINDINGS=OFF -DDP_WARNINGS_AS_ERRORS=ON
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure
```

Build and run the full Python suite:

```bash
python -m pip install -e '.[dev,train]'
pytest -q
ruff check .
```

Run the exploratory American CRR convergence study only when numerical
evidence is requested; it is intentionally not part of CI because its
8,192/8,193-step reference grid is expensive:

```bash
python -m differentiable_pricing.american.convergence \
  --config configs/american_crr_convergence_v1.toml \
  --output artifacts/american-crr-convergence-v1.json
```

Run the independent LSM path/step/basis cross-check only when numerical
evidence is requested; its production-sized path experiments are likewise not
part of CI:

```bash
python -m differentiable_pricing.american.lsm_crosscheck \
  --config configs/american_lsm_crosscheck_v1.toml \
  --output artifacts/american-lsm-crosscheck-v1.json
```

The reviewed cross-check evidence is frozen in
`docs/results/american_lsm_crosscheck_results_v1.json`. Raw reports stay
ignored under `artifacts/` and are never committed. Regenerate the snapshot and
its figures only from a reviewed report, then let the gate enforce them:

```bash
python scripts/freeze_american_lsm_results.py \
  --report artifacts/american-lsm-crosscheck-review-fixed-v1.json \
  --output docs/results/american_lsm_crosscheck_results_v1.json --update
python scripts/plot_american_lsm_results.py
python scripts/freeze_american_lsm_results.py --check
python scripts/plot_american_lsm_results.py --check
```

Both `--check` modes run in the local gate and in CI and must keep working
with `artifacts/` absent: they validate the checked-in snapshot and reconcile
its configuration and C++ provenance digests against current repository files.
Never make CI depend on an ignored artifact. That study selects no production
label policy; task 8E predeclares and runs the CRR label-policy calibration
study separately.

Machine-specific CRR benchmarks belong under ignored `artifacts/` and must
record compiler/build, affinity, thread count, warm-up, repetitions, and batch
size. They are evidence, not portable pass/fail gates.

Run the real-market feasibility audit only when the private archive is
present. It verifies `manifests/sha256sums.txt` before decoding anything,
consults the vendor's per-request `condition.json`, reads every `.dbn.zst` in
bounded chunks through the vendor reader, fails loudly on a truncated payload,
never decompresses or modifies a raw file, and writes normalized partitioned
Parquet atomically beneath the ignored `data/processed/` tree:

```bash
python -m pip install -e '.[dev,market]'
python -m differentiable_pricing.market.ingest \
  --config configs/market_feasibility_v1.toml \
  --output artifacts/market-feasibility-v1-audit3/feasibility-report-v3.json
```

The archive under `data/market-feasibility-v1/` is proprietary quote-level
data. It, and every Parquet file derived from it, stays out of Git. CI never
sees it: `python/tests/market/` builds DBN fixtures programmatically with the
vendor encoder and skips the decode-backed tests when the `market` extra is
absent, so the lightweight `.[dev,data]` job runs the whole suite unchanged.

The audit reports; it does not price, fit, or fill gaps. Three rules keep it
from smuggling a conclusion into its own source:

- **Absent inputs are probed, never assumed.** Every external input the study
  needs is declared in `[[declared_sources]]` as a path and a kind, and the
  audit observes `absent`, `present_but_empty` or `present`. Those are distinct
  facts with distinct remedies, and an empty directory is invisible to a
  SHA-256 manifest, so it has to be probed for. Nothing about dividends,
  borrow, curves or corporate actions is a constant in the code.
- **Capabilities are scoped, not one verdict.** `archive_integrity`,
  `ingestion_feasibility`, `xsp_european_surface_readiness`,
  `spy_american_calibration_readiness` and `replication_readiness` are reported
  separately, each from recorded checks that state the observation deciding
  them.
- **Coverage denominators are the resolved universe, within one population.**
  "Any instrument quoted in this minute" is retained as a descriptive liveness
  metric and is never an acceptance gate; contract-minute coverage, per-contract
  coverage and per-contract synchronization distributions are what the
  capabilities use, and every distribution spans the whole resolved universe
  including contracts that never traded. A product whose payload mixes
  instrument kinds is accumulated per population, so a futures block's
  numerator, denominator and spread distribution are all outrights and calendar
  spreads are reported beside them, never inside them. A strike counts as paired
  only when its call and put are tradable in the *same minute*. Tradability is
  config-versioned in `[tradability]` and requires finite positive sizes on both
  sides; several exploratory relative-spread thresholds are reported and no
  capability check consults any of them.

Capability minima carry their own provenance and it is not uniform. The
structural surface minima are properties of the method. The session-count
minima in `[capabilities]` were chosen after this three-session archive was
observed: they are `provisional_pilot_target`, not desk-grade, and must be
justified against the intended estimator and flipped to `frozen` before any
pilot partition is consumed. Do not write a blanket claim that no threshold in
this audit followed the results.

Put-call parity fitting, implied forwards, dividend inference and forward
moneyness are **task 9B** and are explicitly out of scope here. The archive
carries no independent dividend ground truth, so no artefact of this audit may
describe a dividend as observed or inferred.

The `train` extra includes the `data` dependencies plus PyTorch. It is required
by every test under `python/tests/ml/`, so the full local gate needs it. The
`market` extra adds the `databento-dbn` reader and is optional; only the real
ingestion pipeline needs it.

The suite is split by directory, and CI mirrors that split exactly:

- `python/tests/` (excluding `ml/`) must import only the `data` extra. The
  lightweight CI job installs `.[dev,data]` and runs
  `pytest -q --ignore=python/tests/ml`.
- `python/tests/ml/` holds every PyTorch-dependent test. The CPU-only CI job
  installs `.[dev,train]` and runs `pytest -q python/tests/ml`.

The two commands partition the suite, so every test file is exercised by
exactly one job. `scripts/check_test_partition.py` enforces this statically —
it fails if a module outside `python/tests/ml/` imports `torch` or
`differentiable_pricing.ml`, which would otherwise break collection in the
lightweight job. It runs both in the local gate and in CI. Put new
PyTorch-dependent tests under `python/tests/ml/`; no marker registration is
needed, and no enumerated file list has to be kept in sync.

## Numerical non-negotiables

- State units, conventions, time basis, curve construction, and model
  assumptions explicitly.
- Validate every input at public C++ boundaries; reject NaN and infinities.
- Use `double` / float64 for reference prices, labels, and reported Greeks.
- Never call a network derivative an exact Greek. It is exact for the
  surrogate; validate it against a trusted reference after undoing feature and
  target scaling.
- Compare prices with analytic identities where available. Compare Greeks with
  analytic, algorithmic-differentiation, or convergence-tested bump references.
- Monte Carlo outputs require a seed policy, confidence interval or standard
  error, and a convergence study.
- For LSM, fit the stopping rule on paths independent of final valuation paths.
- Prevent train/test leakage across duplicated states, paths, curve scenarios,
  strikes, expiries, and exercise schedules.
- Use `validation` for architecture and hyperparameter selection. Require
  `interpolation_test` to be named explicitly and do not tune against it.
- Once any test result informs a model change, mark that test as consumed and
  require a fresh-seed replication for the final performance estimate.
- Change one controlled factor per experiment where practical, and keep
  predeclared acceptance gates fixed after results are observed.
- Keep financial feature transforms and physical-unit reconstruction inside
  the differentiable model used by both evaluation and artifact loading.
- Derive differential labels analytically in the declared model coordinates;
  do not use finite differences as training labels.
- Treat output constraints as versioned parts of the mathematical model.
  Record the exact projection and source-weight lineage in the artifact.
- For piecewise-differentiable constraints, test Greeks away from kinks and
  state where derivatives are not unique.
- Report tails and worst regions, not only mean error.
- Benchmark end-to-end latency at fixed batch sizes, including serialization
  and feature transforms.

## Coding rules

- Keep the public API narrow and ownership explicit.
- Prefer standard-library C++ and RAII. No unchecked raw owning pointers.
- Use row-major dense weights as documented by `DenseLayer`.
- Preserve warning-clean builds with `DP_WARNINGS_AS_ERRORS=ON`.
- Tests must fail deterministically and include negative/error cases.
- Python is typed, formatted for 100 columns, and linted with Ruff.
- Markdown math must use GitHub-compatible delimiters: `$...$` inline and `$$`
  on its own line for display, never `\(...\)` or `\[...\]`. Surround display
  blocks with blank lines, keep an inline expression on one line, and keep
  commands and code literals in backticks so no stray `$` becomes math. Stay
  inside core KaTeX: prefer `\mathrm{...}` to `\operatorname{...}`, and never
  use `\DeclareMathOperator`, `\newcommand`, `\require`, or `\label`.
- Do not commit generated datasets, model weights, virtual environments, or
  build directories.
- New dependencies require a concrete benefit and a documented reproducibility
  cost.

## Required workflow

1. Read the relevant source, test, config, and research contract.
2. State the numerical assumption being changed.
3. Make the smallest coherent change.
4. Add or update tests before claiming completion.
5. Run the narrow test, then `./scripts/check.sh`.
6. For non-trivial changes, invoke `code-reviewer`.
7. For pricing, Greeks, sampling, calibration, or ML evaluation, also invoke
   `numerical-reviewer`.
8. Resolve findings in the main context and rerun the gates. Review agents do
   not edit files.

## Review standard

Reviews are findings-first. Include severity, file and line, mechanism of
failure, proposed fix, and a regression test. Avoid style comments already
enforced mechanically. If no issues are found, say so and list residual
risks/testing gaps.

## Git safety

- Inspect `git status` and `git diff` before commits.
- Never discard changes you did not create.
- Never commit secrets, credentials, market-data licenses, or proprietary data.
- Keep commits scoped: pricing core, bindings, experiments, and documentation
  should be separable when practical.
