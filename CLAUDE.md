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
snapshots, verifies that all checked-in figures match deterministic renders,
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

Machine-specific CRR benchmarks belong under ignored `artifacts/` and must
record compiler/build, affinity, thread count, warm-up, repetitions, and batch
size. They are evidence, not portable pass/fail gates.

The `train` extra includes the `data` dependencies plus PyTorch. It is required
by every test under `python/tests/ml/`, so the full local gate needs it.

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
