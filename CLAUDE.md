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

Run the full local gate. It compiles Python and hook sources, validates the
TOML and JSON configuration, runs Ruff and clang-format, builds and runs the
C++ tests, and finishes with the full Python suite (`python3 -m pytest -q`).
It therefore needs the editable install below; a missing pytest fails the gate
rather than skipping it. `--quick` (used by the pre-commit hook) stops after
the C++ tests and does not run the Python suite.

```bash
./scripts/check.sh
```

Build only the C++ core:

```bash
cmake -S . -B build/dev -DDP_BUILD_PYTHON_BINDINGS=OFF -DDP_WARNINGS_AS_ERRORS=ON
cmake --build build/dev --parallel
ctest --test-dir build/dev --output-on-failure
```

Build and test the Python extension:

```bash
python -m pip install -e '.[dev,data]'
pytest -q
ruff check .
```

The `data` extra (NumPy and PyArrow) is required by the dataset tests, so the
full gate needs it too.

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
