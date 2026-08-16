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

Run the controlled task 9C-B PDE label-policy pilot only when its numerical
evidence is requested. It is expensive and is not part of CI. The second
command recomputes every price but reuses the first run's measured timing block
so all deterministic JSON/CSV outputs can be checked byte for byte:

```bash
python -m differentiable_pricing.american.pde_label_policy \
  --config configs/pde_label_policy_pilot_v1.toml \
  --output-directory artifacts/pde-label-policy-pilot-v1-run1
python -m differentiable_pricing.american.pde_label_policy \
  --config configs/pde_label_policy_pilot_v1.toml \
  --output-directory artifacts/pde-label-policy-pilot-v1-run2 \
  --performance-source artifacts/pde-label-policy-pilot-v1-run1/report.json \
  --verify-identical-to artifacts/pde-label-policy-pilot-v1-run1
```

The fixed regular-case gates live in the versioned config. Stress cases are
always retained and are descriptive; they never loosen or decide those gates.
The runner may select the cheapest passing candidate in its declared order or
emit `no_policy_selected`. Its 8/16-worker projections assume ideal scaling and
are not a production-feasibility claim.

Task 9C-C1 added the valuation-time surface beside the scalar price, in the same
`_pde` extension: `dp::finite_difference_valuation_surface` and
`pde_valuation_surface` solve once and return the whole slice with nodewise
price, delta, gamma, exercise classification and Greek eligibility, plus many
requested spots evaluated against that one solve. The scalar API, its results
and its O(N_S) working memory are unchanged, and a surface query at the scalar
spot reproduces the scalar price bitwise. The derivative, classification and
eligibility rules are in `docs/pde-numerical-contract.md` and are load-bearing:
Greeks come from the solved slice and never from a bump, the exercise
classification is certified against the solver's own LCP residual scale rather
than a decimal tolerance, a node whose regime that scale cannot certify is
refused as a Greek label, and the five-node regime stencil and domain-edge
buffer are fixed rules, not tuning knobs. A query's delta and gamma are
interpolated nodewise discrete Greeks, never derivatives of the query price
interpolant. A tiny non-CI demonstration:

```bash
python scripts/demo_pde_valuation_surface.py
```

It is a correctness demonstration, not a throughput claim.

Task 9C-C2a turns those surfaces into dataset rows without pretending they are
independent, in `differentiable_pricing.american.pde_surface_harvest` with the
versioned design `configs/pde_surface_harvest_demo_v1.toml`. It is **exploratory
infrastructure**, not a production dataset. Its rules are load-bearing and are
in `docs/pde-numerical-contract.md`: three versioned SHA-256 canonical
identities over sorted-key JSON with `.17g` floats and negative zero normalized
to positive zero (`partition_group_id` for the base non-spot state, which
deliberately excludes spot, node index, option type, exercise style, bump role,
grid and the reporting-only `contract_multiplier`; `solver_input_id`, also used
as `surface_id`, for the complete actual pricing/numerical call while excluding
group, role, name and multiplier provenance; `row_id` for one exact node);
actual-solve aliases are rejected before assignment, and `plan_harvest` takes
**no solver argument**, so
partition assignment into `train`/`validation`/`interpolation_test` provably
precedes every solve and no group can straddle a partition; duplicate economic
states fail before the first solve; harvesting uses **exact grid nodes only**,
never the off-grid query API, inside a predeclared interior window and outside
the structural boundary buffer; `exercise_state` and Greek eligibility are
copied, never relaxed, so a `numerically_indifferent` node keeps its price row
and can never become an eligible Greek label; per-regime quotas are applied only
after assignment, by a predeclared outcome-independent rule, with every
shortfall reported; and raw rows, independent **design** groups and planned,
attempted, successful, failed, retained and discarded surface counts are
reported separately, with `raw_rows_per_group` (dataset expansion) and
`raw_rows_per_attempted_surface_solve` (numerical-work reuse)
published side by side, each carrying its own integer numerator and denominator.
Rows per group is never the computational multiplier: in the shipped design each
group emits four surfaces, so the two differ by a factor of four.

```bash
python scripts/demo_pde_surface_harvest.py

python -m differentiable_pricing.american.pde_surface_harvest \
  --config configs/pde_surface_harvest_demo_v1.toml \
  --output-directory artifacts/pde-surface-harvest-demo-v1
```

The group count is a design count and is never to be called a statistical
effective sample size. Vega surfaces, parallel execution, production dataset
generation and label policy v2 are still **not** implemented, and task 9C-B
remains `no_policy_selected`: its frozen evidence is never edited, rerun or
reinterpreted.

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

Run the task 9B market-state reconstruction only when the task 9A processed
tree and report are present. It verifies the 9A config, report and raw-manifest
digests before any arithmetic and aborts with nothing written on a mismatch,
reads only the processed Parquet, and writes deterministic, atomic output
beneath the ignored `artifacts/` tree:

```bash
python -m differentiable_pricing.market.reconstruct \
  --config configs/market_state_reconstruction_v1.toml
```

The method, conventions, forbidden names and the task 9C input interface — now
implemented by the task 9C-A solver, still fed only by synthetic inputs — are in
`docs/market-state-reconstruction-contract.md`. Load-bearing rules:

- **The slope sign is mandatory.** Parity gives `y = C - P = a + bK` with
  `b = -D`, so a non-negative fitted slope is reported as a contradiction with
  the offending value intact. Nothing is clamped: `D <= 0`, `F <= 0`, non-finite
  results, singular or ill-conditioned designs and insufficient pairs each get
  their own named diagnostic, and ill conditioning warns rather than invalidates.
- **Bid-ask widths are executable-liquidity measures, never standard errors.**
  The weighted fit uses the inverse square of the combined call/put width; no
  standard error, confidence interval or p-value is derived anywhere.
- **The SPY quantity has exactly one name**: the American parity carry residual,
  `Q = S - D * median_i F_tilde_i`. It is never an observed dividend, an exact
  dividend present value, a borrow rate, an exact SPY forward, or an independent
  market input. Those five names are declared in `[american] forbidden_names` and
  the config parser rejects a configuration that drops one.
- **Every setting is `exploratory_pilot`**, chosen after the archive was
  observed. All three strike windows, both fit methods and all three snapshots
  are reported side by side; the spread across them is a result and no axis may
  be selected as best after seeing results.
- The expiry instant is **assumed**: the archive's definition records carry a
  date at midnight UTC and no settlement time, style or multiplier. Zero-DTE and
  sub-day expiries are fitted but their rate is withheld.
- The SPY ex-date is `officially_scheduled` from the issuer's published SPDR
  distribution schedule, cited by title and URL in
  `[american.ex_date_schedule]`. That fixes the scheduled dates and nothing
  else: the parser requires `amount_status` and `economic_effect_status` to stay
  `not_verified`, no dividend amount may be described as observed or inferred,
  and a carry-residual step coinciding with the ex-date is reported as a
  temporal alignment, never as a measurement of the distribution.
- `pointwise_identified_curve_unconstructed` is not a softer
  `jointly_identifiable_only`. Parity identifies D(T) and F(T) at each quoted
  expiry on their own; the exploratory stability threshold may change the
  stability statement and never the pointwise identifiability statement, and the
  class also records that only knots exist, with no curve interpolated through
  them. `jointly_identifiable_only` stays for genuinely confounded quantities
  such as SPY dividends versus effective borrow/carry.
- Rate-control comparison is descriptive only. No OIS curve is bootstrapped, and
  a Treasury par yield and an SR3 futures rate are neither zero rates nor each
  other.

The generated report, tables and figures are derived from proprietary
quote-level data. Never stage them, nor any fitted curve or inferred market
value; the tests use synthetic fixtures only, so CI runs the whole suite without
the archive.

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
