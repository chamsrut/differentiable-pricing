# CRR dataset admission (task 9E)

The admission record for `data/american-option-v1/`. It answers one question:

> Is the existing `data/american-option-v1/` dataset a valid, reproducible
> input to a bounded American CRR-network experiment?

It is deliberately short. The full inventory is
[data-holdings-catalogue.md](data-holdings-catalogue.md) (task 9D); the label
policy is [american-crr-contract.md](american-crr-contract.md), "Label policy
v1"; the schema contract lives in code, at
`python/src/differentiable_pricing/data/american_schema.py`.

## Decision

**Conditionally admitted, solely for learning the known continuous-yield CRR
mapping in one bounded feasibility pilot**, under the schema
`american-option-dataset/1`, subject to the limitations in "What admission does
not grant" and "Limitations that survive admission". This is not acceptance of
converged American-price accuracy.

Every implemented integrity, internal-identity, identifier-disjointness, and
exact-state-disjointness check in
`python/src/differentiable_pricing/data/american_admission.py` was run over all
250,000 rows and passed. Those checks are only part of task 9E's predeclared
gates. The independent price cross-check and semantic-coverage judgement were
not performed. The near-duplicate threshold was not predeclared before task 9D
observed the distances, so that gate cannot be retroactively satisfied for
this dataset version. No gate is weakened by this reconciliation, no row was
dropped, and the dataset was not modified, moved, or regenerated.

Fresh independent review returned the exact verdict
`APPROVE PLANNING RECONCILIATION` (DEC-035). It approves only this conditional
admission for learning the known continuous-yield CRR mapping. Task 9E's
predeclared admission remains incomplete rather than a completed all-gates
pass, and the approval grants none of the permissions or claims excluded
below.

## What admission does not grant

- **No training is authorized by it.** The first American training run is a
  separate task with its own gate; this decision makes an input available, it
  does not start an experiment.
- **No SPY or PDE training.** This dataset carries a flat continuous dividend
  yield and cannot represent a discrete cash-dividend schedule. **A continuous
  dividend yield is not a discrete SPY cash-dividend schedule**, and nothing
  learned here transfers as a claim about discrete-dividend American pricing.
- **No production label-quality claim.** The labels are synthetic CRR model
  values with lattice discretization error. The stored adjacent-step gap is a
  refinement diagnostic, not an error bound. The outstanding independent
  numerical cross-check must be completed or explicitly resolved before a
  training run is authorized.
- **No certified Greeks.** No Greek is labelled anywhere in the schema, and a
  finite difference of these prices is not a validated Greek.
- **No market claim.** These are model values, not prices anyone traded at.

## Provenance recovered

Read-only Git-object inspection of `feat/american-dataset-v1` at commit
`49ef72a`; no branch switch, no merge, no cherry-pick, and no row regenerated.

**The pricing oracle verifies from this branch.** The manifest's composite
`crr_implementation_sha256` `c69cefc8…` is, by `CMakeLists.txt`, the SHA-256 of
the colon-joined digests of `bindings/python/module.cpp`,
`cpp/include/dp/binomial_tree.hpp`, `cpp/include/dp/black_scholes.hpp`,
`cpp/include/dp/option.hpp`, `cpp/src/binomial_tree.cpp`,
`cpp/src/black_scholes.cpp` and `cpp/src/option.cpp`. Recomputed at HEAD it
reproduces exactly, as do `crr_header_sha256` `eec8f796…` and
`crr_source_sha256` `7dc5d754…`. The engine that produced these labels is
byte-identical to the CRR engine on this branch.

**Ported to this branch, verbatim, and digest-pinned by a test**
(`python/tests/test_american_dataset_schema.py::test_recovered_configuration_digest_is_unchanged`):

| File | SHA-256 | Why |
|---|---|---|
| `configs/american_option_dataset_v1.toml` | `d18485c6…` | the configuration `manifest.json` names and pins |
| `configs/american_label_policy_pilot_v1.toml` | `7858e1d6…` | the recorded label-policy failure |
| `configs/american_label_policy_pilot_v2.toml` | `c5eafa09…` | the configuration the selection report pins |

**Ported as documentation:** `docs/american-crr-contract.md`, "Label policy v1"
— the normative definition of `american-crr-adjacent-average/1`, unchanged
except for one cross-reference.

**Ported as code:** `python/src/differentiable_pricing/data/american_schema.py`
— the schema contract only: column names, order, types, nullability, the
recovered definition of every column, the sampled domain, and a strict manifest
validator. **The generation machinery was deliberately not ported.** Nothing in
this repository samples, prices, or writes a row of this schema, so **the
dataset still cannot be regenerated from tracked sources alone**; that requires
`feat/american-dataset-v1`, which is not an ancestor of `main` and is 94 files
divergent from HEAD.

**The label policy's evidence status, stated plainly.** The two pilot reports
that chose `steps = 1024` were **ignored local artifacts under `artifacts/`,
never frozen `docs/results/` snapshots**, and they remain so. Their
configurations are now tracked and their digests match what the reports record,
but the reports are not project evidence in the sense a frozen snapshot is. No
`docs/results/` snapshot exists for this label policy on any branch, and task 9E
did not invent one.

Two consequences carried forward from the recovered contract section, which
says so itself: pilot v2 was written **after** v1's results were seen and is
**not an uncontaminated predeclaration**; and re-scored under v1's original p99
gate, the selection would have been `N = 4096`, not `N = 1024`. That is a
recorded property of the label policy, not a defect discovered here.

The recovered configuration also references
`configs/american_label_policy_pilot_v1.toml` and `..._v2.toml` and a contract
section; all three now resolve on this branch. It is byte-pinned by the
manifest, so it was ported unedited.

## Reconciliation against the local bytes

Recomputed in this task over all three partitions:

- **Digests.** All three Parquet digests match `manifest.json` exactly.
- **Partitions and counts.** `train` 200,000, `validation` 25,000,
  `interpolation_test` 25,000 — 250,000 rows — matching both the manifest and
  the Parquet footers. Every row's `split` column equals its file's partition.
- **Schema.** All three files satisfy `schema.equals(TABLE_SCHEMA)` against the
  ported contract: 29 columns, exact order, exact types, every column
  non-nullable.
- **Nulls and non-finite values.** Zero, in every column of every partition.
- **Option types.** Exactly `{call, put}`.
- **Domain.** Every sampled field inside the manifest's declared domain, in
  every partition.
- **Label identities**, bitwise, zero tolerance: the American and European
  adjacent averages, the adjacent-step gap, and
  `early_exercise_premium == american_price − european_crr_price`.
- **American dominance.** `american_price ≥ european_crr_price` on every row.
- **Intrinsic bound.** Holds to rounding; worst slack `−4.83e-13` absolute,
  against a spot-scaled `1e-9` rounding allowance. The allowance is a rounding
  budget, never an error bound.
- **`log_moneyness == log(spot/strike)`.** Worst residual `2.78e-16`.
- **Continuous dividend yield.** `dividend_yield` is a flat, continuously
  compounded yield; there is no ex-date, amount, or dividend-count column, and
  the schema cannot represent one.
- **Identifier uniqueness.** `sample_id` unique within each partition.
- **Exact contract-state uniqueness.** No two rows anywhere share a bitwise
  identical `(option_type, spot, strike, maturity, rate, dividend_yield,
  volatility)`.
- **Cross-partition disjointness.** No `sample_id` and no exact contract state
  is shared by any two partitions.

The gate runner reads all three partitions to verify them. That is
verification, not consumption: no model is fitted, no metric is scored, and no
selection is made against `interpolation_test`, so the partition remains
untouched for its one future evaluation. The loader keeps the same discipline —
a caller names the split it wants, and training opens only `train` and
`validation`.

Nearest-neighbour distance between partitions stays **descriptive**. Task 9D
measured it after the fact, so there is no predeclared threshold, and none is
claimed retroactively. It is not an implemented gate, and task 9E's
predeclared near-duplicate gate cannot be satisfied retroactively for this
dataset version.

## Feature sufficiency

The dataset carries every state variable a constant-parameter American CRR
price depends on: spot, strike, expiry, volatility, rate, continuous dividend
yield, and option type. `strike` is stored directly and `log_moneyness` is
`log(spot/strike)` — spot moneyness, not forward moneyness.

The existing ML representations were inspected rather than assumed:

- **`raw_physical_v1`** — `(option_type, spot, strike, maturity, rate,
  dividend_yield, volatility)` — carries the full state.
- **`forward_normalized_v1`** — `(option_type, log(F/K), σ√T)` — **loses
  required state for American options.** It is exactly sufficient for a
  European price scaled by `spot·e^{−qT}`, and insufficient for an American
  one, because the early-exercise boundary depends on `r` and `q` separately
  and not only through the forward.

Measured with the compiled CRR engine at `S = 100`, `T = 1`, `σ = 0.30`,
`N = 512`, comparing two contracts that share an option type, `log(F/K) = 0`
and `σ√T`, so the forward-normalized representation cannot tell them apart:

| Contract pair | European, normalized | American, normalized |
|---|---|---|
| put, `(r,q) = (0.10, 0.00)` vs `(0.12, 0.02)` | agree to `8.9e-16` | `0.141922` vs `0.143785` — differ by `1.86e-3`, **1.31 %** |
| call, `(r,q) = (0.02, 0.10)` vs `(0.04, 0.12)` | agree to `2.5e-15` | `0.136077` vs `0.137677` — differ by `1.60e-3`, **1.18 %** |

The European legs agree at machine precision; the American legs differ by more
than a percent. That gap is early-exercise value that the normalization cannot
see.

**Conclusion.** The raw dataset is feature-sufficient, but the seven-input
`american_raw_physical_v1` form is not minimal. The next task uses
**`american_forward_carry_v1`** with encoded option type,
`log_forward_moneyness = log(F/K)`, `total_volatility = sigma*sqrt(T)`,
`rate_time = r*T`, and `yield_time = q*T`, and learns
`normalized_price = V/(S*exp(-q*T))`. These coordinates retain spot moneyness
because `log(S/K) = log(F/K) - rT + qT`. The measured rejection of the
three-input European `forward_normalized_v1` form remains unchanged; the new
representation extends it with the two separately required carry coordinates.
The normative representation and reconstruction contract is in
[architecture.md](architecture.md), "Phase-1 American surrogate
representation".

No network was defined, built, or trained to reach this conclusion.

## Loader and gates

`python/src/differentiable_pricing/ml/dataset.py` now selects a dataset schema
by **explicit name and version** from a registry of two. A file is read because
its manifest declares a registered schema, never because the bytes happen to
parse; an unregistered `schema_version` is a `DatasetLoadError` raised before
any Parquet file is opened. **There is no permissive compatibility fallback.**

- The **European path is unchanged**: same manifest validator, same table
  schema, same evaluation columns, same feature order, same errors. Its 137 ML
  tests and 147 European dataset tests pass untouched.
- Per-version schema validation is exact equality — a missing, extra,
  reordered, retyped, or nullable column fails.
- Manifest digest and row-count verification apply to every schema.
- Non-finite data and unknown `option_type` values are rejected on load.
- The American target (`american_price`) and its **paired European comparator**
  (`european_crr_price`) are exposed by name through `SplitData.target` and
  `SplitData.european_comparator`, so no caller has to know which column is
  which.

**Training and evaluation code was deliberately not adapted.**
`ml/train.py` and `ml/evaluate.py` still address the label as
`split.columns["price"]`, so handing them an American split raises `KeyError`
rather than reading the wrong column — a hard failure, which is the behaviour
this task wants until a training task exists. `SplitData.target` and
`SplitData.european_comparator` are what that task will use.

`python/src/differentiable_pricing/data/american_admission.py` holds the gates:
schema, manifest reconciliation, partition labels, row identities and bounds,
domain containment, identifier uniqueness, exact state uniqueness, and
cross-partition disjointness. Every gate fails closed.

Tests use small fixtures written to `tmp_path`
(`python/tests/american_admission_fixtures.py`). **No test reads the Git-ignored
47 MB dataset**, so the suite runs in CI unchanged. 75 focused tests were added:
27 for the schema contract and its manifest validator, 27 for the gates, 14 for
named-schema loading, and 7 for feature sufficiency.

One required invariant is **not implemented**: the current checks do not
require every row's `label_policy` to equal `manifest.label_policy.name`, or
every row's `label_steps` to equal `manifest.label_policy.steps`. Both
equalities are mandatory implementation work before training. The experiment
entry checks must also pin the exact dataset `generator_version` (`1.0.0`) and
the recovered configuration SHA-256
`d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98`.
This admission record does not implement those changes.

## Limitations that survive admission

1. **Not reproducible from tracked sources alone.** The generator is on an
   unmerged branch; only the schema, configurations and contract section were
   ported. Regenerating this dataset here is not possible today.
2. **The label policy has no frozen evidence.** Its selection rests on two
   ignored local artifacts, and its own contract section records that the
   selecting pilot was not an uncontaminated predeclaration and that the
   uncontaminated gates would have chosen `N = 4096`.
3. **No independent price cross-check was run.** Every identity verified here is
   internal to one lattice. An LSM or PDE cross-check remains outstanding and
   must be completed or explicitly resolved before training is authorized.
4. **No semantic-coverage judgement is made.** 41,966 of 200,000 training rows
   carry zero early-exercise premium and 25,648 carry no early-exercise node at
   all. Those counts are recorded, not judged: whether that mix makes a good
   learnability probe is a question the experiment itself answers.
5. **One partition only.** There is no boundary, extrapolation/OOD, or scenario
   partition, so any claim from this dataset is an in-envelope interpolation
   claim.
6. **Near-duplicate leakage is unbounded by any gate.** The distances were
   observed before a threshold was fixed, so the predeclared task 9E gate
   cannot be satisfied retroactively for this dataset version.
7. **Parquet files carry no embedded provenance.** The manifest→file digest
   binding is one-directional; a file separated from its manifest identifies
   nothing.
8. **The adjacent-step gap is not an error bound**, so the dataset's own
   numerical-error field does not bound label error.
9. **No training path exists.** `ml/train.py` and `ml/evaluate.py` are
   unchanged and European-specific; wiring them to the named target is work for
   the training task, not for this one.
10. **The row/manifest policy invariants are not enforced.** A row can disagree
    with the manifest's label-policy name or step count without the present
    admission checks rejecting it. Training remains blocked until both
    equalities are implemented and tested.

## Non-claims

- Admission is not training, and not authorization to train.
- Passing the implemented integrity gates says the dataset is internally consistent and
  describable. It does not say the labels are accurate, that the sampling
  design is good, or that a surrogate can learn them.
- A continuous dividend yield is not a discrete cash-dividend schedule.
- **The accepted PDE label policy remains frozen evidence; this task does not
  reinterpret or rerun it**, and nothing here touches the XSP/SPY phase.
