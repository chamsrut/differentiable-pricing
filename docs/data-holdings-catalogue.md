# Local data-holdings catalogue and integrity audit (task 9D)

The written output of task 9D
([tasks/active/task-9d-data-holdings-audit.md](tasks/active/task-9d-data-holdings-audit.md)).
It answers one question: what does this project hold on local disk, and what
about it can actually be established?

**This catalogue admits nothing.** Cataloguing a holding is not accepting it.
`AUTHORIZED_TRAINING_INPUT_STATUSES` is still empty
([decision-log.md](decision-log.md) DEC-011), no holding here is project
evidence, and **no accepted, versioned American training dataset exists.**

## How to read this document

Every material fact is marked:

- **[check]** — recomputed in this audit from the local bytes (hashing,
  Parquet metadata, column reductions, Git object reads).
- **[claim]** — copied from an artefact's own metadata, an existing report, or
  repository source, and *not* independently verified here.

A **[claim]** is never promoted to a **[check]** by restating it. Where a
check contradicts a claim, both are shown.

Digests are SHA-256 over file bytes. Sizes are apparent bytes. Paths are
repository-relative; no absolute or machine-local path, vendor job identifier,
credential, or signed URL appears in this document.

This is an audit record of one point in time, not a normative contract and not
frozen evidence. It does not live under `docs/results/`, has no generator
script, and is superseded by re-auditing rather than by regeneration.

## Classes of holding

The five classes below are kept apart deliberately; conflating them is the
failure this audit exists to prevent.

| Class | Where | In Git? | Status |
|---|---|---|---|
| Local vendor data on this machine | `data/market-feasibility-v1/raw/` | No — ignored | Present, licensed, never redistributable |
| Local generated datasets | `data/american-option-v1/`, `data/european-option-*/` | No — ignored | Present; candidate only |
| Ignored derived outputs | `data/processed/`, `artifacts/`, `runs/` | No — ignored | Reproducible-by-rerun demonstrations, not evidence |
| Versioned repository documentation and manifests | `configs/`, `docs/results/`, `docs/*.md` | **Yes** | Tracked, reviewed, frozen where applicable |
| An accepted, versioned training dataset | — | — | **Does not exist, in any class** |

**[check]** Nothing under `data/`, `artifacts/`, or `runs/` is tracked:
`git ls-files data/ artifacts/ runs/` returns 0 paths, and
`git status --porcelain --untracked-files=all data/` returns 0 lines — every
path is matched by `.gitignore` lines `/data/`, `/artifacts/`, `/runs/`.

## 1. The candidate American CRR dataset

`data/american-option-v1/` — Git-ignored, untracked, local only.

### 1.1 Files, sizes, digests

**[check]** Recomputed from local bytes:

| File | Bytes | SHA-256 |
|---|---:|---|
| `train.parquet` | 38,775,416 | `ff7a114f5bee260dba04c99afc78b0c9098fd9dfdd18c7a9dc952ce249ef50a9` |
| `validation.parquet` | 4,842,322 | `6f53a72e26a917f44324d3df86bffc4924eaf4739655f0563e63a0ad91f839c9` |
| `interpolation_test.parquet` | 4,857,508 | `f006a17cf5818f5950b0402e8942e6738726f2547e2e3c309038a836a1ed1058` |
| `manifest.json` | 20,158 | `252858929f3a981717dafc54c1639e1bd8159c5ad000aec7a0cc0cf24e9630d2` |
| `diagnostics.json` | 350,616 | `a311d5a5615d8693256ed4435f394c3c8fd220d943c88a8231a4c67d989f7408` |

Total 48,846,020 bytes (46.6 MiB) across 5 files.

**[check]** The three Parquet digests recomputed here match, byte for byte,
the digests the dataset's own `manifest.json` and `diagnostics.json` record for
them. `diagnostics.json` additionally records `sha256_verified: true` and byte
sizes that match the recomputed sizes exactly.

### 1.2 Format and partitions

**[check]** Apache Parquet, format version 2.6, snappy compression, written by
`parquet-cpp-arrow version 25.0.0`. Row-group size 25,000 rows: 8 row groups in
`train`, 1 each in `validation` and `interpolation_test`.

**[check]** Three named partitions, one file each — `train` 200,000 rows,
`validation` 25,000 rows, `interpolation_test` 25,000 rows; **250,000 rows in
total**, which corroborates the "approximately 250,000 rows" report. Every row's
`split` column equals its file's partition name.

**[check]** There are **no** `boundary`, `extrapolation`/OOD, or `scenario`
partitions. `docs/research-contract.md` "Data protocol" names four test
partitions; only `interpolation_test` exists here. The dataset's own manifest
records this as a deliberate scope limit **[claim]**.

### 1.3 Schema

**[check]** 29 columns, identical across all three partitions, every column
`not null`. Six `string`, four `int64`, nineteen `double`.

| Group | Columns |
|---|---|
| Identity / partition (string) | `sample_id`, `split`, `stratum`, `exercise_regime`, `option_type`, `label_policy` |
| Contract & model inputs (double) | `spot`, `strike`, `log_moneyness`, `maturity`, `rate`, `dividend_yield`, `volatility` |
| Primary label (double) | `american_price` |
| Label construction (double) | `american_price_at_steps`, `american_price_at_steps_plus_one`, `adjacent_step_gap` |
| European legs (double) | `european_crr_price`, `european_crr_price_at_steps`, `european_crr_price_at_steps_plus_one`, `european_black_scholes_price` |
| Derived diagnostics (double) | `early_exercise_premium`, `intrinsic_value`, `risk_neutral_probability`, `exercise_activity_fraction` |
| Lattice diagnostics (int64) | `label_steps`, `early_exercise_nodes`, `earliest_exercise_step`, `exercise_boundary_layers` |

**[check]** **No Greek is labelled.** There is no delta, gamma, vega, theta, or
rho column. **[claim]** the manifest states numerical American Greeks were
deferred to a later task and that a finite difference of these prices is not a
validated Greek.

**[check]** The Parquet files carry **no key-value file metadata beyond the
Arrow schema block** — no embedded manifest, config digest, or provenance. All
provenance lives in the sibling `manifest.json`, which is not cryptographically
bound to the Parquet files from inside them. The binding is one-directional:
the manifest names the files and their digests, the files do not name the
manifest.

### 1.4 Option types, exercise style, dividends

**[check]** `option_type` is exactly `{call, put}`, split 50/50 in every
partition (train 100,000/100,000; validation and interpolation_test
12,500/12,500 each).

**[check]** There is **no `exercise_style` column**. Each row carries *both* an
American price and a European CRR price computed on the same lattice, so the
row is a paired American/European observation rather than a row tagged with one
exercise style.

**[check]** Dividends are represented as a **continuous dividend yield**, in a
`dividend_yield` column spanning `[0.0, 0.11998901…]` in `train` and similar in
the other partitions. There is **no discrete cash-dividend schedule anywhere in
the schema** — no ex-date, no amount, no dividend count. A zero-yield subset
exists (one stratum pins `dividend_yield = 0` for a non-dividend call control),
but the dataset as a whole is **not** a no-dividend dataset.

> **A continuous dividend yield is not a discrete cash-distribution schedule.
> This dataset cannot model SPY cash dividends** ([decision-log.md](decision-log.md)
> DEC-001, DEC-030). Nothing learned on it transfers as a claim about
> discrete-dividend American pricing.

### 1.5 Feature ranges, missing and non-finite values

**[check]** Zero nulls in all 29 columns of all three partitions; zero
non-finite (NaN/±inf) values in every floating-point column. Recomputed ranges
in `train`, and the declared sampling domain **[claim]** from the manifest:

| Feature | `train` observed range **[check]** | Declared domain **[claim]** |
|---|---|---|
| `spot` | 50.0005 – 149.9998 | [50.0, 150.0] |
| `log_moneyness` | −0.699932 – 0.699931 | [−0.7, 0.7] |
| `maturity` (years) | 0.019195 – 2.999986 | [0.0191781, 3.0] |
| `rate` | −0.019999 – 0.119998 | [−0.02, 0.12] |
| `dividend_yield` | 0.0 – 0.119989 | [0.0, 0.12] |
| `volatility` | 0.050001 – 0.799971 | [0.05, 0.8] |
| `strike` (derived) | 24.87 – 299.26 | not declared directly |

**[check]** Every partition lies inside the declared domain on all six declared
features — zero violations. **[check]** Every one of the 11 declared strata
also holds inside its own per-stratum bounds, and the single option-type-
restricted stratum (`non_dividend_call_control`, calls only) contains calls
only — zero violations across all three partitions.

**[check]** `label_steps` is the constant 1024 in every row.
`earliest_exercise_step` uses `−1` as a sentinel for "no such node"
**[claim, per the manifest's units block]**.

### 1.6 Identifier uniqueness and cross-partition leakage

**[check]** `sample_id` is unique within every partition (200,000 / 25,000 /
25,000 distinct) and **the three partitions share no `sample_id`** — all three
pairwise intersections are empty.

**[check]** Stronger: taking the contract state itself as the key
(`option_type`, `spot`, `strike`, `maturity`, `rate`, `dividend_yield`,
`volatility`), there are **zero exact duplicate states anywhere in the 250,000
rows**, and **zero exact state overlaps between any two partitions**.

**[check]** Near-duplicate check, standardizing the five economic features
(`log_moneyness`, `maturity`, `rate`, `dividend_yield`, `volatility`) by the
training mean and standard deviation and searching each held-out row's nearest
training row of the same option type by brute force: minimum distance 0.0116
(interpolation_test puts), no held-out row closer than 0.01, and 0.12–0.14 % of
held-out rows within 0.05. This is a **descriptive diagnostic, not a gate** —
no near-duplicate threshold is predeclared anywhere in this repository, and
choosing one after seeing these numbers would be exactly the post-hoc criterion
the project forbids. Setting that threshold belongs to the admission task.

**[claim]** `diagnostics.json` reports the same two intersection families
(`sample_id_intersections`, `state_intersections`) as all-zero, with
`status: "ok"` and an empty `findings` list.

### 1.7 Label policy and internal consistency

**[claim]** The manifest declares label policy `american-crr-adjacent-average/1`:
`price = 0.5 · (CRR(N) + CRR(N+1))` at `N = 1024`, with the European leg
averaged the same way on the same lattice.

**[check]** Recomputed on all 250,000 rows, with exact (bitwise) agreement and
zero tolerance:

- `american_price == 0.5 · (american_price_at_steps + …_plus_one)` — max
  residual **0.0**;
- `european_crr_price == 0.5 · (european_crr_price_at_steps + …_plus_one)` —
  max residual **0.0**;
- `adjacent_step_gap == |american_price_at_steps − …_plus_one|` — max residual
  **0.0**;
- `early_exercise_premium == american_price − european_crr_price` — max
  residual **0.0**;
- `american_price ≥ european_crr_price` on every row, minimum difference
  exactly 0.0 (American dominance holds with no tolerance, as the shared
  lattice implies);
- `log_moneyness == log(spot / strike)` — max residual 2.8e-16.

**[check]** `american_price ≥ intrinsic_value` holds up to floating-point
rounding: the largest shortfall anywhere is **−4.83e-13** absolute, far below
the manifest's own `at_intrinsic_over_spot = 1e-9` diagnostic threshold
**[claim]**. This is rounding, not a bound violation, and it is recorded rather
than silently rounded away.

**[check]** Every label-diagnostic count the manifest publishes reconciles
exactly against a recomputation from the Parquet bytes, in all three
partitions: `zero_premium_rows`, `no_early_exercise_rows`, `at_intrinsic_rows`,
`large_adjacent_gap_rows`, `near_zero_price_rows`, `zero_price_rows`, and the
row counts, option-type counts and per-stratum counts. **No discrepancy was
found between the manifest's claims and the data.**

### 1.8 Oracle provenance — what is verifiable today

**[claim]** The manifest names its oracle as `dp::crr_binomial` v0.1.0, a
deterministic C++20 lattice built `Release` with `GNU 13.3.0`, exposed as
`differentiable_pricing._core.crr_price_batch`, and records three digests:
`crr_header_sha256`, `crr_source_sha256`, and a composite
`crr_implementation_sha256`.

**[check]** All three reproduce from the **current branch's** tracked sources:

- `crr_header_sha256` `eec8f796…` equals the SHA-256 of `cpp/include/dp/binomial_tree.hpp` at HEAD;
- `crr_source_sha256` `7dc5d754…` equals the SHA-256 of `cpp/src/binomial_tree.cpp` at HEAD;
- `crr_implementation_sha256` `c69cefc8…` equals the composite that
  `CMakeLists.txt` defines — `SHA256` of the colon-joined digests of
  `bindings/python/module.cpp`, `cpp/include/dp/binomial_tree.hpp`,
  `cpp/include/dp/black_scholes.hpp`, `cpp/include/dp/option.hpp`,
  `cpp/src/binomial_tree.cpp`, `cpp/src/black_scholes.cpp`,
  `cpp/src/option.cpp` — recomputed at HEAD **and** at commit `49ef72a`, both
  giving `c69cefc8…`.

**The pricing oracle that produced these labels is byte-identical to the CRR
engine on the current branch.** That is the strongest provenance result in this
audit.

### 1.9 Generator provenance — what is *not* on this branch

**[check]** The generator, its configuration and its tests are **not present on
this branch and not on `main`**. They exist only on the unmerged branch
`feat/american-dataset-v1`, whose tip is commit `49ef72a` ("Preserve synthetic
American dataset prototype"), read here through Git object commands only — no
branch switch, no checkout, no cherry-pick.

**[check]** `git merge-base --is-ancestor 49ef72a main` is false. The branch
forked from `280ac3b` (before tasks 9A/9B/9C-*), so relative to HEAD it *adds*
the American dataset prototype and *lacks* the entire market and PDE line of
work: a diff against HEAD shows 94 files changed, +6,853 / −83,912 lines. It is
not a fast-forward and merging it forward is not a trivial operation.

**[check]** Artefacts added by `49ef72a`, with digests read from Git objects:

| Path on `feat/american-dataset-v1` | Lines | SHA-256 |
|---|---:|---|
| `configs/american_option_dataset_v1.toml` | 228 | `d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98` |
| `configs/american_label_policy_pilot_v1.toml` | 54 | `7858e1d6cd283f6e35cbd843a0b5cf3aaf02628dcb34e1cd727a606fff7f248b` |
| `configs/american_label_policy_pilot_v2.toml` | 76 | `c5eafa09b19c41dc22044bbee8a43469c077e25c95ce17a24cca028b1910944c` |
| `python/src/differentiable_pricing/data/american_config.py` | 445 | `f52207bc651193282b2910ede8fa8aaa7b70d4b2c33f7b2a58dc32fb65bfc1b1` |
| `python/src/differentiable_pricing/data/american_generate.py` | 1,145 | `9929b7309372f843b771eacb25f6852882caa0780b17bf621eddf66c3c8c3862` |
| `python/src/differentiable_pricing/data/american_diagnose.py` | 1,274 | `eb8b2513600bddceec4424a3ca4a9b530ee6890d95237c65646ffca4312f11f4` |
| `python/src/differentiable_pricing/data/american_label_policy.py` | 667 | `b62347ba2515cad1174256877a199482a01ea62d4d54d132a87c514b5e963efb` |

Tests added alongside them: `python/tests/american_dataset_fixtures.py`,
`test_american_label_policy.py`, `test_american_option_dataset.py`,
`test_american_option_dataset_config.py`, `test_american_option_diagnostics.py`
(2,189 lines in total). None of these run on this branch, because none of them
is on it.

**[check]** The manifest's configuration digest **matches**: it declares
`american_option_dataset_v1.toml` with SHA-256 `d18485c6…`, and the file at
`49ef72a` hashes to exactly that. The dataset is therefore digest-linked to a
recoverable configuration — on a branch that is not this one.

**[check]** The manifest's declared toolchain **[claim]** is Python 3.11.15,
NumPy 2.4.6, PyArrow 25.0.0; the interpreter used for this audit reports
Python 3.11.15, NumPy 2.4.6, PyArrow 25.0.0 — the same versions. The manifest
itself warns **[claim]** that byte-identical regeneration is guaranteed only
for that recorded toolchain.

### 1.10 The label policy's own selection evidence

**[claim]** The manifest attributes the choice `steps = 1024` to a pilot,
citing `configs/american_label_policy_pilot_v2.toml` executed by
`differentiable_pricing.data.american_label_policy`, and notes that pilot v1
selected no candidate.

**[check]** Both raw pilot reports exist, but only as **ignored local
artefacts**: `artifacts/american-label-policy-pilot-v1.json`
(`ded4e2a8…`, 26,543 bytes, `status: "no_candidate_passed"`,
`selected_steps: null`) and `artifacts/american-label-policy-pilot-v2.json`
(`ebba077b…`, 25,910 bytes, `status: "selected"`, `selected_steps: 1024`,
schema `american-crr-label-policy-report/1`). The v2 report's recorded
`config_sha256` `c5eafa09…` and `dataset_config_sha256` `d18485c6…` both match
the corresponding files at `49ef72a` **[check]**.

**[check]** **There is no frozen `docs/results/` snapshot for this label
policy on any branch.** `docs/results/` at HEAD holds five snapshots (European
replication and validation, American LSM cross-check, PDE label policy v1 and
v2); `feat/american-dataset-v1` holds three. Neither set contains a CRR
label-policy result. The selection rests entirely on ignored local artefacts.

**[check]** Correspondingly, the normative section describing this policy —
"Label policy v1" in `docs/american-crr-contract.md` — exists **only on
`feat/american-dataset-v1`**. HEAD's `docs/american-crr-contract.md` has no such
section: its headings run Purpose and scope, Lattice and recursion, Complexity/
batching/determinism, Convergence output, Permanent validation properties, Why
CRR precedes LSM, Current non-claims. **The contract governing this dataset's
label policy is not on the current branch.**

### 1.11 Loader compatibility

**[check]** The existing ML loader cannot read this dataset today, and would
reject it rather than mis-read it. `python/src/differentiable_pricing/ml/dataset.py`
compares each Parquet file's schema with `TABLE_SCHEMA` from
`data/generate.py` using `schema.equals(...)` — an exact equality against the
**European** table (labels `price`, `delta`, `gamma`, `vega`, `theta`, `rho`) —
and `data/diagnose.load_manifest` rejects any manifest whose `schema_version`
is not `european-option-dataset/1` (`data/config.py:18`). This dataset declares
`american-option-dataset/1` with 29 columns and no Greeks, so both gates fail
closed. That is the correct behaviour and it is the concrete work item for the
admission task, not a defect to patch incidentally.

### 1.12 Status

**A local candidate CRR dataset reportedly exists, but it has not yet been
catalogued, validated, reproducibly admitted, or accepted as project
evidence.** This catalogue completes the *cataloguing* half only. It is not
accepted, not reproducible from tracked sources alone, and not training-ready.
Section 8 lists what an admission decision still has to resolve.

## 2. The European datasets (context, not roadmap phase 1)

**[check]** `data/european-option-v1/` (5 files, 32,267,578 bytes) and
`data/european-option-replication-v1/` (5 files, 32,269,836 bytes), both
Git-ignored. Both declare `european-option-dataset/1`, both carry
200,000 / 25,000 / 25,000 rows.

**[check]** For both, every Parquet digest recorded in the dataset's own
manifest matches a recomputation from the local bytes, and each manifest's
declared configuration digest matches a **tracked** file at HEAD:
`configs/european_option_dataset_v1.toml` → `0d2542b8…`, and
`configs/european_option_dataset_replication_v1.toml` → `58ff1517…`.

This is the contrast that matters: the European datasets are digest-linked to
configuration **and** to a generator that both live on this branch
(`python/src/differentiable_pricing/data/generate.py`). The CRR dataset is
digest-linked to a configuration and generator that do not.

The frozen European evidence in `docs/results/` is terminal and is not
revisited here.

## 3. The market-data archive (local vendor data)

`data/market-feasibility-v1/` — Git-ignored, local only, proprietary. Per
[decision-log.md](decision-log.md) DEC-015 it never enters Git and CI never sees
it. **Licensed vendor content may not be redistributed**; everything in this
section is metadata *about* the archive, never its content.

### 3.1 Totals and integrity

**[check]** 120 files, 747,750,918 bytes (713.1 MiB) in total; of those, 114
files and **747,662,190 bytes** are raw vendor data and 6 files are manifests.

**[check]** Integrity: `manifests/sha256sums.txt` lists 119 files; verifying
every one against the bytes on disk gives **119 OK, 0 mismatches, 0 missing**.
The single file on disk not covered by that list is `sha256sums.txt` itself.

**[claim]** `manifests/organization-report.md` states 114 data files at
747,662,190 bytes — which the check above confirms exactly — and a total of
120 files at 747,750,328 bytes. **[check]** The recomputed archive total is
747,750,918 bytes, **590 bytes more** than the claimed total. The discrepancy is
confined to the manifests directory and is explained by ordering: the manifests
were finalized after the report recorded the total, and file timestamps show
`inventory.csv`/`requests.csv` written at 01:01 and
`organization-report.md`/`sha256sums.txt` at 01:02. No raw data file is
implicated, and the checksum verification of all 119 listed files passes. It is
recorded here rather than reconciled away.

### 3.2 Databento holdings

**[check]** All Databento payloads are `.dbn.zst` — DBN, zstd-compressed,
**never decompressed or transcoded in this audit**. Three sessions are present
throughout: **2026-06-17, 2026-06-18, 2026-07-30**.

| Dataset | Product | Schema | Data files | Bytes | Sessions |
|---|---|---|---:|---:|---|
| OPRA.PILLAR | SPY options | `cbbo-1m` | 3 | 365,349,690 | all three |
| OPRA.PILLAR | SPY options | `definition` | 3 | 1,457,609 | all three |
| OPRA.PILLAR | SPY options | `statistics` | 3 | 15,592,336 | all three |
| OPRA.PILLAR | XSP options | `cbbo-1m` | 3 | 358,432,044 | all three |
| OPRA.PILLAR | XSP options | `definition` | 3 | 1,612,079 | all three |
| OPRA.PILLAR | XSP options | `statistics` | 3 | 3,686,547 | all three |
| GLBX.MDP3 | ES futures | `bbo-1m` | 3 | 426,239 | all three |
| GLBX.MDP3 | ES futures | `definition` | 3 | 7,430 | all three |
| GLBX.MDP3 | ES futures | `statistics` | 3 | 152,513 | all three |
| GLBX.MDP3 | SR3 futures | `definition` | 3 | 158,336 | all three |
| GLBX.MDP3 | SR3 futures | `statistics` | 3 | 667,674 | all three |
| EQUS.MINI | SPY underlying | `bbo-1m` | 3 | 54,510 | all three |

**[check]** 36 `.dbn.zst` payloads plus 72 sidecar JSON files (`condition.json`,
`manifest.json`, `metadata.json` per request directory) = 108 Databento files.
Request directories are named by vendor job identifier; those identifiers are
deliberately **not reproduced here**.

**[claim]** `organization-report.md` records 24 unique requests plus one exact
duplicate archive, 0 unclassified, 0 missing expected requests, 0 hash
mismatches, per-provider totals of OPRA 12 requests / 54 files /
746,161,319 bytes, GLBX 10 / 45 / 1,437,900, EQUS.MINI 2 / 9 / 59,617, FRED
6 files / 3,354 bytes. **[check]** The recomputed per-dataset byte totals above
sum consistently with those provider totals, and `raw/_unclassified/`,
`raw/occ/` and `raw/state-street/` are **empty directories** — 0 files each.

**[claim]** `manifests/duplicates.csv` records the one duplicate as
`verified_identical` — 4 files, 1,229,270 bytes, identical relative paths, byte
sizes and SHA-256 — and `manifests/missing-expected-requests.csv` contains a
header row only, i.e. nothing missing.

**Not held:** no OCC data and no State Street data, despite directories being
provisioned for them; no trades, no full order book, no consolidated tape
beyond the 1-minute BBO/CBBO aggregations above; no session outside the three
dates; no XSP or SPY underlying beyond the ETF BBO above.

### 3.3 FRED holdings

**[check]** Six CSV series, 3,354 bytes total, one file per series:
`DGS1MO`, `DGS3MO`, `DGS6MO`, `DGS1`, `DGS2`, `SOFR`. Format is a two-column
CSV, `observation_date` plus the series code, daily rows with **empty values on
non-publication days** (holidays and weekends appear as blank fields, not
zeros). Coverage begins 2026-06-15, spanning the three market sessions.

These are public, freely redistributable series; they are nonetheless kept out
of Git with the rest of the archive, so the archive stays one indivisible,
never-committed unit.

### 3.4 Documented limitations of the archive

**[claim]**, from
[market-state-reconstruction-contract.md](market-state-reconstruction-contract.md)
and [research-contract.md](research-contract.md) "Real-market inputs" — this
audit re-states, and does not re-derive, these limits:

- The archive supports a **feasibility and identifiability study only**. It
  produces no price, no label, no Greek, no implied volatility, and no
  calibrated curve or surface.
- Discount factors and forwards are identified at quoted expiries as knots with
  no interpolation through them.
- **Dividend amounts, borrow and carry, and corporate-action adjustments are
  not available** from this archive.
- Three sessions are not a historical market study. Every threshold in tasks
  9A/9B is exploratory and gates no replication partition.
- The OPRA definition records leave `contract_multiplier` and `exercise_style`
  undefined; the code carries them as `None` rather than defaulting them.

**[check]** That last claim is confirmed against the local bytes in §4.3 below.

## 4. Processed market partitions (ignored derived output)

`data/processed/` — four sibling output trees, all Git-ignored, all produced by
the read-only ingestion pipeline from the archive in §3.

### 4.1 The four trees

**[check]** Every tree has the identical layout
`product=<p>/schema=<s>/date=<YYYY-MM-DD>/part-00000.parquet`, 36 files each,
five products (`spy-options`, `xsp-options`, `spy-underlying`, `es-futures`,
`sr3-futures`) and the three sessions.

| Tree | Files | Bytes |
|---|---:|---:|
| `market-feasibility-v1` | 36 | 733,059,733 |
| `market-feasibility-v1-audit2` | 36 | 758,974,260 |
| `market-feasibility-v1-audit3` | 36 | 758,974,260 |
| `market-feasibility-v1-audit3-verify` | 36 | 758,974,260 |

**[check]** Hashing all 144 files and comparing pairwise:
`audit2`, `audit3` and `audit3-verify` are **byte-identical to each other on
all 36 files**. The original `market-feasibility-v1` differs on 30 of 36 files
and agrees on 6, i.e. it is an earlier revision of the same pipeline output.

Two consequences. The `audit3` / `audit3-verify` pair is genuine **determinism
evidence** — a re-run reproduced its predecessor byte for byte. And of the
3,009,982,513 bytes (2.80 GiB) under `data/processed/`, 1,517,948,520 bytes
(1.41 GiB) are exact duplicates of `audit2`. Nothing is deleted by this audit;
the redundancy is recorded for whoever plans storage.

### 4.2 Row counts and schemas

**[check]** Recomputed from `market-feasibility-v1-audit3` (identical in the
other two audit trees):

| Product | Schema | Files | Rows |
|---|---|---:|---:|
| `spy-options` | `cbbo-1m` | 3 | 16,806,923 |
| `spy-options` | `definition` | 3 | 41,990 |
| `spy-options` | `statistics` | 3 | 3,302,420 |
| `xsp-options` | `cbbo-1m` | 3 | 20,875,144 |
| `xsp-options` | `definition` | 3 | 49,840 |
| `xsp-options` | `statistics` | 3 | 821,922 |
| `spy-underlying` | `bbo-1m` | 3 | 2,473 |
| `es-futures` | `bbo-1m` | 3 | 16,857 |
| `es-futures` | `definition` | 3 | 133 |
| `es-futures` | `statistics` | 3 | 6,630 |
| `sr3-futures` | `definition` | 3 | 3,644 |
| `sr3-futures` | `statistics` | 3 | 30,423 |

**[check]** Quote tables carry 20 columns including `ts_recv_ns` / `ts_event_ns`
(ns, UTC), `bid_price`, `ask_price`, `bid_size`, `ask_size`, `mid_price`,
`quoted_spread`, `relative_spread`, `bid_venue`, `ask_venue`,
`in_regular_session`, `quality`, `vendor_flags`, `vendor_flag_names`,
`bad_ts_recv`, `maybe_bad_book`, `tradable_core`. Definition tables carry 19
columns including `raw_symbol`, `root`, `is_standard_root`, `underlying`,
`option_type`, `strike`, `expiration_ns`, `expiration_date`, `currency`,
`exchange`, `contract_multiplier`, `exercise_style`, `security_update_action`.
Statistics tables carry 11 columns including `stat_type`, `price`, `quantity`,
`update_action`, `is_outright`, `is_repeat_of_earlier_record`.

### 4.3 Are the expected bid/ask fields populated?

**[check]** Yes, with quantified gaps — the pipeline marks missing sides rather
than silently filling them:

| Product / schema | Rows | `bid_price` null | `ask_price` null | both null | `tradable_core` |
|---|---:|---:|---:|---:|---:|
| `spy-options` / `cbbo-1m` | 16,806,923 | 598,942 (3.56 %) | 0 (0.00 %) | 0 | 96.43 % |
| `xsp-options` / `cbbo-1m` | 20,875,144 | 1,674,349 (8.02 %) | 24,173 (0.12 %) | 22,243 | 91.97 % |
| `spy-underlying` / `bbo-1m` | 2,473 | 143 (5.78 %) | 226 (9.14 %) | 64 | 87.67 % |
| `es-futures` / `bbo-1m` | 16,857 | 32 (0.19 %) | 34 (0.20 %) | 29 | 99.45 % |

**[check]** The `quality` label distribution is consistent with those nulls —
e.g. SPY options: 15,318,186 `ok`, 889,358 `stale`, 598,942 `missing`, 435
`locked`, 2 `crossed`; XSP options: 17,694,660 `ok`, 1,504,205 `stale`,
1,676,279 `missing`, and no locked or crossed rows.

**[check]** **`exercise_style` is null on 100 % of option definition rows** —
0 of 41,990 populated for SPY, 0 of 49,840 for XSP — and `contract_multiplier`
is likewise null on 100 % of them. `root` is `SPY` on all SPY rows and `XSP` on
all XSP rows, and `currency` is `USD` throughout.

This is the audit's most consequential market finding for the locked roadmap.
Phase 2's control/target design rests on XSP being European and SPY being
American, and **that distinction is not carried in these bytes**. It is an
external convention — which is exactly how
[market-state-reconstruction-contract.md](market-state-reconstruction-contract.md)
already classifies both fields **[claim]**, and how
`python/src/differentiable_pricing/market/instruments.py` already handles them
in code, deriving `exercise_style` from a CFI code that OPRA does not populate
here and refusing to default it. The finding is not a defect; it is a
constraint phase 2 must plan around, and it is stated here so that phase 2
cannot assume the field.

## 5. Other ignored derived outputs

**[check]** `artifacts/` — 73 files, 20,267,418 bytes (19.3 MiB): raw study
reports and demo outputs for CRR convergence and benchmarking, the CRR
label-policy pilots v1/v2, LSM cross-checks, European neural runs, the market
feasibility/reconstruction audits, and the PDE label-policy pilot v1 and v2
remediation/confirmation runs.

**[check]** `runs/` — 1 file, 13,602 bytes, under `european-neural-replication-v1`.

Both are ignored, refused by `verify_training_input_publication`
([decision-log.md](decision-log.md) DEC-011), and are demonstrations of
machinery on one run — not datasets and not evidence. The two CRR
label-policy pilot reports in `artifacts/` are the *only* record of how this
dataset's label policy was chosen (§1.10).

## 6. What is versioned in Git

**[check]** Tracked and reviewed, for contrast with everything above:
`configs/` (19 versioned TOML configurations, including both European dataset
configs but **not** `american_option_dataset_v1.toml`), and `docs/results/` —
five frozen snapshots: `european_replication_results_v1.json` (10,478 B),
`european_validation_results_v1.json` (8,522 B),
`american_lsm_crosscheck_results_v1.json` (18,396 B),
`american_pde_label_policy_results_v1.json` (1,056,526 B),
`american_pde_label_policy_v2_results_v1.json` (43,160 B).

**The accepted PDE label policy remains frozen evidence; this audit does not
reinterpret or rerun it**, and did not read it except to confirm its presence
and size.

## 7. What is absent

- **An accepted, versioned training dataset — American or otherwise.** None
  exists, in Git or on disk.
- **No accepted, versioned PDE-labelled SPY training dataset exists**, and **no
  American neural surrogate has yet been trained and accepted.** There is no
  SPY-labelled dataset of any kind under `data/`.
- No frozen `docs/results/` snapshot for the CRR label policy (§1.10).
- No OCC and no State Street data, though directories exist for them (§3.2).
- No boundary, extrapolation/OOD, or scenario partition in any dataset (§1.2).
- No Greek labels in the CRR dataset (§1.3).
- No discrete-dividend schedule in any generated dataset (§1.4).
- No S3 bucket, no object-storage client, no vendor API client, and no
  credential material anywhere in this repository. **Cloud storage and vendor
  ingestion are deferred reproducibility work, not prerequisites for the first
  CRR learnability experiment** — planned in
  [tasks/active/task-9f-remote-data-access-plan.md](tasks/active/task-9f-remote-data-access-plan.md),
  which is `On hold`.

## 8. Unresolved limitations, for the admission task

Every item below is a finding of this audit, not a repair. Task 9E
([tasks/active/task-9e-crr-dataset-admission.md](tasks/active/task-9e-crr-dataset-admission.md))
inherited them.

**Update, task 9E.** Items 2 (partly), 6 and, for the exact properties, 5's
framing were addressed: the label policy's configurations and contract section
were recovered onto this branch and digest-pinned, and the loader now admits the
schema by explicit name and version. Items 1, 3, 4, 7 and 8 stand unchanged, the
label policy still has **no frozen evidence**, and item 5's near-duplicate
threshold is still not predeclared. The surviving set is restated in
[american-crr-dataset-admission.md](american-crr-dataset-admission.md),
"Limitations that survive admission".

1. **The generator is not on this branch.** Regenerating or independently
   re-deriving the CRR dataset from tracked sources alone is **not currently
   possible**. It requires `feat/american-dataset-v1`, which is not an ancestor
   of `main` and is 94 files divergent from HEAD (§1.9).
2. **The label policy has no frozen evidence and no contract section on this
   branch.** Its selection lives in two ignored `artifacts/` reports, and the
   "Label policy v1" contract section exists only on the other branch
   (§1.10).
3. **Parquet files carry no embedded provenance.** The manifest→file digest
   binding is one-directional; a file separated from its manifest carries
   nothing identifying (§1.3).
4. **No semantic suitability assessment has been made.** This audit checked
   internal consistency, not whether the sampling design, strata weights,
   moneyness/maturity coverage, or the share of rows carrying no early-exercise
   signal at all (`zero_premium_rows` is 41,966 of 200,000 in `train`, and
   `no_early_exercise_rows` 25,648) make the dataset a *good* learnability
   probe.
5. **No near-duplicate threshold is predeclared.** The measured nearest-
   neighbour distances (§1.6) are descriptive; a leakage gate must be
   predeclared before, not after, looking at them again.
6. **The loader rejects this schema by design** (§1.11). Named-schema support
   is work the admission task must do deliberately.
7. **No independent price cross-check has been run** against the LSM or PDE
   engines on any row of this dataset. Every consistency check in §1.7 is
   internal to one lattice.
8. **The adjacent-step gap is not an error bound** **[claim]**, so the
   dataset's own numerical-error field does not bound label error.
9. **`exercise_style` and `contract_multiplier` are 100 % null** in the
   processed option definitions (§4.3) — a phase-2 constraint, not a phase-1
   one.
10. **The archive's claimed total is 590 bytes below the recomputed total**
    (§3.1), explained by manifest write ordering but left unreconciled.
11. **The market manifests contain machine-local absolute source paths.** They
    are not reproduced in this document and would have to be scrubbed or
    excluded before any manifest is published anywhere.
12. **Parts of the archive tree carry world-writable permissions**
    (`data/market-feasibility-v1/raw/` and `manifests/`). Local hygiene, not a
    data-integrity finding, but it belongs in any storage plan.

## Non-claims

- Nothing here admits, accepts, or authorizes any holding as project evidence
  or as a training input.
- A passing integrity check is not a statement that a dataset is *suitable*.
  Every check in §1.7 is internal consistency of one lattice's output.
- A continuous dividend yield is not a discrete cash-distribution schedule, and
  the CRR dataset cannot model SPY cash dividends.
- This document is an audit record at one point in time. It is not frozen
  evidence, has no generator script, and re-auditing supersedes it.
