# Task 9H: adaptive American price-model development

## Status

`status=active; type=adaptive_exploratory_development; scope=price_only;
branch=experiment/task-9h-american-pricer-development;
partitions_available=train+validation; final_partition_access=forbidden;
infrastructure implemented and hardened; eight attempts completed, none meeting
the criterion; E2b (scratch_residual_smooth_floor_raw_loss_v1) is the recorded
price leader and fails only the two structural gates; both label-free analyses
have run, fixing delta = 1e-4 and leaving E3 unauthorized; E2c
(scratch_residual_smooth_floor_margin_v1) is predeclared and not yet run; no
attempt running; final partition untouched.`

Task 9G is terminal and negative ([decision-log.md](../../decision-log.md)
DEC-038, approved by DEC-039). This task is opened by DEC-041.

## What this task is

**Adaptive exploratory development, not confirmatory research.** Its purpose is
to iterate on capacity, representation, target and architecture until an
American **price** model meets a fixed development criterion on `train` and
`validation`.

Selection happens against `validation`, repeatedly and deliberately. Seeing
validation results is expected and creates selection bias. **No task 9H result
is confirmatory.** Nothing here may be cited as a project result.

**Scope is price only.** Greeks, latency, implied volatility and transfer
learning are separate follow-up stages that begin only after a candidate meets
the criterion. None of their machinery exists in this task, and none is built
speculatively.

Once a candidate works, final confirmation requires a **new protocol and a fresh
partition**, as its own task with its own gates and its own review.

## Rules

1. Only `train` and `validation` data may be used.
2. **No code path may open, hash, stat, import, count or inspect
   `interpolation_test` or another final partition.** The workbench fails closed
   on any such split name or resolved path, strips the dataset manifest to the
   two reachable splits before anything else sees it, and exposes no
   final-evaluation command. The **retained** manifest entries are guarded too:
   a supplied file name goes through the same final-partition and containment
   check as a configured path before anything opens, hashes, stats or counts it,
   so a `train` entry pointing at `../interpolation_test.parquet` is refused
   rather than read. One forbidden-token definition
   (`attempts.FORBIDDEN_PARTITION_TOKENS`) serves both the runtime guard and the
   offline static scan, so the two cannot drift apart.
3. Every attempt is recorded — successful, failed and abandoned alike — in the
   append-only attempt log at the canonical path
   `docs/attempts/task-9h-attempt-log.jsonl`, and nowhere else. An existing
   entry is never rewritten, and only a report of this workbench's schema,
   judged against the canonical criterion, may be recorded.
4. Every real attempt runs from a **clean committed source tree**, and from
   **committed source specifically**. The runner refuses to start with tracked
   worktree modifications, and separately requires the selected attempt
   configuration and every file recorded in `source_digests` — the Task 9H
   package, both scripts, the acceptance configuration and the locked Task 9G
   protocol — to be tracked at `HEAD` and byte-identical to their `HEAD` blob.
   A clean `git status --untracked-files=no` alone would still admit a brand-new
   untracked configuration or module, in which case the recorded commit would
   not describe what ran. Ignored artifacts and runs are unaffected: the
   untracked workspace is **not** required to be empty.
5. **Attempt configurations are immutable after use.** A used configuration is
   never edited; a changed idea gets a new attempt ID and a new file.
6. Agents implement code and configuration and analyze compact summaries.
   **The human invokes every training run.**
7. Later attempts may respond to earlier validation results. That is the point
   of the loop, and it is also exactly what makes the output biased.
8. **The dataset identity is pinned to Task 9G's.** Before training, the
   manifest and both reachable partitions must hash to the `manifest_sha256`,
   `train_sha256` and `validation_sha256` that
   `configs/american_neural_pilot_protocol_v1.toml` already locked, and the
   declared dataset and manifest paths must be the ones that protocol names.
   Only those three identities are read out of the protocol; no other
   partition's declared digest is resolved, compared or recorded. Any mismatch
   fails closed — otherwise an attempt could train on a regenerated dataset and
   still be compared against the Task 9G control. Task 9G's own row-level
   `label_policy` / `label_steps` verification
   (`ml.american_pilot.verify_partition_policy`) is then run over both
   partitions, reused rather than reimplemented.
9. **Every declared configuration field must be behavior that is implemented and
   dispatched.** An unsupported `optimizer.name`, `optimizer.schedule`,
   `checkpoint.metric`, `checkpoint.rule`, `training.shuffle`,
   `row_selection.rule`, `seeds.derivation`, `representation`, `target` or
   `physical_reconstruction` is refused, as is any unknown key in any section.
   A declaration execution would silently ignore is never recorded in the
   attempt log, because the log has to describe what actually ran.

## Predeclared candidate: `scratch_residual_smooth_floor_v1` (E2)

> **This attempt has run**, and did not meet the criterion. The predeclaration
> below is kept **exactly as it was written before the run** — that is what
> makes it a predeclaration. Its outcome, and why the result does not mean what
> it appears to mean, are in
> [What E2 measured, and what it settled](#what-e2-measured-and-what-it-settled)
> and the progress table.

`configs/american_dev_attempt_scratch_residual_smooth_floor_v1.toml`. Relative to
`scratch_residual_architecture_v1` — the direct residual parent that passed all
three price gates — exactly one behavioural field moves: `head` `direct` →
`smooth_lower_floor`, plus the initialization seed the existing rule derives from
the new attempt ID. Everything else is the parent's: the backbone
(`smooth_residual`, width 128, 6 blocks, `tanh`, no normalization, no dropout),
the five base features, the representation, the target, the reconstruction, the
selected rows, the shuffle seed, the validation partition, the optimizer, the
schedule, the budget, the precision, the CPU thread count, the checkpoint
semantics and the criterion.

**The network still predicts the direct normalized price.** `smooth_lower_floor`
is an output transformation, not a reparameterization. The network's output is
character-for-character the direct head's `raw * price_scale + price_mean`; only
the final step differs:

```
floor  = smooth_max(E_analytic / A, intrinsic / A)
output = floor + tau * softplus((direct - floor) / tau)
```

It does **not** predict an American premium. That is the whole point: E1
established that the premium target costs the residual backbone its accuracy.

**The temperature is predeclared, in code, at `tau = 1e-4` normalized.** It lives
in `american_dev.attempts.SMOOTH_FLOOR_TEMPERATURE` rather than in a
configuration file because every attempt configuration must declare exactly the
same top-level keys, so a per-attempt temperature field would have to be added to
the six immutable configurations that already ran. It is therefore pinned in
`source_digests` and recorded in the attempt report.
`attempts.assert_temperature_consistent` re-derives from the digest-pinned
acceptance file that `1e-4` sits strictly between the material violation
tolerance `1e-6` (below which smoothing would be unresolvable) and the normalized
RMSE limit `3e-3` (at or above which the smoothing bias would consume the
accuracy budget it exists to preserve). **The repository's units confirm the
nominal value**; had they contradicted it, the checker raises and the discrepancy
is reported rather than quietly resolved by substituting another number. The
check runs in `scripts/check.sh`, in CI and in the runner's pre-flight.

**What the transformation guarantees, exactly.**

- `smooth_max(a, b) = max(a, b) + tau*log1p(exp(-|a-b|/tau))` — the log-sum-exp,
  written so that a **non-negative** correction is added to the hard maximum.
  The result is at or above `max(a, b)` **bitwise in float64**. Writing it as
  `tau*logsumexp((a,b)/tau)` would lose that: the `a/tau` then `tau*…`
  round-trip can land a unit in the last place *below* the maximum, which is the
  one direction a lower floor may not move.
- The projection adds `tau*softplus(z) >= 0` to the floor, so the output is at or
  above the floor bitwise too. The bound is **non-strict**: `softplus` underflows
  to exactly zero for a very negative `z`, landing the output *on* the floor.
- The floor guarantee is **not bitwise on the reconstructed physical price**.
  `forward` multiplies by `A`, so the price carries an `A * (E/A)` round-trip
  that can land one unit in the last place below the analytic European value — a
  relative shortfall of order `1e-16`, immaterial against a `1e-6*A` material
  tolerance, but a near-bound rather than an exact one. The same qualification
  the premium head carries, for the same arithmetic reason.
- **No exponential can overflow.** `|a-b|` is non-negative so `exp` is only ever
  evaluated at a non-positive argument, and `softplus` switches to its exact
  linear branch above `z = 20`. Widely separated inputs underflow gracefully to
  the hard maximum and to the floor respectively.
- **First and second derivatives are finite everywhere.** The kinks of `max` and
  `abs` cancel: away from a tie the derivative is `sigmoid((a-b)/tau)` exactly,
  and at a tie it is `0.5`, the log-sum-exp value. `softplus`'s linear branch
  leaves the second derivative exactly zero above `z = 20` instead of
  `sigmoid'(20)/tau`, a step of about `2e-5` at `tau = 1e-4`; the function stays
  finite and continuously differentiable throughout.
- **Nothing at inference needs a lattice.** Both floor terms are computed from
  the seven physical contract inputs. **No CRR price is computed, read or
  required.**

**Predeclared interpretation, before it runs.**

- E2 is **exploratory and validation-selected**, like every task 9H attempt.
  Nothing it produces is a project result.
- It tests whether enforcing deployment-computable lower bounds can retain the
  direct residual model's price accuracy.
- The **intrinsic** and **analytic-European** bounds are enforced by
  construction, so the `intrinsic_lower_bound` count is expected to go to zero.
- The **stored CRR European comparator may remain violated**, because it is not
  the analytic value the head enforces. E2 makes **no claim** about that gate,
  and a non-zero `european_comparator_lower_bound` count is not evidence that
  the transformation failed. The schema-2 comparator discrepancy is what bounds
  how much of that gate any analytic floor could ever reach.
- E2 addresses **bounds, not volatility monotonicity**. E1 left 227
  `volatility_monotonicity` violations and nothing in this transformation
  targets them.
- **E3 remains conditional and is not implemented.** Nothing about it is built
  speculatively.

**Its one confound, stated.** The initialization seed differs from the parent's,
because it is derived from the attempt ID and an attempt configuration is
immutable. A single seed cannot separate a small effect from seed noise.

## The definition of "works", fixed before the first attempt

A candidate works when, on `validation`, it satisfies **Task 9G's criterion,
unchanged**:

| Check | Threshold |
|---|---|
| normalized RMSE | `<= 0.003` |
| normalized p99 absolute error | `<= 0.015` |
| normalized maximum absolute error | `<= 0.08` |
| material bound violations | `0` |
| material shape violations | `0` |

Errors are normalized by the discounted spot `A = S*exp(-q*T)`. Bound and shape
violations are Task 9G's: the intrinsic lower bound, the paired European CRR
comparator lower bound, the call and rate-aware put upper bounds, and spot
monotonicity, spot convexity and volatility monotonicity.

**No number above is restated in Task 9H code or configuration.** The criterion
is read from `configs/american_neural_pilot_acceptance_v1.toml`, section
`[validation_final_entry]`, and applied through
`differentiable_pricing.ml.american_pilot.assess_arm`, `sliced_metrics` and
`shape_diagnostics` — the same functions that judged Task 9G. That file is
digest-pinned by the Task 9G protocol and reconciled by
`scripts/check_american_neural_pilot_protocol.py` in both `scripts/check.sh` and
CI, so **the criterion cannot be quietly loosened after an attempt fails**.

There is no development-only revision of it. If a revision ever becomes
necessary, it is argued for in this section first, before the attempt that would
benefit from it runs.

**The reference is enforced, not merely conventional.** An attempt configuration
whose `paths.acceptance_config` is not
`configs/american_neural_pilot_acceptance_v1.toml` is refused; the acceptance
file's schema and `[validation_final_entry]` thresholds are validated before an
attempt directory is created; a report may only be recorded if it cites that
file and that section; and
`python3 scripts/american_dev_attempts.py check` re-verifies offline that every
logged attempt cites that file, that section, and the digest the tracked file
still hashes to.

Task 9G reached this criterion under `[validation_final_entry]` as a gate for
*entering* a final evaluation; task 9H reuses the same numbers as a development
target. The numbers are identical; only what happens next differs, and what
happens next in task 9H is another attempt, never a final evaluation.

## The workflow

1. an agent implements an attempt — code and one immutable configuration — and
   **runs nothing**;
2. the human reviews and commits the exact source state;
3. the human runs the attempt locally;
4. the result is appended to the attempt log;
5. an agent analyzes the compact summary;
6. the next attempt gets a **new** immutable ID and may respond to what the
   previous one measured.

## Initial experiment catalog

Five price-only candidates. All share the same dataset rows, shuffle order,
optimizer, training budget, checkpoint rule and validation metrics; only the
initialization seed label differs per candidate, since the architectures differ.
Every candidate uses the `american_forward_carry_v1` representation and the
`u = V / (S*exp(-q*T))` target, reconstructed as `V = A*u`.

| Attempt | What it changes | What it tests |
|---|---|---|
| `scratch_direct_control_v1` | nothing — `[64,64,64]` `tanh`, direct output | the control every later attempt is measured against |
| `scratch_capacity_v1` | `[256,256,256,256]` | whether Task 9G was capacity-limited |
| `scratch_residual_architecture_v1` | residual blocks, width 128, 6 blocks | whether identity shortcuts optimize better at comparable inputs |
| `scratch_american_premium_v1` | premium head over a European anchor | whether the residual over a known European price is an easier map |
| `scratch_conditioning_v1` | adds `european_price_ratio` and `intrinsic_ratio` | whether conditioning, not capacity, is the binding constraint |

**The premium head's guarantee, stated exactly.** The head produces the
*normalized* value `anchor + price_scale * softplus(raw)`, with the anchor the
analytic continuous-yield European price divided by `A = S*exp(-q*T)`. Two
qualifications keep the claim honest.

- The bound is **non-strict**: `softplus` underflows to exactly zero in float64
  for a sufficiently negative pre-activation, which collapses the price onto the
  anchor rather than below it.
- The bound is **not bitwise on the reconstructed physical price**. The
  reconstruction multiplies by `A`, so the price carries an `A * (E / A)`
  floating-point round-trip and can land **one unit in the last place below the
  European anchor** — a relative shortfall of order `1e-16`, immaterial against
  a `3e-3` normalized RMSE criterion, but a near-bound rather than an exact one.
  The earlier wording, "the reconstructed price cannot fall below it", overstated
  what the arithmetic delivers and is corrected here.

Every other bound remains a reported diagnostic.

**The conditioning features add no economic information.** Each is a
deterministic function of inputs the network already receives, so the candidate
tests conditioning, not extra knowledge.

## Progress: every completed attempt

Eight attempts have been run by the human. **None met the criterion.** Every
number below is a development measurement selected against `validation`; none is
a project result. `RMSE <= 0.003`, `p99 <= 0.015`, `max <= 0.08`, `bound = 0`
and `shape = 0` are the fixed gates.

| # | Attempt | What it changed | RMSE | p99 | max | bound | shape | best epoch | Principal outcome |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| 1 | `scratch_direct_control_v1` | nothing (the control) | 0.005725170440175143 | 0.020336 | 0.11873 | 10,121 | 1,460 | 120 | all five gates failed; the baseline everything else is measured against |
| 2 | `scratch_capacity_v1` | `[256]x4` dense | 0.003500912482274842 | 0.012953 | 0.065981 | 9,083 | 854 | 119 | 22.7x parameters bought price accuracy, not structure |
| 3 | `scratch_american_premium_v1` | premium head, control capacity | 0.006982320321969395 | 0.031738 | 0.13770 | 1,116 | 330 | 120 | European-comparator violations to **0**; price accuracy the worst so far |
| 4 | `scratch_conditioning_v1` | +2 deterministic features | 0.004223135357900415 | 0.015689 | 0.14800 | 9,205 | 1,537 | 119 | conditioning helps the average, not the tail or the structure |
| 5 | `scratch_residual_architecture_v1` | residual 128x6 | **0.002111941927713822** | **0.008157** | **0.031097** | 8,569 | 683 | 117 | **all three price gates passed**; structure untouched. The parent of 6 and 7 |
| 6 | `scratch_residual_premium_v1` (E1) | 5 + premium head | 0.006852205444033097 | 0.032496 | 0.13867 | 6,925 | 312 | 118 | premium target costs 3.2x RMSE at any capacity; analytic anchor leaves 5,839 CRR violations |
| 7 | `scratch_residual_smooth_floor_v1` (E2) | 5 + smooth floor, loss through the projection | 0.007201588210459 | 0.033562 | 0.093093 | 15,507 | **0** | **1** | collapsed onto the floor; gradient saturation, not a floor-versus-accuracy result |
| 8 | `scratch_residual_smooth_floor_raw_loss_v1` (E2b) | 7 + loss on the pre-projection value | **0.0013192586235663635** | **0.005480** | **0.019006** | 4,135 | 532 | 89 | **the price leader**; all three price gates passed with room to spare, intrinsic violations **0**, and the first best epoch selected before the end of the budget |

Reading across the table, three separable findings:

- **Price accuracy is solved by the residual backbone** (row 5), and by nothing
  else tried.
- **Structural validity is solved by architectural floors** (rows 3, 6, 7), and
  by nothing else tried. Row 7 drove shape violations to zero as well.
- **Rows 6 and 7 failed for two *different* reasons**: row 6 because the premium
  target is a harder regression, row 7 because its loss could not reach the
  network at all. Row 8 removed the second cause and obtained price accuracy and
  the enforced floor together, leaving the two structural gates — the stored CRR
  European comparator and shape — as the whole of the remaining failure.

## What E1 measured, and what it settled

`scratch_residual_premium_v1` put the premium head on the residual backbone.
Normalized RMSE **0.006852205444033097**, p99 **0.03249641860634549**, maximum
**0.13867056125662663** — within 2% of the small premium arm's RMSE at 22.7
times the parameters, and **3.2 times worse than its own direct parent's
0.0021119419277138224**, which passed all three price gates.

Two things follow, and both are load-bearing for E2.

- **Capacity was never the explanation for the premium head's price cost.** The
  same reparameterization costs the same accuracy at the smallest and at the
  largest capacity tried. **The premium target is therefore not reused.**
- **An analytic European anchor does not close the CRR comparator gate.** With
  the analytic floor enforced by construction, 5,839 `european_comparator_lower_bound`
  violations remained — with a maximum violation of only 0.0040 price units,
  the signature of a systematic small discrepancy rather than a modelling
  failure. Its structural gains were otherwise real: `spot_monotonicity` went to
  **zero**, `spot_convexity` to 85, shape violations to 312. `volatility_monotonicity`
  did not improve (227).

E1 was predeclared as a decision-gating diagnostic and it gated the decision it
was declared to gate: RMSE `0.00685 >= 0.005`, which is the branch that says
**preserve the direct-price target and implement the floor as an output
transformation**. That is E2.

## What E2 measured, and what it settled

`scratch_residual_smooth_floor_v1` put the smooth lower floor on the direct
residual backbone. It did not meet the criterion: best epoch **1** of 120,
normalized RMSE `0.007201588210459`, p99 `0.033561623029654215`, maximum
`0.09309287875649674`, **15,507** material bound violations and **0** material
shape violations — the first attempt in the loop to reach zero on the shape
gate, and all of the remaining bound violations against the stored CRR European
comparator rather than against the intrinsic floor the head enforces.

**A best epoch of 1 out of 120 is not a slow-learning signal; it is the
signature of a model that never moved.** The mechanism is the projection's
derivative, `sigmoid((raw_u - floor) / tau)`. At `tau = 1e-4` that factor is of
order `1e-44` once a prediction sits a hundredth of a discounted spot below the
floor — which is where a scratch network starts. The gradient reaching the
network is multiplied by it, so the loss could not pull the prediction up, the
model remained pinned to the floor, and every reported number describes the
floor rather than a fitted price model.

**E2 is therefore not evidence that an enforced floor is incompatible with price
accuracy.** It is evidence that the loss must not be differentiated through a
saturated projection. That distinction is what E2b tests.

## Predeclared candidate: `scratch_residual_smooth_floor_raw_loss_v1` (E2b)

`configs/american_dev_attempt_scratch_residual_smooth_floor_raw_loss_v1.toml`.
Relative to E2, exactly one behavioural field moves: `head`
`smooth_lower_floor` → `smooth_lower_floor_raw_loss`, plus the initialization
seed the existing rule derives from the new attempt ID. Everything else is E2's,
which is in turn the direct residual parent's: the backbone, the five base
features, the direct normalized-price target, the reconstruction, the
`tau = 1e-4` floor, the selected training rows, the shuffle seed, the validation
partition, the optimizer, the schedule, the epochs, the batch sizes, the
precision, the CPU thread count, the checkpoint semantics and the criterion.

**The deployed output is E2's, bit for bit.** The two heads apply the same
floor, the same temperature and the same projection — a test asserts their
outputs are bitwise equal for identical weights. They differ only in which
prediction the **training loss** is computed against:

```
raw_standardized = residual_network(features)
raw_u            = raw_standardized * price_scale + price_mean
floor            = smooth_max(E_analytic / A, intrinsic / A)
final_u          = floor + tau * softplus((raw_u - floor) / tau)

loss             = MSE(raw_standardized, standardized_direct_price_target)
```

`AmericanDevPriceModel.training_target` returns `raw_u` for this head and
`normalized_target` — the deployed output — for every other head. That one
method is the entire distinction; there is no loss framework, no configurable
objective and no new dispatch axis.

**Checkpoint selection, price metrics, bound diagnostics and shape diagnostics
all read `final_u`.** The epoch chosen has to be the one that is best at what
will actually be evaluated and deployed, not at the latent problem. Only the
batch loss sees `raw_u`.

**Predeclared interpretation, before it runs.**

- E2b is **exploratory and validation-selected**, like every task 9H attempt.
  Nothing it produces is a project result.
- It tests one thing: whether E2's failure was **optimization** rather than an
  incompatibility between the enforced floor and price accuracy. A result near
  the direct residual parent's `0.002111941927713822` supports the optimization
  explanation; a result near E2's `0.007201588210459` refutes it and means the
  floor itself costs the accuracy.
- The **intrinsic** and **analytic-European** bounds stay enforced by
  construction, because the deployed transformation is unchanged.
- The **stored CRR European comparator may remain violated**, for the reason it
  did in E1 and E2: it is not the analytic value the head enforces. A non-zero
  count there is not evidence that E2b failed.
- E2b targets the optimization failure. It does **not** target volatility
  monotonicity.
- **E3 remains conditional and is not implemented.**

**What it deliberately does not use.** No premium target, no warm start, no
auxiliary loss, no shape penalty, no conditioning feature, no different
architecture, no different temperature, no longer budget.

**Its one confound, stated.** The initialization seed differs from E2's, because
it is derived from the attempt ID and an attempt configuration is immutable. A
single seed cannot separate a small effect from seed noise — though the effect
this attempt is looking for is a factor of three in RMSE, not a small one.

## The two label-free analyses

E2b passed all three price gates and failed only the two structural ones, with
**4,135** of its remaining violations against the **stored CRR** European
comparator rather than against the analytic value its floor enforces. Two
questions follow, and neither needs a label, a partition or a training run.
Both analyses below are exploratory; both write only beneath the ignored
`artifacts/` tree; and **neither modifies any configuration or source file**.
Both have now been run once by the human, and their results are recorded at the
end of each subsection.

### A. The European CRR-versus-Black-Scholes domain characterization

`python/src/differentiable_pricing/ml/american_dev/domain.py`, run by
`python3 scripts/analyze_american_dev_domain.py analyze`.

It measures `(E_CRR - E_BS) / A` over the **declared input domain**, where

- `E_CRR` is the **stored comparator's own semantics**: the European CRR price
  formed the way the label policy `american-crr-adjacent-average/1` forms it,
  `0.5 * (E_CRR(N) + E_CRR(N+1))` at `N = [label].steps = 1024`, on the same
  lattice family and through the compiled engine's batch boundary. **No dataset
  column is read** — the comparator is recomputed;
- `E_BS` is `american_dev.representation.european_price_array`, **the same
  analytic function the `smooth_lower_floor` head enforces at inference**, so
  the measurement and the enforcement cannot drift apart;
- `A = spot * exp(-dividend_yield * maturity)`, the normalized target's own
  reconstruction scale;
- the declared domain is the `[domain]` table of
  `configs/american_option_dataset_v1.toml`, read through a whitelist of its six
  coordinates — the per-partition row table is never resolved — and re-verified
  against the digest the locked Task 9G protocol pins for that file. Its
  `spot`, `volatility` and `log_moneyness` intervals are additionally required
  to agree with the acceptance file's own diagnostics domain.

**The sample.** Four deterministic constructions, each evaluated at **both**
option types: a 131,072-point Halton sequence in the six declared coordinates
(bases 2, 3, 5, 7, 11, 13, leading 1,024 points skipped), a structured sweep of
the numerically hardest regions (both maturity, volatility, rate, dividend-yield
and spot endpoints crossed with a fine near-the-money log-moneyness sweep and
both moneyness endpoints), every vertex of the declared box, and one centred
point per box face. **No sampling location is derived from any partition**, and
the module imports no partition machinery at all.

**The margin rule, predeclared in code before the run:**

```text
domain_supremum = maximum positive normalized CRR-minus-BS gap
candidate       = ceil_to_1e-5(2 * domain_supremum)
delta           = max(1e-4, candidate)
```

It lives in `domain.MARGIN_SAFETY_FACTOR`, `domain.MARGIN_QUANTUM`,
`domain.MINIMUM_MARGIN` and the single function `domain.derive_margin`, so it has
**no free parameter left to tune** once the measurement lands. The report carries
the sample construction and counts, the CRR resolution semantics, the maximum and
the p50/p90/p95/p99/p999 quantiles, the fraction strictly above `0`, `1e-6`,
`1e-5` and `1e-4`, the derived `delta`, and the realized safety factor against
the measured supremum.

**What it is not.** It derives a **candidate margin** and nothing else. It admits
nothing, implements no attempt, and applying the margin would be a separate,
separately predeclared attempt with its own configuration and its own recorded
result. A finite sample bounds the gap where it looks; it is not a proof of a
supremum over the continuum, which is why the rule carries a predeclared safety
factor and a floor rather than the measured maximum itself. The declared domain
is the bounding box of the sampling strata, so a supremum over it is
conservative for the dataset and says nothing about states outside it.

**Disclosed contamination.** The validation-set supremum of this same quantity
was already observed by the geometry analysis and cannot be un-seen. The
derivation uses no partition row and no sampling location derived from one, and
the rule was fixed in code before the run; the report records the exposure rather
than arguing it away.

**Result.** Measured supremum of `(E_CRR - E_BS) / A` over the declared domain:
**`4.192769575172157e-05`**. Under the rule, `2 * supremum =
8.385539150344314e-05`, `ceil_to_1e-5` gives `9e-5`, and the rule's declared
minimum binds: **`delta = 1e-4`**, a realized safety factor of **2.385** over the
measured supremum. The margin is therefore the rule's floor, not a value fitted
to the measurement.

### B. The matched E2b latency diagnostic

`python/src/differentiable_pricing/ml/american_dev/latency.py`, run by
`python3 scripts/benchmark_american_dev_latency.py benchmark`.

**Task 9H's scope is price only and it makes no latency claim.** This is an
**ungated exploratory diagnostic with no validation exposure**, recorded because
the decision about what to build next depends on it. **No gate is applied**:
Task 9G's bar is written into the report as context and explicitly not
evaluated, and a Task 9H measurement neither passes nor fails a Task 9G gate.

It benchmarks the recorded E2b checkpoint at
`artifacts/task-9h/scratch_residual_smooth_floor_raw_loss_v1/checkpoint.pt`,
rebuilt from the **tracked immutable attempt configuration** plus the train-fitted
scaling the attempt report recorded. The configuration must still hash to the
digest the attempt ran, and the attempt must appear exactly once in the
append-only log under that digest, so an unrecorded checkpoint cannot be
benchmarked. **No partition is opened**: the scaling is read, never refitted.

**The contract is Task 9G's, reused rather than restated.** The measurement is
performed by `ml.american_pilot.run_latency` itself, driven by
`configs/american_neural_pilot_latency_cases_v1.toml`, whose digest is
re-verified against the value the locked Task 9G protocol pins. The clock
(`time.perf_counter_ns`), the two warm-ups, the seven repetitions, the
deterministic cyclic measurement rotation, the request shapes (batch 1 and
batch 8), the per-shape thread budgets (1 and 4), the inter-op budget of 1, the
eight fixed synthetic cases, the CRR operation `0.5 * (CRR(N) + CRR(N+1))` priced
through the compiled batch boundary, and the timed neural region — feature
construction, standardization, inference, inverse target transform and physical
reconstruction, which for this head includes the analytic-European and intrinsic
floor and the physical price reconstruction — are all the contract's and the
reused implementation's. **Artifact loading is excluded from the timed region**,
exactly as Task 9G excludes it.

**Two deviations from the Task 9G run, recorded in the report rather than
glossed:** the CRR depth ladder is restricted to the matched depth `N = 1024`
(priced as `1024/1025`), the label policy's own resolution and the depth the
acceptance file names for interpretation; and one model is timed instead of two
arms, so the deterministic rotation alternates between two operations rather than
three. No other semantic changes.

**Reported:** per request shape, the median and p95 nanoseconds of the matched
CRR comparator and of the neural end-to-end path, and the paired median speedup
with the contract's own distribution-free interval. Seven repetitions on one
machine in one process: it characterizes this machine under these conditions and
is not a portable performance claim.

**Result.** Median end-to-end speedup **3.36** at batch 1 and **7.88** at
batch 8. Both are below Task 9G's reference bar of 10, which is recorded here as
context and **not applied**: this diagnostic is ungated and neither passes nor
fails a Task 9G gate.

**What it changes in this task.** Nothing about the criterion, which is price
only and unrevised. One thing about sequencing: a shape-penalty attempt would be
the loop's first tunable weight and its most expensive remaining investment, and
spending it on a model whose end-to-end cost is this far from the objective that
motivates a surrogate is not warranted on this measurement. **E3 is therefore not
authorized**, is not implemented, and nothing about it is built speculatively.

## Predeclared candidate: `scratch_residual_smooth_floor_margin_v1` (E2c)

`configs/american_dev_attempt_scratch_residual_smooth_floor_margin_v1.toml`.
Relative to E2b, exactly one behavioural field moves: `head`
`smooth_lower_floor_raw_loss` → `smooth_lower_floor_margin_raw_loss`, plus the
initialization seed the existing rule derives from the new attempt ID. Everything
else is E2b's: the backbone (`smooth_residual`, width 128, 6 blocks, `tanh`, no
normalization, no dropout), the five base features, the direct normalized-price
target, the raw-price training loss, the reconstruction, `tau = 1e-4`, the
selected training rows, the shuffle seed, the validation partition, the
optimizer, the schedule, the epochs, the batch sizes, the precision, the CPU
thread count, the checkpoint semantics and the criterion.

**The one change.**

```
raw_standardized = residual_network(features)
raw_u            = raw_standardized * price_scale + price_mean
floor            = smooth_max(E_analytic / A + delta, intrinsic / A)
final_u          = floor + tau * softplus((raw_u - floor) / tau)

loss             = MSE(raw_standardized, standardized_direct_price_target)
```

`delta` is added to the **European leg only**, before the existing smooth
maximum. The intrinsic leg is the same object in the floor and in the
`intrinsic_lower_bound` diagnostic, so it carries no discretization gap; lifting
it would buy nothing and would bias the deep-in-the-money region. At `delta = 0`
the floor is **bitwise** the floor that already ran, so no earlier head moves.

**Why a margin, and why not a temperature.** The deployed floor enforces the
**analytic** European value; the `european_comparator_lower_bound` diagnostic
compares against the dataset's **stored CRR** European leg. The two differ by the
lattice's own discretization error, so a prediction resting on the analytic floor
is counted as violating the stored comparator wherever that difference exceeds
the `1e-6` material tolerance. A zero-margin analytic floor therefore cannot
satisfy that gate at any accuracy. Temperature is not a substitute lever: the
smooth maximum's own margin above the hard maximum is `tau*log(2)` only where the
two legs are equal, and decays to zero where the prediction is projected from
below — which is exactly where the violations are.

**`delta` is derived, and it is not in the configuration.** It is pinned in code
as `american_dev.attempts.EUROPEAN_FLOOR_MARGIN = 1e-4`, for the reason `tau` is:
every attempt configuration declares exactly the same top-level keys, so a
per-attempt field would have to be added to the configurations that already ran.
It is therefore in `source_digests` and recorded in the attempt report.
Its value is the label-free domain characterization's, through the rule
predeclared in code before that analysis ran: measured supremum
`4.192769575172157e-05`, `ceil_to_1e-5(2 * supremum) = 9e-5`, and
`delta = max(1e-4, 9e-5) = 1e-4` — the rule's declared minimum, at a realized
safety factor of 2.385. `attempts.assert_margin_consistent` re-derives from the
digest-pinned acceptance file that `1e-4` sits strictly between the material
tolerance `1e-6` and the normalized RMSE limit `3e-3`; the check runs in
`scripts/check.sh`, in CI and in the runner's pre-flight.

**Predeclared interpretation, before it runs.**

- E2c is **exploratory and validation-selected**, like every task 9H attempt.
  Nothing it produces is a project result.
- It tests one thing: whether E2b's residual `european_comparator_lower_bound`
  violations are floor **placement** rather than model quality. Zero material
  violations on that check supports the placement account; a non-zero count with
  `delta` above the measured domain supremum refutes it, and which of the two
  causes — a state where the gap exceeds the characterization, or a wrong account
  of which rows are projected — is diagnosed before anything is changed.
- The **intrinsic** and **margined analytic-European** bounds stay enforced by
  construction. The intrinsic count is expected to stay at zero.
- The price gates are expected to hold. Wherever the projection binds, the margin
  is added to the prediction, and `1e-4` is a thirtieth of the normalized RMSE
  limit; a price gate degrading materially would mean the account of which rows
  are projected is wrong, and is a reason to stop and re-analyze rather than to
  adjust `delta`.
- E2c **does not target shape**. Nothing here addresses spot monotonicity, spot
  convexity or volatility monotonicity, and raising the floor changes which rows
  are projected, so the shape counts may move in either direction without that
  being evidence about this change.
- **E3 remains unauthorized and is not implemented.** No shape penalty, no
  tunable loss weight, no architecture alternative and no framework is built.

**Its confounds, stated.**

- The initialization seed differs from E2b's, because it is derived from the
  attempt ID and an attempt configuration is immutable. A single seed cannot
  separate a small effect from seed noise.
- `delta` was fixed after the validation-set distribution of the same
  CRR-versus-analytic quantity had already been observed by the geometry
  analysis. The rule was predeclared in code before the domain characterization
  ran, the derivation used no partition row and no partition-derived sampling
  location, and the rule's minimum — not the measurement — binds. The exposure is
  disclosed, not argued away.
- Satisfying an exact-zero gate this way is **engineering a characterized margin,
  not a mathematical guarantee** that the analytic floor dominates the stored
  comparator everywhere. A finite deterministic sample bounds the gap where it
  looks. Any confirmation protocol has to carry that qualification.

## The exploratory validation-set geometry analysis

**Predeclared, exploratory, and not a candidate for anything.** Every attempt's
structural failure is counted against a material tolerance of `1e-6` in
normalized units, and how many rows a model can fail that way is a joint
property of the model *and* of how much slack the labels leave above each bound.
That second half has never been measured. Deciding between architectural floor
designs without it would be guessing.

The analysis measures, on `validation` only, in the repository's existing
definitions:

- the normalized European slack `(V_American - V_European_CRR) / A`, using the
  stored CRR European leg — the same comparator the
  `european_comparator_lower_bound` diagnostic uses;
- the normalized intrinsic slack `(V_American - intrinsic) / A`;
- the normalized early-exercise premium, reported for the positive-premium
  population separately;
- for each, quantiles, row counts, and the fraction of rows at or below `1e-6`,
  `1e-4`, `1e-3` and the recorded normalized RMSE of the control, capacity and
  residual attempts — read out of the attempt log, never typed in;

sliced by `ml.american_pilot`'s own validation slices rather than by new ones.

**Schema 2 adds the comparator discrepancy.** E1 showed that enforcing the
analytic European value leaves thousands of stored-CRR violations, so the
remaining question is quantitative: how far above the analytic value does the
stored CRR leg sit? The extension measures

```
(V_European_CRR - V_European_BS) / A
```

with the CRR leg the stored column the `european_comparator_lower_bound`
diagnostic uses and the Black–Scholes leg
`american_dev.representation.european_price_array` — **the same analytic
function the E2 head enforces**, so the measurement and the enforcement cannot
drift apart. It reports counts and fractions **strictly above** `0`, `1e-6` and
`1e-4` (matching `shape_diagnostics`, which counts `value > tolerance`), the
p50, p90, p95, p99 and maximum, on every repository slice and on the rows with
**effectively zero American premium** — defined as normalized premium `<=` the
acceptance file's own `material_normalized_tolerance`, reported beside the exact-zero
`premium_status:zero` slice rather than instead of it. A positive value is not a
model defect; the sign convention is fixed in the report so it cannot be read
backwards.

**Schema 2 writes to `validation-geometry-v2.json`.** The schema-1 report an
earlier run already published is left byte-for-byte intact: a published
measurement is evidence, and this extension adds a file rather than overwriting
one.

**What it is not.** It measures the geometry of the *labels*, not any model's
behavior. It predicts no violation count, admits no candidate, and revises no
threshold. It is measured on the partition this loop selects against, so it
carries the same selection bias as every other task 9H number and is labelled
that way inside its own report payload.

**How it is constrained.** `validation` is a module constant in
`ml.american_dev.geometry`, not an argument, a flag or a configuration key;
`python3 scripts/american_dev_attempts.py check` re-verifies offline that it
still is, that neither the module nor its script spells any other split name as
a literal, and that the script declares exactly `analyze` and `show`. The
analysis reuses the hardened manifest restriction, dataset-identity pinning,
row-level policy verification and committed-source provenance guards unchanged,
so `train` is hashed for identity while only `validation` is read as data — the
report says exactly that. It trains nothing, reserves no attempt directory or
ledger, appends nothing to the attempt log, and mutates no existing attempt
evidence. Its output lives beneath the ignored `artifacts/` tree and is never
committed.

## Predeclared diagnostic: `scratch_residual_premium_v1`

`configs/american_dev_attempt_scratch_residual_premium_v1.toml`. Composition
only: relative to its parent `scratch_residual_architecture_v1` exactly one
behavioral field moves, `head` `direct` → `premium_over_european`, plus the
initialization seed that the existing rule derives from the new attempt ID. The
backbone (`smooth_residual`, width 128, 6 blocks, `tanh`, no normalization, no
dropout), the five base features, the representation, the target, the
reconstruction, the selected training rows, the shuffle seed, the validation
partition, the optimizer, the schedule, the budget, the precision, the CPU
thread count, the checkpoint semantics and the criterion are unchanged. **No
model or training code changes for it**; both components already exist and are
dispatched.

**Predeclared interpretation, before it runs.** This is a **decision-gating
diagnostic, not yet a candidate.** No currently implemented head enforces the
intrinsic floor, so it is **not expected** to satisfy the complete
zero-bound-violation criterion: `scratch_american_premium_v1` left 1,116
`intrinsic_lower_bound` violations, which a European anchor does not address.
Its purpose is to isolate the effect of the premium head at residual-backbone
capacity. Failing the structural criterion is therefore the expected outcome and
is **not** grounds for revising the criterion — the criterion is not revised
because an attempt failed.

**Predeclared decision rule, before it runs.**

- If all three price gates pass **and** European violations are zero, proceed
  toward E2.
- If normalized RMSE is **at most 0.0045**, use the gentler E2 design.
- If normalized RMSE is **at least 0.005**, preserve the direct-price target and
  implement the floor as an **output transformation** instead of reusing the
  premium target.
- If any price metric lands **within 10% of its gate**, run two additional
  initialization seeds before drawing any conclusion.

E2 and E3 are **not implemented and not designed here**. They are conditional on
this result, and nothing about them is built speculatively.

**Its one confound, stated.** The initialization seed differs from the parent's,
because it is derived from the attempt ID and an attempt configuration is
immutable. A single seed cannot separate a small effect from seed noise, which
is exactly what the fourth branch of the decision rule exists to catch.

## Files

| Path | What it is |
|---|---|
| `python/src/differentiable_pricing/ml/american_dev/attempts.py` | partition guard and its single token list, row/seed selection, digests, clean-tree and committed-source checks, strict configuration validation, attempt-log rules. PyTorch-free |
| `python/src/differentiable_pricing/ml/american_dev/representation.py` | the five coordinates, the European anchor, conditioning features, heads (including `smooth_lower_floor` and its smooth maximum and one-sided projection), physical reconstruction |
| `python/src/differentiable_pricing/ml/american_dev/models.py` | the dense and residual networks, and their dispatch |
| `python/src/differentiable_pricing/ml/american_dev/workbench.py` | pre-flight, dataset identity pinning and row-level policy verification, one attempt end to end, evaluated against the reused Task 9G criterion |
| `python/src/differentiable_pricing/ml/american_dev/geometry.py` | the exploratory validation-set geometry of the binding constraints, including the CRR-versus-analytic comparator discrepancy; reads one partition, trains nothing, writes no attempt evidence |
| `python/src/differentiable_pricing/ml/american_dev/domain.py` | the label-free European CRR-versus-Black-Scholes characterization over the declared domain, and the predeclared additive-margin rule; opens no partition, trains nothing, modifies no configuration or source |
| `python/src/differentiable_pricing/ml/american_dev/latency.py` | the ungated matched latency diagnostic for one recorded checkpoint, measured by Task 9G's own `run_latency` under Task 9G's own contract; opens no partition, trains nothing |
| `configs/american_dev_attempt_scratch_*.toml` | the nine immutable attempt configurations |
| `scripts/run_american_dev_attempt.py` | **manual**: `run`, `status` — and no third command |
| `scripts/analyze_american_dev_geometry.py` | **manual**: `analyze`, `show` — and no third command |
| `scripts/analyze_american_dev_domain.py` | **manual**: `analyze`, `show` — and no third command |
| `scripts/benchmark_american_dev_latency.py` | **manual**: `benchmark`, `show` — and no third command |
| `scripts/american_dev_attempts.py` | offline: `record` one attempt, `check` the log, the configurations and the geometry analysis's validation-only restriction |
| `docs/attempts/task-9h-attempt-log.jsonl` | the append-only recorded search |

Checkpoints, reports, compact summaries and the run ledger live beneath the
ignored `artifacts/` and `runs/` trees and are never committed.

## Commands

Training is a manual, terminal-invoked human command. An agent reports it and
stops.

```
python3 scripts/run_american_dev_attempt.py run \
    --config configs/american_dev_attempt_scratch_residual_smooth_floor_margin_v1.toml

python3 scripts/run_american_dev_attempt.py status \
    --config configs/american_dev_attempt_scratch_residual_smooth_floor_margin_v1.toml

python3 scripts/american_dev_attempts.py record \
    --report artifacts/task-9h/<attempt_id>/attempt-report.json \
    --outcome criterion_met|criterion_not_met|abandoned|infrastructure_failure \
    --interpretation "<one honest sentence>" \
    --next-action "continue|stop|change-direction, and why"
```

The exploratory geometry analysis is also a manual, terminal-invoked human
command — it opens a dataset partition, so no test, hook, CI job or repository
check calls it. It trains nothing and records no attempt.

```
python3 scripts/analyze_american_dev_geometry.py analyze

python3 scripts/analyze_american_dev_geometry.py show
```

The two label-free analyses are manual, terminal-invoked human commands too. The
domain characterization prices hundreds of thousands of lattices and the latency
diagnostic is a timing measurement that needs a quiet machine, so no test, hook,
CI job or repository check calls either one. Neither opens a dataset partition,
neither trains anything, neither records an attempt, and neither modifies a
configuration or a source file.

```
python3 scripts/analyze_american_dev_domain.py analyze
python3 scripts/analyze_american_dev_domain.py show

python3 scripts/benchmark_american_dev_latency.py benchmark
python3 scripts/benchmark_american_dev_latency.py show
```

Agent-safe and offline, run by `scripts/check.sh` and CI:

```
python3 scripts/american_dev_attempts.py check
```

There is deliberately no final-evaluation command and no flag that adds one.

## A genuine infrastructure failure

Everything checkable is checked **before** the attempt's output directory is
created: configuration validity, containment inside the repository, the
supported-value rules, the acceptance schema and section, the locked dataset
paths, and committed source. A refusal at that stage reserves nothing — no
directory, no ledger — and the attempt ID is still unused, so the same
configuration may simply be run again once the cause is fixed.

Once the directory exists the attempt ID is **spent**. A crash after that point
leaves `status="failed"` and the recorded failure in the run ledger, the output
directory in place, and `execute_attempt` refuses that ID from then on. The
remedy is to record the dead attempt with `--outcome infrastructure_failure` and
an honest interpretation, then give the retry a **new** attempt ID and a **new**
configuration file. Deleting the output directory to reuse the ID would erase
the evidence that the first run happened, and is never the remedy — that is the
difference between a recorded search and a search that quietly reports only its
last try.

## Non-claims

- Nothing task 9H produces is a project result. It selects against
  `validation`, repeatedly, so every task 9H number carries selection bias.
- Task 9H makes and measures **no Greek, latency or implied-volatility claim**,
  and no transfer-learning claim. The matched latency diagnostic is **ungated**:
  no gate is applied to it, it neither passes nor fails a Task 9G gate, and it is
  not offered as evidence for or against any hypothesis.
- The domain characterization derives a **candidate** additive margin from the
  declared domain. It admits nothing and revises no threshold. E2c applies that
  margin as its own predeclared attempt.
- Satisfying the `european_comparator_lower_bound` gate with an additive margin
  is **engineering a characterized margin, not a mathematical guarantee** that
  the analytic floor dominates the stored CRR comparator over the whole domain.
- **E3 is not authorized and is not implemented.** No shape-penalty machinery,
  tunable loss weight, architecture alternative or framework exists in task 9H.
- The validation-geometry report is an **exploratory measurement of the labels**
  on the partition this loop selects against. It is not frozen evidence, is
  never committed, admits no candidate and revises no threshold.
- A predeclared attempt configuration is a **plan, not a measurement**. Nothing
  about `scratch_residual_smooth_floor_raw_loss_v1` is a result until the human
  has run it.
- **E2 is not evidence about floors versus accuracy.** Its model never left the
  floor, so its price metrics describe the floor, not a fitted model.
- Enforcing the analytic European floor establishes **nothing** about the stored
  CRR European comparator gate. The two differ by the lattice's own
  discretization error, and E2 claims only the bounds it computes.
- Task 9H establishes no H2 result, no converged American-price truth, no OOD
  behavior, no discrete-dividend applicability and no market performance.
- Task 9E's conditional, mapping-only dataset admission is unchanged.

## Recorded limitation

Task 9H adds no wording to any `docs/*-contract.md` or to
[architecture.md](../../architecture.md). Those files are digest-pinned as Task
9G tracked inputs, and editing one would break the historical Task 9G protocol
check that `scripts/check.sh` and CI both run. This specification is therefore
the authority for task 9H's conventions until a later task can promote them.
