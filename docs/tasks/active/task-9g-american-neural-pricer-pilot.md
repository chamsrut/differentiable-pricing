# Task 9G: continuous-yield American neural-pricer feasibility pilot

## Status

`status=validation_gates_failed; outcome=failure_to_learn. Implementation
merged at e930454; the single human-invoked locked run-to-validation completed;
neither arm passed every validation gate; final_evaluation_attempts=0;
final_partition_consumed=false; final evaluation is forbidden under this
protocol.`
This specification defined the active task after the task 9E planning
reconciliation. Fresh independent review of that reconciliation was discharged
by `APPROVE PLANNING RECONCILIATION` (DEC-035). The protocol-and-implementation
change was built (DEC-036); after its first fresh implementation review's
`REQUEST CHANGES` findings were addressed, a fresh top-level review of the
cumulative branch at `8d27c23` returned exactly `APPROVE TASK 9G IMPLEMENTATION
FOR MERGE` (DEC-037), and the branch merged at
`e930454d1c0869e22778a99ae22505f093889e73`. The human operator then invoked the
locked `run-to-validation` once. Its frozen result is recorded below and in
DEC-038. The pilot is closed with a negative, honest outcome: the experiment
ran as predeclared, and the predeclared feasibility gates were not met. One
seed and one budget do not establish H2, and this result does not either.
Material acceptance of this recorded result still requires its own fresh
top-level review of the result-only change.

## Recorded result

The single locked `run-to-validation` run was executed manually by the human
operator, and its raw report was frozen through the designated tool. The frozen
snapshot is
[../../results/american_neural_pilot_results_v1.json](../../results/american_neural_pilot_results_v1.json),
distilled from validation report SHA-256
`9a4acdfd3b96eee3d29ef9c12c467ea304297dc44d561f56fdaeb2292ba999a9` under
protocol SHA-256
`6600a46ea2132bd3c834645ddb704ccc71069a0cca84d3762b4c4ca753aac591`. The
snapshot is authoritative for every number; this section states only the
outcome facts. It is validated offline by
`python scripts/freeze_american_neural_pilot_results.py --check`, which reads
only tracked files and reruns no pricing, training, latency measurement, or IV
inversion.

- **Status `validation_gates_failed`; outcome `failure_to_learn`**, recorded at
  `outcome.phase = "validation"` with `lifecycle.state = "validation_terminal"`.
- **The independent PDE mapping check passed on 21 of 21 rows**, before any
  optimization. It is mapping-consistency evidence only — not converged
  American-price truth and not semantic-coverage evidence — and the fixed-domain
  400x200 versus 800x400 refinement pair does not independently bound
  domain-truncation error.
- **The exact transfer lift passed** at all eight protocol-pinned probes:
  maximum absolute difference `1.4210854715202004e-14` and maximum relative
  difference `3.6214823824845716e-13`, against `1e-12` absolute and relative
  tolerances. No general transfer framework was built.
- **Neither arm passed every validation gate.** `scratch` failed all five
  predeclared checks; `transfer` passed only `normalized_p99_absolute_error`
  and failed the other four. `outcome.arm_accuracy_passed` is `false` for both
  arms, and `all_iv_errors_passed` and `all_reference_speedups_passed` are both
  `false`.
- **Transfer's validation RMSE was strictly better than scratch's**
  (`outcome.transfer_validation_rmse_strictly_better = true`). This is a
  within-pilot comparison only: the two locked arm seeds also produce different
  epoch shuffle permutations, so this one-seed pilot cannot attribute the
  difference solely to transfer initialization. It is not evidence of transfer
  value, and it does not convert a failed gate into a pass.
- **`lifecycle.final_evaluation_attempts = 0` and
  `lifecycle.final_partition_consumed = false`.** `interpolation_test` was
  never opened, hashed, imported, or counted.
- **Final evaluation is forbidden.** `lifecycle.validation_final_entry_passed`
  is `false` and `lifecycle.second_attempt_allowed` is `false`, so the
  protocol's final-entry rule is unmet and `final-evaluate` may not be invoked
  under this protocol — now or later. Any further American neural work requires
  a new predeclared protocol and, when it reaches a final evaluation, a fresh
  final partition. The unconsumed `interpolation_test` partition is never tuned
  against.
- **One seed and one budget do not establish H2.** This pilot cannot, by
  construction, do so; a promising outcome would have required a separate
  replication with at least five seeds and several predeclared label budgets,
  and this outcome was not promising.

## Objective

Determine whether one fixed five-input neural architecture can learn the known
continuous-dividend-yield American CRR mapping at useful in-envelope accuracy,
and whether exact European-weight lifting is promising relative to matched
scratch initialization under one fixed training budget. A positive pilot is a
reason to predeclare a separate H2 replication. Negative transfer, failure to
learn, an unrecoverable source artifact, or an impossible exact lift are valid,
reportable pilot outcomes. None is worked around inside this task.

## Motivation

Task 9E left the local dataset conditionally admitted only for learning its
known CRR mapping; it did not establish independent American-price accuracy or
semantic coverage, and it did not authorize training
([../../american-crr-dataset-admission.md](../../american-crr-dataset-admission.md),
DEC-034). The locked roadmap nevertheless makes this bounded feasibility pilot
the first stage-2 learning experiment. Running it cheaply, with hard entry and
one-shot gates, decides whether the later multi-seed H2 replication and the
XSP/SPY phase are worth their larger label and engineering costs.

## Authoritative inputs

- [../../research-contract.md](../../research-contract.md), especially the
  locked stage-2 roadmap, data protocol, metrics, transfer-learning experiment,
  and phase-1 pilot/H2 distinction.
- [../../architecture.md](../../architecture.md), especially
  `american_forward_carry_v1`, physical reconstruction, artifact identity, and
  the exact transfer-lift contract.
- [../../american-crr-contract.md](../../american-crr-contract.md), including
  the adjacent-average label semantics and its non-claims.
- [../../american-crr-dataset-admission.md](../../american-crr-dataset-admission.md)
  and [task 9E](task-9e-crr-dataset-admission.md), as reconciled and freshly
  reviewed before implementation becomes load-bearing.
- [../../results/european_replication_results_v1.json](../../results/european_replication_results_v1.json),
  the frozen source-model identity and result. Its unconstrained
  `weights.npz` digest is
  `42670774f736383e50818b6e6c1db9374a77988173e35423ffc34b3c4297ecb8`.
- `configs/american_option_dataset_v1.toml`, generator version `1.0.0`, with
  recovered configuration SHA-256
  `d18485c66b92c720c57bef6820e7f6cdb7204159c2dcf8d47d8f9c744cb28c98`.
- The protocol, training configuration, acceptance configuration, latency
  cases, and IV cases added and digest-locked by the protocol/implementation
  PR, all now merged and tracked:
  `configs/american_neural_pilot_protocol_v1.toml` (SHA-256
  `6600a46ea2132bd3c834645ddb704ccc71069a0cca84d3762b4c4ca753aac591`),
  `configs/american_neural_pilot_training_v1.toml`,
  `configs/american_neural_pilot_acceptance_v1.toml`,
  `configs/american_neural_pilot_latency_cases_v1.toml`, and
  `configs/american_neural_pilot_iv_cases_v1.toml`. The executed run is pinned
  to those digests, and this result-only closure changes none of them.

## Entry gates

Every gate is fail-closed and was checked before training opened `train`.
Every gate below passed in the executed run; none was waived, loosened, or
retried:

1. **Admission review — discharged:** fresh independent review returned
   `APPROVE PLANNING RECONCILIATION` (DEC-035), accepting only the reconciled
   task 9E conditional admission for learning the known CRR mapping, not
   converged American-price accuracy or training authorization.
2. **Independent numerical check — discharged in the executed run:** the
   outstanding LSM/PDE price cross-check ran under the predeclared protocol
   inside `run-to-validation`, before optimization, and passed on 21 of 21
   rows. Silence or an internal same-lattice identity would not have been a
   resolution. The check establishes mapping consistency only.
3. **Row/manifest policy invariants — implemented:** implementation and tests
   require every row's `label_policy == manifest.label_policy.name` and every
   row's `label_steps == manifest.label_policy.steps`.
4. **Dataset and generator/config identity pins — discharged in the executed
   run:** entry checks required schema `american-option-dataset/1`, generator
   version `1.0.0`, the recovered configuration digest above, and the
   manifest's declared file identities, and they passed.
   Before training, only `train` and `validation` files were opened and hashed;
   the final split's digest is checked only inside a recorded one-shot final
   evaluation, which never occurred. A mismatch would have stopped the task.
5. **European source-weight recovery and digest verification — preflight discharged, runtime recheck implemented:** the
   original unconstrained European `weights.npz` is
   locally available and hashes to the frozen digest above. The file remains
   ignored. If it cannot be recovered, the transfer arm stops; no retraining or
   substitute artifact is allowed.
6. **Exact transfer-lift verification — discharged in the executed run:**
   before fine-tuning, the lifted five-input network reproduced the source
   model's unconstrained, physically reconstructed predictions at the eight
   fixed, protocol-pinned probe points to float64 numerical precision. Failure
   would have stopped the experiment; no general transfer framework was built.
7. **Seeds, budget, metrics, latency shapes, and IV cases — locked:** the
   architecture, scratch and transfer seeds, optimizer,
   training budget, checkpoint rule, metric definitions and gates, latency
   cases, batch sizes, thread budgets, warm-ups, repetitions, IV cases, and all
   input-file digests are frozen in the protocol PR and reviewed before any
   run.
8. **Partition lifecycle — discharged and terminal:** neither training code
   nor any entry check opened `interpolation_test`. Only `train` and
   `validation` were accessible, the one-shot final-evaluation command was
   never invoked, and the frozen snapshot records
   `final_evaluation_attempts = 0` with `final_partition_consumed = false`.

The already-observed near-duplicate distances remain descriptive. No threshold
is added and task 9E's predeclared near-duplicate gate is not represented as
having passed retroactively.

## Exact in-scope experiment

- One float64 CPU `tanh` MLP with hidden dimensions `[64, 64, 64]`, the same
  hidden architecture as the frozen European source.
- The five-input `american_forward_carry_v1` representation and normalized
  target `V/(S*exp(-q*T))` from the architecture contract.
- Exactly one fixed scratch run and one fixed transfer run, under the same one
  training budget, optimizer schedule, row budget, checkpoint rule, and
  evaluation pipeline. There is no hyperparameter sweep or extra seed.
- The two locked arm seeds also produce different epoch shuffle permutations,
  not only different initializations. This one-seed pilot therefore cannot
  attribute an observed arm difference solely to transfer initialization.
- Validation-only checkpoint selection. Both arms are evaluated once on the
  same `interpolation_test` after the final-evaluation gate is unlocked.
- The row's paired `european_crr_price` as the no-learning baseline: its error
  against the American label is the early-exercise premium. Metrics are
  reported overall and separately on rows with strictly positive stored
  early-exercise premium; zero-premium and no-exercise rows remain visible.
- Price metrics in physical and normalized units: MAE, RMSE, p95, p99, maximum
  absolute error, and the protocol's predeclared shape/bound diagnostics,
  sliced at least by option type, expiry, moneyness, volatility, and
  early-exercise-premium status. No American Greek is a supervised or claimed
  result.
- A fixed CRR depth ladder `N = [256, 512, 1024, 2048, 4096]`. At every depth
  the numerical comparator is the actual adjacent-average operation
  `0.5 * (CRR(N) + CRR(N+1))`, never a single tree described as the label
  generator. The `N=1024` rung is the dataset's label operation.
- A small model-consistent implied-volatility experiment on protocol-pinned
  synthetic CRR cases. It compares IV recovered from the American label with IV
  recovered from each neural arm, under the same inversion method and stopping
  rules. If the cases use one maturity, the output is called an **IV smile
  slice**; only a case set with multiple maturities may be called an **IV
  surface**.

## Exclusions

- No claim that this one-seed, one-budget pilot establishes H2.
- No additional architecture, seed, label budget, training budget,
  hyperparameter sweep, ablation, frozen-trunk arm, or recovery run.
- No silent European retraining or alternate source artifact.
- No American output constraint derived from the European bounds projection.
- No general transfer framework if the exact lift fails.
- No boundary, extrapolation/OOD, or scenario claim; the dataset has no such
  partition.
- No American Greek supervision or accuracy claim.
- No bid--ask-relative or other market-performance claim. Phase 1 is synthetic
  and has no bid or ask.
- No SPY, XSP, discrete-dividend, commodity, cloud, vendor-ingestion, parallel
  generation, or resumability work.
- No change to completed-study configurations, frozen evidence, or consumed
  partitions.

## Fixed protocol items

The protocol/implementation PR must bind, by exact value and SHA-256 before
execution: the two seeds; row/training budget; optimizer and schedule; batch
size; architecture and feature order; standardization and checkpoint rules;
all validation and final metrics and thresholds; the fixed probe points for the
exact lift; the CRR ladder above; latency request shapes, batch sizes, thread
budgets, warm-ups and repetitions; IV cases and inversion rules; hardware and
software metadata schema; artifact paths; and the one-shot final-evaluation
lifecycle. Any item still `OPEN` when that PR is reviewed blocks execution.
This requirement was discharged: no item remained `OPEN` at review, every item
above is bound in the merged configs, and the executed run is pinned to their
digests. Nothing here was changed after execution began, and this result-only
closure changes none of it.

The transfer lift copies the three shared first-layer columns, adds zero
columns for `rT` and `qT`, copies all remaining weights and biases, and
preserves or algebraically rebases feature/target standardization so the
physical source function is unchanged. The source weights are unconstrained;
European bounds projection metadata is lineage only and is not applied to the
American output.

## Validation and one-shot final-evaluation lifecycle

The merged runner `scripts/run_american_neural_pilot.py` has separate
`run-to-validation`, `status`, and
`final-evaluate --confirm-locked-final-evaluation` modes. `run-to-validation`
performs entry checks, prepares the two arms, trains them, and selects
checkpoints using only `train` and `validation`. It cannot import, hash, open,
or count `interpolation_test` rows. The human operator invoked
`run-to-validation` exactly once; `final-evaluate` was never invoked.

Only after the reviewed validation report satisfies the protocol's final-entry
rule may a human invoke `final-evaluate`. That rule was **not** satisfied
(`validation_final_entry_passed = false`), so `final-evaluate` is forbidden
under this protocol. The runner records the attempt before opening
`interpolation_test`, and interruption would consume the attempt; no attempt was
recorded and the partition stays unconsumed. Any further work requires a new
protocol and a fresh final partition; a consumed partition is never tuned
against, and an unconsumed one is not opened to rescue a failed pilot.

## Latency and IV protocol

Reference and neural latency use identical economic requests, request shapes,
batch sizes, thread budgets, warm-ups, and repetition counts on the same
recorded hardware/software environment. The CRR side prices both `N` and
`N+1` and averages them for every requested `N`. Neural end-to-end timing
includes physical-to-network feature transformation, feature standardization,
model execution, inverse target transformation, and reconstruction of
`V = S*exp(-q*T)*u`; serialization or transfer cost is included whenever the
measured API requires it. Kernel-only timings may be diagnostic but cannot
support a speedup claim.

The IV experiment is model-consistent synthetic evidence only. It reports
price-to-IV error and inversion failures under fixed brackets and tolerances;
it reports neither market fit nor bid--ask-relative performance. Terminology is
structural: one maturity is a smile slice, several maturities are a surface.

## Artifact and result policy

The protocol, configs, implementation, tests, and a designated result
freeze/check tool are tracked in the protocol/implementation PR. The dataset,
recovered European artifact, checkpoints, raw reports, timings, and IV outputs
remain beneath ignored paths. The raw experiment report records exact input,
code, environment, artifact, weights, and dataset digests.

After the manual run, a **result-only PR** may add the compact frozen snapshot
and deterministic figures through the designated generator. It may not change
gates, seeds, training code, evaluation code, the protocol, or the consumed
raw report. `--check` must validate the snapshot without ignored artifacts and
must not rerun pricing, training, or IV inversion. This closure is that PR: it
adds the frozen snapshot, wires
`python3 scripts/freeze_american_neural_pilot_results.py --check` into
`scripts/check.sh` and CI, and updates this spec, project state, and the
decision log. It adds no figure and no plotting framework, and it changes no
config, threshold, seed, training-code, evaluation-code, runner, or freezer
behavior.

## Success and honest-failure outcomes

- **Promising pilot:** both arms complete; the frozen report shows the
  predeclared accuracy and latency evidence, and transfer is sufficiently
  promising under the frozen pilot interpretation to justify a separate H2
  replication. This is not itself an H2 pass.
- **Negative transfer:** the transfer arm is no better or is worse than scratch
  under the fixed comparison. Freeze and report it.
- **Failure to learn — this is the recorded outcome.** Neither arm reached the
  predeclared feasibility gates. It is frozen and reported without tuning
  against `interpolation_test`, and without a second attempt.
- **Entry failure:** admission review, independent-check resolution,
  row/manifest invariants, identity, source recovery, or exact lift fails. Stop
  before training and report the failed gate; do not manufacture a two-arm
  result.

If the pilot is promising, the next learning task is a separate H2 replication
with at least five seeds and several predeclared label budgets. It is not an
extension or rerun inside task 9G. **The pilot was not promising**, so no H2
replication is predeclared here. The next intended task is instead task 9H, the
explicitly exploratory development loop described below.

## Next intended task — task 9H (not implemented here)

Task 9H is recorded as intent only. Nothing in this result-only closure
implements, scopes, or authorizes it; it needs its own predeclared spec and its
own review before any work begins.

- **Task 9H: an explicitly exploratory American neural-pricer development
  loop, on `train` and `validation` data only.** It is development, not a
  confirmatory experiment, and it is labelled as such from the start.
- Claude or Codex may iteratively implement capacity, feature, target, and
  architecture experiments; the human runs every one of them locally, under the
  standing rule that long numerical runs are manual, terminal-invoked jobs.
- Every attempt and every failure is documented, including the ones that are
  abandoned. The loop's value is the recorded search, not a single surviving
  configuration.
- `interpolation_test` and every other final partition remain inaccessible for
  the whole loop. No code path in 9H may open, hash, import, or count them.
- Because selection happens against `validation`, any model 9H selects carries
  selection bias and is **not** a result. A selected model requires a separate,
  freshly predeclared confirmation on a fresh partition after development ends;
  that confirmation is a different task with its own gates and its own review.

## Manual-run rule

The independent numerical cross-check, training, latency measurement, and IV
inversion are manual terminal-invoked work. They are never launched by an
agent, hook, CI job, documentation check, or result-freeze check. The
protocol/implementation PR must document the exact commands; a human runs them
only after that PR is reviewed and merged. That is what happened: the human
operator invoked `run-to-validation` once after the implementation merge at
`e930454`. This result-only closure runs none of them, reruns nothing, and does
not repeat the locked run.

## Review requirements

The intended sequence is mandatory:

1. planning-reconciliation PR — merged at `142b4b3` (DEC-035);
2. protocol-and-implementation PR, reviewed and merged before execution —
   approved at `8d27c23` and merged at `e930454` (DEC-036, DEC-037);
3. manual locked experiment run — performed once by the human operator;
4. result-only PR with no gate, seed, training-code, or evaluation-code change
   — the current PR;
5. fresh top-level review of the result — outstanding.

The implementation PR requires code review for lifecycle, artifact, dataset,
lift, and timing code and numerical review for the cross-check resolution,
metrics, CRR comparator, and IV cases. Same-session and review-agent findings
are preliminary. Fresh material approval of task 9E's conditional admission
has been recorded as `APPROVE PLANNING RECONCILIATION` (DEC-035); it does not
authorize execution or approve a neural-pricer result. Material acceptance of
the eventual pilot result still requires its own fresh top-level review. A
two-commit branch first reviewed after results is not an acceptable
predeclaration mechanism.

## Stop conditions

- Any entry gate fails or remains unresolved.
- Any code path accesses `interpolation_test` before the recorded one-shot
  final evaluation.
- The source weights are absent or their digest differs; retraining or
  substitution is proposed.
- The exact lift differs from the source physical function beyond the pinned
  float64 tolerance, or would require a general transfer framework.
- A seed, budget, threshold, case, metric, latency shape, or IV rule is changed
  after execution begins.
- The CRR latency comparator prices only one tree, or timing conditions are not
  matched.
- A one-maturity result is about to be called a surface, or a synthetic result
  is about to be compared with market bid--ask spreads.
- Any expensive run would be agent-, hook-, or CI-triggered.
- Any frozen evidence, completed-study config, ignored dataset, or consumed
  partition would be modified.

## Explicit non-claims

Task 9G cannot establish H2, converged American-price truth, American Greek
accuracy, OOD generalization, discrete-dividend pricing, SPY performance,
market calibration, bid--ask-relative accuracy, production readiness, or live
trading value. Any speedup is conditional on the exact matched benchmark
contract and cannot be generalized beyond its recorded hardware, software,
request shapes, and thread budgets.
The fixed-domain PDE refinement pair does not independently bound domain-
truncation error. Offline snapshot checking detects internal inconsistency and
tracked-input drift, but cannot authenticate a fully coordinated fabricated raw
report and snapshot.

## Completion report

Record every entry gate and its evidence; exact protocol and source digests;
validation and one-shot final outcomes for scratch, transfer, and baseline;
positive-premium slices; the matched latency ladder; IV smile/surface outcome
using correct terminology; failures and stop conditions; partition-consumption
state; and the exact next task. Update
[../../project-state.md](../../project-state.md) and append a decision-log entry
whatever the outcome. State explicitly that one seed and one budget do not
establish H2.

This is discharged as follows. Every entry gate passed and is recorded above
under "Entry gates," with the independent PDE mapping check passing 21 of 21
rows and the exact transfer lift passing all eight pinned probes. The protocol
and source digests, all per-slice validation metrics for scratch, transfer, and
the paired no-learning European-CRR baseline, the positive-premium slices, the
matched CRR-depth latency ladder, and the six-case, two-maturity IV
experiment — correctly termed an **IV surface**, since it spans maturities
`0.25` and `2.0` rather than a single maturity — all live in the frozen
snapshot, which stays authoritative for the numbers. The recorded stop is the
predeclared "failure to learn" outcome: neither arm met the feasibility gates,
so the protocol's final-entry rule failed and the one-shot final evaluation is
forbidden. `final_evaluation_attempts = 0`, `final_partition_consumed = false`,
and `interpolation_test` was never accessed. **One seed and one budget do not
establish H2**, and nothing in this result may be cited as H2 evidence,
converged American-price truth, American Greek accuracy, OOD behavior,
discrete-dividend applicability, market or bid--ask performance, or a portable
latency conclusion. The exact next task is task 9H above, recorded as intent
only and not implemented here.
