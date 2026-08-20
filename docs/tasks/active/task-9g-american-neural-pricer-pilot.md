# Task 9G: continuous-yield American neural-pricer feasibility pilot

## Status

`Protocol and implementation built — REQUEST CHANGES findings addressed;
awaiting fresh approval and merge; no locked run.`
This specification defines the next active task after the task 9E planning
reconciliation. Fresh independent review of that reconciliation is discharged
by `APPROVE PLANNING RECONCILIATION` (DEC-035). The protocol-and-implementation
change is now built (DEC-036). Its first fresh implementation review returned
`REQUEST CHANGES`, so that review gate remains open and no experiment may run
until the fixes receive a subsequent fresh approval and the change is merged.

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
  PR. They do not exist yet and are not invented by this planning change.

## Entry gates

Every gate is fail-closed and is checked before training opens `train`:

1. **Admission review — discharged:** fresh independent review returned
   `APPROVE PLANNING RECONCILIATION` (DEC-035), accepting only the reconciled
   task 9E conditional admission for learning the known CRR mapping, not
   converged American-price accuracy or training authorization.
2. **Independent numerical check — protocol implemented, execution pending:** the outstanding LSM/PDE price
   cross-check is completed under a newly predeclared protocol, or a fresh
   review records an explicit resolution that preserves the mapping-only claim.
   Silence or an internal same-lattice identity is not a resolution. Training
   is unauthorized until one of these is recorded.
3. **Row/manifest policy invariants — implemented:** implementation and tests
   require every row's `label_policy == manifest.label_policy.name` and every
   row's `label_steps == manifest.label_policy.steps`.
4. **Dataset and generator/config identity pins — implemented, execution pending:** entry checks
   require schema `american-option-dataset/1`, generator version `1.0.0`, the
   recovered configuration digest above, and the manifest's declared file
   identities.
   Before training, only `train` and `validation` files are opened and hashed;
   the final split's digest is checked only inside the recorded one-shot final
   evaluation. A mismatch stops the task.
5. **European source-weight recovery and digest verification — preflight discharged, runtime recheck implemented:** the
   original unconstrained European `weights.npz` is
   locally available and hashes to the frozen digest above. The file remains
   ignored. If it cannot be recovered, the transfer arm stops; no retraining or
   substitute artifact is allowed.
6. **Exact transfer-lift verification — implemented and preflight discharged:** before fine-tuning, the
   lifted five-input network reproduces the source model's unconstrained,
   physically reconstructed predictions at fixed, protocol-pinned probe points
   to float64 numerical precision. Failure stops the experiment; no general
   transfer framework is built as a workaround.
7. **Seeds, budget, metrics, latency shapes, and IV cases — locked:** the
   architecture, scratch and transfer seeds, optimizer,
   training budget, checkpoint rule, metric definitions and gates, latency
   cases, batch sizes, thread budgets, warm-ups, repetitions, IV cases, and all
   input-file digests are frozen in the protocol PR and reviewed before any
   run.
8. **Partition lifecycle — implemented, pending review:** neither training code nor
   any entry check opens `interpolation_test`. Only `train` and `validation`
   are accessible before the one-shot final-evaluation command.

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

The transfer lift copies the three shared first-layer columns, adds zero
columns for `rT` and `qT`, copies all remaining weights and biases, and
preserves or algebraically rebases feature/target standardization so the
physical source function is unchanged. The source weights are unconstrained;
European bounds projection metadata is lineage only and is not applied to the
American output.

## Validation and one-shot final-evaluation lifecycle

The planned runner has separate `run-to-validation`, `status`, and
`final-evaluate --confirm-locked-final-evaluation` modes. `run-to-validation`
performs entry checks, prepares the two arms, trains them, and selects
checkpoints using only `train` and `validation`. It cannot import, hash, open,
or count `interpolation_test` rows.

Only after the reviewed validation report satisfies the protocol's final-entry
rule may a human invoke `final-evaluate`. The runner records the attempt before
opening `interpolation_test`; interruption consumes the attempt. Both arms and
the no-learning baseline are evaluated in the same attempt. Any later change
requires a new protocol and a fresh final partition; the consumed partition is
never tuned against.

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
must not rerun pricing, training, or IV inversion.

## Success and honest-failure outcomes

- **Promising pilot:** both arms complete; the frozen report shows the
  predeclared accuracy and latency evidence, and transfer is sufficiently
  promising under the frozen pilot interpretation to justify a separate H2
  replication. This is not itself an H2 pass.
- **Negative transfer:** the transfer arm is no better or is worse than scratch
  under the fixed comparison. Freeze and report it.
- **Failure to learn:** neither arm reaches the predeclared feasibility gates.
  Freeze and report it without tuning against `interpolation_test`.
- **Entry failure:** admission review, independent-check resolution,
  row/manifest invariants, identity, source recovery, or exact lift fails. Stop
  before training and report the failed gate; do not manufacture a two-arm
  result.

If the pilot is promising, the next learning task is a separate H2 replication
with at least five seeds and several predeclared label budgets. It is not an
extension or rerun inside task 9G.

## Manual-run rule

The independent numerical cross-check, training, latency measurement, and IV
inversion are manual terminal-invoked work. They are never launched by an
agent, hook, CI job, documentation check, or result-freeze check. The
protocol/implementation PR must document the exact commands; a human runs them
only after that PR is reviewed and merged. This planning task runs none of
them.

## Review requirements

The intended sequence is mandatory:

1. planning-reconciliation PR — the current PR;
2. protocol-and-implementation PR, reviewed and merged before execution;
3. manual locked experiment run;
4. result-only PR with no gate, seed, training-code, or evaluation-code change;
5. fresh top-level review of the result.

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
