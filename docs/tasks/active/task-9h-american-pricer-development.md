# Task 9H: adaptive American price-model development

## Status

`status=active; type=adaptive_exploratory_development; scope=price_only;
branch=experiment/task-9h-american-pricer-development;
partitions_available=train+validation; final_partition_access=forbidden;
infrastructure implemented and hardened; one attempt recorded
(scratch_direct_control_v1, criterion_not_met); no attempt running.`

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

## Files

| Path | What it is |
|---|---|
| `python/src/differentiable_pricing/ml/american_dev/attempts.py` | partition guard and its single token list, row/seed selection, digests, clean-tree and committed-source checks, strict configuration validation, attempt-log rules. PyTorch-free |
| `python/src/differentiable_pricing/ml/american_dev/representation.py` | the five coordinates, the European anchor, conditioning features, heads, physical reconstruction |
| `python/src/differentiable_pricing/ml/american_dev/models.py` | the dense and residual networks, and their dispatch |
| `python/src/differentiable_pricing/ml/american_dev/workbench.py` | pre-flight, dataset identity pinning and row-level policy verification, one attempt end to end, evaluated against the reused Task 9G criterion |
| `configs/american_dev_attempt_scratch_*.toml` | the five immutable attempt configurations |
| `scripts/run_american_dev_attempt.py` | **manual**: `run`, `status` — and no third command |
| `scripts/american_dev_attempts.py` | offline: `record` one attempt, `check` the log and configurations |
| `docs/attempts/task-9h-attempt-log.jsonl` | the append-only recorded search |

Checkpoints, reports, compact summaries and the run ledger live beneath the
ignored `artifacts/` and `runs/` trees and are never committed.

## Commands

Training is a manual, terminal-invoked human command. An agent reports it and
stops.

```
python3 scripts/run_american_dev_attempt.py run \
    --config configs/american_dev_attempt_scratch_direct_control_v1.toml

python3 scripts/run_american_dev_attempt.py status \
    --config configs/american_dev_attempt_scratch_direct_control_v1.toml

python3 scripts/american_dev_attempts.py record \
    --report artifacts/task-9h/<attempt_id>/attempt-report.json \
    --outcome criterion_met|criterion_not_met|abandoned|infrastructure_failure \
    --interpretation "<one honest sentence>" \
    --next-action "continue|stop|change-direction, and why"
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
  and no transfer-learning claim.
- Task 9H establishes no H2 result, no converged American-price truth, no OOD
  behavior, no discrete-dividend applicability and no market performance.
- Task 9E's conditional, mapping-only dataset admission is unchanged.

## Recorded limitation

Task 9H adds no wording to any `docs/*-contract.md` or to
[architecture.md](../../architecture.md). Those files are digest-pinned as Task
9G tracked inputs, and editing one would break the historical Task 9G protocol
check that `scripts/check.sh` and CI both run. This specification is therefore
the authority for task 9H's conventions until a later task can promote them.
