# Research contract

## Primary question

Can a smooth neural surrogate approximate a trusted derivative pricer and its
useful risk sensitivities, across a declared domain, with lower end-to-end
latency and an error profile suitable for a stated decision?

The project is not trying to show that a network can interpolate an arbitrary
grid. It is testing accuracy, derivative fidelity, generalization, latency,
and whether transfer learning reduces the number of expensive labels needed.

## Hypotheses

### H1: European option sanity check

A compact network trained on Black--Scholes labels can reproduce prices and
delta/vega over a bounded domain. Analytic formulas must expose normalization,
derivative, sampling, and implementation errors before complex products are
introduced.

### H2: European to American transfer

Starting from a European-option representation reduces American-option sample
requirements or training time compared with a matched randomly initialized
model, while reaching equal or better out-of-sample price and Greek error.

### H3: European to Bermudan swaption transfer

A model initialized from European-swaption training reduces the label budget
needed for Bermudan swaption accuracy versus random initialization. The
comparison must control architecture, optimizer, schedules, wall-clock budget,
seeds, and training examples.

## Stages and reference standards

| Stage | Minimum reference | Required checks |
|---|---|---|
| European option | Analytic Black--Scholes | closed-form prices/Greeks, parity, monotonicity, MC convergence |
| American option | Converged CRR tree | European lower bound where applicable, exercise-boundary behaviour, step convergence |
| European swaption | Independently checked Hull--White 1F implementation | curve/convention fixtures, limiting cases, calibration reconstruction |
| Bermudan swaption | Converged tree/PDE or out-of-sample LSM | lower bounds, exercise-policy separation, path/step/basis convergence |

Reference-pricer numerical error must be materially smaller than the surrogate
acceptance threshold. Otherwise the experiment measures label noise.

## Data protocol

Each generated row or partition records:

- contract and model inputs in explicit units;
- price and available reference sensitivities;
- reference method and version;
- seed and random-number policy where applicable;
- convergence controls and estimated numerical error;
- feature-domain identifier and generation timestamp.

Sample in transformed economic coordinates where useful (for example log
moneyness rather than raw spot and strike). Keep named partitions:

- interpolation test: held-out samples inside the training envelope;
- boundary test: short expiry, low volatility, deep moneyness, and exercise
  frontiers;
- extrapolation/OOD test: explicitly outside at least one training bound;
- scenario test: coherent curve/surface shocks rather than independent rows.

Do not random-split rows originating from the same paths, grids, curve
scenario, or near-duplicate contract state.

## Metrics

Report at least:

- price MAE, RMSE, relative error where stable, p95, p99, and maximum;
- delta/vega for equity stages and PV01/vega for rates stages;
- arbitrage and shape violations: bounds, monotonicity, convexity where
  applicable;
- reference-pricer time, feature time, model time, derivative time, and total
  latency for fixed batch sizes;
- model size and peak memory;
- error sliced by expiry, moneyness, volatility, exercise region, and OOD flag.

All latency results require warm-up, pinned software/hardware metadata,
multiple repetitions, and uncertainty intervals. GPU and CPU results are not
interchangeable.

## Transfer-learning experiment

For each target product compare:

1. random initialization;
2. source-product pretrained initialization;
3. optionally, frozen trunk followed by partial/full fine-tuning.

Use at least five training seeds and several label budgets. Select
hyperparameters without looking at the final test set. The primary transfer
claim is the target-label budget needed to cross a predeclared error gate, not
the prettiest single learning curve.

Negative transfer is a valid result and must be reported.

## Provisional gates

Stage-specific gates live in versioned configuration. Before generating the
first large dataset, replace provisional values with tolerances connected to
a use case. Passing average error alone is never sufficient: tail errors,
shape constraints, cross-language parity, and derivative checks must pass.

The first European fresh-seed replication is bound by
`configs/european_neural_replication_protocol_v1.toml`. That protocol preserves
the development sampler and selected model except for independently derived
dataset and training seeds, pins every input file by SHA-256, permits validation
for checkpoint selection, and permits one final evaluation on the fresh
`interpolation_test`. It must be committed before replication data generation.
If the final gate fails, record the failure and version a new protocol; do not
tune against the consumed partition.

Protocol v1 completed with one final-evaluation attempt and passed every
frozen validation and final gate. Its immutable, digest-linked outcome is
`docs/results/european_replication_results_v1.json`. The final partition is
consumed permanently. The result supports synthetic in-envelope European
interpolation only; it does not relax the non-claims below or authorize further
tuning against that dataset.

## Non-claims

- A network derivative is not an exact Greek of the reference model.
- A synthetic-label experiment is not evidence of live trading alpha.
- Faster kernel inference is not faster end-to-end pricing unless feature,
  transfer, batching, and derivative costs are counted.
- Successful interpolation is not proof of extrapolation.
