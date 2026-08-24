# Task 9H American Neural Pricer — Architecture Freeze v2.3

**Status:** Normative. This is the architecture the American neural-pricer roadmap is built and confirmed against, adopted by [decision-log.md](decision-log.md) DEC-049. Once this rebaseline merges into canonical history the document becomes immutable: it is versioned, not amended in place, and any substantive architectural change thereafter requires a superseding freeze (v2.4) plus its own decision entry.  
**Purpose:** Define the architecture the v2 implementation rewrite is built and confirmed against. It resolves the external review of Architecture Freeze v2 and supersedes the previous Task 9H development roadmap.  
**Scope:** Reproducible CRR teacher generation, dataset partitions, adaptive-development controls, deep numerical references, Greek validation, deterministic model artifacts, native C++ inference, and matched native latency benchmarking.  
**Non-status:** This document does not claim that the current E2c model passes final price, Greek, arbitrage, or latency criteria. It introduces no new final pass/fail thresholds.


## 0A. Second-review resolution

This revision incorporates four sequencing changes and resolves one pre-Grid-1 units defect.

1. **Native C++ feasibility and the remaining Greek diagnostics are independent exploratory workstreams and should run in parallel.** E2c is only a convenient concrete 199k-parameter deployment function for the native prototype; the prototype is architecture evidence, not a decision about retaining E2c weights.

2. **E2c cannot be the v2 confirmation candidate.** The v2 confirmation model must be retrained from a dataset reproducible from committed source. Current Greek diagnostics therefore inform the **head/representation design**, not whether historical E2c weights can become the final model.

3. **Reference-method/depth freezing runs independently of model training.** After the v2 architecture, application tolerances, and numerical-reference criteria are committed, the partition-free CRR/PDE reference study may run in parallel with the v2 training campaign. It must complete before final evaluation.

4. **All v2 dataset partitions are generated atomically from one committed configuration and one generator version.** Train, validation, development holdouts, neutral final interpolation, and predeclared economic stress partitions must share the same generation transaction/manifest so sampling semantics cannot drift between runs.

5. **The current Greek degeneracy implementation has a real units defect and must be repaired before Grid 1 is scored.** The declared `1e-8` threshold is documented as a normalized-Vega scale, while the implementation applies it directly to physical-price derivatives and also reuses it for Gamma. Vega can be repaired declaration-faithfully by applying the threshold to `Vega / A`, where `A = S exp(-qT)`. Gamma needs a separate unit-consistent definition; until one is predeclared, Gamma “degeneracy” must not be used as a decision-bearing metric.

6. **The suspected sibling units defect in B.2 eligibility was audited and is *not* present in the implementation.**
   `black_scholes_vega()` returns physical Vega and `label_discretization_proxy()` returns the physical price gap `|E_CRR-E_BS|`; `iv_in_scope()` compares those physical quantities directly. The domain artifact's reported normalized gaps are not fed directly into that rule. `normalized_vega()` is used separately for assessability/degeneracy. The implementation should nevertheless gain a dimensional-invariance regression test so this separation cannot drift later.




---

## 0. Review resolution

The review accepts the core two-axis design and recommends several changes. This revision accepts most of them, with three important qualifications.

### Accepted

1. **Native C++ inference is a first-class performance intervention, not merely benchmark hygiene.**
   The current Python/PyTorch path contains substantial framework, feature-construction, and dispatch overhead. A native implementation may remove much of it. However, no 20–40x speedup is assumed until measured.

2. **Price-reference depth and Greek-reference depth should be decoupled.**
   Price convergence appears substantially easier than Greek convergence. The architecture therefore permits:
   \[
   N_{\mathrm{ref,price}} \neq N_{\mathrm{ref,\Delta}}
   \neq N_{\mathrm{ref,\Gamma}}
   \neq N_{\mathrm{ref,Vega}}.
   \]

3. **The primary final interpolation partition must be neutrally sampled.**
   Model-specific projection/crossover regions do not belong in the denominator of the headline interpolation claim. They belong in a separate in-domain stress study.

4. **Restoring a reproducible American dataset generator is a structural requirement.**
   This is not primarily a sampling upgrade. The current American dataset cannot be regenerated from the presently tracked source alone, so the teacher dataset must be made reproducible before the v2 final experiment.

5. **Attempt-budget and selection-bias controls are architectural components.**
   The attempt ledger, declared attempt budget, development holdouts, and final-partition access controls must survive the rewrite.

6. **Native lattice Delta/Gamma do not answer the narrow projection-crossover Gamma question.**
   A reference-free dense comparison of deployed-autograd Gamma versus raw-autograd Gamma remains a separate required diagnostic.

7. **The independent PDE solver should remain additive reference evidence.**
   A CRR depth ladder demonstrates convergence within one discretization family; a separately converged PDE method gives numerically independent evidence.

### Accepted with qualification

8. **Run an early native C++ proof of concept.**
   This is a good de-risking step, but it must be treated as exploratory architecture feasibility, not as the final latency result. It must first reproduce the frozen Python E2c function numerically, then measure the same matched CRR economic request. Its purpose is to determine whether a native backend is worth the full engineering investment.

9. **Keep the base depth near the current N=1024 unless evidence says otherwise.**
   Current American shallow/deep price evidence indicates that price discretization is already much smaller than current NN price error, but the final v2 base depth should be confirmed from the reproducible generator/reference framework rather than adopted solely from the historical dataset.

### Rejected / modified

10. **Do not qualify the deep price reference using a fixed fraction of the current model's achieved error.**
    A numerical reference should not become “good enough” because the current NN happens to be inaccurate. Reference convergence must be tied to a predeclared application/price tolerance or an independent numerical criterion.

11. **Do not subsample the deep final price reference by default.**
    At the expected final-set scale, deeper price computation is cheap enough that the clean decomposition
    \[
    V_{\rm NN}-L_N,\quad V_{\rm NN}-L_{N_{\rm ref}},\quad L_N-L_{N_{\rm ref}}
    \]
    should be available on every primary final row unless an explicit compute ceiling is reached.

12. **Do not treat memory-bandwidth arithmetic as a latency result.**
    Weight size and FLOP counts are useful plausibility checks only. Small GEMV efficiency, cache residency, activation cost, feature transforms, analytic legs, and implementation details must be measured.

---

# 1. Core scientific question

The target claim is:

> A frozen differentiable neural surrogate can replace repeated finite-depth American CRR pricing calls on unseen in-domain option states while meeting predeclared price, structural, and—where the numerical reference is demonstrably adequate—Greek-quality requirements, at materially lower native inference cost.

The design must separate:

\[
\boxed{
\text{observed disagreement}
=
\text{state-generalization error}
+
\text{NN approximation error}
+
\text{CRR discretization error}
+
\text{Greek-reference error}
}
\]

The current Task 9H development work largely measures the NN against the same finite-depth CRR operator used as its teacher. That is useful but insufficient for the final claim.

---

# 2. Frozen terminology

## 2.1 Option state

\[
x=(\text{option type},S,K,T,r,q,\sigma).
\]

Tree depth \(N\) is not a neural input.

## 2.2 Base teacher

\[
L_N(x)=\frac12\left[CRR_N(x)+CRR_{N+1}(x)\right].
\]

Both component prices are persisted separately.

## 2.3 Deep price reference

\[
L_{N_{\rm ref,price}}(x)
=
\frac12\left[
CRR_{N_{\rm ref,price}}(x)
+
CRR_{N_{\rm ref,price}+1}(x)
\right].
\]

The depth is selected on a deterministic partition-free convergence set before final evaluation.

## 2.4 Greek references

Greek-reference depth and bump rules are selected separately per Greek because differentiation amplifies finite-depth lattice error differently for Delta, Gamma, and Vega.

## 2.5 Out of sample

“Out of sample” means unseen option state \(x\).

Changing \(N\) is a **numerical-reference-quality axis**, not an out-of-sample input test.

---

# 3. CRR tree-depth non-containment

A separately defined \(N\)-step CRR price cannot be recovered exactly from an \(N+1\)-step tree.

For depth \(N\),

\[
\Delta t=T/N,\qquad
u=e^{\sigma\sqrt{\Delta t}},\qquad
d=1/u,
\]

and the risk-neutral probability and discount factor depend on the same \(\Delta t\).

Therefore the whole lattice changes when \(N\) changes.

### Consequence

`CRR_N` and `CRR_{N+1}` are separate computations. The generator may reuse allocation and workspaces, but it must not claim that one exact root price is recoverable from the other tree.

Persist final outputs and diagnostics, not the full \(O(N^2)\) lattice.

---

# 4. Reproducible C++ CRR teacher

The authoritative CRR teacher/reference kernel executes in C++.

A recombining backward-induction implementation should retain approximately:

\[
O(N^2)\text{ time},\qquad O(N)\text{ memory}.
\]

For every base-label state persist at least:

- option inputs;
- `label_steps = N`;
- `CRR_N`;
- `CRR_{N+1}`;
- adjacent average \(L_N\);
- adjacent gap;
- paired European CRR values where used;
- intrinsic;
- early-exercise diagnostics required by the study.

Every generated dataset must record:

- generator source/version;
- CRR source/header/implementation hashes;
- build/compiler identity;
- dtype;
- config digest;
- sampling seed;
- domain;
- split identities/counts;
- label policy;
- per-file hashes.

The v2 repository must contain the generation machinery needed to regenerate the dataset from committed source plus the declared configuration.

---

# 5. Dataset sampling and partitions

## 5.1 Coverage, not a Cartesian grid

The continuous state space is too high-dimensional for a literal tensor-product dense grid. Use deterministic low-discrepancy and stratified coverage with declared regime quotas.

## 5.2 Primary neutral sampling

The primary train/validation/final distribution must be defined without reference to the eventual neural model's observed failures.

It may include economically predeclared strata such as:

- call/put;
- moneyness;
- maturity;
- volatility;
- rate/dividend regimes;
- early-exercise-prone economic regimes.

It must not oversample locations selected because the eventual network or deployment head was observed to behave badly there.

## 5.3 Separate stress evidence

Use two separate stress mechanisms:

1. **Economically declared in-domain stress partition**, defined before model fitting.
2. **Post-freeze label-free model stress scans**, such as dense projection-crossover scans. These may depend on the frozen model because they are explicitly adversarial diagnostics, not the headline interpolation denominator.

## 5.4 Atomic generation

All v2 partitions must be emitted by **one invocation** of one committed generator configuration and described by one manifest. A separate later invocation must not be used merely to create the final partition.

The manifest must bind, in one generation identity:

- generator/config digest;
- sampling seed(s);
- domain and stratum definitions;
- row counts;
- split-allocation rule;
- every partition filename and digest.

This makes “untouched final” a property of one precommitted sampling design rather than a second sampling event.

## 5.5 Development and final partitions

Target partition roles:

- **train** — parameter fitting and train-only scaling;
- **validation** — epoch/model selection;
- **development holdout A** — limited-use diagnostic set;
- **development holdout B** — reserved one-shot pre-final sanity set if the protocol chooses to retain the current H1/H2 pattern;
- **final interpolation test** — untouched confirmation set;
- **in-domain stress set** — separate from primary final statistics.

The exact H1/H2 design should be finalized after the current exploratory phase, but the principle is frozen: adaptive development may not consume the final interpolation set.

---

# 6. Eligibility units audit

The B.2 eligibility rule has been checked for the same physical-vs-normalized defect found in the Greek-degeneracy code.

The declared rule is

\[
0.01\,\mathrm{Vega}_{BS}(x) > 3\,g(x),
\qquad
g(x)=|E_{CRR}(x)-E_{BS}(x)|.
\]

The implementation is dimensionally consistent:

- `black_scholes_vega()` returns physical-price Vega;
- `label_discretization_proxy()` computes `|E_CRR-E_BS|` in physical price units;
- `iv_in_scope()` compares those two physical quantities;
- `normalized_vega()` divides Vega by \(A=S e^{-qT}\) only for the separate assessability/degeneracy analysis.

Thus the actual code implements

\[
0.01\,\mathrm{Vega}_{phys}
>
3\,g_{phys},
\]

which is algebraically equivalent after division by \(A\) to

\[
0.01\,\mathrm{Vega}_{norm}
>
3\,g_{norm}.
\]

The normalized gap values recorded in the domain artifact are reporting/reference constants; they are not directly substituted for the per-row physical gap in `iv_in_scope()`.

### Required hardening test

Add an offline regression test that constructs a row with known scale \(A\), then verifies that eligibility is invariant under the equivalent physical and normalized forms of the inequality. This guards against a future refactor accidentally feeding a normalized domain gap into a physical-Vega comparison.

This audit changes no eligibility result and requires no reclassification of the already-published B.2 price artifacts.

---

# 7. Adaptive-development controls

The rewrite must preserve explicit protection against validation overfitting.

Required components:

- append-only attempt ledger;
- unique attempt/config identity;
- fixed total learning-attempt budget;
- declared handling of failed/unrecordable attempts;
- train-subset identity;
- development-holdout access log;
- final-partition access guard;
- one-shot final evaluation;
- no post-final tuning under the same confirmation claim.

A model selected after many adaptive validation attempts must not be described as if it were a single predeclared fit.

---

# 8. Base teacher and final two-axis price test

For every untouched final state \(x_i\), report all three quantities:

\[
e_{\rm teacher}=V_{\rm NN}(x_i)-L_N(x_i),
\]

\[
e_{\rm deep}=V_{\rm NN}(x_i)-L_{N_{\rm ref,price}}(x_i),
\]

\[
e_{\rm discretization}=L_N(x_i)-L_{N_{\rm ref,price}}(x_i).
\]

This answers separately:

1. Did the NN learn its finite-depth teacher on unseen states?
2. Is the frozen NN also close to a better numerical American-price reference?
3. How much of any final disagreement belongs to teacher discretization rather than the NN?

Do not silently assign \(L_N-L_{N_{\rm ref,price}}\) to neural approximation error.

---

# 9. Selecting the deep price reference

## 9.1 Partition-free selection set

Use a deterministic reference-convergence set independent of train, validation, holdouts, and final rows.

## 9.2 Depth ladder

Freeze a finite ladder before running it, for example:

\[
1024/1025
\rightarrow
2048/2049
\rightarrow
4096/4097
\rightarrow
8192/8193.
\]

The exact ladder and maximum compute budget belong in the v2 reference config.

## 9.3 Convergence criterion

The criterion must be independent of the current model's achieved error.

The **application price-tolerance family** already developed in Task 9H is the starting point for v2 rather than something to be re-derived from the new model:

- the vol-equivalent price criterion where assessable;
- the flat normalized `5.5e-4` practical screen.

The separate question “how small must numerical-reference error be relative to the application tolerance?” is a **reference-design constant** and must be committed before the v2 convergence study. A candidate is 10% of the tight practical normalized screen:

\[
0.10 \times 5.5\times10^{-4}=5.5\times10^{-5}.
\]

That 10% factor is not justified by the current NN error and must never be changed because a model performs better or worse.

The rationale for the 10% reference-error budget is **additive systematic error control**. If the numerical-reference offset can be as large as 10% of the application tolerance and it happens to align with model error, the observed comparison can move by up to 10% of the tolerance boundary. Unlike independent zero-mean noise, this term must not be justified by root-sum-of-squares reasoning. It is acceptable only because the final decomposition reports \(L_N-L_{N_{\rm ref}}\) explicitly, so the systematic term is observable rather than hidden inside NN error.

Preferred form:

> The deeper price reference is adequate when successive-depth disagreement is below the predeclared reference-error budget derived from the application tolerance at declared robust and tail statistics.

Candidate statistics:

- median normalized depth difference;
- p99 normalized depth difference;
- maximum on a deterministic numerical stress set.

Historical Task 9H evidence is only a feasibility check: the observed American `1024/1025` versus `2048/2049` maximum normalized price difference on the existing depth-check set was about `2.88e-5`, below the candidate `5.5e-5` budget by only about **1.9x** on the maximum statistic. That is useful but not generous headroom, especially because v2 will use a different deterministic convergence set.

Therefore the v2 compute budget and ladder must comfortably permit at least the next refinement (for example `4096/4097`) rather than presuming `2048/2049` will qualify. The selection rule still chooses the smallest depth that passes; budgeting for 4096 is not the same thing as preselecting it.

v2 must repeat the convergence study under its committed reference configuration rather than inherit the historical result as the final freeze.

## 9.4 Full final-row evaluation

Once \(N_{\rm ref,price}\) is frozen, compute the deep price reference on every primary final row unless the predeclared compute ceiling makes that impossible.

---

# 10. Independent PDE evidence

A separately validated American PDE solver should provide additive evidence because it does not share the CRR lattice family.

The PDE protocol must itself be convergence-checked.

Preferred use:

- deterministic subset selected independently of NN error;
- at least two PDE grids/resolutions;
- record PDE self-refinement;
- compare the converged/finer PDE result against both \(L_N\), \(L_{N_{\rm ref,price}}\), and the NN.

The PDE is not selected after observing which numerical reference favors the NN.

---

# 11. Neural training contract

Train the surrogate against the base teacher \(L_N\), not the final deep reference.

Reasons:

- base generation remains tractable;
- training target is explicit and reproducible;
- the final deeper reference remains independent of model fitting;
- teacher fidelity and numerical-reference fidelity remain separable.

Deep final-reference values may not influence architecture, loss, epoch selection, deployment-head parameters, or acceptance thresholds.

---

# 12. Deterministic model artifact

The production contract must not depend on Python pickle.

Recommended artifact:

```text
american-neural-v2/
├── manifest.json
├── weights.bin
└── test-vectors.json
```

The manifest records:

- schema version;
- exact feature order;
- representation;
- architecture;
- activation;
- dtype;
- scaling;
- output/deployment head;
- floor/margin/temperature parameters where applicable;
- weight hash;
- dataset identity;
- attempt identity;
- selected epoch;
- training/source provenance;
- limitations.

The format must be deterministic and loadable from C++.

---

# 13. Native C++ inference is a first-class architecture phase

## 13.1 Why

The current Python-facing latency includes framework and dispatch overhead that will not exist in a native deployment.

Therefore the final speed question is not:

> Is PyTorch faster than CRR?

It is:

> Is the exact frozen deployed neural pricing function, implemented natively, faster than the exact matched native CRR economic request?

## 13.2 Early exploratory de-risking prototype

This workstream may run **in parallel with the remaining current Greek diagnostics**. It answers an architecture question, not a question about whether E2c becomes the final model.

Build a minimal C++ implementation of the **current frozen E2c deployed function**:

- load the current frozen weights;
- float64;
- exact current feature representation;
- exact 199k-parameter MLP;
- exact analytic European leg;
- exact intrinsic leg;
- exact smooth floor/projection;
- no retraining;
- no architecture shrinkage;
- no float32.

First require numerical equivalence against fixed Python test vectors.

Then benchmark batch 1 and batch 8 against the same matched C++ CRR adjacent-average request.

The prototype must measure implementation choices that simple FLOP/bandwidth arithmetic cannot settle:

- handwritten/fused dense kernels versus an available BLAS path;
- per-layer dispatch overhead;
- tanh implementation cost;
- cache-hot repeated-call behavior;
- batch-1 versus batch-8 behavior.

A rough 40–160 microsecond batch-1 range may be used as a plausibility hypothesis only, never as evidence.

This prototype is exploratory architecture evidence only. E2c's weights are a concrete specimen; because the v1 teacher dataset is not reproducible from current committed source, E2c is not eligible to become the v2 confirmation model.

## 13.3 Do not assume a speedup

Memory-bandwidth and FLOP calculations may be recorded as engineering estimates, but the architecture makes no 20x, 40x, or 10x assertion until native timings exist.

## 13.4 Production native path

The final native path must include all per-request work:

1. feature construction;
2. scaling;
3. fixed MLP;
4. activation;
5. inverse target transform;
6. analytic European leg if required;
7. intrinsic;
8. floor/projection;
9. physical price.

No timed call may fall back into Python.

---

# 14. Native benchmark protocol

Preferred benchmark:

```text
same executable
same process
same input arrays
same dtype
same CPU/affinity
same thread budget
same timing clock

        native C++ CRR N + N+1
                    vs
        native C++ frozen NN
```

Report at least:

- batch 1 median/p95;
- batch 8 median/p95;
- one larger throughput batch if operationally relevant.

Exclude:

- training;
- dataset generation;
- one-time model export;
- one-time startup load from the primary steady-state latency figure.

If startup/load latency matters, report it separately.

The CRR timed operation must compute both \(N\) and \(N+1\) if the economic reference is their adjacent average.

---

# 15. Greek architecture: three distinct objects

Always distinguish:

1. surrogate derivative;
2. finite-depth CRR operator derivative/Greek;
3. continuous-American Greek.

Agreement with (2) does not establish (3) unless the numerical Greek reference is itself adequately converged.

---

# 16. General Greek reference

## 16.1 Surrogate

Development: PyTorch autograd.

Future native deployment: either native analytic/backprop derivatives or a separately declared price-only first release.

Do not advertise native Greek latency until native Greek code exists.

## 16.2 Delta/Gamma

A future C++ CRR enhancement may expose standard lattice-local Delta/Gamma from early tree layers.

Advantages:

- avoids finite-difference cancellation from repeated full spot repricing;
- gives a natural finite-depth lattice Greek.

Limitation:

- it is still a finite-depth lattice quantity;
- depth convergence remains mandatory;
- its spatial resolution does not resolve arbitrarily narrow model projection spikes.

## 16.3 Vega

Until a differentiated-tree Vega is separately reviewed, use a predeclared volatility bump ladder and adjacent-averaged price operator.

## 16.4 Per-Greek depth selection

Select and report depth convergence separately for:

- Delta;
- Gamma;
- Vega.

The Greek reference may require greater depth than the price reference.

## 16.5 Claim rule

If the reference for a Greek is not converged under its predeclared rule:

- report NN vs finite-depth operator behavior;
- report the reference limitation;
- do not claim continuous-American Greek accuracy.

---

# 17. Sensitivity-degeneracy units

The current Task 9H implementation must be repaired before its Grid-1 degeneracy output is interpreted.

## 17.1 Vega

The deployed and raw pricers return physical price \(V\), so autograd Vega is

\[
\frac{\partial V}{\partial \sigma}.
\]

The declared degeneracy threshold `1e-8` is documented as a **normalized-Vega** scale. For the representation

\[
u=\frac{V}{A},\qquad A=S e^{-qT},
\]

\(A\) is independent of volatility, hence

\[
\frac{\partial u}{\partial \sigma}
=
\frac{1}{A}\frac{\partial V}{\partial \sigma}.
\]

Therefore the declaration-faithful current-study metric is:

\[
|\mathrm{Vega}_{physical}|/A \le 10^{-8}.
\]

Equivalently, the physical threshold is row-dependent: \(A\times10^{-8}\).

Surrogate and CRR-reference Vega must be transformed identically before counting reference degeneracy or **excess** surrogate degeneracy.

This repair changes the degeneracy counts, so unlike the depth-convergence propagation repair it is a **semantic protocol repair**, not merely extra reporting. It must be committed before Grid 1 observes model results.

## 17.2 Gamma

The same `1e-8` normalized-Vega threshold must not be reused for physical Gamma. Gamma has different units and the normalized representation does not reduce to `Gamma/A`.

No Gamma-degeneracy statistic is defined, and none is expected. Any Gamma-near-zero observation is:

- descriptive only if reported with explicit units;
- not an IV/Newton failure metric;
- not decision-bearing.

The IV solver concern is specifically Vega degeneracy. See §17.5 for the condition under which a Gamma statistic could ever be introduced.

## 17.3 Operational Vega-excess buckets

The decision-bearing excess-degeneracy report should not collapse all apparent reference/non-reference differences into one count.

For each row, report three mutually exclusive buckets:

1. **both exactly zero** — deployed Vega and reference Vega are exactly zero; this is a genuine zero-sensitivity region and not manufactured excess;
2. **deployed zero / reference resolvably nonzero** — the deployment has manufactured a zero Jacobian where the reference is demonstrably nonzero; this is the IV/Newton failure signal;
3. **deployed zero / reference unresolved** — the reference is too close to its numerical uncertainty to classify the row; report as inconclusive rather than excess.

“Exactly zero” must be literal floating-point zero only where the underlying implementation/reference actually returns it; near-zero values are handled by the normalized degeneracy threshold and reference-resolution rule.

## 17.4 Reference qualification

Any Vega excess-degeneracy conclusion inherits the Greek reference's depth-convergence and bump-resolution qualification. A row cannot be confidently classified as “reference non-degenerate” when the reference Vega itself is numerically unresolved at the relevant scale.

## 17.5 Gamma degeneracy is not a default metric

Do not invent a Gamma-degeneracy threshold merely for symmetry with Vega.

The Vega metric exists because the declared downstream IV Newton workflow divides by Vega. No currently declared downstream workflow divides by Gamma. Therefore Gamma requires:

- accuracy;
- sign/convexity behavior;
- crossover curvature diagnostics;

but **no degeneracy statistic** unless a future declared workflow has a denominator or failure mechanism that makes Gamma-near-zero operationally relevant.

---

# 18. Projection-crossover Greek study remains reference-free

Native lattice Gamma does **not** replace the narrow crossover diagnostic.

For a frozen deployment transformation, retain dense label-free scans that compare at identical states:

\[
\Gamma_{\rm deployed}^{\rm autograd}
-
\Gamma_{\rm raw}^{\rm autograd}.
\]

Likewise inspect:

\[
\Delta_{\rm deployed}-\Delta_{\rm raw},
\qquad
Vega_{\rm deployed}-Vega_{\rm raw}.
\]

This directly measures curvature/sensitivity manufactured by the deployment transform and carries no CRR finite-difference error.

Run separate spot and volatility crossover scans with frozen spacing and contract construction.

These scans are adversarial structural diagnostics, not primary final-distribution statistics.

---

# 19. Structural/arbitrage checks

The final frozen model must be checked separately for:

- non-negativity;
- intrinsic lower bound;
- valid European lower bound;
- call/put spot monotonicity;
- spot convexity;
- volatility monotonicity where mathematically applicable;
- smooth behavior across deployment crossovers;
- material projection-induced Gamma spikes;
- manufactured sensitivity-degeneracy regions.

Finite-bump shape tests and pointwise derivative tests are different functionals and must remain separately reported.

---

# 20. Final price and Greek metrics

Exact confirmatory thresholds belong in a separate acceptance file and are frozen only after the current exploratory diagnostics are complete.

The threshold design must be economically/numerically motivated, not chosen merely to make the current model pass.

## Price

At minimum:

- MAE;
- RMSE;
- p95;
- p99;
- max;
- signed bias;
- normalized versions;
- option-type/maturity/moneyness/volatility/exercise slices.

Report all against:

- \(L_N\);
- \(L_{N_{\rm ref,price}}\).

## Delta

Absolute-error metrics, sign disagreement, slices, reference qualification.

## Gamma/Vega

Absolute error plus relative error only where the denominator is numerically resolvable.

Always report:

- reference convergence;
- uncertainty;
- excluded relative rows;
- sign violations;
- degenerate-sensitivity regions.

---

# 21. Complexity statement

Allowed wording:

> Standard CRR backward induction has \(O(N^2)\) time per option. A frozen neural network with \(P\) parameters has fixed \(O(P)\) inference work, which is \(O(1)\) with respect to CRR lattice depth \(N\).

Do not describe the neural network as literally constant-complexity with respect to model size.

---

# 22. Implied volatility and calibration boundary

The surrogate replaces the American pricing kernel.

It does not automatically replace:

- root finding for implied volatility;
- calibration optimization algorithms.

Those algorithms can benefit from cheaper repeated pricing calls and, where validated, differentiable sensitivities.

---

# 23. Sequencing after this review

Two exploratory workstreams now run in parallel before the v2 freeze:

```text
current Greek contract repairs ──► Grid 1 / Grid 2 / Grid 2b ──► head-design guidance
                         ╲
                          ╲
                           independent in schedule
                          ╱
                         ╱
native E2c C++ prototype ─────────► handwritten vs BLAS/tanh/cache evidence
```

Neither workstream may open the final interpolation partition or train a new neural attempt.

## Phase 0A — repair and finish the current Greek diagnostics

Before any Grid-1 model result is observed:

1. implement the depth-convergence qualification propagation;
2. record reference-contract digest **and** reference protocol commit in Grid-1 artifacts;
3. repair normalized-Vega degeneracy units;
4. remove Gamma degeneracy as a reported statistic; Gamma is assessed on accuracy, sign/convexity behavior, and crossover curvature, and has no degeneracy metric (§17.2, §17.5);
5. commit/test the semantic repair;
6. make the commit message/decision record state explicitly that **no Grid-1 model result had been observed under either the old or repaired degeneracy definition when the repair was made**;
7. if any artifact exists under a superseded semantic definition, preserve it byte-for-byte and publish the repaired result under a new schema/path with an explicit `supersedes` link; never overwrite historical decision-bearing evidence.

Then run:

- Grid 1 validation;
- Grid 1 H1;
- Grid 2 spot crossover;
- Grid 2b volatility crossover.

The output of this phase is **head/representation guidance**:

- raw bad + deployed bad → learning/representation problem;
- raw good + deployed bad → deployment-head problem;
- crossover spikes → explicit projection-design problem;
- reference not converged → restrict the Greek claim.

It is not a decision that historical E2c weights can become the v2 confirmation artifact.

## Phase 0B — native C++ architecture prototype, in parallel

Using frozen E2c only as a concrete specimen:

1. implement exact float64 native forward inference;
2. validate Python/C++ equivalence;
3. compare handwritten/fused dense kernels and the chosen BLAS option;
4. measure tanh and whole-head costs;
5. benchmark matched batch 1 and batch 8 against native CRR.

This yields exploratory architecture evidence only.

## Phase 1 — freeze v2.3 architecture and acceptance/reference protocol

After Phase 0A/0B are understood, commit:

- architecture/head family allowed for v2;
- application price tolerance;
- price-reference error budget;
- Greek-reference rules;
- attempt budget and development-holdout policy;
- primary neutral sampling design;
- in-domain stress design;
- final-access policy;
- native benchmark contract.

## Phase 2 — restore reproducible generator and atomically generate v2 data

1. restore/rewrite the American generator in committed source;
2. commit the single generation config;
3. generate **all** v2 partitions in one run;
4. publish one manifest binding all partition digests and sampling semantics;
5. verify exact deterministic regeneration on a CI-sized fixture.

The final partition is newly generated but remains untouched because its sampling design was committed before generation and it is inaccessible to development code.

## Phase 3A — v2 training campaign

Train only on train/validation under the fixed attempt budget and holdout rules.

## Phase 3B — reference freeze, in parallel with training

Because the reference study is partition-free and model-independent once the application/reference rules are committed, run in parallel with Phase 3A:

- price-depth convergence;
- per-Greek depth/bump/reference-method convergence;
- independent PDE refinement protocol.

Freeze the resulting numerical reference methods before final evaluation.

No model-error result is used to choose the reference depth.

## Phase 4 — freeze/export final v2 candidate

After training selection:

- freeze the candidate;
- export deterministic artifact;
- implement/verify the production native path.

## Phase 5 — one-shot final evaluation

Open the untouched final partition once.

Compute:

- base teacher;
- deep price reference on all primary rows;
- frozen NN;
- price decomposition;
- Greek evidence under the frozen reference contract;
- structural/arbitrage evidence;
- separately reported in-domain stress results.

## Phase 6 — final native latency benchmark

Run frozen native C++ CRR versus frozen native C++ NN under the committed matched benchmark.

No tuning after the final report under the same confirmation claim.

# 24. Open questions remaining after review

The following are now considered resolved in principle:

- the **application price-tolerance family** transfers from the Task 9H B.2 declaration;
- the reference-error fraction must be independently predeclared rather than tied to achieved NN error;
- native backend selection is an empirical prototype question;
- E2c is not eligible as the v2 confirmation model;
- all v2 partitions are generated atomically;
- the reference study can run in parallel with training after the protocol freeze.
- **no Gamma-degeneracy statistic exists or is required.** §17.5 settles this: the Vega metric exists only because the declared downstream IV Newton workflow divides by Vega, and no Gamma statistic is defined unless a future declared workflow has a denominator or failure mechanism that makes Gamma-near-zero operationally relevant. Nothing is to be invented for symmetry.

Still open:

1. Do the current Greek diagnostics support retaining the residual + smooth-floor **design family**, or should the deployment head change before v2 training?
2. Should the v2 base teacher remain \(N=1024\) after the reproducible generator is restored?
3. Is `10% × 5.5e-4 = 5.5e-5` the final price-reference error budget, and at which robust/tail statistics must it hold?
4. What finite depth ladder and compute ceiling should be frozen separately for Delta, Gamma, and Vega?
5. Should native lattice Delta/Gamma be implemented before v2 final evaluation?
6. Is the first v2 production native API price-only, with native Greeks following separately?
7. Which measured native dense/tanh implementation wins the Phase-0B prototype and is acceptable as a maintained dependency?
8. What deterministic weight format should be used?
9. What exact v2 attempt budget and development-holdout access policy should be frozen?
10. What economically justified Greek tolerances are required for the intended downstream use?
11. What fixed subset and refinement criteria should the independent PDE cross-check use?

# 25. Architecture decision statement

The following principles are considered accepted unless a later versioned review explicitly changes them:

- C++ CRR is the offline teacher/reference.
- \(N\) and \(N+1\) are distinct computations and their outputs are persisted.
- The American dataset must be reproducible from committed source/configuration.
- The base label and deeper numerical reference are separate objects.
- Unseen \(x\) and deeper \(N\) test different things.
- The primary final interpolation set is neutrally sampled.
- All dataset partitions are generated atomically under one committed generator/config/manifest.
- Model-specific crossover scans are separate stress diagnostics.
- Adaptive attempt-budget and holdout controls survive the rewrite.
- The model artifact is deterministic and non-pickle.
- Native C++ NN inference is the production latency path.
- Price-reference and Greek-reference depths may differ.
- Once the application/reference protocol is frozen, numerical-reference selection is model-independent and may run in parallel with training.
- Continuous-American Greek claims require adequate reference convergence.
- Vega degeneracy is evaluated in declared normalized units. Gamma has **no degeneracy statistic**, and none is expected: it is assessed on accuracy, sign/convexity behavior, and crossover curvature, and gains a degeneracy metric only if a future declared workflow makes Gamma-near-zero operationally relevant (§17.5).
- Projection-crossover derivative artifacts are tested by raw-vs-deployed surrogate self-comparison.
- Independent PDE evidence is additive.
- Final evaluation is one-shot and cannot feed another development iteration under the same claim.

These principles are committed. The remaining open questions above are resolved during the exploratory Phase 0 and the Phase 1 protocol freeze; none of them reopens a principle in this section, and none may be resolved by observing how a model performs.
