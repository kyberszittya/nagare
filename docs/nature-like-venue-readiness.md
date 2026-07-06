# Nagare Nature-Like Venue Readiness Plan

Date: 2026-07-05

## Target Claim

Nagare is a closed-form holonomy learning runtime for signed hypergraph neural
computation. It replaces tape-based backpropagation for a defined class of
cycle-pool models with explicit Clifford/quaternion kernels, global entropy
pooling, and projection-gated updates.

The claim should stay precise:

- not "backpropagation is obsolete";
- not "biologically real learning" as a demonstrated biological result;
- yes: "for signed hypergraph models with cycle-pool structure, closed-form
  holonomy updates can match small-task accuracy while reducing CPU latency and
  memory traffic, and the mechanism is compatible with local/global feedback."

## Venue Fit

Primary candidates:

- Nature Machine Intelligence: best fit if the paper becomes a learning-method
  contribution with broad empirical validation across graph, geometric, and
  sequence tasks.
- Nature Methods: plausible only if Nagare becomes a reusable computational
  method that enables a class of analyses or experiments that were impractical
  before.
- Nature Communications / Communications AI & Computing: better fit if the
  result is solid and general but narrower than the top specialist venue.
- Scientific Reports: fallback for a careful, reproducible methods paper with a
  smaller conceptual reach.

## Minimal Evidence Bar

The current moons/spiral/xor results are useful smoke tests, not venue-grade
evidence. To target a Nature-like venue, we need all of the following:

1. A committed reproducibility package.
   - Seeded toy fixtures.
   - Frozen benchmark command lines.
   - CPU model, compiler, Rust version, thread count, and memory method.
   - Code availability and data availability statements.

2. Gradient and kernel correctness.
   - Finite-difference checks for every primitive forward/backward pair.
   - Parity tests against PyTorch reference implementations where applicable.
   - Clifford/quaternion-specific tests that do not collapse to scalar-only
     behavior.

3. Scale ladder.
   - Tiny generated tasks: moons, rings, spiral, xor.
   - Structured non-graph tasks represented as hypergraphs.
   - Real signed graph or hypergraph tasks.
   - Throughput benchmarks at increasing cycle counts.

4. Baselines.
   - PyTorch CPU reference.
   - PyTorch GPU reference where the batch/graph size is large enough to be
     meaningful.
   - Ablations: no entropy, scalar entropy, projection gate, fitted projection,
     quaternion periodic features, Clifford error, global pooling.

5. Mechanistic analysis.
   - Show why global entropy feedback changes the update trajectory.
   - Show where holonomy/projection helps and where it does not.
   - Report failures, instability modes, and sensitivity to projection rank.

## Figure Plan

Main figures:

1. Architecture.
   - HyMeKo parsed IR to Nagare cycle pool.
   - Clifford/quaternion feature lift.
   - Global entropy pooling.
   - Projection-gated closed-form update.

2. Kernel correctness.
   - Finite-difference error distributions.
   - PyTorch parity for forward and backward paths.

3. Toy and medium-scale performance.
   - Accuracy/loss across moons, rings, spiral, xor.
   - Forward time, allocations, and peak RSS.
   - CPU and GPU baseline comparison.

4. Signed graph / hypergraph task.
   - Same model class, real graph structure.
   - AUC/F1/loss versus baselines.

5. Ablation and mechanism.
   - Entropy feedback, projection rank, Clifford error, quaternion periodic
     components.
   - Update trajectory visualization.

Extended data:

- Full command table.
- Hardware table.
- Seed-wise results.
- Kernel microbenchmarks.
- Memory profile.

## Manuscript Spine

Suggested title:

Closed-form holonomy learning for signed hypergraph neural computation

Abstract shape:

1. Problem: neural learning on signed relational structures is usually executed
   through tensor/tape backpropagation, even when the model has sparse
   geometric structure.
2. Approach: Nagare compiles HyMeKo-style hypergraph structure into explicit
   cycle-pool kernels with Clifford/quaternion operations and entropy-mediated
   global feedback.
3. Result: On controlled tasks and signed relational benchmarks, Nagare matches
   baseline accuracy while reducing CPU forward latency and memory traffic.
4. Interpretation: Holonomy updates expose a local/global learning mechanism
   that is algorithmically distinct from generic backpropagation.

## Near-Term Work Items

1. Add HyMeKo model description files for moons, spiral, xor, and rings.
2. Add a PyTorch parity harness for the cycle-pool runtime, separate from the
   existing holonomy toy harness.
3. Add a benchmark manifest that records hardware, threads, seeds, and exact
   commands.
4. Implement real signed-graph benchmark ingestion through HyMeKo/HIVE where
   possible.
5. Extend the PDF report into a paper skeleton with Methods, Code
   Availability, Data Availability, and Reporting Summary sections.

## Editorial Risk Register

- Current evidence is toy-scale.
- The "biological reality" language is too strong unless reframed as
  motivation or analogy.
- Speedups must be reported with strict hardware and workload caveats.
- GPU comparison is only meaningful on workloads large enough to amortize launch
  and transfer overhead.
- The method must be presented as replacing backpropagation only for the
  supported structured model class, not for arbitrary neural networks.

## Reproducibility Commitments

- Public Git repository with exact source used for figures.
- Tagged release or archived snapshot before submission.
- Text fixtures for generated data where possible.
- No hidden Python preprocessing for headline results.
- All custom code necessary to reproduce claims is included or explicitly
  referenced as a HyMeKo dependency.

