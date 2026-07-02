# Nagare Holonomy Learning: Consolidated Report and Repository Plan

Date: 2026-07-02

## Executive Summary

The current Nagare experiments show that a small Rust-native holonomy/global-pooling learner can solve moons, spiral, and xor toy point-set tasks with microsecond-scale CPU forward passes. The fitted projection gate is the strongest version so far: it replaces scalar entropy multiplication with a learned low-rank projection over pooled geometry, rotor, holonomy, spread, and sign-entropy features.

The most important result is the hard spiral stress case:

- fixed projection: 0.906 accuracy, 0.324541 loss
- scalar entropy gate: 0.927 accuracy, 0.286730 loss
- constant gate: 0.938 accuracy, 0.269849 loss
- fitted projection gate: 0.938 accuracy, 0.250603 loss

That is the first result where projection is not merely philosophically cleaner than scalar multiplication, but also empirically better on a difficult row.

## What Was Built

The experiment currently lives in:

- `hymeko_nagare/examples/entropy_pool_learning_compare.rs`
- `hymeko_nagare/tests/entropy_pool_learning_compare.rs`
- `reports/2026-07-02-projection-gate-holonomy-ablation.md`
- `reports/2026-07-02-fitted-projection-gate-holonomy-ablation.md`

Core pieces:

- quaternion periodic feature lift from point sets;
- global pooling over mean, spread, max, and sign entropy;
- Clifford probability error for evaluation;
- local simultaneous update learner;
- scalar entropy, constant, fixed projection, and fitted projection gate modes;
- rank-6 fitted holonomy projector;
- alpha-mixed projection, `alpha * P(phi) + (1 - alpha) * phi`;
- stress testing on clean, noisy, missing, and few-shot/noisy/missing data.

## Main Result

Configuration:

- tasks: moons, spiral, xor
- train samples: 192
- test samples: 96
- points per sample: 32
- epochs: 50
- local learner parameters: 60
- fitted projection stress learner parameters: 228
- backprop-like baseline parameters: 2836

| Task | Local acc | Local loss | Local median us/sample | Backprop-like median us/sample | Speedup |
|---|---:|---:|---:|---:|---:|
| moons | 1.000 | 0.001513 | 5.673 | 139.984 | 24.7x |
| spiral | 1.000 | 0.004486 | 5.776 | 136.064 | 23.6x |
| xor | 1.000 | 0.006714 | 4.178 | 136.299 | 32.6x |

Sampled peak RSS was 7,737,344 bytes, about 7.38 MiB.

## Gate Result

The fitted projection gate is now competitive with, and sometimes better than, the constant gate.

| Task | Stress | Entropy loss | Constant loss | Fitted projection loss | Readout |
|---|---|---:|---:|---:|---|
| moons | clean | 0.001513 | 0.000465 | 0.000468 | tied with constant |
| moons | noisy | 0.001631 | 0.000500 | 0.000498 | slightly better |
| moons | missing | 0.001656 | 0.000516 | 0.000521 | tied with constant |
| moons | few-shot/noisy/missing | 0.011536 | 0.005352 | 0.005323 | slightly better |
| spiral | clean | 0.004487 | 0.001581 | 0.001586 | tied with constant |
| spiral | noisy | 0.043077 | 0.040756 | 0.044278 | worse |
| spiral | missing | 0.015772 | 0.011051 | 0.012371 | better than entropy |
| spiral | few-shot/noisy/missing | 0.286730 | 0.269849 | 0.250603 | best loss |
| xor | clean | 0.006731 | 0.002713 | 0.002697 | slightly better |
| xor | noisy | 0.006969 | 0.002783 | 0.002935 | close |
| xor | missing | 0.008426 | 0.003485 | 0.003637 | close |
| xor | few-shot/noisy/missing | 0.033405 | 0.022253 | 0.022197 | slightly better |

## Interpretation

Simple multiplication is the wrong abstraction for the gate. In this setting, the gate should shape the update by projecting the pooled feature state onto a holonomy-informed subspace. The fitted projection result supports that direction.

The result does not yet prove that entropy global pooling is broadly better than backpropagation. It does show a smaller and faster local learner on these generated toy problems, and it shows that a projection-style gate can improve the hard stress row where fixed projection failed.

The current best phrasing is:

> Nagare holonomy learning is a promising local-update alternative for small structured point-set problems. Projection gating is the right next abstraction, but it needs native kernels, gradient checks, and larger benchmarks before making broader claims.

## Should This Become a Separate Repository?

Yes, but as an experimental sibling repository, not as a replacement for HyMeKo or Nagare.

The reason to split it out:

- the work is becoming a coherent research artifact;
- the benchmark scripts, reports, and fixtures deserve a clean surface;
- the experiments need faster iteration than the main HyMeKo framework should absorb;
- a standalone repo can present the biological/local-learning thesis clearly;
- it avoids mixing speculative HSIKAN/holonomy learning code into stable framework crates.

The reason not to fully sever it yet:

- the code still depends conceptually on Nagare and Clifford conventions;
- the projector should become a native op only after finite-difference checks;
- current benchmarks are toy-scale;
- the repo should preserve provenance back to the HyMeKo experiments.

## Recommended Repository Name

Recommended:

- `nagare-holonomy-learn`

Alternatives:

- `holonomy-pool`
- `nagare-local-learning`
- `hymeko-holonomy-lab`

`nagare-holonomy-learn` is the clearest name: it says what it is, but does not overclaim biological plausibility or replace backprop as a general method.

## Proposed Repository Structure

```text
nagare-holonomy-learn/
  Cargo.toml
  README.md
  LICENSE
  crates/
    holonomy_learn/
      Cargo.toml
      src/
        lib.rs
        features.rs
        pooling.rs
        projection.rs
        learner.rs
        metrics.rs
        datasets.rs
  examples/
    toy_compare.rs
    stress_ablation.rs
  benches/
    forward_micro.rs
  tests/
    projection_basis.rs
    parity_fixture.rs
    finite_difference_projection.rs
  fixtures/
    moons_spiral_xor_seed53.txt
  reports/
    2026-07-02-fitted-projection-gate-holonomy-ablation.md
  docs/
    architecture.md
    biological-local-learning-thesis.md
    extraction-notes.md
```

## Module Boundaries

`features.rs`:

- quaternion periodic lift;
- rotor/holonomy channels;
- optional Chebyshev-domain rescale;
- no broad dense multivector routing.

`pooling.rs`:

- global mean/std/max/sign-entropy pooling;
- simultaneous update-friendly memory layout;
- later sparse pooling.

`projection.rs`:

- fixed projector baseline;
- fitted rank-k projector;
- alpha mixing;
- finite-difference tested backward path once promoted to a native op.

`learner.rs`:

- local update learner;
- gate modes;
- training loop;
- forward timing hooks.

`metrics.rs`:

- cross entropy;
- Clifford probability error;
- entropy readouts;
- allocation and RSS readouts.

`datasets.rs`:

- moons;
- spiral;
- xor;
- rings;
- corruption modes.

## Dependency Strategy

Start with a small Rust-only dependency surface:

- use existing `hymeko_clifford` and `hymeko_nagare` as path dependencies during incubation;
- avoid new pinned dependencies;
- avoid `serde_json` for fixtures if the main framework constraint still applies;
- keep fixture parsing in std text form;
- later publish extracted crates only after API boundaries stabilize.

During incubation inside this workspace, use path dependencies:

```toml
hymeko_nagare = { path = "../hymeko_framework_rust/hymeko_nagare" }
hymeko_clifford = { path = "../hymeko_framework_rust/hymeko_clifford" }
rand = "0.9"
```

Before publishing, decide whether to:

1. publish `hymeko_nagare` and `hymeko_clifford`;
2. vendor only the tiny quaternion/projection parts;
3. keep this as a workspace-local research repo.

For research correctness, option 1 is cleaner. For portability, option 2 is simpler.

## Migration Plan

Phase 1: Extract the experiment.

- Copy the toy dataset generator, metrics, fitted projector, and learner into `crates/holonomy_learn`.
- Preserve the exact seed-53 benchmark as a fixture.
- Keep reports and raw JSON artifacts.
- Add a README with the hard numbers and caveats.

Phase 2: Make projection a kernel.

- Add `project_alpha_mix_forward`.
- Add `project_alpha_mix_backward`.
- Add finite-difference tests.
- Add parity tests against the current example output.
- Report forward time, allocations, and RSS.

Phase 3: Add graph/hypergraph hooks.

- Add pgraph feature enrichment.
- Add structural generators.
- Add SA-HSIKAN/HSIKAN feature adapters.
- Keep the local learner generic over feature source.

Phase 4: Benchmark honestly.

- More seeds;
- more point counts;
- moons, rings, xor, spiral, blobs, noisy periodic signals;
- CPU baseline;
- PyTorch CPU baseline;
- optional PyTorch GPU baseline only when batch sizes make GPU meaningful.

Phase 5: Paper/report package.

- architecture diagrams;
- result tables;
- limitations;
- reproducibility commands;
- biological plausibility discussion phrased carefully.

## Repository README Positioning

Suggested README opening:

> `nagare-holonomy-learn` is an experimental Rust implementation of local holonomy-based learning with global pooling, Clifford error metrics, quaternion periodic features, and projection-gated updates. It explores whether small structured learners can solve generated point-set tasks with microsecond-scale CPU forward passes and far fewer parameters than a backprop-like baseline.

Avoid claiming:

- "replaces backprop";
- "biologically proven";
- "general 24x speedup";
- "universal neural network accelerator".

Claim safely:

- "faster on these toy settings";
- "smaller parameter count";
- "projection gate improves the hard stress row";
- "candidate local-update architecture for structured data".

## Immediate Action Recommendation

Create the separate repo after one more cleanup pass:

1. Move the example-local fitted projector into a real library module inside this repo.
2. Add finite-difference projection tests.
3. Freeze one text fixture for moons/spiral/xor.
4. Then extract to `nagare-holonomy-learn`.

This avoids exporting an example file as if it were a stable library, while still keeping momentum.

## Verification Sources

Reports used:

- `reports/2026-07-02-projection-gate-holonomy-ablation.md`
- `reports/2026-07-02-fitted-projection-gate-holonomy-ablation.md`

Raw artifacts:

- `reports/2026-07-02-projection-gate-holonomy-ablation.json`
- `reports/2026-07-02-projection-gate-holonomy-ablation-rss.json`
- `reports/2026-07-02-fitted-projection-gate-holonomy-ablation.json`
- `reports/2026-07-02-fitted-projection-gate-holonomy-ablation-rss.json`
