# nagare-holonomy-learn

Experimental Rust implementation of local holonomy-based learning with global
pooling, Clifford error metrics, quaternion periodic features, and
projection-gated updates.

This repository explores whether small structured learners can solve generated
point-set tasks with microsecond-scale CPU forward passes and far fewer
parameters than a backprop-like baseline. It is a research artifact, not a
claim that backpropagation is generally replaced.

The crate reuses HyMeKo primitives where the logic is not Nagare-specific. In
particular, quaternion rotor operations and the Clifford probability error come
from `hymeko_clifford` via a path dependency during incubation.

The crate now also carries a Nagare cycle-pool runtime draft that composes
HyMeKo graph cycles through closed-form kernels:

```text
Clifford FIR -> scatter mean -> linear head -> BCE
            <- closed-form backward + Adam <-
```

The FIR implementation and `TopKCyclesBatch` type are reused from
`hymeko_graph`; Nagare owns the composition layer, scatter/linear/loss kernels,
optimizer wiring, and training/inference runtime.

## Current Result

From the 2026-07-02 fitted projection ablation:

| Task | Local acc | Local loss | Local median us/sample | Backprop-like median us/sample | Speedup |
|---|---:|---:|---:|---:|---:|
| moons | 1.000 | 0.001513 | 5.673 | 139.984 | 24.7x |
| spiral | 1.000 | 0.004486 | 5.776 | 136.064 | 23.6x |
| xor | 1.000 | 0.006714 | 4.178 | 136.299 | 32.6x |

Hard stress row:

| Gate | Spiral few-shot/noisy/missing acc | Loss |
|---|---:|---:|
| scalar entropy | 0.927 | 0.286730 |
| constant | 0.938 | 0.269849 |
| fixed projection | 0.906 | 0.324541 |
| fitted projection | 0.938 | 0.250603 |

The fitted projection learner reports 228 parameters in the stress ablation,
compared with 2836 parameters for the original backprop-like baseline.

> **Multi-seed caveat (2026-07-06).** The fitted-projection advantage in the hard
> stress row above is a **single seed (53)**. On a 3-seed median the *constant*
> gate is `<=` the fitted projection in every regime — the projection gate does
> **not** robustly beat a plain linear readout of the same pooled features. A
> point-order-shuffle ablation further shows the holonomy signal is real but
> **spiral-specific and gate-independent** (order-shuffle raises spiral loss
> ~10x, leaves moons/xor unchanged, and hits all gates equally). See
> `reports/2026-07-06-nagare-order-shuffle-ablation.md`.

## Run

```powershell
cargo test
cargo bench --bench parity_gate
cargo run --release --example toy_compare -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --epochs 50 --batch-size 32 --lr 0.05 --seed 53 --out reports/latest-toy-run.json
```

## PDF Report

A compact PDF overview is available at `docs/nagare_hymeko.pdf`; the editable
LaTeX source is `docs/nagare.tex`. The source can also be compiled to the
canonical `docs/nagare.pdf` path when that file is not open in a PDF viewer.

## Publication Target

The current publication-readiness roadmap is in
`docs/nature-like-venue-readiness.md`. It frames Nagare as a closed-form
holonomy learning runtime for signed hypergraph neural computation, with a
Nature-like evidence bar focused on reproducibility, kernel correctness,
baselines, scale, and ablations.

## Layout

- `src/datasets.rs`: generated moons, spiral, xor data and corruption.
- `src/features.rs`: quaternion periodic feature lift using `hymeko_clifford` rotors.
- `src/pooling.rs`: global mean/spread/max/sign-entropy pooling.
- `src/projection.rs`: fixed and fitted holonomy `ProjectionBasis`, applied via the FD-tested `project_alpha_mix` kernel.
- `src/learner.rs`: local learner, gates (entropy/constant/projection), timing, stress ablation incl. point-order shuffle.
- `src/metrics.rs`: cross entropy and Clifford probability error using `hymeko_clifford`.
- `src/runtime.rs`: cycle-pool runtime for FIR, scatter, linear, BCE, backward, Adam.
- `src/ops/`: explicit forward/backward kernels — `project_alpha_mix`, `catmull_rom`/Chebyshev-deploy, `cayley_rotor`, `fsr_mixer`, `fused_entropy_update`, `signed_scatter`, plus adam/linear/loss/scatter/clifford_fir.
- `tests/fixtures/moons_spiral_xor_seed53.txt`: FNV-hashed determinism guard for the frozen seed-53 datasets.
- `examples/toy_compare.rs`: runnable benchmark/example.
- `benches/parity_gate.rs`: Criterion benchmark for FIR/scatter/runtime throughput.
- `reports/`: copied result reports and raw artifacts from the HyMeKo run.

## Status

This is intentionally narrow and honest:

- proven only on small generated toy tasks so far;
- the projection gate now runs on a native FD-tested kernel
  (`project_alpha_mix`, forward + closed-form backward), but on multi-seed
  medians it does **not** beat a plain linear readout (see the caveat above);
- PyTorch GPU comparison is not meaningful at these tiny batch sizes without
  larger throughput benchmarks;
- graph/pgraph and HSIKAN adapters are future work.
