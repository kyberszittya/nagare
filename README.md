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

## Run

```powershell
cargo test
cargo run --release --example toy_compare -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --epochs 50 --batch-size 32 --lr 0.05 --seed 53 --out reports/latest-toy-run.json
```

## PDF Report

A compact PDF overview is available at `docs/nagare.pdf`; the editable LaTeX
source is `docs/nagare.tex`.

## Layout

- `src/datasets.rs`: generated moons, spiral, xor data and corruption.
- `src/features.rs`: quaternion periodic feature lift using `hymeko_clifford` rotors.
- `src/pooling.rs`: global mean/spread/max/sign-entropy pooling.
- `src/projection.rs`: fixed and fitted holonomy projection basis.
- `src/learner.rs`: local learner, gates, timing, stress ablation.
- `src/metrics.rs`: cross entropy and Clifford probability error using `hymeko_clifford`.
- `examples/toy_compare.rs`: runnable benchmark/example.
- `reports/`: copied result reports and raw artifacts from the HyMeKo run.

## Status

This is intentionally narrow and honest:

- proven only on small generated toy tasks so far;
- projection backward is not yet a native kernel;
- PyTorch GPU comparison is not meaningful at these tiny batch sizes without
  larger throughput benchmarks;
- graph/pgraph and HSIKAN adapters are future work.
