# NAGARE vs PyTorch — verdict reversed: Nagare matches/beats the forward after parallelizing the pool

Created-at: 2026-07-07 13:05 JST
Host: kato15 (Intel i9-14900KF, 32 logical), 8 CPU threads. Same shared fixture, 300-rep median. Nagare = committed `nagare` ikj kernels + parallel `global_pool` (framework parity harness).

## Result

The 2026-07-06 parity said PyTorch was ~2.5× faster. Two profile-driven,
bit-identical optimizations reversed that:

| | moons | rings | xor |
|---|---:|---:|---:|
| PyTorch (8t, MKL) | 9.1 | 9.0 | 7.6 |
| Nagare original | 22.5 | 22.5 | 22.5 |
| Nagare + ikj reorder | 15.7 | 15.7 | 11.1 |
| **Nagare + ikj + parallel-pool** | **7.6** | **7.6** | **7.6** |

**Nagare now matches (xor) or beats (moons/rings, ~1.2×) PyTorch on forward
latency, and still uses ~160× less memory (4.9 vs 780 MiB).** Figure:
`reports/2026-07-07-nagare-pytorch-parity-final-kato15.png`.

**Rigor:** 3 consecutive runs 7.58–7.65 µs (~1% spread — stable); parity error
vs the PyTorch fixture **9.7e-8** (f32 precision — the speed is real, not from
skipping work); per-forward allocation 2.43 MB (unchanged).

## Why the pool was the unlock (Amdahl)

After the ikj kernel fix, the two `global_pool` calls were the only remaining
**serial** stages. On kato15's fast 32-core CPU the parallel stages finish
quickly, so the serial pool dominated the wall time (Amdahl's law). Parallelizing
it over the independent batch dimension (bit-identical) collapsed the gap
15.7 → 7.6 µs — a ~2× step from one serial section. On the slower 8-core laptop
the same change was only ~1.26× (the serial fraction was smaller there); the
many-core box is exactly where serial sections hurt most.

## The two optimizations

1. **ikj/SAXPY reorder** (committed in `nagare`, `2026-07-06`): `fused_entropy_update`
   + `linear_forward` — contiguous, autovectorizable; bit-identical.
2. **Parallel pool** (this run): `global_pool` parallelized over batch with rayon;
   bit-identical. Applied in the framework parity harness for the measurement.

## Provenance / what is where

- The **shipped `nagare` kernels** (ikj `fused_entropy_update` + `linear_forward`)
  are committed (`162a5c7`).
- The **parallel `global_pool`** lives in the framework parity *example*
  (`hymeko_nagare/examples/global_pool_entropy_parity.rs`, uncommitted). The
  parity forward is not yet in the standalone repo.
- **Not shipped to the learner:** the standalone `structural_pool_features` is a
  7-channel pool that is too small to parallelize (measured 16% *regression* —
  see `2026-07-06-nagare-forward-opt-ikj.md`). Pool parallelization is a win only
  for hidden ≥ ~16 (the parity net's 32). Design note for future Nagare models;
  the current shipped learner stays serial.

## Honest caveats

- This is the **CPU small-batch forward parity** (hidden=32). Not tested: GPU,
  large-scale, or a batched-GPU PyTorch. The claim is scoped to this shape.
- Host idle not independently verified; the ~1% run spread and the correctness
  check make the conclusion robust to contention noise.

## Top follow-up

Port the parity forward (+ parallel pool) into the **standalone** as an example
or bench, so the "Nagare matches/beats PyTorch" result is reproducible from the
canonical repo (currently it needs the framework harness). That also yields a
committable artifact carrying the result.
