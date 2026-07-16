---
title: HSiKAN edge-chunk parallelization + vs-PyTorch CPU (time & memory)
date: 2026-07-16
component: HSiKAN (Highway-SignedKAN), src/ops/hsikan.rs
finding: F-HSIKAN-PAR
status: complete
---

# HSiKAN edge-chunk parallelization + vs-PyTorch CPU (time & memory)

## Summary

Two questions from the user: *how big is the memory advantage really*, and *can the
closed-form spine accelerate with threads*. Answers, measured on Mac arm64 vs PyTorch
2.12 (CPU), same HSiKAN architecture both sides (sign-routed Chebyshev splines +
Schmidhuber highway gate over signed hyperedges, arity 3, d=16, cheb 6, 2 branches):

1. **Memory advantage is a curve, not 160×.** Tiny model (iris/california KAN+MLP):
   Nagare **6.7 MB** vs PyTorch **281 MB** = **42×** — almost entirely PyTorch's fixed
   ~280 MB import baseline. At HSiKAN scale it narrows to **4–5×** (207 vs 843 MB @50k)
   as Nagare's own forward cache grows. No regime reaches 160×.

2. **The spine was single-threaded** (only `linear` used rayon; `kan`, `hsikan`,
   `clifford_fir`, `gomb_shell` = 0 rayon). So the per-core wins were Nagare-1-thread vs
   PyTorch-1-thread, and PyTorch's 6-thread default took wall-clock.

3. **Fixed:** `hsikan_forward`/`hsikan_backward` now partition the `T` hyperedges into
   `min(threads, T)` contiguous chunks on the rayon pool (each hyperedge is independent;
   param grads + `grad_x` summed once). **Scales 7.3× at 16 threads** while PyTorch
   saturates ~6–8 threads. At **matched 16 threads Nagare is ~5× faster AND ~3× less
   memory**.

## Results

### Single thread (per-core)

| edges | Nagare ms | PyTorch ms | Nagare RSS | PyTorch RSS |
|---|---|---|---|---|
| 20k | 24.1 | 52.5 | 86 MB | 459 MB |
| 50k | 62.0 | 139.7 | 207 MB | 843 MB |
| 100k | 123.8 | 282.7 | 405 MB | 1468 MB |

Per-core: **2.3× faster, 4–5× less memory.**

### Thread scaling (Nagare, 50k edges)

| threads | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| ms/iter | 61.1 | 31.3 | 17.6 | 12.1 | 8.3 |
| speedup | 1.0× | 1.95× | 3.47× | 5.05× | 7.34× |

Checksum identical across all thread counts (−27.1855) — correctness preserved.

### Matched 16 threads

| edges | Nagare ms | PyTorch ms | Nagare RSS | PyTorch RSS |
|---|---|---|---|---|
| 50k | 8.1 | 42.2 | 269 MB | 843 MB |
| 100k | 16.8 | 81.3 | 568 MB | 1468 MB |

**~5× faster, ~3× less memory.** PyTorch barely moved from 6→16 threads (51→42 ms @50k);
Nagare kept scaling.

### Cost of parallelism

Edge-chunk parallelism holds P transient `grad_x` copies (one per chunk) for the final
sum. Nagare RSS 1→16 threads: 196→269 MB (50k), 386→568 MB (100k) — a 37–47% increase,
still 2.6–3.1× under PyTorch. Memory-for-speed trade; the no-tape advantage survives.

## Design

`HsikanCache` (fields private) now wraps `Vec<ChunkCache>` + per-chunk edge counts; the
public API and the opaque cache token are unchanged. The whole-op computation moved to
`forward_serial`/`backward_serial`; `hsikan_forward` splits edges via `chunk_ranges`,
runs the serial path per chunk on `par_iter`, concatenates `h_e` in edge order, and
stores the chunk caches. `hsikan_backward` mirrors the partition and sums the per-chunk
gradients (`reduce_backward`). The sum is exact because every gradient is a per-edge
accumulation. `RAYON_NUM_THREADS=1` forces the serial path.

## Files touched

- `src/ops/hsikan.rs` — ChunkCache split, `forward_serial`/`forward_chunked`/`chunk_ranges`,
  `backward_serial`/`reduce_backward`/`add_into`, public entrypoints rewired, new test
  `chunk_parallel_matches_serial` (~+180 / −25 lines).
- `examples/hsikan_mem_bench.rs` — Nagare fwd+bwd bench, `--edges N`.
- `scripts/dev/hsikan_pytorch.py` — matched PyTorch HSiKAN with autograd.
- `scripts/dev/plot_hsikan_bench.py`, `reports/figures/hsikan-bench.png`,
  `reports/figures/hsikan_pytorch_bench.json`.
- Registries: `canonical_components.json` (HSiKAN → v1.1), `canonical_findings.json`
  (F-HSIKAN-PAR).

## CORE.YAML items touched

None (nagare repo; no CORE.YAML gate on this op). Change is additive + API-preserving.

## Tests

- `cargo test --release --lib`: **147 passed / 0 failed**.
- New `ops::hsikan::tests::chunk_parallel_matches_serial`: forward bit-close (<1e-6) and
  every backward gradient exact (<1e-5) vs the serial whole-set pass, for 2/3/6/16 chunks.
- `backward_matches_finite_difference` + `kb_backward_matches_finite_difference` still pass
  (closed-form gradients unchanged).
- `cargo clippy --all-targets --release`: 0 warnings. `cargo fmt`: clean.

## Reproduce

```
/usr/bin/time -l ./target/release/examples/hsikan_mem_bench --edges 50000   # RAYON_NUM_THREADS to vary
/usr/bin/time -l .venv/bin/python scripts/dev/hsikan_pytorch.py --edges=50000
```

## Provenance

- Host: Hajdus-MacBook-Pro, arm64, CPU. PyTorch 2.12.0.
- Nagare bench seed: LCG 0x12345678 (deterministic synthetic edges/params).
- Peak RSS: `/usr/bin/time -l` "maximum resident set size".
- kato15 (32 cores) synced to this work but has **no torch** — the PyTorch arm was not run
  there; a head-to-head at 32 threads needs a torch install on the cluster.

## Open items

- Only `hsikan` is parallelized; `clifford_fir`/`gomb_shell` stay serial — same edge/bank
  chunk pattern applies if the full Gömb-Soma cascade becomes the bottleneck.
- P-fold `grad_x` memory can be cut with a sparse per-chunk vertex-grad scatter if RSS
  becomes the binding constraint.
