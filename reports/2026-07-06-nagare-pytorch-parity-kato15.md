# NAGARE vs PyTorch — forward parity sanity test on kato15

Created-at: 2026-07-06 19:50 JST
Host: kato15 — Intel **i9-14900KF** (32 logical), Linux 5.15; PyTorch 2.11.0+cu128 (8 CPU threads); rustc/cargo 1.96.1. Nagare timings from the framework `hymeko_nagare` parity example (bit-identical to the standalone repo).

## Question

"Does Nagare outperform PyTorch?" — run the shared-fixture forward parity on a
fast modern host (not the laptop the cached baseline used) and see.

## Method (reused harness — §6.1)

`scripts/dev/global_pool_entropy_parity_fixture.py` builds a shared fixture and
times the PyTorch `ParityEntropyNet` forward; `cargo run --example
global_pool_entropy_parity` reads the **same** fixture and times the equivalent
Nagare forward. Tasks moons/rings/xor, n_test=96, n_points=48, hidden=32,
seed 123, **300 repeats after 20 warm-ups, median µs/sample**. PyTorch pinned to
8 threads (matches the cached baseline); Nagare run at **8 threads (matched)**
and **32 threads (all cores)** via `RAYON_NUM_THREADS`.

## Result

Figure: `reports/2026-07-06-nagare-pytorch-parity-kato15.png`.

| task | PyTorch (8t) | Nagare (8t) | Nagare (32t) |
|---|---:|---:|---:|
| moons | 9.1 | 22.5 | 15.7 |
| rings | 9.0 | 22.5 | 15.6 |
| xor | 7.6 | 22.5 | 15.6 |

**Peak RSS:** Nagare **4.9 MiB** vs PyTorch **780 MiB** (~**160×**).

## Verdict (honest, workload-dependent)

- **Forward speed: PyTorch wins.** 1.8–2.6× faster on this shape. MKL
  vectorization beats Nagare's per-row scalar Rust loops. More cores *narrow*
  the gap (Nagare 22.5→15.6 µs from 8→32 threads, ~1.4×) but do **not** close it
  — this is a per-thread FP-throughput / vectorization gap, not a core-count
  one. On the faster i9 the gap is *wider* than the laptop's ~1.6× (fast
  single-thread + MKL favor PyTorch more).
- **Memory: Nagare wins ~160×** (4.9 vs 780 MiB). Structural: a static Rust
  binary vs the Python+torch runtime + allocator stack. Platform-robust.

So **Nagare does not outperform PyTorch on raw forward latency** for this
entropy-feedback shape; its edge is **memory footprint and deploy** (tiny static
binary, no runtime). This confirms and sharpens the cached verdict on a faster
host. The cached **Chebyshev-deploy** path is the workload where Nagare wins
*speed* too (1.7–1.9×) — i.e. the answer is workload-dependent, and Nagare's
value proposition is deploy-cost/memory, not beating MKL at dense forward.

## Caveats

- **CPU-only.** This harness has no GPU or large-scale path; it is a small-batch
  forward. The architecturally interesting regime (large signed cycle pools,
  where Nagare's SoA MapReduce might diverge from dense tensors) is **not** tested
  here — a separate, larger harness would be needed.
- Host idle not independently verified (§6.5 #17) — a quiet-machine re-run would
  tighten the numbers, but the ~2× speed gap and ~160× memory gap are far larger
  than plausible contention noise.
- Per-engine timers differ (`perf_counter_ns` vs `Instant`) but both are
  median-of-300; the framework's original parity vetted this equivalence.

## Provenance

- Cached baseline (laptop, Ryzen 5900HX): `reports/2026-07-04-nagare-holonomy-package.md`
  (PyTorch 38–43 vs Nagare 64–84 µs; RSS 9 vs 636 MiB).
- Fixture/summaries on kato15 `/tmp/hajdu/parity-{fixture.txt,pytorch.json,nagare-8t.json,nagare-32t.json}` (ephemeral).
- PyTorch net: 4,644 params. Seeds: fixture 123. No GPU used.
