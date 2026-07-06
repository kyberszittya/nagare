# NAGARE — closing the PyTorch forward gap: profile-driven ikj/SAXPY kernel reorder

Created-at: 2026-07-06 20:20 JST
Repo: `nagare_github` (canonical; fix mirrored to the frozen framework crate for the parity remeasure). Host: kato15 (i9-14900KF). **All local; commit held.**

## Why

The 2026-07-06 parity found PyTorch ~2.5× faster than Nagare on the
entropy-feedback forward. I had attributed this to "MKL vectorization" **without
profiling** — a premature-certainty slip (CLAUDE.md operating principles). This
report does the profiling first, then the targeted fix.

## Profile (profiler-free; kato15 has no perf/flamegraph)

Added `examples/forward_profile.rs` (standalone): runs the exact parity op
sequence at the parity shape (batch 96, points 48, hidden 32) with per-stage
`Instant` timing, 300 reps. Before any change (local, µs/sample):

| stage | before | % |
|---|---:|---:|
| **fused_update** (129→32, 4608 rows) | **33.6** | **68.6%** |
| pool2 (serial mean/std/max) | 4.4 | 9.0% |
| pool1 (serial mean/std/max) | 4.1 | 8.3% |
| embed (linear 2→32) | 3.4 | 6.9% |
| head (linear 96→2) | 1.8 | 3.7% |
| first (linear 96→2) | 1.6 | 3.4% |
| entropy | 0.07 | 0.1% |

**The profile redirected the fix.** My suspected target (`linear_forward`) was
only ~14% combined; the real hot spot is **`fused_entropy_update` at 68.6%**.

## Root cause (same pattern in both kernels)

The inner reduction accumulated a scalar while striding W by `out_dim`:

```rust
for j { let mut acc = b[j];
        for i { acc += x[i] * w[i * out_dim + j]; }  // strided gather + scalar reduction
        out[j] = acc; }
```

A strided-gather scalar reduction is **not autovectorizable** by LLVM (and is
cache-hostile: column-major walk of a row-major W).

## Fix — `ikj` (SAXPY) reorder

Initialise the output row with the bias, then for each input `i` broadcast
`x[i]` and add the **contiguous** W-row into the **contiguous** output row:

```rust
out_row.copy_from_slice(&layer.b);
for &xi in x_row {                       // i outer
    for (slot, &w) in out_row.iter_mut().zip(w_row.iter()) {  // j inner: contiguous, no reduction
        *slot += xi * w;                 // autovectorizes to broadcast-multiply + add
    }
}
```

The inner j-loop writes distinct slots (no reduction) over contiguous memory →
LLVM autovectorizes. For a fixed j the additions still run in i-order, so the
result is **bit-identical** (Rust does not contract to FMA by default). **No new
dependency, no SIMD intrinsics, no precision change.** Applied to both
`fused_entropy_update_forward` and `linear_forward`.

## Measured (profile-confirmed, §3)

Per-stage after (local): `fused_update` **33.6 → 9.1 µs (2.9×)**; embed
3.4→2.7; whole timed forward **48.9 → 23.0 µs (2.1×)**. The improvement is
concentrated in the intended kernel (not incidental) — the §3 requirement for a
>10% speedup.

**Parity on kato15 (same fixture, 300-rep median, µs/sample):**

| task | PyTorch (8t) | Nagare before (8t) | Nagare **after** (8t) | after (32t) |
|---|---:|---:|---:|---:|
| moons | 9.1 | 22.5 | **15.7** | 12.7 |
| rings | 9.0 | 22.5 | **15.7** | 12.7 |
| xor | 7.6 | 22.5 | **11.1** | 12.5 |

**Gap: 2.5× → ~1.7× (1.5× on xor).** Figure:
`reports/2026-07-06-nagare-forward-opt-kato15.png`. Memory advantage unchanged
(~160×). All 45 tests green (bit-identical). Run-to-run variance (~15%, e.g.
fused 9.1–11.6 across runs) is far smaller than the effect.

## Remaining levers (not yet done)

1. **Serial pools (~33% now).** `global_pool` is single-threaded with a strided
   double pass; parallelize over batch (rayon) + fuse mean/var into one pass
   (sum + sumsq). Biggest remaining share. (The holonomy learner's
   `pooling::structural_pool_features` has the same serial shape — same fix
   applies and would speed the ablation forward.)
2. **Explicit SIMD** (`std::simd` / `wide`) beyond autovectorization for the
   fused kernel.
3. **Tuned micro-GEMM** (`matrixmultiply`/`gemm` crate) — likely matches MKL,
   but adds a dependency (CORE-gated) and dilutes the pure-Rust story.

## Honest verdict

The gap is **not** MKL magic — it was a naive kernel, and one bit-identical
reorder closed most of it (2.5×→1.7×). Chasing the last ~1.7× toward MKL parity
is a diminishing-returns fight (SIMD/GEMM-crate); worth it only if forward
latency becomes the binding constraint. Nagare's standing edge remains **memory
(~160×) and deploy footprint**, now with a **materially smaller speed deficit**.

## Files touched (local)

- `nagare_github/src/ops/{fused_entropy_update,linear}.rs` — ikj reorder
  (bit-identical); `+ saxpy` helper. Mirrored to `hymeko_nagare/src/ops/` for
  the parity remeasure (keeps the two trees' shared kernels in sync).
- `nagare_github/examples/forward_profile.rs` — new per-stage profiler.
- No CORE.YAML items, no new dependency.
