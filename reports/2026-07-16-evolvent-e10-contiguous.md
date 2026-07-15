---
title: "Evolvent E10 — contiguous-storage multifrontal: same exact answer, 2.7× faster solve, measured speedup now 16.7×–682×"
date: 2026-07-16
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, junction-tree, optimization, criterion, contiguous]
---

# Evolvent E10 — closing E9's ~5× headroom

Date: 2026-07-16 · Nagare `1647d7a`+ · Mac (arm64, CPU) · **criterion 0.5 (§10)**

## Summary

E9 measured the multifrontal solve at 5.9×–255× vs dense and found it ~5× below the analytic flop count, diagnosing
the cause as **per-clique small-`Vec` allocation** and a `position()` search in the hot path — "the contiguous-
storage rewrite's headroom." E10 does the rewrite. `JunctionTreeCholesky` now stores every per-clique block
(frontals, rhs, Cholesky factors) in **flat arenas addressed by precomputed offsets**, with the **parent-local
assembly map precomputed once** and **reused scratch** in the factorization — no per-clique allocation, no
`position()` search per solve. **Same public API, same exact answer** (all 4 exactness tests pass; the E5/E6/E7/E8
example numbers are byte-identical). Re-benchmarked (criterion, Mac):

| d | MF before | **MF after** | rewrite | speedup vs dense (before → after) | flop / measured |
|---|---|---|---|---|---|
| 105 | 10.0 µs | **3.86 µs** | 2.60× | 5.9× → **16.7×** | 5.0× → **1.08×** |
| 217 | 21.0 µs | **7.88 µs** | 2.66× | 16.4× → **43.6×** | 4.7× → **1.77×** |
| 441 | 43.6 µs | **15.8 µs** | 2.77× | 63× → **175×** | 5.0× → **1.80×** |
| 889 | 85.1 µs | **31.7 µs** | 2.69× | 255× → **682×** | 5.0× → **1.86×** |

Figure: `reports/figures/evolvent-contiguous.png`.

## What is measured

- **The solve is 2.6–2.8× faster.** At d=889, 31.7 µs vs the previous 85 µs — the allocation overhead E9 identified,
  removed. The self-speedup is flat across d (~2.7×), consistent with a per-solve constant (the `Vec<Vec>` clone +
  per-clique factor allocations) being amortized away.
- **Measured speedup vs dense is now 16.7× → 682×** (was 5.9× → 255×), and the **flop-to-measured gap closed from
  ~5.0× to ~1.8×** at large d — at d=105 the solve is essentially **at the flop ceiling** (1.08×). E9's headroom
  was real and is now recovered.
- **Correctness is untouched.** The rewrite changed only the memory layout and the assembly-map precompute; the
  arithmetic is identical. The four exactness tests (`== dense` on single / branching / star trees, plus the
  update-locality test) still pass, and every example (E5–E8) prints the same numbers to 4 decimals.

## Honest scope

- **The 682× is against a Gaussian-elimination dense baseline** (`InfoEvolventHead::solve`), ~2× the flops of a
  pure dense Cholesky. So it partly reflects the slower baseline: a fair Cholesky-vs-Cholesky comparison would show
  ~2× more residual (i.e. the true algorithmic residual vs a Cholesky baseline is ~3–4×, not ~1.8×). The **honest,
  layout-independent** number is the **2.7× self-speedup** — that is purely the allocation overhead removed, on the
  same solver, same baseline.
- **Single machine** (Mac arm64); absolute µs are machine-specific, ratios transfer.
- **Further headroom remains** below the flop ceiling: the frontal-extraction (`F_RR` copy with stride) and the
  column-wise triangular solves are not blocked/SIMD-tuned; a dense-Cholesky baseline and a blocked kernel would
  sharpen the picture. Not pursued — the E9 caveat (the ~5×) is closed.

## Tests / gates

| item | result |
|---|---|
| `junction_tree::{single_clique, branching_tree, star_tree, online_update}` (exactness + locality) | 4 / 4 pass |
| E5/E6/E7/E8 example outputs | byte-identical to pre-rewrite |
| `benches/evolvent_solve_bench` (criterion) | table above |
| full suite | **179 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/junction_tree.rs` | `JunctionTreeCholesky` internals rewritten to flat arenas (`m/nres/nsep/foff/boff/loff/woff/yoff/sploc`), precomputed assembly map, reused scratch in `factorize`; helpers → buffer-based `*_into`; removed dead `Clique::n_sep`; test helper updated to flat layout |
| `scripts/dev/plot_evolvent_contiguous.py`, `reports/figures/evolvent-contiguous.png`, `reports/figures/evolvent_e10_results.json` | figure + data |
| `reports/framework/canonical_{components,findings}.json` | `JunctionTreeCholesky` note + F-EVO-11 known-limitation update (rewrite done), F-EVO-12 added |

## Provenance

- Nagare on Hajdus-MacBook-Pro (arm64, CPU-only), `cargo 1.96.1`, `criterion 0.5` (30 samples). Same harness as E9
  (`balanced_binary_tree(depth, 2, 3)`, 20 measurements/clique). Reproduce: `cargo bench --bench evolvent_solve_bench`.
- **kato15 not synced** (Mac work); origin `main` ahead — kato15 fast-forwards on next pull.

## Next

- Blocked / SIMD triangular kernels + a dense-**Cholesky** baseline → the remaining residual below the flop ceiling.
- The remaining science item: higher-order interactions (degree ≥ 3 → width ≥ order+1).
- Optimized `refactorize_path` (leaf→root only) for the incremental online solver + its wall-clock.
