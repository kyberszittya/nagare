---
title: "Evolvent E9 — wall-clock validation: the multifrontal O(d·w³) win is real and growing (5.9× → 255×), ~5× below the analytic flop count"
date: 2026-07-16
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, junction-tree, benchmark, criterion, validation, correction]
---

# Evolvent E9 — closing the biggest standing caveat: measure the time

Date: 2026-07-16 · Nagare `acd3797`+ · Mac (arm64, CPU) · **criterion 0.5 (§10)**

## Summary

Every speedup claimed in E5–E8 was an **analytic flop count** (`Σ|C|³` vs `d³/6`), with each report carrying the
caveat "not a wall-clock benchmark." E9 closes it with `criterion` (the §10-pinned tool, already a dev-dependency):
the two real solvers, `JunctionTreeCholesky::solve` (multifrontal) vs `InfoEvolventHead::solve` (dense Gaussian) —
which give the identical answer (E5) — on the E5 branching tree, depths 4–7:

| d | multifrontal | dense Gauss | **measured speedup** | analytic flops | flop / measured |
|---|---|---|---|---|---|
| 105 | 10.0 µs | 59.0 µs | **5.9×** | 18× | 3.1× |
| 217 | 21.0 µs | 344 µs | **16.4×** | 77× | 4.7× |
| 441 | 43.6 µs | 2.75 ms | **63.0×** | 314× | 5.0× |
| 889 | 85.1 µs | 21.8 ms | **255×** | 1270× | 5.0× |

Figure: `reports/figures/evolvent-wallclock.png`.

## What is measured

- **The win is real, and it grows exactly as the asymptotic predicts.** Wall-clock speedup rises 5.9× → 16.4× →
  63× → 255× over `d` 105 → 889 — the same `(d/w)²` growth as the flop count. At `d=889` the multifrontal solve is
  **85 µs vs 21.8 ms** — a 255× real speedup, and 85 µs is genuinely online-viable. The `O(d·w³)` vs `O(d³)`
  claim survives contact with the clock.
- **But the analytic flop count OVERSTATED the deployable speedup by ~5×.** The `flop / measured` ratio stabilizes
  at **5.0×** for larger `d`. Two honest reasons: (1) the dense baseline is a **contiguous, cache-friendly** single
  `d×d` array, while the multifrontal chases **many small per-clique `Vec`s** (allocation + pointer-chase overhead
  per `solve`); (2) the multifrontal re-allocates its frontals and factors on every `solve` call. So 1270× (the E5
  headline) is the **flop ceiling**, not the measured speedup — the measured number is 255×.
- **The gap is the contiguous-storage rewrite's headroom.** The ~5× is implementation constant, not algorithm: a
  contiguous-storage multifrontal (one arena, no per-clique `Vec`) would recover much of it. The flop count is the
  ceiling that rewrite approaches — E5's flagged follow-up now has a measured target (~5× on the table).

## Correction to the record

The E5/E6/E7 reports and the `JunctionTreeCholesky` registry note quoted the flop speedup (18× → 1270×) as the
solver's advantage. That is the **flop** advantage; the **measured wall-clock** advantage is **5.9× → 255×**
(criterion, this report). F-EVO-7's note and the component are updated to cite both, with the measured number as
the deployable one. The *direction and growth* of F-EVO-7 stand; the *magnitude* is corrected downward ~5×.

## Honest scope

- **Dense baseline is Gaussian elimination** (`InfoEvolventHead::solve`), ~2× the flops of a pure dense Cholesky.
  So the measured ratio is a **mild over-estimate** of the pure-algorithmic win (a dense-Cholesky baseline would
  narrow it by up to ~2×). It is, however, the honest comparison of the two solvers that actually exist in the
  crate and are used in the arc.
- **Single machine, single run of criterion** (30 samples, 3 s warm-up, quiet Mac). Criterion's CIs are tight
  (medians within ~1–2 % of the reported point), so the ratios are stable; absolute µs are machine-specific.
- **Re-factorize-from-scratch** each `solve` (both solvers). The online-incremental path (`refactorize_path`, still
  unbuilt) would change the per-update cost, not this per-solve comparison.

## Tests / gates

| item | result |
|---|---|
| `benches/evolvent_solve_bench` (criterion, 4 sizes × 2 solvers) | table above |
| full suite | **179 / 0** · fmt + clippy clean (incl. bench target) |

## Files touched

| file | change |
|---|---|
| `benches/evolvent_solve_bench.rs` | new — criterion multifrontal-vs-dense solve benchmark |
| `Cargo.toml` | register `[[bench]] evolvent_solve_bench` (criterion already a dev-dep; no new dependency) |
| `scripts/dev/plot_evolvent_wallclock.py`, `reports/figures/evolvent-wallclock.png`, `reports/figures/evolvent_e9_results.json` | figure + data |
| `reports/framework/canonical_{components,findings}.json` | F-EVO-7 note + `JunctionTreeCholesky` note: add measured wall-clock, correct the magnitude |

## Provenance

- Nagare on Hajdus-MacBook-Pro (arm64, CPU-only), `cargo 1.96.1`, `criterion 0.5` (30 samples). Branching binary
  clique tree (`balanced_binary_tree(depth, 2, 3)`), 20 measurements/clique primed outside the timed region; solve
  re-run under criterion. Reproduce: `cargo bench --bench evolvent_solve_bench`.
- **kato15 still not synced** (working on the Mac); origin `main` ahead — kato15 fast-forwards on next pull.

## Next

- **Contiguous-storage multifrontal** (one arena, no per-clique `Vec`) → close the ~5× flop-to-measured gap; the
  flop count is the ceiling.
- A dense-**Cholesky** baseline (not Gaussian) for the pure-algorithmic ratio.
- The remaining science item: higher-order interactions (degree ≥ 3 → width ≥ order+1).
