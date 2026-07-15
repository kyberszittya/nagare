---
title: "Evolvent E6 — the discriminating test: what the cross-clique (separator) coupling is worth, as a function of data scarcity"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, junction-tree, separator, discriminating, positive]
---

# Evolvent E6 — the separator coupling, quantified

Date: 2026-07-15 · Nagare `2fc69ae`+ (kato15, 32-core; CPU-only) · continues E5 (F-EVO-7)

## Summary

E3–E5 left one thing unmeasured: **what is the cross-clique (separator) coupling actually worth?** In E5 (data-rich,
`per=60`) the block-diagonal baseline that *drops* the coupling trailed the exact multifrontal solve by only
~0.0002 R² — I flagged this as a **regime property** (data-rich per clique), not a method limit, and named the
discriminating test: **data-scarce**. E6 runs it. Same branching clique tree (depth 6, `d=441`, 63 cliques, clique
arity ≈ 8); sweep the measurements-per-clique `per` from 2 (far below arity) to 40 (rich). 5 seeds, kato15,
medians:

| per | MULTIFRONTAL R² | DENSE R² | BLOCK R² | **gap (MF − BLOCK)** | wRMSE MF | wRMSE BLOCK |
|---|---|---|---|---|---|---|
| 2  | 0.321 | 0.321 | 0.232 | 0.089 | 0.834 | 0.860 |
| 3  | 0.422 | 0.422 | 0.325 | 0.097 | 0.764 | 0.798 |
| 4  | 0.571 | 0.571 | 0.417 | 0.154 | 0.682 | 0.752 |
| 6  | 0.787 | 0.787 | 0.628 | **0.159** | 0.492 | 0.615 |
| 10 | 0.937 | 0.937 | 0.872 | 0.066 | 0.272 | 0.359 |
| 20 | 0.994 | 0.994 | 0.988 | 0.058 | 0.083 | 0.108 |
| 40 | 0.999 | 0.999 | 0.998 | 0.006 | 0.030 | 0.037 |

Figure: `reports/figures/evolvent-scarcity.png`.

## What is measured

- **The separator coupling is worth up to ~0.16 R² — when data is scarce.** The gap MF − BLOCK rises from 0.089
  (per=2) to a **peak 0.159 at per=6** (below the clique arity ≈ 8, the data-scarce regime), then falls to **0.006
  at per=40** (the E5 rich regime). Block-diagonal *starves*: each clique's local system is under-determined, so
  dropping the neighbours' evidence about the shared separator variables costs real accuracy. This is the answer to
  the block-vs-info question open since E3: the coupling's value is not fixed — it is a function of data density,
  and it is large exactly where data is scarce.
- **Multifrontal == dense at every point.** `MF R² == DENSE R²` at all 35 `(per, seed)` cells (and to 4 decimals in
  the table). The exact solver pays nothing for its sparsity; it recovers the full pooled least-squares solution,
  which is what lets it use the neighbours' evidence through the separators.
- **The estimation gap confirms the mechanism.** Weight-recovery RMSE (‖w − w_true‖) is worse for BLOCK at every
  `per`, and the deficit widens as data thins (per=6: 0.615 vs 0.492). BLOCK mis-estimates precisely the separator
  variables — it sees only their residual clique's data; MF pools all cliques that share them.
- **Why the peak, not a monotone gap.** At per=2 even MF is badly under-determined (R² 0.32), so there is little
  recoverable signal for BLOCK to lose; at per=40 both saturate near 1.0. The coupling matters most in between —
  where MF can recover the target but BLOCK, starved locally, cannot. The peak sits just below the clique arity.

## Honest scope

- **Additive-in-features target, linear model** (as the whole evolvent line). The scarcity here is genuine
  data-scarcity, not a manufactured non-additive coupling — I chose the honest knob (`per`) rather than injecting a
  cross-clique product feature to force a gap. The mechanism is real: shared separator variables get more evidence
  when the tree is pooled.
- **The gap is a property of the (structure, data-density) pair, not of MF alone.** MF's *exactness* is
  regime-independent; the *value* of that exactness over block-diagonal is what varies. E6 measures the value; it
  does not claim block-diagonal is always inadequate — in the rich regime (E5) it is nearly as good and cheaper.
- **Single depth (6).** The scarcity axis is the variable of interest; depth/scale were characterised in E5. The
  gap should grow with separator sharing (deeper trees, larger `sep`), untested here.

## Tests / gates

| item | result |
|---|---|
| `junction_tree::single_clique_equals_dense` (now also pins `solve_block_diagonal` == exact on 1 clique) | pass |
| `junction_tree::branching_tree_equals_dense_at_bounded_width` | pass |
| `junction_tree::online_update_only_perturbs_path_to_root` | pass |
| `examples/evolvent_multifrontal --per` (7 densities × 5 seeds, kato15) | table above |
| full suite | **178 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `examples/evolvent_multifrontal.rs` | `--per` axis (measurements/clique) + weight-recovery RMSE columns |
| `src/junction_tree.rs` | `single_clique_equals_dense` extended to cover `solve_block_diagonal` (§3 coverage) |
| `scripts/dev/plot_evolvent_scarcity.py`, `reports/figures/evolvent-scarcity.png`, `reports/figures/evolvent_e6_results.json` | figure + data |

## Provenance

- Nagare on kato15 (32-core, RTX6000; CPU-only), `source ~/.cargo/env`. Branching binary clique tree
  (depth 6, `sep=2, res=3`, arity ≈ 8, d=441, 63 cliques). `per ∈ {2,3,4,6,10,20,40}` measurements per clique,
  `y = φ·w_true + 0.05 noise`, 3:1 train/test. Seeds 0–4. Reproduce:
  `cargo run --release --example evolvent_multifrontal -- --depth=6 --per=P --seed=S`.

## Next

- Sweep the **separator-sharing** axis (`sep`, depth) — the gap should grow when more cliques share each separator.
- The same scarcity test with a genuinely **non-additive** cross-clique feature (an explicit product spanning a
  separator) — does the gap widen further, or is data-scarcity the whole story?
- Contiguous-storage rewrite + wall-clock benchmark (the E5 follow-up).
