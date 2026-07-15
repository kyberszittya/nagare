---
title: "Evolvent E7 — the separator-sharing axis: sharing matters, but the block deficit saturates with fan-out (multi-seed refutes monotone growth)"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, junction-tree, separator, star, partial-refutation]
---

# Evolvent E7 — how much does the *degree* of separator-sharing matter?

Date: 2026-07-15 · Nagare `d27b018`+ (kato15, 32-core; CPU-only) · continues E6 (F-EVO-8)

## Summary — and a correction

After E6 I stated the next probe would likely show the scarce-regime gap **widen further** with more separator-
sharing. **The multi-seed data does not support that.** E7 builds a **star** clique tree — one separator (3 vars)
shared across `fanout` children, so `fanout + 1` cliques share it — and sweeps `fanout` at fixed data-scarcity
(`per=4`, child arity 6). 5 seeds, kato15, gap = MF − BLOCK, median[IQR]:

| fan-out | cliques/sep | d | MF/DENSE R² | BLOCK R² | **gap MF−BLOCK** |
|---|---|---|---|---|---|
| 1  | 2  | 9   | 0.639 | 0.552 | 0.098 [0.02, 0.10] |
| 2  | 3  | 12  | 0.882 | 0.626 | 0.184 [0.16, 0.35] |
| 4  | 5  | 18  | 0.792 | 0.630 | 0.162 [0.11, 0.29] |
| 8  | 9  | 30  | 0.868 | 0.621 | 0.228 [0.18, 0.29] |
| 16 | 17 | 54  | 0.880 | 0.614 | 0.278 [0.22, 0.31] |
| 32 | 33 | 102 | 0.874 | 0.710 | 0.189 [0.16, 0.24] |

Figure: `reports/figures/evolvent-sharing.png`.

## What is measured

- **Sharing matters (confirmed).** Whenever a separator is shared across children, block-diagonal loses a large
  **~0.16–0.28 R²** — materially more than the minimal-sharing `fanout=1` case (0.10) and comparable-to-larger than
  E6's within-binary-tree gap at the same scarcity (~0.16). Block estimates the shared separator from the root's
  data alone; MF pools every child that touches it. Weight-recovery RMSE confirms block mis-estimates precisely the
  shared separator.
- **But it does NOT grow monotonically with fan-out (hypothesis refuted).** The gap **jumps** when the separator
  first becomes shared (fanout 1→2) and then **plateaus** at ~0.16–0.28 across fanout 2…32 — the IQRs overlap
  heavily and the point estimates are non-monotone (0.184, 0.162, 0.228, 0.278, 0.189). "More cliques per
  separator ⇒ monotonically bigger gap" is **not supported**.
- **MF == DENSE at every one of the 30 cells** — the exact solver is unaffected by the topology change; only the
  block approximation's deficit is at issue.

## Why it saturates (the mechanism, stated as inference)

At **fixed** data-scarcity, two things cap the gap:
1. Each child's **residual** variables (3, with only `per=4` measurements) are under-determined for *both* solvers,
   so both MF and BLOCK are capped below R²≈0.9 by residual estimation — MF's improving separator estimate can only
   recover the ~half of each child's variance that is separator-borne.
2. Block's **per-measurement** error on the shared separator is *set* once the separator is shared (fanout ≥ 2):
   adding more children adds more equally-separator-affected measurements, not more error *per* measurement. MF's
   separator estimate keeps improving (it pools `4·(fanout+1)` measurements), but the *aggregate test-R² gap* it
   buys is bounded by (1).

So the honest reading: **separator-sharing is a threshold effect, not a dose-response.** The coupling's value turns
on sharply when a separator is shared and is bounded by the per-clique data budget — it does not scale with the
number of sharers. This is a refinement of F-EVO-8 (the coupling is worth a lot when data is scarce), not a
contradiction of it.

## Honest scope

- **Small `d` at low fan-out** (`d=9` at fanout=1) makes those points noisy — one seed even had BLOCK *beat* MF on
  the held-out set (an over-regularization win at d=9), which is why the fanout=1 median is unreliable as the "low
  anchor." The plateau conclusion rests on fanout ≥ 2 (d ≥ 12), where the IQRs still overlap.
- **Additive-in-features linear target; single scarcity (`per=4`) and separator width (`sep=3`).** A genuinely
  non-additive cross-clique feature is still untested (the remaining open discriminating knob).
- This is a **partial refutation** reported as such: the single-seed run (seed 0) looked monotone (0.02 → 0.26);
  the 5-seed run does not. Recorded per the "don't force the convenient narrative / multi-seed is the verdict"
  discipline.

## Tests / gates

| item | result |
|---|---|
| `junction_tree::star_tree_equals_dense_with_shared_separator` | pass (MF == dense on a star; assembles many Schur messages into one parent) |
| `junction_tree::{single_clique,branching_tree,online_update}` | pass |
| `examples/evolvent_multifrontal --fanout` (6 fan-outs × 5 seeds, kato15) | table above |
| full suite | **179 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/junction_tree.rs` | new `star_clique_tree` (shared-separator topology) + 1 test |
| `src/lib.rs` | re-export `star_clique_tree` |
| `examples/evolvent_multifrontal.rs` | `--fanout` switch (star topology; fanout=0 ⇒ binary tree) |
| `scripts/dev/plot_evolvent_sharing.py`, `reports/figures/evolvent-sharing.png`, `reports/figures/evolvent_e7_results.json` | figure + data |

## Provenance

- Nagare on kato15 (32-core, RTX6000; CPU-only), `source ~/.cargo/env`. Star clique tree
  `star_clique_tree(fanout, sep=3, res_root=3, res_child=3)`, `per=4` measurements/clique (child arity 6),
  `y = φ·w_true + 0.05 noise`, 3:1 train/test. `fanout ∈ {1,2,4,8,16,32}`, seeds 0–4. Reproduce:
  `cargo run --release --example evolvent_multifrontal -- --fanout=F --per=4 --seed=S`.

## Next

- A genuinely **non-additive** cross-clique feature (an explicit product spanning a separator) — the last open
  discriminating knob; does it break the plateau?
- The scarcity × sharing **joint** sweep at larger `d` (to shed the small-`d` noise) if the question warrants it.
- The E5 engineering follow-ups (contiguous-storage rewrite + wall-clock; optimized `refactorize_path`).
