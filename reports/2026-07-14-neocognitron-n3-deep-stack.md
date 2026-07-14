---
title: "Nagare Neocognitron N3 — the stackable ScBlock backbone, and a deep-stack A/B that separates local orientation-invariance from global rotation-invariance"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, neocognitron, sc-block, deep-stack, rotation-invariance, compositional, conv2d, group-pool, no-autograd, positive-with-limit]
---

# Neocognitron N3 — the deep S/C stack

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `ebdf194`+ · CPU

## Summary

Two deliverables. (1) **`ScBlock`** — a new, FD-verified, stackable S/C block (`conv2d` S-cell → bank of `K`
oriented-unit `group_pool` C-cells → `K`-channel rotation-invariant response map), the joint-discriminative
backbone the pose P1 report asked for. Its output is exactly the input shape of the next block, so blocks stack.
(2) A **2-block deep-stack A/B** on a compositional, rotation-varied task — **corner (L-junction) vs
length-matched straight bar** — with two knobs: depth (1 vs 2 blocks) and C-cell group (C₈ vs C₁). Five seeds.

| arm | train AUROC (median) | held-out-rotation AUROC (median, [min,max]) |
|---|---|---|
| **2-block · C₈** | **1.000** | 0.517  [0.438, 0.610] |
| **2-block · C₁** | **0.549 ≈ chance** | 0.498  [0.461, 0.531] |
| **1-block · C₈** | 1.000 | 0.462  [0.449, 0.503] |

Figure: `reports/figures/neocognitron-deep.png`.

## Two findings, both robust across 5 seeds

**1 — The C-cell is decisively load-bearing for *fitting* the compositional task (positive).** C₈ fits perfectly
(train 1.000, every seed); **C₁ cannot fit at all** (train ~0.39–0.58 ≈ chance, every seed). A corner is two edges
at 90°; distinguishing it from a length-matched bar requires an orientation-invariant edge response that C₁ (no
orbit) cannot build across the two arm orientations, while C₈ pools the oriented response over the group and reads
it. This is the clean win the N2/N2b arc was probing for — and, importantly, **the compositional/deep regime
defeats the N2b implicit-invariance dodge**: in N2b a single learnable conv + global-mean readout found an
orientation-agnostic shortcut so C₈ ≈ C₁; here the compositional corner-vs-bar structure leaves no such shortcut,
so the explicit C-cell becomes necessary. That is exactly the "deeper stack denies the implicit route" hypothesis
N2b left open, now confirmed.

**2 — Local orientation-invariance is *not* global rotation-invariance (the honest limit).** Neither depth
achieves held-out-rotation generalization: 2-block C₈ median 0.517 and 1-block C₈ median 0.462 both hug chance,
with per-seed ranges that overlap 0.5. The stack fits the *trained* orientations perfectly (1.000) but does not
transfer to unseen global rotations. The mechanism is precise and worth stating carefully: the C-cell confers
**per-feature, per-location orientation-invariance** — a rotated *edge* still fires the same channel — but the
**global spatial configuration** of the corner still rotates, and block-2's spatial `conv` is not itself globally
rotation-invariant, so a corner rotated to an unseen angle presents a spatial vertex pattern the conv has not
seen. Two invariances that are easy to conflate are cleanly separated here: *local orientation-invariance of
features* (which `group_pool` provides) vs *global rotation-invariance of a configuration* (which it does not).

## Why this is the informative outcome

A plain "C₈ generalizes across rotations" would have hidden the distinction. The measured result instead pins the
gap to a specific missing primitive: a **globally rotation-invariant top** — a spatial-orbit pool over the final
response map, or a quaternion-attention pose canonicalization (`cayley_rotor`) that aligns the global
configuration before the spatial readout. That is the concrete next step, and it is now motivated by evidence
rather than asserted.

## The backbone (`ScBlock`), verified

`x (C_in,H,W) → conv2d (C_in→2K) → K oriented units (gx,gy) → group_pool per unit → resp (K,H,W)`. The composed
backward is all closed-form FD-verified ops (`group_pool_backward` per unit → `conv2d_backward`), no autograd.
FD tests: the single-block backward (input, conv weights, conv bias, pool filters) **and a two-block stacked
backward** (grad through block-2 then block-1) both match finite differences.

## Tests / gates

| item | result |
|---|---|
| `sc_block::backward_matches_fd` (input, conv.w, conv.b, filt) | pass |
| `sc_block::two_block_stack_backward_matches_fd` | pass (composed grad through 2 blocks = FD) |
| `sc_block::c1_group_is_orientation_specific` | pass |
| `examples/neocognitron_deep` (3 arms × 5 seeds) | table above |
| full suite | **161 / 0** |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Files touched

| file | change |
|---|---|
| `src/ops/sc_block.rs` | new — the stackable S/C block + FD-verified composed backward (3 tests) |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export `ScBlock`, `sc_block_forward/backward`, grads |
| `examples/neocognitron_deep.rs` | new — corner-vs-bar deep-stack A/B (depth × group, seed knob) |
| `scripts/dev/plot_deep.py` | new — median ± spread figure |
| `reports/figures/neocognitron-deep.{png}`, `reports/figures/deep_seeds/*.json` | figure + 5-seed results |

No new deps; no CORE.YAML.

## The Neocognitron so far

- **S-cell** `conv2d` (N0) · **C-cell** `group_pool` (N1) · **S/C stack** with a fixed-Sobel isolation A/B (N2).
- **N2b** — a *learnable* oriented S-cell forced by an energy-matched task; C₈ ≈ C₁ because the learnable conv
  finds the invariance implicitly (explicit prior redundant under a learnable single block).
- **N3 (this)** — the stackable `ScBlock` backbone; in the **compositional deep regime the C-cell becomes
  necessary** (C₁ can't fit), and the residual held-out gap isolates *global* rotation-invariance as the next
  primitive.

## Next

- A **globally rotation-invariant top** (spatial-orbit pool over the final resp map, or a `cayley_rotor`
  quaternion-attention pose canonicalization) — the primitive the held-out gap points to; the discriminating test
  is whether it lifts held-out-rotation AUROC off chance while keeping train at 1.000.
- Feed the `ScBlock` backbone into the SBSH detector / pose `soft_argmax` head (the original P1 unblock).

## Provenance

- Mac (Apple Silicon), Nagare `ebdf194`+; CPU. Analytic data (corner/bar strokes, G=24, ARM=7, light noise).
  5 seeds via `--seed=N` (block/head init + data draw). Train θ∈{0°,90°}, test θ∈{45°,135°,22.5°,67.5°,112.5°,
  157.5°,200°,250°}.
- Reproduce: `cargo run --release --example neocognitron_deep -- [--onelayer] [--c1] [--seed=N]`.
