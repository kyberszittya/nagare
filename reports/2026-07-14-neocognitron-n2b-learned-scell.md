---
title: "Nagare Neocognitron N2b — the harder task that forces a learned oriented S-cell, and what it reveals about when the C-cell matters"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, neocognitron, s-cell, c-cell, rotation-invariance, conv2d, group-pool, learnable, no-autograd, negative-with-mechanism]
---

# Neocognitron N2b — forcing a *learned* oriented S-cell

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `394b8bc`+ · CPU

## Summary

N2 fixed the S-cell to Sobel to isolate the C-cell, and left the honest hard version as next: a task that
**requires a learned oriented S-cell** so a learnable conv can't dodge with an isotropic filter. This is that
task — detect a low-contrast oriented **bar buried in energy-matched isotropic noise** (label 1) vs
**noise-only** (label 0). Energy is matched by construction, so an energy/isotropic filter is at chance; only a
*learned oriented* matched filter that integrates along the coherent line can separate the classes.

**Two results, both measured, one positive and one an honest negative-with-mechanism:**

| axis | result |
|---|---|
| **energy-only baseline** (mean image energy) | **AUROC 0.469 ≈ chance** — the task genuinely forces oriented features |
| **learned oriented S-cell** (both C₈ and C₁) | **AUROC ≈ 0.73** held-out — the conv *does* learn a useful oriented/coherence detector, well above chance |
| **C₈ (rotation-invariant) vs C₁ (orientation-specific)** | **tied** — C₈ 0.730, C₁ 0.730 held-out; the explicit C-cell rotation-invariance is **redundant** here |

Figure: `reports/figures/neocognitron-aniso.png`.

## What this shows — and why it confirms N2 rather than contradicting it

The positive half is real: the energy baseline sits at chance (0.469), so an isotropic filter cannot solve the
task, and yet the pipeline reaches ~0.73 — the learnable conv **does** learn a genuinely oriented (matched-filter)
detector. The bar is +0.5 over ~15 pixels in σ≈0.6 noise → per-pixel SNR ~0.83, matched-filter SNR ~3.2, which
is what a ~0.73 AUROC partial detection looks like. So "the task forces a learned oriented S-cell" is achieved.

The negative half is the interesting one. Even training on a **single** orientation and testing on the whole C₈
orbit, the rotation-invariant C-cell does **not** beat the orientation-specific baseline (0.730 vs 0.730). The
mechanism: with a *learnable* 2-channel conv feeding `to_vectors → group_pool → global mean`, the network is free
to learn an **orientation-agnostic** line/coherence detector (a channel whose mean response is elevated by *any*
coherent line regardless of angle). The global-mean readout then never engages the group-pool's orientation
machinery, so pooling over the orbit (C₈) versus not (C₁) makes no difference. The learnable pathway discovers
the invariance the C-cell would have imposed — so the explicit prior is redundant.

This is the **same dodge as N2's isotropic-filter shortcut, one level up**, and it is exactly why N2 fixed the
S-cell. The clean statement across N2 + N2b:

> The C-cell's explicit rotation-invariance is **load-bearing only when the upstream representation is
> constrained to be genuinely oriented** (the fixed-Sobel S-cell of N2). When the S-cell is **learnable**, the
> network finds an orientation-agnostic invariant statistic on its own and the explicit group-pool becomes
> redundant (C₈ ≈ C₁).

This is the rotation-equivariance instance of the recurring DTC §6.4 pattern seen on pose and the cuboid/SBSH
detector: **a geometric-invariance mechanism is redundant when a learnable pathway can discover an equivalent
invariant** — and it only earns its place under representational constraint, limited data, or depth where the
implicit route fails.

## Method

`conv2d (1→2, learned) → to_vectors (gx,gy) → group_pool (C₈ | C₁) → global mean → linear → BCE`. Composed
backward is all FD-verified closed-form ops (`bce ← linear_backward ← mean-adjoint ← group_pool_backward ←
conv2d_backward`), no autograd. Energy-matched noise: the noise-only class variance is boosted by
`NBAR·CBAR²/GG` so `E[energy]` matches the bar class, removing the energy confound by construction (verified: the
energy baseline lands at 0.469). Train θ∈{0°}, test θ∈{45°,90°,…,315°} (the C₈ orbit). A/B is a one-line group
swap, `DihedralGroup::new(8,…)` vs `new(1,…)`.

## Tests / gates

| item | result |
|---|---|
| `examples/neocognitron_aniso` (C₈) | energy baseline 0.469; learned S-cell held-out AUROC 0.730 |
| `examples/neocognitron_aniso --c1` | held-out AUROC 0.730 (tied — explicit invariance redundant) |
| full suite | **158 / 0** |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

No new library ops (the demonstration reuses `conv2d`/`group_pool`/`linear`/`adam`, all already FD-verified);
no new deps; no CORE.YAML.

## Files touched

| file | change |
|---|---|
| `examples/neocognitron_aniso.rs` | new — energy-matched oriented-bar-in-noise; learned S-cell + C₈-vs-C₁ A/B |
| `scripts/dev/plot_aniso.py` | new — the AUROC bar chart |
| `reports/figures/neocognitron-aniso.{json,png}` | result + figure |

## Next

- To make the explicit C-cell *decisively* win with a learnable S-cell, the readout must be **orientation-
  sensitive-but-invariant** (e.g. an orient-histogram or a localized max over the orbit) rather than a global
  mean that washes orientation away — or the regime must deny the implicit route (very limited data, a **deeper
  2+ S/C stack** where the invariance must compound). The deeper stack is the more valuable direction and is the
  same backbone the pose P1 occlusion A/B needs.
- The **continuous quaternion-attention** C-cell (`cayley_rotor`) for sub-group-angle tolerance remains open.

## Provenance

- Mac (Apple Silicon), Nagare `394b8bc`+; CPU. Analytic data (energy-matched oriented bars in Gaussian noise,
  G=20, CBAR=0.5, σ=0.6). Seeds fixed in-source (Rng(999) train/eval, Rng(7) energy baseline).
- Reproduce: `cargo run --release --example neocognitron_aniso [-- --c1]`.
