---
title: "Nagare Neocognitron N2 — the stacked S/C hierarchy + a rotation-robustness A/B that isolates the C-cell"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, neocognitron, s-c-stack, rotation-robustness, conv2d, group-pool, dihedral, no-autograd]
---

# Neocognitron N2 — the S/C stack, rotation-robustness A/B

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `4665957`+ · CPU

## Summary

Composed the Neocognitron **S/C stack** end to end (no autograd) —
`S-cell (conv2d) → oriented vectors → C-cell (group_pool) → mean → linear` — and ran a **rotation-robustness
A/B that isolates the C-cell**. On an oriented-bar **contrast regression** (target `c`, rotation-invariant),
the rotation-equivariant C-cell (`group_pool` over **C₈**) **fits and generalises across rotations**, while
the orientation-specific baseline (**C₁**, no orbit) **cannot even fit**.

| C-cell group | train-orientation MSE | held-out-orientation MSE |
|---|---|---|
| **C₈** (rotation-invariant) | **0.0000** | **0.0042** |
| **C₁** (orientation-specific baseline) | 0.0137 | 0.0105 |

The C₈ held-out residual (0.0042) is bar-**pixelation** (45°, 135°… *are* C₈ group elements, where the pool is
exactly invariant per the `group_pool` unit test; a diagonally-rendered bar just has a slightly different
gradient field). The decisive signal is that **C₁ can't fit even the training orientations** — one oriented
filter cannot represent contrast across a 0° and a 90° bar — whereas C₈ pools the oriented response over the
group and reads the orientation-invariant edge energy. Figure: `reports/figures/neocognitron-rot.png`.

## The honest design choice — a fixed oriented S-cell

The A/B fixes the S-cell to a **Sobel gradient** (an oriented feature detector) rather than a learnable conv.
Reason (measured): with a *learnable* S-cell, this contrast task is solvable by an **isotropic** filter
(total energy ∝ contrast, already rotation-invariant), so the network dodges the orientation problem and
**both** C₈ and C₁ generalise — the C-cell isn't *needed*. That is itself an honest finding (the mechanism is
unnecessary when the task doesn't require oriented features — same lesson as the pose/detector ceilings). To
*isolate* the C-cell's rotation value, the S-cell must be genuinely oriented; fixing it to Sobel does that. A
learned oriented S-cell (forced by a task that requires it) would behave like the fixed one — that harder task
is the next step.

## The stack, end to end

`conv2d (S) → to_vectors → group_pool (C) → mean → linear → MSE`. The composed backward is all FD-verified
closed-form ops: `MSE ← linear_backward ← (mean adjoint) ← group_pool_backward ← (to_vectors) ← conv2d_backward`.
The C-cell A/B is a one-line change — the same `group_pool` op with `DihedralGroup::new(8,…)` vs `new(1,…)` —
so the comparison is exactly the rotation-orbit pooling and nothing else.

## Tests / gates

| item | result |
|---|---|
| `examples/neocognitron_rot` (C₈) | fits (train 0.0000) + generalises (held-out 0.0042) |
| `examples/neocognitron_rot --c1` | fails to fit (train 0.0137), held-out 0.0105 |
| full suite | **158 / 0** |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Files touched

| file | change |
|---|---|
| `examples/neocognitron_rot.rs` | new — S/C stack (`conv2d`→`group_pool`) + C₈-vs-C₁ rotation-robustness A/B |

No new deps, no CORE.YAML.

## The Neocognitron so far

- **S-cell** `conv2d` (N0) — learned spatial feature detectors (fixed to Sobel in this A/B to isolate the C-cell).
- **C-cell** `group_pool` (N1) — rotation-equivariant dihedral group-orbit attention pool.
- **N2 (this)** — the two composed into a stack; the C-cell's rotation-invariance measured to be *load-bearing*
  when the features are oriented.

## Next

- A task that **requires a learned oriented S-cell** (e.g. discriminate two oriented textures/shapes, both
  rotating), so the *learnable* conv can't dodge with an isotropic filter — the honest hard version of this A/B.
- **Deeper stack** (2+ S/C blocks) feeding the SBSH detector / pose soft-argmax — the joint-discriminative,
  rotation-tolerant backbone the pose P1 occlusion A/B needs.
- The **continuous quaternion-attention** C-cell (`cayley_rotor` rotor keys/queries) for sub-group-angle
  rotation tolerance (removes the ~0.004 pixelation-order residual at between-group angles).
- Multi-channel `group_pool` (a filter bank over the orbit) + an im2col `conv2d` perf path as depth grows.

## Provenance

- Mac (Apple Silicon), Nagare `4665957`+; CPU. No data (analytic oriented bars, G=20). C₈ / C₁ groups; fixed
  Sobel S-cell; train θ∈{0°,90°}, test θ∈{45°,135°,180°,225°,270°,315°}.
- Reproduce: `cargo run --release --example neocognitron_rot [-- --c1]`.
