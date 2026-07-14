---
title: "SBSH Pose P3 — the task that exercises the skeleton: a redundant structural constraint the local backbone can't use but the skeleton hg_conv can (decisive win)"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, sbsh, pose, skeleton-hgconv, redundant-constraint, occlusion, symmetry, no-autograd, positive]
---

# SBSH Pose P3 — the skeleton conv finally pays

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `0990697`+ · CPU

## Summary

P2 showed the skeleton conv is neutral on a 2-link arm because the middle joint is over-constrained (no
structural-only gap). P3 supplies the missing ingredient — a **redundant structural constraint** — and the
skeleton conv **wins decisively** (5 seeds). Two **coupled** arms share the pose (`L = R + shoulder_offset`);
occluding a whole arm leaves it recoverable *only* from its twin, which the local conv head cannot exploit
(it can't couple left↔right) but the skeleton `hg_conv` can.

| condition | backbone only | +skeleton hg_conv |
|---|---|---|
| **left arm occluded** → left-hand err | 6.85 px | **2.61 px** (2.6× better) |
| **clean** → left-hand err | 24.0 px | **5.88 px** (4× better) |

Figure: `reports/figures/pose-symmetry.png`. Medians over 5 seeds; both effects robust (skeleton occluded
2.54–2.77 px across seeds).

## Two wins, one mechanism

**Occlusion recovery.** With the whole left arm masked, the backbone has no local evidence for the left hand and
lands ~6.9 px off. The skeleton's symmetry edge (`L_hand ↔ R_hand`, a `k=2` signed hyperedge) carries the visible
right hand across, and the residual signed `hg_conv` reconstructs the left hand to **2.6 px**. The recovery is a
translation, which this op *can* express: for the symmetry edge with signs `[+1,−1]`, `elin` learns `M ≈ −I/scale²`
and `bias ≈ offset/scale²`, so the occluded joint's garbage raw estimate **cancels** and it is set to
`R_hand + offset`. This is the structural-only recovery P2's over-constrained arm could not stage.

**Disambiguation.** Even *clean*, the backbone alone is poor (~24 px on a 32 px image) because the two arms are
appearance-identical — the local head cannot tell the left hand from the right and swaps them. The skeleton's
distinct connectivity (each hand tied to its own shoulder→elbow chain) breaks the symmetry and pins the correct
assignment (~5.9 px). The structural prior resolves an ambiguity the local representation fundamentally cannot.

## The arc's principle, now bracketed by evidence

Across the Neocognitron arc the same law held: **an explicit structural / geometric prior earns its place exactly
where a bottleneck denies the base mechanism the signal.**

| case | base mechanism | explicit prior | verdict |
|---|---|---|---|
| N2b (learnable S-cell) | learnable conv finds invariance | C-cell C₈ | **redundant** (implicit dodge) |
| N3 (compositional) | learnable conv can't fit | C-cell C₈ | **load-bearing** (fits vs chance) |
| entropy top | mean pool arrangement-blind | covariance eigen-entropy | **load-bearing** (held-out 1.0 vs chance) |
| P2 (2-link arm) | backbone recovers over-constrained joint | skeleton hg_conv | **neutral** (no gap) |
| **P3 (coupled arms)** | local head can't couple/disambiguate | skeleton hg_conv | **decisive win** (2.6 vs 6.9; 5.9 vs 24) |

P2 and P3 are the controlled pair that isolates *when* the skeleton helps: **not** when the joint is locally
recoverable (P2), **decisively** when a redundant long-range constraint exists that the local head cannot use
(P3). That is the honest, complete characterization the arc was after.

## Tests / gates

| item | result |
|---|---|
| `examples/pose_symmetry` (baseline + `--hg`, 5 seeds) | table above |
| full suite | **165 / 0** (reuses FD-verified `sc_block`/`conv2d`/`soft_argmax`/`hg_message`/`linear`) |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

Composes only already-FD-verified ops; no new library op, no new deps, no CORE.YAML.

## Files touched

| file | change |
|---|---|
| `examples/pose_symmetry.rs` | new — coupled-arm figure + symmetry-edge skeleton; whole-arm occlusion A/B |
| `scripts/dev/plot_pose_symmetry.py`, `reports/figures/pose-symmetry.png`, `reports/figures/ps*_*.json` | figure + 5-seed results |

## Next

- A **closed kinematic loop** (4-bar linkage) — the other redundant-constraint archetype; confirm the skeleton
  triangulates a loop-occluded joint the same way.
- **Learned** (not fixed-coupling) symmetry: arms mirrored rather than translated — needs an affine/reflection
  message the current signed hg_conv cannot express with one shared `elin`; a per-edge transform would be the op
  extension.
- Combine with the entropy-pool orientation readout for a full occlusion-robust pose+keypoint head.

## Provenance

- Mac (Apple Silicon), Nagare `0990697`+; CPU. Analytic data (two coupled 2-link arms, G=32, L₁=L₂=7, shoulders
  fixed at (11,9)/(21,9), random θ₁∈[−2.3,−0.8], θ₂∈[−1.0,1.0]). 5 seeds via `--seed=N`. Train 1800 poses, 60%
  random whole-arm occlusion; eval 80 fresh poses, left-arm occlusion.
- Reproduce: `cargo run --release --example pose_symmetry -- [--hg] [--seed=N]`.
