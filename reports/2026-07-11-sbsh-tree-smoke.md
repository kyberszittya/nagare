---
title: "Nagare SBSH — proof-of-concept smoke: both hinges PASS (dynamic tree + canonical-aligned descriptor)"
date: 2026-07-11
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, sbsh, gomb-soma, object-detection, quadtree, phase-pool, hinge-test]
---

# SBSH proof-of-concept — the two hinges (handoff §8)

Date: 2026-07-11 · Mac · Nagare at `50c3142`+ · 5 seeds · **no detector, no training**

The handoff mandated proving two hinges *before* building any detector. Both were tested on a synthetic
scene (K=4 filled oriented rectangles on a flat 96×96 background, ground-truth oriented boxes).
`examples/sbsh_tree_smoke.rs`.

## H1 — the dynamic spatial tree concentrates cells on objects: **PASS (robust, 5/5)**

Quadtree split-by-gradient-energy (structural, no backward — the `cpml_tier` discipline). Metric: fraction
of leaf cells that lie on an object, vs a uniform grid / the object-area baseline.

| seed | adaptive on-object | uniform / obj-area | concentration ×obj-area | leaves |
|---|---|---|---|---|
| 0 | 0.368 | 0.110 | 3.3× | 220 |
| 1 | 0.382 | 0.121 | 3.2× | 220 |
| 2 | 0.324 | 0.091 | 3.6× | 247 |
| 3 | 0.355 | 0.071 | 5.0× | 172 |
| 4 | 0.442 | 0.095 | 4.7× | 163 |

Median **~3.6× concentration** over the object-area baseline, all 5 seeds, and on-object cells are finer
(≈3.3 px) than off-object (≈5.7 px). The visual (`reports/figures/sbsh-tree-smoke.png`) confirms it: fine
leaves hug the oriented rectangles' **edges** (where gradient energy lives), coarse leaves cover the flat
background. **The core novel mechanism — a per-image dynamic quadtree that spends resolution on content —
works.** (Note: a gradient-energy split hugs *boundaries*, leaving flat interiors coarse — good for
localisation, but see "next".)

## H2 — node/shape descriptor rotation-robustness: **initially FAILED, then FIXED (canonical alignment)**

**Initial (raw phase-pool `|DFT|`):** mean relative L2 drift over 8 rotations = **median 0.21** (4/5 seeds
weak, > 0.15). Not rotation-robust on these shapes.

**Diagnosis via discriminating tests (not a guess):**
1. *Is it a test artifact?* Bilinear-rotating a crop resamples sharp edges. Re-measured on **clean
   renders** (the same rect rendered fresh at each angle, no resampling): drift still **0.175** → a
   *genuine descriptor weakness*, not a resampling artifact.
2. *Aliasing hypothesis falsified.* Adding **bins** (18→36→72) and **circular support** made it **worse**
   (0.16→0.29), not better. So the first-guess fixes were wrong.
3. *Mechanism.* A geometric edge produces **near-delta orientation peaks**; `|DFT|` shift-invariance is
   exact only for continuous shifts, so a sub-bin rotation of a delta-peak aliases — and *finer* bins make
   delta peaks sharper, hence *worse*. The opposite of textures (smooth orientation → robust `|DFT|`).

**The fix — canonical-orientation alignment.** Estimate the object's dominant edge via the **2nd circular
moment** `θ₀ = ½·atan2(Σ m·sin2θ, Σ m·cos2θ)` (the long-edge direction of an elongated object) and
histogram `θ−θ₀` — align to the object's own frame instead of relying on `|DFT|` shift-invariance.

| descriptor | clean-render drift | bilinear drift (5-seed) | verdict |
|---|---|---|---|
| raw `|DFT|` b=18 | 0.175 | 0.16 – 0.28 | weak |
| **canon, b=18** | **0.090** | 0.10 – 0.20 | **ROBUST (clean)** |
| **canon, b=12** | — | **0.063 – 0.130 (median 0.075)** | **ROBUST 4/5** |

Canonical alignment nearly halves the clean-render drift (0.175 → **0.090**, below the 0.10 target) and
`b=12+canon` is robust on 4/5 seeds under bilinear rotation (the residual is pure resampling noise — real
same-object-different-orientation instances are not resampled copies, so the **clean-render 0.090 is the
relevant number**).

## Verdict — BOTH hinges pass; unblocked

- **H1 validated** (3.6× cell concentration) and **H2 resolved** (canonical-aligned descriptor, clean drift
  0.090). The SBSH mechanism is sound end-to-end at PoC scale → **Phase 1 (promote the quadtree to an FD-clean
  lib op) is unblocked.**
- **Honest caveats:** (a) rectangles / elongated objects only — the **2nd-moment canonical is ambiguous for
  near-square or rotationally-symmetric objects** (need higher moments, or a learned canonical estimator);
  (b) still synthetic; (c) the descriptor fix is *canonical alignment*, i.e. rotation-*equivariant-then-
  aligned*, not pure `|DFT|` invariance — a deliberate, principled change matching the geometric regime.

Methodology note (on record): the first-guess fixes (more bins, circular support) were **falsified** by
measurement, and the clean-render discriminating test separated *test artifact* from *real weakness* before
any conclusion — the same discipline that reversed the holonomy "robust" claim.

## Files touched

| file | change |
|---|---|
| `examples/sbsh_tree_smoke.rs` | synthetic oriented-scene gen + dynamic quadtree (H1) + descriptor sweep w/ canonical alignment (H2) + clean-render discriminating test + viz dump |
| `scripts/dev/render_sbsh_tree.py`, `reports/figures/sbsh-tree-smoke.png` | tree-overlay visualisation |

No new ops (reuses `rotate_image`; local `phase_desc` will become the lib op in Phase 2), no CORE.YAML,
no new deps. fmt + clippy clean.

## Next (both hinges passed → unblocked)

1. **Phase-1 §2 plan-bundle** → promote the dynamic quadtree to an FD-clean lib op (structural split stays
   backward-free like `cpml_tier`; the *node feature pool* gets the FD-verified backward).
2. **Phase-2** → the **canonical-aligned orientation descriptor** as a lib op (with backward), plus a
   general canonical estimator that degrades gracefully on near-square/symmetric objects (higher circular
   moments or a learned angle) — the one open weakness of the 2nd-moment fix.
3. **Phase-0 novelty search** (4-query) before any external claim.
4. Then the oriented-bbox head (surrogate KLD loss) + assignment; do **not** train a full detector until
   1–2 are FD-clean.

## Provenance

- Mac (Apple Silicon). Synthetic scenes (seeded), no external data. 96×96, K=4 filled oriented rects,
  split thresh 0.05, max_depth 5, min_side 3; descriptor b=18, crop 40, 8 rotation angles.
- Reproduce: `cargo run --release --example sbsh_tree_smoke -- --seed <s>`;
  `uv run --with matplotlib --with numpy scripts/dev/render_sbsh_tree.py /tmp/sbsh`.
