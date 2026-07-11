---
title: "Nagare SBSH — proof-of-concept smoke: dynamic tree PASSES, node descriptor invariance FAILS on geometric shapes"
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

## H2 — node/shape descriptor rotation-robustness: **FAIL on geometric shapes**

`spatial_phase_features` `|DFT|` (the phase-pool invariant) of a shape crop, mean relative L2 drift over
8 rotations:

| seed | 0 | 1 | 2 | 3 | 4 | median |
|---|---|---|---|---|---|---|
| drift | 0.161 | 0.211 | 0.104 | 0.275 | 0.280 | **0.211** |

4/5 seeds "weak" (> 0.15). The phase-pool descriptor is **not** rotation-robust on these shapes.

**Cause (diagnosed, not guessed):** a *filled rectangle* has gradients on only its four straight edges →
a **bimodal, sparse orientation histogram** (two dominant edge directions). `|DFT|` of a sparse two-peak
histogram **aliases badly** as the peaks cross bin boundaries under rotation; plus a crop-boundary artifact
(a fixed crop of an object that rotates partly out). This is the inverse of the CV arc, where phase-pool was
robust on *textures* (dense orientation distribution). **Rotation-invariance that held on textures does not
transfer to clean geometric objects.** Better to know now than after a detector.

## Verdict & what it bounds

- **H1 validated** → the dynamic-tree mechanism is sound; promoting it to a lib op (Phase 1) is justified.
- **H2 must be fixed before the detector** → the node descriptor needs rotation-robustness *on geometric
  objects*, not just textures. Candidate fixes to test next (cheap, all reuse existing ops):
  1. more bins + heavier soft-binning (reduce peak-aliasing of the sparse histogram);
  2. the **rotor-holonomy magnitude** `Re(H)` (found gauge-robust on graphs) as/with the descriptor;
  3. node-exact crops (not a fixed window) to kill the boundary artifact;
  4. accept geometric-shape invariance is inherently harder and add a small canonical-orientation
     alignment (steer to the dominant edge) before pooling.
- **Do not build the detector yet.** The handoff's own gate: both hinges must pass. H2 is the blocker.

Honest framing: this is 1 synthetic shape-type (filled rects), 5 seeds. The H1 result is solid; the H2
failure is a *design signal*, not a dead end — it names the exact next problem (geometric-object rotation
invariance) before any expensive build.

## Files touched

| file | change |
|---|---|
| `examples/sbsh_tree_smoke.rs` | new — synthetic oriented-scene gen + dynamic quadtree + H1/H2 tests + viz dump |
| `scripts/dev/render_sbsh_tree.py`, `reports/figures/sbsh-tree-smoke.png` | tree-overlay visualisation |

No new ops (reuses `spatial_phase_features`, `rotate_image`), no CORE.YAML, no new deps. fmt + clippy clean.

## Next

1. **H2 fix sprint** — test the four candidate fixes above on the same 5-seed smoke; target median drift
   < 0.10 (the "robust" line) on geometric shapes. This is the gate.
2. Only when H2 passes: Phase-1 §2-plan-bundle → promote the quadtree to a FD-clean lib op (feature pool
   with backward; the split stays structural).
3. Phase-0 novelty search (4-query) before any external claim.

## Provenance

- Mac (Apple Silicon). Synthetic scenes (seeded), no external data. 96×96, K=4 filled oriented rects,
  split thresh 0.05, max_depth 5, min_side 3; descriptor b=18, crop 40, 8 rotation angles.
- Reproduce: `cargo run --release --example sbsh_tree_smoke -- --seed <s>`;
  `uv run --with matplotlib --with numpy scripts/dev/render_sbsh_tree.py /tmp/sbsh`.
