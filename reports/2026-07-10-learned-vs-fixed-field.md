---
title: "Nagare CV — learned vs fixed orientation field under the |DFT| invariant"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, computer-vision, phase-pool, backprop, rotation-invariance, negative-result]
---

# Learned vs fixed orientation field under the same `|DFT|` invariant

Date: 2026-07-10 · kato15 (Katolab), Nagare at `99a23cb`, CPU · 5 seeds

## Question

Now that `phase_pool` is differentiable, does a **learned** per-pixel orientation field beat the
**fixed** central-difference gradient *under the same rotation invariant*? This is the test that
decides whether the CV arc is a "clever fixed descriptor" or a "learned invariant representation."

## Design (fair by construction)

The central difference *is* a frozen 3×3 conv (`gx = right−left`, `gy = down−up`). Both arms share the
**identical** pipeline `image → im2col 3×3 → linear(9→2) → field → phase_pool(|DFT|) → linear head`,
differing **only** in the 9→2 kernel:

- **fixed** — kernel frozen to the central difference; train the head only.
- **scratch** — kernel random-init, trained jointly through the pool (learnability from scratch).
- **warmstart** — kernel **init AT the central difference**, then trained jointly. *Decisive arm:* if
  it stays ≈ fixed, central-diff is a local optimum under the pool; if it climbs, learning wins; if it
  drops, the pool gradient drifts away from the good field.

Zero new ops (reuses `phase_pool` + `linear` + `softmax`, all FD-verified). **Plumbing gate passed**:
the frozen field's feature equals `spatial_phase_features(r=1)` to `max gap 0.00e0`, and the fixed arm
reproduces cv_bench's phase-pool R=1 (KTH 0.505 vs 0.492; MNIST 0.42 vs 0.416). Each arm standardises
with its own train-feature stats (per-epoch detached BatchNorm-lite for the learned arms); grad-clip on
the field guards the pool's `1/m²`.

## Results — median [IQR] over 5 seeds

**KTH-TIPS2-b (11 materials, chance 0.091):**

| arm | upright | rotated |
|---|---|---|
| **fixed** | **0.5051** [0.5034, 0.5059] | **0.4116** [0.4057, 0.4141] |
| scratch | 0.3990 [0.3889, 0.4520] | 0.3779 [0.3182, 0.3847] |
| warmstart | 0.4032 [0.3923, 0.4293] | 0.3300 [0.2971, 0.3830] |

Δ(scratch−fixed): up −0.106, ro −0.034 · Δ(warmstart−fixed): up **−0.102**, ro −0.082

**MNIST (10 digits, chance 0.10):**

| arm | upright | rotated |
|---|---|---|
| **fixed** | 0.4205 [0.4200, 0.4210] | **0.2950** [0.2945, 0.2960] |
| scratch | **0.4490** [0.4150, 0.4530] | 0.2160 [0.2025, 0.2700] |
| warmstart | 0.4230 [0.4110, 0.4400] | 0.2700 [0.2525, 0.2940] |

Δ(scratch−fixed): up +0.029, ro **−0.079** · Δ(warmstart−fixed): up +0.003, ro −0.025

Figure: `reports/figures/cv-learned-vs-fixed.png`.

## Verdict — a learned field does NOT beat the fixed central difference

- **Warmstart never beats fixed.** On **KTH-TIPS2 it *degrades* from the warm start** (0.403 vs 0.505,
  IQRs disjoint) — starting *at* the central-diff optimum, gradient training through the pool drifts the
  kernel *away* and *down*. On **MNIST it ties** (0.423 vs 0.4205, IQRs overlap). It improves neither.
- **Scratch** underperforms on textures (0.399 vs 0.505) and, on digits, buys a hair of upright accuracy
  (+0.029) by **trading away rotation invariance** (rotated 0.216 vs 0.295, −0.079) — the H3 "learned
  field becomes less equivariant" mode, measured.
- So under the `|DFT|` invariant, **the hand-designed central difference is at or near the optimum**; a
  learned per-pixel field cannot improve on it and, on textures, is actively degraded by the pool
  gradient.

## Reading (measured / inferred / caveat)

- **Measured.** Warmstart ≤ fixed on both datasets; the KTH degradation-from-warm-start and the MNIST
  scratch invariance-trade are large-margin (IQR-separated on the load-bearing cells).
- **Inferred mechanism.** The phase-pool's discriminative power lives in the **invariant** (`|DFT|` of
  the orientation histogram), not in a learnable field extractor. The central difference already
  produces the right orientation field for this invariant; the pool gradient (lossy — `|DFT|` discards
  phase, plus the `1/m²`/soft-bin kinks) is a **poor teacher** that, from a good init, moves toward
  lower train-CE fields that generalise worse (warmstart train CE fell to ~1.5 while test accuracy fell).
- **Caveat / threat to validity.** This is "learning does not beat central-diff **under matched, standard
  training**" — not "impossible". The learned arm used one lr schedule, detached-BatchNorm standardisation,
  and field-gradient clipping; I did not exhaustively tune it. The robust part of the claim is the
  *warmstart-from-optimum degrades/ties* observation, which does not depend on undertraining (it started
  good).

## Implication for the arc (and for fiber-rotor-spike)

The lever for improving the CV arc is the **invariant / pool design** — multi-scale pooling, the spatial
phase-map `R` knob (already shown domain-tunable), richer invariants — **not** a learned per-pixel front
end. This also **bounds the fiber-rotor-spike direction**: a learned spike front-end feeding *this* pool
would not help; a rotor-spike contribution must **enrich the invariant itself**, not the field under it.

## Files touched

| file | change |
|---|---|
| `examples/cv_learned_field.rs` | new — 3-arm experiment (fixed/scratch/warmstart), plumbing gate, multi-seed |
| `src/cv_data.rs` | new lib module — Split/load_idx/load_raw/load_split/rot_all + feature_stats/standardize_with (pays cv_live loader-dup debt, §6.1) |
| `examples/cv_bench.rs`, `examples/cv_live.rs` | rewired onto the lib loader |
| `scripts/dev/plot_learned_vs_fixed.py`, `reports/figures/cv-learned-vs-fixed.png` | result figure |

No new ops, no CORE.YAML (repo has none), no new deps.

## Provenance

- kato15 (Katolab), Nagare `99a23cb`, CPU. Data: `~/nagare_data/{kth_tips2,mnist}` (repo-external).
- Logs: `/tmp/lvf_kth.log`, `/tmp/lvf_mnist.log` (kato15). Seeds 0–4; epochs 400; b=18; head_lr 0.5,
  conv_lr 0.2, grad-clip 5.0. KTH n_train 3564 / n_test 1188; MNIST 8000 / 2000.
- Reproduce: `cargo run --release --example cv_learned_field -- --dataset {raw|mnist} --data <dir> --seeds 5 --epochs 400`.
- Plan bundle: `docs/plans/2026-07-10-learned-vs-fixed-field/` (tex/pdf/tikz/mmd, gitignored).
- Mac suite green (114); fmt + clippy `-D warnings` clean.
