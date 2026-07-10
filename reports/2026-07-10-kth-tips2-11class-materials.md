---
title: "Nagare CV — KTH-TIPS2-b (11-class materials), the hard texture bench"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, computer-vision, phase-pool, texture, kth-tips2, ablation]
---

# Nagare CV — KTH-TIPS2-b: phase-pool on 11-class materials

Date: 2026-07-10 15:54 JST · kato15 (Katolab), Nagare at `d5b06a8`, CPU

## Summary

Ran `examples/cv_bench` on **KTH-TIPS2-b** (11 materials, 3564 train / 1188 test, 64×64,
75/25 split) — the hardest texture bench in the arc so far, ~4× the images and one more class than
KTH-TIPS v1. Two training regimes (upright, rotation-augmented) × seven arms. Chance = 1/11 = 0.091.
Three findings, all consistent with the phase-pool thesis and one that **sharpens** it:

1. **Phase-pool is 2–3× pixels on materials.** Pixel-linear and patch-embed barely clear chance
   (0.224 / 0.229, ~2.5× chance) — textures have no fixed spatial layout, so a per-pixel linear map
   has almost nothing to grip. The orientation-phase arms sit at **0.49–0.60**, 2–3× the pixel
   baseline. On materials the phase descriptor *is* the signal.
2. **R has an INTERIOR optimum here (R=4).** Upright accuracy climbs 0.492 (R=1, global) →
   **0.599 (R=4)** then falls to 0.508 (R=8). This is between the two earlier datasets: MNIST digits
   want *maximum* locality (monotone-up in R), KTH-TIPS v1 wanted *global* (monotone-down), and
   11-class KTH-TIPS2-b sits in the middle — its materials carry **meso-scale structure** that R=4
   cells capture and R=8 over-fragments. The R-knob's optimum slides continuously with how much
   layout the class carries.
3. **Augmentation makes the phase arms near-perfectly rotation-invariant.** Rotation-augmented
   training drives the rotation drop to **≈0 (even positive)** for every phase arm — phase R=1 drop
   −0.0008, spatial-phase R=4 **+0.0379** — while the best rotated accuracy in the whole table is
   **spatial-phase R=4 augmented = 0.564**. Pixel / patch / mix stay pinned near chance under
   rotation (0.15–0.22). The invariance is a *property of the descriptor*, surfaced once the
   classifier is trained not to lean on the (near-useless) pixel path.

Figure: `reports/figures/cv-kth-tips2-11class.png`.

## Matrix — KTH-TIPS2-b (upright / rotated / drop)

**Train-upright:**

| arm | upright | rotated | drop |
|---|---|---|---|
| raw-pixel linear | 0.2239 | 0.1759 | −0.0480 |
| patch-embed (spatial) | 0.2290 | 0.1902 | −0.0387 |
| phase-pool R=1 (global) | 0.4916 | **0.4040** | −0.0875 |
| spatial-phase R=2 | 0.5598 | **0.4133** | −0.1465 |
| spatial-phase R=4 | **0.5993** | 0.3535 | −0.2458 |
| spatial-phase R=8 | 0.5084 | 0.2795 | −0.2290 |
| mix: pixels ⊕ phase | 0.2273 | 0.1911 | −0.0362 |

**Train-rotation-augmented:**

| arm | upright | rotated | drop |
|---|---|---|---|
| raw-pixel linear | 0.1625 | 0.1532 | −0.0093 |
| patch-embed (spatial) | 0.2003 | 0.2214 | +0.0210 |
| phase-pool R=1 (global) | 0.4579 | 0.4571 | −0.0008 |
| spatial-phase R=2 | 0.4975 | 0.5253 | +0.0278 |
| spatial-phase R=4 | 0.5261 | **0.5640** | +0.0379 |
| spatial-phase R=8 | 0.4731 | 0.4756 | +0.0025 |
| mix: pixels ⊕ phase | 0.2045 | 0.1987 | −0.0059 |

## Reading (measured / inferred)

- **Measured — pixels can't do materials; phase can.** Pixel/patch at ~0.22 vs phase at 0.49–0.60.
  This is the clean version of the "rotation-nuisance domain" claim: with no fixed layout, the linear
  pixel map is near-blind, and the orientation-phase histogram carries the class.
- **Measured — the mix collapses to the pixel level (0.227 ≈ pixel 0.224).** New vs KTH-TIPS v1
  (where the mix was mid-pack). *Mechanism (inferred):* the concatenated descriptor is dominated by
  the 4096-dim pixel block, which here carries almost no signal but plenty of variance; the linear
  classifier spends its capacity on the large, noisy block and loses the phase advantage. Naive
  concat is governed by the higher-dimensional block, not the more informative one — so fusion only
  helps when the spatial path is itself strong (MNIST), and *hurts* when it is near-useless
  (materials). Standardisation does not rescue it. This is a concrete argument against crude
  early-fusion and for keeping the phase descriptor as its own head.
- **Measured — R=4 is the interior optimum for both upright and augmented-robust.** Unlike the two
  earlier datasets (monotone in R), KTH-TIPS2-b peaks *inside* the sweep. Across three real datasets
  the R-optimum now traces a continuum: **MNIST → R=7 (max locality), KTH-TIPS2-b → R=4 (meso),
  KTH-TIPS v1 → R=1 (global)**. R is a single, interpretable domain knob, and its optimum is set by
  how much spatial layout the class actually carries.
- **Measured — augmentation ⇒ descriptor-level invariance.** The phase arms' drop → ≈0 (often
  positive) under augmented training; the exact-invariance property of `|DFT|` becomes usable once
  the classifier stops leaning on the pixel path. Best rotation-robust arm on materials =
  spatial-phase R=4 (0.564 rotated), well above the pure-global phase-pool (0.457).

## Files touched

| file | change |
|---|---|
| `scripts/dev/plot_kth_tips2.py` | new — R-sweep + augmentation-robustness figure |
| `reports/figures/cv-kth-tips2-11class.png` | new — the KTH-TIPS2-b result figure |
| `reports/2026-07-10-kth-tips2-11class-materials.md` | this report |

No `src/` or `examples/` change — this run reuses the existing `cv_bench` harness and
`spatial_phase_features` unchanged. No `CORE.YAML` items touched.

## Provenance

- kato15 (Katolab), Nagare at `d5b06a8`; CPU. Data: `~/nagare_data/kth_tips2/` (KTH-TIPS2-b,
  `kth-tips2-b_col_200x200`, decoded to raw 64×64, 11 classes, 75/25 split), repo-external.
- Reproduce: `cargo run --release --example cv_bench -- --dataset raw --data ~/nagare_data/kth_tips2 --n-train 3564 --n-test 1188 [--augment]`.
- Seed fixed inside `cv_bench` (linear-fit seed 7). Single split — accuracies are point estimates on
  the fixed 75/25 partition, not multi-seed medians; the *rankings* (phase ≫ pixel, R=4 interior
  optimum, augmentation ⇒ drop≈0) are the load-bearing claims and are large-margin.
- Mac suite green; `cargo clippy -D warnings` + `cargo fmt --check` clean (plot is Python only).

## Reading against the arc

This closes the "native-harder texture bench" open item from
`reports/2026-07-10-cv-ablation-results.md`. The phase-pool story is now shown on **three real
datasets** spanning the layout↔stationarity axis: digits (spatial), KTH-TIPS (stationary), and
KTH-TIPS2-b (meso-structured materials) — with a single R knob whose optimum slides across them, and
augmentation converting the descriptor's exact `|DFT|` invariance into measured rotation-robustness.
Signed-graph link prediction remains the flagship; this is the CV-expansion evidence.
