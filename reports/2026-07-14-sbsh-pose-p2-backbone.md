---
title: "SBSH Pose P2 — the P1 unblock: a real spatial backbone (ScBlock) does multi-pose keypoint localization without coord channels; skeleton conv is neutral (over-constrained joint)"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, sbsh, pose, keypoint, soft-argmax, sc-block, spatial-backbone, skeleton-hgconv, no-autograd, positive-with-neutral]
---

# SBSH Pose P2 — the spatial-backbone unblock

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `a97a896`+ · CPU

## Summary

P1 was confounded: coord channels + a single memorised pose made localisation free and occlusion harmless, so
the skeleton benefit was undemonstrable. P2 fixes the confound with the Neocognitron pieces built this session —
a **real spatial backbone** (`ScBlock` → conv head → `soft_argmax`) and **randomised poses**, no coord channels.

| finding | result (5 seeds) |
|---|---|
| **P1 UNBLOCK — multi-pose localization** | all-joint MAE **1.76 px** median (28-px image; 4/5 seeds 1.5–1.8 px, one outlier 5.5 px) |
| Elbow occlusion (large patch) | recoverable by backbone **alone** — 0.84 px (≤ clean) |
| Skeleton `hg_conv` A/B | **neutral** — median MAE 1.43 (skel) vs 1.76 (backbone); per-seed mixed |

Figure: `reports/figures/pose-backbone.png`. Stack: `img → ScBlock(1→K) → conv head(K→J) → J heatmaps →
soft_argmax → coords → [skeleton hg_conv] → coords'`, MSE + bone-length limb loss, all closed-form / FD-verified /
no autograd.

## The unblock (positive)

The spatial backbone localizes the joints of a 2-link arm across **random** poses (random θ₁, θ₂) to **~1.7 px**
with **no coord channels** — exactly what P1 could not do. P1's own report diagnosed the blocker ("local features
can't localize → needs coord channels → but coord channels memorize a single pose"); the `conv` S-cell + `ScBlock`
supplies real spatial features so localisation is learned from appearance and generalises across the pose
distribution. This closes the P1 gap: the pose net now stands on a genuine spatial representation, not a
coordinate lookup.

## The skeleton conv is neutral here — and why (honest secondary)

I hypothesised that occluding the elbow would create a structural-only recovery gap the skeleton `hg_conv` fills.
It does not, for a measured structural reason: on a 2-link arm the **middle joint is over-constrained**. A large
occlusion patch leaves a hole bounded by the two visible endpoint tips, and the true elbow sits at the hole
centre — so the backbone recovers it *without* the skeleton (occluded elbow error **0.84 px ≤ clean 1.64 px**).
There is no unique gap for the global prior to fill, so the skeleton conv is **neutral** (median all-joint MAE
1.43 skel vs 1.76 backbone — the difference is the skeleton smoothing one unstable seed, within 5-seed noise; it
even hurts clean localisation on some seeds). This is the *minor/neutral* verdict the "don't devolve a foundational
basis on noise" rule prescribes — **not** a negative on the skeleton mechanism, which is the framework's structural
substrate; only *this task* fails to exercise it.

The two joints of a 2-link arm bracket the regimes: the **middle** joint is over-constrained (recoverable without
structure), the **endpoint** (hand) is under-constrained under random θ₂ (unrecoverable *by anything* once its
evidence is gone). Neither leaves the structural-only-recoverable gap a skeleton prior needs. Demonstrating a
skeleton benefit requires a task with a genuine **redundant structural constraint** — a closed kinematic loop, a
symmetric multi-limb figure (occluded limb recovered from its mirror), or partial occlusion that preserves a
direction cue. That is the next step, not a defect here.

This is the same redundant-mechanism pattern the whole Neocognitron arc measured: an explicit structural prior
earns its place only where a bottleneck denies the base mechanism the signal (as the entropy pool did for global
rotation-invariance); where the base mechanism already recovers, the prior is redundant.

## Tests / gates

| item | result |
|---|---|
| `examples/pose_backbone` (baseline + `--hg`, 5 seeds) | table above |
| full suite | **165 / 0** (reuses FD-verified `sc_block`, `conv2d`, `soft_argmax`, `hg_message`, `linear`) |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

The example composes only already-FD-verified ops; the composed training convergence (~1.7 px) is the integration
check. No new library op, no new deps, no CORE.YAML.

## Files touched

| file | change |
|---|---|
| `examples/pose_backbone.rs` | new — spatial-backbone pose net (ScBlock→conv head→soft_argmax) + skeleton hg_conv A/B |
| `scripts/dev/plot_pose_backbone.py`, `reports/figures/pose-backbone.png`, `reports/figures/pb*_*.json` | figure + 5-seed results |

## Next

- **Qualitative visual** — a prediction-overlay panel / GIF of predicted vs GT skeletons across poses (§9 spatial
  output); cheap follow-up on this example.
- **A task that exercises the skeleton** — closed kinematic loop / symmetric multi-limb figure / partial occlusion,
  where a joint is recoverable *only* from redundant structure. That is where the skeleton `hg_conv` (and the
  entropy pool's structural read) should finally pay off.
- Wire the entropy-pool pose readout (global orientation) alongside the per-joint `soft_argmax` for a full
  pose+keypoint head.

## Provenance

- Mac (Apple Silicon), Nagare `a97a896`+; CPU. Analytic data (2-link arm, G=28, L₁=L₂=7, shoulder fixed, random
  θ₁∈[−2.2,−0.9], θ₂∈[−1.1,1.1]). 5 seeds via `--seed=N`. Train 1600 poses, 60% random-joint patch occlusion; eval
  80 fresh poses, elbow occlusion radius 6.
- Reproduce: `cargo run --release --example pose_backbone -- [--hg] [--seed=N]`.
