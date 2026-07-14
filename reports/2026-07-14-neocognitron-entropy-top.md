---
title: "Nagare Neocognitron — the entropy global-pooling top: rotation-invariant recognition + equivariant pose + real-time update (hypothesis confirmed)"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, neocognitron, entropy-pool, global-invariant-top, pose, rotation-invariance, real-time, no-autograd, positive]
---

# Neocognitron — the entropy global-pool top

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `b943873`+ · CPU

## Summary

The N3 held-out gap identified the missing primitive: a **globally rotation-invariant top**. This builds it as
the crate's own **entropy global pooling** and tests the user's hypothesis — *in a Neocognitron the entropy
feedback can learn object recognition AND pose detection, with a fast, real-time weight update*. **All three
claims confirmed, robustly (5 seeds).**

| claim | measurement (median over 5 seeds) |
|---|---|
| **Recognition** (invariant `Hs`) | entropy-top held-out-rotation AUROC **1.000** (all seeds) vs mean-top **0.500** |
| **Pose** (equivariant angle) | principal-axis MAE **0.9°** [0.5°, 1.4°] |
| **Speed** (weight update) | **~300 µs/update ≈ 3300 updates/s** — real-time (kHz, CPU) |

Figure: `reports/figures/neocognitron-entropy.png`.

## The op — `global_entropy_pool` (rotation-invariant, arrangement-sensitive)

A purely permutation-invariant value pool (mean/std/entropy of the response *multiset*) is **blind to spatial
arrangement**, so it cannot separate a length-matched L-corner from a bar — which is exactly why the N3 mean-top
sat at chance. The fix reads the **response-weighted spatial covariance**, whose rotation *invariants* keep
arrangement:

```
w_i = resp_i²  →  weighted covariance [[a,b],[b,d]]
T = a+d (trace),  Dt = ad−b² (det),  q = Dt/T²
Hs = eigenvalue-distribution entropy  (−Σ e·ln e, e = (1±√(1−4q))/2)
feat = [mean, T, Hs] per channel     (all rotation-invariant)
```

`Hs → 0` for an elongated bar (rank-1 covariance), `Hs → ln 2` for an isotropic corner — a **continuous**
rotation-invariant that separates the two shapes at *every* angle (unit test `bar_has_lower_entropy_than_corner`;
`entropy_is_rotation_invariant` holds `Hs` constant across 0–90°). The closed-form backward
(`∂a/∂w_i = ((x−cx)²−a)/M`, `∂Hs/∂q = ln(e1/e2)/disc`) is FD-verified. O(H·W) per channel, one 2×2 eigen — **no
`|G|` steering**, which is why it is fast.

## One pool, two readouts — recognition *and* pose

The same covariance yields both halves of the hypothesis:

- **Recognition** — the rotation-**invariant** `Hs` (+`T`, `mean`) → linear head → corner-vs-bar. Held-out-rotation
  AUROC **1.000 on all 5 seeds**, generalising to non-group angles (22.5°, 67.5°, 200°, 250°) too, because the
  covariance invariant is *continuous*, unlike the discrete C₈ C-cell. The 1-block **mean-top** cannot even fit
  (train 0.500 — length-matched shapes share a mean); this is the cleanest possible contrast.
- **Pose** — the rotation-**equivariant** principal-axis angle `½·atan2(2b, a−d)` recovers a bar's orientation to
  **≈0.9° MAE** (mod π). The invariant recognises the object; the equivariant angle localises it — from one
  forward pass (`principal_angle_tracks_bar_orientation` unit test).

## Speed — the weight update is real-time

Per-sample **forward + backward + Adam update** wall time is **~300 µs** (~3300 updates/s) on CPU, measured over
2400 updates. That is ~100× the 30 Hz real-time bar; the closed-form no-autograd op and the O(H·W) pool are what
keep it there. The hypothesis's "fast, under a real-time measure" holds with large margin.

## Honest caveats

- **Oriented warm-start required.** With a *random* conv, 2/5 seeds stalled at train 0.5: a noise resp map has an
  isotropic covariance, so `Hs` starts uninformative and the conv never receives an edge-forming gradient
  (chicken-and-egg, same shape as the CR warm-start earlier in this arc). Seeding the S-cell as an oriented Sobel
  bank (it stays fully learnable) fixes it — 5/5 seeds then hit 1.000. Reported, not hidden.
- **Controlled task.** Two rigid shapes, single object, clean background. The result validates the *mechanism*
  (entropy feed → invariant recognition + equivariant pose + real-time); richer multi-object scenes are the scale
  test, not claimed here.
- **Pose on bars.** A corner's pose is ambiguous (two arms); MAE is measured on bars, where orientation is defined.

## Tests / gates

| item | result |
|---|---|
| `global_entropy_pool::backward_matches_fd` | pass (closed-form = FD) |
| `global_entropy_pool::entropy_is_rotation_invariant` | pass (`Hs` const across rotation) |
| `global_entropy_pool::bar_has_lower_entropy_than_corner` | pass (the discriminant) |
| `global_entropy_pool::principal_angle_tracks_bar_orientation` | pass (pose readout) |
| `examples/neocognitron_entropy` (entropy vs mean, 5 seeds) | table above |
| full suite | **165 / 0** |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Files touched

| file | change |
|---|---|
| `src/ops/global_entropy_pool.rs` | new — the rotation-invariant entropy pool + equivariant pose readout, FD-verified (4 tests) |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export |
| `examples/neocognitron_entropy.rs` | new — the 3-claim hypothesis test (recognition/pose/speed) + oriented warm-start |
| `scripts/dev/plot_entropy.py`, `reports/figures/neocognitron-entropy.png`, `reports/figures/ent*_*.json` | figure + 5-seed results |

No new deps; no CORE.YAML.

## Where this lands the arc

The Neocognitron now has the full stack **and** its group-invariant top, all closed-form / FD-verified / no
autograd: `conv2d` S-cell (N0) · `group_pool` C-cell (N1) · S/C isolation (N2) · learnable-dodge (N2b) ·
`ScBlock` deep stack (N3) · **`global_entropy_pool` top (this)**. N3 showed the C-cell gives *local* orientation-
invariance; the entropy top supplies the *global* rotation-invariance N3 lacked — and confirms the user's
hypothesis that the entropy feed carries both recognition and pose at real-time update cost.

## Next

- Feed this top into the **SBSH detector / pose `soft_argmax`** head — the original pose-P1 unblock, now with a
  proven rotation-invariant backbone + pose readout.
- Multi-object / cluttered scenes (per-region entropy pools) — the scale test the caveats flag.

## Provenance

- Mac (Apple Silicon), Nagare `b943873`+; CPU. Analytic data (corner/bar strokes, G=24). 5 seeds via `--seed=N`.
  Train θ∈{0°,90°}; recognition test θ∈{45°,135°,22.5°,67.5°,112.5°,157.5°,200°,250°}; pose over θ∈[0°,180°) step
  15°. Timing over 2400 updates on the described host.
- Reproduce: `cargo run --release --example neocognitron_entropy -- [--mean-top] [--seed=N]`.
