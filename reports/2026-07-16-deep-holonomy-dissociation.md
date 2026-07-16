---
title: "Deep holonomy net learns useful features THROUGH DEPTH — the empirical crux of 'one step to deep-representation learning', double dissociation, closed-form/no-autograd"
date: 2026-07-16
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, deep, entropy, dissociation, clifford, positive]
---

# Deep holonomy dissociation — the empirical crux, positive

Date: 2026-07-16 · Mac (arm64, CPU) · on-mission · no autograd

## Summary

The mechanism half (a deep rotor-mesh net composes through depth with a closed-form, FD-verified backward) was
proven earlier. This is the **empirical half**: does the holonomy net *learn useful deep features*, readable by the
entropy (arrangement) signal? **Yes — a clean double dissociation.**

Pipeline (all closed-form, no autograd tape): `RotorMeshNet` (deep learned rotors + mesh mix) → readout → logistic,
trained by the composed closed-form gradient (BCE → logistic → readout backward → `RotorMeshNet::backward` →
bivectors). Task: a ring mesh, class 0 = a **coherent twist** `v_i = R(θ·i)·u` (anisotropic covariance), class 1 =
**isotropic** random directions, **zero-mean by construction** so a raw mean is chance. Two readouts on the *same*
learned net: **entropy** (normalized spectral eigen-entropy, arrangement-sensitive) vs **mean** (arrangement-blind).
5 seeds, held-out AUROC median:

| | entropy readout | mean readout |
|---|---|---|
| **deep (L=3)** | **0.759** | 0.542 |
| shallow (L=1) | 0.561 | 0.544 |

Figure: `reports/figures/holonomy-dissociation.png`.

## The double dissociation

- **deep + entropy = 0.759** — the winner.
- **Remove depth** (deep→shallow, entropy readout): 0.759 → **0.561** (~chance). Depth is load-bearing.
- **Remove the entropy readout** (entropy→mean, deep): 0.759 → **0.542** (~chance). The arrangement is load-bearing.
- **Both removed**: 0.544 (chance).

Neither axis alone suffices; only their conjunction wins. So the deep holonomy net **learns features through depth**
that are (a) not there at a single layer, and (b) visible only to the arrangement-sensitive (entropy) readout — the
mean pooling is blind to them regardless of depth. That is exactly "deep-representation learning" in the sense that
was the open question: the representation is *learned* (the per-layer bivectors), the learning *propagates through
depth* (via the closed-form adjoint/inverse-rotor transport), and it is read by *entropy, not a supervised feature*.

## Integrity — the gradient is verified, the result is not a phantom

Before trusting any training number, a **hard FD gate** checked the entire end-to-end closed-form gradient (BCE →
logistic → readout → deep net → bivectors) against finite differences: **max |analytic − FD| = 6.2e-5 (entropy),
5.9e-5 (mean)** — PASS. A wrong gradient would have made the AUROC meaningless; it is verified. Live loss was
printed during training (never run blind). AUROC is reported as separability (symmetric under label flip), 5-seed
median, on a held-out set drawn the same way.

## Honest scope

- **Trained by GD on the closed-form gradient — the stepping stone, not yet the thesis.** The learning signal here
  is supervised BCE, and *entropy* is the **readout/objective feature**; the update is iterative gradient descent on
  the (hand-derived, no-autograd) holonomy gradient. The pure thesis — an **instantaneous** entropy/holonomy
  feedback that updates the bivectors in one shot — is the remaining refinement. What is proven here is that the
  deep holonomy representation is *learnable and useful*, and that entropy (not mean) is what reads it.
- **AUROC 0.759 is above chance, not saturated.** The task is deliberately hard (zero-mean, arrangement-only); the
  training is gentle (loss 0.6931 → 0.6885 over 200 epochs). The dissociation is clear even though the absolute
  score is modest — pushing LR/epochs/scale would likely raise it, untested.
- **Small, controlled** (N=12 ring, 120 train/test, 5 seeds). A snapshot proof-of-mechanism, not a scaled benchmark.

## Clifford / simplicial grounding

Rotors = unit quaternions (even subalgebra of Cl(3,0), Spin(3)=SU(2)); the mesh mix is the simplicial
coboundary/boundary contraction; the entropy readout is HSiKAN's spectral eigen-entropy (`spectral_reg_value_grad`).
Every gradient is hand-derived and FD-verified — no autograd anywhere in the pipeline.

## Tests / gates

| item | result |
|---|---|
| FD gate (end-to-end gradient, entropy + mean) | PASS (max err 6e-5) |
| `examples/holonomy_deep_dissociation` (5 seeds × 2×2) | table above |
| full suite | **185 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `examples/holonomy_deep_dissociation.rs` | new — task, closed-form pipeline, FD gate, training, 2×2 sweep |
| `scripts/dev/plot_holonomy_dissociation.py`, `reports/figures/holonomy-dissociation.png`, `reports/figures/holonomy_dissociation_results.json` | figure + data |

Reuses `RotorMeshNet` + `spectral_reg_value_grad` + `MeshTopology` — no new library code (the experiment is the
config-over-framework layer).

## Provenance

Nagare `5ea9b8c`+ on Hajdus-MacBook-Pro (arm64, CPU-only), `cargo 1.96.1`. Seeds 0–4. Reproduce:
`cargo run --release --example holonomy_deep_dissociation` (FD gate + smoke + 2×2). CPU-only — no kato15.

## Next

- **Replace GD with the instantaneous entropy/holonomy feedback** — the on-thesis learning rule (one-shot bivector
  update from the entropy signal), the last gap to the pure claim.
- Push scale/score (deeper, larger mesh, tuned LR) now the dissociation is established.
- The `rotor_holonomy` (ordered loop product) as an explicit per-layer holonomy feature alongside the entropy readout.
