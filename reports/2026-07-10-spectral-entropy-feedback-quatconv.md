# Nagare CV — spectral-entropy feedback on the quat-conv (the entropy thread, closed)

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Ran the framework-faithful "entropy feedback": `spectral_entropy` regularisation on the quaternion
conv's **learned** pooled feature matrix `P (batch × M)` during training (not a static feature).
**It's neutral-to-slightly-harmful** — at a strong, verified-active reg (`λ_eff = 1.0`, pushing the
concentrated feature spectrum `H_norm ≈ 0.10` toward τ=0.6) it *lowers* accuracy by Δ −0.025 (3/4
seeds). Combined with the two earlier entropy experiments, the thread closes with a clear,
mechanistically-grounded conclusion: **entropy feedback does not help supervised CV in Nagare, and
I can say precisely why in each form.**

## The three entropy experiments (all consistent)

| form | task | result | mechanism |
|---|---|---|---|
| entropy as a **feature** | shape ID | neutral | task saturates at 0.97 (no headroom) |
| entropy as a **feature** | orientation-disorder textures (headroom) | neutral (Δ −0.005) | entropy ≈ a function of `\|DFT\|` magnitudes — redundant |
| entropy as a **regulariser** on learned `P` | shape ID | **hurts (Δ −0.025)** | a spread prior fights the low-entropy concentration supervised discrimination wants |

## This experiment (spectral-entropy reg on `P`, 4 seeds, rotated shapes)

`SpectralEntropyReg::step(P)` each epoch → reg gradient added to the classification gradient,
backpropagated through the conv. Strong config (`lam_0=1.0`, τ=0.6), Lyapunov-scheduled.

| | seed 0 | seed 1 | seed 2 | seed 3 | median |
|---|---|---|---|---|---|
| no reg | 0.663 | 0.556 | 0.600 | 0.594 | **0.600** |
| entropy reg | 0.675 | 0.544 | 0.575 | 0.569 | 0.575 |

Diagnostics confirm the reg is **active, not too weak**: `λ_eff = 1.0` (clamped ceiling), and the
learned feature spectrum sits at `H_norm ≈ 0.10` — highly *concentrated*, far below the τ=0.6 target
— so the reg gradient is large and genuinely pushes toward spread. It changes the features; it just
doesn't help (slightly hurts).

## Reading (measured / inferred)

- **Measured:** an active spectral-entropy spread-prior on the supervised feature map lowers
  accuracy slightly (−0.025).
- **Inferred (mechanism):** supervised classification concentrates discriminative information in a
  **low-entropy** subspace (a few directions) — hence `H_norm ≈ 0.10` naturally. Forcing the
  spectrum toward high entropy (τ=0.6) *dilutes* that concentration → a small accuracy cost. The
  spectral-entropy reg is an **anti-collapse / diversity prior**; its home is regimes where
  representation collapse is the failure mode (self-/un-supervised objectives, the HSiKAN local
  entropy-gated update it was built for), **not** a shallow supervised classifier where the label
  already prevents collapse.
- **Honest close:** across feature-form and regulariser-form, entropy does not aid this supervised
  CV task. This is a well-characterised negative, not "entropy is useless" — it names the regime
  (supervised, non-collapsing) where the prior is the wrong one.

## Files touched

| file | change |
|---|---|
| `tests/vision_entropy_reg.rs` | **new** — quat-conv with vs without `spectral_entropy` reg on `P`, `H_norm`/`λ_eff` diagnostics |

The measurement is `#[ignore]`d (~50 s: per-epoch eigendecomposition); run with `cargo test --test
vision_entropy_reg -- --ignored`.

## CORE / deps

**None.** Reuses `spectral_entropy` + `common::vision`; no dependency change.

## Test results

- Full suite **106 / 0** (3 ignored: heavy group-conv + entropy-reg measurements); clippy
  `-D warnings` + fmt clean. Mac-only.

## Open / next

- Entropy thread **closed** for supervised CV. If revisited, the right setting is an
  **unsupervised / collapse-prone** objective (where the diversity prior earns its keep), not
  supervised classification.
- CV winner remains the **quaternion-phase pool** (`|DFT|`, 0.97 shapes / 0.80 textures). Real
  datasets (MNIST/CIFAR) and a spatial phase map are the open CV increments. Signed-graph link
  prediction stays the flagship.
