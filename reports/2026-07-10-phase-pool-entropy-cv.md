# Nagare CV — quaternion-phase pooling + entropy feedback (the rotor-pool, done right)

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

The vector rotor-pool failed 3 ways because it rotated *non-equivariant feature channels*
(scramble). The fix (Hajdu's): **pool the rotor's phase, not the rotated vector.** Each patch's
dominant gradient is a z-rotor whose phase is its orientation `θ_p`; pooling the magnitude-weighted
phases is an orientation histogram, and its **rotation-invariant** `|DFT|` classifies rotated
shapes at **0.969** — crushing every earlier approach (vector rotor-pool 0.52, single-θ canonical
0.60, C_8 group-conv 0.65). **The best CV result of the session, by +0.32.**

## The idea

Orientation is a genuine geometric **phase** (`e^{iθ}` = a unit z-quaternion). Under a global image
rotation `φ`, every patch phase `θ_p → θ_p + φ`, so the orientation histogram `h` **circularly
shifts**. Therefore:
- `|DFT(h)|_k` (circular-shift-invariant magnitudes) are **exact rotation invariants** of the *full*
  orientation distribution — not a canonicalisation to one frame, not an approximate group average;
- the **phase entropy** `H(h)` is likewise rotation-invariant — the entropy the framework's
  machinery feeds back.

No feature vector is ever rotated — only phases are pooled. This is precisely what a "rotor pool"
should be, and why it beats rotating features (which requires equivariance the learned channels
lack).

## Result (test accuracy, 4 seeds, randomly-rotated shapes; linear classifier on fixed features)

| feature | seed 0 | seed 1 | seed 2 | seed 3 | median |
|---|---|---|---|---|---|
| raw histogram (covariant) | 0.606 | 0.587 | 0.519 | 0.631 | 0.606 |
| **phase-pool `\|DFT\|`** | 0.919 | 0.969 | 0.962 | 0.969 | **0.969** |
| phase-pool + entropy | 0.938 | 0.969 | 0.962 | 0.969 | 0.969 |

**Full approach ladder** (rotated-shape median acc, from across the session's vision tests):

| vector rotor-pool | raw histogram | single-θ canonical | C_8 group-conv | **phase-pool** |
|---|---|---|---|---|
| 0.52 (failed) | 0.61 | 0.60 | 0.65 | **0.97** |

Plot: `reports/figures/phase-pool-cv-ladder.png`.

## Reading (measured / inferred)

- **Measured:** the phase-pool `|DFT|` is rotation-invariant *and* discriminative (bar/cross/L/T
  have distinct orientation spectra), giving 0.97 median — +0.36 over the covariant raw histogram
  and +0.32 over the previous best (group-conv). 4/4 seeds ≥ 0.92.
- **Why so much stronger than canonicalisation / group-conv:** those commit to one frame (single-θ)
  or average over a discrete group (C_8) — both lossy. The `|DFT|` of the phase histogram is the
  **exact** invariant of the *entire* orientation distribution, so it loses nothing to rotation
  while keeping all the shape-discriminative orientation structure.
- **Entropy feedback (honest):** neutral at the median because `|DFT|` alone already saturates the
  task (~0.97); it helps on the one non-saturated seed (0.919→0.938). So `H(h)` is a positive
  invariant feature whose value is masked by the ceiling here — it should matter on a harder task
  (real textures, more classes) with headroom.
- **No leak:** features come only from the input image gradients (leakage-free); the raw histogram
  (0.61) is the same information *without* the invariance, and it is much worse — so the +0.36 is
  the invariance, not information the label leaks.

## Files touched

| file | change |
|---|---|
| `tests/vision_phase_pool.rs` | **new** — phase histogram + `\|DFT\|`/entropy invariants, 3-arm ablation |
| `scripts/dev/plot_phase_pool.py`, `reports/figures/phase-pool-cv-ladder.png` | **new** — approach ladder |

## CORE / deps

**None.** Reuses `common::vision` + `softmax_k`; no dependency change.

## Test results

- Full suite **105 / 0** (2 ignored: the heavy group-conv measurement); phase-pool runs in ~1.7 s.
  clippy `-D warnings` + fmt clean. Mac-only.

## Open / next

- **Entropy on a task with headroom** — real textures / more classes, where `H(h)` (and the
  framework's spectral-entropy machinery) can show its contribution over `|DFT|` alone.
- Per-patch phase pooling into a spatial map (keep some layout) before the invariant transform, for
  shapes not separable by the global orientation distribution.
- Real datasets (MNIST/CIFAR). Signed-graph link prediction remains the flagship.
