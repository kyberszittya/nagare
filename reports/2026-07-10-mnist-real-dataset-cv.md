# Nagare CV on a real dataset (MNIST) — the phase-pool's regime, measured

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Moved the CV work to real data (MNIST), on **kato15** (reconnected). Three closed-form arms trained
on upright digits, evaluated on **upright** and **randomly-rotated** test sets. The result cleanly
characterises each approach's regime: the **spatial** arms (raw-pixel, patch-embed) fit upright
digits well but **collapse ~5× harder under rotation**, while the rotation-invariant **phase-pool
holds** and **wins the rotated regime**. The framework's patch machinery also validates on real
data (85% upright).

## Result (MNIST, 8k train / 2k test, kato15)

| arm | upright | rotated | drop |
|---|---|---|---|
| raw-pixel linear (baseline) | 0.8805 | 0.2485 | **−0.632** |
| patch-embed (`patch_project`, spatial) | 0.8515 | 0.2595 | **−0.592** |
| **phase-pool `\|DFT\|`** (rotation-invariant) | 0.4155 | 0.2870 | **−0.129** |

Plot: `reports/figures/mnist-rotation-robustness.png`.

## Reading (measured / inferred)

- **Upright:** spatial arms win decisively (0.88 / 0.85 vs 0.42). Orientation *is* discriminative
  for upright digits and rotation is not a nuisance, so throwing away spatial layout + absolute
  orientation (what the phase-pool does) is a liability. The **patch-embed works on real MNIST**
  (0.85) — the framework's `patch_project` machinery validated beyond synthetic shapes.
- **Rotated:** the spatial arms collapse to ~0.25 (−0.6 drop) — they never saw rotation and encode
  it into the weights. The **phase-pool holds** (0.287, −0.13 drop) and **is the best arm under
  rotation**. Its 5× smaller drop is the point: rotation-invariance by construction.
- **Honest caveats** (why the phase-pool isn't *perfectly* invariant on rotated MNIST):
  1. bilinear rotation + the 28×28 boundary introduce interpolation/clipping artifacts, so a
     rotated digit isn't a clean circular shift of the orientation histogram → `|DFT|` isn't
     exactly preserved (0.42 → 0.287);
  2. digits are **not** truly rotation-invariant (a rotated 6 is a 9), so full rotation-invariance
     is a *ceiling* on rotated-digit ID — it caps everyone, and the phase-pool most.
- **Net:** MNIST confirms the phase-pool's scope predicted from the synthetic arc — it is the
  **rotation-robust** option (for rotation-nuisance tasks), not a general upright-digit recogniser.
  The right real-data showcase for it is a *texture / rotation-nuisance* dataset, not digits.

## Files touched

| file | change |
|---|---|
| `src/vision.rs` | (prior commit) grid-general `orientation_histogram` + `phase_features` |
| `examples/mnist_cv.rs` | IDX loader + 3 arms, upright + rotated eval (bilinear rotation) |
| `scripts/dev/plot_mnist_rotation.py`, `reports/figures/mnist-rotation-robustness.png` | plot |

## Provenance

- **kato15 reconnected** (Katolab RTX 6000 Ada); Nagare synced Mac→GitHub→kato15 to `5105c72`.
- MNIST fetched to kato15 `~/nagare_data/mnist/` (SNAP/ossci mirror IDX; repo-external). CPU run.
- Reproduce: `cargo run --release --example mnist_cv -- --data ~/nagare_data/mnist`.
- Mac suite **107/0** (3 heavy measurements ignored); clippy `-D warnings` + fmt clean.

## Open / next

- **Texture / rotation-nuisance real dataset** — the phase-pool's genuine home (KTH-TIPS, DTD, or
  a rotated-texture benchmark), where rotation-invariance is an asset not a liability.
- A **spatial phase map** (keep patch layout before the invariant transform) to lift the phase-pool
  on tasks like digits where spatial structure matters.
- CIFAR (color) for the patch/spatial arms. Signed-graph link prediction stays the flagship.
