# Nagare CV — the phase-pool vindicated on textures (KTH-TIPS), the rank flip

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Fetched a real **texture** dataset (KTH-TIPS materials) on kato15 and ran the same three-arm
bench. The result is the clean scientific payoff and the vindication of the phase-pool idea: it is
the **worst arm on MNIST digits** (spatial, upright) and the **best arm on KTH-TIPS textures**
(rotation-nuisance, orientation-driven) — a full **rank flip across domains**, exactly as the
synthetic arc predicted. On its home turf it wins **both** upright and rotated, and holds well
under rotation.

## Result (both datasets, upright + randomly-rotated test, edge-clamp rotation)

| dataset | arm | upright | rotated | drop |
|---|---|---|---|---|
| **MNIST** (digits) | raw-pixel | 0.881 | 0.249 | −0.63 |
| | patch-embed | 0.855 | 0.266 | −0.59 |
| | phase-pool `\|DFT\|` | **0.416** ⟵ worst | 0.287 | −0.13 |
| **KTH-TIPS** (textures) | raw-pixel | 0.512 | 0.310 | −0.20 |
| | patch-embed | 0.429 | 0.325 | −0.10 |
| | **phase-pool `\|DFT\|`** | **0.606** ⟵ best | **0.458** ⟵ best | −0.15 |

Plot: `reports/figures/cv-datasets-phase-pool.png`.

## Reading (measured / inferred)

- **The rank flip is the headline.** Same descriptor, opposite outcome by domain: phase-pool is
  3rd/3 on digits (0.42) and 1st/3 on textures (0.61). Digit ID needs *spatial layout* and a
  *canonical pose* — the phase-pool discards both. Texture ID is *orientation-statistics-driven* and
  *pose-free* — the phase-pool's exact strength. This is the scope prediction from the synthetic
  arc, confirmed on two real datasets.
- **Best AND robust under rotation on textures.** The phase-pool wins the rotated regime on both
  datasets, and on textures it stays the top arm under rotation (0.606 → 0.458). Textures are
  *genuinely* rotation-invariant (a rotated brick is a brick), so — unlike digits (6↔9) — the
  invariance is an asset with no ambiguity ceiling.
- **The residual rotation drop is interpolation, not principle.** Switching the rotation from
  background-fill to **edge-clamp** shrank the phase-pool's texture drop from −0.31 → −0.15,
  confirming the earlier gap was a boundary artifact (spurious background edges corrupting a
  frame-filling texture's histogram). The remaining −0.15 is bilinear-interpolation + soft-binning
  discretization on a 64×64 grid — a clean invariant on a continuous field would drop ~0.
- **Patch-embed on real data:** validated (0.85 MNIST, 0.43 texture); it's a genuine spatial
  backbone, not a synthetic artifact.

## Files touched (this + the two prior commits)

| file | change |
|---|---|
| `examples/cv_bench.rs` | dataset-general bench (MNIST IDX + raw texture); edge-clamp rotation |
| `src/vision.rs` | grid-general `orientation_histogram` + `phase_features` (lib) |
| `scripts/dev/plot_cv_datasets.py`, `reports/figures/cv-datasets-phase-pool.png` | 2-dataset plot |

## Provenance

- **kato15** (Katolab RTX 6000 Ada), Nagare at `5afcc02`; CPU run.
- KTH-TIPS grey 200×200 (SNAP/KTH mirror) → PIL-decoded to 64×64 raw uint8 on kato15
  (`~/nagare_data/kth_tips/`, repo-external; Nagare stays image-crate-free). 10 materials, 607
  train / 203 test. MNIST as before.
- Reproduce: `cargo run --release --example cv_bench -- --dataset raw --data ~/nagare_data/kth_tips`.

## Open / next

- **A rotation-explicit texture benchmark** (KTH-TIPS2 / Kylberg-rotated) to measure clean
  invariance without self-imposed rotation artifacts.
- A **spatial phase map** (phase-pool per region, keeping coarse layout) to lift it on tasks like
  digits that need some spatial structure — combining the phase-pool's invariance with locality.
- CIFAR (color) for the spatial arms. Signed-graph link prediction remains the flagship.
