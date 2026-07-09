# Nagare — N-dimensional patch projection op

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Added an **N-dimensional patch-projection** op (a rank-general patch-embed) with a
closed-form backward — per the user's preference for *n-dimensional* patches over the
2-D/ViT-only form. An input laid out as a `k`-D grid (`dims`) of `channels`-vectors is cut
into non-overlapping patches (`patch` per axis), each flattened to `∏ patch_i · channels`
and projected by a **shared** linear map `W (patch_vol, proj_dim)` (+ bias) →
`(n_patches, proj_dim)`. Works for any rank (1-D windows, 2-D image patches, 3-D/volumetric,
…) and needs **no image-specific data path** — it patchifies any tensor whose feature axis
factors as a grid.

## Design

The patchify is a **fixed gather** (precomputed once, cached), so the op is a gather + a
shared linear layer: `y[patch] = b + Σ_v x[cell(patch,v)] · W[v]`. The backward is the
standard linear backward scattered back through the gather map — `grad_x` (scatter),
`grad_w`, `grad_b`.

## Verification (FD-green, both machines)

- `patch_count_and_shape_generalise_over_rank` — a 3-D grid (4×4×2, patch 2×2×1, 3 ch) →
  8 patches, correct `patch_vol`.
- `gather_covers_every_cell_once` — the patchify is a **clean partition** (every input cell
  in exactly one patch).
- `backward_matches_finite_difference` — `grad_x`, `grad_w`, `grad_b` all match central-diff.

## Files touched

| file | change |
|---|---|
| `src/ops/patch_projection.rs` | **new** — `PatchConfig` + `patch_project_forward/backward` + tests |
| `src/ops/mod.rs`, `src/lib.rs` | +mod / +re-export |

## CORE / deps

**None.** Standalone; no new dependency.

## Test results

- Full suite **82 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.

## Notes / next

- It's a *shared*-weight patch embed (one `W` for all patches, like ViT). Per-patch weights
  or a positional term are easy extensions if wanted.
- Not yet wired into a model — a downstream classifier/regressor on patch tokens is the
  natural next use (e.g. patch tokens → readout, or → a sequence model).
- Still open from the same session: HSiKAN spline-pluggable (CR-Chebyshev vs KB) + the
  HSiKAN-on-tabular-graph comparison.

## Provenance

Repo `github.com/kyberszittya/nagare`. Developed on kato15, mirrored via the Mac. Rust 1.96.1.
