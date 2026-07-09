//! N-dimensional patch projection (a rank-general patch-embed) with closed-form backward.
//!
//! Generalises the ViT patch-embed to **arbitrary spatial rank**: an input laid out as a
//! `k`-D grid (`dims`) of `channels`-vectors is cut into non-overlapping patches
//! (`patch` size per axis), each patch flattened to `patch_vol = ∏ patch_i · channels`
//! and projected by a **shared** linear map `W (patch_vol, proj_dim)` (+ bias) →
//! `(n_patches, proj_dim)`. Needs no image path — it works on any tensor whose feature
//! axis factors as a grid. The patchify is a fixed gather, so the op is a gather + a
//! shared linear layer; the backward is the standard linear backward scattered back.
//!
//! Layout: `x` is flat `(n, ∏dims · channels)`, row-major over `dims` with `channels`
//! innermost per cell.

/// Shape of an N-D patch projection.
#[derive(Debug, Clone)]
pub struct PatchConfig {
    /// Spatial grid dims (rank `k`).
    pub dims: Vec<usize>,
    /// Patch size per axis (must divide `dims`).
    pub patch: Vec<usize>,
    /// Channels per grid cell.
    pub channels: usize,
    /// Projection output dim per patch.
    pub proj_dim: usize,
}

impl PatchConfig {
    /// Construct + validate.
    ///
    /// # Panics
    /// Panics if `dims`/`patch` differ in rank, a patch does not divide its axis, or a
    /// dimension is 0.
    pub fn new(dims: Vec<usize>, patch: Vec<usize>, channels: usize, proj_dim: usize) -> Self {
        assert_eq!(
            dims.len(),
            patch.len(),
            "dims and patch must have equal rank"
        );
        assert!(!dims.is_empty() && channels >= 1 && proj_dim >= 1);
        for (&dd, &pp) in dims.iter().zip(&patch) {
            assert!(
                pp >= 1 && dd >= 1 && dd % pp == 0,
                "patch {pp} must divide dim {dd}"
            );
        }
        Self {
            dims,
            patch,
            channels,
            proj_dim,
        }
    }

    /// Grid cells `∏ dims`.
    pub fn cell_count(&self) -> usize {
        self.dims.iter().product()
    }
    /// Patches `∏ (dims_i / patch_i)`.
    pub fn n_patches(&self) -> usize {
        self.dims
            .iter()
            .zip(&self.patch)
            .map(|(d, p)| d / p)
            .product()
    }
    /// Flattened patch length `∏ patch_i · channels`.
    pub fn patch_vol(&self) -> usize {
        self.patch.iter().product::<usize>() * self.channels
    }
    fn input_len(&self) -> usize {
        self.cell_count() * self.channels
    }
}

/// Row-major unravel of `flat` against `radix`.
fn unravel(mut flat: usize, radix: &[usize]) -> Vec<usize> {
    let mut out = vec![0usize; radix.len()];
    for i in (0..radix.len()).rev() {
        out[i] = flat % radix[i];
        flat /= radix[i];
    }
    out
}

/// For each patch, the `patch_vol` input-cell offsets (within one sample). Fixed gather.
fn gather_map(cfg: &PatchConfig) -> Vec<Vec<usize>> {
    let k = cfg.dims.len();
    let mut strides = vec![1usize; k]; // row-major over dims
    for i in (0..k - 1).rev() {
        strides[i] = strides[i + 1] * cfg.dims[i + 1];
    }
    let grid: Vec<usize> = cfg
        .dims
        .iter()
        .zip(&cfg.patch)
        .map(|(d, p)| d / p)
        .collect();
    let patch_cells: usize = cfg.patch.iter().product();
    let mut map = Vec::with_capacity(cfg.n_patches());
    for q in 0..cfg.n_patches() {
        let qm = unravel(q, &grid);
        let mut cells = Vec::with_capacity(cfg.patch_vol());
        for r in 0..patch_cells {
            let rm = unravel(r, &cfg.patch);
            let cell_flat: usize = (0..k)
                .map(|i| (qm[i] * cfg.patch[i] + rm[i]) * strides[i])
                .sum();
            for c in 0..cfg.channels {
                cells.push(cell_flat * cfg.channels + c);
            }
        }
        map.push(cells);
    }
    map
}

/// Cache for the patch-projection backward (the fixed gather map + dims).
#[derive(Debug, Clone)]
pub struct PatchCache {
    map: Vec<Vec<usize>>,
    n: usize,
    input_len: usize,
}

/// Forward N-D patch projection. `x` flat `(n, ∏dims·channels)`, `w` `(patch_vol, proj_dim)`,
/// `b` `(proj_dim)`. Returns `y (n, n_patches·proj_dim)` and a backward cache.
///
/// # Panics
/// Panics if `x`/`w`/`b` lengths are inconsistent with `cfg`/`n`.
pub fn patch_project_forward(
    x: &[f32],
    w: &[f32],
    b: &[f32],
    n: usize,
    cfg: &PatchConfig,
) -> (Vec<f32>, PatchCache) {
    let (pv, pd, npatch) = (cfg.patch_vol(), cfg.proj_dim, cfg.n_patches());
    assert_eq!(x.len(), n * cfg.input_len());
    assert_eq!(w.len(), pv * pd);
    assert_eq!(b.len(), pd);
    let map = gather_map(cfg);
    let mut y = vec![0.0f32; n * npatch * pd];
    for s in 0..n {
        let xs = &x[s * cfg.input_len()..(s + 1) * cfg.input_len()];
        for (q, cells) in map.iter().enumerate() {
            let out = &mut y[(s * npatch + q) * pd..(s * npatch + q) * pd + pd];
            out.copy_from_slice(b);
            for (v, &cell) in cells.iter().enumerate() {
                let xv = xs[cell];
                let wrow = &w[v * pd..v * pd + pd];
                for (o, &wo) in wrow.iter().enumerate() {
                    out[o] += xv * wo;
                }
            }
        }
    }
    (
        y,
        PatchCache {
            map,
            n,
            input_len: cfg.input_len(),
        },
    )
}

/// Backward N-D patch projection → `(grad_x, grad_w, grad_b)`.
///
/// # Panics
/// Panics if `grad_y` length is inconsistent with `cache`/`cfg`.
pub fn patch_project_backward(
    x: &[f32],
    w: &[f32],
    cache: &PatchCache,
    grad_y: &[f32],
    cfg: &PatchConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (pv, pd, npatch) = (cfg.patch_vol(), cfg.proj_dim, cfg.n_patches());
    assert_eq!(grad_y.len(), cache.n * npatch * pd);
    let mut grad_x = vec![0.0f32; cache.n * cache.input_len];
    let mut grad_w = vec![0.0f32; pv * pd];
    let mut grad_b = vec![0.0f32; pd];
    for s in 0..cache.n {
        let xs = &x[s * cache.input_len..(s + 1) * cache.input_len];
        let gxs = &mut grad_x[s * cache.input_len..(s + 1) * cache.input_len];
        for (q, cells) in cache.map.iter().enumerate() {
            let gy = &grad_y[(s * npatch + q) * pd..(s * npatch + q) * pd + pd];
            for (o, &g) in gy.iter().enumerate() {
                grad_b[o] += g;
            }
            for (v, &cell) in cells.iter().enumerate() {
                let xv = xs[cell];
                let wrow = &w[v * pd..v * pd + pd];
                let gwrow = &mut grad_w[v * pd..v * pd + pd];
                let mut gx = 0.0f32;
                for (o, &g) in gy.iter().enumerate() {
                    gwrow[o] += xv * g;
                    gx += wrow[o] * g;
                }
                gxs[cell] += gx;
            }
        }
    }
    (grad_x, grad_w, grad_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn patch_count_and_shape_generalise_over_rank() {
        // 3-D grid 4×4×2, patch 2×2×1, 3 channels → (4/2)(4/2)(2/1)=8 patches.
        let cfg = PatchConfig::new(vec![4, 4, 2], vec![2, 2, 1], 3, 5);
        assert_eq!(cfg.n_patches(), 8);
        assert_eq!(cfg.patch_vol(), 12); // 2·2·1 patch cells × 3 channels
        let n = 2;
        let x: Vec<f32> = (0..n * cfg.input_len()).map(|i| 0.01 * i as f32).collect();
        let w: Vec<f32> = (0..cfg.patch_vol() * cfg.proj_dim)
            .map(|i| 0.1 * ((i as f32).sin()))
            .collect();
        let b = vec![0.05f32; cfg.proj_dim];
        let (y, _) = patch_project_forward(&x, &w, &b, n, &cfg);
        assert_eq!(y.len(), n * cfg.n_patches() * cfg.proj_dim);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn gather_covers_every_cell_once() {
        // A partition: each input cell appears in exactly one patch (non-overlapping).
        let cfg = PatchConfig::new(vec![4, 6], vec![2, 3], 2, 4);
        let map = gather_map(&cfg);
        let mut seen = vec![0u32; cfg.input_len()];
        for cells in &map {
            for &c in cells {
                seen[c] += 1;
            }
        }
        assert!(
            seen.iter().all(|&s| s == 1),
            "patchify is not a clean partition"
        );
    }

    #[test]
    fn backward_matches_finite_difference() {
        let cfg = PatchConfig::new(vec![4, 4], vec![2, 2], 2, 3);
        let n = 2;
        let x: Vec<f32> = (0..n * cfg.input_len())
            .map(|i| 0.2 * ((i as f32 * 0.7).sin()))
            .collect();
        let w: Vec<f32> = (0..cfg.patch_vol() * cfg.proj_dim)
            .map(|i| 0.15 * ((i as f32 * 1.1).cos()))
            .collect();
        let b = vec![0.1f32, -0.05, 0.2];
        let (y, cache) = patch_project_forward(&x, &w, &b, n, &cfg);
        let grad_y = vec![1.0f32; y.len()];
        let (gx, gw, gb) = patch_project_backward(&x, &w, &cache, &grad_y, &cfg);
        let eps = 1e-3;
        let sum = |xf: &[f32], wf: &[f32], bf: &[f32]| -> f32 {
            patch_project_forward(xf, wf, bf, n, &cfg).0.iter().sum()
        };
        for (idx, &g) in gx.iter().enumerate() {
            let (mut xp, mut xm) = (x.clone(), x.clone());
            xp[idx] += eps;
            xm[idx] -= eps;
            let num = (sum(&xp, &w, &b) - sum(&xm, &w, &b)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-2, "grad_x[{idx}] {g} vs {num}");
        }
        for (idx, &g) in gw.iter().enumerate() {
            let (mut wp, mut wm) = (w.clone(), w.clone());
            wp[idx] += eps;
            wm[idx] -= eps;
            let num = (sum(&x, &wp, &b) - sum(&x, &wm, &b)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-2, "grad_w[{idx}] {g} vs {num}");
        }
        for (idx, &g) in gb.iter().enumerate() {
            let (mut bp, mut bm) = (b.clone(), b.clone());
            bp[idx] += eps;
            bm[idx] -= eps;
            let num = (sum(&x, &w, &bp) - sum(&x, &w, &bm)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-2, "grad_b[{idx}] {g} vs {num}");
        }
    }
}
