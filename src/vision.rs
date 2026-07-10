//! Reusable computer-vision primitives: gradient **orientation statistics** and the
//! **quaternion-phase-pool** rotation invariants.
//!
//! Grid-general (any `g×g` image, any bin count `b`) so examples on real datasets (MNIST/CIFAR)
//! and the synthetic tests share one implementation. The phase-pool pools per-pixel gradient
//! **orientation phases** into a circular histogram; under image rotation the histogram shifts, so
//! `|DFT(h)|` is rotation-invariant (`phase_features`). See `reports/2026-07-10-phase-pool-*`.

use std::f32::consts::TAU;

/// Central-difference gradient `(∂x, ∂y)` at pixel `(i, j)` of a `g×g` image (border-clamped).
fn grad_at(img: &[f32], g: usize, i: usize, j: usize) -> (f32, f32) {
    let at = |a: i32, b: i32| {
        let (a, b) = (
            a.clamp(0, g as i32 - 1) as usize,
            b.clamp(0, g as i32 - 1) as usize,
        );
        img[a * g + b]
    };
    (
        at(i as i32, j as i32 + 1) - at(i as i32, j as i32 - 1),
        at(i as i32 + 1, j as i32) - at(i as i32 - 1, j as i32),
    )
}

/// Per-image magnitude-weighted **orientation histogram** `(n, b)`: every pixel's gradient
/// contributes its orientation `θ = atan2(∂y, ∂x)` (weighted by magnitude) to the circular
/// histogram (soft-binned). Under image rotation `φ`, `θ → θ + φ` shifts the histogram.
///
/// # Preconditions
/// `imgs.len() == n * g * g`, `b >= 2`.
pub fn orientation_histogram(imgs: &[f32], n: usize, g: usize, b: usize) -> Vec<f32> {
    assert_eq!(imgs.len(), n * g * g);
    assert!(b >= 2);
    let mut h = vec![0.0f32; n * b];
    for s in 0..n {
        let img = &imgs[s * g * g..(s + 1) * g * g];
        for i in 0..g {
            for j in 0..g {
                let (gx, gy) = grad_at(img, g, i, j);
                let m = (gx * gx + gy * gy).sqrt();
                if m < 1e-6 {
                    continue;
                }
                let theta = gy.atan2(gx).rem_euclid(TAU);
                let pos = theta / TAU * b as f32;
                let lo = pos.floor() as usize % b;
                let frac = pos - pos.floor();
                h[s * b + lo] += m * (1.0 - frac);
                h[s * b + (lo + 1) % b] += m * frac;
            }
        }
    }
    h
}

/// Which phase-pool feature to build from an orientation histogram.
#[derive(Clone, Copy)]
pub enum PhaseFeature {
    /// The histogram itself — rotation-*covariant* (a floor baseline).
    Raw,
    /// `|DFT(h)|_{0..b/2}` — rotation-*invariant* (circular-shift-invariant magnitudes).
    Dft,
    /// `|DFT(h)|` ⊕ phase entropy `H(h)` — invariant, plus the nonlinear entropy feature.
    DftEntropy,
}

/// Feature dimension of one histogram under `mode` (`b` bins).
pub fn phase_feature_dim(b: usize, mode: PhaseFeature) -> usize {
    match mode {
        PhaseFeature::Raw => b,
        PhaseFeature::Dft => b / 2 + 1,
        PhaseFeature::DftEntropy => b / 2 + 2,
    }
}

/// One histogram `hs (b)` → its feature vector (`Raw`, `|DFT|`, or `|DFT|`⊕entropy).
fn hist_feature(hs: &[f32], b: usize, mode: PhaseFeature) -> Vec<f32> {
    if let PhaseFeature::Raw = mode {
        return hs.to_vec();
    }
    let nk = b / 2 + 1;
    let mut f = Vec::with_capacity(nk + 1);
    for k in 0..nk {
        let (mut re, mut im) = (0.0f32, 0.0f32);
        for (bi, &hv) in hs.iter().enumerate() {
            let ang = -TAU * k as f32 * bi as f32 / b as f32;
            re += hv * ang.cos();
            im += hv * ang.sin();
        }
        f.push((re * re + im * im).sqrt());
    }
    if matches!(mode, PhaseFeature::DftEntropy) {
        let tot: f32 = hs.iter().sum::<f32>() + 1e-6;
        let mut ent = 0.0f32;
        for &hv in hs {
            let pr = hv / tot;
            if pr > 1e-9 {
                ent -= pr * pr.ln();
            }
        }
        f.push(ent);
    }
    f
}

/// Build the requested feature from per-image histograms `(n, b)` → `(features, dim)`.
///
/// # Preconditions
/// `hist.len() == n * b`.
pub fn phase_features(hist: &[f32], n: usize, b: usize, mode: PhaseFeature) -> (Vec<f32>, usize) {
    assert_eq!(hist.len(), n * b);
    let dim = phase_feature_dim(b, mode);
    let mut f = vec![0.0f32; n * dim];
    for s in 0..n {
        let fs = hist_feature(&hist[s * b..s * b + b], b, mode);
        f[s * dim..s * dim + dim].copy_from_slice(&fs);
    }
    (f, dim)
}

/// **Spatial phase map**: divide each `g×g` image into an `r×r` grid of cells, build a per-cell
/// orientation histogram, take its phase feature, and concatenate across cells → `(n, r·r·cell_dim)`.
///
/// `r = 1` is the global [`phase_features`] (full global-rotation invariance, no layout); larger `r`
/// keeps coarser spatial layout — each cell is *locally* rotation-invariant, but the arrangement of
/// cells is not global-rotation-invariant. Sweeping `r` traces the invariance↔locality trade.
///
/// # Preconditions
/// `imgs.len() == n * g * g`, `1 <= r <= g`, `b >= 2`.
pub fn spatial_phase_features(
    imgs: &[f32],
    n: usize,
    g: usize,
    r: usize,
    b: usize,
    mode: PhaseFeature,
) -> (Vec<f32>, usize) {
    assert_eq!(imgs.len(), n * g * g);
    assert!(r >= 1 && r <= g && b >= 2);
    let cell_dim = phase_feature_dim(b, mode);
    let dim = r * r * cell_dim;
    let mut f = vec![0.0f32; n * dim];
    for s in 0..n {
        let img = &imgs[s * g * g..(s + 1) * g * g];
        for cr in 0..r {
            for cc in 0..r {
                let (y0, y1, x0, x1) = (cr * g / r, (cr + 1) * g / r, cc * g / r, (cc + 1) * g / r);
                let mut h = vec![0.0f32; b];
                for i in y0..y1 {
                    for j in x0..x1 {
                        let (gx, gy) = grad_at(img, g, i, j);
                        let m = (gx * gx + gy * gy).sqrt();
                        if m < 1e-6 {
                            continue;
                        }
                        let theta = gy.atan2(gx).rem_euclid(TAU);
                        let pos = theta / TAU * b as f32;
                        let lo = pos.floor() as usize % b;
                        let frac = pos - pos.floor();
                        h[lo] += m * (1.0 - frac);
                        h[(lo + 1) % b] += m * frac;
                    }
                }
                let cf = hist_feature(&h, b, mode);
                let base = s * dim + (cr * r + cc) * cell_dim;
                f[base..base + cell_dim].copy_from_slice(&cf);
            }
        }
    }
    (f, dim)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dft_is_rotation_invariant_but_raw_is_not() {
        // A 6×6 image with a single vertical edge; and the same image "rotated" by shifting the
        // orientation histogram. |DFT| must match; the raw histogram must differ.
        let g = 6;
        let mut img = vec![0.0f32; g * g];
        for i in 0..g {
            for j in 0..g {
                img[i * g + j] = if j < g / 2 { -1.0 } else { 1.0 }; // vertical edge → horizontal grad
            }
        }
        let b = 12;
        let h = orientation_histogram(&img, 1, g, b);
        // Manually circularly shift the histogram by 3 bins (a 90° "rotation" of orientations).
        let mut hr = vec![0.0f32; b];
        for k in 0..b {
            hr[(k + 3) % b] = h[k];
        }
        let (fd, _) = phase_features(&h, 1, b, PhaseFeature::Dft);
        let (fdr, _) = phase_features(&hr, 1, b, PhaseFeature::Dft);
        for (a, c) in fd.iter().zip(&fdr) {
            assert!(
                (a - c).abs() < 1e-4,
                "|DFT| not shift-invariant: {a} vs {c}"
            );
        }
        // Raw differs (it's covariant).
        let (raw, _) = phase_features(&h, 1, b, PhaseFeature::Raw);
        let (rawr, _) = phase_features(&hr, 1, b, PhaseFeature::Raw);
        assert!(raw.iter().zip(&rawr).any(|(a, c)| (a - c).abs() > 1e-3));
    }

    #[test]
    fn spatial_phase_r1_equals_global() {
        // r=1 spatial phase map == the global phase feature.
        let g = 8;
        let img: Vec<f32> = (0..g * g).map(|i| (i as f32 * 0.3).sin()).collect();
        let b = 12;
        let (global, gd) = phase_features(
            &orientation_histogram(&img, 1, g, b),
            1,
            b,
            PhaseFeature::Dft,
        );
        let (spatial, sd) = spatial_phase_features(&img, 1, g, 1, b, PhaseFeature::Dft);
        assert_eq!(gd, sd);
        for (a, c) in global.iter().zip(&spatial) {
            assert!((a - c).abs() < 1e-4, "r=1 spatial != global: {a} vs {c}");
        }
        // r=2 has 4× the cells → 4× the dim.
        let (_s2, sd2) = spatial_phase_features(&img, 1, g, 2, b, PhaseFeature::Dft);
        assert_eq!(sd2, 4 * sd);
    }
}
