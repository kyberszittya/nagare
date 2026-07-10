//! Differentiable **global orientation phase-pool** — the backward that turns the CV rotation
//! invariant from a *fixed descriptor* into a *trainable representation*.
//!
//! A learned front-end emits a per-pixel 2-vector **field** `(gx, gy)`; this op pools it into a
//! circular orientation histogram (soft-binned, magnitude-weighted) and returns the DFT magnitudes
//! `|DFT(h)|_{0..b/2}`, which are **rotation invariants** (rotating every field vector by `φ`
//! circularly shifts `h`, and `|DFT|` is shift-invariant). Unlike [`crate::vision`] (which computes
//! a *fixed* central-difference gradient and has no backward), this op is differentiable **w.r.t. the
//! field**, so gradients flow from the invariant feature back into an upstream learned operator
//! (quat-conv / dihedral-steer / HSiKAN). This is the global-pooling backpropagation the arc needed.
//!
//! # Forward
//! ```text
//!   m = ‖(gx,gy)‖,  θ = atan2(gy,gx) mod 2π          (pixels with m<ε are skipped)
//!   p = θ·b/2π,  ℓ = ⌊p⌋ mod b,  φ = p − ⌊p⌋         (soft bin)
//!   h[ℓ]     += m·(1−φ);   h[(ℓ+1)%b] += m·φ
//!   re_k = Σ_β h[β]·cos(−2πkβ/b);  im_k = Σ_β h[β]·sin(−2πkβ/b)
//!   feat_k = √(re_k² + im_k²),   k = 0 .. b/2
//! ```
//!
//! # Backward (given ḡ_k = ∂L/∂feat_k)
//! ```text
//!   re̅_k = ḡ_k·re_k/feat_k;   im̅_k = ḡ_k·im_k/feat_k         (feat_k>ε, else 0)
//!   h̄[β] = Σ_k re̅_k·cos(−2πkβ/b) + im̅_k·sin(−2πkβ/b)
//!   dL/dm = h̄[ℓ](1−φ) + h̄[(ℓ+1)%b]φ
//!   dL/dθ = m·(h̄[(ℓ+1)%b] − h̄[ℓ])·b/2π
//!   ∂L/∂gx = dL/dm·gx/m − dL/dθ·gy/m²
//!   ∂L/∂gy = dL/dm·gy/m + dL/dθ·gx/m²
//! ```
//! The bin index `ℓ` is piecewise-constant in `θ` but the soft weight `φ` is continuous, so the map
//! is differentiable a.e.; singularities are guarded and documented (cf. `cayley_rotor` at 180°).

use std::f32::consts::TAU;

/// Below this gradient magnitude a pixel carries no orientation and is skipped (grad 0). Matches
/// [`crate::vision::orientation_histogram`] so the two stay bit-consistent on the same field.
const MIN_MAG: f32 = 1e-6;

/// Feature dimension of one phase-pool descriptor: `|DFT(h)|_{0..b/2}` → `b/2 + 1` magnitudes.
pub fn phase_pool_dim(b: usize) -> usize {
    b / 2 + 1
}

/// Forward output: the invariant `feat` plus the histogram `hist`, saved so backward reproduces the
/// DFT without re-binning.
pub struct PhasePoolOut {
    /// Rotation-invariant features, flat `(n * (b/2+1))`.
    pub feat: Vec<f32>,
    /// Per-image orientation histograms, flat `(n * b)` — backward input.
    pub hist: Vec<f32>,
}

/// Precomputed DFT basis: `cos`/`sin` of `−2πkβ/b` for `k∈0..nk`, `β∈0..b`, row-major `k*b+β`.
fn dft_tables(b: usize, nk: usize) -> (Vec<f32>, Vec<f32>) {
    let mut cos_t = vec![0.0f32; nk * b];
    let mut sin_t = vec![0.0f32; nk * b];
    for k in 0..nk {
        for beta in 0..b {
            let ang = -TAU * k as f32 * beta as f32 / b as f32;
            cos_t[k * b + beta] = ang.cos();
            sin_t[k * b + beta] = ang.sin();
        }
    }
    (cos_t, sin_t)
}

/// Soft-bin one field vector: `(m, ℓ, φ)` or `None` if the vector is below [`MIN_MAG`].
#[inline]
fn soft_bin(gx: f32, gy: f32, b: usize) -> Option<(f32, usize, f32)> {
    let m = (gx * gx + gy * gy).sqrt();
    if m < MIN_MAG {
        return None;
    }
    let pos = gy.atan2(gx).rem_euclid(TAU) / TAU * b as f32;
    let lo = pos.floor() as usize % b;
    Some((m, lo, pos - pos.floor()))
}

/// Global orientation phase-pool forward. See the module docs for the math.
///
/// # Preconditions
/// `field.len() == n*g*g*2` (interleaved `(gx,gy)` per pixel, row-major); `b >= 2`.
///
/// # Postconditions
/// `feat.len() == n*(b/2+1)`, `hist.len() == n*b`; `feat` is invariant to a global rotation of the
/// field (see the `rotation_invariant` test).
///
/// # Panics
/// If `field.len() != n*g*g*2` or `b < 2`.
pub fn phase_pool_forward(field: &[f32], n: usize, g: usize, b: usize) -> PhasePoolOut {
    assert_eq!(field.len(), n * g * g * 2);
    assert!(b >= 2);
    let nk = phase_pool_dim(b);
    let (cos_t, sin_t) = dft_tables(b, nk);
    let mut hist = vec![0.0f32; n * b];
    let mut feat = vec![0.0f32; n * nk];
    for s in 0..n {
        let f = &field[s * g * g * 2..(s + 1) * g * g * 2];
        let h = &mut hist[s * b..s * b + b];
        for px in 0..g * g {
            if let Some((m, lo, frac)) = soft_bin(f[2 * px], f[2 * px + 1], b) {
                h[lo] += m * (1.0 - frac);
                h[(lo + 1) % b] += m * frac;
            }
        }
        let ft = &mut feat[s * nk..s * nk + nk];
        for (k, fk) in ft.iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for beta in 0..b {
                re += h[beta] * cos_t[k * b + beta];
                im += h[beta] * sin_t[k * b + beta];
            }
            *fk = (re * re + im * im).sqrt();
        }
    }
    PhasePoolOut { feat, hist }
}

/// Global orientation phase-pool backward. Given `grad_feat = ∂L/∂feat`, returns `∂L/∂field`.
///
/// # Preconditions
/// `field.len() == n*g*g*2`, `out.hist.len() == n*b`, `grad_feat.len() == n*(b/2+1)`, `b >= 2`.
///
/// # Postconditions
/// `grad.len() == n*g*g*2`; pixels below [`MIN_MAG`] and zero DFT modes contribute 0 (guarded).
///
/// # Panics
/// If the length preconditions do not hold.
pub fn phase_pool_backward(
    field: &[f32],
    out: &PhasePoolOut,
    grad_feat: &[f32],
    n: usize,
    g: usize,
    b: usize,
) -> Vec<f32> {
    assert_eq!(field.len(), n * g * g * 2);
    assert!(b >= 2);
    let nk = phase_pool_dim(b);
    assert_eq!(out.hist.len(), n * b);
    assert_eq!(grad_feat.len(), n * nk);
    let (cos_t, sin_t) = dft_tables(b, nk);
    let mut grad = vec![0.0f32; n * g * g * 2];
    for s in 0..n {
        let h = &out.hist[s * b..s * b + b];
        let gf = &grad_feat[s * nk..s * nk + nk];
        // ∂L/∂h[β] accumulated over frequencies.
        let mut hbar = vec![0.0f32; b];
        for (k, &gfk) in gf.iter().enumerate() {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for beta in 0..b {
                re += h[beta] * cos_t[k * b + beta];
                im += h[beta] * sin_t[k * b + beta];
            }
            let featk = (re * re + im * im).sqrt();
            if featk < 1e-12 {
                continue; // zero mode: |DFT| singular at 0, grad 0 (documented)
            }
            let (rebar, imbar) = (gfk * re / featk, gfk * im / featk);
            for beta in 0..b {
                hbar[beta] += rebar * cos_t[k * b + beta] + imbar * sin_t[k * b + beta];
            }
        }
        // Propagate h̄ through the soft-bin to the field.
        let f = &field[s * g * g * 2..(s + 1) * g * g * 2];
        let gout = &mut grad[s * g * g * 2..(s + 1) * g * g * 2];
        for px in 0..g * g {
            let (gx, gy) = (f[2 * px], f[2 * px + 1]);
            if let Some((m, lo, frac)) = soft_bin(gx, gy, b) {
                let (hb_lo, hb_hi) = (hbar[lo], hbar[(lo + 1) % b]);
                let dl_dm = hb_lo * (1.0 - frac) + hb_hi * frac;
                let dl_dtheta = m * (hb_hi - hb_lo) * (b as f32 / TAU);
                let m2 = m * m;
                gout[2 * px] = dl_dm * gx / m - dl_dtheta * gy / m2;
                gout[2 * px + 1] = dl_dm * gy / m + dl_dtheta * gx / m2;
            }
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Directional-derivative gradient check: `⟨∇f, u⟩` (analytic) vs central FD of `f` along `u`,
    /// over several directions. Robust to the op's a.e. kinks — a single pixel sitting near a
    /// soft-bin edge is averaged over the whole field instead of failing one component — where a
    /// per-component check is fragile. A real backward error is order-unity relative, not 1e-3.
    fn assert_dir_grad(loss: impl Fn(&[f32]) -> f32, grad: &[f32], field: &[f32]) {
        let eps = 1e-3;
        for d in 0..5 {
            let u: Vec<f32> = (0..field.len())
                .map(|i| ((i as f32 + d as f32 * 11.0) * 0.6).sin())
                .collect();
            let ana: f32 = grad.iter().zip(&u).map(|(g, ui)| g * ui).sum();
            let fp: Vec<f32> = field.iter().zip(&u).map(|(f, ui)| f + eps * ui).collect();
            let fm: Vec<f32> = field.iter().zip(&u).map(|(f, ui)| f - eps * ui).collect();
            let num = (loss(&fp) - loss(&fm)) / (2.0 * eps);
            assert!(
                (ana - num).abs() < 5e-3 + 2e-3 * num.abs(),
                "dir {d}: analytic {ana} vs fd {num}"
            );
        }
    }

    /// A smooth random field (no pixel lands on a bin edge) so the a.e.-differentiable soft-bin is
    /// FD-comparable.
    fn smooth_field(n: usize, g: usize) -> Vec<f32> {
        (0..n * g * g * 2)
            .map(|i| 0.5 * ((i as f32 * 0.7).sin() + 0.3 * (i as f32 * 1.9).cos()))
            .collect()
    }

    #[test]
    fn backward_matches_fd_scalar_sum() {
        let (n, g, b) = (2, 5, 12);
        let field = smooth_field(n, g);
        let out = phase_pool_forward(&field, n, g, b);
        let grad_feat = vec![1.0f32; out.feat.len()]; // L = Σ feat
        let ana = phase_pool_backward(&field, &out, &grad_feat, n, g, b);
        assert_dir_grad(
            |ff| phase_pool_forward(ff, n, g, b).feat.iter().sum(),
            &ana,
            &field,
        );
    }

    #[test]
    fn backward_matches_fd_weighted() {
        // Non-uniform upstream grad (a linear readout of feat) — exercises re/im mixing.
        let (n, g, b) = (2, 6, 16);
        let field = smooth_field(n, g);
        let out = phase_pool_forward(&field, n, g, b);
        let nk = phase_pool_dim(b);
        let w: Vec<f32> = (0..nk).map(|k| (k as f32 * 0.9).cos()).collect();
        let grad_feat: Vec<f32> = (0..n).flat_map(|_| w.iter().copied()).collect();
        let ana = phase_pool_backward(&field, &out, &grad_feat, n, g, b);
        assert_dir_grad(
            |ff| {
                phase_pool_forward(ff, n, g, b)
                    .feat
                    .chunks(nk)
                    .map(|c| c.iter().zip(&w).map(|(a, b)| a * b).sum::<f32>())
                    .sum()
            },
            &ana,
            &field,
        );
    }

    #[test]
    fn rotation_invariant() {
        // Rotating every field vector by a WHOLE-bin angle φ = 2π·j/b shifts θ by exactly j bins
        // (m unchanged, frac unchanged) → the histogram is an exact circular shift → |DFT| exactly
        // invariant. (For a fractional φ, invariance holds only up to soft-bin discretisation; the
        // *exact* claim is the whole-bin one, as in `vision.rs`.)
        let (n, g, b) = (1, 8, 16);
        let field = smooth_field(n, g);
        let phi = TAU * 3.0 / b as f32; // exactly 3 bins
        let (c, s) = (phi.cos(), phi.sin());
        let rot: Vec<f32> = field
            .chunks(2)
            .flat_map(|v| [v[0] * c - v[1] * s, v[0] * s + v[1] * c])
            .collect();
        let f0 = phase_pool_forward(&field, n, g, b).feat;
        let f1 = phase_pool_forward(&rot, n, g, b).feat;
        for (a, bb) in f0.iter().zip(&f1) {
            assert!(
                (a - bb).abs() < 1e-3,
                "not invariant under whole-bin rotation: {a} vs {bb}"
            );
        }
    }

    #[test]
    fn dc_mode_equals_total_mass() {
        // feat_0 = |Σ_β h[β]| = total gradient mass (k=0 DFT bin).
        let (n, g, b) = (1, 4, 8);
        let field = smooth_field(n, g);
        let out = phase_pool_forward(&field, n, g, b);
        let mass: f32 = out.hist.iter().sum();
        assert!(
            (out.feat[0] - mass).abs() < 1e-4,
            "dc {} vs mass {mass}",
            out.feat[0]
        );
    }

    #[test]
    fn zero_field_is_zero_and_finite() {
        // All-zero field: no pixel binned → feat all 0, backward all 0 (guards, no NaN).
        let (n, g, b) = (1, 4, 8);
        let field = vec![0.0f32; n * g * g * 2];
        let out = phase_pool_forward(&field, n, g, b);
        assert!(out.feat.iter().all(|&v| v == 0.0));
        let gf = vec![1.0f32; out.feat.len()];
        let grad = phase_pool_backward(&field, &out, &gf, n, g, b);
        assert!(grad.iter().all(|&v| v == 0.0 && v.is_finite()));
    }
}
