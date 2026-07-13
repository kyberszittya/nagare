//! Differentiable **soft-argmax** keypoint head — a per-joint heatmap → expected
//! sub-pixel coordinate. The canonical no-autograd keypoint primitive for the
//! SBSH pose line (image → SBSH grid → signed hg-conv over the skeleton →
//! per-joint heatmap → *this* → `(x,y)`).
//!
//! # Forward
//! Per joint, softmax over the `g×g` grid at temperature `tau` (max-subtracted,
//! overflow-safe), then the expected position:
//! ```text
//! p_ij = softmax(s_ij / tau);  x_hat = Σ_ij p_ij·j,  y_hat = Σ_ij p_ij·i
//! ```
//! `x` is the column (horizontal), `y` the row (vertical), in grid-index units
//! `[0, g-1]`. Sub-pixel and differentiable, unlike a hard argmax.
//!
//! # Backward
//! The softmax first-moment adjoint:
//! ```text
//! ∂x_hat/∂s_ij = (p_ij/tau)(j - x_hat);  ∂y_hat/∂s_ij = (p_ij/tau)(i - y_hat)
//! s_bar_ij = (p_ij/tau)·( x_bar·(j - x_hat) + y_bar·(i - y_hat) )
//! ```
//! Elementary, FD-verified.
//!
//! **No novelty claimed** (integral pose regression, Sun et al. ECCV 2018; DSNT,
//! Nibali et al. 2018). The contribution is the closed-form, hand-derived,
//! FD-verified op in the Nagare no-autograd discipline.
//!
//! # Preconditions
//! - `heat.len() == n*g*g`, `g >= 1`, `tau > 0` (clamped to `MIN_TAU`).
//! # Postconditions
//! - `coord in [0, g-1]^2`; a sharp single peak → its location; a flat map → the
//!   grid centroid. Multi-modal input → the *mean* of the modes (a documented
//!   soft-argmax property, not a defect).

const MIN_TAU: f32 = 1e-4;

/// Forward output: expected coordinates + the softmax maps for the backward.
pub struct SoftArgmaxOut {
    /// Expected coordinates, flat `(n, 2)` = `[x_hat, y_hat]` per joint.
    pub coord: Vec<f32>,
    /// Softmax probability maps, flat `(n, g*g)`.
    p: Vec<f32>,
    /// Grid side (`g*g` positions per joint).
    g: usize,
}

/// Soft-argmax forward. See the module docs.
///
/// # Panics
/// If `heat.len() != n*g*g`.
pub fn soft_argmax_forward(heat: &[f32], n: usize, g: usize, tau: f32) -> SoftArgmaxOut {
    assert_eq!(heat.len(), n * g * g);
    let tau = tau.max(MIN_TAU);
    let np = g * g;
    let mut p = vec![0.0f32; n * np];
    let mut coord = vec![0.0f32; n * 2];
    for j in 0..n {
        let s = &heat[j * np..(j + 1) * np];
        let mx = s.iter().copied().fold(f32::MIN, f32::max);
        let pj = &mut p[j * np..(j + 1) * np];
        let mut sum = 0.0f32;
        for (o, &sv) in pj.iter_mut().zip(s) {
            let e = ((sv - mx) / tau).exp();
            *o = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let (mut xh, mut yh) = (0.0f32, 0.0f32);
        for (idx, pv) in pj.iter_mut().enumerate() {
            *pv *= inv;
            xh += *pv * (idx % g) as f32;
            yh += *pv * (idx / g) as f32;
        }
        coord[j * 2] = xh;
        coord[j * 2 + 1] = yh;
    }
    SoftArgmaxOut { coord, p, g }
}

/// Soft-argmax backward. Given `grad_coord` (`n*2`), returns `grad_heat` (`n*g*g`).
///
/// # Panics
/// If `grad_coord.len() != n*2`.
pub fn soft_argmax_backward(
    out: &SoftArgmaxOut,
    grad_coord: &[f32],
    n: usize,
    tau: f32,
) -> Vec<f32> {
    assert_eq!(grad_coord.len(), n * 2);
    let g = out.g;
    let np = g * g;
    let inv_tau = 1.0 / tau.max(MIN_TAU);
    let mut grad = vec![0.0f32; n * np];
    for j in 0..n {
        let (xb, yb) = (grad_coord[j * 2], grad_coord[j * 2 + 1]);
        let (xh, yh) = (out.coord[j * 2], out.coord[j * 2 + 1]);
        let pj = &out.p[j * np..(j + 1) * np];
        let gj = &mut grad[j * np..(j + 1) * np];
        for (idx, gv) in gj.iter_mut().enumerate() {
            let (col, row) = ((idx % g) as f32, (idx / g) as f32);
            *gv = pj[idx] * inv_tau * (xb * (col - xh) + yb * (row - yh));
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single Gaussian bump centred at (row=cy, col=cx) on a g×g grid.
    fn bump(g: usize, cx: f32, cy: f32, sharp: f32) -> Vec<f32> {
        let mut h = vec![0.0f32; g * g];
        for i in 0..g {
            for k in 0..g {
                let d2 = (k as f32 - cx).powi(2) + (i as f32 - cy).powi(2);
                h[i * g + k] = -sharp * d2;
            }
        }
        h
    }

    #[test]
    fn backward_matches_fd() {
        let (n, g, tau) = (2usize, 6usize, 0.7f32);
        let mut heat = bump(g, 3.5, 1.8, 0.4);
        heat.extend(bump(g, 1.0, 4.2, 0.3));
        let out = soft_argmax_forward(&heat, n, g, tau);
        let gc: Vec<f32> = vec![0.7, -0.4, -0.3, 0.6];
        let grad = soft_argmax_backward(&out, &gc, n, tau);
        let dot = |h: &[f32]| -> f32 {
            soft_argmax_forward(h, n, g, tau)
                .coord
                .iter()
                .zip(&gc)
                .map(|(&c, &g)| c * g)
                .sum()
        };
        let eps = 1e-2;
        for i in 0..heat.len() {
            let mut hp = heat.clone();
            hp[i] += eps;
            let mut hm = heat.clone();
            hm[i] -= eps;
            let num = (dot(&hp) - dot(&hm)) / (2.0 * eps);
            // 2% relative + a 1.5e-3 absolute floor: far-from-peak pixels have
            // near-zero gradients (~1e-6) that the FD (O(eps^2) bias) cannot
            // resolve; the floor accepts those, the relative term checks the rest.
            assert!(
                (grad[i] - num).abs() < 1.5e-3 + 2e-2 * num.abs(),
                "grad[{i}] {} vs fd {num}",
                grad[i]
            );
        }
    }

    #[test]
    fn sharp_peak_recovers_argmax() {
        let g = 9;
        let h = bump(g, 6.0, 2.0, 3.0); // sharp
        let out = soft_argmax_forward(&h, 1, g, 0.2);
        assert!((out.coord[0] - 6.0).abs() < 0.15, "x {} vs 6", out.coord[0]);
        assert!((out.coord[1] - 2.0).abs() < 0.15, "y {} vs 2", out.coord[1]);
    }

    #[test]
    fn flat_gives_centroid() {
        let g = 8;
        let h = vec![0.0f32; g * g];
        let out = soft_argmax_forward(&h, 1, g, 1.0);
        let c = (g - 1) as f32 / 2.0;
        assert!((out.coord[0] - c).abs() < 1e-4 && (out.coord[1] - c).abs() < 1e-4);
    }

    #[test]
    fn translation_equivariant() {
        let g = 10;
        let a = soft_argmax_forward(&bump(g, 3.0, 3.0, 1.0), 1, g, 0.3);
        let b = soft_argmax_forward(&bump(g, 5.0, 6.0, 1.0), 1, g, 0.3);
        assert!((b.coord[0] - a.coord[0] - 2.0).abs() < 0.1, "dx");
        assert!((b.coord[1] - a.coord[1] - 3.0).abs() < 0.1, "dy");
    }
}
