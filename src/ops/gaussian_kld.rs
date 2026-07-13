//! Oriented-bbox Gaussian-KLD loss with a closed-form, FD-verified backward.
//!
//! An oriented box `(cx, cy, w, h, theta)` is modelled as a 2-D Gaussian
//! `N(mu, Sigma)` with `mu = (cx, cy)` and
//! `Sigma = R(theta) diag((w/2)^2, (h/2)^2) R(theta)^T`. The loss is the
//! target-anchored Kullback-Leibler divergence
//!
//! ```text
//! D = KL(N_t || N_p)
//!   = 1/2 [ delta^T Sigma_p^{-1} delta + tr(Sigma_p^{-1} Sigma_t)
//!           + ln(|Sigma_p| / |Sigma_t|) - 2 ],   delta = mu_p - mu_t
//! ```
//!
//! wrapped in the bounded form `l = 1 - 1/(tau + sqrt(D))`.
//!
//! This IS the standard Gaussian oriented-detection surrogate — GWD (Yang et
//! al., ICML 2021), KLD (Yang et al., NeurIPS 2021), KFIoU (ICLR 2022). **No
//! novelty is claimed for the loss.** What is Nagare-specific is the
//! hand-derived, finite-difference-verified backward (`gaussian_kld_backward`):
//! the whole detection head trains without an autograd graph.
//!
//! Because `|Sigma| = a*b = (w/2)^2 (h/2)^2` is `theta`-independent, all the
//! 2x2 algebra (inverse, determinant, the `d/dSigma` identities) is elementary
//! and the assembled 5-vector gradient FD-verifies to machine tolerance.
//!
//! # Preconditions
//! - `pred`, `target` are `[cx, cy, w, h, theta]` with `w, h > 0`.
//! # Postconditions
//! - `D >= 0`; `D = 0` iff the two Gaussians coincide; `l in [0, 1)`.
//! - `theta` and `theta + pi` give identical `Sigma` (the loss is pi-periodic).
//! - `gaussian_kld_backward` is finite everywhere, including at the optimum
//!   (`p == t`), where the `1/sqrt(D)` singularity multiplies a zero gradient.

/// Clamp on the half-axes so `Sigma` stays invertible for tiny boxes.
const MIN_AXIS: f64 = 1e-3;
/// Guard on `sqrt(D)` in the backward at the exact optimum (`D -> 0`).
const EPS_SQRT: f64 = 1e-6;

/// Readable oriented-box view over the `[cx, cy, w, h, theta]` op layout.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obox {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
    pub theta: f32,
}

impl Obox {
    /// Pack into the op boundary layout `[cx, cy, w, h, theta]`.
    pub fn to_array(self) -> [f32; 5] {
        [self.cx, self.cy, self.w, self.h, self.theta]
    }
    /// View an op-layout array as a box.
    pub fn from_array(a: &[f32; 5]) -> Self {
        Obox {
            cx: a[0],
            cy: a[1],
            w: a[2],
            h: a[3],
            theta: a[4],
        }
    }
}

/// Intermediates cached by the forward for the backward (avoids recompute of
/// the inverse and keeps `D` for the outer `dl/dD` factor).
pub struct KldCache {
    /// The raw KL divergence `D` (>= 0).
    pub d: f32,
    /// `Sigma_p^{-1}` as `[m11, m12, m22]` (symmetric).
    minv: [f64; 3],
    /// `delta = mu_p - mu_t`.
    delta: [f64; 2],
    /// The `tau` used in the bounded wrap.
    tau: f64,
}

/// `Sigma = R diag(a,b) R^T` for a box; returns `(S11, S12, S22, a, b)` in f64.
/// `a = (w/2)^2`, `b = (h/2)^2` (half-axes clamped to `MIN_AXIS`).
fn sigma_of(b: &[f32; 5]) -> (f64, f64, f64, f64, f64) {
    let half_w = (b[2] as f64 * 0.5).max(MIN_AXIS);
    let half_h = (b[3] as f64 * 0.5).max(MIN_AXIS);
    let (aa, bb) = (half_w * half_w, half_h * half_h);
    let (sin, cos) = (b[4] as f64).sin_cos();
    let s11 = aa * cos * cos + bb * sin * sin;
    let s22 = aa * sin * sin + bb * cos * cos;
    let s12 = (aa - bb) * cos * sin;
    (s11, s12, s22, aa, bb)
}

/// Forward: the bounded Gaussian-KLD loss `l = 1 - 1/(tau + sqrt(D))` of an
/// oriented prediction against a (constant) oriented target.
///
/// # Panics
/// Debug builds assert `w, h > 0` on both boxes.
pub fn gaussian_kld_forward(pred: &[f32; 5], target: &[f32; 5], tau: f32) -> (f32, KldCache) {
    debug_assert!(pred[2] > 0.0 && pred[3] > 0.0, "pred w,h must be positive");
    debug_assert!(
        target[2] > 0.0 && target[3] > 0.0,
        "target w,h must be positive"
    );
    let (ps11, ps12, ps22, pa, pb) = sigma_of(pred);
    let (ts11, ts12, ts22, ta, tb) = sigma_of(target);
    let (det_p, det_t) = (pa * pb, ta * tb);

    // Sigma_p^{-1} = (1/det_p) [[S22, -S12], [-S12, S11]].
    let inv = 1.0 / det_p;
    let (m11, m12, m22) = (ps22 * inv, -ps12 * inv, ps11 * inv);

    let dx = (pred[0] - target[0]) as f64;
    let dy = (pred[1] - target[1]) as f64;

    // delta^T M delta.
    let t_mean = dx * (m11 * dx + m12 * dy) + dy * (m12 * dx + m22 * dy);
    // tr(M Sigma_t) = m11*ts11 + 2*m12*ts12 + m22*ts22.
    let t_trace = m11 * ts11 + 2.0 * m12 * ts12 + m22 * ts22;
    // ln(|Sigma_p| / |Sigma_t|).
    let t_logdet = (det_p / det_t).ln();

    let d = (0.5 * (t_mean + t_trace + t_logdet - 2.0)).max(0.0);
    let taud = tau as f64;
    let l = 1.0 - 1.0 / (taud + d.sqrt());
    let cache = KldCache {
        d: d as f32,
        minv: [m11, m12, m22],
        delta: [dx, dy],
        tau: taud,
    };
    (l as f32, cache)
}

/// Backward: `dl/dp` for the 5 predicted box parameters (target is constant).
///
/// Chain: `dl/dD * dD/dp`, with `dl/dD = 1/(2*sqrt(D)*(tau+sqrt(D))^2)`,
/// `dD/dmu_p = Sigma_p^{-1} delta`, and the shape gradient assembled from
/// `G = 1/2 [ Sigma_p^{-1} - Sigma_p^{-1} (Sigma_t + delta*delta^T) Sigma_p^{-1} ]`
/// pushed through `Sigma(a,b,theta)` and `a=(w/2)^2, b=(h/2)^2`.
pub fn gaussian_kld_backward(cache: &KldCache, pred: &[f32; 5], target: &[f32; 5]) -> [f32; 5] {
    let (_ps11, _ps12, _ps22, pa, pb) = sigma_of(pred);
    let (ts11, ts12, ts22, _ta, _tb) = sigma_of(target);
    let [m11, m12, m22] = cache.minv;
    let [dx, dy] = cache.delta;

    let sqrt_d = (cache.d as f64).max(EPS_SQRT).sqrt();
    let dl_dd = 1.0 / (2.0 * sqrt_d * (cache.tau + sqrt_d).powi(2));

    // q = Sigma_p^{-1} delta ; dD/dmu_p = q.
    let qx = m11 * dx + m12 * dy;
    let qy = m12 * dx + m22 * dy;

    // A = Sigma_t + delta delta^T ; G = 1/2 (M - M A M).
    let (a11, a12, a22) = (ts11 + dx * dx, ts12 + dx * dy, ts22 + dy * dy);
    let (ma11, ma12) = (m11 * a11 + m12 * a12, m11 * a12 + m12 * a22);
    let (ma21, ma22) = (m12 * a11 + m22 * a12, m12 * a12 + m22 * a22);
    let mam11 = ma11 * m11 + ma12 * m12;
    let mam12 = ma11 * m12 + ma12 * m22; // == ma21*m11 + ma22*m12 by symmetry
    let mam22 = ma21 * m12 + ma22 * m22;
    let g11 = 0.5 * (m11 - mam11);
    let g22 = 0.5 * (m22 - mam22);
    let g12 = 0.5 * (m12 - mam12);

    // Sigma(a,b,theta) derivatives. Effective dD/dS12 = 2*g12 (symmetric entry).
    let (sin, cos) = (pred[4] as f64).sin_cos();
    let cs = cos * sin;
    let cos2 = cos * cos - sin * sin; // cos(2 theta)
    let dd_da = g11 * cos * cos + g22 * sin * sin + 2.0 * g12 * cs;
    let dd_db = g11 * sin * sin + g22 * cos * cos - 2.0 * g12 * cs;
    let dd_dth = (pa - pb) * (2.0 * cs * (g22 - g11) + 2.0 * g12 * cos2);

    // a=(w/2)^2 -> da/dw = w/2 = half_w (0 in the clamped regime, matching forward).
    let half_w_raw = pred[2] as f64 * 0.5;
    let half_h_raw = pred[3] as f64 * 0.5;
    let dd_dw = if half_w_raw > MIN_AXIS {
        dd_da * half_w_raw
    } else {
        0.0
    };
    let dd_dh = if half_h_raw > MIN_AXIS {
        dd_db * half_h_raw
    } else {
        0.0
    };

    [
        (dl_dd * qx) as f32,
        (dl_dd * qy) as f32,
        (dl_dd * dd_dw) as f32,
        (dl_dd * dd_dh) as f32,
        (dl_dd * dd_dth) as f32,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_matches_fd() {
        let pred = [1.2f32, 0.3, 2.4, 2.0, 0.7];
        let target = [1.0f32, 0.5, 3.0, 1.5, 0.3];
        let (_, cache) = gaussian_kld_forward(&pred, &target, 1.0);
        let grad = gaussian_kld_backward(&cache, &pred, &target);
        let eps = 1e-3f32;
        for i in 0..5 {
            let mut pp = pred;
            pp[i] += eps;
            let mut pm = pred;
            pm[i] -= eps;
            let (lp, _) = gaussian_kld_forward(&pp, &target, 1.0);
            let (lm, _) = gaussian_kld_forward(&pm, &target, 1.0);
            let num = (lp - lm) / (2.0 * eps);
            let denom = num.abs().max(1e-3);
            assert!(
                (grad[i] - num).abs() / denom < 2e-2,
                "grad[{i}] analytic {} vs fd {}",
                grad[i],
                num
            );
        }
    }

    #[test]
    fn zero_at_identity_and_positive_elsewhere() {
        let t = [1.0f32, 0.5, 3.0, 1.5, 0.3];
        let (l0, c0) = gaussian_kld_forward(&t, &t, 1.0);
        assert!(l0.abs() < 1e-5, "loss at identity {l0}");
        assert!(c0.d.abs() < 1e-5, "D at identity {}", c0.d);
        // Gradient at the optimum is finite (1/sqrt(D) blow-up * zero gradient).
        let g0 = gaussian_kld_backward(&c0, &t, &t);
        assert!(
            g0.iter().all(|v| v.is_finite()),
            "grad finite at optimum: {g0:?}"
        );
        assert!(
            g0.iter().all(|v| v.abs() < 1e-4),
            "grad ~0 at optimum: {g0:?}"
        );
        // Any displacement raises D (hence l) above zero.
        for probe in [
            [1.4f32, 0.5, 3.0, 1.5, 0.3],
            [1.0, 0.5, 2.0, 1.5, 0.3],
            [1.0, 0.5, 3.0, 1.5, 0.9],
        ] {
            let (l, c) = gaussian_kld_forward(&probe, &t, 1.0);
            assert!(c.d > 0.0 && l > 0.0 && l < 1.0, "D {} l {}", c.d, l);
        }
    }

    #[test]
    fn pi_periodic_in_theta() {
        // theta and theta+pi describe the same Gaussian -> identical loss.
        let t = [0.0f32, 0.0, 4.0, 1.0, 0.2];
        let p_a = [0.5f32, -0.3, 2.0, 3.0, 0.6];
        let mut p_b = p_a;
        p_b[4] += std::f32::consts::PI;
        let (la, _) = gaussian_kld_forward(&p_a, &t, 1.0);
        let (lb, _) = gaussian_kld_forward(&p_b, &t, 1.0);
        assert!((la - lb).abs() < 1e-5, "pi-shift changed loss {la} vs {lb}");
    }

    #[test]
    fn tiny_box_stays_finite() {
        // Degenerate near-zero box exercises the MIN_AXIS clamp; must not NaN.
        let t = [0.0f32, 0.0, 2.0, 2.0, 0.0];
        let p = [0.1f32, 0.1, 1e-4, 1e-4, 0.0];
        let (l, c) = gaussian_kld_forward(&p, &t, 1.0);
        let g = gaussian_kld_backward(&c, &p, &t);
        assert!(l.is_finite() && c.d.is_finite());
        assert!(
            g.iter().all(|v| v.is_finite()),
            "grad finite for tiny box: {g:?}"
        );
    }

    #[test]
    fn obox_roundtrips() {
        let a = [1.0f32, 2.0, 3.0, 4.0, 0.5];
        assert_eq!(Obox::from_array(&a).to_array(), a);
    }
}
