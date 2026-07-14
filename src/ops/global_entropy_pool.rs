//! `global_entropy_pool` — a **rotation-invariant global pool** for the top of
//! the Neocognitron stack, no autograd. It reduces each channel of a response
//! map to three rotation-invariant scalars via the crate's global-pooling idiom
//! (`structural_pool` = "mean/std/max/**entropy** pooling"), specialised to
//! recover *spatial arrangement* — which a permutation-invariant value pool
//! (mean/std/entropy of the value multiset) cannot see.
//!
//! Key idea: a global input rotation permutes spatial locations but leaves the
//! **response-weighted spatial covariance** rotationally-transformed, so its
//! rotation *invariants* — trace `T` and the eigenvalue **entropy** `Hs` — are
//! global-rotation-invariant *and* arrangement-sensitive. An elongated bar has a
//! rank-1 covariance (`Hs → 0`); an L-corner spreads in two directions
//! (`Hs → ln 2`). This is the "entropy global pooling" feed the N3 held-out gap
//! asked for: O(H·W) per channel, closed-form 2×2 eigen, no `|G|` steering.
//!
//! Per channel `c` (weights `w_i = resp[c,i]²`, coords normalised to `[0,1]`):
//! ```text
//!   M = Σ w_i,  mean = M/N,  (cx,cy) = weighted centroid
//!   a = Var_x,  d = Var_y,  b = Cov_xy            (weighted, centred)
//!   T = a+d,  Dt = a·d − b²,  q = Dt/T² ∈ [0, 1/4]
//!   disc = √(1−4q),  e = (1±disc)/2,  Hs = −Σ e·ln e
//!   feat[c] = [mean, T, Hs]                        (all rotation-invariant)
//! ```
//!
//! # Backward (FD-verified)
//! `∂a/∂w_i = ((x_i−cx)² − a)/M` (and `d`, `b` analogues); `∂Hs/∂q =
//! ln(e1/e2)/disc`; `grad_resp[c,i] = 2·resp[c,i]·∂L/∂w_i`. Degenerate channels
//! (`M<ε` or `T<ε`) contribute zero feature and zero gradient.
//!
//! **No novelty claimed** — second-moment shape descriptors and covariance
//! eigen-entropy are classical; the Nagare part is the closed-form no-autograd
//! op and its role as the group-invariant top of the rotor Neocognitron.

const EPS: f32 = 1e-6;

/// Forward output: `feat` `(K·3)` = `[mean, trace, entropy]` per channel, plus
/// the per-channel moment cache the backward reuses.
pub struct GlobalEntropyPoolOut {
    /// Rotation-invariant features, flat `(k·3)`: `[mean, T, Hs]` per channel.
    pub feat: Vec<f32>,
    /// Per-channel `(M, cx, cy, a, b, d, T, Dt)` cache, flat `(k·8)`.
    cache: Vec<f32>,
    k: usize,
    h: usize,
    w: usize,
}

/// Number of features per channel.
pub const FEATS_PER_CHANNEL: usize = 3;

impl GlobalEntropyPoolOut {
    /// Rotation-**equivariant** pose readout: the principal-axis angle (mod π) of
    /// channel `c`'s response-weighted covariance, `½·atan2(2b, a−d)`. The
    /// invariant entropy `Hs` recognises the object; this angle localises its
    /// pose. Returns `None` for a degenerate (near-blank) channel.
    ///
    /// # Panics
    /// If `c >= k`.
    pub fn principal_angle(&self, c: usize) -> Option<f32> {
        assert!(c < self.k, "channel out of range");
        let cc = &self.cache[c * 8..c * 8 + 8];
        let (m, a, b, d) = (cc[0], cc[3], cc[4], cc[5]);
        if m < EPS {
            return None;
        }
        Some(0.5 * (2.0 * b).atan2(a - d))
    }

    /// Total response mass of channel `c` (for picking the dominant channel).
    pub fn mass(&self, c: usize) -> f32 {
        self.cache[c * 8]
    }
}

#[inline]
fn norm_coord(i: usize, n: usize) -> f32 {
    if n <= 1 {
        0.0
    } else {
        i as f32 / (n - 1) as f32
    }
}

/// Eigenvalue-distribution entropy of a 2×2 covariance with trace `t`, det `dt`.
/// Returns `(Hs, disc)` where `disc = √(1−4q)`, `q = dt/t²`.
#[inline]
fn eigen_entropy(t: f32, dt: f32) -> (f32, f32) {
    let q = (dt / (t * t)).clamp(0.0, 0.25);
    let disc = (1.0 - 4.0 * q).max(0.0).sqrt();
    let e1 = 0.5 * (1.0 + disc);
    let e2 = 0.5 * (1.0 - disc);
    let term = |e: f32| if e > EPS { -e * e.ln() } else { 0.0 };
    (term(e1) + term(e2), disc)
}

/// `global_entropy_pool` forward. `resp` is `(K, H, W)` flat.
///
/// # Preconditions
/// `resp.len() == k*h*w`.
///
/// # Postconditions
/// `feat.len() == k*3`; every feature is invariant to a global rotation of the
/// spatial response map (up to interpolation), by construction.
///
/// # Panics
/// If `resp.len() != k*h*w`.
pub fn global_entropy_pool_forward(
    resp: &[f32],
    k: usize,
    h: usize,
    w: usize,
) -> GlobalEntropyPoolOut {
    assert_eq!(resp.len(), k * h * w, "resp must be K*H*W");
    let n = h * w;
    let mut feat = vec![0.0f32; k * FEATS_PER_CHANNEL];
    let mut cache = vec![0.0f32; k * 8];
    for c in 0..k {
        let (mut m, mut sx, mut sy, mut sxx, mut syy, mut sxy) = (0.0f32, 0.0, 0.0, 0.0, 0.0, 0.0);
        for r in 0..h {
            for col in 0..w {
                let wi = {
                    let v = resp[c * n + r * w + col];
                    v * v
                };
                let (x, y) = (norm_coord(col, w), norm_coord(r, h));
                m += wi;
                sx += wi * x;
                sy += wi * y;
                sxx += wi * x * x;
                syy += wi * y * y;
                sxy += wi * x * y;
            }
        }
        if m < EPS {
            // blank channel → zero feature, zero grad (cache stays zeros).
            continue;
        }
        let (cx, cy) = (sx / m, sy / m);
        let a = sxx / m - cx * cx;
        let d = syy / m - cy * cy;
        let b = sxy / m - cx * cy;
        let t = (a + d).max(EPS);
        let dt = (a * d - b * b).max(0.0);
        let (hs, _disc) = eigen_entropy(t, dt);
        feat[c * 3] = m / n as f32;
        feat[c * 3 + 1] = t;
        feat[c * 3 + 2] = hs;
        let cc = &mut cache[c * 8..c * 8 + 8];
        cc.copy_from_slice(&[m, cx, cy, a, b, d, t, dt]);
    }
    GlobalEntropyPoolOut {
        feat,
        cache,
        k,
        h,
        w,
    }
}

/// `global_entropy_pool` backward. `grad_feat` is `(K·3)`; returns
/// `grad_resp (K,H,W)`.
///
/// # Panics
/// If `grad_feat.len() != k*3`.
pub fn global_entropy_pool_backward(
    out: &GlobalEntropyPoolOut,
    resp: &[f32],
    grad_feat: &[f32],
) -> Vec<f32> {
    let (k, h, w) = (out.k, out.h, out.w);
    assert_eq!(
        grad_feat.len(),
        k * FEATS_PER_CHANNEL,
        "grad_feat must be K*3"
    );
    let n = h * w;
    let mut grad = vec![0.0f32; k * n];
    for c in 0..k {
        let cc = &out.cache[c * 8..c * 8 + 8];
        let (m, cx, cy, a, b, d, t, dt) = (cc[0], cc[1], cc[2], cc[3], cc[4], cc[5], cc[6], cc[7]);
        if m < EPS {
            continue;
        }
        let (g_mean, g_t, g_hs) = (grad_feat[c * 3], grad_feat[c * 3 + 1], grad_feat[c * 3 + 2]);
        // ∂Hs/∂q = ln(e1/e2)/disc (limit → 2 as disc → 0).
        let disc = (1.0 - 4.0 * (dt / (t * t)).clamp(0.0, 0.25))
            .max(0.0)
            .sqrt();
        let dhs_dq = if disc < 1e-4 {
            2.0
        } else {
            let (e1, e2) = (0.5 * (1.0 + disc), 0.5 * (1.0 - disc));
            (e1 / e2.max(EPS)).ln() / disc
        };
        // q = Dt/T²; ∂q/∂{a,d,b} via ∂Dt/∂· and ∂T/∂·.
        let inv_t2 = 1.0 / (t * t);
        let dq_da = (d - 2.0 * dt / t) * inv_t2;
        let dq_dd = (a - 2.0 * dt / t) * inv_t2;
        let dq_db = (-2.0 * b) * inv_t2;
        // ∂L/∂{a,d,b} through T (trace) and Hs.
        let dl_da = g_t + g_hs * dhs_dq * dq_da;
        let dl_dd = g_t + g_hs * dhs_dq * dq_dd;
        let dl_db = g_hs * dhs_dq * dq_db;
        let inv_m = 1.0 / m;
        for r in 0..h {
            for col in 0..w {
                let (x, y) = (norm_coord(col, w), norm_coord(r, h));
                let (dx, dy) = (x - cx, y - cy);
                // ∂a/∂w_i = ((x−cx)²−a)/M etc.
                let dl_dw = g_mean / n as f32
                    + dl_da * (dx * dx - a) * inv_m
                    + dl_dd * (dy * dy - d) * inv_m
                    + dl_db * (dx * dy - b) * inv_m;
                grad[c * n + r * w + col] = 2.0 * resp[c * n + r * w + col] * dl_dw;
            }
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::PI;

    fn rot_map(k: usize, g: usize, theta: f32, seed: u64) -> Vec<f32> {
        // an L-shape (two perpendicular arms) rotated by theta, as a K=1 map.
        let mut xs = seed;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let cc = (g - 1) as f32 / 2.0;
        let (ct, st) = (theta.cos(), theta.sin());
        let mut m = vec![0.0f32; k * g * g];
        for r in 0..g {
            for col in 0..g {
                let (dx, dy) = (col as f32 - cc, r as f32 - cc);
                // rotate into the shape's frame
                let (u, v) = (dx * ct + dy * st, -dx * st + dy * ct);
                let on = (v.abs() < 1.2 && (0.0..7.0).contains(&u))
                    || (u.abs() < 1.2 && (0.0..7.0).contains(&v));
                for kk in 0..k {
                    m[kk * g * g + r * g + col] = if on { 1.0 } else { 0.0 } + 0.02 * nx();
                }
            }
        }
        m
    }

    #[test]
    fn entropy_is_rotation_invariant() {
        // The eigen-entropy feature of a rotated L should be ~constant.
        let g = 20;
        let hs: Vec<f32> = [0.0f32, 30.0, 60.0, 90.0]
            .iter()
            .map(|deg| {
                let m = rot_map(1, g, deg * PI / 180.0, 7);
                global_entropy_pool_forward(&m, 1, g, g).feat[2]
            })
            .collect();
        let (lo, hi) = (
            hs.iter().cloned().fold(f32::MAX, f32::min),
            hs.iter().cloned().fold(f32::MIN, f32::max),
        );
        assert!(hi - lo < 0.05, "entropy varies with rotation: {hs:?}");
    }

    #[test]
    fn bar_has_lower_entropy_than_corner() {
        let g = 20;
        // straight bar (rank-1 covariance) vs L-corner (two directions).
        let mut bar = vec![0.0f32; g * g];
        let cc = (g - 1) / 2;
        for col in 4..16 {
            bar[cc * g + col] = 1.0;
        }
        let corner = rot_map(1, g, 0.0, 1);
        let hb = global_entropy_pool_forward(&bar, 1, g, g).feat[2];
        let hc = global_entropy_pool_forward(&corner, 1, g, g).feat[2];
        assert!(hb < hc, "bar entropy {hb} should be < corner entropy {hc}");
    }

    #[test]
    fn principal_angle_tracks_bar_orientation() {
        // A bar rendered at angle θ: the equivariant pose readout ≈ θ (mod π).
        let g = 24;
        for deg in [0.0f32, 30.0, 60.0, 120.0] {
            let th = deg * PI / 180.0;
            let cc = (g - 1) as f32 / 2.0;
            let (ct, st) = (th.cos(), th.sin());
            let mut m = vec![0.0f32; g * g];
            for r in 0..g {
                for col in 0..g {
                    let (dx, dy) = (col as f32 - cc, r as f32 - cc);
                    let (along, perp) = (dx * ct + dy * st, -dx * st + dy * ct);
                    if along.abs() <= 8.0 && perp.abs() <= 1.0 {
                        m[r * g + col] = 1.0;
                    }
                }
            }
            let ang = global_entropy_pool_forward(&m, 1, g, g)
                .principal_angle(0)
                .unwrap();
            // compare mod π (bar direction is a line, defined mod π).
            let diff = (ang - th).rem_euclid(PI);
            let err = diff.min(PI - diff);
            assert!(err < 0.15, "θ={deg}°: predicted {ang}, err {err}");
        }
    }

    #[test]
    fn backward_matches_fd() {
        let (k, g) = (2usize, 8usize);
        let mut xs: u64 = 33;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.3
        };
        let resp: Vec<f32> = (0..k * g * g).map(|_| nx()).collect();
        let gw: Vec<f32> = (0..k * FEATS_PER_CHANNEL).map(|_| nx()).collect();
        let out = global_entropy_pool_forward(&resp, k, g, g);
        let grad = global_entropy_pool_backward(&out, &resp, &gw);
        let loss = |r: &[f32]| -> f32 {
            global_entropy_pool_forward(r, k, g, g)
                .feat
                .iter()
                .zip(&gw)
                .map(|(f, w)| f * w)
                .sum()
        };
        let eps = 1e-3f32;
        for &i in &[0usize, 5, 30, 64, 100, 120] {
            let mut rp = resp.clone();
            rp[i] += eps;
            let mut rm = resp.clone();
            rm[i] -= eps;
            let num = (loss(&rp) - loss(&rm)) / (2.0 * eps);
            assert!(
                (grad[i] - num).abs() < 2e-2 + 3e-2 * num.abs(),
                "grad[{i}] ana {} vs fd {num}",
                grad[i]
            );
        }
    }
}
