//! Evolvent (incremental / online) learning — the closed-form per-sample update
//! that does NOT require a backward sweep, the counterpoint to slow batch
//! backprop. `EvolventHead` is an exact **forgetting recursive-least-squares**
//! (RLS) readout over a fixed feature basis: it consumes one `(phi, y)` at a
//! time, updates in closed form via Sherman–Morrison in `O(d^2)`, and — unlike
//! SGD — reaches the (regularised) least-squares optimum in one pass and TRACKS
//! a drifting target through the forgetting factor `lambda`.
//!
//! Verified (the closed-form analogue of FD-verification): on stationary data the
//! recursion converges to the batch ridge normal-equations solution
//! `w = (Phi^T Phi + ridge·I)^{-1} Phi^T y` (see `online::tests`).
//!
//! Contract:
//! - Preconditions: `phi.len() == d`; `ridge > 0`; `lambda in (0, 1]`.
//! - Postconditions: after `update`, `w` is the forgetting-RLS estimate; `predict`
//!   returns `phi·w`.

/// Exact forgetting-RLS readout: `w in R^d`, precision `P = (Phi^T Phi + ridge I)^{-1}`
/// maintained incrementally. `lambda == 1` is standard RLS (stationary);
/// `lambda < 1` forgets old data to track drift.
#[derive(Clone, Debug)]
pub struct EvolventHead {
    /// Weight vector `(d,)`.
    pub w: Vec<f32>,
    /// Inverse-covariance / precision matrix `(d*d,)` row-major.
    p: Vec<f32>,
    d: usize,
    lambda: f32,
    /// Windup guard: max allowed `trace(P)`. With `lambda < 1`, `P` inflates by
    /// `1/lambda` each step in un-excited directions (covariance windup) and can
    /// diverge; capping the trace bounds every eigenvalue → bounds the gain →
    /// prevents blow-up. Bounded by default (see `new`).
    p_trace_max: f32,
}

impl EvolventHead {
    /// New head over `d` features. `ridge > 0` sets the prior precision
    /// (`P0 = (1/ridge) I`); `lambda in (0,1]` is the forgetting factor. The
    /// covariance-windup guard is set to a generous finite default
    /// (`1e4 · trace(P0)`), so forgetting is safe out of the box; tune with
    /// [`EvolventHead::with_trace_cap`].
    ///
    /// # Panics
    /// If `ridge <= 0` or `lambda` is outside `(0, 1]`.
    pub fn new(d: usize, ridge: f32, lambda: f32) -> Self {
        assert!(ridge > 0.0, "ridge must be > 0");
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0,1]");
        let mut p = vec![0.0f32; d * d];
        for i in 0..d {
            p[i * d + i] = 1.0 / ridge;
        }
        EvolventHead {
            w: vec![0.0; d],
            p,
            d,
            lambda,
            p_trace_max: 1e4 * d as f32 / ridge,
        }
    }

    /// Override the covariance-windup trace cap (larger = faster adaptation but
    /// weaker windup protection).
    pub fn with_trace_cap(mut self, cap: f32) -> Self {
        assert!(cap > 0.0, "trace cap must be > 0");
        self.p_trace_max = cap;
        self
    }

    /// Prediction `phi · w`.
    ///
    /// # Panics
    /// If `phi.len() != d`.
    pub fn predict(&self, phi: &[f32]) -> f32 {
        assert_eq!(phi.len(), self.d);
        phi.iter().zip(&self.w).map(|(&a, &b)| a * b).sum()
    }

    /// One evolvent (forgetting-RLS) update from a single sample — closed-form,
    /// `O(d^2)`, no backward sweep. Returns the pre-update prediction error
    /// `y - phi·w` (the prequential residual).
    ///
    /// Recursion (with forgetting `lambda`):
    /// ```text
    ///   Pphi = P phi
    ///   g    = Pphi / (lambda + phi^T Pphi)      (Kalman gain)
    ///   err  = y - phi^T w
    ///   w   += g * err
    ///   P    = (P - g (Pphi)^T) / lambda
    /// ```
    ///
    /// # Panics
    /// If `phi.len() != d`.
    pub fn update(&mut self, phi: &[f32], y: f32) -> f32 {
        let d = self.d;
        assert_eq!(phi.len(), d);
        // Pphi = P phi
        let mut pphi = vec![0.0f32; d];
        for (i, pphi_i) in pphi.iter_mut().enumerate() {
            let row = &self.p[i * d..i * d + d];
            *pphi_i = row.iter().zip(phi).map(|(&a, &b)| a * b).sum();
        }
        let denom = self.lambda + phi.iter().zip(&pphi).map(|(&a, &b)| a * b).sum::<f32>();
        let inv_denom = 1.0 / denom;
        let err = y - self.predict(phi);
        // w += g * err, with g = Pphi / denom
        for (wi, &pp) in self.w.iter_mut().zip(&pphi) {
            *wi += pp * inv_denom * err;
        }
        // P = (P - g Pphi^T) / lambda = (P - (Pphi Pphi^T)/denom) / lambda
        let inv_lambda = 1.0 / self.lambda;
        let mut trace = 0.0f32;
        for i in 0..d {
            let gi = pphi[i] * inv_denom;
            let row = &mut self.p[i * d..i * d + d];
            for (j, pij) in row.iter_mut().enumerate() {
                *pij = (*pij - gi * pphi[j]) * inv_lambda;
            }
            trace += self.p[i * d + i];
        }
        // covariance-windup guard: scale P down if its trace exceeds the cap
        // (bounds every eigenvalue, hence the gain). Only bites when lambda < 1.
        if trace > self.p_trace_max {
            let s = self.p_trace_max / trace;
            for v in self.p.iter_mut() {
                *v *= s;
            }
        }
        err
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Closed-form verification: on STATIONARY data, forgetting-RLS with
    /// lambda=1 converges to the batch ridge normal-equations solution.
    #[test]
    fn converges_to_batch_ridge() {
        let (d, n, ridge) = (5usize, 400usize, 1.0f32);
        // deterministic stream
        let mut xs: u64 = 7;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let w_true: Vec<f32> = (0..d).map(|_| nx()).collect();
        let phis: Vec<Vec<f32>> = (0..n).map(|_| (0..d).map(|_| nx()).collect()).collect();
        let ys: Vec<f32> = phis
            .iter()
            .map(|p| p.iter().zip(&w_true).map(|(&a, &b)| a * b).sum::<f32>() + 0.05 * nx())
            .collect();

        // online RLS (lambda = 1)
        let mut head = EvolventHead::new(d, ridge, 1.0);
        for (p, &y) in phis.iter().zip(&ys) {
            head.update(p, y);
        }

        // batch ridge: (A + ridge I) w = b, solved by Gaussian elimination
        let mut a = vec![0.0f32; d * d];
        let mut b = vec![0.0f32; d];
        for (p, &y) in phis.iter().zip(&ys) {
            for i in 0..d {
                b[i] += p[i] * y;
                for j in 0..d {
                    a[i * d + j] += p[i] * p[j];
                }
            }
        }
        for i in 0..d {
            a[i * d + i] += ridge;
        }
        let w_batch = solve(&mut a, &mut b, d);
        for (i, (&hw, &wb)) in head.w.iter().zip(&w_batch).enumerate() {
            assert!((hw - wb).abs() < 1e-3, "coef {i}: rls {hw} vs batch {wb}");
        }
    }

    /// The evolvent head TRACKS an abrupt drift: after the target flips, error
    /// recovers within a bounded number of samples (lambda<1), which a frozen
    /// estimate would not.
    #[test]
    fn tracks_a_drift() {
        let d = 4usize;
        let mut xs: u64 = 3;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let mut head = EvolventHead::new(d, 1.0, 0.95);
        let w1: Vec<f32> = (0..d).map(|_| nx()).collect();
        let w2: Vec<f32> = w1.iter().map(|&v| -v).collect(); // flipped target
        let teach = |p: &[f32], w: &[f32]| p.iter().zip(w).map(|(&a, &b)| a * b).sum::<f32>();
        for _ in 0..300 {
            let p: Vec<f32> = (0..d).map(|_| nx()).collect();
            head.update(&p, teach(&p, &w1));
        }
        for _ in 0..300 {
            let p: Vec<f32> = (0..d).map(|_| nx()).collect();
            head.update(&p, teach(&p, &w2));
        }
        // after retraining on w2, error on fresh w2 samples is small
        let mut e = 0.0f32;
        for _ in 0..50 {
            let p: Vec<f32> = (0..d).map(|_| nx()).collect();
            e += (head.predict(&p) - teach(&p, &w2)).powi(2);
        }
        assert!(e / 50.0 < 1e-2, "did not track the drift: mse {}", e / 50.0);
    }

    /// Regression (bug F-EVO-1, 2026-07-15): forgetting-RLS with many features
    /// must NOT diverge from covariance windup. With lambda<1 and d>>1 and
    /// poorly-excited directions, the un-guarded recursion blew up (RMSE ~1e4);
    /// the trace-cap guard must keep w and predictions finite and bounded.
    #[test]
    fn windup_guard_keeps_it_bounded() {
        let d = 200usize;
        let mut xs: u64 = 11;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let mut head = EvolventHead::new(d, 1.0, 0.99);
        // sparse, poorly-exciting stream (only a few features active per sample)
        for _ in 0..5000 {
            let mut phi = vec![0.0f32; d];
            for _ in 0..5 {
                phi[((nx() + 0.5) * d as f32) as usize % d] = nx();
            }
            let y = nx();
            head.update(&phi, y);
        }
        assert!(head.w.iter().all(|v| v.is_finite()), "weights diverged");
        let wnorm: f32 = head.w.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(wnorm < 1e3, "weights blew up: |w| = {wnorm}");
    }

    /// Tiny Gaussian-elimination solver for the test's batch reference.
    fn solve(a: &mut [f32], b: &mut [f32], d: usize) -> Vec<f32> {
        for col in 0..d {
            let mut piv = col;
            for r in col + 1..d {
                if a[r * d + col].abs() > a[piv * d + col].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for j in 0..d {
                    a.swap(col * d + j, piv * d + j);
                }
                b.swap(col, piv);
            }
            let diag = a[col * d + col];
            for r in 0..d {
                if r == col {
                    continue;
                }
                let f = a[r * d + col] / diag;
                for j in 0..d {
                    a[r * d + j] -= f * a[col * d + j];
                }
                b[r] -= f * b[col];
            }
        }
        (0..d).map(|i| b[i] / a[i * d + i]).collect()
    }
}
