//! Evolvent (incremental / online) learning — the closed-form per-sample update
//! that does NOT require a backward sweep, the counterpoint to slow batch
//! backprop. `EvolventHead` is an exact **forgetting recursive-least-squares**
//! (RLS) readout with `C` outputs: `C=1` for regression, `C=n_classes` one-hot
//! (`argmax`) for classification. It consumes one `(phi, y)` at a time, updates
//! in closed form via Sherman–Morrison, and — unlike SGD — reaches the
//! (regularised) least-squares optimum in one pass and TRACKS a drifting target
//! through the forgetting factor `lambda`.
//!
//! **Multi-output is cheap:** the precision matrix `P` (`d×d`) is SHARED across
//! all `C` outputs (they share the feature covariance), so a step costs
//! `O(d^2 + d·C)`, not `C·O(d^2)`.
//!
//! Verified (the closed-form analogue of FD-verification): on stationary data the
//! recursion converges to the batch ridge normal-equations solution
//! `w = (Phi^T Phi + ridge·I)^{-1} Phi^T y` (see `online::tests`).

/// Exact forgetting-RLS readout with `C` outputs. `w` is `(C, d)` row-major (row
/// `k` = output `k`'s weights); precision `P = (Phi^T Phi + ridge I)^{-1}` is
/// maintained incrementally and shared across outputs. `lambda == 1` is standard
/// RLS (stationary); `lambda < 1` forgets old data to track drift.
#[derive(Clone, Debug)]
pub struct EvolventHead {
    /// Weights `(C, d)` row-major.
    pub w: Vec<f32>,
    /// Inverse-covariance / precision matrix `(d*d,)` row-major.
    p: Vec<f32>,
    d: usize,
    c: usize,
    lambda: f32,
    /// Windup guard: max allowed `trace(P)`. With `lambda < 1`, `P` inflates by
    /// `1/lambda` each step in un-excited directions (covariance windup) and can
    /// diverge; capping the trace bounds every eigenvalue → bounds the gain →
    /// prevents blow-up. Bounded by default (see `new`).
    p_trace_max: f32,
}

impl EvolventHead {
    /// New head: `d` features, `c` outputs (`c=1` regression, `c=n_classes`
    /// one-hot classification). `ridge > 0` sets the prior precision
    /// (`P0 = (1/ridge) I`); `lambda in (0,1]` is the forgetting factor. The
    /// covariance-windup guard defaults to a generous finite `1e4 · trace(P0)`;
    /// tune with [`EvolventHead::with_trace_cap`].
    ///
    /// # Panics
    /// If `c == 0`, `ridge <= 0`, or `lambda` is outside `(0, 1]`.
    pub fn new(d: usize, c: usize, ridge: f32, lambda: f32) -> Self {
        assert!(c >= 1, "need at least one output");
        assert!(ridge > 0.0, "ridge must be > 0");
        assert!(lambda > 0.0 && lambda <= 1.0, "lambda must be in (0,1]");
        let mut p = vec![0.0f32; d * d];
        for i in 0..d {
            p[i * d + i] = 1.0 / ridge;
        }
        EvolventHead {
            w: vec![0.0; c * d],
            p,
            d,
            c,
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

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.c
    }

    /// Per-output predictions `phi · w_k`, length `C`.
    ///
    /// # Panics
    /// If `phi.len() != d`.
    pub fn predict(&self, phi: &[f32]) -> Vec<f32> {
        assert_eq!(phi.len(), self.d);
        (0..self.c)
            .map(|k| {
                let wk = &self.w[k * self.d..k * self.d + self.d];
                phi.iter().zip(wk).map(|(&a, &b)| a * b).sum()
            })
            .collect()
    }

    /// `argmax_k predict(phi)` — the classification decision.
    pub fn predict_class(&self, phi: &[f32]) -> usize {
        let pred = self.predict(phi);
        (0..self.c)
            .max_by(|&a, &b| pred[a].partial_cmp(&pred[b]).unwrap())
            .unwrap_or(0)
    }

    /// One evolvent (forgetting-RLS) update from a single sample — closed-form,
    /// `O(d^2 + d·C)`, no backward sweep. Returns the per-output pre-update
    /// residuals `y - phi·w` (the prequential errors).
    ///
    /// Recursion (with forgetting `lambda`), `P` shared across outputs:
    /// ```text
    ///   Pphi = P phi;   g = Pphi / (lambda + phi^T Pphi)
    ///   for each output k:  w_k += g * (y_k - phi^T w_k)
    ///   P = (P - g (Pphi)^T) / lambda
    /// ```
    ///
    /// # Panics
    /// If `phi.len() != d` or `y.len() != C`.
    pub fn update(&mut self, phi: &[f32], y: &[f32]) -> Vec<f32> {
        let (d, c) = (self.d, self.c);
        assert_eq!(phi.len(), d);
        assert_eq!(y.len(), c);
        // Pphi = P phi
        let mut pphi = vec![0.0f32; d];
        for (i, pphi_i) in pphi.iter_mut().enumerate() {
            let row = &self.p[i * d..i * d + d];
            *pphi_i = row.iter().zip(phi).map(|(&a, &b)| a * b).sum();
        }
        let denom = self.lambda + phi.iter().zip(&pphi).map(|(&a, &b)| a * b).sum::<f32>();
        let inv_denom = 1.0 / denom;
        // per-output residual + weight update (shared gain g = Pphi/denom)
        let pred = self.predict(phi);
        let residuals: Vec<f32> = (0..c).map(|k| y[k] - pred[k]).collect();
        for (k, &res) in residuals.iter().enumerate() {
            let ge = inv_denom * res;
            let wk = &mut self.w[k * d..k * d + d];
            for (wi, &pp) in wk.iter_mut().zip(&pphi) {
                *wi += pp * ge;
            }
        }
        // P = (P - g Pphi^T) / lambda   (shared)
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
        // covariance-windup guard (F-EVO-1): cap trace(P) → bound the gain.
        if trace > self.p_trace_max {
            let s = self.p_trace_max / trace;
            for v in self.p.iter_mut() {
                *v *= s;
            }
        }
        residuals
    }
}

/// Block-structured (hyperedge-clique) precision evolvent — the alternative to
/// the dense O(d^2) pairwise precision. Features are grouped into contiguous
/// blocks (one per hyperedge); the precision is kept **block-diagonal** (a small
/// `b_e x b_e` matrix per block) with a SHARED residual and denominator, so a
/// step costs `O(sum b_e^2)` = `O(d * w)` for width `w`, not `O(d^2)`.
///
/// This is exact RLS when the true precision is block-diagonal (feature-disjoint
/// hyperedges over independent, mean-zero inputs); it is an approximation that
/// drops cross-block (separator) coupling when hyperedges overlap. With a single
/// block it reduces to the dense [`EvolventHead`] exactly (see tests).
#[derive(Clone, Debug)]
pub struct BlockEvolventHead {
    /// Full weight vector `(d,)`.
    pub w: Vec<f32>,
    /// `(offset, size)` per block (contiguous over the feature vector).
    blocks: Vec<(usize, usize)>,
    /// Per-block precision `b_e x b_e` (row-major).
    p: Vec<Vec<f32>>,
    lambda: f32,
    caps: Vec<f32>,
}

impl BlockEvolventHead {
    /// New head over contiguous blocks of the given sizes. `ridge > 0`, `lambda in (0,1]`.
    ///
    /// # Panics
    /// If any size is 0, `ridge <= 0`, or `lambda` outside `(0,1]`.
    pub fn new(block_sizes: &[usize], ridge: f32, lambda: f32) -> Self {
        assert!(ridge > 0.0 && lambda > 0.0 && lambda <= 1.0);
        let mut blocks = Vec::with_capacity(block_sizes.len());
        let mut p = Vec::with_capacity(block_sizes.len());
        let mut caps = Vec::with_capacity(block_sizes.len());
        let mut off = 0;
        for &b in block_sizes {
            assert!(b >= 1, "block size must be >= 1");
            blocks.push((off, b));
            let mut pb = vec![0.0f32; b * b];
            for i in 0..b {
                pb[i * b + i] = 1.0 / ridge;
            }
            p.push(pb);
            caps.push(1e4 * b as f32 / ridge);
            off += b;
        }
        BlockEvolventHead {
            w: vec![0.0; off],
            blocks,
            p,
            lambda,
            caps,
        }
    }

    /// Total feature dimension.
    pub fn dim(&self) -> usize {
        self.w.len()
    }

    /// Nonzeros the precision stores — `sum b_e^2` (vs `d^2` dense).
    pub fn precision_nnz(&self) -> usize {
        self.blocks.iter().map(|&(_, b)| b * b).sum()
    }

    /// Prediction `sum_e phi_e . w_e`.
    pub fn predict(&self, phi: &[f32]) -> f32 {
        assert_eq!(phi.len(), self.w.len());
        phi.iter().zip(&self.w).map(|(&a, &b)| a * b).sum()
    }

    /// One block-diagonal RLS update — shared residual/denominator across blocks,
    /// per-block precision. Returns the prequential residual.
    ///
    /// # Panics
    /// If `phi.len() != d`.
    pub fn update(&mut self, phi: &[f32], y: f32) -> f32 {
        assert_eq!(phi.len(), self.w.len());
        // per-block Pphi and shared denom = lambda + sum_e phi_e^T P_e phi_e
        let mut pphi: Vec<Vec<f32>> = Vec::with_capacity(self.blocks.len());
        let mut denom = self.lambda;
        for (bi, &(off, b)) in self.blocks.iter().enumerate() {
            let pe = &self.p[bi];
            let phe = &phi[off..off + b];
            let mut v = vec![0.0f32; b];
            for (i, vi) in v.iter_mut().enumerate() {
                let row = &pe[i * b..i * b + b];
                *vi = row.iter().zip(phe).map(|(&pij, &pj)| pij * pj).sum();
            }
            denom += phe.iter().zip(&v).map(|(&p, &vi)| p * vi).sum::<f32>();
            pphi.push(v);
        }
        let inv = 1.0 / denom;
        let resid = y - self.predict(phi);
        // per-block weight + precision update
        for (bi, &(off, b)) in self.blocks.iter().enumerate() {
            let v = &pphi[bi];
            for (wi, &vi) in self.w[off..off + b].iter_mut().zip(v) {
                *wi += vi * inv * resid;
            }
            let il = 1.0 / self.lambda;
            let pe = &mut self.p[bi];
            let mut trace = 0.0f32;
            for (i, &vi) in v.iter().enumerate() {
                let gi = vi * inv;
                let row = &mut pe[i * b..i * b + b];
                for (pij, &vj) in row.iter_mut().zip(v) {
                    *pij = (*pij - gi * vj) * il;
                }
                trace += row[i];
            }
            if trace > self.caps[bi] {
                let s = self.caps[bi] / trace;
                for x in pe.iter_mut() {
                    *x *= s;
                }
            }
        }
        resid
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut xs = seed;
        move || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        }
    }

    /// Closed-form verification: on STATIONARY data, single-output forgetting-RLS
    /// with lambda=1 converges to the batch ridge normal-equations solution.
    #[test]
    fn converges_to_batch_ridge() {
        let (d, n, ridge) = (5usize, 400usize, 1.0f32);
        let mut nx = lcg(7);
        let w_true: Vec<f32> = (0..d).map(|_| nx()).collect();
        let phis: Vec<Vec<f32>> = (0..n).map(|_| (0..d).map(|_| nx()).collect()).collect();
        let ys: Vec<f32> = phis
            .iter()
            .map(|p| p.iter().zip(&w_true).map(|(&a, &b)| a * b).sum::<f32>() + 0.05 * nx())
            .collect();

        let mut head = EvolventHead::new(d, 1, ridge, 1.0);
        for (p, &y) in phis.iter().zip(&ys) {
            head.update(p, &[y]);
        }

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

    /// Multi-output one-hot RLS separates two Gaussian blobs (classification).
    #[test]
    fn one_hot_classifies_blobs() {
        let d = 2usize;
        let mut nx = lcg(3);
        let mut head = EvolventHead::new(d, 2, 1.0, 1.0);
        let sample = |cls: usize, nx: &mut dyn FnMut() -> f32| -> [f32; 2] {
            let mu = if cls == 0 { -1.0 } else { 1.0 };
            [mu + 0.3 * nx(), mu + 0.3 * nx()]
        };
        for _ in 0..300 {
            for cls in 0..2 {
                let x = sample(cls, &mut nx);
                let y = if cls == 0 { [1.0, 0.0] } else { [0.0, 1.0] };
                head.update(&x, &y);
            }
        }
        let mut correct = 0;
        for _ in 0..100 {
            for cls in 0..2 {
                if head.predict_class(&sample(cls, &mut nx)) == cls {
                    correct += 1;
                }
            }
        }
        assert!(correct >= 190, "blob accuracy too low: {correct}/200");
    }

    /// Forgetting tracks an abrupt drift (single output).
    #[test]
    fn tracks_a_drift() {
        let d = 4usize;
        let mut nx = lcg(3);
        let mut head = EvolventHead::new(d, 1, 1.0, 0.95);
        let w1: Vec<f32> = (0..d).map(|_| nx()).collect();
        let w2: Vec<f32> = w1.iter().map(|&v| -v).collect();
        let teach = |p: &[f32], w: &[f32]| p.iter().zip(w).map(|(&a, &b)| a * b).sum::<f32>();
        for _ in 0..300 {
            let p: Vec<f32> = (0..d).map(|_| nx()).collect();
            head.update(&p, &[teach(&p, &w1)]);
        }
        for _ in 0..300 {
            let p: Vec<f32> = (0..d).map(|_| nx()).collect();
            head.update(&p, &[teach(&p, &w2)]);
        }
        let mut e = 0.0f32;
        for _ in 0..50 {
            let p: Vec<f32> = (0..d).map(|_| nx()).collect();
            e += (head.predict(&p)[0] - teach(&p, &w2)).powi(2);
        }
        assert!(e / 50.0 < 1e-2, "did not track the drift: mse {}", e / 50.0);
    }

    /// Regression (F-EVO-1 guard): many-feature forgetting-RLS stays bounded.
    #[test]
    fn windup_guard_keeps_it_bounded() {
        let d = 200usize;
        let mut nx = lcg(11);
        let mut head = EvolventHead::new(d, 1, 1.0, 0.99);
        for _ in 0..5000 {
            let mut phi = vec![0.0f32; d];
            for _ in 0..5 {
                phi[((nx() + 0.5) * d as f32) as usize % d] = nx();
            }
            head.update(&phi, &[nx()]);
        }
        assert!(head.w.iter().all(|v| v.is_finite()), "weights diverged");
        let wnorm: f32 = head.w.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!(wnorm < 1e3, "weights blew up: |w| = {wnorm}");
    }

    /// Block precision with a SINGLE block reduces to the dense EvolventHead exactly.
    #[test]
    fn single_block_equals_dense() {
        let (d, n) = (6usize, 300usize);
        let mut nx = lcg(21);
        let mut dense = EvolventHead::new(d, 1, 1.0, 1.0);
        let mut block = BlockEvolventHead::new(&[d], 1.0, 1.0);
        for _ in 0..n {
            let phi: Vec<f32> = (0..d).map(|_| nx()).collect();
            let y = nx();
            dense.update(&phi, &[y]);
            block.update(&phi, y);
        }
        for (i, (&a, &b)) in dense.w.iter().zip(&block.w).enumerate() {
            assert!((a - b).abs() < 1e-4, "w[{i}] dense {a} vs block {b}");
        }
        assert_eq!(block.precision_nnz(), d * d);
    }

    /// Block-diagonal RLS on feature-disjoint-support data matches dense RLS
    /// (the exact regime) at a fraction of the precision storage.
    #[test]
    fn block_matches_dense_when_separable() {
        // 2 blocks of 3; block A active on even samples, block B on odd -> Phi^T Phi
        // is EXACTLY block-diagonal, so dense and block RLS agree.
        let (b, nb) = (3usize, 2usize);
        let d = b * nb;
        let mut nx = lcg(5);
        let mut dense = EvolventHead::new(d, 1, 1.0, 1.0);
        let mut block = BlockEvolventHead::new(&[b, b], 1.0, 1.0);
        for t in 0..600 {
            let mut phi = vec![0.0f32; d];
            let blk = t % nb;
            for i in 0..b {
                phi[blk * b + i] = nx();
            }
            let y = nx();
            dense.update(&phi, &[y]);
            block.update(&phi, y);
        }
        // predictions agree on fresh block-structured inputs
        let mut maxdiff = 0.0f32;
        for t in 0..50 {
            let mut phi = vec![0.0f32; d];
            let blk = t % nb;
            for i in 0..b {
                phi[blk * b + i] = nx();
            }
            maxdiff = maxdiff.max((dense.predict(&phi)[0] - block.predict(&phi)).abs());
        }
        assert!(maxdiff < 1e-3, "dense vs block prediction diff {maxdiff}");
        assert!(
            block.precision_nnz() < d * d,
            "block must store fewer than d^2"
        );
    }

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
