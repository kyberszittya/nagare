//! Kolmogorov-Arnold (KAN) layer with closed-form backward.
//!
//! `y[n][j] = Σ_i φ_{ij}(x[n][i])`, where each edge function `φ_{ij}` is a Chebyshev-CR
//! spline — reusing [`crate::ops::catmull_rom`]'s `chebyshev_cr_forward/backward` (the
//! shipped train-CR / deploy-Chebyshev primitive). This is the generic tabular learner
//! for T1/T2 (Nagare's closed-form ops as a KAN); no signed-graph structure.
//!
//! For each output `j`, one `chebyshev_cr` call applies the `d_in` splines `φ_{·j}` to the
//! `d_in` input channels, then the results are summed over `i`. Coefficients are flat
//! `(d_out, d_in, cheb_k)`.
//!
//! # Preconditions
//! Inputs are clamped to `[-1,1]` inside the spline; **standardise tabular features into
//! `[-1,1]`** upstream (the spline's trusted range) or the KAN saturates.

use crate::ops::catmull_rom::{chebyshev_cr_backward, chebyshev_cr_forward, CatmullRomCache};

/// Shape of a KAN layer.
#[derive(Debug, Clone, Copy)]
pub struct KanConfig {
    /// Input feature count.
    pub d_in: usize,
    /// Output feature count.
    pub d_out: usize,
    /// Catmull-Rom control points (≥ 4).
    pub grid: usize,
    /// Chebyshev order.
    pub cheb_k: usize,
}

impl KanConfig {
    /// Construct + validate.
    ///
    /// # Panics
    /// Panics if `grid < 4`, `cheb_k == 0`, or a dimension is 0.
    pub fn new(d_in: usize, d_out: usize, grid: usize, cheb_k: usize) -> Self {
        assert!(grid >= 4 && cheb_k >= 1 && d_in >= 1 && d_out >= 1);
        Self {
            d_in,
            d_out,
            grid,
            cheb_k,
        }
    }
    fn block(&self) -> usize {
        self.d_in * self.cheb_k
    }
}

/// Cache for the KAN backward.
#[derive(Debug, Clone)]
pub struct KanCache {
    caches: Vec<CatmullRomCache>,
    controls: Vec<Vec<f32>>,
    basis: Vec<f32>,
    n: usize,
}

/// Forward KAN layer. `coef` flat `(d_out, d_in, cheb_k)`, `x` flat `(n, d_in)`.
///
/// # Postconditions
/// Returns `y (n, d_out)` and a backward cache.
///
/// # Panics
/// Panics if `coef`/`x` lengths do not match `cfg`/`n`.
pub fn kan_forward(coef: &[f32], x: &[f32], n: usize, cfg: KanConfig) -> (Vec<f32>, KanCache) {
    assert_eq!(coef.len(), cfg.d_out * cfg.block());
    assert_eq!(x.len(), n * cfg.d_in);
    let mut y = vec![0.0f32; n * cfg.d_out];
    let mut caches = Vec::with_capacity(cfg.d_out);
    let mut controls = Vec::with_capacity(cfg.d_out);
    let mut basis = Vec::new();
    for j in 0..cfg.d_out {
        let coef_j = &coef[j * cfg.block()..(j + 1) * cfg.block()]; // (d_in, cheb_k)
        let (sp, cache, control, b) =
            chebyshev_cr_forward(coef_j, x, n, cfg.d_in, cfg.grid, cfg.cheb_k); // (n, d_in)
        for row in 0..n {
            y[row * cfg.d_out + j] = sp[row * cfg.d_in..row * cfg.d_in + cfg.d_in].iter().sum();
        }
        caches.push(cache);
        controls.push(control);
        basis = b;
    }
    (
        y,
        KanCache {
            caches,
            controls,
            basis,
            n,
        },
    )
}

/// Backward KAN layer → `(grad_x (n, d_in), grad_coef (d_out, d_in, cheb_k))`.
///
/// # Panics
/// Panics if `grad_y.len() != cache.n · d_out`.
pub fn kan_backward(cache: &KanCache, grad_y: &[f32], cfg: KanConfig) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(grad_y.len(), cache.n * cfg.d_out);
    let mut grad_x = vec![0.0f32; cache.n * cfg.d_in];
    let mut grad_coef = vec![0.0f32; cfg.d_out * cfg.block()];
    for j in 0..cfg.d_out {
        // y[:,j] = Σ_i sp[:,i] ⇒ ∂y[:,j]/∂sp[:,i] = 1, so grad_sp[:,i] = grad_y[:,j].
        let mut grad_sp = vec![0.0f32; cache.n * cfg.d_in];
        for row in 0..cache.n {
            let gy = grad_y[row * cfg.d_out + j];
            for slot in grad_sp[row * cfg.d_in..row * cfg.d_in + cfg.d_in].iter_mut() {
                *slot = gy;
            }
        }
        let bw = chebyshev_cr_backward(
            &cache.controls[j],
            &cache.basis,
            &cache.caches[j],
            &grad_sp,
            cfg.cheb_k,
        );
        for (acc, g) in grad_x.iter_mut().zip(&bw.grad_x) {
            *acc += g;
        }
        grad_coef[j * cfg.block()..(j + 1) * cfg.block()].copy_from_slice(&bw.grad_coef);
    }
    (grad_x, grad_coef)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> (Vec<f32>, Vec<f32>, usize, KanConfig) {
        let cfg = KanConfig::new(3, 2, 6, 4);
        let n = 4;
        let x: Vec<f32> = (0..n * cfg.d_in)
            .map(|i| 0.3 * ((i as f32 * 1.1).sin()))
            .collect();
        let coef: Vec<f32> = (0..cfg.d_out * cfg.block())
            .map(|i| 0.2 * ((i as f32 * 0.7).cos()))
            .collect();
        (coef, x, n, cfg)
    }

    #[test]
    fn forward_shape_and_finite() {
        let (coef, x, n, cfg) = fixture();
        let (y, _) = kan_forward(&coef, &x, n, cfg);
        assert_eq!(y.len(), n * cfg.d_out);
        assert!(y.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn backward_matches_finite_difference() {
        let (coef, x, n, cfg) = fixture();
        let (y, cache) = kan_forward(&coef, &x, n, cfg);
        let grad_y = vec![1.0f32; y.len()]; // L = Σ y
        let (gx, gc) = kan_backward(&cache, &grad_y, cfg);
        let eps = 1e-3;
        let sum_fwd = |c: &[f32], xf: &[f32]| -> f32 { kan_forward(c, xf, n, cfg).0.iter().sum() };
        for (idx, &g) in gx.iter().enumerate() {
            let mut xp = x.clone();
            xp[idx] += eps;
            let mut xm = x.clone();
            xm[idx] -= eps;
            let num = (sum_fwd(&coef, &xp) - sum_fwd(&coef, &xm)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-2, "grad_x[{idx}] {g} vs {num}");
        }
        for (idx, &g) in gc.iter().enumerate() {
            let mut cp = coef.clone();
            cp[idx] += eps;
            let mut cm = coef.clone();
            cm[idx] -= eps;
            let num = (sum_fwd(&cp, &x) - sum_fwd(&cm, &x)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-2, "grad_coef[{idx}] {g} vs {num}");
        }
    }
}
