//! Linear layer (`y = xW + b`) with closed-form backward.
//!
//! Forward:
//! ```text
//!   y[n][j] = Σ_i x[n][i] · W[i][j] + b[j]
//! ```
//!
//! Backward (chain rule):
//! ```text
//!   ∂L/∂x[n][i] = Σ_j (∂L/∂y[n][j]) · W[i][j]
//!   ∂L/∂W[i][j] = Σ_n x[n][i] · (∂L/∂y[n][j])
//!   ∂L/∂b[j]    = Σ_n (∂L/∂y[n][j])
//! ```
//!
//! No autograd: gradients are explicit Rust functions over flat
//! `&[f32]` buffers.

use rayon::prelude::*;

/// Linear layer parameters.
#[derive(Debug, Clone)]
pub struct LinearLayer {
    /// Weights `(in_dim, out_dim)` row-major.
    pub w: Vec<f32>,
    /// Biases `(out_dim,)`.
    pub b: Vec<f32>,
    /// Input dim.
    pub in_dim: usize,
    /// Output dim.
    pub out_dim: usize,
}

impl LinearLayer {
    /// New layer with Glorot-uniform init.
    pub fn new(in_dim: usize, out_dim: usize, seed: u64) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let scale = (6.0 / (in_dim + out_dim) as f32).sqrt();
        let mut w = vec![0.0f32; in_dim * out_dim];
        for v in &mut w {
            *v = (rng.random::<f32>() * 2.0 - 1.0) * scale;
        }
        Self {
            w,
            b: vec![0.0; out_dim],
            in_dim,
            out_dim,
        }
    }

    /// Gradient buffer of zeros, same shape as `self`.
    pub fn zero_grad(&self) -> LinearLayer {
        LinearLayer {
            w: vec![0.0; self.w.len()],
            b: vec![0.0; self.b.len()],
            in_dim: self.in_dim,
            out_dim: self.out_dim,
        }
    }
}

/// Linear forward. `x` is `(n, in_dim)` flat; returns `(n, out_dim)` flat.
///
/// Uses `ikj` (SAXPY) accumulation: initialise each output row with the bias,
/// then for each input `i` broadcast `x[i]` and add the *contiguous* W-row into
/// the *contiguous* output row. The inner j-loop writes distinct slots (no
/// reduction), so it autovectorises; and for a fixed j the additions run in
/// i-order, so the result is bit-identical to a scalar-accumulate form.
pub fn linear_forward(layer: &LinearLayer, x: &[f32]) -> Vec<f32> {
    let n = x.len() / layer.in_dim;
    assert_eq!(x.len(), n * layer.in_dim);
    let out_dim = layer.out_dim;
    let mut out = vec![0.0f32; n * out_dim];
    out.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(ni, row)| {
            let x_row = &x[ni * layer.in_dim..(ni + 1) * layer.in_dim];
            row.copy_from_slice(&layer.b);
            let mut w_base = 0usize;
            for &xi in x_row {
                let w_row = &layer.w[w_base..w_base + out_dim];
                for (slot, &w) in row.iter_mut().zip(w_row.iter()) {
                    *slot += xi * w;
                }
                w_base += out_dim;
            }
        });
    out
}

/// Linear backward. Returns `(grad_x, grad_layer)`.
pub fn linear_backward(layer: &LinearLayer, x: &[f32], grad_y: &[f32]) -> (Vec<f32>, LinearLayer) {
    let n = x.len() / layer.in_dim;
    assert_eq!(grad_y.len(), n * layer.out_dim);
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_w = vec![0.0f32; layer.w.len()];
    let mut grad_b = vec![0.0f32; layer.b.len()];
    // grad_x[n][i] = Σ_j grad_y[n][j] · W[i][j]   — parallel over n.
    grad_x
        .par_chunks_mut(layer.in_dim)
        .enumerate()
        .for_each(|(ni, gx_row)| {
            let gy_row = &grad_y[ni * layer.out_dim..(ni + 1) * layer.out_dim];
            for (i, slot) in gx_row.iter_mut().enumerate() {
                let mut a = 0.0f32;
                for (j, &gy_j) in gy_row.iter().enumerate() {
                    a += gy_j * layer.w[i * layer.out_dim + j];
                }
                *slot = a;
            }
        });
    // grad_W[i][j] = Σ_n x[n][i] · grad_y[n][j]
    // grad_b[j]    = Σ_n grad_y[n][j]
    // Sequential — small in-dim and out-dim in our use case (≤64).
    for ni in 0..n {
        let x_row = &x[ni * layer.in_dim..(ni + 1) * layer.in_dim];
        let gy_row = &grad_y[ni * layer.out_dim..(ni + 1) * layer.out_dim];
        for i in 0..layer.in_dim {
            let xi = x_row[i];
            for j in 0..layer.out_dim {
                grad_w[i * layer.out_dim + j] += xi * gy_row[j];
            }
        }
        for j in 0..layer.out_dim {
            grad_b[j] += gy_row[j];
        }
    }
    (
        grad_x,
        LinearLayer {
            w: grad_w,
            b: grad_b,
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_backward_matches_numerical() {
        let layer = LinearLayer::new(3, 2, 42);
        let x = vec![0.1, 0.2, 0.3, -0.4, 0.5, -0.6]; // n=2, in_dim=3
        let y = linear_forward(&layer, &x);
        let grad_y = vec![1.0; y.len()]; // L = sum(y)
        let (gx, gl) = linear_backward(&layer, &x, &grad_y);

        let eps = 1e-3;
        // Check grad_w
        for i in 0..layer.in_dim {
            for j in 0..layer.out_dim {
                let idx = i * layer.out_dim + j;
                let mut l_p = layer.clone();
                l_p.w[idx] += eps;
                let mut l_m = layer.clone();
                l_m.w[idx] -= eps;
                let y_p: f32 = linear_forward(&l_p, &x).iter().sum();
                let y_m: f32 = linear_forward(&l_m, &x).iter().sum();
                let num = (y_p - y_m) / (2.0 * eps);
                let ana = gl.w[idx];
                assert!(
                    (ana - num).abs() < 1e-2,
                    "w[{},{}]: ana={} num={}",
                    i,
                    j,
                    ana,
                    num
                );
            }
        }
        // Check grad_b
        for j in 0..layer.out_dim {
            let mut l_p = layer.clone();
            l_p.b[j] += eps;
            let mut l_m = layer.clone();
            l_m.b[j] -= eps;
            let y_p: f32 = linear_forward(&l_p, &x).iter().sum();
            let y_m: f32 = linear_forward(&l_m, &x).iter().sum();
            let num = (y_p - y_m) / (2.0 * eps);
            assert!((gl.b[j] - num).abs() < 1e-2);
        }
        // Check grad_x
        for i in 0..x.len() {
            let mut x_p = x.clone();
            x_p[i] += eps;
            let mut x_m = x.clone();
            x_m[i] -= eps;
            let y_p: f32 = linear_forward(&layer, &x_p).iter().sum();
            let y_m: f32 = linear_forward(&layer, &x_m).iter().sum();
            let num = (y_p - y_m) / (2.0 * eps);
            assert!((gx[i] - num).abs() < 1e-2);
        }
    }
}
