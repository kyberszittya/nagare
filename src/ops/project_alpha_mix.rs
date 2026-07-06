//! Alpha-mixed subspace projection kernel.
//!
//! Given a (not necessarily normalised) basis `U = {u_r}` of `rank` row
//! vectors in `R^dim`, the forward computes, per sample `x`:
//!
//! ```text
//!   y = alpha * P(x) + (1 - alpha) * x,
//!   P(x) = sum_r (x . u_r / ||u_r||^2) u_r
//! ```
//!
//! `P` is the (oblique-safe) projector onto `span(U)`; rows with
//! `||u_r||^2 <= 1e-12` are skipped, so zero-padded bases are valid. The
//! backward is closed-form (`P` is symmetric):
//!
//! ```text
//!   dL/dx   = alpha * P(g) + (1 - alpha) * g
//!   dL/du_r = alpha * [ (g . u_r) * (x - 2 s_r u_r) / ||u_r||^2 + s_r g ],
//!             s_r = x . u_r / ||u_r||^2,   g = dL/dy
//! ```
//!
//! summed over samples. This is the kernel form of the fitted projection
//! gate from `reports/2026-07-02-fitted-projection-gate-holonomy-ablation.md`.

/// Shape of a projection call: `rank` basis rows of `dim` components.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProjectAlphaMixShape {
    /// Feature dimensionality of each sample and basis row.
    pub dim: usize,
    /// Number of basis rows (rows may be zero and are then skipped).
    pub rank: usize,
}

/// Backward result for [`project_alpha_mix_backward`].
#[derive(Debug, Clone)]
pub struct ProjectAlphaMixBackward {
    /// Gradient with respect to the inputs, shape `(n, dim)`.
    pub grad_x: Vec<f32>,
    /// Gradient with respect to the basis rows, shape `(rank, dim)`.
    pub grad_basis: Vec<f32>,
}

/// Alpha-mixed projection forward over a batch of `n = x.len() / dim` samples.
///
/// # Preconditions
/// * `x.len()` is a multiple of `shape.dim`.
/// * `basis.len() == shape.rank * shape.dim` (row-major).
/// * `alpha` is finite (typically in `[0, 1]`).
///
/// # Postconditions
/// * Returns `y` with `y.len() == x.len()`; `alpha == 0.0` reproduces `x`
///   exactly; for an orthonormal basis and `alpha == 1.0` the result is the
///   orthogonal projection onto `span(basis)`.
///
/// # Invariants
/// * `x` and `basis` are unmodified; basis rows with squared norm
///   `<= 1e-12` contribute nothing.
pub fn project_alpha_mix_forward(
    x: &[f32],
    basis: &[f32],
    alpha: f32,
    shape: ProjectAlphaMixShape,
) -> Vec<f32> {
    let ProjectAlphaMixShape { dim, rank } = shape;
    assert!(dim > 0, "dim must be positive");
    assert_eq!(x.len() % dim, 0, "x length must be a multiple of dim");
    assert_eq!(basis.len(), rank * dim, "basis must be rank x dim");
    let mut y = x.to_vec();
    let n = x.len() / dim;
    let mut projected = vec![0.0f32; dim];
    for sample in 0..n {
        let row = &x[sample * dim..(sample + 1) * dim];
        projected.iter_mut().for_each(|v| *v = 0.0);
        for r in 0..rank {
            let axis = &basis[r * dim..(r + 1) * dim];
            let norm2 = axis.iter().map(|v| v * v).sum::<f32>();
            if norm2 <= 1.0e-12 {
                continue;
            }
            let dot = row
                .iter()
                .zip(axis.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
            let scale = dot / norm2;
            for (dst, &axis_value) in projected.iter_mut().zip(axis.iter()) {
                *dst += scale * axis_value;
            }
        }
        let out_row = &mut y[sample * dim..(sample + 1) * dim];
        for i in 0..dim {
            out_row[i] = alpha * projected[i] + (1.0 - alpha) * row[i];
        }
    }
    y
}

/// Alpha-mixed projection backward over a batch of `n = x.len() / dim` samples.
///
/// # Preconditions
/// * Same as [`project_alpha_mix_forward`], plus `grad_y.len() == x.len()`.
///
/// # Postconditions
/// * `grad_x` has the shape of `x`; `grad_basis` has the shape of `basis`
///   and accumulates over all samples. Zero-norm basis rows receive zero
///   gradient.
pub fn project_alpha_mix_backward(
    x: &[f32],
    basis: &[f32],
    alpha: f32,
    grad_y: &[f32],
    shape: ProjectAlphaMixShape,
) -> ProjectAlphaMixBackward {
    let ProjectAlphaMixShape { dim, rank } = shape;
    assert!(dim > 0, "dim must be positive");
    assert_eq!(x.len() % dim, 0, "x length must be a multiple of dim");
    assert_eq!(basis.len(), rank * dim, "basis must be rank x dim");
    assert_eq!(grad_y.len(), x.len(), "grad_y must match x");
    let n = x.len() / dim;
    let mut grad_x = vec![0.0f32; x.len()];
    let mut grad_basis = vec![0.0f32; basis.len()];
    for sample in 0..n {
        let row = &x[sample * dim..(sample + 1) * dim];
        let g = &grad_y[sample * dim..(sample + 1) * dim];
        let gx = &mut grad_x[sample * dim..(sample + 1) * dim];
        // (1 - alpha) * g pass-through.
        for i in 0..dim {
            gx[i] += (1.0 - alpha) * g[i];
        }
        for r in 0..rank {
            let axis = &basis[r * dim..(r + 1) * dim];
            let norm2 = axis.iter().map(|v| v * v).sum::<f32>();
            if norm2 <= 1.0e-12 {
                continue;
            }
            let dot_x = row
                .iter()
                .zip(axis.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>();
            let dot_g = g.iter().zip(axis.iter()).map(|(&a, &b)| a * b).sum::<f32>();
            let s = dot_x / norm2;
            // alpha * P(g): P is symmetric, so reuse the same rank-1 terms.
            let g_scale = dot_g / norm2;
            for (dst, &axis_value) in gx.iter_mut().zip(axis.iter()) {
                *dst += alpha * g_scale * axis_value;
            }
            let gb = &mut grad_basis[r * dim..(r + 1) * dim];
            for i in 0..dim {
                gb[i] += alpha * (dot_g * (row[i] - 2.0 * s * axis[i]) / norm2 + s * g[i]);
            }
        }
    }
    ProjectAlphaMixBackward { grad_x, grad_basis }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_zero_is_identity() {
        let shape = ProjectAlphaMixShape { dim: 3, rank: 2 };
        let x = vec![0.3, -1.2, 0.7, 2.0, 0.1, -0.4];
        let basis = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let y = project_alpha_mix_forward(&x, &basis, 0.0, shape);
        assert_eq!(x, y);
    }

    #[test]
    fn alpha_one_orthonormal_projection_is_idempotent() {
        let shape = ProjectAlphaMixShape { dim: 3, rank: 2 };
        let x = vec![0.3, -1.2, 0.7];
        let basis = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let once = project_alpha_mix_forward(&x, &basis, 1.0, shape);
        let twice = project_alpha_mix_forward(&once, &basis, 1.0, shape);
        for (a, b) in once.iter().zip(twice.iter()) {
            assert!((a - b).abs() < 1.0e-6, "a={a} b={b}");
        }
        assert!(once[2].abs() < 1.0e-6, "component outside span must vanish");
    }

    #[test]
    fn zero_rows_are_skipped() {
        let shape = ProjectAlphaMixShape { dim: 2, rank: 2 };
        let x = vec![0.5, -0.25];
        let basis = vec![0.0, 0.0, 0.0, 1.0];
        let y = project_alpha_mix_forward(&x, &basis, 1.0, shape);
        assert!(y[0].abs() < 1.0e-9);
        assert!((y[1] + 0.25).abs() < 1.0e-6);
        let grads = project_alpha_mix_backward(&x, &basis, 1.0, &[1.0, 1.0], shape);
        assert_eq!(&grads.grad_basis[..2], &[0.0, 0.0]);
    }
}
