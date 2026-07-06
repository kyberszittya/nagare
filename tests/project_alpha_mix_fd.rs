//! Finite-difference verification of the alpha-mix projection kernel.
//!
//! Checks both closed-form gradients (`grad_x`, `grad_basis`) of
//! `project_alpha_mix_backward` against central differences of the scalar
//! loss `L = sum(grad_y elementwise-times y)` through
//! `project_alpha_mix_forward`.

use holonomy_learn::{
    default_holonomy_basis, project_alpha_mix_backward, project_alpha_mix_forward,
    ProjectAlphaMixShape, PROJECTION_ALPHA,
};

const EPS: f32 = 1.0e-3;
const TOL_ABS: f32 = 2.0e-3;
const TOL_REL: f32 = 2.0e-2;

fn loss(x: &[f32], basis: &[f32], alpha: f32, grad_y: &[f32], shape: ProjectAlphaMixShape) -> f64 {
    project_alpha_mix_forward(x, basis, alpha, shape)
        .iter()
        .zip(grad_y.iter())
        .map(|(&y, &g)| f64::from(y) * f64::from(g))
        .sum()
}

fn assert_close(analytic: f32, numeric: f64, what: &str, index: usize) {
    let numeric = numeric as f32;
    let err = (analytic - numeric).abs();
    let tol = TOL_ABS + TOL_REL * numeric.abs().max(analytic.abs());
    assert!(
        err <= tol,
        "{what}[{index}]: analytic={analytic} numeric={numeric} err={err} tol={tol}"
    );
}

/// Deterministic quasi-random values in roughly `[-0.9, 0.9]`.
fn pseudo_values(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            ((h >> 8) as f32 / (1u32 << 24) as f32) * 1.8 - 0.9
        })
        .collect()
}

fn run_fd_case(dim: usize, rank: usize, n: usize, alpha: f32, salt: u32) {
    let shape = ProjectAlphaMixShape { dim, rank };
    let x = pseudo_values(n * dim, salt);
    let mut basis = pseudo_values(rank * dim, salt + 1);
    // Keep one zero row in multi-row bases to exercise the skip path.
    if rank > 1 {
        for v in &mut basis[(rank - 1) * dim..] {
            *v = 0.0;
        }
    }
    let grad_y = pseudo_values(n * dim, salt + 2);
    let analytic = project_alpha_mix_backward(&x, &basis, alpha, &grad_y, shape);

    for i in 0..x.len() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[i] += EPS;
        minus[i] -= EPS;
        let numeric = (loss(&plus, &basis, alpha, &grad_y, shape)
            - loss(&minus, &basis, alpha, &grad_y, shape))
            / (2.0 * f64::from(EPS));
        assert_close(analytic.grad_x[i], numeric, "grad_x", i);
    }
    for i in 0..basis.len() {
        let mut plus = basis.clone();
        let mut minus = basis.clone();
        plus[i] += EPS;
        minus[i] -= EPS;
        let numeric = (loss(&x, &plus, alpha, &grad_y, shape)
            - loss(&x, &minus, alpha, &grad_y, shape))
            / (2.0 * f64::from(EPS));
        assert_close(analytic.grad_basis[i], numeric, "grad_basis", i);
    }
}

#[test]
fn finite_difference_small_dense_case() {
    run_fd_case(4, 2, 3, 0.72, 11);
}

#[test]
fn finite_difference_alpha_one_pure_projection() {
    run_fd_case(5, 3, 2, 1.0, 23);
}

#[test]
fn finite_difference_alpha_zero_identity_gradients() {
    let shape = ProjectAlphaMixShape { dim: 3, rank: 2 };
    let x = pseudo_values(6, 5);
    let basis = pseudo_values(6, 6);
    let grad_y = pseudo_values(6, 7);
    let out = project_alpha_mix_backward(&x, &basis, 0.0, &grad_y, shape);
    for (gx, gy) in out.grad_x.iter().zip(grad_y.iter()) {
        assert!((gx - gy).abs() < 1.0e-6);
    }
    assert!(out.grad_basis.iter().all(|&g| g == 0.0));
}

#[test]
fn finite_difference_production_shaped_case() {
    // The learner's real shape: the fitted 6 x 28 holonomy basis.
    let basis = default_holonomy_basis();
    let shape = ProjectAlphaMixShape {
        dim: basis.dim(),
        rank: basis.rank(),
    };
    let x = pseudo_values(2 * basis.dim(), 31);
    let grad_y = pseudo_values(2 * basis.dim(), 37);
    let analytic =
        project_alpha_mix_backward(&x, basis.vectors(), PROJECTION_ALPHA, &grad_y, shape);
    for i in 0..x.len() {
        let mut plus = x.clone();
        let mut minus = x.clone();
        plus[i] += EPS;
        minus[i] -= EPS;
        let numeric = (loss(&plus, basis.vectors(), PROJECTION_ALPHA, &grad_y, shape)
            - loss(&minus, basis.vectors(), PROJECTION_ALPHA, &grad_y, shape))
            / (2.0 * f64::from(EPS));
        assert_close(analytic.grad_x[i], numeric, "grad_x", i);
    }
}
