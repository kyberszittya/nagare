//! Native Catmull--Rom and Chebyshev-CR activation kernels.
//!
//! These operators mirror `hymeko_neuro.core.splines`: per-channel uniform CR on
//! `[-1, 1]`, plus Chebyshev-generated CR control points and direct Chebyshev
//! deploy evaluation. All gradients are explicit closed-form buffers.

/// Cache from Catmull--Rom forward needed by the closed-form backward.
#[derive(Debug, Clone)]
pub struct CatmullRomCache {
    /// Clamped input values, flat `(n, channels)`.
    pub x_clamped: Vec<f32>,
    /// Segment index per input element.
    pub indices: Vec<usize>,
    /// Local segment coordinate per input element.
    pub t: Vec<f32>,
    /// Number of rows in the flat input.
    pub n: usize,
    /// Channel count.
    pub channels: usize,
    /// Number of CR control points per channel.
    pub grid: usize,
}

/// Result of a Catmull--Rom backward pass.
#[derive(Debug, Clone)]
pub struct CatmullRomBackward {
    /// Gradient w.r.t. input `x`, flat `(n, channels)`.
    pub grad_x: Vec<f32>,
    /// Gradient w.r.t. CR control points, flat `(channels, grid)`.
    pub grad_coef: Vec<f32>,
}

/// Result of Chebyshev-CR train-mode backward.
#[derive(Debug, Clone)]
pub struct ChebyshevCrBackward {
    /// Gradient w.r.t. input `x`, flat `(n, channels)`.
    pub grad_x: Vec<f32>,
    /// Gradient w.r.t. Chebyshev coefficients, flat `(channels, k)`.
    pub grad_coef: Vec<f32>,
}

/// Forward Catmull--Rom activation.
///
/// # Preconditions
/// - `grid >= 4`
/// - `x.len() == n * channels`
/// - `coef.len() == channels * grid`
///
/// # Postconditions
/// Returns output flat `(n, channels)` and a cache for backward. Inputs are
/// clamped to `[-1, 1]`, matching the canonical Python evaluator.
pub fn catmull_rom_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
) -> (Vec<f32>, CatmullRomCache) {
    assert!(grid >= 4);
    assert_eq!(x.len(), n * channels);
    assert_eq!(coef.len(), channels * grid);

    let mut out = vec![0.0; x.len()];
    let mut x_clamped = vec![0.0; x.len()];
    let mut indices = vec![0usize; x.len()];
    let mut ts = vec![0.0; x.len()];

    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let xc = x[idx].clamp(-1.0, 1.0);
            let u = (xc + 1.0) * 0.5 * (grid - 1) as f32;
            let i = (u.floor() as usize).min(grid - 2);
            let t = u - i as f32;
            let weights = cr_weights(t);
            let control = cr_control_indices(i, grid);
            let base = ch * grid;
            out[idx] = weights[0] * coef[base + control[0]]
                + weights[1] * coef[base + control[1]]
                + weights[2] * coef[base + control[2]]
                + weights[3] * coef[base + control[3]];
            x_clamped[idx] = xc;
            indices[idx] = i;
            ts[idx] = t;
        }
    }

    (
        out,
        CatmullRomCache {
            x_clamped,
            indices,
            t: ts,
            n,
            channels,
            grid,
        },
    )
}

/// Backward Catmull--Rom activation.
///
/// # Preconditions
/// `grad_y` has the same flat shape as the forward input/output.
///
/// # Postconditions
/// Returns gradients for input and control points. The input gradient is zero
/// outside the clamp interval.
pub fn catmull_rom_backward(
    coef: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
) -> CatmullRomBackward {
    assert_eq!(grad_y.len(), cache.n * cache.channels);
    assert_eq!(coef.len(), cache.channels * cache.grid);

    let mut grad_x = vec![0.0; grad_y.len()];
    let mut grad_coef = vec![0.0; coef.len()];
    let du_dx = 0.5 * (cache.grid - 1) as f32;

    for row in 0..cache.n {
        for ch in 0..cache.channels {
            let idx = row * cache.channels + ch;
            let gy = grad_y[idx];
            let i = cache.indices[idx];
            let t = cache.t[idx];
            let weights = cr_weights(t);
            let dweights = cr_weight_derivatives(t);
            let control = cr_control_indices(i, cache.grid);
            let base = ch * cache.grid;
            for k in 0..4 {
                grad_coef[base + control[k]] += gy * weights[k];
            }
            if (-1.0..=1.0).contains(&cache.x_clamped[idx]) {
                let mut dy_dt = 0.0;
                for k in 0..4 {
                    dy_dt += dweights[k] * coef[base + control[k]];
                }
                grad_x[idx] = gy * dy_dt * du_dx;
            }
        }
    }

    CatmullRomBackward { grad_x, grad_coef }
}

/// Chebyshev basis evaluated at uniform CR knots in `[-1, 1]`.
///
/// Returns flat `(grid, k)`.
pub fn chebyshev_knot_basis(grid: usize, k: usize) -> Vec<f32> {
    assert!(grid >= 2);
    assert!(k >= 1);
    let mut out = vec![0.0; grid * k];
    for g in 0..grid {
        let x = -1.0 + 2.0 * g as f32 / (grid - 1) as f32;
        let terms = chebyshev_terms(x, k);
        out[g * k..(g + 1) * k].copy_from_slice(&terms);
    }
    out
}

/// Build Chebyshev-CR control points.
///
/// `coef` is flat `(channels, k)`, `basis` is flat `(grid, k)`, and the return
/// is flat `(channels, grid)`.
pub fn chebyshev_control_points(
    coef: &[f32],
    basis: &[f32],
    channels: usize,
    grid: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(coef.len(), channels * k);
    assert_eq!(basis.len(), grid * k);
    let mut out = vec![0.0; channels * grid];
    for ch in 0..channels {
        for g in 0..grid {
            let mut acc = 0.0;
            for term in 0..k {
                acc += coef[ch * k + term] * basis[g * k + term];
            }
            out[ch * grid + g] = acc;
        }
    }
    out
}

/// Forward Chebyshev-CR train path: generate CR control points, then run CR.
pub fn chebyshev_cr_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    grid: usize,
    k: usize,
) -> (Vec<f32>, CatmullRomCache, Vec<f32>, Vec<f32>) {
    let basis = chebyshev_knot_basis(grid, k);
    let control = chebyshev_control_points(coef, &basis, channels, grid, k);
    let (y, cache) = catmull_rom_forward(&control, x, n, channels, grid);
    (y, cache, control, basis)
}

/// Backward Chebyshev-CR train path.
pub fn chebyshev_cr_backward(
    control_points: &[f32],
    basis: &[f32],
    cache: &CatmullRomCache,
    grad_y: &[f32],
    k: usize,
) -> ChebyshevCrBackward {
    let cr = catmull_rom_backward(control_points, cache, grad_y);
    let mut grad_coef = vec![0.0; cache.channels * k];
    for ch in 0..cache.channels {
        for term in 0..k {
            let mut acc = 0.0;
            for g in 0..cache.grid {
                acc += cr.grad_coef[ch * cache.grid + g] * basis[g * k + term];
            }
            grad_coef[ch * k + term] = acc;
        }
    }
    ChebyshevCrBackward {
        grad_x: cr.grad_x,
        grad_coef,
    }
}

/// Direct Chebyshev deploy forward, avoiding CR gather.
///
/// `coef` is flat `(channels, k)`, `x` is flat `(n, channels)`.
pub fn chebyshev_deploy_forward(
    coef: &[f32],
    x: &[f32],
    n: usize,
    channels: usize,
    k: usize,
) -> Vec<f32> {
    assert_eq!(coef.len(), channels * k);
    assert_eq!(x.len(), n * channels);
    let mut out = vec![0.0; x.len()];
    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let xc = x[idx].clamp(-1.0, 1.0);
            out[idx] = chebyshev_eval_channel(coef, ch, k, xc);
        }
    }
    out
}

/// Direct Chebyshev deploy backward.
///
/// Returns gradients for input and Chebyshev coefficients.
pub fn chebyshev_deploy_backward(
    coef: &[f32],
    x: &[f32],
    grad_y: &[f32],
    n: usize,
    channels: usize,
    k: usize,
) -> ChebyshevCrBackward {
    assert_eq!(coef.len(), channels * k);
    assert_eq!(x.len(), n * channels);
    assert_eq!(grad_y.len(), x.len());
    let mut grad_x = vec![0.0; x.len()];
    let mut grad_coef = vec![0.0; coef.len()];
    for row in 0..n {
        for ch in 0..channels {
            let idx = row * channels + ch;
            let xc = x[idx].clamp(-1.0, 1.0);
            let gy = grad_y[idx];
            accumulate_chebyshev_coef_grad(&mut grad_coef, ch, k, xc, gy);
            if (-1.0..=1.0).contains(&x[idx]) {
                grad_x[idx] = gy * chebyshev_eval_derivative_channel(coef, ch, k, xc);
            }
        }
    }
    ChebyshevCrBackward { grad_x, grad_coef }
}

fn chebyshev_eval_channel(coef: &[f32], ch: usize, k: usize, x: f32) -> f32 {
    if k == 0 {
        return 0.0;
    }
    let base = ch * k;
    let mut acc = coef[base];
    if k == 1 {
        return acc;
    }
    let mut prev = 1.0;
    let mut curr = x;
    acc += curr * coef[base + 1];
    for term in 2..k {
        let next = 2.0 * x * curr - prev;
        acc += next * coef[base + term];
        prev = curr;
        curr = next;
    }
    acc
}

fn chebyshev_eval_derivative_channel(coef: &[f32], ch: usize, k: usize, x: f32) -> f32 {
    if k <= 1 {
        return 0.0;
    }
    let base = ch * k;
    let mut prev = 1.0;
    let mut curr = x;
    let mut d_prev = 0.0;
    let mut d_curr = 1.0;
    let mut acc = coef[base + 1];
    for term in 2..k {
        let next = 2.0 * x * curr - prev;
        let d_next = 2.0 * curr + 2.0 * x * d_curr - d_prev;
        acc += coef[base + term] * d_next;
        prev = curr;
        curr = next;
        d_prev = d_curr;
        d_curr = d_next;
    }
    acc
}

fn accumulate_chebyshev_coef_grad(grad_coef: &mut [f32], ch: usize, k: usize, x: f32, gy: f32) {
    if k == 0 {
        return;
    }
    let base = ch * k;
    grad_coef[base] += gy;
    if k == 1 {
        return;
    }
    let mut prev = 1.0;
    let mut curr = x;
    grad_coef[base + 1] += gy * curr;
    for term in 2..k {
        let next = 2.0 * x * curr - prev;
        grad_coef[base + term] += gy * next;
        prev = curr;
        curr = next;
    }
}

fn cr_weights(t: f32) -> [f32; 4] {
    let t2 = t * t;
    let t3 = t2 * t;
    [
        0.5 * (-t3 + 2.0 * t2 - t),
        0.5 * (3.0 * t3 - 5.0 * t2 + 2.0),
        0.5 * (-3.0 * t3 + 4.0 * t2 + t),
        0.5 * (t3 - t2),
    ]
}

fn cr_weight_derivatives(t: f32) -> [f32; 4] {
    let t2 = t * t;
    [
        0.5 * (-3.0 * t2 + 4.0 * t - 1.0),
        0.5 * (9.0 * t2 - 10.0 * t),
        0.5 * (-9.0 * t2 + 8.0 * t + 1.0),
        0.5 * (3.0 * t2 - 2.0 * t),
    ]
}

fn cr_control_indices(i: usize, grid: usize) -> [usize; 4] {
    [
        i.saturating_sub(1).min(grid - 1),
        i.min(grid - 1),
        (i + 1).min(grid - 1),
        (i + 2).min(grid - 1),
    ]
}

fn chebyshev_terms(x: f32, k: usize) -> Vec<f32> {
    let mut terms = vec![0.0; k];
    if k == 0 {
        return terms;
    }
    terms[0] = 1.0;
    if k > 1 {
        terms[1] = x;
    }
    for term in 2..k {
        terms[term] = 2.0 * x * terms[term - 1] - terms[term - 2];
    }
    terms
}

#[cfg(test)]
mod tests {
    use super::*;

    fn loss_sum_cr(coef: &[f32], x: &[f32], n: usize, channels: usize, grid: usize) -> f32 {
        catmull_rom_forward(coef, x, n, channels, grid)
            .0
            .iter()
            .sum()
    }

    #[test]
    fn catmull_rom_interpolates_control_points() {
        let coef = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let x = vec![-1.0, -0.5, 0.0, 0.5, 1.0];
        let (y, _) = catmull_rom_forward(&coef, &x, 5, 1, 5);
        for (got, want) in y.iter().zip(coef.iter()) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn catmull_rom_backward_matches_finite_difference() {
        let n = 3;
        let channels = 2;
        let grid = 5;
        let coef = vec![0.1, -0.2, 0.3, 0.7, -0.1, 0.5, -0.4, 0.2, 0.1, 0.9];
        let x = vec![-0.7, -0.2, 0.15, 0.4, 0.72, -0.55];
        let (_, cache) = catmull_rom_forward(&coef, &x, n, channels, grid);
        let grad_y = vec![1.0; x.len()];
        let grad = catmull_rom_backward(&coef, &cache, &grad_y);
        let eps = 1e-3;

        for idx in 0..coef.len() {
            let mut plus = coef.clone();
            let mut minus = coef.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let num = (loss_sum_cr(&plus, &x, n, channels, grid)
                - loss_sum_cr(&minus, &x, n, channels, grid))
                / (2.0 * eps);
            assert!(
                (grad.grad_coef[idx] - num).abs() < 1e-3,
                "coef[{idx}] analytic={} numeric={num}",
                grad.grad_coef[idx]
            );
        }

        for idx in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let num = (loss_sum_cr(&coef, &plus, n, channels, grid)
                - loss_sum_cr(&coef, &minus, n, channels, grid))
                / (2.0 * eps);
            assert!(
                (grad.grad_x[idx] - num).abs() < 1e-3,
                "x[{idx}] analytic={} numeric={num}",
                grad.grad_x[idx]
            );
        }
    }

    #[test]
    fn chebyshev_control_points_match_manual_basis() {
        let basis = chebyshev_knot_basis(3, 3);
        assert_eq!(basis, vec![1.0, -1.0, 1.0, 1.0, 0.0, -1.0, 1.0, 1.0, 1.0]);
        let coef = vec![2.0, 3.0, 5.0];
        let control = chebyshev_control_points(&coef, &basis, 1, 3, 3);
        assert_eq!(control, vec![4.0, -3.0, 10.0]);
    }

    #[test]
    fn chebyshev_deploy_backward_matches_finite_difference() {
        let n = 3;
        let channels = 2;
        let k = 4;
        let coef = vec![0.2, -0.3, 0.1, 0.05, -0.4, 0.7, -0.2, 0.3];
        let x = vec![-0.8, -0.3, 0.1, 0.35, 0.7, -0.6];
        let grad_y = vec![1.0; x.len()];
        let grad = chebyshev_deploy_backward(&coef, &x, &grad_y, n, channels, k);
        let eps = 1e-3;

        for idx in 0..coef.len() {
            let mut plus = coef.clone();
            let mut minus = coef.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let yp: f32 = chebyshev_deploy_forward(&plus, &x, n, channels, k)
                .iter()
                .sum();
            let ym: f32 = chebyshev_deploy_forward(&minus, &x, n, channels, k)
                .iter()
                .sum();
            let num = (yp - ym) / (2.0 * eps);
            assert!((grad.grad_coef[idx] - num).abs() < 1e-3);
        }

        for idx in 0..x.len() {
            let mut plus = x.clone();
            let mut minus = x.clone();
            plus[idx] += eps;
            minus[idx] -= eps;
            let yp: f32 = chebyshev_deploy_forward(&coef, &plus, n, channels, k)
                .iter()
                .sum();
            let ym: f32 = chebyshev_deploy_forward(&coef, &minus, n, channels, k)
                .iter()
                .sum();
            let num = (yp - ym) / (2.0 * eps);
            assert!((grad.grad_x[idx] - num).abs() < 1e-3);
        }
    }

    #[test]
    fn chebyshev_cr_backward_matches_chain_rule_shape() {
        let n = 2;
        let channels = 2;
        let grid = 6;
        let k = 3;
        let coef = vec![0.1, 0.2, -0.1, 0.4, -0.3, 0.2];
        let x = vec![-0.6, -0.1, 0.25, 0.8];
        let (y, cache, control, basis) = chebyshev_cr_forward(&coef, &x, n, channels, grid, k);
        assert_eq!(y.len(), x.len());
        let grad = chebyshev_cr_backward(&control, &basis, &cache, &vec![1.0; y.len()], k);
        assert_eq!(grad.grad_x.len(), x.len());
        assert_eq!(grad.grad_coef.len(), coef.len());
        assert!(grad.grad_x.iter().all(|v| v.is_finite()));
        assert!(grad.grad_coef.iter().all(|v| v.is_finite()));
    }
}
