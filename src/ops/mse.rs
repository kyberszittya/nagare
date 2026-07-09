//! Mean-squared-error loss with closed-form backward (the T2 regression head loss).
//!
//! Forward:  `L = mean_n (pred_n − target_n)²`.
//! Backward: `∂L/∂pred_n = 2 (pred_n − target_n) / n`.

/// Mean squared error over `n` scalar predictions.
///
/// # Panics
/// Panics if `pred.len() != target.len()`.
pub fn mse_forward(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len());
    let n = pred.len();
    if n == 0 {
        return 0.0;
    }
    let sum: f64 = pred
        .iter()
        .zip(target)
        .map(|(&p, &t)| {
            let d = (p - t) as f64;
            d * d
        })
        .sum();
    (sum / n as f64) as f32
}

/// Gradient of the mean squared error w.r.t. each prediction.
pub fn mse_backward(pred: &[f32], target: &[f32]) -> Vec<f32> {
    assert_eq!(pred.len(), target.len());
    let inv_n = 1.0 / pred.len() as f32;
    pred.iter()
        .zip(target)
        .map(|(&p, &t)| 2.0 * (p - t) * inv_n)
        .collect()
}

/// Coefficient of determination `R² = 1 − SS_res / SS_tot` (scale-free regression score).
pub fn r2_score(pred: &[f32], target: &[f32]) -> f32 {
    assert_eq!(pred.len(), target.len());
    let n = target.len();
    if n == 0 {
        return 0.0;
    }
    let mean: f32 = target.iter().sum::<f32>() / n as f32;
    let ss_res: f32 = pred
        .iter()
        .zip(target)
        .map(|(&p, &t)| (p - t) * (p - t))
        .sum();
    let ss_tot: f32 = target.iter().map(|&t| (t - mean) * (t - mean)).sum();
    if ss_tot <= 0.0 {
        0.0
    } else {
        1.0 - ss_res / ss_tot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_matches_finite_difference() {
        let pred = vec![0.5f32, -0.3, 1.2, 0.1, -0.8];
        let target = vec![0.2f32, 0.4, 1.0, -0.2, -0.5];
        let grad = mse_backward(&pred, &target);
        let eps = 1e-3;
        for (idx, &g) in grad.iter().enumerate() {
            let mut pp = pred.clone();
            pp[idx] += eps;
            let mut pm = pred.clone();
            pm[idx] -= eps;
            let num = (mse_forward(&pp, &target) - mse_forward(&pm, &target)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-3, "grad[{idx}] {g} vs {num}");
        }
    }

    #[test]
    fn perfect_and_r2() {
        let t = vec![1.0f32, 2.0, 3.0, 4.0];
        assert!(mse_forward(&t, &t) < 1e-9);
        assert!((r2_score(&t, &t) - 1.0).abs() < 1e-6);
        // Predicting the mean → R² = 0.
        let mean_pred = vec![2.5f32; 4];
        assert!(r2_score(&mean_pred, &t).abs() < 1e-6);
    }
}
