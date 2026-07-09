//! K-class softmax + cross-entropy with closed-form backward.
//!
//! Generalises the 2-class `metrics::{softmax2, cross_entropy}` to `K` classes for the
//! tabular KAN classifier (Iris is 3-class). Numerically stable (max-subtraction).
//!
//! Forward: `L = mean_n ( -log softmax(logits_n)[y_n] )`.
//! Backward: `∂L/∂logit[n][c] = (softmax(logits_n)[c] − 1[c = y_n]) / n`.

/// Row-wise softmax of `logits (n, k)` → `probs (n, k)` (stable).
pub fn softmax_k(logits: &[f32], n: usize, k: usize) -> Vec<f32> {
    assert_eq!(logits.len(), n * k);
    let mut probs = vec![0.0f32; n * k];
    for row in 0..n {
        let l = &logits[row * k..row * k + k];
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for (p, &v) in probs[row * k..row * k + k].iter_mut().zip(l) {
            *p = (v - m).exp();
            sum += *p;
        }
        for p in probs[row * k..row * k + k].iter_mut() {
            *p /= sum;
        }
    }
    probs
}

/// Mean K-class cross-entropy loss.
///
/// # Preconditions
/// `logits.len() == n·k`; every `labels[i] < k`.
///
/// # Panics
/// Panics if `logits.len() != n·k` or a label is out of range.
pub fn cross_entropy_k_forward(logits: &[f32], labels: &[usize], n: usize, k: usize) -> f32 {
    assert_eq!(logits.len(), n * k);
    assert_eq!(labels.len(), n);
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for (row, &y) in labels.iter().enumerate() {
        assert!(y < k, "label {y} out of range for k={k}");
        let l = &logits[row * k..row * k + k];
        let m = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let logsumexp = m + l.iter().map(|&v| (v - m).exp()).sum::<f32>().ln();
        sum += (logsumexp - l[y]) as f64;
    }
    (sum / n as f64) as f32
}

/// Gradient of the mean cross-entropy w.r.t. each logit, flat `(n, k)`.
pub fn cross_entropy_k_backward(logits: &[f32], labels: &[usize], n: usize, k: usize) -> Vec<f32> {
    let mut grad = softmax_k(logits, n, k);
    let inv_n = 1.0 / n as f32;
    for (row, &y) in labels.iter().enumerate() {
        grad[row * k + y] -= 1.0;
        for g in grad[row * k..row * k + k].iter_mut() {
            *g *= inv_n;
        }
    }
    grad
}

/// Argmax-accuracy of `logits (n, k)` against `labels`.
pub fn accuracy_k(logits: &[f32], labels: &[usize], n: usize, k: usize) -> f32 {
    let mut correct = 0usize;
    for (row, &y) in labels.iter().enumerate() {
        let l = &logits[row * k..row * k + k];
        let pred = (0..k).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap();
        correct += usize::from(pred == y);
    }
    correct as f32 / n as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_normalises_and_is_stable() {
        let logits = vec![1.0, 2.0, 3.0, 1000.0, 1000.0, 1000.0];
        let p = softmax_k(&logits, 2, 3);
        for row in 0..2 {
            let s: f32 = p[row * 3..row * 3 + 3].iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "row {row} sums to {s}");
            assert!(p[row * 3..row * 3 + 3].iter().all(|&v| v.is_finite()));
        }
        // Equal large logits → uniform, no overflow.
        assert!((p[3] - 1.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn backward_matches_finite_difference() {
        let n = 4;
        let k = 3;
        let logits = vec![
            0.3, -0.7, 1.2, -0.4, 0.9, 0.1, 1.5, -1.1, 0.2, -0.2, 0.6, -0.9,
        ];
        let labels = vec![2usize, 0, 1, 2];
        let grad = cross_entropy_k_backward(&logits, &labels, n, k);
        let eps = 1e-3;
        for (idx, &g) in grad.iter().enumerate() {
            let mut lp = logits.clone();
            lp[idx] += eps;
            let mut lm = logits.clone();
            lm[idx] -= eps;
            let num = (cross_entropy_k_forward(&lp, &labels, n, k)
                - cross_entropy_k_forward(&lm, &labels, n, k))
                / (2.0 * eps);
            assert!((g - num).abs() < 1e-3, "grad[{idx}] {g} vs {num}");
        }
    }

    #[test]
    fn perfect_prediction_zero_loss_and_full_acc() {
        let logits = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0];
        let labels = vec![0usize, 1];
        assert!(cross_entropy_k_forward(&logits, &labels, 2, 3) < 1e-3);
        assert!((accuracy_k(&logits, &labels, 2, 3) - 1.0).abs() < 1e-6);
    }
}
