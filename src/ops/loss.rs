//! BCE-with-logits loss with closed-form backward.
//!
//! Numerically-stable formulation (LogSumExp trick):
//! ```text
//!   L_n = max(s_n, 0) - s_n · y_n + log(1 + exp(-|s_n|))
//!   L = mean_n L_n
//! ```
//!
//! Backward:
//! ```text
//!   ∂L/∂s_n = (1/N) · (σ(s_n) - y_n)
//! ```
//! where σ is the logistic sigmoid. Single closed-form expression;
//! no autograd required.

/// Sigmoid (numerically stable for both positive and negative inputs).
#[inline]
fn sigmoid(s: f32) -> f32 {
    if s >= 0.0 {
        let e = (-s).exp();
        1.0 / (1.0 + e)
    } else {
        let e = s.exp();
        e / (1.0 + e)
    }
}

/// BCE-with-logits forward.  Returns the scalar mean loss across the
/// `N` samples in `logits` and `targets`.
pub fn bce_with_logits_forward(logits: &[f32], targets: &[f32]) -> f32 {
    assert_eq!(logits.len(), targets.len());
    let n = logits.len();
    if n == 0 {
        return 0.0;
    }
    let mut sum = 0.0f64;
    for (&s, &y) in logits.iter().zip(targets.iter()) {
        let abs = s.abs();
        let log_one_plus = (-abs).exp().ln_1p();
        let per = s.max(0.0) - s * y + log_one_plus;
        sum += per as f64;
    }
    (sum / n as f64) as f32
}

/// BCE-with-logits backward. Returns `(N,)` gradient of the *mean*
/// loss w.r.t. each logit: `(1/N) (σ(s) − y)`.
pub fn bce_with_logits_backward(logits: &[f32], targets: &[f32]) -> Vec<f32> {
    let n = logits.len();
    let inv_n = 1.0 / n as f32;
    let mut out = vec![0.0f32; n];
    for i in 0..n {
        out[i] = (sigmoid(logits[i]) - targets[i]) * inv_n;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_consistency() {
        // Logistic must equal 1/(1+e^{-s}) for both signs.
        for &s in &[-3.0_f32, -1.0, 0.0, 1.0, 3.0] {
            let expected = 1.0 / (1.0 + (-s).exp());
            assert!((sigmoid(s) - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn bce_backward_matches_numerical() {
        let logits = vec![-1.5_f32, 0.3, 2.7, -0.8];
        let targets = vec![0.0_f32, 1.0, 1.0, 0.0];
        let grad = bce_with_logits_backward(&logits, &targets);
        let eps = 1e-3;
        for i in 0..logits.len() {
            let mut lp = logits.clone();
            lp[i] += eps;
            let mut lm = logits.clone();
            lm[i] -= eps;
            let num = (bce_with_logits_forward(&lp, &targets)
                - bce_with_logits_forward(&lm, &targets))
                / (2.0 * eps);
            assert!(
                (grad[i] - num).abs() < 1e-3,
                "logit[{}]: ana={} num={}",
                i,
                grad[i],
                num
            );
        }
    }

    #[test]
    fn bce_perfect_prediction_zero_loss() {
        // logit large positive + target=1 → near-zero loss.
        let loss = bce_with_logits_forward(&[10.0], &[1.0]);
        assert!(loss < 0.001);
        // logit large negative + target=0 → near-zero loss.
        let loss = bce_with_logits_forward(&[-10.0], &[0.0]);
        assert!(loss < 0.001);
    }
}
