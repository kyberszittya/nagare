//! Adam optimizer step (Kingma & Ba 2014).
//!
//! Maintains first-moment (`m`) and second-moment (`v`) running
//! averages per parameter, with bias correction. Plain Rust struct,
//! no autograd integration — caller passes in the gradient directly.

/// Per-parameter Adam state buffers.
#[derive(Debug, Clone)]
pub struct AdamState {
    /// First moment (mean of grads).
    pub m: Vec<f32>,
    /// Second moment (mean of squared grads).
    pub v: Vec<f32>,
    /// Step counter.
    pub t: u64,
    /// β₁.
    pub beta1: f32,
    /// β₂.
    pub beta2: f32,
    /// ε.
    pub eps: f32,
}

impl AdamState {
    /// New zero-initialised state for a parameter of length `n`.
    pub fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
        }
    }
}

/// In-place Adam step:
/// ```text
///   m ← β1 m + (1-β1) g
///   v ← β2 v + (1-β2) g²
///   m̂ ← m / (1-β1^t),   v̂ ← v / (1-β2^t)
///   θ ← θ − lr · m̂ / (√v̂ + ε)
/// ```
pub fn adam_step(params: &mut [f32], grad: &[f32], state: &mut AdamState, lr: f32) {
    assert_eq!(params.len(), grad.len());
    assert_eq!(state.m.len(), params.len());
    state.t += 1;
    let t = state.t as i32;
    let bc1 = 1.0 - state.beta1.powi(t);
    let bc2 = 1.0 - state.beta2.powi(t);
    for i in 0..params.len() {
        let g = grad[i];
        state.m[i] = state.beta1 * state.m[i] + (1.0 - state.beta1) * g;
        state.v[i] = state.beta2 * state.v[i] + (1.0 - state.beta2) * g * g;
        let m_hat = state.m[i] / bc1;
        let v_hat = state.v[i] / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + state.eps);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adam_reduces_constant_gradient_loss() {
        // Toy: minimize L = θ² with gradient = 2θ; Adam should drive
        // θ → 0 in a handful of steps.
        let mut p = vec![1.0f32];
        let mut state = AdamState::new(1);
        for _ in 0..200 {
            let g = vec![2.0 * p[0]];
            adam_step(&mut p, &g, &mut state, 0.05);
        }
        assert!(p[0].abs() < 0.1, "θ after 200 steps = {}", p[0]);
    }

    #[test]
    fn adam_bias_correction_first_step() {
        // After 1 step with β1=0.9, β2=0.999, the bias-corrected m̂
        // should equal g exactly (since (1-β1^1)/(1-β1) = 1).
        let mut p = vec![0.0f32];
        let g = vec![3.0f32];
        let mut state = AdamState::new(1);
        // Capture state before step
        adam_step(&mut p, &g, &mut state, 1.0);
        // After step: m = 0.1·3 = 0.3; m̂ = 0.3 / 0.1 = 3.0
        // v = 0.001·9 = 0.009; v̂ = 0.009 / 0.001 = 9.0; √v̂ = 3.0
        // Δθ = 1.0 · 3.0 / (3.0 + 1e-8) ≈ 1.0
        // p was 0.0, gradient was +3 → minimize → p decreases by ~1.0.
        assert!(
            (p[0] - (-1.0)).abs() < 1e-3,
            "expected θ ≈ -1.0 after 1 Adam step, got {}",
            p[0]
        );
    }
}
