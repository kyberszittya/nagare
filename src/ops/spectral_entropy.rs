//! Spectral-entropy Lyapunov-safe regulariser as a closed-form Nagare op.
//!
//! Port of `hymeko_neuro/hyperedge/entropy_reg.py::EntropyRegulariser` — HSiKAN's
//! refined spectral-entropy regulariser — with a hand-derived closed-form gradient
//! (no autograd). For a matrix `A ∈ ℝ^{n×d}` (the HSiKAN node-embedding matrix, or a
//! coef tensor reshaped to `(S·C, G)`):
//!
//! ```text
//!   G = AᵀA (d×d)          (or AAᵀ when n<d)
//!   (λ, U) = eigh(G)       symmetric eigendecomposition (Jacobi)
//!   s = max(λ, 0);  p = s / (Σs + ε)         spectral distribution
//!   H = -Σ p log₂ p;  H_norm = H / log₂(rank)
//!   reg = λ_eff · ( a·(H_norm − τ)² + b·H_norm + κ·(1 − H_norm) )
//! ```
//!
//! `λ_eff` is set by the detached Lyapunov schedule (KL-gated `exp`, optional EMA),
//! so it is **constant for the gradient**. The backward is
//! `∇_A reg = 2·A·M`, `M = U·diag(w)·Uᵀ`, `w_j = ∂reg/∂λ_j` — well-defined even under
//! eigenvalue degeneracy (`w` depends only on `λ`). See the plan for the full
//! derivation.
//!
//! # Preconditions
//! `a.len() == n·d`, `n ≥ 1`, `d ≥ 1`.

use std::f32::consts::LOG2_E; // = 1/ln 2

/// Configuration of the spectral-entropy regulariser (mirrors `EntropyRegConfig`).
#[derive(Debug, Clone, Copy)]
pub struct SpectralEntropyConfig {
    /// Base regularisation strength `λ₀`.
    pub lam_0: f32,
    /// Weight of the `(H_norm − τ)²` attractor.
    pub lam_a: f32,
    /// Weight of the `H_norm` (rank-collapse) term.
    pub lam_b: f32,
    /// Weight of the `1 − H_norm` (spread-prior) term.
    pub lam_kl: f32,
    /// KL-feedback rate `η` in the Lyapunov schedule.
    pub eta: f32,
    /// Target normalised entropy `τ`.
    pub target: f32,
    /// Numerical floor.
    pub eps: f32,
    /// Divide the schedule KL by `log₂(rank)` (scale-invariant `η`).
    pub kl_normalized: bool,
    /// EMA momentum on `λ_eff` (0 = off).
    pub momentum: f32,
}

impl Default for SpectralEntropyConfig {
    fn default() -> Self {
        Self {
            lam_0: 0.01,
            lam_a: 1.0,
            lam_b: 1.0,
            lam_kl: 0.0,
            eta: 5.0,
            target: 0.5,
            eps: 1e-12,
            kl_normalized: false,
            momentum: 0.0,
        }
    }
}

/// Off-diagonal squared Frobenius norm of a symmetric `m×m` matrix.
fn offdiag_sq(a: &[f32], m: usize) -> f32 {
    let mut off = 0.0f32;
    for p in 0..m {
        for q in (p + 1)..m {
            off += a[p * m + q] * a[p * m + q];
        }
    }
    off
}

/// Apply one symmetric Jacobi rotation annihilating `a[p][q]`, accumulating into `u`.
fn apply_rotation(a: &mut [f32], u: &mut [f32], m: usize, p: usize, q: usize) {
    let apq = a[p * m + q];
    let (app, aqq) = (a[p * m + p], a[q * m + q]);
    let tau = (aqq - app) / (2.0 * apq);
    let t = tau.signum() / (tau.abs() + (tau * tau + 1.0).sqrt());
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;
    for i in 0..m {
        if i != p && i != q {
            let (aip, aiq) = (a[i * m + p], a[i * m + q]);
            a[i * m + p] = c * aip - s * aiq;
            a[p * m + i] = a[i * m + p];
            a[i * m + q] = s * aip + c * aiq;
            a[q * m + i] = a[i * m + q];
        }
    }
    a[p * m + p] = app - t * apq;
    a[q * m + q] = aqq + t * apq;
    a[p * m + q] = 0.0;
    a[q * m + p] = 0.0;
    for i in 0..m {
        let (uip, uiq) = (u[i * m + p], u[i * m + q]);
        u[i * m + p] = c * uip - s * uiq;
        u[i * m + q] = s * uip + c * uiq;
    }
}

/// Symmetric eigendecomposition of a flat `m×m` matrix via cyclic Jacobi.
///
/// # Preconditions
/// `gram.len() == m·m` and `gram` is symmetric.
///
/// # Postconditions
/// Returns `(eigvals, eigvecs)` with `eigvals.len() == m`, `eigvecs` flat `m×m`
/// where column `j` (`eigvecs[i*m + j]`) is the unit eigenvector for `eigvals[j]`;
/// `G ≈ U·diag(λ)·Uᵀ` and `UᵀU = I`.
pub fn jacobi_eigh(gram: &[f32], m: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(gram.len(), m * m);
    let mut a = gram.to_vec();
    let mut u = vec![0.0f32; m * m];
    for i in 0..m {
        u[i * m + i] = 1.0;
    }
    for _ in 0..100 {
        if offdiag_sq(&a, m) <= 1e-20 {
            break;
        }
        for p in 0..m {
            for q in (p + 1)..m {
                if a[p * m + q].abs() > 1e-30 {
                    apply_rotation(&mut a, &mut u, m, p, q);
                }
            }
        }
    }
    let eigvals = (0..m).map(|i| a[i * m + i]).collect();
    (eigvals, u)
}

fn gram_ata(a: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; d * d];
    for k in 0..n {
        let row = &a[k * d..k * d + d];
        for i in 0..d {
            for j in 0..d {
                g[i * d + j] += row[i] * row[j];
            }
        }
    }
    g
}

fn gram_aat(a: &[f32], n: usize, d: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0f32;
            for c in 0..d {
                acc += a[i * d + c] * a[j * d + c];
            }
            g[i * n + j] = acc;
        }
    }
    g
}

/// Normalised spectral distribution `p` and `H_norm` from eigenvalues.
fn spectral_distribution(lam: &[f32], eps: f32) -> (Vec<f32>, f32) {
    let s: Vec<f32> = lam.iter().map(|&v| v.max(0.0)).collect();
    let sum_s = s.iter().sum::<f32>() + eps;
    let p: Vec<f32> = s.iter().map(|&v| v / sum_s).collect();
    let h: f32 = p.iter().map(|&pi| -pi * pi.max(eps).log2()).sum();
    let h_max = (lam.len().max(2) as f32).log2();
    (p, h / h_max)
}

/// `∂reg/∂λ` (the `w` vector) from the spectral distribution and the scalar `g_H`.
fn spectral_lambda_grad(
    lam: &[f32],
    p: &[f32],
    g_h: f32,
    h_max: f32,
    sum_s: f32,
    eps: f32,
) -> Vec<f32> {
    let c = g_h / h_max;
    let dreg_dp: Vec<f32> = p
        .iter()
        .map(|&pi| c * (-pi.max(eps).log2() - LOG2_E))
        .collect();
    let dot: f32 = dreg_dp.iter().zip(p).map(|(g, pi)| g * pi).sum();
    lam.iter()
        .zip(&dreg_dp)
        .map(|(&l, &gp)| if l > 0.0 { (gp - dot) / sum_s } else { 0.0 })
        .collect()
}

/// `M = U·diag(w)·Uᵀ`, flat `m×m`.
fn build_m(u: &[f32], w: &[f32], m: usize) -> Vec<f32> {
    let mut mmat = vec![0.0f32; m * m];
    for p in 0..m {
        for q in 0..m {
            let mut acc = 0.0f32;
            for j in 0..m {
                acc += u[p * m + j] * w[j] * u[q * m + j];
            }
            mmat[p * m + q] = acc;
        }
    }
    mmat
}

/// Regularisation value and `∇_A reg`, given a fixed `lam_eff`.
///
/// # Postconditions
/// Returns `(reg, grad_a, h_norm)`; `grad_a` is flat `(n, d)`.
///
/// # Panics
/// Panics if `a.len() != n·d`.
pub fn spectral_reg_value_grad(
    a: &[f32],
    n: usize,
    d: usize,
    cfg: &SpectralEntropyConfig,
    lam_eff: f32,
) -> (f32, Vec<f32>, f32) {
    assert_eq!(a.len(), n * d);
    let ata = n >= d;
    let m = if ata { d } else { n };
    let gram = if ata {
        gram_ata(a, n, d)
    } else {
        gram_aat(a, n, d)
    };
    let (lam, u) = jacobi_eigh(&gram, m);
    let (p, h_norm) = spectral_distribution(&lam, cfg.eps);
    let sum_s = lam.iter().map(|&v| v.max(0.0)).sum::<f32>() + cfg.eps;
    let h_max = (m.max(2) as f32).log2();

    let reg = lam_eff
        * (cfg.lam_a * (h_norm - cfg.target).powi(2)
            + cfg.lam_b * h_norm
            + cfg.lam_kl * (1.0 - h_norm));

    let g_h = lam_eff * (2.0 * cfg.lam_a * (h_norm - cfg.target) + cfg.lam_b - cfg.lam_kl);
    let w = spectral_lambda_grad(&lam, &p, g_h, h_max, sum_s, cfg.eps);
    let mmat = build_m(&u, &w, m);

    let mut grad_a = vec![0.0f32; n * d];
    if ata {
        // grad_A[k][j] = 2 Σ_r A[k][r] M[r][j]
        for k in 0..n {
            for j in 0..d {
                let mut acc = 0.0f32;
                for r in 0..d {
                    acc += a[k * d + r] * mmat[r * d + j];
                }
                grad_a[k * d + j] = 2.0 * acc;
            }
        }
    } else {
        // grad_A[k][j] = 2 Σ_r M[k][r] A[r][j]
        for k in 0..n {
            for j in 0..d {
                let mut acc = 0.0f32;
                for r in 0..n {
                    acc += mmat[k * n + r] * a[r * d + j];
                }
                grad_a[k * d + j] = 2.0 * acc;
            }
        }
    }
    (reg, grad_a, h_norm)
}

/// Stateful spectral-entropy regulariser carrying the Lyapunov schedule
/// (`prev_spectrum`, `λ_eff` EMA) across calls. Mirrors `EntropyRegulariser`.
#[derive(Debug, Clone)]
pub struct SpectralEntropyReg {
    cfg: SpectralEntropyConfig,
    prev_spectrum: Option<Vec<f32>>,
    lam_eff_ema: Option<f32>,
    /// Last `H_norm` (for logging).
    pub last_h_norm: f32,
    /// Last effective `λ_eff` (for logging).
    pub last_lam_eff: f32,
}

impl SpectralEntropyReg {
    /// New regulariser with the given config.
    pub fn new(cfg: SpectralEntropyConfig) -> Self {
        Self {
            cfg,
            prev_spectrum: None,
            lam_eff_ema: None,
            last_h_norm: f32::NAN,
            last_lam_eff: f32::NAN,
        }
    }

    /// KL(prev ‖ curr) in bits, matching `entropy_reg._kl_bits`.
    fn kl_bits(&self, curr: &[f32]) -> f32 {
        match &self.prev_spectrum {
            Some(prev) if prev.len() == curr.len() => curr
                .iter()
                .zip(prev)
                .map(|(&c, &pv)| {
                    let cc = c.max(self.cfg.eps);
                    cc * (cc.log2() - pv.max(self.cfg.eps).log2())
                })
                .sum(),
            _ => 0.0,
        }
    }

    /// Update the detached Lyapunov schedule and return `λ_eff`.
    fn schedule(&mut self, p: &[f32], h_max: f32) -> f32 {
        let kl = self.kl_bits(p);
        self.prev_spectrum = Some(p.to_vec());
        let kl_sched = if self.cfg.kl_normalized {
            kl / h_max.max(self.cfg.eps)
        } else {
            kl
        };
        let raw = self.cfg.lam_0 * (-self.cfg.eta * kl_sched).exp();
        let mut lam_eff = raw.clamp(0.1 * self.cfg.lam_0, 10.0 * self.cfg.lam_0);
        if self.cfg.momentum > 0.0 {
            let ema = match self.lam_eff_ema {
                Some(prev) => self.cfg.momentum * prev + (1.0 - self.cfg.momentum) * lam_eff,
                None => lam_eff,
            };
            self.lam_eff_ema = Some(ema);
            lam_eff = ema;
        }
        lam_eff
    }

    /// One regulariser call: `(reg, ∇_A reg)`. Updates the schedule (detached).
    ///
    /// # Panics
    /// Panics if `a.len() != n·d`.
    pub fn step(&mut self, a: &[f32], n: usize, d: usize) -> (f32, Vec<f32>) {
        assert_eq!(a.len(), n * d);
        let m = n.min(d);
        let gram = if n >= d {
            gram_ata(a, n, d)
        } else {
            gram_aat(a, n, d)
        };
        let (lam, _u) = jacobi_eigh(&gram, m);
        let (p, _h_norm) = spectral_distribution(&lam, self.cfg.eps);
        let h_max = (m.max(2) as f32).log2();
        let lam_eff = self.schedule(&p, h_max);
        let (reg, grad_a, h_norm) = spectral_reg_value_grad(a, n, d, &self.cfg, lam_eff);
        self.last_h_norm = h_norm;
        self.last_lam_eff = lam_eff;
        (reg, grad_a)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn jacobi_reconstructs_and_is_orthonormal() {
        // Symmetric 3×3.
        let g = vec![2.0, -1.0, 0.3, -1.0, 3.0, 0.4, 0.3, 0.4, 1.5];
        let (lam, u) = jacobi_eigh(&g, 3);
        // UᵀU = I.
        for a in 0..3 {
            for b in 0..3 {
                let dot: f32 = (0..3).map(|i| u[i * 3 + a] * u[i * 3 + b]).sum();
                let want = if a == b { 1.0 } else { 0.0 };
                assert!(approx(dot, want, 1e-5), "UtU[{a}][{b}]={dot}");
            }
        }
        // Reconstruct G = U diag(lam) Uᵀ.
        for r in 0..3 {
            for c in 0..3 {
                let g_rc: f32 = (0..3).map(|j| u[r * 3 + j] * lam[j] * u[c * 3 + j]).sum();
                assert!(
                    approx(g_rc, g[r * 3 + c], 1e-4),
                    "G[{r}][{c}] {g_rc} vs {}",
                    g[r * 3 + c]
                );
            }
        }
    }

    fn fixture() -> (Vec<f32>, usize, usize, SpectralEntropyConfig) {
        // Full-rank n>d so AᵀA is positive-definite (smooth spectrum for FD).
        let (n, d) = (6usize, 4usize);
        let a: Vec<f32> = (0..n * d)
            .map(|i| 0.5 * ((i as f32 * 1.3 + 0.7).sin()) + 0.1 * i as f32)
            .collect();
        let cfg = SpectralEntropyConfig {
            lam_0: 1.0,
            lam_a: 1.0,
            lam_b: 1.0,
            lam_kl: 0.5,
            ..Default::default()
        };
        (a, n, d, cfg)
    }

    #[test]
    fn grad_matches_finite_difference() {
        let (a, n, d, cfg) = fixture();
        let lam_eff = 1.0; // fixed schedule — matches the gradient's assumption.
        let (_, grad, _) = spectral_reg_value_grad(&a, n, d, &cfg, lam_eff);
        let eps = 1e-3;
        for (idx, &g) in grad.iter().enumerate() {
            let mut ap = a.clone();
            ap[idx] += eps;
            let mut am = a.clone();
            am[idx] -= eps;
            let rp = spectral_reg_value_grad(&ap, n, d, &cfg, lam_eff).0;
            let rm = spectral_reg_value_grad(&am, n, d, &cfg, lam_eff).0;
            let num = (rp - rm) / (2.0 * eps);
            assert!(
                approx(g, num, 1e-2),
                "grad_a[{idx}] analytic={g} numeric={num}"
            );
        }
    }

    #[test]
    fn schedule_first_call_is_lam0_then_moves() {
        let (a, n, d, _) = fixture();
        let cfg = SpectralEntropyConfig {
            lam_0: 0.02,
            eta: 5.0,
            ..Default::default()
        };
        let mut reg = SpectralEntropyReg::new(cfg);
        let _ = reg.step(&a, n, d);
        // First call: KL=0 → lam_eff = lam_0.
        assert!(approx(reg.last_lam_eff, cfg.lam_0, 1e-7));
        assert!(reg.last_h_norm.is_finite() && (0.0..=1.0).contains(&reg.last_h_norm));
        // A moved spectrum on the second call keeps lam_eff in the clamp band.
        let a2: Vec<f32> = a.iter().map(|v| v * 1.7 + 0.05).collect();
        let (r2, g2) = reg.step(&a2, n, d);
        assert!(r2.is_finite() && g2.iter().all(|v| v.is_finite()));
        assert!(reg.last_lam_eff >= 0.1 * cfg.lam_0 - 1e-9);
        assert!(reg.last_lam_eff <= 10.0 * cfg.lam_0 + 1e-9);
    }
}
