//! Narrow-tuned **rotor-spike** op — biological orientation-selective tuning.
//!
//! A bank of `K` orientation-selective units, each tuned to a preferred
//! orientation `mu_k = pi*k/K` (orientation, period pi) with a von-Mises
//! concentration `kappa` that sets the tuning width. Large `kappa` => narrow
//! tuning => a sharp "spike" (V1 simple-cell selectivity); small `kappa` =>
//! broad (like the existing [`crate::ops::phase_pool`] histogram). A divisive
//! normalisation (Carandini--Heeger gain control) sparsifies the bank into a
//! peaked orientation distribution.
//!
//! Input a per-pixel 2-vector field `(gx,gy)` (`n` samples x `np` pixels,
//! interleaved — the `phase_pool`/`oriented_descriptor` layout). Per pixel
//! `m = ||(gx,gy)||`, `theta = atan2(gy,gx)` (skip `m<eps`), and with
//! `phi_pk = 2*theta_p - 2*mu_k`:
//!
//! ```text
//! e_{s,k} = sum_p m_p * exp(kappa*(cos(phi_pk) - 1))   (shifted: overflow-safe, in (0,1])
//! Z_s     = eps_Z + sum_k e_{s,k}
//! y_{s,k} = e_{s,k} / Z_s                              (the normalised "spike")
//! ```
//!
//! The `-1` shift factors an `exp(-kappa)` out of every term; it cancels in the
//! ratio `y` (the op is defined with the shift, and the FD test verifies that
//! definition). **No novelty is claimed for the tuning model** (von-Mises /
//! Gabor tuning + divisive normalisation are classical V1 models, Hubel &
//! Wiesel, Carandini & Heeger). The contribution is a closed-form, hand-derived,
//! FD-verified orientation-tuning op in the no-autograd Nagare discipline.
//!
//! # Preconditions
//! - `field.len() == n*np*2` (interleaved `(gx,gy)`), `k >= 1`, `kappa >= 0`.
//! # Postconditions
//! - `spike.len() == n*k`; each sample's `k` spikes are non-negative and sum to
//!   `<= 1` (exactly `Z_s`-normalised bar the `eps_Z` floor).
//! - Larger `kappa` gives a sharper (lower-entropy) spike for oriented input.
//! - Backward is finite everywhere (isotropic pixels `m<eps` contribute 0).

use std::f32::consts::PI;

const MIN_MAG: f32 = 1e-6;
const EPS_Z: f32 = 1e-6;

/// Output feature dim: one spike per preferred orientation.
pub fn rotor_spike_dim(k: usize) -> usize {
    k
}

/// Preferred orientation of bin `ki` (of `k`): `pi*ki/k`.
#[inline]
fn mu(ki: usize, k: usize) -> f32 {
    PI * ki as f32 / k as f32
}

/// Forward output: the normalised spikes plus the state the backward needs
/// (`e` = shifted tuning sums, `z` = partition).
pub struct RotorSpikeOut {
    /// Normalised orientation spikes, flat `(n*k)`.
    pub spike: Vec<f32>,
    e: Vec<f32>,
    z: Vec<f32>,
}

/// Rotor-spike forward. See the module docs.
///
/// # Panics
/// If `field.len() != n*np*2` or `k == 0`.
pub fn rotor_spike_forward(
    field: &[f32],
    n: usize,
    np: usize,
    k: usize,
    kappa: f32,
) -> RotorSpikeOut {
    assert_eq!(field.len(), n * np * 2);
    assert!(k >= 1, "need >=1 orientation bin");
    let mut e = vec![0.0f32; n * k];
    let mut z = vec![0.0f32; n];
    for s in 0..n {
        let ek = &mut e[s * k..(s + 1) * k];
        for p in 0..np {
            let gx = field[(s * np + p) * 2];
            let gy = field[(s * np + p) * 2 + 1];
            let m = (gx * gx + gy * gy).sqrt();
            if m < MIN_MAG {
                continue;
            }
            let theta = gy.atan2(gx);
            for (ki, ekv) in ek.iter_mut().enumerate() {
                let phi = 2.0 * theta - 2.0 * mu(ki, k);
                *ekv += m * (kappa * (phi.cos() - 1.0)).exp();
            }
        }
        z[s] = EPS_Z + ek.iter().sum::<f32>();
    }
    let spike: Vec<f32> = (0..n * k).map(|i| e[i] / z[i / k]).collect();
    RotorSpikeOut { spike, e, z }
}

/// Rotor-spike backward. Given `grad_spike` (`n*k`), returns `grad_field`
/// (`n*np*2`).
///
/// # Panics
/// If the length preconditions do not hold.
pub fn rotor_spike_backward(
    field: &[f32],
    out: &RotorSpikeOut,
    grad_spike: &[f32],
    n: usize,
    np: usize,
    k: usize,
    kappa: f32,
) -> Vec<f32> {
    assert_eq!(field.len(), n * np * 2);
    assert_eq!(grad_spike.len(), n * k);
    let mut grad = vec![0.0f32; n * np * 2];
    for s in 0..n {
        let z = out.z[s];
        let ek = &out.e[s * k..(s + 1) * k];
        let gy = &grad_spike[s * k..(s + 1) * k];
        // Softmax-like normaliser adjoint: e_bar_j = gy_j/Z - (sum_k gy_k e_k)/Z^2.
        let a: f32 = (0..k).map(|ki| gy[ki] * ek[ki]).sum();
        let ebar: Vec<f32> = (0..k).map(|ki| gy[ki] / z - a / (z * z)).collect();
        for p in 0..np {
            let gxv = field[(s * np + p) * 2];
            let gyv = field[(s * np + p) * 2 + 1];
            let m = (gxv * gxv + gyv * gyv).sqrt();
            if m < MIN_MAG {
                continue;
            }
            let theta = gyv.atan2(gxv);
            // Accumulate d(sum ebar*e)/dm and /dtheta over the bank.
            let (mut dm, mut dth) = (0.0f32, 0.0f32);
            for (ki, &eb) in ebar.iter().enumerate() {
                let phi = 2.0 * theta - 2.0 * mu(ki, k);
                let ex = (kappa * (phi.cos() - 1.0)).exp();
                dm += eb * ex;
                dth += eb * (-2.0 * kappa * m * ex * phi.sin());
            }
            // Angle chain: dm/d(gx,gy)=(gx,gy)/m ; dtheta/d(gx,gy)=(-gy,gx)/m^2.
            let inv_m = 1.0 / m;
            let inv_m2 = inv_m * inv_m;
            grad[(s * np + p) * 2] = dm * (gxv * inv_m) + dth * (-gyv * inv_m2);
            grad[(s * np + p) * 2 + 1] = dm * (gyv * inv_m) + dth * (gxv * inv_m2);
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A field of `np` pixels all at orientation `theta0`, unit magnitude.
    fn oriented_field(np: usize, theta0: f32) -> Vec<f32> {
        let (s, c) = theta0.sin_cos();
        let mut f = vec![0.0f32; np * 2];
        for p in 0..np {
            f[p * 2] = c;
            f[p * 2 + 1] = s;
        }
        f
    }

    fn entropy(y: &[f32]) -> f32 {
        let sum: f32 = y.iter().sum::<f32>().max(1e-9);
        -y.iter()
            .map(|&v| {
                let p = v / sum;
                if p > 1e-9 {
                    p * p.ln()
                } else {
                    0.0
                }
            })
            .sum::<f32>()
    }

    #[test]
    fn backward_matches_fd() {
        // A mildly varied field (distinct orientations) so gradients are generic.
        let np = 6;
        let mut field = vec![0.0f32; np * 2];
        for p in 0..np {
            let th = 0.3 + 0.4 * p as f32;
            let (s, c) = th.sin_cos();
            let m = 0.5 + 0.1 * p as f32;
            field[p * 2] = m * c;
            field[p * 2 + 1] = m * s;
        }
        let (n, k, kappa) = (1usize, 8usize, 3.0f32);
        let out = rotor_spike_forward(&field, n, np, k, kappa);
        // Arbitrary upstream gradient; L = <grad_spike, spike>.
        let gsp: Vec<f32> = (0..k).map(|i| 0.2 - 0.05 * i as f32).collect();
        let grad = rotor_spike_backward(&field, &out, &gsp, n, np, k, kappa);
        let dot = |f: &[f32]| -> f32 {
            let o = rotor_spike_forward(f, n, np, k, kappa);
            o.spike.iter().zip(&gsp).map(|(&s, &g)| s * g).sum()
        };
        let eps = 1e-3;
        for i in 0..field.len() {
            let mut fp = field.clone();
            fp[i] += eps;
            let mut fm = field.clone();
            fm[i] -= eps;
            let num = (dot(&fp) - dot(&fm)) / (2.0 * eps);
            let denom = num.abs().max(1e-3);
            assert!(
                (grad[i] - num).abs() / denom < 2e-2,
                "grad[{i}] {} vs fd {num}",
                grad[i]
            );
        }
    }

    #[test]
    fn narrower_with_kappa() {
        // A single-orientation stimulus: the spike distribution must sharpen
        // (lower entropy) as kappa grows — the biological narrowing.
        let (np, k) = (16usize, 16usize);
        let field = oriented_field(np, 0.7);
        let e_lo = entropy(&rotor_spike_forward(&field, 1, np, k, 0.5).spike);
        let e_mid = entropy(&rotor_spike_forward(&field, 1, np, k, 3.0).spike);
        let e_hi = entropy(&rotor_spike_forward(&field, 1, np, k, 12.0).spike);
        assert!(
            e_hi < e_mid && e_mid < e_lo,
            "entropy not monotone: {e_lo} {e_mid} {e_hi}"
        );
    }

    #[test]
    fn peak_at_stimulus_orientation() {
        let (np, k) = (16usize, 16usize);
        let argmax = |y: &[f32]| {
            y.iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0
        };
        // theta0=0 -> peak bin 0 ; theta0=pi/2 -> peak bin k/2.
        let y0 = rotor_spike_forward(&oriented_field(np, 0.0), 1, np, k, 8.0).spike;
        assert_eq!(argmax(&y0), 0);
        let yq = rotor_spike_forward(&oriented_field(np, PI / 2.0), 1, np, k, 8.0).spike;
        assert_eq!(argmax(&yq), k / 2);
    }

    #[test]
    fn isotropic_flat_and_finite() {
        // Pixels spanning all orientations -> near-uniform spikes, no NaN.
        let (np, k) = (16usize, 16usize);
        let mut field = vec![0.0f32; np * 2];
        for p in 0..np {
            let (s, c) = (PI * p as f32 / np as f32).sin_cos();
            field[p * 2] = c;
            field[p * 2 + 1] = s;
        }
        let out = rotor_spike_forward(&field, 1, np, k, 6.0);
        assert!(out.spike.iter().all(|v| v.is_finite()));
        let (mx, mn) = out
            .spike
            .iter()
            .fold((f32::MIN, f32::MAX), |(a, b), &v| (a.max(v), b.min(v)));
        assert!(
            mx - mn < 0.1,
            "isotropic input not near-uniform: spread {}",
            mx - mn
        );
        // Backward finite too.
        let g = rotor_spike_backward(&field, &out, &vec![0.1f32; k], 1, np, k, 6.0);
        assert!(g.iter().all(|v| v.is_finite()));
    }
}
