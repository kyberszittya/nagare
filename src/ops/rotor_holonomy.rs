//! **Rotor holonomy** — the order-sensitive running rotor product over a signed cycle.
//!
//! Clifford-FIR reframed as a *transmission channel* rather than an outer compressor. A signal
//! transmits around a cycle; each edge applies a rotor (a unit quaternion, from [`crate::cayley_rotor`]
//! upstream); the **ordered product** `H_c = q_{k-1} ⋯ q_1 q_0` is the accumulated rotation — the
//! **holonomy**. Because SO(3) rotors do not commute, `H_c` is *order-sensitive* (a genuine holonomy,
//! not the trivial abelian sum-of-angles), and it generalizes signed-graph *balance* (the sign product
//! `e^{iπ·#neg}`) into a learned geometric quantity. See `docs/plans/2026-07-10-rotor-holonomy`.
//!
//! # Forward
//! Per cycle, with `P_{-1} = 1` (identity): `P_i = q_i · P_{i-1}`, and `H_c = P_{k-1}`. The prefix
//! products `P_0 … P_{k-1}` are saved for the backward.
//!
//! # Backward (given `grad_H = ∂L/∂H`)
//! With `H = S_i · q_i · P_{i-1}` where `S_i = q_{k-1} ⋯ q_{i+1}` (suffix), and the quaternion adjoint
//! identities `⟨qx, y⟩ = ⟨x, q̄ y⟩`, `⟨xq, y⟩ = ⟨x, y q̄⟩` (Euclidean inner product on ℝ⁴, hold for
//! non-unit `q`):
//! ```text
//!   ∂L/∂q_i = conj(S_i) · grad_H · conj(P_{i-1})
//! ```
//! Suffixes accumulate in the backward pass: `S_{k-1} = 1`, `S_{i-1} = S_i · q_i`.

/// Identity quaternion `(w, x, y, z) = (1, 0, 0, 0)`.
const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// Hamilton product `a · b` of two quaternions `(w, x, y, z)`.
#[inline]
fn qmul(a: [f32; 4], b: [f32; 4]) -> [f32; 4] {
    [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]
}

/// Quaternion conjugate `(w, -x, -y, -z)`.
#[inline]
fn qconj(q: [f32; 4]) -> [f32; 4] {
    [q[0], -q[1], -q[2], -q[3]]
}

#[inline]
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// Rotor-holonomy forward: ordered quaternion product per cycle.
///
/// # Preconditions
/// `edge_quats.len() == n_cycles * k * 4`, `k >= 1`.
///
/// # Postconditions
/// Returns `(holo, prefixes)` with `holo.len() == n_cycles * 4` (the holonomy per cycle) and
/// `prefixes.len() == n_cycles * k * 4` (the saved prefix products `P_0 … P_{k-1}`).
///
/// # Panics
/// If `edge_quats.len() != n_cycles * k * 4` or `k == 0`.
pub fn rotor_holonomy_forward(
    edge_quats: &[f32],
    n_cycles: usize,
    k: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert!(k >= 1);
    assert_eq!(edge_quats.len(), n_cycles * k * 4);
    let mut holo = vec![0.0f32; n_cycles * 4];
    let mut prefixes = vec![0.0f32; n_cycles * k * 4];
    for c in 0..n_cycles {
        let base = c * k * 4;
        let mut p = q_at(edge_quats, c * k); // P_0 = q_0
        prefixes[base..base + 4].copy_from_slice(&p);
        for i in 1..k {
            p = qmul(q_at(edge_quats, c * k + i), p); // P_i = q_i · P_{i-1}
            prefixes[base + i * 4..base + i * 4 + 4].copy_from_slice(&p);
        }
        holo[c * 4..c * 4 + 4].copy_from_slice(&p);
    }
    (holo, prefixes)
}

/// Rotor-holonomy backward. Given `grad_holo = ∂L/∂H`, returns `∂L/∂edge_quats`.
///
/// # Preconditions
/// `edge_quats.len() == prefixes.len() == n_cycles * k * 4`, `grad_holo.len() == n_cycles * 4`, `k >= 1`.
///
/// # Panics
/// If the length preconditions do not hold.
pub fn rotor_holonomy_backward(
    edge_quats: &[f32],
    prefixes: &[f32],
    grad_holo: &[f32],
    n_cycles: usize,
    k: usize,
) -> Vec<f32> {
    assert!(k >= 1);
    assert_eq!(edge_quats.len(), n_cycles * k * 4);
    assert_eq!(prefixes.len(), n_cycles * k * 4);
    assert_eq!(grad_holo.len(), n_cycles * 4);
    let mut grad = vec![0.0f32; n_cycles * k * 4];
    for c in 0..n_cycles {
        let base = c * k * 4;
        let grad_h = q_at(grad_holo, c);
        let mut suffix = IDENT; // S_{k-1} = 1
        for i in (0..k).rev() {
            let p_prev = if i == 0 {
                IDENT
            } else {
                q_at(prefixes, c * k + i - 1)
            };
            let gqi = qmul(qmul(qconj(suffix), grad_h), qconj(p_prev));
            grad[base + i * 4..base + i * 4 + 4].copy_from_slice(&gqi);
            suffix = qmul(suffix, q_at(edge_quats, c * k + i)); // S_{i-1} = S_i · q_i
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand_quats(n: usize) -> Vec<f32> {
        // Deterministic non-unit, non-coplanar quaternions (so rotors don't commute).
        (0..n * 4)
            .map(|i| (i as f32 * 0.7 + 0.3).sin() + 0.2 * (i as f32 * 1.9).cos())
            .collect()
    }

    /// Directional-derivative gradient check: `⟨∇, u⟩` vs central FD of `L = Σ (H · w)` along `u`.
    #[test]
    fn backward_matches_fd() {
        let (n, k) = (3usize, 4usize);
        let eq = rand_quats(n * k);
        let (_h, pre) = rotor_holonomy_forward(&eq, n, k);
        let w: Vec<f32> = (0..n * 4).map(|i| (i as f32 * 1.3).cos()).collect();
        let grad_holo: Vec<f32> = w.clone();
        let ana = rotor_holonomy_backward(&eq, &pre, &grad_holo, n, k);
        let loss = |q: &[f32]| -> f32 {
            let (h, _) = rotor_holonomy_forward(q, n, k);
            h.iter().zip(&w).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-3;
        for d in 0..5 {
            let u: Vec<f32> = (0..eq.len())
                .map(|i| ((i as f32 + d as f32 * 7.0) * 0.5).sin())
                .collect();
            let dir: f32 = ana.iter().zip(&u).map(|(g, ui)| g * ui).sum();
            let fp: Vec<f32> = eq.iter().zip(&u).map(|(a, ui)| a + eps * ui).collect();
            let fm: Vec<f32> = eq.iter().zip(&u).map(|(a, ui)| a - eps * ui).collect();
            let num = (loss(&fp) - loss(&fm)) / (2.0 * eps);
            assert!(
                (dir - num).abs() < 2e-3 + 2e-3 * num.abs(),
                "dir {d}: {dir} vs fd {num}"
            );
        }
    }

    #[test]
    fn k1_is_identity_map() {
        let eq = rand_quats(5);
        let (h, _) = rotor_holonomy_forward(&eq, 5, 1);
        assert_eq!(h, eq); // H = q_0
    }

    #[test]
    fn all_identity_edges_give_identity_holonomy() {
        let n = 4;
        let k = 3;
        let eq: Vec<f32> = (0..n)
            .flat_map(|_| IDENT.iter().chain(&IDENT).chain(&IDENT).copied())
            .collect();
        let (h, _) = rotor_holonomy_forward(&eq, n, k);
        for c in 0..n {
            assert!((h[c * 4] - 1.0).abs() < 1e-6);
            assert!(h[c * 4 + 1..c * 4 + 4].iter().all(|&v| v.abs() < 1e-6));
        }
    }

    #[test]
    fn order_sensitive_for_noncommuting_rotors() {
        // Same two edges in swapped order → different holonomy (proves non-abelian / "running").
        let a = [0.7f32, 0.2, -0.3, 0.5];
        let b = [0.1f32, 0.6, 0.4, -0.2];
        let ab: Vec<f32> = a.iter().chain(&b).copied().collect();
        let ba: Vec<f32> = b.iter().chain(&a).copied().collect();
        let (h_ab, _) = rotor_holonomy_forward(&ab, 1, 2);
        let (h_ba, _) = rotor_holonomy_forward(&ba, 1, 2);
        let gap: f32 = h_ab.iter().zip(&h_ba).map(|(x, y)| (x - y).abs()).sum();
        assert!(gap > 1e-2, "holonomy should be order-sensitive, gap {gap}");
    }

    #[test]
    fn matches_hand_product_k3() {
        let eq = rand_quats(3);
        let (h, _) = rotor_holonomy_forward(&eq, 1, 3);
        let expect = qmul(q_at(&eq, 2), qmul(q_at(&eq, 1), q_at(&eq, 0)));
        for j in 0..4 {
            assert!((h[j] - expect[j]).abs() < 1e-6);
        }
    }
}
