//! **The non-commutativity task** — the test of holonomy *specificity*. Two regional holonomies
//! `H_A = R(a_A, θ)`, `H_B = R(a_B, θ)` (same angle θ, matched) built as ordered edge-rotor loop
//! products; the class is whether they **commute** (`[H_A,H_B]=I`, parallel axes) or not (perp axes).
//!
//! The signal lives *only* in the commutator — an ordered, non-abelian quantity — while every
//! magnitude and every per-edge marginal is matched. A scalar pooler / mean / covariance-entropy of
//! the edges is structurally blind (the signal is in the ordered product, not any unordered
//! statistic); a conv/MLP over the raw edges must discover loop composition + quaternion
//! multiplication (which the F-HOLO-3 MLP failed at); and the abelian magnitude (`θ_A,θ_B`) is
//! matched, so even a loop-product-*angle* method is at chance. Only the non-commutative composition
//! ([`commutator_angle`], via [`crate::rotor_holonomy_forward`]) reveals the class.
//!
//! Reuses `curvature_task` (`haar_quat`, `axis_angle_quat`, `rotor_angle`, `Rng`),
//! `rotor_holonomy_forward`, and `hymeko_clifford::{quat_mul, quat_conjugate}` — no re-implementation.

use crate::curvature_task::{axis_angle_quat, haar_quat, rotor_angle, Rng};
use crate::rotor_holonomy_forward;
use hymeko_clifford::{quat_conjugate, quat_mul};

const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// A uniform unit 3-vector (Haar on the sphere).
fn haar_axis(rng: &mut Rng) -> [f32; 3] {
    loop {
        let v = [rng.g(), rng.g(), rng.g()];
        let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if n > 1e-4 {
            return [v[0] / n, v[1] / n, v[2] / n];
        }
    }
}

/// A random unit vector orthogonal to `a` (uniform on the great circle ⟂ `a`).
fn perp_axis(a: [f32; 3], rng: &mut Rng) -> [f32; 3] {
    loop {
        let v = haar_axis(rng);
        let dot = v[0] * a[0] + v[1] * a[1] + v[2] * a[2];
        let p = [v[0] - dot * a[0], v[1] - dot * a[1], v[2] - dot * a[2]];
        let n = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if n > 1e-3 {
            return [p[0] / n, p[1] / n, p[2] / n];
        }
    }
}

/// `k` edge rotors whose *ordered product* (`rotor_holonomy`, `H = q_{k-1}⋯q_0`) equals `target`:
/// `k-1` random Haar edges and one closing edge `= target · (product of the rest)⁻¹`. Every edge is
/// Haar-marginal (the closing edge is a fixed rotor times a Haar element).
fn edges_with_product(rng: &mut Rng, k: usize, target: [f32; 4]) -> Vec<f32> {
    assert!(k >= 2);
    let mut edges = vec![0.0f32; k * 4];
    let mut prod = IDENT; // running q_{i}⋯q_0
    for e in edges.chunks_mut(4).take(k - 1) {
        let r = haar_quat(rng);
        e.copy_from_slice(&r);
        prod = quat_mul(r, prod);
    }
    let last = quat_mul(target, quat_conjugate(prod)); // last · prod = target
    edges[(k - 1) * 4..k * 4].copy_from_slice(&last);
    edges
}

/// Sample the non-commutativity task: `2k` edge rotors (region A `[0,k)`, region B `[k,2k)`), each a
/// cycle whose loop holonomy is `R(axis, θ)`. `class 0` = commute (parallel axes), `class 1` =
/// non-commute (perpendicular axes). Both regions use the same `theta` (matched magnitude); the axis
/// marginals are Haar in both classes, so only the axis *correlation* — the commutator — differs.
///
/// # Preconditions
/// `k >= 2`, `theta ∈ (0, π)`, `class ∈ {0,1}`.
pub fn sample_noncommute(rng: &mut Rng, k: usize, theta: f32, class: u8) -> Vec<f32> {
    assert!(k >= 2 && (0.0..=std::f32::consts::PI).contains(&theta) && class <= 1);
    let a_a = haar_axis(rng);
    let a_b = if class == 0 {
        if rng.f() < 0.5 {
            a_a
        } else {
            [-a_a[0], -a_a[1], -a_a[2]]
        }
    } else {
        perp_axis(a_a, rng)
    };
    let h_a = axis_angle_quat(a_a, theta);
    let h_b = axis_angle_quat(a_b, theta);
    let mut edges = vec![0.0f32; 2 * k * 4];
    edges[0..k * 4].copy_from_slice(&edges_with_product(rng, k, h_a));
    edges[k * 4..2 * k * 4].copy_from_slice(&edges_with_product(rng, k, h_b));
    edges
}

/// The ordered loop holonomy of a region: `rotor_holonomy` over its `k` edges.
pub fn region_holonomy(edges: &[f32], start: usize, k: usize) -> [f32; 4] {
    let slice = &edges[start * 4..(start + k) * 4];
    let (h, _) = rotor_holonomy_forward(slice, 1, k);
    [h[0], h[1], h[2], h[3]]
}

/// **Framework non-abelian readout**: the angle of the commutator
/// `[H_A, H_B] = H_A H_B H_A⁻¹ H_B⁻¹`. Zero iff the two regional holonomies commute (parallel axes),
/// positive otherwise. The order-sensitive quantity `rotor_holonomy` is built for and no unordered
/// statistic of the edges can see.
pub fn commutator_angle(edges: &[f32], k: usize) -> f32 {
    let h_a = region_holonomy(edges, 0, k);
    let h_b = region_holonomy(edges, k, k);
    let c = quat_mul(
        quat_mul(h_a, h_b),
        quat_mul(quat_conjugate(h_a), quat_conjugate(h_b)),
    );
    rotor_angle(c)
}

/// Abelian control readout: the sum of the two regional holonomy *angles* (magnitude info only).
/// Matched across classes by construction ⇒ at chance.
pub fn regional_angle_sum(edges: &[f32], k: usize) -> f32 {
    rotor_angle(region_holonomy(edges, 0, k)) + rotor_angle(region_holonomy(edges, k, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_product_equals_target() {
        // the construction is correct: the ordered product equals the prescribed holonomy angle.
        let mut rng = Rng(3);
        let edges = edges_with_product(&mut rng, 6, axis_angle_quat([0.0, 0.0, 1.0], 1.2));
        let h = region_holonomy(&edges, 0, 6);
        assert!(
            (rotor_angle(h) - 1.2).abs() < 1e-2,
            "loop product angle {}",
            rotor_angle(h)
        );
    }

    #[test]
    fn commute_class_has_zero_commutator_noncommute_positive() {
        let (mut c0, mut c1) = (0.0f32, 0.0f32);
        for s in 0..60u64 {
            let e0 = sample_noncommute(&mut Rng(10 + s), 6, 1.3, 0);
            let e1 = sample_noncommute(&mut Rng(200 + s), 6, 1.3, 1);
            c0 += commutator_angle(&e0, 6);
            c1 += commutator_angle(&e1, 6);
        }
        c0 /= 60.0;
        c1 /= 60.0;
        assert!(c0 < 0.05, "commute-class commutator not ~0: {c0}");
        assert!(c1 > 0.5, "non-commute-class commutator too small: {c1}");
    }

    #[test]
    fn regional_angle_matched_across_classes() {
        // the abelian magnitude carries no class info: mean regional-angle-sum equal across classes.
        let mean = |class: u8| -> f32 {
            let mut s = 0.0f32;
            for seed in 0..60u64 {
                let e = sample_noncommute(&mut Rng(500 + seed), 6, 1.3, class);
                s += regional_angle_sum(&e, 6);
            }
            s / 60.0
        };
        let (m0, m1) = (mean(0), mean(1));
        assert!(
            (m0 - m1).abs() < 0.02,
            "regional angle leaks class: {m0} vs {m1}"
        );
    }

    #[test]
    fn sample_deterministic() {
        let a = sample_noncommute(&mut Rng(7), 5, 1.0, 1);
        let b = sample_noncommute(&mut Rng(7), 5, 1.0, 1);
        assert_eq!(a, b);
    }
}
