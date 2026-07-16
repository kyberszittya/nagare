//! **The closed Clifford approximation** — a one-shot, closed-form curvature estimator
//! (auto-holonomy Step 2). Given *only* the graph topology and the observed `SO(3)`
//! connection (not the generative flux), it estimates curvature in a single pass:
//!
//! 1. **gauge-fix by tree transport** — propagate node frames along the spanning tree from
//!    the root (`R̂_root = I`, `R̂_node = g · R̂_parent`). This is the exact gauge solve; the
//!    B1b `tree_consistency` backbone / Nagare balance theorem T1b, generalized to a tree.
//! 2. **cotree loop-closure residual** — for each cotree edge `(a→b)`, the fundamental
//!    cycle's holonomy is `ρ = R̂_b⁻¹ · g_{a→b} · R̂_a` (identity iff that cycle is flat).
//! 3. **readout** — curvature energy `= mean_e ‖log ρ_e‖` (a rotation angle), or the
//!    residual-angle distribution's entropy.
//!
//! This is "differentiation as connection transport" made **instantaneous and closed-form**:
//! the estimate is a holonomy (transport around fundamental loops) obtained by an exact
//! Clifford (quaternion) gauge solve — no gradient descent, `O(|E|)`, replacing the iterative
//! rotor learning of [`crate::RotorMeshNet`]. The oracle ([`oracle_curvature`]) is the same
//! holonomy read from the *true* plaquettes via [`crate::rotor_holonomy_forward`]; the
//! estimator recovers it from topology alone (a correctness identity, tested below).
//!
//! Reuses `hymeko_clifford::{quat_mul, quat_conjugate}` and `rotor_holonomy_forward` — no
//! quaternion algebra or loop-product is re-implemented (§6.1).

use crate::curvature_task::{rotor_angle, ConnGraph, IDENT};
use crate::rotor_holonomy_forward;
use hymeko_clifford::{quat_conjugate, quat_mul};

#[inline]
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// Step 1 — gauge-fix by spanning-tree transport. Returns node frames flat `(n_nodes·4)`,
/// with the root frame the identity.
///
/// # Preconditions
/// `edge_q.len() == g.edges.len() * 4`; `g.tree_transport` is in BFS order from the root.
///
/// # Postconditions
/// The root's frame is the identity; every frame is a unit quaternion (product of units).
///
/// # Panics
/// If `edge_q` has the wrong length.
pub fn tree_gauge_frames(g: &ConnGraph, edge_q: &[f32]) -> Vec<f32> {
    assert_eq!(edge_q.len(), g.edges.len() * 4, "edge_q must be (|E|, 4)");
    let mut frames = vec![0.0f32; g.n_nodes * 4];
    // root frame = identity (index 0 is the hub / root by construction)
    frames[0..4].copy_from_slice(&IDENT);
    for &(node, edge_idx, forward) in &g.tree_transport {
        let (a, b) = g.edges[edge_idx];
        let g_edge = q_at(edge_q, edge_idx);
        // parent = the endpoint that is NOT `node`; transport applies g (forward) or its
        // inverse (reverse) to the parent's already-set frame.
        let (parent, transport) = if forward {
            (a as usize, g_edge) // edge parent→node
        } else {
            (b as usize, quat_conjugate(g_edge)) // edge node→parent, use inverse
        };
        debug_assert!(
            (forward && b as usize == node) || (!forward && a as usize == node),
            "tree_transport orientation inconsistent"
        );
        let fp = q_at(&frames, parent);
        let fnode = quat_mul(transport, fp);
        frames[node * 4..node * 4 + 4].copy_from_slice(&fnode);
    }
    frames
}

/// Step 2 — cotree loop-closure residuals `ρ_e = R̂_b⁻¹ · g_{a→b} · R̂_a`, flat `(m·4)`
/// over the `m = g.cotree.len()` fundamental cycles. Identity on a flat connection.
///
/// # Preconditions
/// `frames.len() == g.n_nodes * 4`, `edge_q.len() == g.edges.len() * 4`.
pub fn cotree_residuals(g: &ConnGraph, edge_q: &[f32], frames: &[f32]) -> Vec<f32> {
    assert_eq!(frames.len(), g.n_nodes * 4);
    assert_eq!(edge_q.len(), g.edges.len() * 4);
    let mut res = vec![0.0f32; g.cotree.len() * 4];
    for (k, &e) in g.cotree.iter().enumerate() {
        let (a, b) = g.edges[e];
        let g_edge = q_at(edge_q, e);
        let fa = q_at(frames, a as usize);
        let fb = q_at(frames, b as usize);
        // rho = conj(fb) * g * fa
        let rho = quat_mul(quat_conjugate(fb), quat_mul(g_edge, fa));
        res[k * 4..k * 4 + 4].copy_from_slice(&rho);
    }
    res
}

/// Step 3 — curvature energy: the mean rotation angle over cotree residuals. `≈ 0` for a
/// flat connection, `> 0` for a curved one. A single scalar; rank samples by it (no training).
pub fn curvature_energy(residuals: &[f32]) -> f32 {
    let m = residuals.len() / 4;
    if m == 0 {
        return 0.0;
    }
    let s: f32 = (0..m).map(|k| rotor_angle(q_at(residuals, k))).sum();
    s / m as f32
}

/// The full one-shot closed Clifford estimate: `curvature_energy(cotree_residuals(gauge))`.
/// The auto-holonomy Step-2 readout, computed from topology + connection alone.
pub fn closed_clifford_curvature(g: &ConnGraph, edge_q: &[f32]) -> f32 {
    let frames = tree_gauge_frames(g, edge_q);
    let res = cotree_residuals(g, edge_q, &frames);
    curvature_energy(&res)
}

/// The **oracle** curvature: mean plaquette holonomy angle read from the *true* fundamental
/// cycles via [`crate::rotor_holonomy_forward`] (the Rust holonomy op). The ceiling the
/// estimator is measured against; equals the estimator by the gauge identity (tested).
///
/// # Preconditions
/// `edge_q.len() == g.edges.len() * 4`; every cycle has length 3 (wheel plaquettes).
pub fn oracle_curvature(g: &ConnGraph, edge_q: &[f32]) -> f32 {
    let k = 3usize;
    let n_cycles = g.cycles.len();
    // build ordered edge-quats per cycle (with reversed edges conjugated), then reuse the op.
    let mut ordered = vec![0.0f32; n_cycles * k * 4];
    for (c, cyc) in g.cycles.iter().enumerate() {
        for (i, &(e, fwd)) in cyc.iter().enumerate() {
            let q = q_at(edge_q, e);
            let q = if fwd { q } else { quat_conjugate(q) };
            ordered[(c * k + i) * 4..(c * k + i) * 4 + 4].copy_from_slice(&q);
        }
    }
    let (holo, _prefixes) = rotor_holonomy_forward(&ordered, n_cycles, k);
    let s: f32 = (0..n_cycles).map(|c| rotor_angle(q_at(&holo, c))).sum();
    s / n_cycles as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curvature_task::{sample_connection, wheel_graph, Rng};

    #[test]
    fn tree_gauge_root_is_identity_and_unit() {
        let g = wheel_graph(12);
        let eq = sample_connection(&g, &mut Rng(3), true, 0.7);
        let frames = tree_gauge_frames(&g, &eq);
        assert!(rotor_angle(q_at(&frames, 0)) < 1e-6, "root not identity");
        for i in 0..g.n_nodes {
            let q = q_at(&frames, i);
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((n - 1.0).abs() < 1e-4, "frame {i} not unit");
        }
    }

    #[test]
    fn estimator_equals_oracle() {
        // The correctness identity: the closed-form gauge estimator recovers the exact
        // plaquette holonomy the oracle reads via rotor_holonomy_forward.
        let g = wheel_graph(16);
        for (seed, curved) in [(1u64, false), (2, true), (3, true)] {
            let eq = sample_connection(&g, &mut Rng(seed), curved, 0.6);
            let est = closed_clifford_curvature(&g, &eq);
            let ora = oracle_curvature(&g, &eq);
            assert!(
                (est - ora).abs() < 1e-3,
                "estimator {est} != oracle {ora} (curved={curved})"
            );
        }
    }

    #[test]
    fn flat_energy_near_zero_curved_positive() {
        let g = wheel_graph(20);
        let flat = closed_clifford_curvature(&g, &sample_connection(&g, &mut Rng(5), false, 0.9));
        let curved = closed_clifford_curvature(&g, &sample_connection(&g, &mut Rng(5), true, 0.9));
        assert!(flat < 1e-2, "flat energy not ~0: {flat}");
        assert!(curved > 0.8, "curved energy too small: {curved}");
        assert!(
            curved > flat + 0.5,
            "no separation: flat {flat} curved {curved}"
        );
    }

    /// Performance: the one-shot closed Clifford estimate is cheap (`O(|E|)`). Assert a
    /// per-sample latency budget on the wheel-24 (the deploy/inference axis, §3). Median of
    /// 5 timed batches after warm-up; budget is generous for CI-host jitter.
    #[test]
    fn perf_estimator_latency_budget() {
        use std::time::Instant;
        let g = wheel_graph(24);
        let samples: Vec<Vec<f32>> = (0..500)
            .map(|s| sample_connection(&g, &mut Rng(s), s % 2 == 0, 0.6))
            .collect();
        // warm-up
        let mut acc = 0.0f32;
        for x in &samples {
            acc += closed_clifford_curvature(&g, x);
        }
        let mut per_sample_us = vec![];
        for _ in 0..5 {
            let t = Instant::now();
            for x in &samples {
                acc += closed_clifford_curvature(&g, x);
            }
            per_sample_us.push(t.elapsed().as_secs_f64() * 1e6 / samples.len() as f64);
        }
        per_sample_us.sort_by(|a, b| a.total_cmp(b));
        let median = per_sample_us[2];
        assert!(acc.is_finite());
        assert!(
            median < 50.0,
            "closed Clifford estimate median {median:.2} us/sample exceeds 50 us budget (wheel-24)"
        );
    }
}
