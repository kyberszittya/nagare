//! Generalising the quaternion convolution to **hypergraph convolution × dihedral rotation
//! groups**: the vision quat-conv canonicalised a geometric field by one continuous rotor; here
//! the geometric messages of the signed hypergraph conv (`hg_message`) are steered across a
//! finite **dihedral group** `D_n`, giving a dihedral-equivariant hypergraph convolution.
//!
//! Two rigorous properties (no training — these are the *definition* of a group-equivariant conv):
//!  1. **Equivariance** — steering the node features by any `g ∈ D_n` then message-passing equals
//!     message-passing then steering by `g` (`hg_message` commutes with the group action, because
//!     it is a per-component weighted sum and the group acts as a fixed linear map on each
//!     3-vector). So the hypergraph conv is `D_n`-equivariant.
//!  2. **Invariance of the group-pool** — a group convolution (steer to all `|G|` frames →
//!     message-pass → pool over the group) is invariant under a `D_n` transform of the input
//!     (the transform permutes the group index; an orderless pool is unmoved).

use holonomy_learn::{
    dihedral_steer_forward, hg_edge_to_node_forward, hg_node_to_edge_forward, DihedralGroup,
};

// 3 triangles over 6 nodes; geometric node features are 3-vectors (d = 3).
const N: usize = 6;
const K: usize = 3;
const D: usize = 3;

fn fixture() -> (Vec<u32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let cycles = vec![0u32, 1, 2, 2, 3, 4, 1, 4, 5];
    let signs = vec![1.0f32, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, 1.0];
    let scale = vec![0.7f32, 1.1, 0.5, 0.9, 1.3, 0.8];
    let x: Vec<f32> = (0..N * D).map(|i| 0.4 * ((i as f32 * 1.3).sin())).collect();
    (cycles, signs, scale, x)
}

/// Steer a single node/edge field `(m, 3)` by group element `gi` (slice of the full steer).
fn steer_one(v: &[f32], g: DihedralGroup, m: usize, gi: usize) -> Vec<f32> {
    let all = dihedral_steer_forward(v, g, m);
    all[gi * m * D..(gi + 1) * m * D].to_vec()
}

#[test]
fn hg_message_is_dihedral_equivariant() {
    let (cycles, signs, scale, x) = fixture();
    let g = DihedralGroup::new(4, true); // D_4 (8 elements)
    let n_e = cycles.len() / K;

    // Message-pass the un-steered field once.
    let he = hg_node_to_edge_forward(&x, &cycles, &signs, &scale, n_e, K, D);

    for gi in 0..g.order() {
        // steer_g(x) → message-pass
        let x_g = steer_one(&x, g, N, gi);
        let he_from_steered = hg_node_to_edge_forward(&x_g, &cycles, &signs, &scale, n_e, K, D);
        // message-pass(x) → steer_g
        let he_then_steer = steer_one(&he, g, n_e, gi);
        for (a, b) in he_from_steered.iter().zip(&he_then_steer) {
            assert!(
                (a - b).abs() < 1e-5,
                "g{gi}: hg∘steer != steer∘hg ({a} vs {b})"
            );
        }
    }
}

#[test]
fn edge_to_node_is_dihedral_equivariant() {
    let (cycles, signs, scale, _x) = fixture();
    let g = DihedralGroup::new(6, false); // C_6
    let n_e = cycles.len() / K;
    let he: Vec<f32> = (0..n_e * D)
        .map(|i| 0.3 * ((i as f32 * 0.7).cos()))
        .collect();
    let out = hg_edge_to_node_forward(&he, &cycles, &signs, &scale, N, K, D);
    for gi in 0..g.order() {
        let he_g = steer_one(&he, g, n_e, gi);
        let out_from_steered = hg_edge_to_node_forward(&he_g, &cycles, &signs, &scale, N, K, D);
        let out_then_steer = steer_one(&out, g, N, gi);
        for (a, b) in out_from_steered.iter().zip(&out_then_steer) {
            assert!((a - b).abs() < 1e-5, "g{gi}: scatter mismatch ({a} vs {b})");
        }
    }
}

/// One dihedral group convolution round on the hypergraph → per-node group-**pooled** feature.
/// For each group element: steer node features → node→edge → edge→node; then sum over the group
/// (an orderless pool) → `(N, D)`.
fn group_hyperconv(
    x: &[f32],
    cycles: &[u32],
    signs: &[f32],
    scale: &[f32],
    g: DihedralGroup,
) -> Vec<f32> {
    let n_e = cycles.len() / K;
    let steered = dihedral_steer_forward(x, g, N); // (|G|, N, 3)
    let mut pooled = vec![0.0f32; N * D];
    for gi in 0..g.order() {
        let x_g = &steered[gi * N * D..(gi + 1) * N * D];
        let he = hg_node_to_edge_forward(x_g, cycles, signs, scale, n_e, K, D);
        let back = hg_edge_to_node_forward(&he, cycles, signs, scale, N, K, D);
        for (p, v) in pooled.iter_mut().zip(&back) {
            *p += v / g.order() as f32;
        }
    }
    pooled
}

#[test]
fn group_hyperconv_is_dihedral_invariant() {
    let (cycles, signs, scale, x) = fixture();
    let g = DihedralGroup::new(4, true); // D_4
    let base = group_hyperconv(&x, &cycles, &signs, &scale, g);
    // Transform the input by each group element; the group-pooled output must be unchanged.
    for gi in 0..g.order() {
        let x_h = steer_one(&x, g, N, gi);
        let out = group_hyperconv(&x_h, &cycles, &signs, &scale, g);
        for (a, b) in out.iter().zip(&base) {
            assert!(
                (a - b).abs() < 1e-4,
                "group-conv not D_n-invariant under g{gi} ({a} vs {b})"
            );
        }
    }
    eprintln!(
        "Dihedral hypergraph conv: |D_4|={} — equivariant node→edge & edge→node, group-pool invariant.",
        g.order()
    );
}
