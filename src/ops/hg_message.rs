//! Signed hypergraph message-passing kernels — the propagation core of a signed HGNN
//! convolution on cycles-as-hyperedges.
//!
//! Port of the propagation in `hymeko_neuro/hyperedge/cpml.py` `CapsuleHypergraphRouter`:
//! one node→edge→node round on the **signed** cycle hypergraph, where corner signs
//! `σ_{c,i} ∈ {±1}` scale both incidence directions (star expansion) and a per-node scale
//! `s_v` (e.g. `D_v^{-1/2}`) does symmetric degree normalisation.
//!
//! Two dual kernels compose the full conv (the learnable `vertex_proj` / `edge_mlp` /
//! `route_head` are plain `linear`/MLP, reused — not re-implemented here):
//! - [`hg_node_to_edge_forward`] — signed, scaled **mean** of a cycle's corners → per-edge.
//! - [`hg_edge_to_node_forward`] — signed **sum** of incident edges, per-node scaled → per-node.
//!
//! The per-node scale `s_v` is a **fixed structural quantity** (from degrees), like `signs`
//! and `cycles` — treated as constant, so no gradient flows through it.

/// Signed node→edge pool: `h_e[c] = (1/k) Σ_i σ[c,i]·s[v]·x[v]`, `v = cycles[c,i]`.
///
/// # Preconditions
/// `cycles.len() == n_edges*k`, `signs.len() == n_edges*k`, `x.len() == n_nodes*d`,
/// `scale.len() == n_nodes`; every `cycles` entry `< n_nodes`.
///
/// # Postconditions
/// Returns `h_e` flat `(n_edges, d)`.
pub fn hg_node_to_edge_forward(
    x: &[f32],
    cycles: &[u32],
    signs: &[f32],
    scale: &[f32],
    n_edges: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    assert_eq!(cycles.len(), n_edges * k);
    assert_eq!(signs.len(), n_edges * k);
    let inv_k = 1.0 / k as f32;
    let mut h_e = vec![0.0f32; n_edges * d];
    for c in 0..n_edges {
        let out = &mut h_e[c * d..c * d + d];
        for i in 0..k {
            let v = cycles[c * k + i] as usize;
            let w = signs[c * k + i] * scale[v] * inv_k;
            let xv = &x[v * d..v * d + d];
            for (o, &xj) in out.iter_mut().zip(xv) {
                *o += w * xj;
            }
        }
    }
    h_e
}

/// Backward of [`hg_node_to_edge_forward`] → `grad_x` flat `(n_nodes, d)`.
pub fn hg_node_to_edge_backward(
    cycles: &[u32],
    signs: &[f32],
    scale: &[f32],
    grad_he: &[f32],
    n_nodes: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    let n_edges = cycles.len() / k;
    let inv_k = 1.0 / k as f32;
    let mut grad_x = vec![0.0f32; n_nodes * d];
    for c in 0..n_edges {
        let g = &grad_he[c * d..c * d + d];
        for i in 0..k {
            let v = cycles[c * k + i] as usize;
            let w = signs[c * k + i] * scale[v] * inv_k;
            let gx = &mut grad_x[v * d..v * d + d];
            for (o, &gj) in gx.iter_mut().zip(g) {
                *o += w * gj;
            }
        }
    }
    grad_x
}

/// Signed edge→node scatter: `out[v] = s[v] · Σ_{(c,i): cycles[c,i]=v} σ[c,i]·h_e[c]`.
///
/// # Preconditions
/// `cycles.len() == n_edges*k`, `signs.len() == n_edges*k`, `h_e.len() == n_edges*d`,
/// `scale.len() == n_nodes`.
///
/// # Postconditions
/// Returns `out` flat `(n_nodes, d)`.
pub fn hg_edge_to_node_forward(
    h_e: &[f32],
    cycles: &[u32],
    signs: &[f32],
    scale: &[f32],
    n_nodes: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    let n_edges = cycles.len() / k;
    let mut out = vec![0.0f32; n_nodes * d];
    for c in 0..n_edges {
        let he = &h_e[c * d..c * d + d];
        for i in 0..k {
            let v = cycles[c * k + i] as usize;
            let sg = signs[c * k + i];
            let ov = &mut out[v * d..v * d + d];
            for (o, &hj) in ov.iter_mut().zip(he) {
                *o += sg * hj;
            }
        }
    }
    for v in 0..n_nodes {
        let s = scale[v];
        for o in out[v * d..v * d + d].iter_mut() {
            *o *= s;
        }
    }
    out
}

/// Backward of [`hg_edge_to_node_forward`] → `grad_he` flat `(n_edges, d)`.
pub fn hg_edge_to_node_backward(
    cycles: &[u32],
    signs: &[f32],
    scale: &[f32],
    grad_out: &[f32],
    n_edges: usize,
    k: usize,
    d: usize,
) -> Vec<f32> {
    assert_eq!(cycles.len(), n_edges * k);
    let mut grad_he = vec![0.0f32; n_edges * d];
    for c in 0..n_edges {
        let g = &mut grad_he[c * d..c * d + d];
        for i in 0..k {
            let v = cycles[c * k + i] as usize;
            let w = signs[c * k + i] * scale[v];
            let go = &grad_out[v * d..v * d + d];
            for (o, &gj) in g.iter_mut().zip(go) {
                *o += w * gj;
            }
        }
    }
    grad_he
}

#[cfg(test)]
mod tests {
    use super::*;

    // 3 triangles over 5 nodes, d=2.
    fn fixture() -> (Vec<u32>, Vec<f32>, Vec<f32>, usize, usize, usize) {
        let cycles = vec![0u32, 1, 2, 2, 3, 4, 0, 3, 4];
        let signs = vec![1.0f32, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0];
        let scale = vec![0.7f32, 1.1, 0.5, 0.9, 1.3];
        (cycles, signs, scale, 3, 3, 2) // n_edges, k, d
    }

    fn fd_grad(f: impl Fn(&[f32]) -> f32, p: &[f32]) -> Vec<f32> {
        let eps = 1e-3;
        (0..p.len())
            .map(|i| {
                let (mut a, mut b) = (p.to_vec(), p.to_vec());
                a[i] += eps;
                b[i] -= eps;
                (f(&a) - f(&b)) / (2.0 * eps)
            })
            .collect()
    }

    #[test]
    fn node_to_edge_backward_matches_fd() {
        let (cycles, signs, scale, ne, k, d) = fixture();
        let n = 5;
        let x: Vec<f32> = (0..n * d).map(|i| 0.3 * ((i as f32 * 1.3).sin())).collect();
        let he = hg_node_to_edge_forward(&x, &cycles, &signs, &scale, ne, k, d);
        let grad_he = vec![1.0f32; he.len()];
        let gx = hg_node_to_edge_backward(&cycles, &signs, &scale, &grad_he, n, k, d);
        let num = fd_grad(
            |xf| {
                hg_node_to_edge_forward(xf, &cycles, &signs, &scale, ne, k, d)
                    .iter()
                    .sum()
            },
            &x,
        );
        for (a, b) in gx.iter().zip(&num) {
            assert!((a - b).abs() < 1e-3, "grad_x {a} vs {b}");
        }
    }

    #[test]
    fn edge_to_node_backward_matches_fd() {
        let (cycles, signs, scale, ne, k, d) = fixture();
        let n = 5;
        let he: Vec<f32> = (0..ne * d)
            .map(|i| 0.4 * ((i as f32 * 0.9).cos()))
            .collect();
        let out = hg_edge_to_node_forward(&he, &cycles, &signs, &scale, n, k, d);
        let grad_out = vec![1.0f32; out.len()];
        let ghe = hg_edge_to_node_backward(&cycles, &signs, &scale, &grad_out, ne, k, d);
        let num = fd_grad(
            |hf| {
                hg_edge_to_node_forward(hf, &cycles, &signs, &scale, n, k, d)
                    .iter()
                    .sum()
            },
            &he,
        );
        for (a, b) in ghe.iter().zip(&num) {
            assert!((a - b).abs() < 1e-3, "grad_he {a} vs {b}");
        }
    }

    #[test]
    fn node_to_edge_is_signed_and_scaled() {
        // A single edge, k=1: h_e = σ·s·x (mean of one term). σ=-1, s=0.5.
        let x = vec![2.0f32, -3.0];
        let he = hg_node_to_edge_forward(&x, &[0], &[-1.0], &[0.5], 1, 1, 2);
        assert!((he[0] - (-(0.5 * 2.0))).abs() < 1e-6); // -1.0
        assert!((he[1] - (-(0.5 * -3.0))).abs() < 1e-6); // +1.5
    }

    #[test]
    fn round_trip_composes() {
        // node→edge then edge→node runs and stays finite (the conv propagation shape).
        let (cycles, signs, scale, ne, k, d) = fixture();
        let n = 5;
        let x: Vec<f32> = (0..n * d).map(|i| 0.1 * i as f32).collect();
        let he = hg_node_to_edge_forward(&x, &cycles, &signs, &scale, ne, k, d);
        let out = hg_edge_to_node_forward(&he, &cycles, &signs, &scale, n, k, d);
        assert_eq!(out.len(), n * d);
        assert!(out.iter().all(|v| v.is_finite()));
    }
}
