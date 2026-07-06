//! Scatter-mean: aggregate per-cycle features into per-vertex
//! embeddings.
//!
//! Forward:
//! ```text
//!   H[v][j] = mean_{c: v ∈ c} per_cycle[c][j]
//!         = (Σ_{c: v ∈ c} per_cycle[c][j]) / count[v]
//! ```
//!
//! Backward (commutative reduce, so the gradient is straightforward):
//! ```text
//!   ∂L/∂per_cycle[c][j] = Σ_{i: v_i ∈ c} (∂L/∂H[v_i][j]) / count[v_i]
//! ```
//!
//! Lockless atomic-add not needed since each (c, j) gradient is
//! written exactly once.

use rayon::prelude::*;

/// Scatter-mean forward. Returns `(n_vertices, d)` per-vertex
/// aggregated features.
///
/// # Args
/// - `cycles` : flat `(n_cycles * k)` cycle vertex indices.
/// - `k`      : cycle length.
/// - `per_cycle_features` : flat `(n_cycles * d)` per-cycle features.
/// - `d`      : feature dim.
/// - `n_vertices` : number of vertices.
///
/// # Returns
/// `(per_vertex_features, counts)` — counts is `(n_vertices,)` so
/// backward can reproduce the mean denominator.
pub fn scatter_mean_forward(
    cycles: &[u32],
    k: usize,
    per_cycle_features: &[f32],
    d: usize,
    n_vertices: usize,
) -> (Vec<f32>, Vec<u32>) {
    let n_cycles = cycles.len() / k;
    assert_eq!(per_cycle_features.len(), n_cycles * d);
    let mut out = vec![0.0f32; n_vertices * d];
    let mut counts = vec![0u32; n_vertices];
    for ci in 0..n_cycles {
        let c_start = ci * k;
        let o_start = ci * d;
        for i in 0..k {
            let v = cycles[c_start + i] as usize;
            counts[v] += 1;
            let vd = v * d;
            let pc = &per_cycle_features[o_start..o_start + d];
            for j in 0..d {
                out[vd + j] += pc[j];
            }
        }
    }
    for v in 0..n_vertices {
        if counts[v] > 0 {
            let inv = 1.0 / counts[v] as f32;
            for j in 0..d {
                out[v * d + j] *= inv;
            }
        }
    }
    (out, counts)
}

/// Scatter-mean backward. Given `∂L/∂H` (per-vertex grad), compute
/// `∂L/∂per_cycle` (per-cycle grad).
pub fn scatter_mean_backward(
    cycles: &[u32],
    k: usize,
    grad_per_vertex: &[f32],
    d: usize,
    counts: &[u32],
    n_vertices: usize,
) -> Vec<f32> {
    let n_cycles = cycles.len() / k;
    assert_eq!(grad_per_vertex.len(), n_vertices * d);
    assert_eq!(counts.len(), n_vertices);
    let mut out = vec![0.0f32; n_cycles * d];
    // Compute 1 / count[v] once per vertex.
    let inv_counts: Vec<f32> = counts
        .iter()
        .map(|&c| if c > 0 { 1.0 / c as f32 } else { 0.0 })
        .collect();
    out.par_chunks_mut(d).enumerate().for_each(|(ci, gpc)| {
        let c_start = ci * k;
        for i in 0..k {
            let v = cycles[c_start + i] as usize;
            let inv = inv_counts[v];
            let vd = v * d;
            for j in 0..d {
                gpc[j] += grad_per_vertex[vd + j] * inv;
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scatter_mean_uniform_features() {
        // 2 cycles (0,1,2) and (1,2,3), features all 1s → vertex 0 has
        // 1 cycle, 1,2 have 2, 3 has 1. Mean = 1.0 everywhere.
        let cycles = vec![0, 1, 2, 1, 2, 3];
        let pcf = vec![1.0; 2 * 4]; // (2 cycles, d=4)
        let (out, counts) = scatter_mean_forward(&cycles, 3, &pcf, 4, 4);
        assert_eq!(counts, vec![1, 2, 2, 1]);
        for v in 0..4 {
            for j in 0..4 {
                assert!((out[v * 4 + j] - 1.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn scatter_mean_backward_matches_numerical() {
        let cycles = vec![0u32, 1, 2, 1, 2, 3];
        let k = 3;
        let d = 2;
        let n_v = 4;
        // Random per-cycle feats.
        let pcf = vec![0.3, -0.7, 1.2, 0.5];
        let (_h, counts) = scatter_mean_forward(&cycles, k, &pcf, d, n_v);
        // Loss = sum(h)
        let grad_pv = vec![1.0; n_v * d];
        let grad_pc = scatter_mean_backward(&cycles, k, &grad_pv, d, &counts, n_v);
        let eps = 1e-3;
        let n_cycles = pcf.len() / d;
        for ci in 0..n_cycles {
            for j in 0..d {
                let mut pcf_p = pcf.clone();
                pcf_p[ci * d + j] += eps;
                let mut pcf_m = pcf.clone();
                pcf_m[ci * d + j] -= eps;
                let (h_p, _) = scatter_mean_forward(&cycles, k, &pcf_p, d, n_v);
                let (h_m, _) = scatter_mean_forward(&cycles, k, &pcf_m, d, n_v);
                let l_p: f32 = h_p.iter().sum();
                let l_m: f32 = h_m.iter().sum();
                let num = (l_p - l_m) / (2.0 * eps);
                let ana = grad_pc[ci * d + j];
                assert!(
                    (ana - num).abs() < 1e-2,
                    "cycle {} dim {}: ana={} num={}",
                    ci,
                    j,
                    ana,
                    num
                );
            }
        }
    }
}
