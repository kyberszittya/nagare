//! T3 — build a signed cycle pool *from tabular features* (graph-from-tabular).
//!
//! Samples become graph nodes; a **kNN graph** (Euclidean distance in the standardised
//! feature space) gives the edges; each edge's **sign is the sign of the two samples'
//! feature correlation** — a **leakage-free** signal (features only, never labels), so a
//! node classifier over this graph does not cheat. Signed triangles are enumerated with
//! `hymeko_graph::enumerate_simple_cycles_noprune`, giving a `TopKCyclesBatch` the Gömb /
//! Clifford-FIR machinery consumes. This lets the *signed-cycle core* (not just the KAN)
//! run on Iris/California, so T4 can ask whether the graph structure earns its keep.

use std::collections::HashSet;

use hymeko_graph::{enumerate_simple_cycles_noprune, SignedGraph, TopKCyclesBatch};

/// Undirected kNN edges `(min, max)` by Euclidean distance in feature space.
fn knn_edges(x: &[f32], n: usize, d: usize, k_nn: usize) -> Vec<(u32, u32)> {
    let mut set: HashSet<(u32, u32)> = HashSet::new();
    let mut dist = vec![0.0f32; n];
    for i in 0..n {
        for j in 0..n {
            dist[j] = if i == j {
                f32::INFINITY
            } else {
                x[i * d..i * d + d]
                    .iter()
                    .zip(&x[j * d..j * d + d])
                    .map(|(a, b)| (a - b) * (a - b))
                    .sum()
            };
        }
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| dist[a].total_cmp(&dist[b]));
        for &j in idx.iter().take(k_nn) {
            let (u, v) = (i.min(j) as u32, i.max(j) as u32);
            set.insert((u, v));
        }
    }
    let mut edges: Vec<(u32, u32)> = set.into_iter().collect();
    edges.sort();
    edges
}

/// Sign of the centred correlation between samples `i` and `j` (leakage-free: features only).
fn correlation_sign(x: &[f32], i: usize, j: usize, d: usize) -> i8 {
    let (a, b) = (&x[i * d..i * d + d], &x[j * d..j * d + d]);
    let ma = a.iter().sum::<f32>() / d as f32;
    let mb = b.iter().sum::<f32>() / d as f32;
    let dot: f32 = a
        .iter()
        .zip(b)
        .map(|(&ai, &bi)| (ai - ma) * (bi - mb))
        .sum();
    if dot >= 0.0 {
        1
    } else {
        -1
    }
}

/// Result of building the graph: the signed cycle pool + how many nodes appear in a cycle.
pub struct GraphPool {
    /// The signed cycle pool (`k`-cycles over the samples).
    pub batch: TopKCyclesBatch,
    /// Number of enumerated cycles.
    pub n_cycles: usize,
    /// Number of undirected kNN edges.
    pub n_edges: usize,
}

/// Build a `cycle_k`-cycle signed pool from tabular features (kNN + correlation signs).
///
/// # Preconditions
/// `x.len() == n·d`; `cycle_k >= 3`.
///
/// # Panics
/// Panics if `cycle_k < 3` or shapes mismatch.
pub fn build_signed_cycle_pool(
    x: &[f32],
    n: usize,
    d: usize,
    k_nn: usize,
    cycle_k: usize,
) -> GraphPool {
    assert_eq!(x.len(), n * d);
    assert!(cycle_k >= 3);
    let edges = knn_edges(x, n, d, k_nn);
    let (mut eu, mut ev, mut esign) = (Vec::new(), Vec::new(), Vec::new());
    // Sign lookup for both directions.
    let mut sign_of = std::collections::HashMap::<(u32, u32), i8>::new();
    for &(u, v) in &edges {
        let s = correlation_sign(x, u as usize, v as usize, d);
        eu.push(u);
        ev.push(v);
        esign.push(s);
        sign_of.insert((u, v), s);
        sign_of.insert((v, u), s);
    }
    let graph = SignedGraph::from_parts(n as u32, &eu, &ev, &esign);
    let cycles = enumerate_simple_cycles_noprune(&graph, cycle_k);

    let mut cyc_flat = Vec::with_capacity(cycles.len() * cycle_k);
    let mut sign_flat = Vec::with_capacity(cycles.len() * cycle_k);
    for c in &cycles {
        for i in 0..cycle_k {
            cyc_flat.push(c[i]);
            let (a, b) = (c[i], c[(i + 1) % cycle_k]);
            sign_flat.push(*sign_of.get(&(a, b)).unwrap_or(&1));
        }
    }
    GraphPool {
        batch: TopKCyclesBatch {
            cycles: cyc_flat,
            signs: sign_flat,
            scores: vec![0.0f64; cycles.len()],
            k: cycle_k,
        },
        n_cycles: cycles.len(),
        n_edges: edges.len(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builds_triangles_with_edge_signs() {
        // 6 samples in 2 tight clusters → the kNN graph has triangles.
        let x = vec![
            0.0, 0.0, 0.1, 0.1, -0.1, 0.05, // cluster A
            2.0, 2.0, 2.1, 1.9, 1.9, 2.1, // cluster B
        ];
        let pool = build_signed_cycle_pool(&x, 6, 2, 4, 3);
        assert_eq!(pool.batch.k, 3);
        assert_eq!(pool.batch.cycles.len(), pool.n_cycles * 3);
        assert_eq!(pool.batch.signs.len(), pool.n_cycles * 3);
        assert!(pool.batch.signs.iter().all(|&s| s == 1 || s == -1));
        assert!(pool.batch.cycles.iter().all(|&v| (v as usize) < 6));
    }
}
