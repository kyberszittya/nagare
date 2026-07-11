//! Integration — gradients flow through `node_pool` end to end: a learned field is trained to make
//! its per-cell means match a target. `linear(W): x → field  →  node_pool → node  →  MSE(target)`.
//! Proves the composed backward (`x ← W ← node_pool ← loss`) reduces the loss.

use holonomy_learn::{
    linear_backward, linear_forward, node_pool_backward, node_pool_forward, LinearLayer,
};

#[test]
fn learns_through_node_pool() {
    let (npx, in_dim, d, n_cells) = (60usize, 5usize, 3usize, 6usize);
    // Deterministic assignment covering every cell (a stand-in for a quadtree's `assign`).
    let assign: Vec<u32> = (0..npx).map(|p| (p % n_cells) as u32).collect();
    let x: Vec<f32> = (0..npx * in_dim).map(|i| (i as f32 * 0.3).sin()).collect();
    // Reachable target: the pooled node features of a *known* linear map (so loss → ~0 is achievable).
    let w_true = LinearLayer::new(in_dim, d, 99);
    let (target, _) = node_pool_forward(&linear_forward(&w_true, &x), &assign, n_cells, d);
    let n = n_cells * d;

    let mut w = LinearLayer::new(in_dim, d, 3);
    let lr = 0.6;
    let loss_of = |w: &LinearLayer| -> f32 {
        let (node, _) = node_pool_forward(&linear_forward(w, &x), &assign, n_cells, d);
        node.iter()
            .zip(&target)
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / n as f32
    };
    let l0 = loss_of(&w);

    for _ in 0..2000 {
        let field = linear_forward(&w, &x);
        let (node, counts) = node_pool_forward(&field, &assign, n_cells, d);
        let gl: Vec<f32> = node
            .iter()
            .zip(&target)
            .map(|(a, b)| 2.0 * (a - b) / n as f32)
            .collect();
        let grad_field = node_pool_backward(&assign, &gl, &counts, n_cells, d);
        let (_gx, gw) = linear_backward(&w, &x, &grad_field);
        for (v, g) in w.w.iter_mut().zip(&gw.w) {
            *v -= lr * g;
        }
        for (v, g) in w.b.iter_mut().zip(&gw.b) {
            *v -= lr * g;
        }
    }
    let l1 = loss_of(&w);
    assert!(
        l1 < 0.05 * l0,
        "loss did not converge through node_pool: {l0} -> {l1}"
    );
}
