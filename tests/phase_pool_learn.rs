//! Integration — a learned front-end trained end-to-end **through the invariant phase-pool**.
//!
//! `linear(W): x → field  →  phase_pool → feat  →  linear(head) → logits  →  cross-entropy`.
//! This is the global-pooling-backpropagation claim exercised as a whole: gradients must flow
//! `x ← W ← pool ← head` and reduce the loss. Before `phase_pool_backward` existed, the pool was a
//! dead-end forward and `W` could not be trained under the invariant at all.

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, cross_entropy_k_forward, linear_backward, linear_forward,
    phase_pool_backward, phase_pool_dim, phase_pool_forward, LinearLayer,
};

#[test]
fn learns_through_phase_pool() {
    let (g, b, k) = (6usize, 12usize, 2usize);
    let in_dim = 8;
    let field_dim = g * g * 2;
    let nk = phase_pool_dim(b);
    let per_class = 20;
    let n = per_class * k;

    // Distinct per-class inputs (class c activates channel c) + deterministic noise.
    let mut x = vec![0.0f32; n * in_dim];
    let mut y = vec![0usize; n];
    for c in 0..k {
        for r in 0..per_class {
            let s = c * per_class + r;
            y[s] = c;
            x[s * in_dim + c] = 1.0;
            for d in 0..in_dim {
                x[s * in_dim + d] += 0.05 * (((s * in_dim + d) as f32) * 0.7).sin();
            }
        }
    }

    let mut front = LinearLayer::new(in_dim, field_dim, 11);
    let mut head = LinearLayer::new(nk, k, 23);
    let lr = 0.05;
    let clip = 5.0f32; // the pool's 1/m² near the skip threshold can spike grad_field; clip its norm.

    let loss_of = |front: &LinearLayer, head: &LinearLayer| {
        let field = linear_forward(front, &x);
        let feat = phase_pool_forward(&field, n, g, b).feat;
        cross_entropy_k_forward(&linear_forward(head, &feat), &y, n, k)
    };
    let l0 = loss_of(&front, &head);

    for _ in 0..800 {
        let field = linear_forward(&front, &x);
        let out = phase_pool_forward(&field, n, g, b);
        let logits = linear_forward(&head, &out.feat);
        let gl = cross_entropy_k_backward(&logits, &y, n, k);
        let (grad_feat, head_g) = linear_backward(&head, &out.feat, &gl);
        // The load-bearing leg: grad flows from the invariant feature back to the field.
        let mut grad_field = phase_pool_backward(&field, &out, &grad_feat, n, g, b);
        let norm = grad_field.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > clip {
            let sc = clip / norm;
            for v in &mut grad_field {
                *v *= sc;
            }
        }
        let (_grad_x, front_g) = linear_backward(&front, &x, &grad_field);
        for (w, gw) in head.w.iter_mut().zip(&head_g.w) {
            *w -= lr * gw;
        }
        for (bb, gb) in head.b.iter_mut().zip(&head_g.b) {
            *bb -= lr * gb;
        }
        for (w, gw) in front.w.iter_mut().zip(&front_g.w) {
            *w -= lr * gw;
        }
        for (bb, gb) in front.b.iter_mut().zip(&front_g.b) {
            *bb -= lr * gb;
        }
    }

    let l1 = loss_of(&front, &head);
    let field = linear_forward(&front, &x);
    let feat = phase_pool_forward(&field, n, g, b).feat;
    let acc = accuracy_k(&linear_forward(&head, &feat), &y, n, k);

    assert!(
        l1 < 0.6 * l0,
        "loss did not fall through the pool: {l0} -> {l1}"
    );
    assert!(acc >= 0.9, "train acc through the pool too low: {acc}");
}
