//! Wire the N-D patch-projection op into a working closed-form model.
//!
//! Task: 8×8 single-channel grids; each of 4 classes places a Gaussian blob in a distinct
//! quadrant (+ noise) — the label is *where* the blob is, a spatial/patch-scale signal.
//! Model: `grid → patch_project (2×2 patches → tokens) → flatten → linear → softmax₄`,
//! trained purely by the composed closed-form backward (softmax_k → linear → patch).
//! Demonstrates the patch-embed learns end-to-end; asserts high held-out accuracy.

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, cross_entropy_k_forward, linear_backward, linear_forward,
    patch_project_backward, patch_project_forward, LinearLayer, PatchConfig,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const G: usize = 8; // grid side
const K: usize = 4; // classes / quadrants

/// Generate `n` labelled 8×8 blob grids, flat `(n, 64)` + labels, scaled to ~[-1,1].
fn make_grids(n: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<usize>) {
    let centers = [(2usize, 2usize), (2, 6), (6, 2), (6, 6)];
    let sigma2 = 1.5f32 * 1.5;
    let mut x = vec![0.0f32; n * G * G];
    let mut y = vec![0usize; n];
    for s in 0..n {
        let c = rng.random_range(0..K);
        y[s] = c;
        let (cy, cx) = centers[c];
        for i in 0..G {
            for j in 0..G {
                let d2 = ((i as f32 - cy as f32).powi(2)) + ((j as f32 - cx as f32).powi(2));
                let v = (-d2 / (2.0 * sigma2)).exp() + 0.15 * (rng.random::<f32>() * 2.0 - 1.0);
                x[s * G * G + i * G + j] = 2.0 * v - 1.0;
            }
        }
    }
    (x, y)
}

#[test]
fn patch_embed_classifies_spatial_grid() {
    let mut rng = StdRng::seed_from_u64(7);
    let (x_tr, y_tr) = make_grids(320, &mut rng);
    let (x_te, y_te) = make_grids(120, &mut rng);
    let (n_tr, n_te) = (320usize, 120usize);

    let cfg = PatchConfig::new(vec![G, G], vec![2, 2], 1, 4); // 16 patches × proj 4 = 64 tokens
    let tok = cfg.n_patches() * cfg.proj_dim;

    // Params: patch W (patch_vol, proj) + b (proj); readout linear (tok → K).
    let mut irng = StdRng::seed_from_u64(99);
    let mut w: Vec<f32> = (0..cfg.patch_vol() * cfg.proj_dim)
        .map(|_| (irng.random::<f32>() * 2.0 - 1.0) * 0.2)
        .collect();
    let mut b = vec![0.0f32; cfg.proj_dim];
    let mut readout = LinearLayer::new(tok, K, 1);
    let lr = 0.1;

    let forward = |w: &[f32], b: &[f32], readout: &LinearLayer, x: &[f32], n: usize| -> Vec<f32> {
        let (tokens, _) = patch_project_forward(x, w, b, n, &cfg);
        linear_forward(readout, &tokens)
    };

    let mut last_loss = 0.0f32;
    for _ in 0..300 {
        let (tokens, pcache) = patch_project_forward(&x_tr, &w, &b, n_tr, &cfg);
        let logits = linear_forward(&readout, &tokens);
        last_loss = cross_entropy_k_forward(&logits, &y_tr, n_tr, K);
        let gl = cross_entropy_k_backward(&logits, &y_tr, n_tr, K);
        let (grad_tokens, grad_readout) = linear_backward(&readout, &tokens, &gl);
        let (_gx, gw, gb) = patch_project_backward(&x_tr, &w, &pcache, &grad_tokens, &cfg);
        for (wi, g) in w.iter_mut().zip(&gw) {
            *wi -= lr * g;
        }
        for (bi, g) in b.iter_mut().zip(&gb) {
            *bi -= lr * g;
        }
        for (wi, g) in readout.w.iter_mut().zip(&grad_readout.w) {
            *wi -= lr * g;
        }
        for (bi, g) in readout.b.iter_mut().zip(&grad_readout.b) {
            *bi -= lr * g;
        }
    }

    let tr_acc = accuracy_k(&forward(&w, &b, &readout, &x_tr, n_tr), &y_tr, n_tr, K);
    let te_acc = accuracy_k(&forward(&w, &b, &readout, &x_te, n_te), &y_te, n_te, K);
    eprintln!(
        "N-D patch-embed on 8×8 grid ({} patches → {tok} tokens): train {tr_acc:.3}  test {te_acc:.3}  loss {last_loss:.4}",
        cfg.n_patches()
    );
    assert!(
        te_acc >= 0.9,
        "patch-embed failed to classify the spatial grid: {te_acc:.3}"
    );
}
