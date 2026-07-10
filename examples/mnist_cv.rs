//! Nagare CV on a **real dataset** (MNIST) — three arms, closed-form:
//!   1. **raw-pixel linear** — logistic on the 784 pixels (the standard MNIST baseline, context);
//!   2. **patch-embed** — `patch_project` (7×7 patches → tokens) → flatten → linear → softmax₁₀
//!      (the framework's spatial patch machinery);
//!   3. **phase-pool `|DFT|`** — rotation-invariant orientation-histogram descriptor → linear.
//!
//! Expectation (honest): the *spatial* models (pixel, patch) fit upright digits well; the
//! phase-pool is rotation-**invariant** and thus spatially blind, so on upright MNIST — where
//! orientation is discriminative and rotation is NOT a nuisance — it should be much weaker. That
//! characterises the phase-pool's scope on real data (it's for rotation-nuisance tasks, not
//! upright digit ID), and motivates a spatial phase map.
//!
//! Run: `cargo run --release --example mnist_cv -- --data ~/nagare_data/mnist [--n-train 8000 --n-test 2000]`

use std::path::Path;

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, linear_backward, linear_forward, orientation_histogram,
    patch_project_backward, patch_project_forward, phase_features, LinearLayer, PatchConfig,
    PhaseFeature,
};

const G: usize = 28;
const KC: usize = 10; // MNIST classes

fn arg_str(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}
fn arg_usize(name: &str, d: usize) -> usize {
    arg_str(name).and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Read an IDX image file → (`n*784` in [-1,1], n). Header is 16 bytes (big-endian counts).
fn read_images(path: &Path, cap: usize) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read images");
    let n = u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize;
    let n = n.min(cap);
    let px = &b[16..16 + n * G * G];
    (
        px.iter().map(|&p| p as f32 / 255.0 * 2.0 - 1.0).collect(),
        n,
    )
}
fn read_labels(path: &Path, cap: usize) -> Vec<usize> {
    let b = std::fs::read(path).expect("read labels");
    b[8..].iter().take(cap).map(|&l| l as usize).collect()
}

/// SGD a linear softmax classifier on fixed features `(n, dim)`; return test accuracy.
fn train_linear(f_tr: &[f32], dim: usize, y_tr: &[usize], f_te: &[f32], y_te: &[usize]) -> f32 {
    let (n_tr, n_te) = (y_tr.len(), y_te.len());
    let mut layer = LinearLayer::new(dim, KC, 7);
    for _ in 0..200 {
        let logits = linear_forward(&layer, f_tr);
        let gl = cross_entropy_k_backward(&logits, y_tr, n_tr, KC);
        let (_gx, grad) = linear_backward(&layer, f_tr, &gl);
        for (w, g) in layer.w.iter_mut().zip(&grad.w) {
            *w -= 0.5 * g;
        }
        for (b, g) in layer.b.iter_mut().zip(&grad.b) {
            *b -= 0.5 * g;
        }
    }
    accuracy_k(&linear_forward(&layer, f_te), y_te, n_te, KC)
}

/// Standardise features by train stats (helps the fixed-feature arms converge).
fn standardize(f_tr: &mut [f32], f_te: &mut [f32], dim: usize) {
    let n = f_tr.len() / dim;
    let mut mu = vec![0.0f32; dim];
    let mut sd = vec![0.0f32; dim];
    for r in f_tr.chunks(dim) {
        for j in 0..dim {
            mu[j] += r[j] / n as f32;
        }
    }
    for r in f_tr.chunks(dim) {
        for j in 0..dim {
            sd[j] += (r[j] - mu[j]).powi(2) / n as f32;
        }
    }
    for s in &mut sd {
        *s = s.sqrt() + 1e-6;
    }
    for buf in [f_tr, f_te] {
        for r in buf.chunks_mut(dim) {
            for j in 0..dim {
                r[j] = (r[j] - mu[j]) / sd[j];
            }
        }
    }
}

/// Patch-embed: `patch_project` → flatten tokens → linear → softmax₁₀. Trains W, b, readout.
fn run_patch_embed(x_tr: &[f32], y_tr: &[usize], x_te: &[f32], y_te: &[usize]) -> f32 {
    const PD: usize = 8;
    let cfg = PatchConfig::new(vec![G, G], vec![7, 7], 1, PD);
    let tok = cfg.n_patches() * PD;
    let (n_tr, n_te) = (y_tr.len(), y_te.len());
    let mut w: Vec<f32> = (0..cfg.patch_vol() * PD)
        .enumerate()
        .map(|(i, _)| 0.15 * ((i as f32 * 0.7).sin()))
        .collect();
    let mut b = vec![0.0f32; PD];
    let mut readout = LinearLayer::new(tok, KC, 11);
    for _ in 0..150 {
        let (tokens, pc) = patch_project_forward(x_tr, &w, &b, n_tr, &cfg);
        let logits = linear_forward(&readout, &tokens);
        let gl = cross_entropy_k_backward(&logits, y_tr, n_tr, KC);
        let (grad_tok, grad_ro) = linear_backward(&readout, &tokens, &gl);
        let (_gx, gw, gb) = patch_project_backward(x_tr, &w, &pc, &grad_tok, &cfg);
        for (wi, g) in w.iter_mut().zip(&gw) {
            *wi -= 0.2 * g;
        }
        for (bi, g) in b.iter_mut().zip(&gb) {
            *bi -= 0.2 * g;
        }
        for (wi, g) in readout.w.iter_mut().zip(&grad_ro.w) {
            *wi -= 0.2 * g;
        }
        for (bi, g) in readout.b.iter_mut().zip(&grad_ro.b) {
            *bi -= 0.2 * g;
        }
    }
    let (tok_te, _) = patch_project_forward(x_te, &w, &b, n_te, &cfg);
    accuracy_k(&linear_forward(&readout, &tok_te), y_te, n_te, KC)
}

fn main() {
    let dir = arg_str("--data").unwrap_or_else(|| {
        format!(
            "{}/nagare_data/mnist",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let d = Path::new(&dir);
    let n_tr = arg_usize("--n-train", 8000);
    let n_te = arg_usize("--n-test", 2000);

    let (x_tr, n_tr) = read_images(&d.join("train-images-idx3-ubyte"), n_tr);
    let y_tr = read_labels(&d.join("train-labels-idx1-ubyte"), n_tr);
    let (x_te, n_te) = read_images(&d.join("t10k-images-idx3-ubyte"), n_te);
    let y_te = read_labels(&d.join("t10k-labels-idx1-ubyte"), n_te);
    println!("MNIST: {n_tr} train, {n_te} test (28×28)");

    // Arm 1 — raw-pixel linear (standardised).
    let (mut px_tr, mut px_te) = (x_tr.clone(), x_te.clone());
    standardize(&mut px_tr, &mut px_te, G * G);
    let pixel = train_linear(&px_tr, G * G, &y_tr, &px_te, &y_te);

    // Arm 2 — patch-embed (spatial).
    let patch = run_patch_embed(&x_tr, &y_tr, &x_te, &y_te);

    // Arm 3 — phase-pool |DFT| (rotation-invariant orientation stats).
    let b = 18usize;
    let (htr, hte) = (
        orientation_histogram(&x_tr, n_tr, G, b),
        orientation_histogram(&x_te, n_te, G, b),
    );
    let (mut ftr, dim) = phase_features(&htr, n_tr, b, PhaseFeature::Dft);
    let (mut fte, _) = phase_features(&hte, n_te, b, PhaseFeature::Dft);
    standardize(&mut ftr, &mut fte, dim);
    let phase = train_linear(&ftr, dim, &y_tr, &fte, &y_te);

    println!("  raw-pixel linear   test acc {pixel:.4}");
    println!("  patch-embed        test acc {patch:.4}");
    println!("  phase-pool |DFT|   test acc {phase:.4}  (rotation-invariant; spatially blind)");
    println!(
        "  reading: spatial arms fit upright digits; the phase-pool is rotation-invariant → weak here\n           (its regime is rotation-nuisance tasks, not upright digit ID)."
    );
}
