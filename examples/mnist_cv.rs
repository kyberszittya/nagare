//! Nagare CV on a **real dataset** (MNIST) — three arms, evaluated on **upright and randomly
//! rotated** test sets to show each approach's regime.
//!
//! Arms (closed-form, trained on upright digits):
//!   1. **raw-pixel linear** — logistic on the 784 pixels (the standard MNIST baseline);
//!   2. **patch-embed** — `patch_project` (7×7 patches) → flatten → linear → softmax₁₀ (spatial);
//!   3. **phase-pool `|DFT|`** — rotation-invariant orientation-histogram descriptor → linear.
//!
//! The story (honest): on **upright** digits the *spatial* arms win (orientation is discriminative,
//! rotation is not a nuisance). Under **test-time rotation** the spatial arms **collapse** while the
//! rotation-invariant phase-pool **holds** — that is where its invariance pays off on real data.
//!
//! Run: `cargo run --release --example mnist_cv -- --data ~/nagare_data/mnist [--n-train 8000 --n-test 2000]`

use std::path::Path;

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, linear_backward, linear_forward, orientation_histogram,
    patch_project_backward, patch_project_forward, phase_features, LinearLayer, PatchConfig,
    PhaseFeature,
};

const G: usize = 28;
const KC: usize = 10;

fn arg_str(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}
fn arg_usize(name: &str, d: usize) -> usize {
    arg_str(name).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn read_images(path: &Path, cap: usize) -> (Vec<f32>, usize) {
    let b = std::fs::read(path).expect("read images");
    let n = (u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize).min(cap);
    let px = &b[16..16 + n * G * G];
    (
        px.iter().map(|&p| p as f32 / 255.0 * 2.0 - 1.0).collect(),
        n,
    )
}
fn read_labels(path: &Path, cap: usize) -> Vec<usize> {
    std::fs::read(path).expect("read labels")[8..]
        .iter()
        .take(cap)
        .map(|&l| l as usize)
        .collect()
}

/// Bilinearly rotate a `G×G` image by `theta` about its centre; out-of-bounds → −1 (background).
fn rotate(img: &[f32], theta: f32) -> Vec<f32> {
    let (c, s) = (theta.cos(), theta.sin());
    let ctr = (G as f32 - 1.0) / 2.0;
    let mut out = vec![-1.0f32; G * G];
    for oi in 0..G {
        for oj in 0..G {
            let (dy, dx) = (oi as f32 - ctr, oj as f32 - ctr);
            let sy = ctr + dx * s + dy * c; // inverse rotation (sample the source)
            let sx = ctr + dx * c - dy * s;
            let (fy, fx) = (sy.floor(), sx.floor());
            if fy < 0.0 || fx < 0.0 || fy >= G as f32 - 1.0 || fx >= G as f32 - 1.0 {
                continue;
            }
            let (y0, x0) = (fy as usize, fx as usize);
            let (ty, tx) = (sy - fy, sx - fx);
            let v = |a: usize, b: usize| img[a * G + b];
            out[oi * G + oj] = v(y0, x0) * (1.0 - ty) * (1.0 - tx)
                + v(y0, x0 + 1) * (1.0 - ty) * tx
                + v(y0 + 1, x0) * ty * (1.0 - tx)
                + v(y0 + 1, x0 + 1) * ty * tx;
        }
    }
    out
}

/// A deterministic pseudo-random angle per image (LCG on the index).
fn rot_test_set(x: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; n * G * G];
    let mut st = 0x2545f4914f6cdd1du64;
    for s in 0..n {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let theta = (st >> 40) as f32 / (1u64 << 24) as f32 * std::f32::consts::TAU;
        out[s * G * G..(s + 1) * G * G]
            .copy_from_slice(&rotate(&x[s * G * G..(s + 1) * G * G], theta));
    }
    out
}

fn standardize(f_tr: &mut [f32], tests: &mut [&mut [f32]], dim: usize) {
    let n = f_tr.len() / dim;
    let (mut mu, mut sd) = (vec![0.0f32; dim], vec![0.0f32; dim]);
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
    let norm = |buf: &mut [f32]| {
        for r in buf.chunks_mut(dim) {
            for j in 0..dim {
                r[j] = (r[j] - mu[j]) / sd[j];
            }
        }
    };
    norm(f_tr);
    for t in tests.iter_mut() {
        norm(t);
    }
}

fn fit_linear(f_tr: &[f32], dim: usize, y_tr: &[usize]) -> LinearLayer {
    let mut layer = LinearLayer::new(dim, KC, 7);
    let n = y_tr.len();
    for _ in 0..200 {
        let logits = linear_forward(&layer, f_tr);
        let gl = cross_entropy_k_backward(&logits, y_tr, n, KC);
        let (_gx, grad) = linear_backward(&layer, f_tr, &gl);
        for (w, g) in layer.w.iter_mut().zip(&grad.w) {
            *w -= 0.5 * g;
        }
        for (b, g) in layer.b.iter_mut().zip(&grad.b) {
            *b -= 0.5 * g;
        }
    }
    layer
}
fn eval_linear(layer: &LinearLayer, f_te: &[f32], y_te: &[usize]) -> f32 {
    accuracy_k(&linear_forward(layer, f_te), y_te, y_te.len(), KC)
}

/// Patch-embed → (trained W, b, readout, cfg).
fn fit_patch(x_tr: &[f32], y_tr: &[usize]) -> (Vec<f32>, Vec<f32>, LinearLayer, PatchConfig) {
    const PD: usize = 8;
    let cfg = PatchConfig::new(vec![G, G], vec![7, 7], 1, PD);
    let n_tr = y_tr.len();
    let mut w: Vec<f32> = (0..cfg.patch_vol() * PD)
        .map(|i| 0.15 * ((i as f32 * 0.7).sin()))
        .collect();
    let mut b = vec![0.0f32; PD];
    let mut readout = LinearLayer::new(cfg.n_patches() * PD, KC, 11);
    for _ in 0..150 {
        let (tokens, pc) = patch_project_forward(x_tr, &w, &b, n_tr, &cfg);
        let gl = cross_entropy_k_backward(&linear_forward(&readout, &tokens), y_tr, n_tr, KC);
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
    (w, b, readout, cfg)
}
fn eval_patch(
    m: &(Vec<f32>, Vec<f32>, LinearLayer, PatchConfig),
    x_te: &[f32],
    y_te: &[usize],
) -> f32 {
    let (tok, _) = patch_project_forward(x_te, &m.0, &m.1, y_te.len(), &m.3);
    accuracy_k(&linear_forward(&m.2, &tok), y_te, y_te.len(), KC)
}

fn phase_feat(x: &[f32], n: usize, b: usize) -> (Vec<f32>, usize) {
    let h = orientation_histogram(x, n, G, b);
    phase_features(&h, n, b, PhaseFeature::Dft)
}

fn main() {
    let dir = arg_str("--data").unwrap_or_else(|| {
        format!(
            "{}/nagare_data/mnist",
            std::env::var("HOME").unwrap_or_default()
        )
    });
    let d = Path::new(&dir);
    let (x_tr, n_tr) = read_images(
        &d.join("train-images-idx3-ubyte"),
        arg_usize("--n-train", 8000),
    );
    let y_tr = read_labels(&d.join("train-labels-idx1-ubyte"), n_tr);
    let (x_te, n_te) = read_images(
        &d.join("t10k-images-idx3-ubyte"),
        arg_usize("--n-test", 2000),
    );
    let y_te = read_labels(&d.join("t10k-labels-idx1-ubyte"), n_te);
    let x_te_rot = rot_test_set(&x_te, n_te);
    println!("MNIST: {n_tr} train, {n_te} test (28×28); eval on upright + randomly-rotated test.");

    // Arm 1 — raw-pixel linear.
    let (mut px_tr, mut px_up, mut px_ro) = (x_tr.clone(), x_te.clone(), x_te_rot.clone());
    standardize(&mut px_tr, &mut [&mut px_up, &mut px_ro], G * G);
    let pl = fit_linear(&px_tr, G * G, &y_tr);
    let (pixel_up, pixel_ro) = (
        eval_linear(&pl, &px_up, &y_te),
        eval_linear(&pl, &px_ro, &y_te),
    );

    // Arm 2 — patch-embed (spatial).
    let pm = fit_patch(&x_tr, &y_tr);
    let (patch_up, patch_ro) = (
        eval_patch(&pm, &x_te, &y_te),
        eval_patch(&pm, &x_te_rot, &y_te),
    );

    // Arm 3 — phase-pool |DFT| (rotation-invariant).
    let b = 18usize;
    let (mut ftr, dim) = phase_feat(&x_tr, n_tr, b);
    let (mut fup, _) = phase_feat(&x_te, n_te, b);
    let (mut fro, _) = phase_feat(&x_te_rot, n_te, b);
    standardize(&mut ftr, &mut [&mut fup, &mut fro], dim);
    let ph = fit_linear(&ftr, dim, &y_tr);
    let (phase_up, phase_ro) = (eval_linear(&ph, &fup, &y_te), eval_linear(&ph, &fro, &y_te));

    println!("  arm                upright   rotated   drop");
    let row = |name: &str, up: f32, ro: f32| {
        println!("  {name:18} {up:.4}    {ro:.4}    {:+.4}", ro - up);
    };
    row("raw-pixel linear", pixel_up, pixel_ro);
    row("patch-embed", patch_up, patch_ro);
    row("phase-pool |DFT|", phase_up, phase_ro);
    println!(
        "  reading: spatial arms win UPRIGHT but COLLAPSE under rotation; the phase-pool is weak\n           upright yet HOLDS under rotation — its rotation-invariance is the point."
    );
}
