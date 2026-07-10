//! Nagare CV bench on **real datasets** — MNIST (digits) and KTH-TIPS (textures) — comparing three
//! closed-form arms on **upright and randomly-rotated** test sets, so each approach's regime shows.
//!
//! Arms (trained on upright images):
//!   1. **raw-pixel linear** — logistic on the pixels (baseline);
//!   2. **patch-embed** — `patch_project` → flatten → linear → softmax (spatial);
//!   3. **phase-pool `|DFT|`** — rotation-invariant orientation-histogram descriptor → linear.
//!
//! Story: the *spatial* arms win when the target is a spatial pattern with a canonical pose
//! (upright digits) but **collapse under rotation**. The rotation-invariant **phase-pool** shines
//! when the target is a **rotation-nuisance** texture (KTH-TIPS: a rotated brick is still a brick),
//! where it should both hold under rotation *and* be genuinely discriminative.
//!
//! Datasets (dispatch on `--dataset`):
//!   - `mnist` — IDX files `{train,t10k}-{images,labels}-idx*-ubyte` (big-endian header);
//!   - `raw`   — little-endian `{train,test}-{images,labels}.bin` (`n,h,w:u32` + `u8` pixels).
//!
//! Run: `cargo run --release --example cv_bench -- --dataset raw --data ~/nagare_data/kth_tips`

use std::path::Path;

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, feature_stats, linear_backward, linear_forward,
    load_split, patch_project_backward, patch_project_forward, rot_all, spatial_phase_features,
    standardize_with, LinearLayer, PatchConfig, PhaseFeature,
};

fn arg_str(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}
fn arg_usize(name: &str, d: usize) -> usize {
    arg_str(name).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn standardize(f_tr: &mut [f32], tests: &mut [&mut [f32]], dim: usize) {
    let (mu, sd) = feature_stats(f_tr, dim);
    standardize_with(f_tr, &mu, &sd, dim);
    for t in tests.iter_mut() {
        standardize_with(t, &mu, &sd, dim);
    }
}

fn fit_linear(f_tr: &[f32], dim: usize, k: usize, y_tr: &[usize], epochs: usize) -> LinearLayer {
    let mut layer = LinearLayer::new(dim, k, 7);
    let n = y_tr.len();
    for _ in 0..epochs {
        let gl = cross_entropy_k_backward(&linear_forward(&layer, f_tr), y_tr, n, k);
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
fn acc_linear(layer: &LinearLayer, f: &[f32], y: &[usize], k: usize) -> f32 {
    accuracy_k(&linear_forward(layer, f), y, y.len(), k)
}

type PatchModel = (Vec<f32>, Vec<f32>, LinearLayer, PatchConfig);

/// Patch-embed model (trained W, b, readout, cfg). `patch` divides `g`.
fn fit_patch(x: &[f32], g: usize, k: usize, y: &[usize], patch: usize) -> PatchModel {
    const PD: usize = 8;
    let cfg = PatchConfig::new(vec![g, g], vec![patch, patch], 1, PD);
    let n = y.len();
    let mut w: Vec<f32> = (0..cfg.patch_vol() * PD)
        .map(|i| 0.1 * ((i as f32 * 0.7).sin()))
        .collect();
    let mut b = vec![0.0f32; PD];
    let mut ro = LinearLayer::new(cfg.n_patches() * PD, k, 11);
    for _ in 0..150 {
        let (tok, pc) = patch_project_forward(x, &w, &b, n, &cfg);
        let gl = cross_entropy_k_backward(&linear_forward(&ro, &tok), y, n, k);
        let (gt, gro) = linear_backward(&ro, &tok, &gl);
        let (_gx, gw, gb) = patch_project_backward(x, &w, &pc, &gt, &cfg);
        for (wi, gg) in w.iter_mut().zip(&gw) {
            *wi -= 0.2 * gg;
        }
        for (bi, gg) in b.iter_mut().zip(&gb) {
            *bi -= 0.2 * gg;
        }
        for (wi, gg) in ro.w.iter_mut().zip(&gro.w) {
            *wi -= 0.2 * gg;
        }
        for (bi, gg) in ro.b.iter_mut().zip(&gro.b) {
            *bi -= 0.2 * gg;
        }
    }
    (w, b, ro, cfg)
}
fn acc_patch(m: &PatchModel, x: &[f32], y: &[usize], k: usize) -> f32 {
    let (tok, _) = patch_project_forward(x, &m.0, &m.1, y.len(), &m.3);
    accuracy_k(&linear_forward(&m.2, &tok), y, y.len(), k)
}

fn main() {
    let ds = arg_str("--dataset").unwrap_or_else(|| "mnist".into());
    let dir = arg_str("--data").expect("--data <dir>");
    let d = Path::new(&dir);
    let (n_tr, n_te) = (arg_usize("--n-train", 8000), arg_usize("--n-test", 2000));
    let augment = std::env::args().any(|a| a == "--augment");
    let b = 18usize;

    let (tr, te) = (
        load_split(&ds, d, true, n_tr),
        load_split(&ds, d, false, n_te),
    );
    let g = tr.g;
    let k = tr.y.iter().chain(&te.y).copied().max().unwrap() + 1;
    let (nt, ne) = (tr.y.len(), te.y.len());
    let patch = if g.is_multiple_of(7) { 7 } else { g / 8 };
    let x_te_rot = rot_all(&te.x, ne, g);
    // Rotation-augment the TRAINING set when requested (spatial arms learn rotation-robustness).
    let train_x = if augment {
        rot_all(&tr.x, nt, g)
    } else {
        tr.x.clone()
    };
    println!(
        "{ds}: {nt} train, {ne} test, {g}×{g}, {k} classes; train={}, eval upright + rotated.",
        if augment { "rot-augmented" } else { "upright" }
    );

    // Shared fixed-feature arm: build train/upright/rotated features, standardise, fit, eval both.
    type FeatBuild<'a> = dyn Fn(&[f32], usize) -> (Vec<f32>, usize) + 'a;
    let eval_fixed = |build: &FeatBuild| -> (f32, f32) {
        let (mut ftr, dim) = build(&train_x, nt);
        let (mut fup, _) = build(&te.x, ne);
        let (mut fro, _) = build(&x_te_rot, ne);
        standardize(&mut ftr, &mut [&mut fup, &mut fro], dim);
        let m = fit_linear(&ftr, dim, k, &tr.y, 200);
        (
            acc_linear(&m, &fup, &te.y, k),
            acc_linear(&m, &fro, &te.y, k),
        )
    };
    let sp = |r: usize| {
        move |x: &[f32], n: usize| spatial_phase_features(x, n, g, r, b, PhaseFeature::Dft)
    };

    // Fixed-feature arms.
    let pixel = eval_fixed(&|x: &[f32], _n| (x.to_vec(), g * g));
    let phase1 = eval_fixed(&sp(1)); // = global phase-pool
    let phase2 = eval_fixed(&sp(2));
    let phase4 = eval_fixed(&sp(4));
    let phasep = eval_fixed(&sp(patch));
    // Mix: raw pixels ⊕ global phase (spatial signal + rotation-invariant signal).
    let mix = eval_fixed(&|x: &[f32], n| {
        let (ph, pd) = spatial_phase_features(x, n, g, 1, b, PhaseFeature::Dft);
        let dim = g * g + pd;
        let mut f = vec![0.0f32; n * dim];
        for s in 0..n {
            f[s * dim..s * dim + g * g].copy_from_slice(&x[s * g * g..(s + 1) * g * g]);
            f[s * dim + g * g..(s + 1) * dim].copy_from_slice(&ph[s * pd..(s + 1) * pd]);
        }
        (f, dim)
    });

    // Learned patch-embed (spatial).
    let pm = fit_patch(&train_x, g, k, &tr.y, patch);
    let patch_arm = (
        acc_patch(&pm, &te.x, &te.y, k),
        acc_patch(&pm, &x_te_rot, &te.y, k),
    );

    println!("  arm                     upright   rotated   drop");
    let row = |name: &str, r: (f32, f32)| {
        println!("  {name:23} {:.4}    {:.4}    {:+.4}", r.0, r.1, r.1 - r.0)
    };
    row("raw-pixel linear", pixel);
    row("patch-embed (spatial)", patch_arm);
    row("phase-pool R=1 (global)", phase1);
    row("spatial-phase R=2", phase2);
    row("spatial-phase R=4", phase4);
    row(&format!("spatial-phase R={patch}"), phasep);
    row("mix: pixels + phase", mix);
}
