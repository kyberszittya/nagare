//! Discriminating experiment — **learned vs fixed orientation field, same `|DFT|` invariant**.
//!
//! The fixed central-difference gradient *is* a frozen 3×3 conv (`gx = right−left`, `gy = down−up`).
//! Both arms share the identical pipeline
//!   `image → im2col 3×3 → linear(9→2) → field → phase_pool(|DFT|) → linear head → softmax`
//! and differ ONLY in whether the 9→2 kernel is **frozen** to the central difference (ARM A) or
//! **trained jointly** with the head (ARM B). This asks: now that `phase_pool` is differentiable,
//! does a learned field beat the hand-designed gradient under the same rotation invariant?
//!
//! Both eval axes (upright + randomly-rotated), multi-seed median/IQR. The frozen arm's feature must
//! equal `spatial_phase_features(r=1)` — a built-in plumbing gate.
//!
//! Run: `cargo run --release --example cv_learned_field -- --dataset raw --data ~/nagare_data/kth_tips2 --n-train 3564 --n-test 1188`

use std::path::Path;

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, cross_entropy_k_forward, feature_stats, linear_backward,
    linear_forward, load_split, phase_pool_backward, phase_pool_dim, phase_pool_forward, rot_all,
    spatial_phase_features, standardize_with, LinearLayer, PhaseFeature,
};

fn arg_str(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}
fn arg_usize(name: &str, d: usize) -> usize {
    arg_str(name).and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Edge-clamped 3×3 im2col: `imgs (n,g*g)` → windows `(n*g*g, 9)`, row-major window
/// `[(−1,−1)..(1,1)]` so index 1=up, 3=left, 5=right, 7=down.
fn im2col(imgs: &[f32], n: usize, g: usize) -> Vec<f32> {
    let mut w = vec![0.0f32; n * g * g * 9];
    for s in 0..n {
        let img = &imgs[s * g * g..(s + 1) * g * g];
        for i in 0..g {
            for j in 0..g {
                let base = (s * g * g + i * g + j) * 9;
                let mut idx = 0;
                for di in -1..=1i32 {
                    for dj in -1..=1i32 {
                        let ii = (i as i32 + di).clamp(0, g as i32 - 1) as usize;
                        let jj = (j as i32 + dj).clamp(0, g as i32 - 1) as usize;
                        w[base + idx] = img[ii * g + jj];
                        idx += 1;
                    }
                }
            }
        }
    }
    w
}

/// Frozen central-difference kernel as a 9→2 linear layer: `gx = w[5]−w[3]`, `gy = w[7]−w[1]`.
fn central_diff_conv() -> LinearLayer {
    let mut l = LinearLayer::new(9, 2, 0);
    // w[in*2 + out]; gx (out 0) = w5−w3 (right−left), gy (out 1) = w7−w1 (down−up).
    l.w = vec![0.0f32; 18];
    l.w[10] = 1.0; // in 5 (right) → gx
    l.w[6] = -1.0; // in 3 (left)  → gx
    l.w[15] = 1.0; // in 7 (down)  → gy
    l.w[3] = -1.0; // in 1 (up)    → gy
    l.b = vec![0.0f32; 2];
    l
}

/// Pooled invariant feature of a conv applied to windows: `windows → field → |DFT|`.
fn pooled_feat(conv: &LinearLayer, windows: &[f32], n: usize, g: usize, b: usize) -> Vec<f32> {
    let field = linear_forward(conv, windows);
    phase_pool_forward(&field, n, g, b).feat
}

/// Standardised copy of a feature buffer.
fn standardized(feat: &[f32], mu: &[f32], sd: &[f32], dim: usize) -> Vec<f32> {
    let mut f = feat.to_vec();
    standardize_with(&mut f, mu, sd, dim);
    f
}

fn median_iqr(mut v: Vec<f32>) -> (f32, f32, f32) {
    v.sort_by(|a, b| a.total_cmp(b));
    let q = |p: f32| v[((p * (v.len() - 1) as f32).round() as usize).min(v.len() - 1)];
    (q(0.5), q(0.25), q(0.75))
}

fn main() {
    let ds = arg_str("--dataset").unwrap_or_else(|| "mnist".into());
    let dir = arg_str("--data").expect("--data <dir>");
    let d = Path::new(&dir);
    let (n_tr, n_te) = (arg_usize("--n-train", 8000), arg_usize("--n-test", 2000));
    let seeds = arg_usize("--seeds", 5);
    let epochs = arg_usize("--epochs", 300);
    let b = arg_usize("--b", 18);
    let (head_lr, conv_lr, clip) = (0.5f32, 0.2f32, 5.0f32);

    let tr = load_split(&ds, d, true, n_tr);
    let te = load_split(&ds, d, false, n_te);
    let g = tr.g;
    let k = tr.y.iter().chain(&te.y).copied().max().unwrap() + 1;
    let (nt, ne) = (tr.y.len(), te.y.len());
    let nk = phase_pool_dim(b);

    // Build im2col windows once (train, upright test, rotated test).
    let win_tr = im2col(&tr.x, nt, g);
    let x_te_rot = rot_all(&te.x, ne, g);
    let win_up = im2col(&te.x, ne, g);
    let win_ro = im2col(&x_te_rot, ne, g);
    println!(
        "{ds}: {nt} train, {ne} test, {g}×{g}, {k} classes, b={b} (nk={nk}); learned vs fixed field, {seeds} seeds."
    );

    // Fixed preconditioner + plumbing gate: central-diff feat must equal spatial_phase_features(r=1).
    let cd = central_diff_conv();
    let feat_tr_cd = pooled_feat(&cd, &win_tr, nt, g, b);
    let (mu_cd, sd_cd) = feature_stats(&feat_tr_cd, nk);
    let n_ref = nt.min(4);
    let (ref_feat, ref_dim) =
        spatial_phase_features(&tr.x[..n_ref * g * g], n_ref, g, 1, b, PhaseFeature::Dft);
    assert_eq!(ref_dim, nk);
    let max_gap = ref_feat
        .iter()
        .zip(&feat_tr_cd[..ref_feat.len()])
        .map(|(a, c)| (a - c).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_gap < 1e-3,
        "PLUMBING GATE FAILED: cd-conv+phase_pool != phase-pool R=1 (max gap {max_gap})"
    );
    println!("  plumbing gate ok: frozen field == phase-pool R=1 (max gap {max_gap:.2e})");

    let feat_tr_cd_std = standardized(&feat_tr_cd, &mu_cd, &sd_cd, nk);

    // Each arm standardises with ITS OWN train-feature stats (the fixed arm's are the constant
    // central-diff stats; the learned arm recomputes them for its trained kernel). Only the kernel
    // differs — the standardisation is the same procedure applied to each arm's own features.
    let eval_arm = |conv: &LinearLayer, head: &LinearLayer, mu: &[f32], sd: &[f32]| -> (f32, f32) {
        let up = standardized(&pooled_feat(conv, &win_up, ne, g, b), mu, sd, nk);
        let ro = standardized(&pooled_feat(conv, &win_ro, ne, g, b), mu, sd, nk);
        (
            accuracy_k(&linear_forward(head, &up), &te.y, ne, k),
            accuracy_k(&linear_forward(head, &ro), &te.y, ne, k),
        )
    };

    // Train the 9→2 kernel + head jointly through the pool from an arbitrary initial kernel.
    // Per-epoch detached (BatchNorm-lite) standardisation keeps the head input O(1) as the kernel
    // moves. Returns (upright acc, rotated acc, final train CE) — CE surfaces undertraining.
    let train_learned =
        |conv0: LinearLayer, head_seed: u64, tag: &str, seed: usize| -> (f32, f32, f32) {
            let mut conv = conv0;
            let mut head = LinearLayer::new(nk, k, head_seed);
            let mut last_ce = 0.0f32;
            for ep in 0..epochs {
                let field = linear_forward(&conv, &win_tr);
                let out = phase_pool_forward(&field, nt, g, b);
                let (mu_e, sd_e) = feature_stats(&out.feat, nk);
                let feat_std = standardized(&out.feat, &mu_e, &sd_e, nk);
                let logits = linear_forward(&head, &feat_std);
                let gl = cross_entropy_k_backward(&logits, &tr.y, nt, k);
                let (grad_feat_std, hg) = linear_backward(&head, &feat_std, &gl);
                let mut grad_feat = grad_feat_std; // ∂/∂feat = grad_std / sd (detached stats)
                for r in grad_feat.chunks_mut(nk) {
                    for j in 0..nk {
                        r[j] /= sd_e[j];
                    }
                }
                let mut grad_field = phase_pool_backward(&field, &out, &grad_feat, nt, g, b);
                let norm = grad_field.iter().map(|v| v * v).sum::<f32>().sqrt();
                if norm > clip {
                    let sc = clip / norm;
                    grad_field.iter_mut().for_each(|v| *v *= sc);
                }
                let (_gx, cg) = linear_backward(&conv, &win_tr, &grad_field);
                for (w, g) in conv.w.iter_mut().zip(&cg.w) {
                    *w -= conv_lr * g;
                }
                for (bi, g) in conv.b.iter_mut().zip(&cg.b) {
                    *bi -= conv_lr * g;
                }
                for (w, g) in head.w.iter_mut().zip(&hg.w) {
                    *w -= head_lr * g;
                }
                for (bi, g) in head.b.iter_mut().zip(&hg.b) {
                    *bi -= head_lr * g;
                }
                if ep % 150 == 0 || ep == epochs - 1 {
                    last_ce =
                        cross_entropy_k_forward(&linear_forward(&head, &feat_std), &tr.y, nt, k);
                    println!("    seed {seed} {tag:9} ep {ep:4}/{epochs}  CE {last_ce:.4}");
                }
            }
            let (mu_ln, sd_ln) = feature_stats(&pooled_feat(&conv, &win_tr, nt, g, b), nk);
            let (up, ro) = eval_arm(&conv, &head, &mu_ln, &sd_ln);
            (up, ro, last_ce)
        };

    let mut fx_up = vec![];
    let mut fx_ro = vec![];
    let mut sc_up = vec![];
    let mut sc_ro = vec![];
    let mut ws_up = vec![];
    let mut ws_ro = vec![];
    for seed in 0..seeds {
        // ARM A (fixed): frozen central-diff kernel, train head only.
        let mut head_fx = LinearLayer::new(nk, k, 7 + seed as u64);
        for _ in 0..epochs {
            let gl =
                cross_entropy_k_backward(&linear_forward(&head_fx, &feat_tr_cd_std), &tr.y, nt, k);
            let (_g, hg) = linear_backward(&head_fx, &feat_tr_cd_std, &gl);
            for (w, g) in head_fx.w.iter_mut().zip(&hg.w) {
                *w -= head_lr * g;
            }
            for (bi, g) in head_fx.b.iter_mut().zip(&hg.b) {
                *bi -= head_lr * g;
            }
        }
        let (a_up, a_ro) = eval_arm(&cd, &head_fx, &mu_cd, &sd_cd);
        // ARM B (learned-scratch): random-init kernel — learnability from scratch.
        let (s_up, s_ro, s_ce) = train_learned(
            LinearLayer::new(9, 2, 100 + seed as u64),
            200 + seed as u64,
            "scratch",
            seed,
        );
        // ARM C (learned-warmstart): kernel INIT at central-diff — can training improve the
        // hand-designed field? (the decisive arm: stay ⇒ local optimum, up ⇒ learning wins.)
        let (w_up, w_ro, w_ce) =
            train_learned(central_diff_conv(), 300 + seed as u64, "warmstart", seed);

        println!(
            "  seed {seed}: fixed {a_up:.4}/{a_ro:.4} | scratch {s_up:.4}/{s_ro:.4} (ce {s_ce:.3}) | warmstart {w_up:.4}/{w_ro:.4} (ce {w_ce:.3})"
        );
        fx_up.push(a_up);
        fx_ro.push(a_ro);
        sc_up.push(s_up);
        sc_ro.push(s_ro);
        ws_up.push(w_up);
        ws_ro.push(w_ro);
    }

    let row = |name: &str, v: &[f32]| {
        let (m, lo, hi) = median_iqr(v.to_vec());
        println!("  {name:18} {m:.4}  [{lo:.4}, {hi:.4}]");
    };
    println!("\n== median [IQR] over {seeds} seeds (upright then rotated) ==");
    row("fixed     up", &fx_up);
    row("fixed     ro", &fx_ro);
    row("scratch   up", &sc_up);
    row("scratch   ro", &sc_ro);
    row("warmstart up", &ws_up);
    row("warmstart ro", &ws_ro);
    let med = |v: &[f32]| median_iqr(v.to_vec()).0;
    println!(
        "\n  Δ(scratch−fixed):   up {:+.4}, ro {:+.4}\n  Δ(warmstart−fixed): up {:+.4}, ro {:+.4}",
        med(&sc_up) - med(&fx_up),
        med(&sc_ro) - med(&fx_ro),
        med(&ws_up) - med(&fx_up),
        med(&ws_ro) - med(&fx_ro),
    );
}
