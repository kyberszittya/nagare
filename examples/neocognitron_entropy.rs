//! Neocognitron — the **entropy global-pooling top**, testing the hypothesis
//! (user, 2026-07-14): *in a Neocognitron the entropy feedback can learn object
//! recognition AND pose detection, with a fast/real-time weight update.* No
//! autograd. Three measured claims:
//!
//! 1. **Recognition** — a 1-block `ScBlock` + `global_entropy_pool` top learns
//!    corner-vs-bar and GENERALISES to HELD-OUT rotations (the invariant `Hs`),
//!    where the N3 mean-top hugged chance. A/B: `--mean-top` (the N3 baseline).
//! 2. **Pose** — the *same* pool's principal-axis angle (rotation-EQUIVARIANT)
//!    recovers a bar's orientation; MAE reported (mod π).
//! 3. **Speed** — per-sample forward+backward+update wall time → real-time check.
//!
//! Stack: `x → ScBlock(1→K) → resp → [entropy pool | channel-mean] → linear → BCE`.
//! Run: `cargo run --release --example neocognitron_entropy -- [--mean-top] [--seed=N]`

use holonomy_learn::{
    adam_step, auroc, global_entropy_pool_backward, global_entropy_pool_forward, linear_backward,
    linear_forward, oriented_sobel_bank, sc_block_backward, sc_block_forward, AdamState, ConvShape,
    DihedralGroup, LinearLayer, ScBlock, FEATS_PER_CHANNEL,
};
use std::f32::consts::PI;
use std::io::Write;
use std::time::Instant;

const G: usize = 24;
const GG: usize = G * G;
const ARM: f32 = 7.0;
const WIDTH: f32 = 1.1;

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

fn seg_dist(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> f32 {
    let (vx, vy) = (bx - ax, by - ay);
    let (wx, wy) = (px - ax, py - ay);
    let len2 = vx * vx + vy * vy;
    let t = if len2 > 1e-6 {
        ((wx * vx + wy * vy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let (cx, cy) = (ax + t * vx, ay + t * vy);
    ((px - cx).powi(2) + (py - cy).powi(2)).sqrt()
}

fn render(theta: f32, corner: bool, rng: &mut u64) -> Vec<f32> {
    let mut nx = || {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*rng >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    let cc = (G - 1) as f32 / 2.0;
    let (ct, st) = (theta.cos(), theta.sin());
    let arm1 = (cc + ARM * ct, cc + ARM * st);
    let segs: [(f32, f32, f32, f32); 2] = if corner {
        [
            (cc, cc, arm1.0, arm1.1),
            (cc, cc, cc - ARM * st, cc + ARM * ct),
        ]
    } else {
        [
            (cc - ARM * ct, cc - ARM * st, arm1.0, arm1.1),
            (cc, cc, cc, cc),
        ]
    };
    let mut img = vec![0.0f32; GG];
    for r in 0..G {
        for c in 0..G {
            let (px, py) = (c as f32, r as f32);
            let on = segs.iter().any(|&(ax, ay, bx, by)| {
                (ax, ay) != (bx, by) && seg_dist(px, py, ax, ay, bx, by) <= WIDTH
            });
            img[r * G + c] = if on { 1.0 } else { 0.0 } + 0.15 * nx();
        }
    }
    img
}

/// Oriented warm-start for the S-cell conv `(2K,1,3,3)`: unit `u` gets a rotated
/// Sobel gradient pair at angle `u·π/K`, so the resp map is structured from step
/// 0 and the entropy-pool gradient engages (cold-start fix; cf. the CR
/// warm-start). The conv stays fully learnable — this only seeds it oriented.
fn main() {
    let mean_top = flag("--mean-top");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/neocognitron-entropy.json".into());
    let seed_base: u64 = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--seed=")
                .map(|s| s.parse::<u64>().unwrap_or(0))
        })
        .unwrap_or(0);
    let group = DihedralGroup::new(8, false);
    let tau = 0.3f32;
    let k = 4usize;
    let feat_dim = if mean_top { k } else { k * FEATS_PER_CHANNEL };

    let mut b1 = ScBlock::new(1, k, 3, 3, group, tau, 11 + seed_base);
    b1.conv.w = oriented_sobel_bank(k); // oriented warm-start (cold-start fix)
    let mut head = LinearLayer::new(feat_dim, 1, 13 + seed_base);
    let s1 = ConvShape {
        c_in: 1,
        h: G,
        w: G,
        pad: 1,
    };
    let (mut aw, mut ab, mut af) = (
        AdamState::new(b1.conv.w.len()),
        AdamState::new(b1.conv.b.len()),
        AdamState::new(b1.filt.len()),
    );
    let (mut ahw, mut ahb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));

    // feature extraction (both tops), returning feat + a closure-free backward token.
    let chan_mean = |resp: &[f32]| -> Vec<f32> {
        (0..k)
            .map(|c| resp[c * GG..c * GG + GG].iter().sum::<f32>() / GG as f32)
            .collect()
    };

    let train_ang: Vec<f32> = [0.0f32, 90.0].iter().map(|d| d * PI / 180.0).collect();
    let test_ang: Vec<f32> = [45.0f32, 135.0, 22.5, 67.5, 112.5, 157.5, 200.0, 250.0]
        .iter()
        .map(|d| d * PI / 180.0)
        .collect();

    let mut rng: u64 = 424242 + seed_base.wrapping_mul(7919);
    let pick = |v: &[f32], r: &mut u64| {
        *r = r.wrapping_mul(6364136223846793005).wrapping_add(1);
        v[((*r >> 33) as usize) % v.len()]
    };

    let n_steps = 1200usize;
    let mut update_ns: u128 = 0;
    for _ in 0..n_steps {
        for corner in [true, false] {
            let th = pick(&train_ang, &mut rng);
            let img = render(th, corner, &mut rng);
            let t0 = Instant::now();
            let (resp, cache1) = sc_block_forward(&b1, &img, s1);
            // forward: pick top
            let (fv, ep) = if mean_top {
                (chan_mean(&resp), None)
            } else {
                let ep = global_entropy_pool_forward(&resp, k, G, G);
                (ep.feat.clone(), Some(ep))
            };
            let logit = linear_forward(&head, &fv)[0];
            let p = 1.0 / (1.0 + (-logit).exp());
            let gl = vec![p - if corner { 1.0 } else { 0.0 }];
            let (gfeat, ghead) = linear_backward(&head, &fv, &gl);
            // grad_feat → grad_resp map
            let gresp = if let Some(ep) = ep.as_ref() {
                global_entropy_pool_backward(ep, &resp, &gfeat)
            } else {
                let mut g = vec![0.0f32; k * GG];
                for c in 0..k {
                    let gc = gfeat[c] / GG as f32;
                    for x in g[c * GG..c * GG + GG].iter_mut() {
                        *x = gc;
                    }
                }
                g
            };
            let (_gx, g1) = sc_block_backward(&b1, &img, s1, &cache1, &gresp);
            adam_step(&mut b1.conv.w, &g1.conv.w, &mut aw, 0.02);
            adam_step(&mut b1.conv.b, &g1.conv.b, &mut ab, 0.02);
            adam_step(&mut b1.filt, &g1.filt, &mut af, 0.02);
            adam_step(&mut head.w, &ghead.w, &mut ahw, 0.02);
            adam_step(&mut head.b, &ghead.b, &mut ahb, 0.02);
            update_ns += t0.elapsed().as_nanos();
        }
    }
    let n_updates = n_steps * 2;
    let us_per_update = update_ns as f64 / n_updates as f64 / 1000.0;
    let updates_per_s = 1e9 * n_updates as f64 / update_ns as f64;

    // recognition eval (held-out rotation AUROC)
    let feat_of = |b1: &ScBlock, img: &[f32]| -> Vec<f32> {
        let (resp, _) = sc_block_forward(b1, img, s1);
        if mean_top {
            chan_mean(&resp)
        } else {
            global_entropy_pool_forward(&resp, k, G, G).feat
        }
    };
    let eval = |angs: &[f32], rng: &mut u64| -> f64 {
        let (mut sc, mut lb) = (Vec::new(), Vec::new());
        for &th in angs {
            for _ in 0..30 {
                for corner in [true, false] {
                    let img = render(th, corner, rng);
                    sc.push(linear_forward(&head, &feat_of(&b1, &img))[0]);
                    lb.push(corner as u8);
                }
            }
        }
        auroc(&sc, &lb)
    };
    let train_auc = eval(&train_ang, &mut rng);
    let test_auc = eval(&test_ang, &mut rng);

    // pose eval: principal angle of the dominant channel vs true θ (bars, mod π).
    let mut pose_err = 0.0f64;
    let mut npose = 0u32;
    if !mean_top {
        for deg in (0..180).step_by(15) {
            let th = deg as f32 * PI / 180.0;
            let img = render(th, false, &mut rng);
            let (resp, _) = sc_block_forward(&b1, &img, s1);
            let ep = global_entropy_pool_forward(&resp, k, G, G);
            let cbest = (0..k)
                .max_by(|&i, &j| ep.mass(i).partial_cmp(&ep.mass(j)).unwrap())
                .unwrap();
            if let Some(ang) = ep.principal_angle(cbest) {
                let diff = (ang - th).rem_euclid(PI);
                pose_err += diff.min(PI - diff) as f64;
                npose += 1;
            }
        }
    }
    let pose_mae_deg = if npose > 0 {
        pose_err / npose as f64 * 180.0 / PI as f64
    } else {
        -1.0
    };

    let top = if mean_top {
        "mean-top (N3 baseline)"
    } else {
        "entropy-top"
    };
    println!(
        "[{top}] recog: train AUROC {train_auc:.3}  HELD-OUT {test_auc:.3} | pose MAE {pose_mae_deg:.1}° | update {us_per_update:.1}µs ({updates_per_s:.0}/s)"
    );
    let json = format!(
        "{{\n  \"mean_top\": {mean_top},\n  \"train_auc\": {train_auc:.4},\n  \"test_auc\": {test_auc:.4},\n  \"pose_mae_deg\": {pose_mae_deg:.3},\n  \"us_per_update\": {us_per_update:.3},\n  \"updates_per_s\": {updates_per_s:.1}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
