//! Neocognitron N3 — a **2-block S/C deep stack** on a compositional,
//! rotation-varied task, no autograd. Classify a **corner** (L-junction: two
//! edges meeting at 90°, label 1) vs a **straight bar** (one edge, label 0),
//! LENGTH-matched (corner = two arms of length A; bar = one segment of length
//! 2A) so the discriminator is the *configuration* (bent vs straight), not total
//! edge energy. This is the classic Fukushima hierarchy: block-1 S/C detects
//! oriented edges, block-2 S/C detects the vertex as a configuration of edges,
//! and rotation-tolerance must compound through both blocks.
//!
//! Stack: `x → ScBlock(1→K1) → resp1 → ScBlock(K1→K2) → resp2 → mean → linear → BCE`.
//! A/B knobs: `--onelayer` (a single S/C block — does depth help?) and `--c1`
//! (C-cell group C₁ orientation-specific vs the default C₈). Train on two
//! orientations, test AUROC on HELD-OUT orientations.
//!
//! Run: `cargo run --release --example neocognitron_deep -- [--onelayer] [--c1] [out.json]`

use holonomy_learn::{
    adam_step, linear_backward, linear_forward, sc_block_backward, sc_block_forward, AdamState,
    ConvShape, DihedralGroup, LinearLayer, ScBlock,
};
use std::f32::consts::PI;
use std::io::Write;

const G: usize = 24;
const GG: usize = G * G;
const ARM: f32 = 7.0; // arm length A (corner arms A, bar length 2A → matched)
const WIDTH: f32 = 1.1; // stroke half-width

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Distance from point `p` to segment `a→b`.
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

/// Render a corner (`corner=true`) or a straight bar at angle `theta`, plus light
/// noise. Length-matched: corner is two arms of length ARM from centre; bar is a
/// single 2·ARM segment through centre.
fn render(theta: f32, corner: bool, rng: &mut u64) -> Vec<f32> {
    let mut nx = || {
        *rng = rng
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*rng >> 33) as f32) / (u32::MAX as f32) - 0.5
    };
    let cc = (G - 1) as f32 / 2.0;
    let (ct, st) = (theta.cos(), theta.sin());
    // segment endpoints in image coords
    let arm1 = (cc + ARM * ct, cc + ARM * st);
    let segs: [(f32, f32, f32, f32); 2] = if corner {
        // two arms from centre at 90°: along θ and along θ+90°.
        [
            (cc, cc, arm1.0, arm1.1),
            (cc, cc, cc - ARM * st, cc + ARM * ct),
        ]
    } else {
        // one straight bar of length 2A through centre.
        [
            (cc - ARM * ct, cc - ARM * st, arm1.0, arm1.1),
            (cc, cc, cc, cc),
        ]
    };
    let mut img = vec![0.0f32; GG];
    for r in 0..G {
        for c in 0..G {
            let (px, py) = (c as f32, r as f32);
            let mut on = false;
            for &(ax, ay, bx, by) in &segs {
                if (ax, ay) == (bx, by) {
                    continue;
                }
                if seg_dist(px, py, ax, ay, bx, by) <= WIDTH {
                    on = true;
                }
            }
            img[r * G + c] = if on { 1.0 } else { 0.0 } + 0.15 * nx();
        }
    }
    img
}

fn auroc(scores: &[f32], labels: &[u8]) -> f64 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut rs, mut np) = (0.0f64, 0u64);
    for (r, &i) in idx.iter().enumerate() {
        if labels[i] == 1 {
            rs += (r + 1) as f64;
            np += 1;
        }
    }
    let nn = scores.len() as u64 - np;
    if np == 0 || nn == 0 {
        return 0.5;
    }
    (rs - (np * (np + 1) / 2) as f64) / (np * nn) as f64
}

fn main() {
    let one = flag("--onelayer");
    let c1 = flag("--c1");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/neocognitron-deep.json".into());
    let group = if c1 {
        DihedralGroup::new(1, false)
    } else {
        DihedralGroup::new(8, false)
    };
    let tau = 0.3f32;
    let (k1, k2) = (4usize, 4usize);
    // optional `--seed=N` to vary init + data draw for multi-seed medians.
    let seed_base: u64 = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--seed=")
                .map(|s| s.parse::<u64>().unwrap_or(0))
        })
        .unwrap_or(0);

    let mut b1 = ScBlock::new(1, k1, 3, 3, group, tau, 11 + seed_base);
    let mut b2 = ScBlock::new(k1, k2, 3, 3, group, tau, 12 + seed_base);
    let kc = if one { k1 } else { k2 };
    let mut head = LinearLayer::new(kc, 1, 13 + seed_base);
    let s1 = ConvShape {
        c_in: 1,
        h: G,
        w: G,
        pad: 1,
    };
    let s2 = ConvShape {
        c_in: k1,
        h: G,
        w: G,
        pad: 1,
    };

    let (mut a1w, mut a1b, mut a1f) = (
        AdamState::new(b1.conv.w.len()),
        AdamState::new(b1.conv.b.len()),
        AdamState::new(b1.filt.len()),
    );
    let (mut a2w, mut a2b, mut a2f) = (
        AdamState::new(b2.conv.w.len()),
        AdamState::new(b2.conv.b.len()),
        AdamState::new(b2.filt.len()),
    );
    let (mut ahw, mut ahb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));

    // forward → the last block's resp map (K_last, G, G).
    let forward = |b1: &ScBlock, b2: &ScBlock, img: &[f32]| -> Vec<f32> {
        let (r1, _c1) = sc_block_forward(b1, img, s1);
        if one {
            r1
        } else {
            let (r2, _c2) = sc_block_forward(b2, &r1, s2);
            r2
        }
    };
    // per-channel mean feature vector (kc long) for the linear head.
    let chan_feats = |resp: &[f32]| -> Vec<f32> {
        (0..kc)
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

    for _ in 0..1200 {
        for corner in [true, false] {
            let th = pick(&train_ang, &mut rng);
            let img = render(th, corner, &mut rng);
            // forward keeping caches
            let (r1, cache1) = sc_block_forward(&b1, &img, s1);
            let (resp, cache2, r1_for_b2) = if one {
                (r1.clone(), None, r1)
            } else {
                let (r2, c2) = sc_block_forward(&b2, &r1, s2);
                (r2, Some(c2), r1)
            };
            let fv = chan_feats(&resp);
            let logit = linear_forward(&head, &fv)[0];
            let p = 1.0 / (1.0 + (-logit).exp());
            let gl = vec![p - if corner { 1.0 } else { 0.0 }];
            let (gfeat, ghead) = linear_backward(&head, &fv, &gl);
            // grad of per-channel mean → grad_resp map
            let mut gresp = vec![0.0f32; kc * GG];
            for c in 0..kc {
                let g = gfeat[c] / GG as f32;
                for x in gresp[c * GG..c * GG + GG].iter_mut() {
                    *x = g;
                }
            }
            if one {
                let (_gx, g1) = sc_block_backward(&b1, &img, s1, &cache1, &gresp);
                adam_step(&mut b1.conv.w, &g1.conv.w, &mut a1w, 0.02);
                adam_step(&mut b1.conv.b, &g1.conv.b, &mut a1b, 0.02);
                adam_step(&mut b1.filt, &g1.filt, &mut a1f, 0.02);
            } else {
                let (grad_r1, g2) =
                    sc_block_backward(&b2, &r1_for_b2, s2, cache2.as_ref().unwrap(), &gresp);
                let (_gx, g1) = sc_block_backward(&b1, &img, s1, &cache1, &grad_r1);
                adam_step(&mut b2.conv.w, &g2.conv.w, &mut a2w, 0.02);
                adam_step(&mut b2.conv.b, &g2.conv.b, &mut a2b, 0.02);
                adam_step(&mut b2.filt, &g2.filt, &mut a2f, 0.02);
                adam_step(&mut b1.conv.w, &g1.conv.w, &mut a1w, 0.02);
                adam_step(&mut b1.conv.b, &g1.conv.b, &mut a1b, 0.02);
                adam_step(&mut b1.filt, &g1.filt, &mut a1f, 0.02);
            }
            adam_step(&mut head.w, &ghead.w, &mut ahw, 0.02);
            adam_step(&mut head.b, &ghead.b, &mut ahb, 0.02);
        }
    }

    let eval = |angs: &[f32], rng: &mut u64| -> f64 {
        let (mut sc, mut lb) = (Vec::new(), Vec::new());
        for &th in angs {
            for _ in 0..30 {
                for corner in [true, false] {
                    let img = render(th, corner, rng);
                    let resp = forward(&b1, &b2, &img);
                    sc.push(linear_forward(&head, &chan_feats(&resp))[0]);
                    lb.push(corner as u8);
                }
            }
        }
        auroc(&sc, &lb)
    };
    let train_auc = eval(&train_ang, &mut rng);
    let test_auc = eval(&test_ang, &mut rng);

    let depth = if one { "1-block" } else { "2-block" };
    let grp = if c1 { "C_1" } else { "C_8" };
    println!(
        "neocognitron deep [{depth} · {grp}]: train AUROC {train_auc:.3}  HELD-OUT-rotation AUROC {test_auc:.3}"
    );
    let json = format!(
        "{{\n  \"onelayer\": {one},\n  \"c1\": {c1},\n  \"train_auc\": {train_auc:.4},\n  \"test_auc\": {test_auc:.4}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
