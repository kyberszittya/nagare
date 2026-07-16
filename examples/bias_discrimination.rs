//! **Bias discrimination** — does the framework's regional 2nd-order pooling close the gap a generic
//! learner cannot, on the Gate-2 task (F-HOLO-7)? All arms train on the SAME data (5 seeds, held-out
//! separability AUROC); the difference is the INPUT REPRESENTATION (the bias), not the credit rule.
//!
//!   generic-MLP        — 2-layer MLP over the raw m×m extracted field. NO bias (must discover). ~0.66.
//!   block-Laplacian    — G=4 uniform column-block Laplacian roughness features + light MLP head.
//!                        GENERIC 2nd-order spatial bias (control).
//!   block-entropy      — G=4 block covariance eigen-entropy (framework `spectral_reg_value_grad`) +
//!                        light MLP head. The FRAMEWORK's 2nd-order op, applied regionally.
//!   oracle             — |roughness(A)−roughness(B)|, a fixed regional closed-form. ~0.94 ceiling.
//!
//! Reading: block-entropy ≈ oracle ≫ generic-MLP ⇒ the framework's regional 2nd-order pooling is the
//! lever; block-Laplacian ≈ block-entropy ⇒ ANY 2nd-order regional bias (not holonomy-specific).
//!
//! Run: `cargo run --release --example bias_discrimination [-- --json <path>]`

use holonomy_learn::{
    adam_step, auroc, block_entropy_features, block_laplacian_features, extract_curvature_field,
    grid_graph, region_roughness_diff, sample_regional_curvature, AdamState, CurvatureRng,
    GridGraph,
};

const L: usize = 12;
const K_GEN: usize = 3;
const NOISE: f32 = 0.05;
const N_TRAIN: usize = 600;
const N_TEST: usize = 300;
const SEEDS: u64 = 5;
const G_BLOCKS: usize = 4;

struct Sample {
    field: Vec<f32>, // extracted m*m field
    ent: Vec<f32>,   // block-entropy features
    lap: Vec<f32>,   // block-Laplacian features
    oracle: f32,     // |rA-rB|
    y: u8,
}

fn gen_set(g: &GridGraph, rng: &mut CurvatureRng, n: usize) -> Vec<Sample> {
    let m = g.m;
    (0..n)
        .map(|_| {
            let (eq, _t, y) = sample_regional_curvature(g, rng, K_GEN, NOISE);
            let field = extract_curvature_field(g, &eq);
            Sample {
                ent: block_entropy_features(&field, m, G_BLOCKS),
                lap: block_laplacian_features(&field, m, G_BLOCKS),
                oracle: region_roughness_diff(&field, m),
                field,
                y,
            }
        })
        .collect()
}

fn sep(scores: &[f32], labels: &[u8]) -> f64 {
    let a = auroc(scores, labels);
    a.max(1.0 - a)
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Training hyperparameters for [`run_mlp`].
#[derive(Clone, Copy)]
struct TrainCfg {
    hidden: usize,
    lr: f32,
    epochs: usize,
    seed: u64,
}

/// 2-layer MLP over arbitrary feature vectors, Adam-trained. Held-out separability AUROC.
fn run_mlp(xtr: &[Vec<f32>], ytr: &[u8], xte: &[Vec<f32>], yte: &[u8], cfg: TrainCfg) -> f64 {
    let TrainCfg {
        hidden,
        lr,
        epochs,
        seed,
    } = cfg;
    let din = xtr[0].len();
    let mut rng = CurvatureRng(777 + seed);
    let mut w1: Vec<f32> = (0..hidden * din)
        .map(|_| 0.3 * rng.g() / (din as f32).sqrt())
        .collect();
    let mut b1 = vec![0.0f32; hidden];
    let mut w2: Vec<f32> = (0..hidden)
        .map(|_| 0.3 * rng.g() / (hidden as f32).sqrt())
        .collect();
    let mut b2 = 0.0f32;
    let (mut s1, mut sb1, mut s2, mut sb2) = (
        AdamState::new(hidden * din),
        AdamState::new(hidden),
        AdamState::new(hidden),
        AdamState::new(1),
    );
    let fwd = |w1: &[f32], b1: &[f32], w2: &[f32], b2: f32, x: &[f32]| -> (f32, Vec<f32>) {
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut z = b1[j];
            for k in 0..din {
                z += w1[j * din + k] * x[k];
            }
            h[j] = z.tanh();
        }
        let logit = b2 + (0..hidden).map(|j| w2[j] * h[j]).sum::<f32>();
        (logit, h)
    };
    for _ in 0..epochs {
        let (mut gw1, mut gb1, mut gw2, mut gb2) = (
            vec![0.0f32; hidden * din],
            vec![0.0f32; hidden],
            vec![0.0f32; hidden],
            0.0f32,
        );
        for (x, &y) in xtr.iter().zip(ytr) {
            let (logit, h) = fwd(&w1, &b1, &w2, b2, x);
            let dl = sigmoid(logit) - y as f32;
            gb2 += dl;
            for j in 0..hidden {
                gw2[j] += dl * h[j];
                let dz = dl * w2[j] * (1.0 - h[j] * h[j]);
                gb1[j] += dz;
                for k in 0..din {
                    gw1[j * din + k] += dz * x[k];
                }
            }
        }
        let m = xtr.len() as f32;
        for gi in gw1.iter_mut() {
            *gi /= m;
        }
        for gi in gb1.iter_mut() {
            *gi /= m;
        }
        for gi in gw2.iter_mut() {
            *gi /= m;
        }
        gb2 /= m;
        adam_step(&mut w1, &gw1, &mut s1, lr);
        adam_step(&mut b1, &gb1, &mut sb1, lr);
        adam_step(&mut w2, &gw2, &mut s2, lr);
        let mut bb = [b2];
        adam_step(&mut bb, &[gb2], &mut sb2, lr);
        b2 = bb[0];
    }
    let scores: Vec<f32> = xte.iter().map(|x| fwd(&w1, &b1, &w2, b2, x).0).collect();
    sep(&scores, yte)
}

fn main() {
    let g = grid_graph(L);
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let t0 = std::time::Instant::now();

    println!(
        "== Bias discrimination on the Gate-2 task (G={G_BLOCKS} blocks, {N_TRAIN}/{N_TEST}, {SEEDS} seeds) ==\n"
    );

    let names = [
        "generic-MLP",
        "block-Laplacian",
        "block-entropy",
        "oracle |rA-rB|",
    ];
    let mut cols: Vec<Vec<f64>> = vec![vec![]; names.len()];
    for s in 0..SEEDS {
        let mut rng = CurvatureRng(101 + s);
        let tr = gen_set(&g, &mut rng, N_TRAIN);
        let te = gen_set(&g, &mut rng, N_TEST);
        let (ytr, yte): (Vec<u8>, Vec<u8>) = (
            tr.iter().map(|s| s.y).collect(),
            te.iter().map(|s| s.y).collect(),
        );
        let feat = |set: &[Sample], sel: fn(&Sample) -> Vec<f32>| -> Vec<Vec<f32>> {
            set.iter().map(sel).collect()
        };
        // arms — same head hyperparameters for the two block-pooled arms (a fair bias comparison)
        let raw_cfg = TrainCfg {
            hidden: 128,
            lr: 0.01,
            epochs: 700,
            seed: s,
        };
        let head_cfg = TrainCfg {
            hidden: 16,
            lr: 0.03,
            epochs: 400,
            seed: s,
        };
        let mlp = run_mlp(
            &feat(&tr, |s| s.field.clone()),
            &ytr,
            &feat(&te, |s| s.field.clone()),
            &yte,
            raw_cfg,
        );
        let lap = run_mlp(
            &feat(&tr, |s| s.lap.clone()),
            &ytr,
            &feat(&te, |s| s.lap.clone()),
            &yte,
            head_cfg,
        );
        let ent = run_mlp(
            &feat(&tr, |s| s.ent.clone()),
            &ytr,
            &feat(&te, |s| s.ent.clone()),
            &yte,
            head_cfg,
        );
        let orc = sep(&te.iter().map(|s| s.oracle).collect::<Vec<_>>(), &yte);
        for (i, v) in [mlp, lap, ent, orc].into_iter().enumerate() {
            cols[i].push(v);
        }
        println!("  seed {s} done  [{:.1}s]", t0.elapsed().as_secs_f32());
    }
    let med: Vec<f64> = cols.iter().map(|c| median(c)).collect();
    println!("\n  {:<18} {:>8}", "arm", "median");
    for (n, mv) in names.iter().zip(&med) {
        println!("  {n:<18} {mv:>8.3}");
    }

    let (mlp, lap, ent, orc) = (med[0], med[1], med[2], med[3]);
    println!("\n== VERDICT — does regional 2nd-order pooling close the gap? ==");
    let ent_closes = ent >= orc - 0.05;
    let ent_beats_mlp = ent >= mlp + 0.1;
    let holonomy_specific = ent >= lap + 0.05;
    println!(
        "  (1) framework block-entropy {ent:.3} vs generic MLP {mlp:.3} (oracle {orc:.3}) => {}",
        if ent_beats_mlp && ent_closes {
            "CLOSES the gap (regional 2nd-order pooling is the lever)"
        } else if ent_beats_mlp {
            "beats MLP but below oracle"
        } else {
            "does NOT beat the generic MLP"
        }
    );
    println!(
        "  (2) holonomy-specific? entropy {ent:.3} vs generic Laplacian control {lap:.3} => {}",
        if holonomy_specific {
            "framework entropy specifically better"
        } else {
            "NO — any 2nd-order regional bias suffices (not holonomy-specific)"
        }
    );
    println!("\n== done in {:.1}s ==", t0.elapsed().as_secs_f32());

    if let Some(path) = json_path {
        let arms: Vec<String> = names
            .iter()
            .zip(&med)
            .map(|(n, v)| format!("{{\"arm\":\"{n}\",\"median_auroc\":{v:.4}}}"))
            .collect();
        let out = format!(
            "{{\n  \"task\":\"bias_discrimination_gate2\",\"blocks\":{G_BLOCKS},\"seeds\":{SEEDS},\n  \
             \"generic_mlp\":{mlp:.4},\"block_laplacian\":{lap:.4},\"block_entropy\":{ent:.4},\
             \"oracle\":{orc:.4},\n  \"arms\":[{}]\n}}\n",
            arms.join(",")
        );
        std::fs::write(&path, out).expect("write json");
        println!("wrote {path}");
    }
}
