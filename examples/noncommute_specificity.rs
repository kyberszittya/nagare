//! **Non-commutativity — the test of holonomy specificity.** Does the framework's `rotor_holonomy`
//! (ordered, non-abelian composition) capture a signal every generic / abelian method is structurally
//! blind to? Task (F-HOLO-9 candidate): two regional holonomies with MATCHED magnitude θ; the class is
//! whether they COMMUTE (parallel axes) or not (perp axes) — signal only in the commutator.
//!
//! Arms (5 seeds, held-out separability AUROC; input = raw edge rotors unless noted):
//!   trivial-entropy      — covariance eigen-entropy of edge log-rotors → chance (edges Haar).
//!   generic-MLP (raw)    — MLP over the 2k×4 raw edges → chance (can't compose loops, cf. F-HOLO-3).
//!   abelian-angle        — θ_A + θ_B via `rotor_holonomy` → chance (magnitudes matched).
//!   MLP-on-holonomies    — MLP over (H_A, H_B) 8-dim (holonomies GIVEN) → isolates extraction vs commutator.
//!   framework-commutator — angle([H_A,H_B]) via `rotor_holonomy`, fixed → SOLVES (the non-abelian op).
//!
//! Verdict: commutator solves AND trivial/generic-MLP(raw)/abelian at chance ⇒ holonomy-specificity
//! demonstrated (the signal is non-abelian; only rotor_holonomy captures it).
//!
//! Run: `cargo run --release --example noncommute_specificity [-- --json <path>]`

use holonomy_learn::{
    adam_step, auroc, commutator_angle, edge_log_field, region_holonomy, regional_angle_sum,
    sample_noncommute, spectral_reg_value_grad, AdamState, CurvatureRng, SpectralEntropyConfig,
};

const K: usize = 6; // edges per region (2K = 12 edges total)
const N_TRAIN: usize = 400;
const N_TEST: usize = 400;
const SEEDS: u64 = 5;

struct Sample {
    edges: Vec<f32>, // 2K*4 raw edge rotors
    holo: Vec<f32>,  // [H_A(4), H_B(4)]
    y: u8,
}

fn gen_set(rng: &mut CurvatureRng, n: usize) -> Vec<Sample> {
    (0..n)
        .map(|i| {
            let class = (i % 2) as u8;
            let theta = 0.8 + rng.f() * 1.0; // same distribution for both classes
            let edges = sample_noncommute(rng, K, theta, class);
            let ha = region_holonomy(&edges, 0, K);
            let hb = region_holonomy(&edges, K, K);
            let holo = [ha[0], ha[1], ha[2], ha[3], hb[0], hb[1], hb[2], hb[3]].to_vec();
            Sample {
                edges,
                holo,
                y: class,
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

fn entropy_cfg() -> SpectralEntropyConfig {
    SpectralEntropyConfig {
        lam_0: 1.0,
        lam_a: 0.0,
        lam_b: 1.0,
        lam_kl: 0.0,
        ..SpectralEntropyConfig::default()
    }
}

#[derive(Clone, Copy)]
struct TrainCfg {
    hidden: usize,
    lr: f32,
    epochs: usize,
    seed: u64,
}

/// 2-layer MLP over feature vectors, Adam-trained. Held-out separability AUROC.
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
    let cfg = entropy_cfg();
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let t0 = std::time::Instant::now();

    println!("== Non-commutativity specificity test (K={K} edges/region, {N_TRAIN}/{N_TEST}, {SEEDS} seeds) ==\n");

    let names = [
        "trivial-entropy",
        "generic-MLP (raw)",
        "abelian-angle",
        "MLP-on-holonomies",
        "framework-commutator",
    ];
    let mut cols: Vec<Vec<f64>> = vec![vec![]; names.len()];
    for s in 0..SEEDS {
        let mut rng = CurvatureRng(101 + s);
        let tr = gen_set(&mut rng, N_TRAIN);
        let te = gen_set(&mut rng, N_TEST);
        let (ytr, yte): (Vec<u8>, Vec<u8>) = (
            tr.iter().map(|x| x.y).collect(),
            te.iter().map(|x| x.y).collect(),
        );
        // fixed scalar arms (rank the readout)
        let triv: Vec<f32> = te
            .iter()
            .map(|x| {
                let f = edge_log_field(&x.edges);
                spectral_reg_value_grad(&f, f.len() / 3, 3, &cfg, 1.0).0
            })
            .collect();
        let ab: Vec<f32> = te.iter().map(|x| regional_angle_sum(&x.edges, K)).collect();
        let com: Vec<f32> = te.iter().map(|x| commutator_angle(&x.edges, K)).collect();
        // learned arms
        let raw = TrainCfg {
            hidden: 64,
            lr: 0.01,
            epochs: 500,
            seed: s,
        };
        let holo_cfg = TrainCfg {
            hidden: 32,
            lr: 0.02,
            epochs: 500,
            seed: s,
        };
        let mlp_raw = run_mlp(
            &tr.iter().map(|x| x.edges.clone()).collect::<Vec<_>>(),
            &ytr,
            &te.iter().map(|x| x.edges.clone()).collect::<Vec<_>>(),
            &yte,
            raw,
        );
        let mlp_holo = run_mlp(
            &tr.iter().map(|x| x.holo.clone()).collect::<Vec<_>>(),
            &ytr,
            &te.iter().map(|x| x.holo.clone()).collect::<Vec<_>>(),
            &yte,
            holo_cfg,
        );
        let vals = [
            sep(&triv, &yte),
            mlp_raw,
            sep(&ab, &yte),
            mlp_holo,
            sep(&com, &yte),
        ];
        for (i, v) in vals.into_iter().enumerate() {
            cols[i].push(v);
        }
        println!("  seed {s} done  [{:.1}s]", t0.elapsed().as_secs_f32());
    }
    let med: Vec<f64> = cols.iter().map(|c| median(c)).collect();
    println!("\n  {:<22} {:>8}", "arm", "median");
    for (n, mv) in names.iter().zip(&med) {
        println!("  {n:<22} {mv:>8.3}");
    }

    let (triv, mlp_raw, ab, mlp_holo, com) = (med[0], med[1], med[2], med[3], med[4]);
    let generic_max = triv.max(mlp_raw).max(ab);
    let specific = com >= 0.90 && generic_max <= 0.60;
    println!("\n== VERDICT — is the signal holonomy-specific (non-abelian)? ==");
    println!(
        "  framework-commutator {com:.3} vs generic/abelian max {generic_max:.3} (trivial {triv:.3}, MLP-raw {mlp_raw:.3}, abelian {ab:.3}) => {}",
        if specific {
            "HOLONOMY-SPECIFIC — only rotor_holonomy captures the non-abelian signal"
        } else if com >= generic_max + 0.15 {
            "commutator >> generic (specificity supported; a generic arm slightly above chance — see below)"
        } else {
            "NOT demonstrated — a generic arm also captures it"
        }
    );
    println!(
        "  nuance — MLP-on-holonomies {mlp_holo:.3}: {}",
        if mlp_holo >= 0.85 {
            "given the holonomies, an MLP learns the commutator ⇒ the specificity is in the LOOP EXTRACTION (which generic raw methods can't do)"
        } else {
            "even given the holonomies the non-abelian commutator resists a generic MLP"
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
            "{{\n  \"task\":\"noncommutativity_specificity\",\"k\":{K},\"seeds\":{SEEDS},\n  \
             \"holonomy_specific\":{specific},\"framework_commutator\":{com:.4},\
             \"generic_max\":{generic_max:.4},\"mlp_on_holonomies\":{mlp_holo:.4},\n  \
             \"arms\":[{}]\n}}\n",
            arms.join(",")
        );
        std::fs::write(&path, out).expect("write json");
        println!("wrote {path}");
    }
}
