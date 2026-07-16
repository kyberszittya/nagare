//! **Auto-holonomy dissociation** — the discriminating task where trivial entropy fails
//! but holonomy succeeds (Step 1 gate), then the closed Clifford one-shot estimator vs the
//! trained (iterative) baseline (Step 2 A/B). No autograd anywhere; the holonomy arms are
//! closed-form and untrained.
//!
//! Arms (identical data per seed, held-out AUROC, separability `max(a, 1-a)`):
//!   trivial-entropy  — covariance eigen-entropy of edge log-rotors (`spectral_reg_value_grad`).
//!                      MUST be at chance (matched Haar marginals) → the F-HOLO-2 gate.
//!   trivial-mean     — ‖mean edge log-rotor‖ (sanity → chance).
//!   MLP              — 2-layer MLP over flattened edge rotors, Adam-trained (the learned/
//!                      iterative baseline; the only arm that trains).
//!   oracle           — mean plaquette holonomy angle via `rotor_holonomy_forward` (ceiling).
//!   closed-Clifford  — tree gauge-fix + cotree residual curvature energy (ONE-SHOT, no train).
//!
//! Gate (5 seeds): trivial-entropy ≤ 0.60  AND  oracle ≥ 0.90  ⇒ the metric can rank
//! auto-holonomy. Then closed-Clifford is the headline: it should reach the oracle without
//! training and beat trivial by a wide margin.
//!
//! Run: `cargo run --release --example auto_holonomy_dissociation`
//! JSON: `--json <path>` writes the per-arm / per-flux results.

use holonomy_learn::{
    adam_step, auroc, closed_clifford_curvature, edge_log_field, oracle_curvature,
    sample_connection, spectral_reg_value_grad, wheel_graph, AdamState, ConnGraph, CurvatureRng,
    SpectralEntropyConfig,
};
use std::time::Instant;

const N_RIM: usize = 24; // 48 edges, 25 nodes, 24 plaquettes
const N_TRAIN: usize = 200;
const N_TEST: usize = 200;
const SEEDS: u64 = 5;

/// A labelled sample set: edge-rotor vectors and 0/1 (flat/curved) labels.
struct Data {
    x: Vec<Vec<f32>>,
    y: Vec<u8>,
}

/// Generate `n` interleaved-label samples on graph `g`.
fn gen_set(g: &ConnGraph, rng: &mut CurvatureRng, n: usize, theta_min: f32) -> Data {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for k in 0..n {
        let lab = (k % 2) as u8; // 0 flat, 1 curved
        x.push(sample_connection(g, rng, lab == 1, theta_min));
        y.push(lab);
    }
    Data { x, y }
}

/// Separability = symmetric AUROC (robust to which class scores higher).
fn sep(scores: &[f32], labels: &[u8]) -> f64 {
    let a = auroc(scores, labels);
    a.max(1.0 - a)
}

fn entropy_cfg() -> SpectralEntropyConfig {
    // pass-through entropy: reg = H_norm (lam_a=0, lam_b=1, lam_kl=0)
    SpectralEntropyConfig {
        lam_0: 1.0,
        lam_a: 0.0,
        lam_b: 1.0,
        lam_kl: 0.0,
        ..SpectralEntropyConfig::default()
    }
}

/// trivial-entropy score: covariance eigen-entropy of the edge log-rotor field.
fn trivial_entropy(eq: &[f32], cfg: &SpectralEntropyConfig) -> f32 {
    let field = edge_log_field(eq);
    let n_edges = field.len() / 3;
    spectral_reg_value_grad(&field, n_edges, 3, cfg, 1.0).0
}

/// trivial-mean score: magnitude of the mean edge log-rotor (sanity → chance).
fn trivial_mean(eq: &[f32]) -> f32 {
    let field = edge_log_field(eq);
    let n = field.len() / 3;
    let mut m = [0.0f32; 3];
    for e in 0..n {
        for c in 0..3 {
            m[c] += field[e * 3 + c];
        }
    }
    (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt() / n as f32
}

/// The learned/iterative baseline: a 2-layer MLP over the flattened edge rotors, Adam-trained
/// on the SAME data. Returns held-out separability AUROC. (Reuses `adam_step`/`AdamState`.)
fn run_mlp(train: &Data, test: &Data, hidden: usize, lr: f32, epochs: usize, seed: u64) -> f64 {
    let din = train.x[0].len();
    let mut rng = CurvatureRng(777 + seed);
    let mut w1: Vec<f32> = (0..hidden * din)
        .map(|_| 0.3 * rng.g() / (din as f32).sqrt())
        .collect();
    let mut b1 = vec![0.0f32; hidden];
    let mut w2: Vec<f32> = (0..hidden)
        .map(|_| 0.3 * rng.g() / (hidden as f32).sqrt())
        .collect();
    let mut b2 = 0.0f32;
    let (mut sw1, mut sb1, mut sw2, mut sb2) = (
        AdamState::new(hidden * din),
        AdamState::new(hidden),
        AdamState::new(hidden),
        AdamState::new(1),
    );
    let sigmoid = |x: f32| 1.0 / (1.0 + (-x).exp());
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
        for (x, &y) in train.x.iter().zip(&train.y) {
            let (logit, h) = fwd(&w1, &b1, &w2, b2, x);
            let dlogit = sigmoid(logit) - y as f32;
            gb2 += dlogit;
            for j in 0..hidden {
                gw2[j] += dlogit * h[j];
                let dz = dlogit * w2[j] * (1.0 - h[j] * h[j]);
                gb1[j] += dz;
                for k in 0..din {
                    gw1[j * din + k] += dz * x[k];
                }
            }
        }
        let m = train.x.len() as f32;
        for g in gw1.iter_mut() {
            *g /= m;
        }
        for g in gb1.iter_mut() {
            *g /= m;
        }
        for g in gw2.iter_mut() {
            *g /= m;
        }
        gb2 /= m;
        adam_step(&mut w1, &gw1, &mut sw1, lr);
        adam_step(&mut b1, &gb1, &mut sb1, lr);
        adam_step(&mut w2, &gw2, &mut sw2, lr);
        let mut bb = [b2];
        adam_step(&mut bb, &[gb2], &mut sb2, lr);
        b2 = bb[0];
    }
    let scores: Vec<f32> = test.x.iter().map(|x| fwd(&w1, &b1, &w2, b2, x).0).collect();
    sep(&scores, &test.y)
}

/// Median of a slice (sorts a copy).
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

/// Per-arm separability for one seed at one flux level.
struct ArmScores {
    triv_ent: f64,
    triv_mean: f64,
    mlp: f64,
    mlp_big: f64,
    oracle: f64,
    closed: f64,
}

fn run_seed(g: &ConnGraph, seed: u64, theta_min: f32, cfg: &SpectralEntropyConfig) -> ArmScores {
    let mut rng = CurvatureRng(101 + seed);
    // the strong MLP gets MORE data (3x) so a weak score cannot be blamed on undertraining.
    let train = gen_set(g, &mut rng, 3 * N_TRAIN, theta_min);
    let test = gen_set(g, &mut rng, N_TEST, theta_min);
    // unsupervised scalar arms — score test directly, no training
    let te: Vec<f32> = test.x.iter().map(|x| trivial_entropy(x, cfg)).collect();
    let tm: Vec<f32> = test.x.iter().map(|x| trivial_mean(x)).collect();
    let or: Vec<f32> = test.x.iter().map(|x| oracle_curvature(g, x)).collect();
    let cc: Vec<f32> = test
        .x
        .iter()
        .map(|x| closed_clifford_curvature(g, x))
        .collect();
    // capacity-matched MLP (hidden 16) and a strong one (hidden 64, 3x data, 400 epochs)
    let train_small = Data {
        x: train.x[..N_TRAIN].to_vec(),
        y: train.y[..N_TRAIN].to_vec(),
    };
    ArmScores {
        triv_ent: sep(&te, &test.y),
        triv_mean: sep(&tm, &test.y),
        mlp: run_mlp(&train_small, &test, 16, 0.05, 200, seed),
        mlp_big: run_mlp(&train, &test, 64, 0.03, 400, seed),
        oracle: sep(&or, &test.y),
        closed: sep(&cc, &test.y),
    }
}

fn main() {
    let cfg = entropy_cfg();
    let g = wheel_graph(N_RIM);
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let t0 = Instant::now();

    println!(
        "== auto-holonomy dissociation: wheel n_rim={N_RIM} ({} edges, {} plaquettes), \
         {N_TRAIN} train / {N_TEST} test, {SEEDS} seeds ==",
        2 * N_RIM,
        N_RIM
    );

    // ---- main table at theta_min = 0.6 ----
    let theta_main = 0.6f32;
    let mut per_arm: Vec<(&str, Vec<f64>)> = vec![
        ("trivial-entropy", vec![]),
        ("trivial-mean", vec![]),
        ("MLP h16", vec![]),
        ("MLP h64 (strong)", vec![]),
        ("oracle", vec![]),
        ("closed-Clifford", vec![]),
    ];
    for s in 0..SEEDS {
        let a = run_seed(&g, s, theta_main, &cfg);
        per_arm[0].1.push(a.triv_ent);
        per_arm[1].1.push(a.triv_mean);
        per_arm[2].1.push(a.mlp);
        per_arm[3].1.push(a.mlp_big);
        per_arm[4].1.push(a.oracle);
        per_arm[5].1.push(a.closed);
        println!(
            "  seed {s}: triv-ent {:.3}  triv-mean {:.3}  MLP16 {:.3}  MLP64 {:.3}  oracle {:.3}  \
             closed {:.3}  [{:.1}s]",
            a.triv_ent,
            a.triv_mean,
            a.mlp,
            a.mlp_big,
            a.oracle,
            a.closed,
            t0.elapsed().as_secs_f32()
        );
    }
    println!("\n  {:<18} {:>8}", "arm", "median");
    let med: Vec<(String, f64)> = per_arm
        .iter()
        .map(|(name, v)| (name.to_string(), median(v)))
        .collect();
    for (name, m) in &med {
        println!("  {name:<18} {m:>8.3}");
    }

    // ---- Step 1 GATE ----
    let triv = med[0].1;
    let mlp_strong = med[3].1;
    let oracle = med[4].1;
    let closed = med[5].1;
    let gate = triv <= 0.60 && oracle >= 0.90;
    println!(
        "\n== STEP 1 GATE: trivial-entropy {triv:.3} <= 0.60 AND oracle {oracle:.3} >= 0.90  =>  {} ==",
        if gate { "PASS (metric is valid)" } else { "FAIL (broken/leaky metric — STOP)" }
    );
    if gate {
        let verdict = if closed >= oracle - 0.02 {
            "closed-Clifford REACHES the oracle (one-shot, no training)"
        } else if closed >= mlp_strong {
            "closed-Clifford matches/beats the trained MLP"
        } else {
            "closed-Clifford below MLP — investigate"
        };
        println!(
            "== STEP 2 A/B: closed-Clifford {closed:.3} vs oracle {oracle:.3} vs strong-MLP {mlp_strong:.3} vs trivial {triv:.3}\n   => {verdict} ==",
        );
    }

    // ---- flux sweep (worst-case input): AUROC vs theta_min ----
    // light path: the h16 MLP + the (untrained) closed / oracle / trivial arms — the strong
    // MLP is not re-trained per flux level (it never approaches the closed-form arm).
    println!("\n== flux sweep: separability vs theta_min (median over {SEEDS} seeds) ==");
    println!(
        "  {:>9} {:>10} {:>8} {:>10}",
        "theta_min", "triv-ent", "MLP16", "closed"
    );
    let thetas = [0.1f32, 0.2, 0.4, 0.6, 0.9, 1.4];
    let mut sweep: Vec<(f32, f64, f64, f64, f64)> = vec![];
    for &th in &thetas {
        let mut te = vec![];
        let mut mlp = vec![];
        let mut cc = vec![];
        let mut or = vec![];
        for s in 0..SEEDS {
            let mut rng = CurvatureRng(101 + s);
            let train = gen_set(&g, &mut rng, N_TRAIN, th);
            let test = gen_set(&g, &mut rng, N_TEST, th);
            let tev: Vec<f32> = test.x.iter().map(|x| trivial_entropy(x, &cfg)).collect();
            let ccv: Vec<f32> = test
                .x
                .iter()
                .map(|x| closed_clifford_curvature(&g, x))
                .collect();
            let orv: Vec<f32> = test.x.iter().map(|x| oracle_curvature(&g, x)).collect();
            te.push(sep(&tev, &test.y));
            mlp.push(run_mlp(&train, &test, 16, 0.05, 200, s));
            cc.push(sep(&ccv, &test.y));
            or.push(sep(&orv, &test.y));
        }
        let (mte, mmlp, mcc, mor) = (median(&te), median(&mlp), median(&cc), median(&or));
        sweep.push((th, mte, mmlp, mcc, mor));
        println!("  {th:>9.2} {mte:>10.3} {mmlp:>8.3} {mcc:>10.3}");
    }

    println!("\n== done in {:.1}s ==", t0.elapsed().as_secs_f32());

    if let Some(path) = json_path {
        let arms_json: Vec<String> = med
            .iter()
            .map(|(name, m)| format!("{{\"arm\":\"{name}\",\"median_auroc\":{m:.4}}}"))
            .collect();
        let sweep_json: Vec<String> = sweep
            .iter()
            .map(|(th, te, mlp, cc, or)| {
                format!(
                    "{{\"theta_min\":{th:.2},\"trivial_entropy\":{te:.4},\"mlp\":{mlp:.4},\
                     \"closed_clifford\":{cc:.4},\"oracle\":{or:.4}}}"
                )
            })
            .collect();
        let out = format!(
            "{{\n  \"task\":\"auto_holonomy_curvature_wheel\",\n  \"n_rim\":{N_RIM},\
             \"n_edges\":{},\"n_train\":{N_TRAIN},\"n_test\":{N_TEST},\"seeds\":{SEEDS},\
             \"theta_main\":{theta_main},\n  \"gate_pass\":{gate},\n  \"main\":[{}],\n  \
             \"flux_sweep\":[{}]\n}}\n",
            2 * N_RIM,
            arms_json.join(","),
            sweep_json.join(",")
        );
        std::fs::write(&path, out).expect("write json");
        println!("wrote {path}");
    }
}
