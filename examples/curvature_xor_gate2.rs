//! **Gate 2 — a task where learning is NECESSARY.** XOR-of-regional-roughness on the F-HOLO-4
//! lattice: the two column-halves are each independently smooth/rough, and the class is whether
//! they DIFFER (`(A rough) ⊕ (B rough)`). The discriminative quantity is `|roughness(A)−roughness(B)|`
//! — invisible to any sum/mean-like global scalar (non-monotonic in the XOR), so every fixed
//! closed-form readout is at chance and only a learned nonlinear readout clears it.
//!
//! Arms (5 seeds, held-out separability AUROC). MEASURED verdict (F-HOLO-7): the strict gate FAILS,
//! informatively — every fixed GLOBAL scalar is at chance, the oracle is high, but a generic MLP does
//! NOT close the gap, and the oracle is itself a *fixed* regional closed-form, so the task is not
//! strictly learning-necessary:
//!   trivial-entropy   — covariance eigen-entropy of edge log-rotors → chance (matched marginals).
//!   constant-rotor    — mean plaquette holonomy angle → chance.
//!   global-ChebyCR    — fixed low-order Chebyshev roughness of the WHOLE field → chance.
//!   global-Laplacian  — mean local Laplacian roughness (sum-like) → chance.
//!   linear-on-field   — logistic over the extracted field → chance (XOR not linearly separable).
//!   learned-MLP       — 2-layer MLP over the extracted field, Adam → ~0.66 (partial; CANNOT reach
//!                       the oracle — the regional 2nd-order feature is hard for a generic learner).
//!   oracle            — |roughness(A)−roughness(B)|, a FIXED regional closed-form → ~0.94 (ceiling).
//!
//! Reading: the task defeats every fixed GLOBAL readout AND a generic MLP, but is solvable by a fixed
//! REGIONAL closed-form. So it is a BIAS-discriminating task (the right regional 2nd-order pooling is
//! needed), not a strictly learning-necessary one — a fixed regional feature suffices, and a generic
//! MLP is insufficient. See report `reports/2026-07-17-gate2-learning-necessary.md`.
//!
//! Run: `cargo run --release --example curvature_xor_gate2 [-- --json <path>]`

use holonomy_learn::{
    adam_step, auroc, chebycr_roughness, constant_rotor_energy, edge_log_field,
    extract_curvature_field, grid_graph, laplacian_roughness, region_roughness_diff,
    sample_regional_curvature, spectral_reg_value_grad, AdamState, CurvatureRng, GridGraph,
    SpectralEntropyConfig,
};

const L: usize = 12;
const K_GEN: usize = 3;
const K_FIT: usize = 3;
const NOISE: f32 = 0.05;
const N_TRAIN: usize = 600;
const N_TEST: usize = 300;
const SEEDS: u64 = 5;

struct Data {
    fields: Vec<Vec<f32>>, // extracted plaquette-angle field (m*m) per sample
    edges: Vec<Vec<f32>>,
    y: Vec<u8>,
}

fn gen_set(g: &GridGraph, rng: &mut CurvatureRng, n: usize) -> Data {
    let mut fields = Vec::with_capacity(n);
    let mut edges = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    for _ in 0..n {
        let (eq, _theta, class) = sample_regional_curvature(g, rng, K_GEN, NOISE);
        fields.push(extract_curvature_field(g, &eq));
        edges.push(eq);
        y.push(class);
    }
    Data { fields, edges, y }
}

fn sep(scores: &[f32], labels: &[u8]) -> f64 {
    let a = auroc(scores, labels);
    a.max(1.0 - a)
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

fn trivial_entropy(eq: &[f32], cfg: &SpectralEntropyConfig) -> f32 {
    let f = edge_log_field(eq);
    spectral_reg_value_grad(&f, f.len() / 3, 3, cfg, 1.0).0
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// MLP (`hidden > 0`) or linear (`hidden == 0`) over the extracted field, Adam-trained. AUROC.
fn run_net(train: &Data, test: &Data, hidden: usize, lr: f32, epochs: usize, seed: u64) -> f64 {
    let din = train.fields[0].len();
    let mut rng = CurvatureRng(777 + seed);
    if hidden == 0 {
        // plain logistic (linear)
        let mut w = vec![0.0f32; din];
        let mut b = 0.0f32;
        let (mut sw, mut sb) = (AdamState::new(din), AdamState::new(1));
        for _ in 0..epochs {
            let (mut gw, mut gb) = (vec![0.0f32; din], 0.0f32);
            for (x, &y) in train.fields.iter().zip(&train.y) {
                let logit = x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f32>() + b;
                let dl = sigmoid(logit) - y as f32;
                for k in 0..din {
                    gw[k] += dl * x[k];
                }
                gb += dl;
            }
            let m = train.fields.len() as f32;
            for gi in gw.iter_mut() {
                *gi /= m;
            }
            adam_step(&mut w, &gw, &mut sw, lr);
            let mut bb = [b];
            adam_step(&mut bb, &[gb / m], &mut sb, lr);
            b = bb[0];
        }
        let scores: Vec<f32> = test
            .fields
            .iter()
            .map(|x| x.iter().zip(&w).map(|(xi, wi)| xi * wi).sum::<f32>() + b)
            .collect();
        return sep(&scores, &test.y);
    }
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
        for (x, &y) in train.fields.iter().zip(&train.y) {
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
        let m = train.fields.len() as f32;
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
    let scores: Vec<f32> = test
        .fields
        .iter()
        .map(|x| fwd(&w1, &b1, &w2, b2, x).0)
        .collect();
    sep(&scores, &test.y)
}

fn main() {
    let cfg = entropy_cfg();
    let g = grid_graph(L);
    let m = g.m;
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let t0 = std::time::Instant::now();

    println!(
        "== Gate 2: XOR-of-regional-roughness (learning NECESSARY) — grid L={L} ({m}x{m}), \
         {N_TRAIN}/{N_TEST}, {SEEDS} seeds ==\n"
    );

    let names = [
        "trivial-entropy",
        "constant-rotor",
        "global-ChebyCR",
        "global-Laplacian",
        "linear-on-field",
        "learned-MLP",
        "oracle |rA-rB|",
    ];
    let mut cols: Vec<Vec<f64>> = vec![vec![]; names.len()];
    for s in 0..SEEDS {
        let mut rng = CurvatureRng(101 + s);
        let train = gen_set(&g, &mut rng, N_TRAIN);
        let test = gen_set(&g, &mut rng, N_TEST);
        let te: Vec<f32> = test
            .edges
            .iter()
            .map(|e| trivial_entropy(e, &cfg))
            .collect();
        let cr: Vec<f32> = test
            .fields
            .iter()
            .map(|f| constant_rotor_energy(f))
            .collect();
        let cby: Vec<f32> = test
            .fields
            .iter()
            .map(|f| chebycr_roughness(f, m, K_FIT))
            .collect();
        let lap: Vec<f32> = test
            .fields
            .iter()
            .map(|f| laplacian_roughness(f, m))
            .collect();
        let orc: Vec<f32> = test
            .fields
            .iter()
            .map(|f| region_roughness_diff(f, m))
            .collect();
        let vals = [
            sep(&te, &test.y),
            sep(&cr, &test.y),
            sep(&cby, &test.y),
            sep(&lap, &test.y),
            run_net(&train, &test, 0, 0.05, 300, s), // linear
            run_net(&train, &test, 128, 0.01, 700, s), // strong MLP (fair shot)
            sep(&orc, &test.y),
        ];
        for (i, v) in vals.into_iter().enumerate() {
            cols[i].push(v);
        }
        println!("  seed {s} done  [{:.1}s]", t0.elapsed().as_secs_f32());
    }
    let med: Vec<f64> = cols.iter().map(|c| median(c)).collect();
    println!("\n  {:<18} {:>8}", "arm", "median");
    for (n, mv) in names.iter().zip(&med) {
        println!("  {n:<18} {mv:>8.3}");
    }

    // gate
    let fixed_max = med[0..5].iter().cloned().fold(0.0f64, f64::max);
    let learned = med[5];
    let oracle = med[6];
    let gate = fixed_max <= 0.60 && learned >= 0.90 && oracle >= 0.90;
    println!(
        "\n== GATE 2: max(fixed/linear) {fixed_max:.3}<=.60 AND learned-MLP {learned:.3}>=.90 AND oracle {oracle:.3}>=.90 => {} ==",
        if gate {
            "PASS — LEARNING IS NECESSARY (fixed closed-form insufficient; nonlinear learned readout required)"
        } else if learned >= fixed_max + 0.15 && learned >= 0.85 {
            "SOFT-PASS — learned >> best fixed (learning necessary; a fixed arm above chance, see below)"
        } else {
            "FAIL — either a fixed arm solves it (not learning-necessary) or the MLP can't learn it (report)"
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
            "{{\n  \"task\":\"gate2_xor_regional_roughness\",\"L\":{L},\"m\":{m},\
             \"n_train\":{N_TRAIN},\"n_test\":{N_TEST},\"seeds\":{SEEDS},\n  \
             \"gate_pass\":{gate},\"fixed_max\":{fixed_max:.4},\"learned\":{learned:.4},\
             \"oracle\":{oracle:.4},\n  \"arms\":[{}]\n}}\n",
            arms.join(",")
        );
        std::fs::write(&path, out).expect("write json");
        println!("wrote {path}");
    }
}
