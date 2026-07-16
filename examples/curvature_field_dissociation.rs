//! **Nonlinear-curvature dissociation (B+C)** — a spline-parameterized curvature field where
//! trivial entropy AND the F-HOLO-3 constant-rotor solve are BOTH blind, but a closed-form
//! ChebyCR roughness readout recovers the field. No autograd; the holonomy arms are untrained.
//!
//! Classes (matched flux multiset + total): smooth (low-order Chebyshev angle field) vs rough
//! (spatial permutation). Arms (identical data per seed, held-out separability AUROC):
//!   trivial-entropy   — covariance eigen-entropy of edge log-rotors → chance (matched marginals).
//!   constant-rotor    — mean plaquette holonomy angle (the F-HOLO-3 estimator) → chance (BLIND;
//!                       permutation-invariant + matched total). THE sharper gate.
//!   MLP               — 2-layer MLP over raw edge rotors, Adam-trained (learned baseline).
//!   Laplacian         — discrete-Laplacian roughness of the extracted field (no-solve check).
//!   ChebyCR           — low-order 2-D Chebyshev fit residual of the extracted field (the method).
//!   oracle            — ChebyCR roughness of the TRUE field (ceiling).
//!
//! Gate (5 seeds): trivial ≤ 0.60 AND constant-rotor ≤ 0.60 AND oracle ≥ 0.90 ⇒ the exact scalar
//! solve is insufficient and a spatial readout is necessary. Then ChebyCR should ≫ both blind
//! arms and ≈ oracle, one-shot.
//!
//! Run: `cargo run --release --example curvature_field_dissociation [-- --json <path>]`

use holonomy_learn::{
    adam_step, auroc, chebycr_roughness, constant_rotor_energy, edge_log_field,
    extract_curvature_field, grid_graph, laplacian_roughness, sample_curvature_field,
    spectral_reg_value_grad, AdamState, CurvatureRng, GridGraph, SpectralEntropyConfig,
};
use std::time::Instant;

const L: usize = 12; // 11x11 = 121 plaquettes, 264 edges
const K_FIT: usize = 3; // fixed low-order Chebyshev fit
const K_GEN: usize = 3; // generating field order (main table)
const NOISE: f32 = 0.05;
const N_TRAIN: usize = 200;
const N_TEST: usize = 200;
const SEEDS: u64 = 5;

struct Data {
    x: Vec<Vec<f32>>, // edge rotors per sample
    y: Vec<u8>,
    theta: Vec<Vec<f32>>, // true field per sample (for the oracle)
}

fn gen_set(g: &GridGraph, rng: &mut CurvatureRng, n: usize, k_gen: usize) -> Data {
    let mut x = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut theta = Vec::with_capacity(n);
    for k in 0..n {
        let lab = (k % 2) as u8; // 0 smooth, 1 rough
        let (eq, th) = sample_curvature_field(g, rng, lab == 1, k_gen, NOISE);
        x.push(eq);
        y.push(lab);
        theta.push(th);
    }
    Data { x, y, theta }
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
    let field = edge_log_field(eq);
    let n_edges = field.len() / 3;
    spectral_reg_value_grad(&field, n_edges, 3, cfg, 1.0).0
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

/// 2-layer MLP over flattened edge rotors, Adam-trained. Held-out separability AUROC.
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

struct Arms {
    triv: f64,
    const_rotor: f64,
    mlp: f64,
    laplacian: f64,
    chebycr: f64,
    oracle: f64,
}

fn run_seed(
    g: &GridGraph,
    seed: u64,
    k_gen: usize,
    cfg: &SpectralEntropyConfig,
    with_mlp: bool,
) -> Arms {
    let mut rng = CurvatureRng(101 + seed);
    let train = gen_set(g, &mut rng, N_TRAIN, k_gen);
    let test = gen_set(g, &mut rng, N_TEST, k_gen);
    let m = g.m;
    let fields: Vec<Vec<f32>> = test
        .x
        .iter()
        .map(|x| extract_curvature_field(g, x))
        .collect();
    let triv: Vec<f32> = test.x.iter().map(|x| trivial_entropy(x, cfg)).collect();
    let cr: Vec<f32> = fields.iter().map(|f| constant_rotor_energy(f)).collect();
    let lap: Vec<f32> = fields.iter().map(|f| laplacian_roughness(f, m)).collect();
    let cby: Vec<f32> = fields
        .iter()
        .map(|f| chebycr_roughness(f, m, K_FIT))
        .collect();
    let orc: Vec<f32> = test
        .theta
        .iter()
        .map(|th| chebycr_roughness(th, m, K_FIT))
        .collect();
    Arms {
        triv: sep(&triv, &test.y),
        const_rotor: sep(&cr, &test.y),
        mlp: if with_mlp {
            run_mlp(&train, &test, 16, 0.05, 150, seed)
        } else {
            f64::NAN
        },
        laplacian: sep(&lap, &test.y),
        chebycr: sep(&cby, &test.y),
        oracle: sep(&orc, &test.y),
    }
}

fn main() {
    let cfg = entropy_cfg();
    let g = grid_graph(L);
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let t0 = Instant::now();

    println!(
        "== nonlinear-curvature dissociation: grid L={L} ({}x{} plaquettes, {} edges), \
         k_gen={K_GEN} k_fit={K_FIT} noise={NOISE}, {N_TRAIN}/{N_TEST}, {SEEDS} seeds ==",
        g.m, g.m, g.n_edges
    );

    let names = [
        "trivial-entropy",
        "constant-rotor",
        "MLP (trained)",
        "Laplacian",
        "ChebyCR",
        "oracle",
    ];
    let mut cols: Vec<Vec<f64>> = vec![vec![]; 6];
    for s in 0..SEEDS {
        let a = run_seed(&g, s, K_GEN, &cfg, true);
        for (i, v) in [
            a.triv,
            a.const_rotor,
            a.mlp,
            a.laplacian,
            a.chebycr,
            a.oracle,
        ]
        .into_iter()
        .enumerate()
        {
            cols[i].push(v);
        }
        println!(
            "  seed {s}: triv {:.3}  const {:.3}  MLP {:.3}  lap {:.3}  ChebyCR {:.3}  oracle {:.3}  [{:.1}s]",
            a.triv, a.const_rotor, a.mlp, a.laplacian, a.chebycr, a.oracle, t0.elapsed().as_secs_f32()
        );
    }
    let med: Vec<f64> = cols.iter().map(|c| median(c)).collect();
    println!("\n  {:<18} {:>8}", "arm", "median");
    for (n, m) in names.iter().zip(&med) {
        println!("  {n:<18} {m:>8.3}");
    }

    let (triv, const_rotor, chebycr, oracle) = (med[0], med[1], med[4], med[5]);
    let gate = triv <= 0.60 && const_rotor <= 0.60 && oracle >= 0.90;
    println!(
        "\n== GATE: trivial {triv:.3}<=.60 AND constant-rotor {const_rotor:.3}<=.60 AND oracle {oracle:.3}>=.90 => {} ==",
        if gate { "PASS (exact scalar solve is INSUFFICIENT; spatial readout necessary)" } else { "FAIL (a baseline is not blind — STOP, do not force the claim)" }
    );
    if gate {
        let verdict = if chebycr >= oracle - 0.03 {
            "ChebyCR REACHES the oracle, one-shot, where the constant-rotor solve is blind"
        } else if chebycr > const_rotor + 0.15 {
            "ChebyCR clears the blind baselines (below oracle — investigate)"
        } else {
            "ChebyCR did not clear the blind baselines — investigate"
        };
        println!("== A/B: ChebyCR {chebycr:.3} vs oracle {oracle:.3} vs constant-rotor {const_rotor:.3} => {verdict} ==");
    }

    // ---- contrast sweep: AUROC vs generating-field order k_gen (smoothness contrast falls as k_gen rises) ----
    println!("\n== contrast sweep: separability vs k_gen (fixed k_fit={K_FIT}, no MLP) ==");
    println!(
        "  {:>6} {:>10} {:>10} {:>10} {:>8}",
        "k_gen", "triv", "const", "ChebyCR", "oracle"
    );
    let kgs = [2usize, 3, 4, 6, 8];
    let mut sweep: Vec<(usize, f64, f64, f64, f64)> = vec![];
    for &kg in &kgs {
        let (mut tv, mut cv, mut chv, mut ov) = (vec![], vec![], vec![], vec![]);
        for s in 0..SEEDS {
            let a = run_seed(&g, s, kg, &cfg, false);
            tv.push(a.triv);
            cv.push(a.const_rotor);
            chv.push(a.chebycr);
            ov.push(a.oracle);
        }
        let (mt, mc, mch, mo) = (median(&tv), median(&cv), median(&chv), median(&ov));
        sweep.push((kg, mt, mc, mch, mo));
        println!("  {kg:>6} {mt:>10.3} {mc:>10.3} {mch:>10.3} {mo:>8.3}");
    }

    println!("\n== done in {:.1}s ==", t0.elapsed().as_secs_f32());

    if let Some(path) = json_path {
        let arms_json: Vec<String> = names
            .iter()
            .zip(&med)
            .map(|(n, m)| format!("{{\"arm\":\"{n}\",\"median_auroc\":{m:.4}}}"))
            .collect();
        let sweep_json: Vec<String> = sweep
            .iter()
            .map(|(kg, t, c, ch, o)| {
                format!("{{\"k_gen\":{kg},\"trivial\":{t:.4},\"constant_rotor\":{c:.4},\"chebycr\":{ch:.4},\"oracle\":{o:.4}}}")
            })
            .collect();
        let out = format!(
            "{{\n  \"task\":\"nonlinear_curvature_field_grid\",\n  \"L\":{L},\"m\":{},\"n_edges\":{},\
             \"k_gen\":{K_GEN},\"k_fit\":{K_FIT},\"noise\":{NOISE},\"seeds\":{SEEDS},\n  \
             \"gate_pass\":{gate},\n  \"main\":[{}],\n  \"contrast_sweep\":[{}]\n}}\n",
            g.m,
            g.n_edges,
            arms_json.join(","),
            sweep_json.join(",")
        );
        std::fs::write(&path, out).expect("write json");
        println!("wrote {path}");
    }
}
