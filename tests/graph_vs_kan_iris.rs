//! T4 — does the signed-graph structure beat the plain KAN on Iris?
//!
//! Two closed-form models on the SAME splits:
//!   - **KAN** (T1): `x → KAN(4→3) + bias → softmax₃`.
//!   - **Graph** (T3): build a leakage-free kNN signed graph (features only), enumerate
//!     signed triangles, then `x → gomb_outer (M FIR banks) → scatter_mean (cycle→sample)
//!     → linear → softmax₃`, transductive (train on train labels, eval on test).
//!
//! The verdict is **reported**, not assumed — the plan flagged the graph may NOT win.

use holonomy_learn::{
    accuracy_k, build_signed_cycle_pool, cross_entropy_k_backward, gomb_outer_backward,
    gomb_outer_forward, kan_backward, kan_forward, linear_backward, linear_forward, load_csv,
    scatter_mean_backward, scatter_mean_forward, softmax_k, KanConfig, LinearLayer, Tabular,
};
use hymeko_graph::CliffordFIR;
use rand::{rngs::StdRng, Rng, SeedableRng};

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn test_acc(logits: &[f32], y: &[usize], idx: &[usize], k: usize) -> f32 {
    let sub_logits: Vec<f32> = idx
        .iter()
        .flat_map(|&i| logits[i * k..i * k + k].to_vec())
        .collect();
    let sub_y: Vec<usize> = idx.iter().map(|&i| y[i]).collect();
    accuracy_k(&sub_logits, &sub_y, idx.len(), k)
}

/// KAN baseline (T1) — median-style single run; returns test accuracy.
fn run_kan(iris: &Tabular, tr: &[usize], te: &[usize], seed: u64) -> f32 {
    let (x_tr, y_tr) = iris.gather(tr);
    let (n_tr, k) = (tr.len(), iris.n_classes);
    let cfg = KanConfig::new(iris.d, k, 8, 6);
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(100));
    let mut coef: Vec<f32> = (0..cfg.d_out * cfg.d_in * cfg.cheb_k)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
        .collect();
    let mut bias = vec![0.0f32; k];
    for _ in 0..300 {
        let (kout, cache) = kan_forward(&coef, &x_tr, n_tr, cfg);
        let mut l = kout.clone();
        for r in 0..n_tr {
            for j in 0..k {
                l[r * k + j] += bias[j];
            }
        }
        let gl = cross_entropy_k_backward(&l, &y_tr, n_tr, k);
        let (_g, gc) = kan_backward(&cache, &gl, cfg);
        for (c, g) in coef.iter_mut().zip(&gc) {
            *c -= 0.1 * g;
        }
        for j in 0..k {
            bias[j] -= 0.1 * (0..n_tr).map(|r| gl[r * k + j]).sum::<f32>();
        }
    }
    // Eval on the full set, score test rows.
    let (kout, _) = kan_forward(&coef, &iris.x, iris.n, cfg);
    let mut logits = kout;
    for r in 0..iris.n {
        for j in 0..k {
            logits[r * k + j] += bias[j];
        }
    }
    test_acc(&logits, &iris.y, te, k)
}

/// Graph model (T3) — transductive node classifier over the signed triangles.
fn run_graph(iris: &Tabular, tr: &[usize], te: &[usize], seed: u64) -> (f32, usize) {
    let (n, d, k) = (iris.n, iris.d, iris.n_classes);
    let m = 4usize; // FIR banks
    let pool = build_signed_cycle_pool(&iris.x, n, d, 10, 3);
    let width = m * d;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(200));
    let mut banks: Vec<CliffordFIR> = (0..m)
        .map(|_| {
            let a = (0..3)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.4)
                .collect();
            let b = (0..3)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.4)
                .collect();
            CliffordFIR::new(a, b)
        })
        .collect();
    let mut readout = LinearLayer::new(width, k, seed.wrapping_add(201));
    let inv_ntr = 1.0 / tr.len() as f32;

    for _ in 0..300 {
        let y = gomb_outer_forward(&pool.batch, &iris.x, &banks, n, d);
        let (xg, counts) = scatter_mean_forward(&pool.batch.cycles, 3, &y, width, n);
        let logits = linear_forward(&readout, &xg);
        let probs = softmax_k(&logits, n, k);
        // Masked CE gradient: train rows only, mean over train.
        let mut grad_logits = vec![0.0f32; n * k];
        for &r in tr {
            for j in 0..k {
                grad_logits[r * k + j] = (probs[r * k + j] - f32::from(j == iris.y[r])) * inv_ntr;
            }
        }
        let (grad_xg, grad_readout) = linear_backward(&readout, &xg, &grad_logits);
        let grad_y = scatter_mean_backward(&pool.batch.cycles, 3, &grad_xg, width, &counts, n);
        let (_gf, grad_banks) = gomb_outer_backward(&pool.batch, &iris.x, &banks, &grad_y, n, d);
        for (bank, gb) in banks.iter_mut().zip(&grad_banks) {
            for (a, ga) in bank.a.iter_mut().zip(&gb.a) {
                *a -= 0.1 * ga;
            }
            for (b, gbb) in bank.b.iter_mut().zip(&gb.b) {
                *b -= 0.1 * gbb;
            }
        }
        for (w, gw) in readout.w.iter_mut().zip(&grad_readout.w) {
            *w -= 0.1 * gw;
        }
        for (b, gb) in readout.b.iter_mut().zip(&grad_readout.b) {
            *b -= 0.1 * gb;
        }
    }
    let y = gomb_outer_forward(&pool.batch, &iris.x, &banks, n, d);
    let (xg, _c) = scatter_mean_forward(&pool.batch.cycles, 3, &y, width, n);
    let logits = linear_forward(&readout, &xg);
    (test_acc(&logits, &iris.y, te, k), pool.n_cycles)
}

#[test]
fn graph_vs_kan_on_iris() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/iris.csv");
    let iris = load_csv(&std::fs::read_to_string(path).expect("iris fixture"));

    let (mut kan_accs, mut graph_accs) = (Vec::new(), Vec::new());
    let mut n_cycles = 0;
    for seed in 0..5u64 {
        let (tr, te) = iris.split(0.25, seed);
        let ka = run_kan(&iris, &tr, &te, seed);
        let (ga, nc) = run_graph(&iris, &tr, &te, seed);
        n_cycles = nc;
        eprintln!("seed {seed}: KAN {ka:.3}  graph {ga:.3}");
        kan_accs.push(ka);
        graph_accs.push(ga);
    }
    let (km, gm) = (median(kan_accs), median(graph_accs));
    eprintln!(
        "Iris median held-out accuracy: KAN {km:.3}  vs  graph {gm:.3}  (triangles={n_cycles})"
    );
    eprintln!(
        "  verdict: signed-graph structure {} the plain KAN (Δ {:.3})",
        if gm > km { "beats" } else { "does NOT beat" },
        gm - km
    );
    // Both must learn a working classifier; which wins is the measurement, not the gate.
    assert!(
        km >= 0.85 && gm >= 0.7,
        "a model failed to classify: KAN {km:.3}, graph {gm:.3}"
    );
}
