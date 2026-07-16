//! Nagare vs PyTorch — CPU wall-clock + accuracy on simple datasets (no autograd).
//!
//! Same MLP (d→64→out, ReLU, Adam, 300 full-batch epochs) in Rust, plus the Nagare
//! KAN (Chebyshev-CR spline learner). Exports the exact standardized train/test split
//! to /tmp so `scripts/dev/pytorch_bench.py` trains an identical PyTorch MLP on the
//! identical data. Reports test accuracy (iris) / R² (california) + median train ms.
//!
//! Run: `cargo run --release --example pytorch_bench`

use holonomy_learn::{
    accuracy_k, adam_step, cross_entropy_k_backward, kan_backward, kan_forward, linear_backward,
    linear_forward, load_csv, load_csv_regression, mse_backward, r2_score, AdamState, KanConfig,
    LinearLayer,
};
use std::time::Instant;

fn relu(z: &[f32]) -> Vec<f32> {
    z.iter().map(|&v| v.max(0.0)).collect()
}

/// Median of a few timed training runs (ms).
fn median_ms(mut f: impl FnMut() -> f64) -> f64 {
    let mut v: Vec<f64> = (0..3).map(|_| f()).collect();
    v.sort_by(|a, b| a.total_cmp(b));
    v[1]
}

/// 80/20 split of `(x (n,d), y)` by a fixed stride; returns (xtr, ytr, xte, yte).
fn split<T: Clone>(x: &[f32], y: &[T], n: usize, d: usize) -> (Vec<f32>, Vec<T>, Vec<f32>, Vec<T>) {
    let (mut xtr, mut ytr, mut xte, mut yte) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        let row = &x[i * d..i * d + d];
        if i % 5 == 0 {
            xte.extend_from_slice(row);
            yte.push(y[i].clone());
        } else {
            xtr.extend_from_slice(row);
            ytr.push(y[i].clone());
        }
    }
    (xtr, ytr, xte, yte)
}

/// Write a split to /tmp as plain CSV (feature cols then label) for the PyTorch arm.
fn export(name: &str, x: &[f32], y: &[f32], d: usize) {
    let n = y.len();
    let mut s = String::new();
    for i in 0..n {
        for c in 0..d {
            s.push_str(&format!("{},", x[i * d + c]));
        }
        s.push_str(&format!("{}\n", y[i]));
    }
    std::fs::write(format!("/tmp/nb_{name}.csv"), s).unwrap();
}

fn adam_layer(
    l: &mut LinearLayer,
    g: &LinearLayer,
    sw: &mut AdamState,
    sb: &mut AdamState,
    lr: f32,
) {
    adam_step(&mut l.w, &g.w, sw, lr);
    adam_step(&mut l.b, &g.b, sb, lr);
}

const H: usize = 64;
const EPOCHS: usize = 300;
const LR: f32 = 0.01;

/// Train the Nagare MLP; returns (metric, median_ms). cls: accuracy, reg: R².
#[allow(clippy::too_many_arguments)]
fn nagare_mlp(
    xtr: &[f32],
    ytr_c: &[usize],
    ytr_r: &[f32],
    xte: &[f32],
    yte_c: &[usize],
    yte_r: &[f32],
    d: usize,
    k: usize,
    cls: bool,
) -> (f64, f64) {
    let ntr = if cls { ytr_c.len() } else { ytr_r.len() };
    let mk = || -> (f64, f64) {
        let mut l1 = LinearLayer::new(d, H, 1);
        let mut l2 = LinearLayer::new(H, k, 2);
        let (mut s1w, mut s1b) = (AdamState::new(l1.w.len()), AdamState::new(l1.b.len()));
        let (mut s2w, mut s2b) = (AdamState::new(l2.w.len()), AdamState::new(l2.b.len()));
        let t = Instant::now();
        for _ in 0..EPOCHS {
            let z1 = linear_forward(&l1, xtr);
            let h = relu(&z1);
            let out = linear_forward(&l2, &h);
            let grad_out = if cls {
                cross_entropy_k_backward(&out, ytr_c, ntr, k)
            } else {
                mse_backward(&out, ytr_r)
            };
            let (grad_h, g2) = linear_backward(&l2, &h, &grad_out);
            let grad_z1: Vec<f32> = grad_h
                .iter()
                .zip(&z1)
                .map(|(&g, &z)| if z > 0.0 { g } else { 0.0 })
                .collect();
            let (_gx, g1) = linear_backward(&l1, xtr, &grad_z1);
            adam_layer(&mut l1, &g1, &mut s1w, &mut s1b, LR);
            adam_layer(&mut l2, &g2, &mut s2w, &mut s2b, LR);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let out = linear_forward(&l2, &relu(&linear_forward(&l1, xte)));
        let metric = if cls {
            accuracy_k(&out, yte_c, yte_c.len(), k) as f64
        } else {
            r2_score(&out, yte_r) as f64
        };
        (metric, ms)
    };
    let (metric, _) = mk();
    (metric, median_ms(|| mk().1))
}

/// Train the Nagare KAN (single Chebyshev-CR layer d→k); returns (metric, median_ms).
#[allow(clippy::too_many_arguments)]
fn nagare_kan(
    xtr: &[f32],
    ytr_c: &[usize],
    ytr_r: &[f32],
    xte: &[f32],
    yte_c: &[usize],
    yte_r: &[f32],
    d: usize,
    k: usize,
    cls: bool,
) -> (f64, f64) {
    let ntr = if cls { ytr_c.len() } else { ytr_r.len() };
    let cfg = KanConfig::new(d, k, 6, 6);
    let mk = || -> (f64, f64) {
        let mut coef = vec![0.0f32; k * d * 6];
        let mut st = AdamState::new(coef.len());
        let mut seed = 7u64;
        for c in coef.iter_mut() {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *c = 0.1 * (((seed >> 33) as f32) / (u32::MAX as f32) - 0.5);
        }
        let t = Instant::now();
        for _ in 0..EPOCHS {
            let (out, cache) = kan_forward(&coef, xtr, ntr, cfg);
            let grad_out = if cls {
                cross_entropy_k_backward(&out, ytr_c, ntr, k)
            } else {
                mse_backward(&out, ytr_r)
            };
            let (_gx, gc) = kan_backward(&cache, &grad_out, cfg);
            adam_step(&mut coef, &gc, &mut st, LR);
        }
        let ms = t.elapsed().as_secs_f64() * 1e3;
        let (out, _) = kan_forward(&coef, xte, yte_c.len().max(yte_r.len()), cfg);
        let metric = if cls {
            accuracy_k(&out, yte_c, yte_c.len(), k) as f64
        } else {
            r2_score(&out, yte_r) as f64
        };
        (metric, ms)
    };
    let (metric, _) = mk();
    (metric, median_ms(|| mk().1))
}

fn main() {
    println!(
        "Nagare (Rust, no autograd) — MLP d→64→out + KAN, 300 epochs, Adam. rayon threads: {}",
        rayon::current_num_threads()
    );
    println!(
        "{:<12} {:<10} {:>10} {:>10}",
        "dataset", "arm", "metric", "train_ms"
    );

    // iris (classification, accuracy)
    let t = load_csv(&std::fs::read_to_string("tests/fixtures/iris.csv").unwrap());
    let (xtr, ytr, xte, yte) = split(&t.x, &t.y, t.n, t.d);
    let (dummy_r, _e): (Vec<f32>, _) = (vec![], ());
    export(
        "iris_train",
        &xtr,
        &ytr.iter().map(|&c| c as f32).collect::<Vec<_>>(),
        t.d,
    );
    export(
        "iris_test",
        &xte,
        &yte.iter().map(|&c| c as f32).collect::<Vec<_>>(),
        t.d,
    );
    let (m, ms) = nagare_mlp(
        &xtr,
        &ytr,
        &dummy_r,
        &xte,
        &yte,
        &dummy_r,
        t.d,
        t.n_classes,
        true,
    );
    println!("{:<12} {:<10} {:>9.3}  {:>9.1}", "iris", "MLP", m, ms);
    let (m, ms) = nagare_kan(
        &xtr,
        &ytr,
        &dummy_r,
        &xte,
        &yte,
        &dummy_r,
        t.d,
        t.n_classes,
        true,
    );
    println!("{:<12} {:<10} {:>9.3}  {:>9.1}", "iris", "KAN", m, ms);

    // california (regression, R²)
    let r = load_csv_regression(&std::fs::read_to_string("tests/fixtures/california.csv").unwrap());
    let (xtr, ytr, xte, yte) = split(&r.x, &r.target, r.n, r.d);
    let (dummy_c, _e): (Vec<usize>, _) = (vec![], ());
    export("cali_train", &xtr, &ytr, r.d);
    export("cali_test", &xte, &yte, r.d);
    let (m, ms) = nagare_mlp(&xtr, &dummy_c, &ytr, &xte, &dummy_c, &yte, r.d, 1, false);
    println!("{:<12} {:<10} {:>9.3}  {:>9.1}", "california", "MLP", m, ms);
    let (m, ms) = nagare_kan(&xtr, &dummy_c, &ytr, &xte, &dummy_c, &yte, r.d, 1, false);
    println!("{:<12} {:<10} {:>9.3}  {:>9.1}", "california", "KAN", m, ms);
    println!("(exported /tmp/nb_*.csv for the PyTorch arm)");
}
