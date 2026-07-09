//! T2 — a closed-form Chebyshev-KAN **regressor** on **California housing** (the standard
//! tabular regression benchmark; 8 numeric features → median house value). Model:
//! `x → KAN(8→1) + bias → scalar → MSE`, trained by hand-derived closed-form gradients
//! (`kan_backward` + `mse_backward`). Features min-max standardised to `[-1,1]`, target
//! z-scored (loader). Reports held-out R² (multi-seed) + RMSE in original $ units.
//!
//! Runs on a committed 2000-row subset (self-contained); the full 20 433-row set lives in
//! the repo-external data dir (`scripts/dev/fetch_tabular_datasets.sh`) for larger runs.

use holonomy_learn::{
    kan_backward, kan_forward, load_csv_regression, mse_forward, r2_score, KanConfig, TabularReg,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

/// Train on a seed's split; returns (test_r2, test_rmse_original_units).
fn train_eval(data: &TabularReg, seed: u64) -> (f32, f32) {
    let (tr, te) = data.split(0.2, seed);
    let (x_tr, t_tr) = data.gather(&tr);
    let (x_te, t_te) = data.gather(&te);
    let (n_tr, n_te) = (tr.len(), te.len());
    let cfg = KanConfig::new(data.d, 1, 8, 6); // additive spline model: 8 features → 1

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(100));
    let mut coef: Vec<f32> = (0..cfg.d_out * cfg.d_in * cfg.cheb_k)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
        .collect();
    let mut bias = 0.0f32;
    let lr = 0.05;

    let pred = |coef: &[f32], bias: f32, x: &[f32], n: usize| -> Vec<f32> {
        let (kout, _) = kan_forward(coef, x, n, cfg);
        kout.iter().map(|&v| v + bias).collect()
    };

    for _ in 0..600 {
        let (kout, cache) = kan_forward(&coef, &x_tr, n_tr, cfg);
        let p: Vec<f32> = kout.iter().map(|&v| v + bias).collect();
        // ∂MSE/∂pred = 2(pred−target)/n; pred = kan_out + bias.
        let inv_n = 1.0 / n_tr as f32;
        let grad_pred: Vec<f32> = p
            .iter()
            .zip(&t_tr)
            .map(|(&pi, &ti)| 2.0 * (pi - ti) * inv_n)
            .collect();
        let (_gx, gc) = kan_backward(&cache, &grad_pred, cfg);
        for (c, g) in coef.iter_mut().zip(&gc) {
            *c -= lr * g;
        }
        bias -= lr * grad_pred.iter().sum::<f32>();
    }

    let p_te = pred(&coef, bias, &x_te, n_te);
    let r2 = r2_score(&p_te, &t_te);
    // RMSE back in original $ units (target was z-scored by target_std).
    let rmse = mse_forward(&p_te, &t_te).sqrt() * data.target_std;
    (r2, rmse)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn kan_regresses_california() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/california.csv");
    let data =
        load_csv_regression(&std::fs::read_to_string(path).expect("california fixture present"));
    assert_eq!(data.d, 8, "8 numeric features");
    assert!(data.n >= 1000);

    let mut r2s = Vec::new();
    for seed in 0..5u64 {
        let (r2, rmse) = train_eval(&data, seed);
        eprintln!("seed {seed}: test R² {r2:.3}  RMSE ${rmse:.0}");
        r2s.push(r2);
    }
    let med = median(r2s);
    eprintln!("Chebyshev-KAN regressor on California (closed-form): median held-out R² = {med:.3}");
    // An additive spline model should explain a solid fraction of the variance.
    assert!(med >= 0.5, "median California R² too low: {med:.3}");
}
