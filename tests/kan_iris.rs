//! T1 — a closed-form Chebyshev-KAN classifier on **Iris** (the standard 150×4, 3-class
//! tabular benchmark). Demonstrates Nagare's closed-form ops generalise beyond signed
//! graphs to standard tabular data. Model: `x → KAN(4→3) + per-class bias → softmax₃ → CE`,
//! trained purely by hand-derived closed-form gradients (`kan_backward` +
//! `cross_entropy_k_backward`). Features are min-max standardised into `[-1,1]` (the spline
//! range) by the loader. Multi-seed held-out accuracy (§3).

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, cross_entropy_k_forward, kan_backward, kan_forward,
    load_csv, KanConfig, Tabular,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn logits(kout: &[f32], bias: &[f32], n: usize, k: usize) -> Vec<f32> {
    let mut l = kout.to_vec();
    for row in 0..n {
        for j in 0..k {
            l[row * k + j] += bias[j];
        }
    }
    l
}

/// Train on a seed's split; return (train_acc, test_acc, final_train_loss).
fn train_eval(iris: &Tabular, seed: u64) -> (f32, f32, f32) {
    let (tr, te) = iris.split(0.25, seed);
    let (x_tr, y_tr) = iris.gather(&tr);
    let (x_te, y_te) = iris.gather(&te);
    let (n_tr, n_te) = (tr.len(), te.len());
    let cfg = KanConfig::new(iris.d, iris.n_classes, 8, 6);
    let k = cfg.d_out;

    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(100));
    let mut coef: Vec<f32> = (0..cfg.d_out * cfg.d_in * cfg.cheb_k)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
        .collect();
    let mut bias = vec![0.0f32; k];
    let lr = 0.1;

    for _ in 0..300 {
        let (kout, cache) = kan_forward(&coef, &x_tr, n_tr, cfg);
        let l = logits(&kout, &bias, n_tr, k);
        let gl = cross_entropy_k_backward(&l, &y_tr, n_tr, k);
        let (_gx, gc) = kan_backward(&cache, &gl, cfg);
        for (c, g) in coef.iter_mut().zip(&gc) {
            *c -= lr * g;
        }
        for j in 0..k {
            let gb: f32 = (0..n_tr).map(|row| gl[row * k + j]).sum();
            bias[j] -= lr * gb;
        }
    }

    let eval = |x: &[f32], y: &[usize], n: usize| -> f32 {
        let (kout, _) = kan_forward(&coef, x, n, cfg);
        accuracy_k(&logits(&kout, &bias, n, k), y, n, k)
    };
    let (ktr, _) = kan_forward(&coef, &x_tr, n_tr, cfg);
    let final_loss = cross_entropy_k_forward(&logits(&ktr, &bias, n_tr, k), &y_tr, n_tr, k);
    (
        eval(&x_tr, &y_tr, n_tr),
        eval(&x_te, &y_te, n_te),
        final_loss,
    )
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn kan_classifies_iris() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/iris.csv");
    let iris = load_csv(&std::fs::read_to_string(path).expect("iris fixture present"));
    assert_eq!((iris.n, iris.d, iris.n_classes), (150, 4, 3));

    let mut test_accs = Vec::new();
    for seed in 0..5u64 {
        let (tr_acc, te_acc, loss) = train_eval(&iris, seed);
        eprintln!("seed {seed}: train_acc {tr_acc:.3}  test_acc {te_acc:.3}  loss {loss:.4}");
        test_accs.push(te_acc);
    }
    let med = median(test_accs);
    eprintln!(
        "Chebyshev-KAN on Iris (closed-form): median held-out accuracy over 5 seeds = {med:.3}",
    );
    // Iris is easy; a working closed-form KAN should clear this floor comfortably.
    assert!(med >= 0.9, "median Iris test accuracy too low: {med:.3}");
}
