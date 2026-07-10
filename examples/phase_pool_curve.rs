//! Demo — train a learned front-end **through** the differentiable invariant phase-pool and dump
//! the loss curve, showing global-pooling backpropagation reduces the loss it defines.
//!
//! `linear(W): x → field → phase_pool → linear(head) → cross-entropy`, gradient descent with the
//! grad flowing `x ← W ← phase_pool_backward ← head`. Writes `step,loss` CSV to `--out`
//! (default `reports/figures/phase_pool_loss.csv`), which `scripts/dev/plot_phase_pool_loss.py`
//! renders. Run: `cargo run --release --example phase_pool_curve`.

use std::io::Write;

use holonomy_learn::{
    cross_entropy_k_backward, cross_entropy_k_forward, linear_backward, linear_forward,
    phase_pool_backward, phase_pool_dim, phase_pool_forward, LinearLayer,
};

fn arg(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}

fn main() {
    let out = arg("--out").unwrap_or_else(|| "reports/figures/phase_pool_loss.csv".into());
    let (g, b, k, in_dim) = (6usize, 12usize, 2usize, 8usize);
    let (field_dim, nk, per_class) = (g * g * 2, phase_pool_dim(b), 40usize);
    let n = per_class * k;

    let mut x = vec![0.0f32; n * in_dim];
    let mut y = vec![0usize; n];
    for c in 0..k {
        for r in 0..per_class {
            let s = c * per_class + r;
            y[s] = c;
            x[s * in_dim + c] = 1.0;
            for d in 0..in_dim {
                x[s * in_dim + d] += 0.05 * (((s * in_dim + d) as f32) * 0.7).sin();
            }
        }
    }

    let mut front = LinearLayer::new(in_dim, field_dim, 11);
    let mut head = LinearLayer::new(nk, k, 23);
    let (lr, clip, steps) = (0.05f32, 5.0f32, 800usize);
    let mut curve = String::from("step,loss\n");

    for step in 0..steps {
        let field = linear_forward(&front, &x);
        let o = phase_pool_forward(&field, n, g, b);
        let logits = linear_forward(&head, &o.feat);
        if step % 10 == 0 {
            curve.push_str(&format!(
                "{step},{}\n",
                cross_entropy_k_forward(&logits, &y, n, k)
            ));
        }
        let gl = cross_entropy_k_backward(&logits, &y, n, k);
        let (grad_feat, head_g) = linear_backward(&head, &o.feat, &gl);
        let mut grad_field = phase_pool_backward(&field, &o, &grad_feat, n, g, b);
        let norm = grad_field.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > clip {
            let sc = clip / norm;
            grad_field.iter_mut().for_each(|v| *v *= sc);
        }
        let (_gx, front_g) = linear_backward(&front, &x, &grad_field);
        for (w, gw) in head.w.iter_mut().zip(&head_g.w) {
            *w -= lr * gw;
        }
        for (bb, gb) in head.b.iter_mut().zip(&head_g.b) {
            *bb -= lr * gb;
        }
        for (w, gw) in front.w.iter_mut().zip(&front_g.w) {
            *w -= lr * gw;
        }
        for (bb, gb) in front.b.iter_mut().zip(&front_g.b) {
            *bb -= lr * gb;
        }
    }

    let l_final = {
        let field = linear_forward(&front, &x);
        let feat = phase_pool_forward(&field, n, g, b).feat;
        cross_entropy_k_forward(&linear_forward(&head, &feat), &y, n, k)
    };
    std::fs::File::create(&out)
        .and_then(|mut f| f.write_all(curve.as_bytes()))
        .expect("write csv");
    println!("wrote {out}; final loss {l_final:.4}");
}
