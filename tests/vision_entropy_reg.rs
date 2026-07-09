//! Nagare CV — the genuine **entropy feedback**: `spectral_entropy` regularisation on the
//! quaternion conv's **learned feature map** (not a static feature).
//!
//! The headroom-texture test showed entropy is redundant with `|DFT|` *as a feature*. The
//! framework's entropy machinery is a **regulariser on a learned representation** (entropy-gated
//! HSiKAN; the `spectral_entropy` op with its Lyapunov schedule). So this targets the quat-conv's
//! learned pooled feature matrix `P (batch × M)`: each epoch, `SpectralEntropyReg::step(P)` returns
//! the reg gradient (pushing `P`'s spectral entropy toward the target τ), added to the
//! classification gradient and backpropagated through the conv. A rotation-invariance prior lives
//! in the canonicalised gradient input; the entropy prior shapes the *learned* features.
//!
//! Ablation (reported, §3), 4 seeds, rotated shapes: quat-conv with vs without spectral-entropy
//! feedback on `P`. Does regularising the learned feature spectrum help where the conv has headroom?

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, softmax_k, SpectralEntropyConfig, SpectralEntropyReg,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

mod common;
use common::vision::{make_set, patch_gradient_field, CELLS, K, NP};

const DESC: usize = CELLS * 2;
const M: usize = 12;

/// Single-θ-canonical per-patch gradient descriptors `(n, NP, DESC)` (rotation-invariant input).
fn canonical_desc(x: &[f32], n: usize) -> Vec<f32> {
    let (field, theta) = patch_gradient_field(x, n);
    let mut d = vec![0.0f32; n * NP * DESC];
    for s in 0..n {
        for p in 0..NP {
            let t = -theta[s * NP + p];
            let (c, sn) = (t.cos(), t.sin());
            for cell in 0..CELLS {
                let base = ((s * NP + p) * CELLS + cell) * 3;
                let (gx, gy) = (field[base], field[base + 1]);
                d[(s * NP + p) * DESC + cell * 2] = gx * c - gy * sn;
                d[(s * NP + p) * DESC + cell * 2 + 1] = gx * sn + gy * c;
            }
        }
    }
    d
}

/// Train the quat-conv (filter bank + tanh → mean-pool `P` → readout); `entropy` toggles
/// spectral-entropy feedback on `P`. Returns test accuracy.
fn train_eval(
    d_tr: &[f32],
    y_tr: &[usize],
    d_te: &[f32],
    y_te: &[usize],
    seed: u64,
    entropy: bool,
) -> (f32, f32, f32) {
    let (n_tr, n_te) = (y_tr.len(), y_te.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let rv = |n: usize, sc: f32, r: &mut StdRng| -> Vec<f32> {
        (0..n)
            .map(|_| (r.random::<f32>() * 2.0 - 1.0) * sc)
            .collect()
    };
    let mut cw = rv(M * DESC, 0.2, &mut rng);
    let mut cb = vec![0.0f32; M];
    let mut rw = rv(K * M, 0.2, &mut rng);
    let mut rb = vec![0.0f32; K];
    let lr = 0.15;
    // A strong spread prior on the learned feature spectrum (τ=0.6, Lyapunov-scheduled).
    let mut reg = SpectralEntropyReg::new(SpectralEntropyConfig {
        lam_0: 1.0,
        target: 0.6,
        ..Default::default()
    });
    let (mut last_h, mut last_lam) = (f32::NAN, f32::NAN);

    // Forward → (per-patch tanh act, pooled P (n,M)).
    let fwd = |cw: &[f32], cb: &[f32], d: &[f32], n: usize| -> (Vec<f32>, Vec<f32>) {
        let mut act = vec![0.0f32; n * NP * M];
        let mut p = vec![0.0f32; n * M];
        for s in 0..n {
            for pt in 0..NP {
                for m in 0..M {
                    let mut z = cb[m];
                    for j in 0..DESC {
                        z += d[(s * NP + pt) * DESC + j] * cw[m * DESC + j];
                    }
                    let a = z.tanh();
                    act[(s * NP + pt) * M + m] = a;
                    p[s * M + m] += a / NP as f32;
                }
            }
        }
        (act, p)
    };
    let logits = |rw: &[f32], rb: &[f32], p: &[f32], n: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; n * K];
        for s in 0..n {
            for k in 0..K {
                let mut z = rb[k];
                for m in 0..M {
                    z += p[s * M + m] * rw[k * M + m];
                }
                l[s * K + k] = z;
            }
        }
        l
    };

    for _ in 0..280 {
        let (act, pooled) = fwd(&cw, &cb, d_tr, n_tr);
        let lg = logits(&rw, &rb, &pooled, n_tr);
        // Classification grad on P via the readout.
        let gl = cross_entropy_k_backward(&lg, y_tr, n_tr, K);
        let mut grad_p = vec![0.0f32; n_tr * M];
        let mut grad_rw = [0.0f32; K * M];
        let mut grad_rb = [0.0f32; K];
        for s in 0..n_tr {
            for k in 0..K {
                let dz = gl[s * K + k];
                grad_rb[k] += dz;
                for m in 0..M {
                    grad_rw[k * M + m] += dz * pooled[s * M + m];
                    grad_p[s * M + m] += dz * rw[k * M + m];
                }
            }
        }
        // Entropy feedback: add the spectral-entropy reg gradient on P (n × M).
        if entropy {
            let (_r, grad_reg) = reg.step(&pooled, n_tr, M);
            for (g, gr) in grad_p.iter_mut().zip(&grad_reg) {
                *g += gr;
            }
        }
        last_h = reg.last_h_norm;
        last_lam = reg.last_lam_eff;
        // Backprop grad_p through mean-pool + tanh → conv grads.
        let (mut gcw, mut gcb) = (vec![0.0f32; M * DESC], vec![0.0f32; M]);
        for s in 0..n_tr {
            for pt in 0..NP {
                for m in 0..M {
                    let a = act[(s * NP + pt) * M + m];
                    let gz = grad_p[s * M + m] / NP as f32 * (1.0 - a * a);
                    gcb[m] += gz;
                    for j in 0..DESC {
                        gcw[m * DESC + j] += gz * d_tr[(s * NP + pt) * DESC + j];
                    }
                }
            }
        }
        for i in 0..cw.len() {
            cw[i] -= lr * gcw[i];
        }
        for i in 0..M {
            cb[i] -= lr * gcb[i];
        }
        for i in 0..rw.len() {
            rw[i] -= lr * grad_rw[i];
        }
        for i in 0..K {
            rb[i] -= lr * grad_rb[i];
        }
    }
    let (_a, pooled) = fwd(&cw, &cb, d_te, n_te);
    let _ = softmax_k(&logits(&rw, &rb, &pooled, n_te), n_te, K); // (probs unused; acc argmaxes)
    (
        accuracy_k(&logits(&rw, &rb, &pooled, n_te), y_te, n_te, K),
        last_h,
        last_lam,
    )
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
#[ignore = "heavy (~50s: per-epoch eigendecomposition × 2 arms × 4 seeds); run with --ignored"]
fn spectral_entropy_feedback_on_quat_conv() {
    let (mut off, mut on) = (Vec::new(), Vec::new());
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (x_tr, y_tr) = make_set(360, &mut rng);
        let (x_te, y_te) = make_set(160, &mut rng);
        let (dtr, dte) = (canonical_desc(&x_tr, 360), canonical_desc(&x_te, 160));
        let (no, _, _) = train_eval(&dtr, &y_tr, &dte, &y_te, seed + 1, false);
        let (ye, h, lam) = train_eval(&dtr, &y_tr, &dte, &y_te, seed + 1, true);
        eprintln!(
            "seed {seed}: no-reg {no:.3}  entropy-reg {ye:.3}  (H_norm {h:.3}, λ_eff {lam:.4})"
        );
        off.push(no);
        on.push(ye);
    }
    let (om, ym) = (median(off), median(on));
    eprintln!("Nagare CV — spectral-entropy feedback on the quat-conv learned feature map:");
    eprintln!("  no-reg {om:.3}   entropy-reg {ym:.3}");
    eprintln!(
        "  verdict: spectral-entropy feedback {} (Δ {:+.3})",
        if ym > om + 0.01 {
            "HELPS"
        } else if ym < om - 0.01 {
            "hurts"
        } else {
            "neutral"
        },
        ym - om
    );
    // Gate: both learn above chance; the reg-vs-no-reg delta is the measurement.
    assert!(
        om > 0.4 && ym > 0.4,
        "quat-conv failed to learn: {om:.3}/{ym:.3}"
    );
}
