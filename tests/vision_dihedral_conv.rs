//! Nagare CV — wire the **dihedral group-convolution** into the vision model and measure it
//! against the single-θ canonicalisation (the quat-conv winner) on rotated shapes.
//!
//! One group-max conv backbone serves all three arms (`|G|` steered frames → shared filter +
//! tanh → group-max over `|G|` → mean-pool over patches → readout):
//!   - **raw** — `|G|=1`, the gradient field untouched (rotation-blind floor);
//!   - **single-θ canonical** — `|G|=1`, each patch's cells rotated by −θ_p (continuous, exact,
//!     one data-dependent frame; the quat-conv strategy);
//!   - **D_n group-conv** — `|G|=n`, the field steered to all `n` dihedral frames (`dihedral_steer`),
//!     group-max over them (discrete, data-independent, learned combination).
//!
//! Question: does the discrete `D_n` group-conv match/beat the continuous single-frame
//! canonicalisation? Ablation (reported, §3): raw vs canonical vs `C_8` group-conv, 4 seeds.

use holonomy_learn::{accuracy_k, dihedral_steer_forward, softmax_k, DihedralGroup};
use rand::{rngs::StdRng, Rng, SeedableRng};

mod common;
use common::vision::{make_set, patch_gradient_field, CELLS, K, NP};

const DESC: usize = CELLS * 2; // per-patch descriptor (9 (gx,gy) pairs)
const M: usize = 12; // conv filter bank

/// Raw / single-θ-canonical descriptors as a single group frame → `(n, 1, NP, DESC)`.
/// `canonical` rotates each patch's cell gradients by −θ_p (2-D rotor action, exact).
fn single_frame(x: &[f32], n: usize, canonical: bool) -> Vec<f32> {
    let (field, theta) = patch_gradient_field(x, n);
    let mut d = vec![0.0f32; n * NP * DESC];
    for s in 0..n {
        for p in 0..NP {
            let (c, sgn) = if canonical {
                let t = -theta[s * NP + p];
                (t.cos(), t.sin())
            } else {
                (1.0, 0.0)
            };
            for cell in 0..CELLS {
                let base = ((s * NP + p) * CELLS + cell) * 3;
                let (gx, gy) = (field[base], field[base + 1]);
                d[(s * NP + p) * DESC + cell * 2] = gx * c - gy * sgn;
                d[(s * NP + p) * DESC + cell * 2 + 1] = gx * sgn + gy * c;
            }
        }
    }
    d
}

/// D_n group-conv descriptors: steer the gradient field to all `|G|` frames → `(n, |G|, NP, DESC)`.
fn group_frames(x: &[f32], n: usize, group: DihedralGroup) -> (Vec<f32>, usize) {
    let (field, _theta) = patch_gradient_field(x, n); // (n·NP·CELLS, 3)
    let nv = n * NP * CELLS;
    let steered = dihedral_steer_forward(&field, group, nv); // (|G|, nv, 3)
    let go = group.order();
    let mut d = vec![0.0f32; n * go * NP * DESC];
    for g in 0..go {
        for s in 0..n {
            for p in 0..NP {
                for cell in 0..CELLS {
                    let sb = (g * nv + (s * NP + p) * CELLS + cell) * 3;
                    let db = ((s * go + g) * NP + p) * DESC + cell * 2;
                    d[db] = steered[sb];
                    d[db + 1] = steered[sb + 1];
                }
            }
        }
    }
    (d, go)
}

/// Train the group-max conv (`|G|` frames → shared filter+tanh → group-max → mean-pool → readout)
/// on fixed descriptors `(n, |G|, NP, DESC)`; return test accuracy.
fn train_eval(
    d_tr: &[f32],
    go: usize,
    y_tr: &[usize],
    d_te: &[f32],
    y_te: &[usize],
    seed: u64,
) -> f32 {
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

    // Forward → (argmax-frame per (s,p,m), group-max activation, mean-pooled (n,M)).
    let fwd = |cw: &[f32], cb: &[f32], d: &[f32], n: usize| -> (Vec<usize>, Vec<f32>, Vec<f32>) {
        let mut arg = vec![0usize; n * NP * M];
        let mut amax = vec![0.0f32; n * NP * M];
        let mut pooled = vec![0.0f32; n * M];
        for s in 0..n {
            for p in 0..NP {
                for m in 0..M {
                    let (mut best, mut bg) = (f32::NEG_INFINITY, 0usize);
                    for g in 0..go {
                        let db = ((s * go + g) * NP + p) * DESC;
                        let mut z = cb[m];
                        for j in 0..DESC {
                            z += d[db + j] * cw[m * DESC + j];
                        }
                        let a = z.tanh();
                        if a > best {
                            best = a;
                            bg = g;
                        }
                    }
                    arg[(s * NP + p) * M + m] = bg;
                    amax[(s * NP + p) * M + m] = best;
                    pooled[s * M + m] += best / NP as f32;
                }
            }
        }
        (arg, amax, pooled)
    };
    let logits = |rw: &[f32], rb: &[f32], pooled: &[f32], n: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; n * K];
        for s in 0..n {
            for k in 0..K {
                let mut z = rb[k];
                for m in 0..M {
                    z += pooled[s * M + m] * rw[k * M + m];
                }
                l[s * K + k] = z;
            }
        }
        l
    };

    for _ in 0..280 {
        let (arg, amax, pooled) = fwd(&cw, &cb, d_tr, n_tr);
        let probs = softmax_k(&logits(&rw, &rb, &pooled, n_tr), n_tr, K);
        let mut grad_pool = vec![0.0f32; n_tr * M];
        let (mut grw, mut grb) = (vec![0.0f32; K * M], vec![0.0f32; K]);
        for s in 0..n_tr {
            for k in 0..K {
                let dz = (probs[s * K + k] - f32::from(y_tr[s] == k)) / n_tr as f32;
                grb[k] += dz;
                for m in 0..M {
                    grw[k * M + m] += dz * pooled[s * M + m];
                    grad_pool[s * M + m] += dz * rw[k * M + m];
                }
            }
        }
        // Conv grads route through the mean-pool + group-max (only the argmax frame gets grad).
        let (mut gcw, mut gcb) = (vec![0.0f32; M * DESC], vec![0.0f32; M]);
        for s in 0..n_tr {
            for p in 0..NP {
                for m in 0..M {
                    let a = amax[(s * NP + p) * M + m];
                    let gz = grad_pool[s * M + m] / NP as f32 * (1.0 - a * a);
                    let g = arg[(s * NP + p) * M + m];
                    let db = ((s * go + g) * NP + p) * DESC;
                    gcb[m] += gz;
                    for j in 0..DESC {
                        gcw[m * DESC + j] += gz * d_tr[db + j];
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
            rw[i] -= lr * grw[i];
        }
        for i in 0..K {
            rb[i] -= lr * grb[i];
        }
    }
    let (_a, _m, pooled) = fwd(&cw, &cb, d_te, n_te);
    accuracy_k(&logits(&rw, &rb, &pooled, n_te), y_te, n_te, K)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
#[ignore = "heavy group-conv measurement (~140s, |G|=8 frames); run with --ignored. Correctness is gated by dihedral_hypergraph.rs"]
fn dihedral_group_conv_vs_canonical_vs_raw() {
    let group = DihedralGroup::new(8, false); // C_8 — 8 rotation frames (45° steps)
    let (mut raw, mut canon, mut grp) = (Vec::new(), Vec::new(), Vec::new());
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (x_tr, y_tr) = make_set(360, &mut rng);
        let (x_te, y_te) = make_set(160, &mut rng);
        let r = train_eval(
            &single_frame(&x_tr, 360, false),
            1,
            &y_tr,
            &single_frame(&x_te, 160, false),
            &y_te,
            seed + 1,
        );
        let c = train_eval(
            &single_frame(&x_tr, 360, true),
            1,
            &y_tr,
            &single_frame(&x_te, 160, true),
            &y_te,
            seed + 1,
        );
        let (gtr, go) = group_frames(&x_tr, 360, group);
        let (gte, _) = group_frames(&x_te, 160, group);
        let g = train_eval(&gtr, go, &y_tr, &gte, &y_te, seed + 1);
        eprintln!("seed {seed}: raw {r:.3}  canonical {c:.3}  C_8 group-conv {g:.3}");
        raw.push(r);
        canon.push(c);
        grp.push(g);
    }
    let (rm, cm, gm) = (median(raw), median(canon), median(grp));
    eprintln!("Nagare CV — dihedral group-conv vs single-θ canonicalization (rotated shapes):");
    eprintln!("  raw {rm:.3}   single-θ canonical {cm:.3}   C_8 group-conv {gm:.3}");
    eprintln!(
        "  verdict: C_8 group-conv {} single-θ canonicalization (Δ {:+.3}); both beat raw (+{:.3}/{:+.3})",
        if gm > cm + 0.01 {
            "beats"
        } else if gm < cm - 0.01 {
            "trails"
        } else {
            "matches"
        },
        gm - cm,
        cm - rm,
        gm - rm
    );
    // Gate: the rotation-aware arms clear raw; which of canonical/group wins is the measurement.
    assert!(
        cm > rm && gm > rm,
        "rotation-aware arms failed to beat raw: raw {rm:.3} canon {cm:.3} grp {gm:.3}"
    );
}
