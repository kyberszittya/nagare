//! Nagare CV — **quaternion patch convolution** (the user's "convolution → CV, not pool").
//!
//! Prior rotor-*pool* probes all failed to beat baselines: free-learned rotors on arbitrary
//! channels HURT (0.52 vs 0.71), geometric-angle rotors on non-equivariant learned tokens HURT
//! (0.56 vs 0.71), rotor on the equivariant gradient field only TIED (0.45 vs 0.44). Lesson:
//! the rotor must (a) act on an **equivariant** quantity (the gradient field) and (b) live in
//! the **convolution** (which keeps discriminative structure), not an orderless pool.
//!
//! This design: the rotor canonicalises the gradient field per patch (θ_p = dominant grad
//! orientation → z-rotation by −θ_p; gradients co-rotate exactly, so this is rotation-invariant),
//! then a **learned filter bank** convolves the canonical per-patch descriptor (a 1×1 patch conv
//! + tanh) → feature map → mean-pool → readout. Rotor = invariance; conv = capacity.
//!
//! Ablation (reported, §3): the SAME conv on **rotor-canonical** vs **raw** gradient descriptors,
//! 4 seeds, randomly-rotated shapes. Signed-graph link prediction stays the flagship.

use holonomy_learn::{accuracy_k, cayley_rotor_forward, softmax_k};
use rand::{rngs::StdRng, Rng, SeedableRng};

const G: usize = 12;
const K: usize = 4;
const NP: usize = 16;
const PS: usize = 3;
const PR: usize = 4;
const CELLS: usize = PS * PS;
const DESC: usize = CELLS * 2; // per-patch canonical gradient descriptor (9 (gx,gy) pairs)
const M: usize = 12; // conv filter bank size

fn strokes(class: usize) -> Vec<(f32, f32)> {
    let line = |ax: f32, ay: f32, t0: f32, t1: f32| -> Vec<(f32, f32)> {
        (0..7)
            .map(|i| {
                let t = t0 + (t1 - t0) * i as f32 / 6.0;
                (ax * t, ay * t)
            })
            .collect()
    };
    match class {
        0 => line(1.0, 0.0, -0.6, 0.6),
        1 => [line(1.0, 0.0, -0.6, 0.6), line(0.0, 1.0, -0.6, 0.6)].concat(),
        2 => [line(1.0, 0.0, 0.0, 0.6), line(0.0, 1.0, 0.0, 0.6)].concat(),
        _ => [line(1.0, 0.0, -0.6, 0.6), line(0.0, 1.0, -0.6, 0.0)].concat(),
    }
}

fn render(class: usize, theta: f32, rng: &mut StdRng) -> Vec<f32> {
    let (c, s) = (theta.cos(), theta.sin());
    let pts: Vec<(f32, f32)> = strokes(class)
        .iter()
        .map(|&(x, y)| (x * c - y * s, x * s + y * c))
        .collect();
    let sig2 = 0.12f32 * 0.12;
    let mut img = vec![0.0f32; G * G];
    for i in 0..G {
        for j in 0..G {
            let cy = (i as f32 + 0.5) / G as f32 * 2.0 - 1.0;
            let cx = (j as f32 + 0.5) / G as f32 * 2.0 - 1.0;
            let mut v = 0.0f32;
            for &(px, py) in &pts {
                v += (-((cx - px).powi(2) + (cy - py).powi(2)) / (2.0 * sig2)).exp();
            }
            v += 0.08 * (rng.random::<f32>() * 2.0 - 1.0);
            img[i * G + j] = 2.0 * v.min(1.0) - 1.0;
        }
    }
    img
}

fn make_set(n: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<usize>) {
    let mut x = vec![0.0f32; n * G * G];
    let mut y = vec![0usize; n];
    for s in 0..n {
        let c = rng.random_range(0..K);
        let theta = rng.random::<f32>() * std::f32::consts::TAU;
        y[s] = c;
        x[s * G * G..(s + 1) * G * G].copy_from_slice(&render(c, theta, rng));
    }
    (x, y)
}

fn grad_at(img: &[f32], i: usize, j: usize) -> (f32, f32) {
    let at = |a: i32, b: i32| {
        let (a, b) = (
            a.clamp(0, G as i32 - 1) as usize,
            b.clamp(0, G as i32 - 1) as usize,
        );
        img[a * G + b]
    };
    (
        at(i as i32, j as i32 + 1) - at(i as i32, j as i32 - 1),
        at(i as i32 + 1, j as i32) - at(i as i32 - 1, j as i32),
    )
}

/// Per-patch gradient descriptor `(n, NP, DESC)`. When `canonical`, each patch's 9 cell gradients
/// are rotor-rotated by −θ_p into its canonical frame (rotation-invariant); else raw.
fn descriptors(x: &[f32], n: usize, canonical: bool) -> Vec<f32> {
    let ntri = n * NP * CELLS;
    let mut v = vec![0.0f32; ntri * 3];
    let mut biv = vec![0.0f32; ntri * 3];
    for s in 0..n {
        let img = &x[s * G * G..(s + 1) * G * G];
        for p in 0..NP {
            let (prow, pcol) = (p / PR, p % PR);
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            let mut cell = [(0.0f32, 0.0f32); CELLS];
            for a in 0..PS {
                for bb in 0..PS {
                    let g = grad_at(img, prow * PS + a, pcol * PS + bb);
                    cell[a * PS + bb] = g;
                    sx += g.0;
                    sy += g.1;
                }
            }
            let tz = if canonical {
                (-0.5 * sy.atan2(sx)).tan()
            } else {
                0.0
            };
            for (c, &(gx, gy)) in cell.iter().enumerate() {
                let base = ((s * NP + p) * CELLS + c) * 3;
                v[base] = gx;
                v[base + 1] = gy;
                biv[base + 2] = tz;
            }
        }
    }
    let (rot, _q) = cayley_rotor_forward(&biv, &v, ntri);
    let mut desc = vec![0.0f32; n * NP * DESC];
    for s in 0..n {
        for p in 0..NP {
            for c in 0..CELLS {
                let base = ((s * NP + p) * CELLS + c) * 3;
                desc[(s * NP + p) * DESC + c * 2] = rot[base];
                desc[(s * NP + p) * DESC + c * 2 + 1] = rot[base + 1];
            }
        }
    }
    desc
}

/// Train the quaternion patch conv (learned filter bank + tanh → mean-pool → readout) on fixed
/// per-patch descriptors; return test accuracy.
fn train_eval(d_tr: &[f32], y_tr: &[usize], d_te: &[f32], y_te: &[usize], seed: u64) -> f32 {
    let (n_tr, n_te) = (y_tr.len(), y_te.len());
    let mut rng = StdRng::seed_from_u64(seed);
    let rv = |n: usize, sc: f32, r: &mut StdRng| -> Vec<f32> {
        (0..n)
            .map(|_| (r.random::<f32>() * 2.0 - 1.0) * sc)
            .collect()
    };
    let mut cw = rv(M * DESC, 0.2, &mut rng); // conv filter bank (M × DESC)
    let mut cb = vec![0.0f32; M];
    let mut rw = rv(K * M, 0.2, &mut rng); // readout (K × M)
    let mut rb = vec![0.0f32; K];
    let lr = 0.15;

    // Conv+tanh over patches → activation map (n, NP, M); mean-pool → (n, M).
    let conv = |cw: &[f32], cb: &[f32], d: &[f32], n: usize| -> (Vec<f32>, Vec<f32>) {
        let mut act = vec![0.0f32; n * NP * M];
        let mut pooled = vec![0.0f32; n * M];
        for s in 0..n {
            for p in 0..NP {
                for m in 0..M {
                    let mut z = cb[m];
                    for j in 0..DESC {
                        z += d[(s * NP + p) * DESC + j] * cw[m * DESC + j];
                    }
                    let a = z.tanh();
                    act[(s * NP + p) * M + m] = a;
                    pooled[s * M + m] += a / NP as f32;
                }
            }
        }
        (act, pooled)
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
        let (act, pooled) = conv(&cw, &cb, d_tr, n_tr);
        let lg = logits(&rw, &rb, &pooled, n_tr);
        let probs = softmax_k(&lg, n_tr, K);
        // Readout grads + grad wrt pooled.
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
        // Conv grads through tanh + mean-pool.
        let (mut gcw, mut gcb) = (vec![0.0f32; M * DESC], vec![0.0f32; M]);
        for s in 0..n_tr {
            for p in 0..NP {
                for m in 0..M {
                    let a = act[(s * NP + p) * M + m];
                    let gz = grad_pool[s * M + m] / NP as f32 * (1.0 - a * a);
                    gcb[m] += gz;
                    for j in 0..DESC {
                        gcw[m * DESC + j] += gz * d_tr[(s * NP + p) * DESC + j];
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
    let (_a, pooled) = conv(&cw, &cb, d_te, n_te);
    accuracy_k(&logits(&rw, &rb, &pooled, n_te), y_te, n_te, K)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn quat_conv_canonical_vs_raw_on_rotated_shapes() {
    let (mut cn, mut rw, mut wins) = (Vec::new(), Vec::new(), 0);
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (x_tr, y_tr) = make_set(360, &mut rng);
        let (x_te, y_te) = make_set(160, &mut rng);
        let c = train_eval(
            &descriptors(&x_tr, 360, true),
            &y_tr,
            &descriptors(&x_te, 160, true),
            &y_te,
            seed + 1,
        );
        let r = train_eval(
            &descriptors(&x_tr, 360, false),
            &y_tr,
            &descriptors(&x_te, 160, false),
            &y_te,
            seed + 1,
        );
        eprintln!("seed {seed}: quat-conv canonical {c:.3}  raw {r:.3}");
        if c > r {
            wins += 1;
        }
        cn.push(c);
        rw.push(r);
    }
    let (cm, rm) = (median(cn), median(rw));
    eprintln!("Nagare CV — quaternion patch convolution (rotor-canonical vs raw, rotated shapes):");
    eprintln!("  canonical median acc {cm:.3}   raw median acc {rm:.3}");
    eprintln!(
        "  verdict: rotor canonicalization in the conv {} raw — better on {wins}/4 seeds (Δ {:+.3})",
        if wins >= 3 { "HELPS" } else { "does not robustly beat" },
        cm - rm
    );
    assert!(
        cm > 0.4 && rm > 0.25,
        "conv failed to learn: canon {cm:.3}, raw {rm:.3}"
    );
}
