//! Nagare CV — **quaternion-phase pooling + entropy feedback** (the user's fix for the failed
//! rotor-pool).
//!
//! The vector rotor-pool failed because it rotated *non-equivariant feature channels* (scramble).
//! This pools the **rotor phase** instead: each patch's dominant gradient is a z-rotor whose phase
//! is its orientation `θ_p` (`e^{iθ}` = a unit quaternion). Pooling the magnitude-weighted phases
//! over patches is an **orientation histogram** `h`. Under a global image rotation `φ`, every
//! `θ_p→θ_p+φ`, so `h` **circularly shifts** — hence the rotation-**invariant** summaries are
//! `|DFT(h)|_k` (shift-invariant magnitudes) and the **phase entropy** `H(h)` (the entropy the
//! framework's machinery feeds back). No feature vector is ever rotated — only phases are pooled.
//!
//! Three arms (linear classifier on a fixed per-image feature), 4 seeds, randomly-rotated shapes:
//!   - **raw histogram** — `h` itself (rotation-*covariant* floor);
//!   - **phase-pool** — `|DFT(h)|` (rotation-invariant);
//!   - **phase-pool + entropy** — `|DFT(h)|` ⊕ `H(h)` (the entropy-feedback feature).

use holonomy_learn::{accuracy_k, softmax_k};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f32::consts::TAU;

mod common;
use common::vision::{make_set, patch_gradient_field, CELLS, K, NP};

const B: usize = 12; // orientation bins
const NK: usize = B / 2 + 1; // DFT magnitudes kept (k = 0..B/2)

/// Per-image magnitude-weighted orientation histogram `(n, B)` — the pooled rotor phases.
fn phase_histogram(x: &[f32], n: usize) -> Vec<f32> {
    let (field, _theta) = patch_gradient_field(x, n);
    let mut h = vec![0.0f32; n * B];
    for s in 0..n {
        for p in 0..NP {
            // Patch dominant gradient (sum over its cells) → phase θ_p + magnitude m_p.
            let (mut gx, mut gy) = (0.0f32, 0.0f32);
            for c in 0..CELLS {
                let base = ((s * NP + p) * CELLS + c) * 3;
                gx += field[base];
                gy += field[base + 1];
            }
            let m = (gx * gx + gy * gy).sqrt();
            let theta = gy.atan2(gx).rem_euclid(TAU); // phase in [0, 2π)
                                                      // Soft (linear) binning into the circular histogram.
            let pos = theta / TAU * B as f32;
            let lo = pos.floor() as usize % B;
            let frac = pos - pos.floor();
            h[s * B + lo] += m * (1.0 - frac);
            h[s * B + (lo + 1) % B] += m * frac;
        }
    }
    h
}

/// Rotation-invariant features from the histogram: `|DFT(h)|_{0..B/2}`, optionally ⊕ entropy.
fn features(h: &[f32], n: usize, raw: bool, with_entropy: bool) -> (Vec<f32>, usize) {
    if raw {
        return (h.to_vec(), B); // covariant floor
    }
    let dim = NK + usize::from(with_entropy);
    let mut f = vec![0.0f32; n * dim];
    for s in 0..n {
        let hs = &h[s * B..s * B + B];
        // |DFT| magnitudes — invariant to circular shift (i.e. to image rotation).
        for (k, fk) in f[s * dim..s * dim + NK].iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (b, &hv) in hs.iter().enumerate() {
                let ang = -TAU * k as f32 * b as f32 / B as f32;
                re += hv * ang.cos();
                im += hv * ang.sin();
            }
            *fk = (re * re + im * im).sqrt();
        }
        // Phase entropy H(h) — the rotation-invariant entropy-feedback feature.
        if with_entropy {
            let tot: f32 = hs.iter().sum::<f32>() + 1e-6;
            let mut ent = 0.0f32;
            for &hv in hs {
                let pr = hv / tot;
                if pr > 1e-9 {
                    ent -= pr * pr.ln();
                }
            }
            f[s * dim + NK] = ent;
        }
    }
    (f, dim)
}

/// Train a linear softmax classifier on fixed per-image features; return test accuracy.
fn train_eval(
    f_tr: &[f32],
    dim: usize,
    y_tr: &[usize],
    f_te: &[f32],
    y_te: &[usize],
    seed: u64,
) -> f32 {
    let (n_tr, n_te) = (y_tr.len(), y_te.len());
    // Standardise features by train stats (DFT magnitudes and entropy live on different scales).
    let mut mu = vec![0.0f32; dim];
    let mut sd = vec![0.0f32; dim];
    for r in f_tr.chunks(dim) {
        for j in 0..dim {
            mu[j] += r[j] / n_tr as f32;
        }
    }
    for r in f_tr.chunks(dim) {
        for j in 0..dim {
            sd[j] += (r[j] - mu[j]).powi(2) / n_tr as f32;
        }
    }
    for s in &mut sd {
        *s = s.sqrt() + 1e-6;
    }
    let norm = |f: &[f32], n: usize| -> Vec<f32> {
        let mut o = vec![0.0f32; n * dim];
        for i in 0..n {
            for j in 0..dim {
                o[i * dim + j] = (f[i * dim + j] - mu[j]) / sd[j];
            }
        }
        o
    };
    let (ftr, fte) = (norm(f_tr, n_tr), norm(f_te, n_te));

    let mut rng = StdRng::seed_from_u64(seed);
    let mut w: Vec<f32> = (0..dim * K)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
        .collect();
    let mut b = vec![0.0f32; K];
    let logits = |w: &[f32], b: &[f32], f: &[f32], n: usize| -> Vec<f32> {
        let mut l = vec![0.0f32; n * K];
        for i in 0..n {
            for k in 0..K {
                let mut z = b[k];
                for j in 0..dim {
                    z += f[i * dim + j] * w[k * dim + j];
                }
                l[i * K + k] = z;
            }
        }
        l
    };
    for _ in 0..400 {
        let probs = softmax_k(&logits(&w, &b, &ftr, n_tr), n_tr, K);
        for k in 0..K {
            let mut gb = 0.0f32;
            let mut gw = vec![0.0f32; dim];
            for i in 0..n_tr {
                let d = (probs[i * K + k] - f32::from(y_tr[i] == k)) / n_tr as f32;
                gb += d;
                for j in 0..dim {
                    gw[j] += d * ftr[i * dim + j];
                }
            }
            b[k] -= 0.3 * gb;
            for j in 0..dim {
                w[k * dim + j] -= 0.3 * gw[j];
            }
        }
    }
    accuracy_k(&logits(&w, &b, &fte, n_te), y_te, n_te, K)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn phase_pool_entropy_vs_raw_histogram() {
    let (mut raw, mut phase, mut phent) = (Vec::new(), Vec::new(), Vec::new());
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (x_tr, y_tr) = make_set(400, &mut rng);
        let (x_te, y_te) = make_set(160, &mut rng);
        let (htr, hte) = (phase_histogram(&x_tr, 400), phase_histogram(&x_te, 160));
        let ev = |raw: bool, ent: bool| {
            let (ftr, dim) = features(&htr, 400, raw, ent);
            let (fte, _) = features(&hte, 160, raw, ent);
            train_eval(&ftr, dim, &y_tr, &fte, &y_te, seed + 1)
        };
        let (r, p, pe) = (ev(true, false), ev(false, false), ev(false, true));
        eprintln!("seed {seed}: raw-hist {r:.3}  phase-pool {p:.3}  phase+entropy {pe:.3}");
        raw.push(r);
        phase.push(p);
        phent.push(pe);
    }
    let (rm, pm, pem) = (median(raw), median(phase), median(phent));
    eprintln!("Nagare CV — quaternion-phase pooling + entropy feedback (rotated shapes):");
    eprintln!("  raw histogram {rm:.3}   phase-pool |DFT| {pm:.3}   phase-pool + entropy {pem:.3}");
    eprintln!(
        "  verdict: phase-pool {} raw (Δ {:+.3}); entropy feedback {} phase-pool alone (Δ {:+.3})",
        if pm > rm + 0.01 {
            "beats"
        } else if pm < rm - 0.01 {
            "trails"
        } else {
            "matches"
        },
        pm - rm,
        if pem > pm + 0.01 {
            "helps"
        } else if pem < pm - 0.01 {
            "hurts"
        } else {
            "neutral"
        },
        pem - pm
    );
    // Gate: the invariant arms must at least learn above chance; the ranking is the measurement.
    assert!(
        pm > 0.35 && pem > 0.35,
        "phase-pool failed to learn: {pm:.3}/{pem:.3}"
    );
}
