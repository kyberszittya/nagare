//! Shared synthetic-vision scaffolding: randomly-rotated shape rendering + the per-patch
//! gradient field. Used by `vision_quat_conv` (single-θ canonicalisation) and
//! `vision_dihedral_conv` (D_n group-conv) so the task + gradient extraction live in one place.

use holonomy_learn::{accuracy_k, softmax_k};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f32::consts::TAU;

pub const B: usize = 12; // orientation-histogram bins
pub const NK: usize = B / 2 + 1; // DFT magnitudes kept (k = 0..B/2)
pub const G: usize = 12; // grid side
pub const K: usize = 4; // shape classes (bar / cross / L / T)
pub const PS: usize = 3; // patch side
pub const PR: usize = 4; // patches per row (G/PS)
pub const NP: usize = 16; // patches
pub const CELLS: usize = PS * PS; // gradient cells per patch

/// Stroke points (centred, radius ≤ 0.6) for each shape class — distinguishable by arm topology,
/// hence rotation-invariant.
pub fn strokes(class: usize) -> Vec<(f32, f32)> {
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

/// Render one shape at rotation `theta` (+noise) → flat `G*G` in ~[-1,1].
pub fn render(class: usize, theta: f32, rng: &mut StdRng) -> Vec<f32> {
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

/// `n` randomly-rotated labelled shapes: flat `(n, G*G)` + labels.
pub fn make_set(n: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<usize>) {
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

/// Central-difference image gradient at cell `(i,j)` (clamped at borders).
pub fn grad_at(img: &[f32], i: usize, j: usize) -> (f32, f32) {
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

/// Per-patch per-cell gradient 3-vectors `(gx, gy, 0)`, flat `(n·NP·CELLS, 3)` — the equivariant
/// field both vision tests transform. Also returns per-patch dominant orientation `θ_p`
/// (`atan2(Σ∂y, Σ∂x)`), flat `(n·NP)`.
pub fn patch_gradient_field(x: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut field = vec![0.0f32; n * NP * CELLS * 3];
    let mut theta = vec![0.0f32; n * NP];
    for s in 0..n {
        let img = &x[s * G * G..(s + 1) * G * G];
        for p in 0..NP {
            let (prow, pcol) = (p / PR, p % PR);
            let (mut sx, mut sy) = (0.0f32, 0.0f32);
            for a in 0..PS {
                for b in 0..PS {
                    let (gx, gy) = grad_at(img, prow * PS + a, pcol * PS + b);
                    let base = ((s * NP + p) * CELLS + a * PS + b) * 3;
                    field[base] = gx;
                    field[base + 1] = gy;
                    sx += gx;
                    sy += gy;
                }
            }
            theta[s * NP + p] = sy.atan2(sx);
        }
    }
    (field, theta)
}

/// Which phase-pool feature to build from the orientation histogram.
#[derive(Clone, Copy)]
pub enum PhaseFeature {
    /// The histogram itself — rotation-*covariant* (a floor baseline).
    Raw,
    /// `|DFT(h)|_{0..B/2}` — rotation-*invariant* (circular-shift-invariant magnitudes).
    Dft,
    /// `|DFT(h)|` ⊕ phase entropy `H(h)` — invariant, plus the nonlinear entropy feature.
    DftEntropy,
}

/// Per-image magnitude-weighted **orientation histogram** `(n, B)` — the pooled rotor phases:
/// each patch's dominant gradient contributes its orientation `θ_p` (weighted by magnitude) to the
/// circular histogram (soft-binned). Under image rotation, `θ_p → θ_p + φ` shifts the histogram.
pub fn phase_histogram(x: &[f32], n: usize) -> Vec<f32> {
    let (field, _theta) = patch_gradient_field(x, n);
    let mut h = vec![0.0f32; n * B];
    for s in 0..n {
        for p in 0..NP {
            let (mut gx, mut gy) = (0.0f32, 0.0f32);
            for c in 0..CELLS {
                let base = ((s * NP + p) * CELLS + c) * 3;
                gx += field[base];
                gy += field[base + 1];
            }
            let m = (gx * gx + gy * gy).sqrt();
            let theta = gy.atan2(gx).rem_euclid(TAU);
            let pos = theta / TAU * B as f32;
            let lo = pos.floor() as usize % B;
            let frac = pos - pos.floor();
            h[s * B + lo] += m * (1.0 - frac);
            h[s * B + (lo + 1) % B] += m * frac;
        }
    }
    h
}

/// Build the requested phase-pool feature from the histogram `(n, B)` → `(features, dim)`.
pub fn phase_features(h: &[f32], n: usize, mode: PhaseFeature) -> (Vec<f32>, usize) {
    if let PhaseFeature::Raw = mode {
        return (h.to_vec(), B);
    }
    let with_entropy = matches!(mode, PhaseFeature::DftEntropy);
    let dim = NK + usize::from(with_entropy);
    let mut f = vec![0.0f32; n * dim];
    for s in 0..n {
        let hs = &h[s * B..s * B + B];
        for (k, fk) in f[s * dim..s * dim + NK].iter_mut().enumerate() {
            let (mut re, mut im) = (0.0f32, 0.0f32);
            for (b, &hv) in hs.iter().enumerate() {
                let ang = -TAU * k as f32 * b as f32 / B as f32;
                re += hv * ang.cos();
                im += hv * ang.sin();
            }
            *fk = (re * re + im * im).sqrt();
        }
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

/// Train a linear softmax classifier on fixed per-image features (train-standardised); return
/// test accuracy. Shared by the phase-pool tests.
pub fn train_linear(
    f_tr: &[f32],
    dim: usize,
    y_tr: &[usize],
    f_te: &[f32],
    y_te: &[usize],
    seed: u64,
) -> f32 {
    let (n_tr, n_te) = (y_tr.len(), y_te.len());
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
