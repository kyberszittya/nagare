//! Shared synthetic-vision scaffolding: randomly-rotated shape rendering + the per-patch
//! gradient field. Used by `vision_quat_conv` (single-θ canonicalisation) and
//! `vision_dihedral_conv` (D_n group-conv) so the task + gradient extraction live in one place.

use rand::{rngs::StdRng, Rng};

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
