//! SBSH proof-of-concept smoke (handoff §8) — prove the TWO HINGES before building any detector:
//!   H1. the dynamic spatial tree (quadtree, split-by-gradient-energy) concentrates cells on OBJECTS
//!       more than a uniform grid at the same cell budget;
//!   H2. a node/shape descriptor (`spatial_phase_features` |DFT|, the phase-pool invariant) is
//!       ROTATION-ROBUST.
//! Synthetic scene = K filled oriented rectangles on a flat background (ground-truth oriented boxes).
//! No detector, no training — this is the cheapest test that the central mechanism is sound.
//! Dumps a viz (`scene.bin` + `boxes.txt`) for `scripts/dev/render_sbsh_tree.py`.
//!
//! Run: `cargo run --release --example sbsh_tree_smoke -- [--out /tmp/sbsh] [--seed 0]`

use std::io::Write;
use std::path::Path;

use holonomy_learn::{rotate_image, spatial_phase_features, PhaseFeature};
use rand::{rngs::StdRng, Rng, SeedableRng};

fn arg(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}

const G: usize = 96; // canvas side

/// Oriented ground-truth box (pixel coords, radians).
#[derive(Clone, Copy)]
struct Obj {
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    theta: f32,
}

impl Obj {
    /// Is pixel (i=row, j=col) inside this oriented rectangle?
    fn contains(&self, i: usize, j: usize) -> bool {
        let (dx, dy) = (j as f32 - self.cx, i as f32 - self.cy);
        let (c, s) = (self.theta.cos(), self.theta.sin());
        let rx = dx * c + dy * s;
        let ry = -dx * s + dy * c;
        rx.abs() <= self.w * 0.5 && ry.abs() <= self.h * 0.5
    }
}

/// Synthetic scene: `k` filled oriented rects (value +1) on a flat background (−1) + ground truth.
fn gen_scene(k: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<Obj>) {
    let mut img = vec![-1.0f32; G * G];
    let mut objs = Vec::with_capacity(k);
    for _ in 0..k {
        let w: f32 = rng.random_range(12.0..26.0);
        let h: f32 = rng.random_range(8.0..18.0);
        let m = w.max(h) * 0.6;
        let cx: f32 = rng.random_range(m..(G as f32 - m));
        let cy: f32 = rng.random_range(m..(G as f32 - m));
        let theta: f32 = rng.random_range(0.0..std::f32::consts::PI);
        let o = Obj {
            cx,
            cy,
            w,
            h,
            theta,
        };
        for i in 0..G {
            for j in 0..G {
                if o.contains(i, j) {
                    img[i * G + j] = 1.0;
                }
            }
        }
        objs.push(o);
    }
    (img, objs)
}

fn central_diff(img: &[f32], i: usize, j: usize) -> (f32, f32) {
    let at = |a: i32, b: i32| {
        let a = a.clamp(0, G as i32 - 1) as usize;
        let b = b.clamp(0, G as i32 - 1) as usize;
        img[a * G + b]
    };
    (
        at(i as i32, j as i32 + 1) - at(i as i32, j as i32 - 1),
        at(i as i32 + 1, j as i32) - at(i as i32 - 1, j as i32),
    )
}

/// Axis-aligned tree cell `[y0,y1) × [x0,x1)`.
#[derive(Clone, Copy)]
struct Cell {
    y0: usize,
    x0: usize,
    y1: usize,
    x1: usize,
}

/// Mean gradient magnitude in a cell — the split score (content ⇒ high, flat background ⇒ ~0).
fn cell_energy(img: &[f32], c: &Cell) -> f32 {
    let mut e = 0.0f32;
    let mut n = 0usize;
    for i in c.y0..c.y1 {
        for j in c.x0..c.x1 {
            let (gx, gy) = central_diff(img, i, j);
            e += (gx * gx + gy * gy).sqrt();
            n += 1;
        }
    }
    e / n.max(1) as f32
}

/// Dynamic quadtree: split a cell into 4 while its gradient energy exceeds `thresh` (up to
/// `max_depth`, min side `min_side`). Structural — no backward (the `cpml_tier` discipline). Returns
/// the leaf cells.
fn build_tree(
    img: &[f32],
    c: Cell,
    depth: usize,
    max_depth: usize,
    thresh: f32,
    min_side: usize,
    out: &mut Vec<Cell>,
) {
    let side = (c.y1 - c.y0).min(c.x1 - c.x0);
    if depth < max_depth && side >= 2 * min_side && cell_energy(img, &c) > thresh {
        let my = (c.y0 + c.y1) / 2;
        let mx = (c.x0 + c.x1) / 2;
        for (y0, y1, x0, x1) in [
            (c.y0, my, c.x0, mx),
            (c.y0, my, mx, c.x1),
            (my, c.y1, c.x0, mx),
            (my, c.y1, mx, c.x1),
        ] {
            build_tree(
                img,
                Cell { y0, x0, y1, x1 },
                depth + 1,
                max_depth,
                thresh,
                min_side,
                out,
            );
        }
    } else {
        out.push(c);
    }
}

/// Fraction of a cell's pixels that lie on some object (ground-truth coverage of the cell).
fn cell_on_object(objs: &[Obj], c: &Cell) -> f32 {
    let (mut on, mut n) = (0usize, 0usize);
    for i in c.y0..c.y1 {
        for j in c.x0..c.x1 {
            if objs.iter().any(|o| o.contains(i, j)) {
                on += 1;
            }
            n += 1;
        }
    }
    on as f32 / n.max(1) as f32
}

/// Crop a `s×s` region centred on an object and return it (edge-clamped) for the descriptor test.
fn crop(img: &[f32], cy: f32, cx: f32, s: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; s * s];
    let (y0, x0) = (cy as i32 - s as i32 / 2, cx as i32 - s as i32 / 2);
    for i in 0..s {
        for j in 0..s {
            let a = (y0 + i as i32).clamp(0, G as i32 - 1) as usize;
            let b = (x0 + j as i32).clamp(0, G as i32 - 1) as usize;
            out[i * s + j] = img[a * G + b];
        }
    }
    out
}

fn main() {
    let out = arg("--out").unwrap_or_else(|| "/tmp/sbsh".into());
    let seed = arg("--seed").and_then(|s| s.parse().ok()).unwrap_or(0u64);
    let mut rng = StdRng::seed_from_u64(seed);
    let k = 4;
    let (img, objs) = gen_scene(k, &mut rng);

    // --- Dynamic tree ---
    let root = Cell {
        y0: 0,
        x0: 0,
        y1: G,
        x1: G,
    };
    let (max_depth, min_side) = (5usize, 3usize);
    let thresh = 0.05; // mean-gradient threshold: background ~0, object edges >0
    let mut leaves = Vec::new();
    build_tree(&img, root, 0, max_depth, thresh, min_side, &mut leaves);

    // H1: does the adaptive tree concentrate cells on objects vs a uniform grid at equal budget?
    let n_leaf = leaves.len();
    let adaptive_on: f32 = leaves
        .iter()
        .filter(|c| cell_on_object(&objs, c) > 0.5)
        .count() as f32
        / n_leaf as f32;
    // Uniform grid with ~n_leaf cells (side gu×gu).
    let gu = (n_leaf as f32).sqrt().round() as usize;
    let step = G / gu.max(1);
    let mut uni = Vec::new();
    let mut y = 0;
    while y < G {
        let mut x = 0;
        while x < G {
            uni.push(Cell {
                y0: y,
                x0: x,
                y1: (y + step).min(G),
                x1: (x + step).min(G),
            });
            x += step;
        }
        y += step;
    }
    let uniform_on: f32 = uni
        .iter()
        .filter(|c| cell_on_object(&objs, c) > 0.5)
        .count() as f32
        / uni.len() as f32;
    // Object area fraction (the neutral baseline: a uniform grid's on-object fraction ≈ this).
    let obj_area: f32 = (0..G * G)
        .filter(|&p| objs.iter().any(|o| o.contains(p / G, p % G)))
        .count() as f32
        / (G * G) as f32;

    // Mean leaf side on objects vs off — the tree should make on-object cells FINER.
    let mut on_side = (0.0f32, 0usize);
    let mut off_side = (0.0f32, 0usize);
    for c in &leaves {
        let s = (c.y1 - c.y0) as f32;
        if cell_on_object(&objs, c) > 0.5 {
            on_side.0 += s;
            on_side.1 += 1;
        } else {
            off_side.0 += s;
            off_side.1 += 1;
        }
    }
    let mean_on_side = on_side.0 / on_side.1.max(1) as f32;
    let mean_off_side = off_side.0 / off_side.1.max(1) as f32;

    // H2: rotation-robustness of a shape crop's |DFT| phase-pool descriptor.
    let b = 18;
    let s = 40usize;
    let o0 = objs[0];
    let cr = crop(&img, o0.cy, o0.cx, s);
    let (d0, dim) = spatial_phase_features(&cr, 1, s, 1, b, PhaseFeature::Dft);
    let mut drift_sum = 0.0f32;
    let n_ang = 8;
    for a in 1..=n_ang {
        let phi = a as f32 / n_ang as f32 * std::f32::consts::TAU;
        let rot = rotate_image(&cr, s, phi);
        let (dphi, _) = spatial_phase_features(&rot, 1, s, 1, b, PhaseFeature::Dft);
        let l2: f32 = d0
            .iter()
            .zip(&dphi)
            .map(|(a, c)| (a - c).powi(2))
            .sum::<f32>()
            .sqrt();
        let norm: f32 = d0.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-6;
        drift_sum += l2 / norm;
    }
    let mean_rel_drift = drift_sum / n_ang as f32;

    // --- Report ---
    println!("SBSH tree smoke (seed {seed}, {G}×{G}, k={k} objects, obj_area {obj_area:.3})");
    println!("  H1 concentration:");
    println!("     adaptive leaves        : {n_leaf}");
    println!("     on-object cell fraction: adaptive {adaptive_on:.3}  vs  uniform {uniform_on:.3}  (obj_area {obj_area:.3})");
    println!("     mean leaf side         : on-object {mean_on_side:.1}px  vs  off-object {mean_off_side:.1}px  (finer-on-object = {})",
        if mean_on_side < mean_off_side { "YES" } else { "no" });
    println!("  H2 rotation robustness:");
    println!("     phase-pool |DFT| dim {dim}, mean rel. drift over {n_ang} rotations: {mean_rel_drift:.4}  ({})",
        if mean_rel_drift < 0.15 { "ROBUST" } else { "weak" });

    // --- Viz dump ---
    std::fs::create_dir_all(&out).unwrap();
    let bytes: Vec<u8> = img
        .iter()
        .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8)
        .collect();
    std::fs::write(Path::new(&out).join("scene.bin"), &bytes).unwrap();
    let mut f = std::fs::File::create(Path::new(&out).join("boxes.txt")).unwrap();
    writeln!(f, "{G}").unwrap();
    writeln!(f, "GT {}", objs.len()).unwrap();
    for o in &objs {
        writeln!(
            f,
            "{:.2} {:.2} {:.2} {:.2} {:.4}",
            o.cx, o.cy, o.w, o.h, o.theta
        )
        .unwrap();
    }
    writeln!(f, "LEAF {}", leaves.len()).unwrap();
    for c in &leaves {
        writeln!(f, "{} {} {} {}", c.x0, c.y0, c.x1, c.y1).unwrap();
    }
    println!("  wrote viz → {out}/scene.bin + boxes.txt");
}
