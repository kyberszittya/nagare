//! Neocognitron N2 — the S/C stack and a ROTATION-ROBUSTNESS A/B that isolates
//! the C-cell, no autograd. An oriented bar at angle θ with contrast c is
//! rendered; the target is c (rotation-INVARIANT). The stack:
//!   `S-cell (fixed Sobel gx,gy) → oriented vectors → group_pool (C-cell) → mean → linear → c`
//!
//! The A/B is the C-cell's group: **C₈** (steer over 8 rotations → rotation-
//! INVARIANT) vs **C₁** (`--c1`, no orbit → orientation-SPECIFIC baseline). Same
//! op, only the group differs. The S-cell is a *fixed* oriented filter so the
//! network cannot dodge the orientation problem with an isotropic filter (a
//! learnable conv does exactly that on this easy task — then both generalise,
//! which is why the S-cell is fixed here to isolate the C-cell). Train on two
//! orientations, test on HELD-OUT group-element orientations.
//!
//! Run: `cargo run --release --example neocognitron_rot -- [--c1] [out.json]`

use holonomy_learn::{
    adam_step, conv2d_forward, group_pool_backward, group_pool_forward, linear_backward,
    linear_forward, AdamState, ConvLayer, ConvShape, DihedralGroup, LinearLayer,
};
use std::f32::consts::PI;
use std::io::Write;

const G: usize = 20;
const GG: usize = G * G;

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// A bar through the centre at angle `theta` (rad), amplitude `c`, on a G×G grid.
fn render_bar(theta: f32, c: f32) -> Vec<f32> {
    let mut img = vec![0.0f32; GG];
    let cc = (G - 1) as f32 / 2.0;
    let (ct, st) = (theta.cos(), theta.sin());
    for r in 0..G {
        for col in 0..G {
            let (dx, dy) = (col as f32 - cc, r as f32 - cc);
            let perp = (-dx * st + dy * ct).abs();
            let along = (dx * ct + dy * st).abs();
            if perp <= 1.2 && along <= 7.0 {
                img[r * G + col] = c;
            }
        }
    }
    img
}

/// Fixed oriented S-cell: 3×3 Sobel gx (channel 0) and gy (channel 1).
fn sobel_scell() -> ConvLayer {
    let mut c = ConvLayer::new(1, 2, 3, 3, 0);
    c.w = vec![
        -1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0, // gx
        -1.0, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0, // gy
    ];
    c.b = vec![0.0, 0.0];
    c
}

/// Per-location oriented vectors (GG × 3) = `(gx, gy, 0)` from the S-cell output.
fn to_vectors(y: &[f32]) -> Vec<f32> {
    let mut v = vec![0.0f32; GG * 3];
    for p in 0..GG {
        v[p * 3] = y[p];
        v[p * 3 + 1] = y[GG + p];
    }
    v
}

fn main() {
    let c1 = flag("--c1");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/neocognitron-rot.json".into());
    let group = if c1 {
        DihedralGroup::new(1, false)
    } else {
        DihedralGroup::new(8, false)
    };
    let tau = 0.3f32;
    let scell = sobel_scell();
    let s = ConvShape {
        c_in: 1,
        h: G,
        w: G,
        pad: 1,
    };

    let train_ang: Vec<f32> = [0.0f32, 90.0].iter().map(|d| d * PI / 180.0).collect();
    let test_ang: Vec<f32> = [45.0f32, 135.0, 180.0, 225.0, 270.0, 315.0]
        .iter()
        .map(|d| d * PI / 180.0)
        .collect();

    let mut filt = vec![0.6f32, -0.3, 0.2];
    let mut head = LinearLayer::new(1, 1, 9);
    let mut sfilt = AdamState::new(3);
    let (mut shw, mut shb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));

    let mut st: u64 = 12345;
    let mut nextc = || {
        st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
        0.4 + 0.6 * (((st >> 33) as f32) / (u32::MAX as f32))
    };

    // feature = mean over locations of the group_pool response (1 oriented unit).
    let feat_of = |filt: &[f32], img: &[f32]| -> (f32, holonomy_learn::GroupPoolOut) {
        let (y, _, _) = conv2d_forward(&scell, img, s);
        let v = to_vectors(&y);
        let gp = group_pool_forward(&v, group, filt, tau);
        (gp.resp.iter().sum::<f32>() / GG as f32, gp)
    };
    let eval =
        |filt: &[f32], head: &LinearLayer, angs: &[f32], nc: &mut dyn FnMut() -> f32| -> f32 {
            let mut e = 0.0f32;
            let mut n = 0usize;
            for &th in angs {
                for _ in 0..8 {
                    let c = nc();
                    let (f, _) = feat_of(filt, &render_bar(th, c));
                    let p = linear_forward(head, &[f])[0];
                    e += (p - c).powi(2);
                    n += 1;
                }
            }
            e / n as f32
        };

    for _ in 0..2000 {
        for &th in &train_ang {
            let c = nextc();
            let img = render_bar(th, c);
            let (f, gp) = feat_of(&filt, &img);
            let pred = linear_forward(&head, &[f])[0];
            let (gfeat, ghead) = linear_backward(&head, &[f], &[2.0 * (pred - c)]);
            // grad_feat → grad_resp (mean adjoint) → group_pool_backward → grad_filt.
            let gresp = vec![gfeat[0] / GG as f32; GG];
            let (_gv, gfilt) = group_pool_backward(&gp, group, &filt, tau, &gresp);
            adam_step(&mut filt, &gfilt, &mut sfilt, 0.02);
            adam_step(&mut head.w, &ghead.w, &mut shw, 0.02);
            adam_step(&mut head.b, &ghead.b, &mut shb, 0.02);
        }
    }

    let train_mse = eval(&filt, &head, &train_ang, &mut nextc);
    let test_mse = eval(&filt, &head, &test_ang, &mut nextc);
    println!(
        "neocognitron rot ({}): train MSE {train_mse:.4}  held-out-rotation MSE {test_mse:.4}  gap {:.1}x",
        if c1 { "C_1 baseline (no rotation pool)" } else { "C_8 C-cell (rotation-invariant)" },
        test_mse / train_mse.max(1e-6)
    );
    let json = format!(
        "{{\n  \"c1\": {c1},\n  \"train_mse\": {train_mse:.6},\n  \"test_mse\": {test_mse:.6}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
