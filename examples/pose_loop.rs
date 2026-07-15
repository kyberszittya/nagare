//! SBSH Pose P4 — a CLOSED KINEMATIC LOOP, the other redundant-constraint
//! archetype, no autograd. A 4-bar **parallelogram** linkage A-B-C-D (ground
//! A-D fixed, crank A-B rotates, coupler C = B + (D−A)). The skeleton is a true
//! 4-CYCLE `[A,B],[B,C],[C,D],[D,A]`. Occluding the coupler joint C **and both
//! its bars** removes its local evidence and the limbs that point at it, so C is
//! recoverable only from the loop (B visible, A/D fixed) — a redundant *cyclic*
//! constraint the local conv head cannot use.
//!
//! HONEST CAVEAT (derived, then measured): with a single shared `elin` the loop
//! edges treat C's two neighbours (B, D) symmetrically, so the residual signed
//! hg_conv recovers toward their **midpoint**, not the exact parallelogram point
//! `C=B+D−A`. This example measures how far that gets — the expected outcome is a
//! *partial* loop win that motivates a per-edge transform (which could express
//! the asymmetric closure exactly). Contrast P3, whose recovery was a single-
//! neighbour translation the shared `elin` expresses exactly.
//!
//! Stack as P2/P3. A/B: `--hg`. Eval occludes the coupler and reports its error.
//! Run: `cargo run --release --example pose_loop -- [--hg] [--seed=N] [out.json]`

use holonomy_learn::{
    adam_step, conv2d_backward, conv2d_forward, hg_edge_to_node_backward, hg_edge_to_node_forward,
    hg_node_to_edge_backward, hg_node_to_edge_forward, linear_backward, linear_forward,
    oriented_sobel_bank, sc_block_backward, sc_block_forward, soft_argmax_backward,
    soft_argmax_forward, AdamState, ConvLayer, ConvShape, DihedralGroup, LinearLayer, ScBlock,
};
use std::io::Write;

const G: usize = 32;
const GG: usize = G * G;
const J: usize = 4; // A(ground), B(crank), C(coupler), D(ground)
const K: usize = 6;
const CRANK: f32 = 8.0;
const A: [f32; 2] = [9.0, 22.0];
const D: [f32; 2] = [23.0, 22.0];

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
    fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.f()
    }
}

/// A random configuration of the parallelogram 4-bar: crank angle θ is the 1 DOF.
/// Returns [A, B, C, D]; C = B + (D − A).
fn sample_pose(rng: &mut Rng) -> [[f32; 2]; J] {
    let th = rng.range(-2.4, -0.7);
    let b = [A[0] + CRANK * th.cos(), A[1] + CRANK * th.sin()];
    let c = [b[0] + (D[0] - A[0]), b[1] + (D[1] - A[1])];
    [A, b, c, D]
}

/// Render the 4 bars; `occ_coupler` masks a box over C and its two bars (B-C, C-D).
fn render(joints: &[[f32; 2]; J], occ_coupler: bool) -> Vec<f32> {
    let mut img = vec![0.0f32; GG];
    let bars = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];
    for &(a, b) in &bars {
        let (p, q) = (joints[a], joints[b]);
        for s in 0..=72 {
            let t = s as f32 / 72.0;
            let (x, y) = (p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t);
            let (ci, ri) = (x.round() as i32, y.round() as i32);
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let (cc, rr) = (ci + dx, ri + dy);
                    if cc >= 0 && rr >= 0 && (cc as usize) < G && (rr as usize) < G {
                        img[rr as usize * G + cc as usize] = 1.0;
                    }
                }
            }
        }
    }
    if occ_coupler {
        // mask C and both its bars: the box spanning B, C, D (leaves A + crank A-B).
        let (b, c, d) = (joints[1], joints[2], joints[3]);
        let xs = [b[0], c[0], d[0]];
        let ys = [b[1], c[1], d[1]];
        let x0 = (xs.iter().cloned().fold(f32::MAX, f32::min) - 2.0).max(0.0) as usize;
        let x1 = (xs.iter().cloned().fold(f32::MIN, f32::max) + 2.0).min((G - 1) as f32) as usize;
        let y0 = (ys.iter().cloned().fold(f32::MAX, f32::min) - 2.0).max(0.0) as usize;
        let y1 = (ys.iter().cloned().fold(f32::MIN, f32::max) + 2.0).min((G - 1) as f32) as usize;
        for r in y0..=y1 {
            for c in x0..=x1 {
                img[r * G + c] = 0.0;
            }
        }
    }
    img
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    let use_hg = flag("--hg");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/pose-loop.json".into());
    let seed_base: u64 = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--seed=")
                .map(|s| s.parse::<u64>().unwrap_or(0))
        })
        .unwrap_or(0);
    let tau = 0.6f32;
    let group = DihedralGroup::new(8, false);
    let s1 = ConvShape {
        c_in: 1,
        h: G,
        w: G,
        pad: 1,
    };
    let hs = ConvShape {
        c_in: K,
        h: G,
        w: G,
        pad: 2,
    };

    let mut b1 = ScBlock::new(1, K, 3, 3, group, tau, 11 + seed_base);
    b1.conv.w = oriented_sobel_bank(K);
    let mut head = ConvLayer::new(K, J, 5, 5, 21 + seed_base);
    // closed-loop skeleton: the 4-cycle A-B-C-D-A.
    let edges = [[0usize, 1], [1, 2], [2, 3], [3, 0]];
    let crank = CRANK;
    let ground = dist(A, D);
    let bone = [crank, ground, crank, ground]; // AB, BC(∥AD), CD(∥AB), DA
    let cycles: Vec<u32> = edges
        .iter()
        .flat_map(|e| [e[0] as u32, e[1] as u32])
        .collect();
    let signs: Vec<f32> = edges.iter().flat_map(|_| [1.0f32, -1.0]).collect();
    let mut deg = [0.0f32; J];
    for e in &edges {
        deg[e[0]] += 1.0;
        deg[e[1]] += 1.0;
    }
    let scale: Vec<f32> = deg.iter().map(|&d| d.max(1.0).powf(-0.5)).collect();
    let mut elin = LinearLayer::new(2, 2, 31 + seed_base);

    let (mut abw, mut abb, mut abf) = (
        AdamState::new(b1.conv.w.len()),
        AdamState::new(b1.conv.b.len()),
        AdamState::new(b1.filt.len()),
    );
    let (mut ahw, mut ahb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));
    let (mut aew, mut aeb) = (AdamState::new(elin.w.len()), AdamState::new(elin.b.len()));

    let refine = |coord: &[f32], elin: &LinearLayer| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        if !use_hg {
            return (coord.to_vec(), Vec::new(), Vec::new());
        }
        let e = hg_node_to_edge_forward(coord, &cycles, &signs, &scale, edges.len(), 2, 2);
        let e2 = linear_forward(elin, &e);
        let r = hg_edge_to_node_forward(&e2, &cycles, &signs, &scale, J, 2, 2);
        let cp: Vec<f32> = coord.iter().zip(&r).map(|(&a, &b)| a + b).collect();
        (cp, e, e2)
    };

    let lambda = 0.1f32;
    let mut rng = Rng(1234 + seed_base.wrapping_mul(7919));
    for _ in 0..1800 {
        let joints = sample_pose(&mut rng);
        let occ = rng.f() < 0.55; // train with random coupler occlusion
        let img = render(&joints, occ);
        let gt: Vec<f32> = joints.iter().flatten().copied().collect();
        let (resp, bcache) = sc_block_forward(&b1, &img, s1);
        let (heat, _, _) = conv2d_forward(&head, &resp, hs);
        let sa = soft_argmax_forward(&heat, J, G, tau);
        let (coord, e, e2) = refine(&sa.coord, &elin);
        let mut gc = vec![0.0f32; J * 2];
        for k in 0..J * 2 {
            gc[k] = 2.0 * (coord[k] - gt[k]) / (J * 2) as f32;
        }
        for (li, ed) in edges.iter().enumerate() {
            let (a, b) = (
                [coord[ed[0] * 2], coord[ed[0] * 2 + 1]],
                [coord[ed[1] * 2], coord[ed[1] * 2 + 1]],
            );
            let d = dist(a, b).max(1e-4);
            let coef = lambda * 2.0 * (d - bone[li]) / d / edges.len() as f32;
            for t in 0..2 {
                let g = coef * (a[t] - b[t]);
                gc[ed[0] * 2 + t] += g;
                gc[ed[1] * 2 + t] -= g;
            }
        }
        let (graw, gelin) = if use_hg {
            let ge2 = hg_edge_to_node_backward(&cycles, &signs, &scale, &gc, edges.len(), 2, 2);
            let (ge, gl) = linear_backward(&elin, &e, &ge2);
            let _ = &e2;
            let gconv = hg_node_to_edge_backward(&cycles, &signs, &scale, &ge, J, 2, 2);
            let graw: Vec<f32> = gc.iter().zip(&gconv).map(|(&a, &b)| a + b).collect();
            (graw, Some(gl))
        } else {
            (gc, None)
        };
        let gheat = soft_argmax_backward(&sa, &graw, J, tau);
        let (gresp, ghead) = conv2d_backward(&head, &resp, hs, &gheat);
        let (_gx, gblk) = sc_block_backward(&b1, &img, s1, &bcache, &gresp);
        adam_step(&mut b1.conv.w, &gblk.conv.w, &mut abw, 0.01);
        adam_step(&mut b1.conv.b, &gblk.conv.b, &mut abb, 0.01);
        adam_step(&mut b1.filt, &gblk.filt, &mut abf, 0.01);
        adam_step(&mut head.w, &ghead.w, &mut ahw, 0.01);
        adam_step(&mut head.b, &ghead.b, &mut ahb, 0.01);
        if let Some(gl) = gelin {
            adam_step(&mut elin.w, &gl.w, &mut aew, 0.01);
            adam_step(&mut elin.b, &gl.b, &mut aeb, 0.01);
        }
    }

    let predict = |img: &[f32], elin: &LinearLayer| -> Vec<f32> {
        let (resp, _) = sc_block_forward(&b1, img, s1);
        let (heat, _, _) = conv2d_forward(&head, &resp, hs);
        let sa = soft_argmax_forward(&heat, J, G, tau);
        refine(&sa.coord, elin).0
    };
    let mut ergn = Rng(9999 + seed_base);
    // also report the neighbour-midpoint oracle for the coupler (the shared-elin ceiling).
    let (mut clean, mut occ_c, mut mid_ref) = (0.0f64, 0.0f64, 0.0f64);
    let n_eval = 80;
    for _ in 0..n_eval {
        let joints = sample_pose(&mut ergn);
        let pc = predict(&render(&joints, false), &elin);
        clean += dist([pc[4], pc[5]], joints[2]) as f64; // coupler, clean
        let po = predict(&render(&joints, true), &elin);
        occ_c += dist([po[4], po[5]], joints[2]) as f64; // coupler, occluded
        let mid = [
            (joints[1][0] + joints[3][0]) / 2.0,
            (joints[1][1] + joints[3][1]) / 2.0,
        ];
        mid_ref += dist(mid, joints[2]) as f64; // midpoint(B,D) vs true C — the shared-elin ceiling
    }
    clean /= n_eval as f64;
    occ_c /= n_eval as f64;
    mid_ref /= n_eval as f64;

    println!(
        "pose loop (hg={use_hg}): coupler err CLEAN {clean:.2}px  OCCLUDED {occ_c:.2}px  [midpoint-oracle {mid_ref:.2}px]"
    );
    let json = format!(
        "{{\n  \"hg\": {use_hg},\n  \"clean_coupler\": {clean:.4},\n  \"occ_coupler\": {occ_c:.4},\n  \"midpoint_oracle\": {mid_ref:.4}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
