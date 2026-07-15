//! SBSH Pose P3 — the task that actually EXERCISES the skeleton, no autograd. Two
//! COUPLED (parallel) arms share the pose (same θ₁,θ₂), so each arm is a fixed
//! translation of the other: `L = R + shoulder_offset`. Occluding an ENTIRE arm
//! removes its local evidence, but the other arm + the symmetry edge determine it
//! — a **redundant structural constraint** the local conv head cannot use (it
//! can't couple left↔right) but the skeleton `hg_conv` can. This is the missing
//! piece from P2, where the 2-link arm's middle joint was over-constrained and
//! left no structural-only gap.
//!
//! The recovery is a TRANSLATION, which the residual signed hg_conv expresses:
//! for the symmetry edge (Lᵢ,Rᵢ) with signs [+1,−1], `elin` can learn
//! `M=−I/scale²`, `bias=offset/scale²`, so the occluded joint's garbage estimate
//! cancels and it is set to `R + offset`.
//!
//! Stack as P2 (`ScBlock → conv head → soft_argmax → skeleton hg_conv`). A/B:
//! `--hg`. Eval occludes the whole LEFT arm and reports the left-hand error.
//!
//! Run: `cargo run --release --example pose_symmetry -- [--hg] [--seed=N] [out.json]`

use holonomy_learn::{
    adam_step, conv2d_backward, conv2d_forward, hg_edge_to_node_backward, hg_edge_to_node_forward,
    hg_node_to_edge_backward, hg_node_to_edge_forward, linear_backward, linear_forward,
    oriented_sobel_bank, sc_block_backward, sc_block_forward, soft_argmax_backward,
    soft_argmax_forward, AdamState, ConvLayer, ConvShape, DihedralGroup, LinearLayer, ScBlock,
};
use std::io::Write;

const G: usize = 32;
const GG: usize = G * G;
const J: usize = 6; // Lsh, Lel, Lha, Rsh, Rel, Rha
const K: usize = 6;
const L1: f32 = 7.0;
const L2: f32 = 7.0;
const LSH: [f32; 2] = [11.0, 9.0];
const RSH: [f32; 2] = [21.0, 9.0];

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

/// Two COUPLED arms (same θ₁,θ₂): each joint of the left arm is `RSH−LSH` off its
/// right twin. Returns [Lsh, Lel, Lha, Rsh, Rel, Rha].
fn sample_pose(rng: &mut Rng) -> [[f32; 2]; J] {
    let th1 = rng.range(-2.3, -0.8);
    let th2 = rng.range(-1.0, 1.0);
    let arm = |sh: [f32; 2]| -> ([f32; 2], [f32; 2]) {
        let el = [sh[0] + L1 * th1.cos(), sh[1] + L1 * th1.sin()];
        let ha = [
            el[0] + L2 * (th1 + th2).cos(),
            el[1] + L2 * (th1 + th2).sin(),
        ];
        (el, ha)
    };
    let (lel, lha) = arm(LSH);
    let (rel, rha) = arm(RSH);
    [LSH, lel, lha, RSH, rel, rha]
}

/// Which arm to occlude for masking: 0 = none, 1 = left (joints 1,2), 2 = right (4,5).
fn render(joints: &[[f32; 2]; J], occ_arm: u8) -> Vec<f32> {
    let mut img = vec![0.0f32; GG];
    let limbs = [(0usize, 1usize), (1, 2), (3, 4), (4, 5)];
    for &(a, b) in &limbs {
        let (p, q) = (joints[a], joints[b]);
        for s in 0..=64 {
            let t = s as f32 / 64.0;
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
    // occlude a whole arm: mask a box spanning its elbow+hand.
    if occ_arm != 0 {
        let (el, ha) = if occ_arm == 1 {
            (joints[1], joints[2])
        } else {
            (joints[4], joints[5])
        };
        let (x0, x1) = (
            (el[0].min(ha[0]) - 3.0).max(0.0) as usize,
            (el[0].max(ha[0]) + 3.0).min((G - 1) as f32) as usize,
        );
        let (y0, y1) = (
            (el[1].min(ha[1]) - 3.0).max(0.0) as usize,
            (el[1].max(ha[1]) + 3.0).min((G - 1) as f32) as usize,
        );
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
        .unwrap_or_else(|| "reports/figures/pose-symmetry.json".into());
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
    // skeleton: 4 bones + 2 SYMMETRY edges (Lel-Rel, Lha-Rha) — the redundancy.
    let edges = [[0usize, 1], [1, 2], [3, 4], [4, 5], [1, 4], [2, 5]];
    let off = dist(LSH, RSH); // symmetry-edge "bone" length = shoulder separation
    let bone = [L1, L2, L1, L2, off, off];
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
        // train with random whole-arm occlusion (either arm) so the skeleton
        // learns to reconstruct an occluded arm from its coupled twin.
        let occ_arm = if rng.f() < 0.6 {
            1 + (rng.f() * 2.0) as u8 % 2
        } else {
            0
        };
        let img = render(&joints, occ_arm);
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
    let (mut clean, mut occ_lha) = (0.0f64, 0.0f64);
    let n_eval = 80;
    for _ in 0..n_eval {
        let joints = sample_pose(&mut ergn);
        let pc = predict(&render(&joints, 0), &elin);
        clean += dist([pc[4], pc[5]], joints[2]) as f64; // left-hand, clean
        let po = predict(&render(&joints, 1), &elin); // occlude LEFT arm
        occ_lha += dist([po[4], po[5]], joints[2]) as f64; // left-hand, left-arm occluded
    }
    clean /= n_eval as f64;
    occ_lha /= n_eval as f64;

    println!(
        "pose symmetry (hg={use_hg}): left-hand err CLEAN {clean:.2}px  LEFT-ARM-OCCLUDED {occ_lha:.2}px"
    );
    let json = format!(
        "{{\n  \"hg\": {use_hg},\n  \"clean_lhand\": {clean:.4},\n  \"occ_lhand\": {occ_lha:.4}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
