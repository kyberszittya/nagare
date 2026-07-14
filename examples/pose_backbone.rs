//! SBSH Pose P2 — the P1 unblock: a **real spatial backbone** (the `ScBlock`
//! Neocognitron) + `soft_argmax` keypoint head + the skeleton `hg_conv`, no
//! autograd. P1 was confounded — coord channels + a single memorised pose made
//! localisation free and occlusion harmless, so the skeleton benefit was not
//! demonstrable. This fixes all three: (a) an `ScBlock` spatial backbone (no
//! coord channels), (b) RANDOMISED poses (nothing to memorise), (c) occlusion of
//! the **elbow** of a 2-link arm — a joint whose position is *recoverable from
//! structure* (pinned by the two visible endpoints + fixed bone lengths), so the
//! skeleton conv has something real to reconstruct.
//!
//! Stack: `img → ScBlock(1→K) → resp → conv head(K→J) → J heatmaps →
//! soft_argmax → coords → [skeleton hg_conv refine] → coords'`, trained on a
//! coordinate MSE + a bone-length limb-consistency loss, over random poses with
//! random-joint patch occlusion. A/B: `--hg` (skeleton on/off). Eval occludes the
//! ELBOW and reports its localisation error.
//!
//! Run: `cargo run --release --example pose_backbone -- [--hg] [--seed=N] [out.json]`

use holonomy_learn::{
    adam_step, conv2d_backward, conv2d_forward, hg_edge_to_node_backward, hg_edge_to_node_forward,
    hg_node_to_edge_backward, hg_node_to_edge_forward, linear_backward, linear_forward,
    sc_block_backward, sc_block_forward, soft_argmax_backward, soft_argmax_forward, AdamState,
    ConvLayer, ConvShape, DihedralGroup, LinearLayer, ScBlock,
};
use std::f32::consts::PI;
use std::io::Write;

const G: usize = 28;
const GG: usize = G * G;
const J: usize = 3; // shoulder, elbow, hand
const K: usize = 6; // backbone units
const L1: f32 = 7.0;
const L2: f32 = 7.0;
const SHOULDER: [f32; 2] = [14.0, 23.0];

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

/// A random 2-link arm pose: shoulder fixed, θ1 (upper) and θ2 (elbow bend) random.
fn sample_pose(rng: &mut Rng) -> [[f32; 2]; J] {
    let th1 = rng.range(-2.2, -0.9); // pointing generally upward (image y-down)
    let th2 = rng.range(-1.1, 1.1); // elbow bend
    let elbow = [SHOULDER[0] + L1 * th1.cos(), SHOULDER[1] + L1 * th1.sin()];
    let hand = [
        elbow[0] + L2 * (th1 + th2).cos(),
        elbow[1] + L2 * (th1 + th2).sin(),
    ];
    [SHOULDER, elbow, hand]
}

/// Render the arm (two limbs) to a G×G image, optionally masking a patch at `occ`.
fn render(joints: &[[f32; 2]; J], occ: Option<[f32; 2]>) -> Vec<f32> {
    let mut img = vec![0.0f32; GG];
    let limbs = [(0usize, 1usize), (1, 2)];
    for &(a, b) in &limbs {
        let (p, q) = (joints[a], joints[b]);
        for s in 0..=60 {
            let t = s as f32 / 60.0;
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
    if let Some(c) = occ {
        // large patch: removes the joint's LOCAL evidence and the inner limbs,
        // leaving only the endpoint tips → the middle joint must be triangulated
        // from structure (bone lengths + visible endpoints), not read locally.
        let rad = 6i32;
        let (cx, cy) = (c[0].round() as i32, c[1].round() as i32);
        for dy in -rad..=rad {
            for dx in -rad..=rad {
                let (cc, rr) = (cx + dx, cy + dy);
                if cc >= 0 && rr >= 0 && (cc as usize) < G && (rr as usize) < G {
                    img[rr as usize * G + cc as usize] = 0.0;
                }
            }
        }
    }
    img
}

fn oriented_conv_init(k: usize) -> Vec<f32> {
    let gx = [-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let gy = [-1.0f32, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let mut w = vec![0.0f32; 2 * k * 9];
    for u in 0..k {
        let phi = u as f32 * PI / k as f32;
        let (cp, sp) = (phi.cos(), phi.sin());
        for t in 0..9 {
            w[(2 * u) * 9 + t] = cp * gx[t] + sp * gy[t];
            w[(2 * u + 1) * 9 + t] = -sp * gx[t] + cp * gy[t];
        }
    }
    w
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}

fn main() {
    let use_hg = flag("--hg");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/pose-backbone.json".into());
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

    // backbone + head
    let mut b1 = ScBlock::new(1, K, 3, 3, group, tau, 11 + seed_base);
    b1.conv.w = oriented_conv_init(K);
    let mut head = ConvLayer::new(K, J, 5, 5, 21 + seed_base);
    // skeleton: chain 0-1-2, two bones as k=2 hyperedges
    let edges = [[0usize, 1], [1, 2]];
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
    let bone = [L1, L2];
    let mut elin = LinearLayer::new(2, 2, 31 + seed_base);

    let (mut abw, mut abb, mut abf) = (
        AdamState::new(b1.conv.w.len()),
        AdamState::new(b1.conv.b.len()),
        AdamState::new(b1.filt.len()),
    );
    let (mut ahw, mut ahb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));
    let (mut aew, mut aeb) = (AdamState::new(elin.w.len()), AdamState::new(elin.b.len()));

    // refine coords over the skeleton (residual hg-conv), returns (coord', e, e2).
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
    for _ in 0..1600 {
        let joints = sample_pose(&mut rng);
        // random-joint patch occlusion (train the skeleton to reconstruct any joint)
        let occ = if rng.f() < 0.6 {
            Some(joints[1 + (rng.f() * 2.0) as usize % 2])
        } else {
            None
        };
        let img = render(&joints, occ);
        let gt: Vec<f32> = joints.iter().flatten().copied().collect();
        // forward
        let (resp, bcache) = sc_block_forward(&b1, &img, s1);
        let (heat, _, _) = conv2d_forward(&head, &resp, hs);
        let sa = soft_argmax_forward(&heat, J, G, tau);
        let (coord, e, e2) = refine(&sa.coord, &elin);
        // coord MSE grad + bone-length limb grad → grad_coord' (J,2)
        let mut gc = vec![0.0f32; J * 2];
        for k in 0..J * 2 {
            gc[k] = 2.0 * (coord[k] - gt[k]) / (J * 2) as f32;
        }
        for (li, e) in edges.iter().enumerate() {
            let (a, b) = (
                [coord[e[0] * 2], coord[e[0] * 2 + 1]],
                [coord[e[1] * 2], coord[e[1] * 2 + 1]],
            );
            let d = dist(a, b).max(1e-4);
            let coef = lambda * 2.0 * (d - bone[li]) / d / edges.len() as f32;
            for t in 0..2 {
                let g = coef * (a[t] - b[t]);
                gc[e[0] * 2 + t] += g;
                gc[e[1] * 2 + t] -= g;
            }
        }
        // backward through skeleton refine → grad_raw_coord
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
        // soft_argmax backward → grad_heat → conv head backward → grad_resp → block backward
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

    // eval: per-joint error, clean vs ELBOW-occluded, over fresh poses.
    let predict = |img: &[f32], elin: &LinearLayer| -> Vec<f32> {
        let (resp, _) = sc_block_forward(&b1, img, s1);
        let (heat, _, _) = conv2d_forward(&head, &resp, hs);
        let sa = soft_argmax_forward(&heat, J, G, tau);
        refine(&sa.coord, elin).0
    };
    let mut ergn = Rng(9999 + seed_base);
    let (mut all_clean, mut clean_err, mut occ_elbow_err) = (0.0f64, 0.0f64, 0.0f64);
    let n_eval = 80;
    for _ in 0..n_eval {
        let joints = sample_pose(&mut ergn);
        let gt: Vec<[f32; 2]> = joints.to_vec();
        let pc = predict(&render(&joints, None), &elin);
        // mean localisation error over ALL joints (the P1-unblock number).
        all_clean += (0..J)
            .map(|k| dist([pc[k * 2], pc[k * 2 + 1]], gt[k]) as f64)
            .sum::<f64>()
            / J as f64;
        clean_err += dist([pc[2], pc[3]], gt[1]) as f64; // elbow, clean
        let po = predict(&render(&joints, Some(joints[1])), &elin);
        occ_elbow_err += dist([po[2], po[3]], gt[1]) as f64; // elbow, occluded
    }
    all_clean /= n_eval as f64;
    clean_err /= n_eval as f64;
    occ_elbow_err /= n_eval as f64;

    println!(
        "pose backbone (hg={use_hg}): all-joint MAE {all_clean:.2}px | elbow CLEAN {clean_err:.2}px  ELBOW-OCCLUDED {occ_elbow_err:.2}px"
    );
    let json = format!(
        "{{\n  \"hg\": {use_hg},\n  \"all_joint_mae\": {all_clean:.4},\n  \"clean_elbow_err\": {clean_err:.4},\n  \"occ_elbow_err\": {occ_elbow_err:.4}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
