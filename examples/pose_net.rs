//! SBSH Pose P1 — the forward pose net on the signed-hg_conv path, no autograd:
//! `image feature (incl. coord channels) → per-joint query → [signed hg_conv over
//! the skeleton] → correlation heatmap → soft_argmax → (x,y)`, trained with a
//! coordinate MSE + a bone-length **limb-consistency** loss (the skeleton
//! structure). The `--hg` toggle turns the signed skeleton conv on/off for an
//! A/B: does refining joint queries over the skeleton help when some joints are
//! occluded (their image evidence removed)?
//!
//! Localization uses coord channels (CoordConv): local line appearance is the
//! same at every joint, so the query must read position from the coord channels
//! — the same "local features can't localize" lesson as the detector.
//!
//! Run: `cargo run --release --example pose_net -- [--hg] [--occlude] [out.json]`

use holonomy_learn::{
    adam_step, hg_edge_to_node_backward, hg_edge_to_node_forward, hg_node_to_edge_backward,
    hg_node_to_edge_forward, linear_backward, linear_forward, soft_argmax_backward,
    soft_argmax_forward, AdamState, LinearLayer,
};
use std::io::Write;

const G: usize = 32;
const C: usize = 5; // feature channels: intensity, gx, gy, xnorm, ynorm

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

/// Render a stick figure (limbs as lines, value 1) to a G×G image.
fn render(joints: &[[f32; 2]], edges: &[[usize; 2]], occlude: Option<usize>) -> Vec<f32> {
    let mut img = vec![0.0f32; G * G];
    for (li, e) in edges.iter().enumerate() {
        if Some(li) == occlude {
            continue; // drop this limb's evidence (occlusion)
        }
        let (a, b) = (joints[e[0]], joints[e[1]]);
        let steps = 60;
        for s in 0..=steps {
            let t = s as f32 / steps as f32;
            let x = a[0] + (b[0] - a[0]) * t;
            let y = a[1] + (b[1] - a[1]) * t;
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
    img
}

/// Per-pixel feature (P, C): [intensity, gx, gy, xnorm, ynorm]; 0..2 standardised.
fn features(img: &[f32]) -> Vec<f32> {
    let at = |r: i32, c: i32| -> f32 {
        let r = r.clamp(0, G as i32 - 1) as usize;
        let c = c.clamp(0, G as i32 - 1) as usize;
        img[r * G + c]
    };
    let mut f = vec![0.0f32; G * G * C];
    for r in 0..G {
        for c in 0..G {
            let p = r * G + c;
            let gx = at(r as i32, c as i32 + 1) - at(r as i32, c as i32 - 1);
            let gy = at(r as i32 + 1, c as i32) - at(r as i32 - 1, c as i32);
            f[p * C] = img[p];
            f[p * C + 1] = gx;
            f[p * C + 2] = gy;
            f[p * C + 3] = c as f32 / (G - 1) as f32;
            f[p * C + 4] = r as f32 / (G - 1) as f32;
        }
    }
    // standardise channels 0..2
    for ch in 0..3 {
        let mu: f32 = (0..G * G).map(|p| f[p * C + ch]).sum::<f32>() / (G * G) as f32;
        let sd: f32 = ((0..G * G)
            .map(|p| (f[p * C + ch] - mu).powi(2))
            .sum::<f32>()
            / (G * G) as f32)
            .sqrt()
            + 1e-6;
        for p in 0..G * G {
            f[p * C + ch] = (f[p * C + ch] - mu) / sd;
        }
    }
    f
}

fn main() {
    let use_hg = flag("--hg");
    let occlude = flag("--occlude");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/pose-net.json".into());
    let tau = 0.5f32;

    let gt: Vec<[f32; 2]> = vec![
        [16.0, 4.0],
        [16.0, 12.0],
        [8.0, 9.0],
        [24.0, 9.0],
        [11.0, 26.0],
        [21.0, 26.0],
    ];
    let edges = [[0usize, 1], [1, 2], [1, 3], [1, 4], [1, 5]];
    let j = gt.len();
    let l = edges.len();
    let occ_limb = if occlude { Some(3) } else { None }; // limb 1-4 → joint 4 occluded
    let img = render(&gt, &edges, occ_limb);
    let feat = features(&img);
    let gt_flat: Vec<f32> = gt.iter().flatten().copied().collect();
    let bone: Vec<f32> = edges.iter().map(|e| dist(gt[e[0]], gt[e[1]])).collect();

    // Skeleton signed hypergraph: each limb = a k=2 hyperedge; sign +1/-1 per end.
    let cycles: Vec<u32> = edges
        .iter()
        .flat_map(|e| [e[0] as u32, e[1] as u32])
        .collect();
    let signs: Vec<f32> = edges.iter().flat_map(|_| [1.0f32, -1.0]).collect();
    let mut deg = vec![0.0f32; j];
    for e in &edges {
        deg[e[0]] += 1.0;
        deg[e[1]] += 1.0;
    }
    let scale: Vec<f32> = deg.iter().map(|&d| (d.max(1.0)).powf(-0.5)).collect();

    // Params: per-joint query q (J,C); the hg-conv edge transform.
    let mut q = vec![0.0f32; j * C];
    {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        for v in q.iter_mut() {
            *v = (rng.random::<f32>() - 0.5) * 0.2;
        }
    }
    let mut elin = LinearLayer::new(C, C, 11);
    let mut sq = AdamState::new(q.len());
    let mut se_w = AdamState::new(elin.w.len());
    let mut se_b = AdamState::new(elin.b.len());

    // q' = q + hg_edge_to_node(elin(hg_node_to_edge(q))) if --hg, else q.
    let refine = |q: &[f32], elin: &LinearLayer| -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        if !use_hg {
            return (q.to_vec(), Vec::new(), Vec::new());
        }
        let e = hg_node_to_edge_forward(q, &cycles, &signs, &scale, l, 2, C);
        let e2 = linear_forward(elin, &e);
        let r = hg_edge_to_node_forward(&e2, &cycles, &signs, &scale, j, 2, C);
        let qp: Vec<f32> = q.iter().zip(&r).map(|(&a, &b)| a + b).collect();
        (qp, e, e2)
    };
    // heat[jt,p] = Σ_c q'[jt,c]·f[p,c]
    let heatmap = |qp: &[f32]| -> Vec<f32> {
        let mut h = vec![0.0f32; j * G * G];
        for jt in 0..j {
            for p in 0..G * G {
                let mut s = 0.0f32;
                for c in 0..C {
                    s += qp[jt * C + c] * feat[p * C + c];
                }
                h[jt * G * G + p] = s;
            }
        }
        h
    };
    let coords_of = |q: &[f32], elin: &LinearLayer| -> Vec<f32> {
        let (qp, _, _) = refine(q, elin);
        soft_argmax_forward(&heatmap(&qp), j, G, tau).coord
    };

    let mse = |c: &[f32]| {
        c.iter()
            .zip(&gt_flat)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / (j * 2) as f32
    };
    let l0 = mse(&coords_of(&q, &elin));
    let lambda = 0.05f32;
    let mut curve = Vec::new();

    for it in 0..2500 {
        let (qp, e, e2) = refine(&q, &elin);
        let sa = soft_argmax_forward(&heatmap(&qp), j, G, tau);
        // coordinate-MSE grad + bone-length limb grad → grad_coord (J,2).
        let mut gc = vec![0.0f32; j * 2];
        for k in 0..j * 2 {
            gc[k] = 2.0 * (sa.coord[k] - gt_flat[k]) / (j * 2) as f32;
        }
        for (li, e) in edges.iter().enumerate() {
            let (a, b) = (
                [sa.coord[e[0] * 2], sa.coord[e[0] * 2 + 1]],
                [sa.coord[e[1] * 2], sa.coord[e[1] * 2 + 1]],
            );
            let d = dist(a, b).max(1e-4);
            let coef = lambda * 2.0 * (d - bone[li]) / d / l as f32;
            for t in 0..2 {
                let g = coef * (a[t] - b[t]);
                gc[e[0] * 2 + t] += g;
                gc[e[1] * 2 + t] -= g;
            }
        }
        // soft_argmax backward → grad_heat → grad_q'.
        let gheat = soft_argmax_backward(&sa, &gc, j, tau);
        let mut gqp = vec![0.0f32; j * C];
        for jt in 0..j {
            for c in 0..C {
                let mut s = 0.0f32;
                for p in 0..G * G {
                    s += gheat[jt * G * G + p] * feat[p * C + c];
                }
                gqp[jt * C + c] = s;
            }
        }
        // through the residual hg-conv (if on): grad_q = gqp + conv-path; grad_elin.
        let (gq, gelin) = if use_hg {
            // r = hg_edge_to_node(e2); q' = q + r, so grad_r = gqp.
            let ge2 = hg_edge_to_node_backward(&cycles, &signs, &scale, &gqp, l, 2, C);
            let (ge, gl) = linear_backward(&elin, &e, &ge2);
            let _ = &e2;
            let gq_conv = hg_node_to_edge_backward(&cycles, &signs, &scale, &ge, j, 2, C);
            let gq: Vec<f32> = gqp.iter().zip(&gq_conv).map(|(&a, &b)| a + b).collect();
            (gq, Some(gl))
        } else {
            (gqp, None)
        };
        adam_step(&mut q, &gq, &mut sq, 0.05);
        if let Some(gl) = gelin {
            adam_step(&mut elin.w, &gl.w, &mut se_w, 0.05);
            adam_step(&mut elin.b, &gl.b, &mut se_b, 0.05);
        }
        if it % 100 == 0 {
            curve.push((it, mse(&sa.coord)));
        }
    }
    let pred = coords_of(&q, &elin);
    let l1 = mse(&pred);
    let max_err = (0..j)
        .map(|k| dist([pred[k * 2], pred[k * 2 + 1]], gt[k]))
        .fold(0.0f32, f32::max);
    println!(
        "pose net (hg={use_hg} occlude={occlude}): MSE {l0:.3} -> {l1:.4}; max joint err {max_err:.3} px"
    );

    let arr2 = |v: &[[f32; 2]]| {
        v.iter()
            .map(|p| format!("[{:.2},{:.2}]", p[0], p[1]))
            .collect::<Vec<_>>()
            .join(",")
    };
    let pred_pairs: Vec<[f32; 2]> = (0..j).map(|k| [pred[k * 2], pred[k * 2 + 1]]).collect();
    let json = format!(
        "{{\n  \"g\": {G}, \"hg\": {use_hg}, \"occlude\": {occlude}, \"occ_limb\": {},\n  \"gt\": [{}],\n  \"pred\": [{}],\n  \"edges\": [{}],\n  \"img\": [{}],\n  \"loss_curve\": [{}],\n  \"max_err\": {max_err:.4}\n}}\n",
        occ_limb.map(|x| x as i32).unwrap_or(-1),
        arr2(&gt),
        arr2(&pred_pairs),
        edges.iter().map(|e| format!("[{},{}]", e[0], e[1])).collect::<Vec<_>>().join(","),
        img.iter().map(|v| format!("{v:.0}")).collect::<Vec<_>>().join(","),
        curve.iter().map(|(it, m)| format!("[{it},{m:.4}]")).collect::<Vec<_>>().join(","),
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
    println!("wrote {out_path}");
}

fn dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}
