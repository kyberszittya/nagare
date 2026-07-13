//! Demo — train the SBSH hypergraph-grid detector on synthetic oriented-rect
//! scenes and evaluate detection precision/recall, then dump one eval scene
//! (image + GT + detections + the quadtree grid) to JSON for rendering.
//!
//! The detector's "grid" is a dynamic quadtree (adaptive), not a fixed YOLO
//! grid. Only the linear head trains; features are fixed (descriptor+geometry).
//!
//! Run: `cargo run --release --example sbsh_detector_demo -- <out.json>`

use holonomy_learn::{gen_scene, leaf_center_object, DetectorConfig, NodePred, Obox, SbshDetector};
use rand::{Rng, SeedableRng};
use std::io::Write;

/// Greedy match: each GT is claimed by the nearest detection whose centre is
/// within `radius` px and not already used. Returns (tp, fp, fn).
fn match_dets(dets: &[NodePred], gt: &[Obox], radius: f32) -> (usize, usize, usize) {
    let mut used = vec![false; dets.len()];
    let mut tp = 0usize;
    for o in gt {
        let mut best = None;
        let mut best_d = radius;
        for (di, d) in dets.iter().enumerate() {
            if used[di] {
                continue;
            }
            let dist = ((d.bbox[0] - o.cx).powi(2) + (d.bbox[1] - o.cy).powi(2)).sqrt();
            if dist < best_d {
                best_d = dist;
                best = Some(di);
            }
        }
        if let Some(di) = best {
            used[di] = true;
            tp += 1;
        }
    }
    let fp = dets.len() - tp;
    let fn_ = gt.len() - tp;
    (tp, fp, fn_)
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reports/figures/sbsh-detections.json".to_string());
    let g = 64;
    let mut det = SbshDetector::new(DetectorConfig::new(g), 5);
    let mut rng = rand::rngs::StdRng::seed_from_u64(20);

    // Train on many random scenes (generalisation, not single-scene overfit).
    let n_train = 600;
    let mut curve = Vec::new();
    for it in 0..n_train {
        let k = rng.random_range(1..=4);
        let (img, objs) = gen_scene(g, k, &mut rng);
        let (ol, bl) = det.train_step(&img, &objs);
        if it % 10 == 0 {
            curve.push((it, ol, bl));
        }
    }

    // Diagnostics on a few held-out scenes: is objectness selective for
    // centre-region leaves, and do detection centres land near GT centres?
    {
        let mut drng = rand::rngs::StdRng::seed_from_u64(4242);
        let (mut pcen, mut pbg, mut ncen, mut nbg) = (0.0f32, 0.0f32, 0usize, 0usize);
        let (mut derr, mut dcnt) = (0.0f32, 0usize);
        for _ in 0..20 {
            let k = drng.random_range(1..=3);
            let (img, objs) = gen_scene(g, k, &mut drng);
            let (_qt, preds) = det.forward(&img);
            for p in &preds {
                if leaf_center_object(&objs, &p.anchor, det.cfg.center_frac).is_some() {
                    pcen += p.prob();
                    ncen += 1;
                } else {
                    pbg += p.prob();
                    nbg += 1;
                }
            }
            for d in det.detections(&img) {
                let nn = objs
                    .iter()
                    .map(|o| ((d.bbox[0] - o.cx).powi(2) + (d.bbox[1] - o.cy).powi(2)).sqrt())
                    .fold(f32::INFINITY, f32::min);
                derr += nn;
                dcnt += 1;
            }
        }
        println!(
            "DIAG: obj prob centre={:.3} (n={}) vs background={:.3} (n={}); mean det->GT centre dist={:.2}px (n={})",
            pcen / ncen.max(1) as f32, ncen, pbg / nbg.max(1) as f32, nbg, derr / dcnt.max(1) as f32, dcnt
        );
    }

    // Evaluate on held-out scenes.
    let mut eval_rng = rand::rngs::StdRng::seed_from_u64(9999);
    let (mut tp, mut fp, mut fn_) = (0usize, 0usize, 0usize);
    for _ in 0..80 {
        let k = eval_rng.random_range(1..=4);
        let (img, objs) = gen_scene(g, k, &mut eval_rng);
        let dets = det.detections(&img);
        let (a, b, c) = match_dets(&dets, &objs, 10.0);
        tp += a;
        fp += b;
        fn_ += c;
    }
    let precision = tp as f32 / (tp + fp).max(1) as f32;
    let recall = tp as f32 / (tp + fn_).max(1) as f32;
    let f1 = 2.0 * precision * recall / (precision + recall).max(1e-6);
    println!(
        "SBSH detector eval (80 held-out scenes): P={precision:.3} R={recall:.3} F1={f1:.3}  (tp={tp} fp={fp} fn={fn_})"
    );

    // Dump one representative eval scene for rendering (post-NMS detections).
    let (img, objs) = gen_scene(g, 3, &mut eval_rng);
    let (qt, _preds) = det.forward(&img);
    let dets = det.detections(&img);

    let box_json = |b: &[f32; 5]| {
        format!(
            "[{:.3},{:.3},{:.3},{:.3},{:.4}]",
            b[0], b[1], b[2], b[3], b[4]
        )
    };
    let obj_json = |o: &Obox| {
        format!(
            "[{:.3},{:.3},{:.3},{:.3},{:.4}]",
            o.cx, o.cy, o.w, o.h, o.theta
        )
    };
    let cells_json = qt
        .cells
        .iter()
        .map(|c| format!("[{},{},{},{}]", c[0], c[1], c[2], c[3]))
        .collect::<Vec<_>>()
        .join(",");
    let gt_json = objs.iter().map(obj_json).collect::<Vec<_>>().join(",");
    let det_json = dets
        .iter()
        .map(|p| format!("{{\"box\":{},\"prob\":{:.3}}}", box_json(&p.bbox), p.prob()))
        .collect::<Vec<_>>()
        .join(",");
    let curve_json = curve
        .iter()
        .map(|(it, ol, bl)| format!("[{it},{ol:.5},{bl:.5}]"))
        .collect::<Vec<_>>()
        .join(",");
    let img_json = img
        .iter()
        .map(|v| format!("{v:.1}"))
        .collect::<Vec<_>>()
        .join(",");

    let json = format!(
        "{{\n  \"g\": {g},\n  \"precision\": {precision:.4}, \"recall\": {recall:.4}, \"f1\": {f1:.4},\n  \"cells\": [{cells_json}],\n  \"gt\": [{gt_json}],\n  \"dets\": [{det_json}],\n  \"loss_curve\": [{curve_json}],\n  \"img\": [{img_json}]\n}}\n"
    );
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = std::fs::File::create(&out_path).expect("create json");
    f.write_all(json.as_bytes()).expect("write json");
    println!("wrote {out_path}");
}
