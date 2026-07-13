//! Demo — train the oriented head end to end under the closed-form Gaussian-KLD
//! loss (no autograd) and dump target vs learned oriented boxes to JSON for
//! rendering (`scripts/dev/render_oriented_boxes.py`).
//!
//! Mirrors the `tests/oriented_head_learn.rs` integration path; this binary
//! exists only to emit the plottable artifact required by CLAUDE.md Section 9.
//!
//! Run: `cargo run --release --example oriented_head_demo -- <out.json>`

use holonomy_learn::{
    adam_step, decode_backward, decode_forward, gaussian_kld_backward, gaussian_kld_forward,
    linear_backward, linear_forward, AdamState, Anchor, LinearLayer,
};
use std::io::Write;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reports/figures/oriented-head-boxes.json".to_string());

    let (n_nodes, feat_dim, out_dim, tau) = (4usize, 4usize, 5usize, 1.0f32);
    let mut feats = vec![0.0f32; n_nodes * feat_dim];
    for k in 0..n_nodes {
        feats[k * feat_dim + k] = 1.0;
    }
    let anchors = [
        Anchor { cx: 4.0, cy: 4.0, w: 4.0, h: 2.0 },
        Anchor { cx: 12.0, cy: 5.0, w: 3.0, h: 3.0 },
        Anchor { cx: 6.0, cy: 11.0, w: 5.0, h: 2.0 },
        Anchor { cx: 13.0, cy: 12.0, w: 2.0, h: 4.0 },
    ];
    let targets = [
        [5.0f32, 3.5, 5.0, 1.5, 0.3],
        [11.0f32, 6.0, 2.2, 4.0, -0.5],
        [7.0f32, 10.0, 3.5, 2.5, 0.9],
        [12.5f32, 13.0, 4.0, 2.0, 1.2],
    ];

    let mut w = LinearLayer::new(feat_dim, out_dim, 7);
    let boxes_of = |w: &LinearLayer| -> Vec<[f32; 5]> {
        let raw = linear_forward(w, &feats);
        (0..n_nodes)
            .map(|k| {
                let r: [f32; 5] = raw[k * out_dim..k * out_dim + out_dim].try_into().unwrap();
                decode_forward(&r, &anchors[k])
            })
            .collect()
    };
    let loss_of = |w: &LinearLayer| -> f32 {
        boxes_of(w)
            .iter()
            .zip(&targets)
            .map(|(b, t)| gaussian_kld_forward(b, t, tau).0)
            .sum::<f32>()
            / n_nodes as f32
    };

    let init_boxes = boxes_of(&w);
    let l0 = loss_of(&w);
    let lr = 0.05f32;
    let inv_n = 1.0 / n_nodes as f32;
    let (mut st_w, mut st_b) = (AdamState::new(w.w.len()), AdamState::new(w.b.len()));
    let mut curve = Vec::new();
    for it in 0..3000 {
        let raw = linear_forward(&w, &feats);
        let mut grad_raw = vec![0.0f32; n_nodes * out_dim];
        for k in 0..n_nodes {
            let r: [f32; 5] = raw[k * out_dim..k * out_dim + out_dim].try_into().unwrap();
            let boxp = decode_forward(&r, &anchors[k]);
            let (_, cache) = gaussian_kld_forward(&boxp, &targets[k], tau);
            let mut gbox = gaussian_kld_backward(&cache, &boxp, &targets[k]);
            for v in &mut gbox {
                *v *= inv_n;
            }
            let graw = decode_backward(&gbox, &boxp);
            grad_raw[k * out_dim..k * out_dim + out_dim].copy_from_slice(&graw);
        }
        let (_gx, gw) = linear_backward(&w, &feats, &grad_raw);
        adam_step(&mut w.w, &gw.w, &mut st_w, lr);
        adam_step(&mut w.b, &gw.b, &mut st_b, lr);
        if it % 50 == 0 {
            curve.push((it, loss_of(&w)));
        }
    }
    let learned = boxes_of(&w);
    let l1 = loss_of(&w);
    println!("KLD head demo: loss {l0:.4} -> {l1:.6} over 3000 Adam steps");

    // Minimal hand-rolled JSON (no serde dep).
    let box_json = |b: &[f32; 5]| {
        format!("[{:.4},{:.4},{:.4},{:.4},{:.4}]", b[0], b[1], b[2], b[3], b[4])
    };
    let arr = |bs: &[[f32; 5]]| {
        bs.iter().map(box_json).collect::<Vec<_>>().join(",")
    };
    let curve_json = curve
        .iter()
        .map(|(it, l)| format!("[{it},{l:.6}]"))
        .collect::<Vec<_>>()
        .join(",");
    let json = format!(
        "{{\n  \"anchors\": [{}],\n  \"targets\": [{}],\n  \"init\": [{}],\n  \"learned\": [{}],\n  \"loss_curve\": [{}],\n  \"l0\": {:.6}, \"l1\": {:.6}\n}}\n",
        anchors
            .iter()
            .map(|a| format!("[{:.4},{:.4},{:.4},{:.4}]", a.cx, a.cy, a.w, a.h))
            .collect::<Vec<_>>()
            .join(","),
        arr(&targets),
        arr(&init_boxes),
        arr(&learned),
        curve_json,
        l0,
        l1,
    );
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut f = std::fs::File::create(&out_path).expect("create json");
    f.write_all(json.as_bytes()).expect("write json");
    println!("wrote {out_path}");
}
