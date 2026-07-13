//! SBSH Pose P0 smoke — regress a stick-figure's joint coordinates end to end
//! through the differentiable soft-argmax head, with NO autograd:
//! `per-joint feature → linear → heatmap → soft_argmax → (x,y)`, MSE to the GT
//! joints. Proves the composed backward
//! (`features ← W ← soft_argmax ← MSE`) drives the predicted joints onto the
//! skeleton. Dumps GT + predicted joints (+ skeleton edges) to JSON for
//! `scripts/dev/render_pose.py`.
//!
//! Run: `cargo run --release --example pose_smoke -- [out.json]`

use holonomy_learn::{
    adam_step, linear_backward, linear_forward, mse_backward, mse_forward, soft_argmax_backward,
    soft_argmax_forward, AdamState, LinearLayer,
};
use std::io::Write;

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reports/figures/pose-smoke.json".into());
    let g = 24usize;
    let tau = 0.6f32;
    // A 6-joint stick figure (col=x, row=y) on the g×g grid + skeleton edges.
    let gt: Vec<[f32; 2]> = vec![
        [12.0, 3.0],  // 0 head
        [12.0, 9.0],  // 1 neck/chest
        [6.0, 7.0],   // 2 left hand
        [18.0, 7.0],  // 3 right hand
        [8.0, 19.0],  // 4 left foot
        [16.0, 19.0], // 5 right foot
    ];
    let edges = [[0, 1], [1, 2], [1, 3], [1, 4], [1, 5]];
    let n = gt.len();
    let np = g * g;
    // One-hot per-joint features → each joint's heatmap is independently fittable.
    let mut feats = vec![0.0f32; n * n];
    for j in 0..n {
        feats[j * n + j] = 1.0;
    }
    let gt_flat: Vec<f32> = gt.iter().flatten().copied().collect();

    let mut head = LinearLayer::new(n, np, 7); // feat → heatmap
    let (mut sw, mut sb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));
    let coords_of = |head: &LinearLayer| -> Vec<f32> {
        let heat = linear_forward(head, &feats);
        soft_argmax_forward(&heat, n, g, tau).coord
    };
    let l0 = mse_forward(&coords_of(&head), &gt_flat);

    let mut curve = Vec::new();
    for it in 0..1500 {
        let heat = linear_forward(&head, &feats);
        let sa = soft_argmax_forward(&heat, n, g, tau);
        let gcoord = mse_backward(&sa.coord, &gt_flat);
        let gheat = soft_argmax_backward(&sa, &gcoord, n, tau);
        let (_gx, gh) = linear_backward(&head, &feats, &gheat);
        adam_step(&mut head.w, &gh.w, &mut sw, 0.05);
        adam_step(&mut head.b, &gh.b, &mut sb, 0.05);
        if it % 50 == 0 {
            curve.push((it, mse_forward(&sa.coord, &gt_flat)));
        }
    }
    let pred = coords_of(&head);
    let l1 = mse_forward(&pred, &gt_flat);
    let max_err = gt
        .iter()
        .enumerate()
        .map(|(j, p)| ((pred[j * 2] - p[0]).powi(2) + (pred[j * 2 + 1] - p[1]).powi(2)).sqrt())
        .fold(0.0f32, f32::max);
    println!("pose smoke: MSE {l0:.3} -> {l1:.5}; max joint error {max_err:.3} px (g={g})");

    let arr2 = |v: &[[f32; 2]]| {
        v.iter()
            .map(|p| format!("[{:.2},{:.2}]", p[0], p[1]))
            .collect::<Vec<_>>()
            .join(",")
    };
    let pred_pairs: Vec<[f32; 2]> = (0..n).map(|j| [pred[j * 2], pred[j * 2 + 1]]).collect();
    let json = format!(
        "{{\n  \"g\": {g},\n  \"gt\": [{}],\n  \"pred\": [{}],\n  \"edges\": [{}],\n  \"loss_curve\": [{}],\n  \"max_err\": {max_err:.4}\n}}\n",
        arr2(&gt),
        arr2(&pred_pairs),
        edges.iter().map(|e| format!("[{},{}]", e[0], e[1])).collect::<Vec<_>>().join(","),
        curve.iter().map(|(it, l)| format!("[{it},{l:.5}]")).collect::<Vec<_>>().join(","),
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
