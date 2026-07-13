//! Integration — the full oriented head trains end to end under the closed-form
//! Gaussian-KLD loss, with NO autograd:
//! `features -> linear -> raw5 -> decode(anchor) -> box -> gaussian_kld(target)`.
//! Proves the composed hand-derived backward
//! (`features <- W <- decode <- gaussian_kld`) drives the per-node oriented
//! boxes onto distinct targets.

use holonomy_learn::{
    adam_step, decode_backward, decode_forward, gaussian_kld_backward, gaussian_kld_forward,
    linear_backward, linear_forward, AdamState, Anchor, LinearLayer,
};

#[test]
fn learns_oriented_boxes_through_kld() {
    let n_nodes = 4usize;
    let feat_dim = n_nodes; // one-hot features -> each node's raw5 is independently fittable
    let out_dim = 5usize;
    let tau = 1.0f32;

    // One-hot per-node features (n_nodes, feat_dim).
    let mut feats = vec![0.0f32; n_nodes * feat_dim];
    for k in 0..n_nodes {
        feats[k * feat_dim + k] = 1.0;
    }
    // Distinct anchors and distinct (reachable, positive-size) oriented targets.
    let anchors = [
        Anchor {
            cx: 4.0,
            cy: 4.0,
            w: 4.0,
            h: 2.0,
        },
        Anchor {
            cx: 12.0,
            cy: 5.0,
            w: 3.0,
            h: 3.0,
        },
        Anchor {
            cx: 6.0,
            cy: 11.0,
            w: 5.0,
            h: 2.0,
        },
        Anchor {
            cx: 13.0,
            cy: 12.0,
            w: 2.0,
            h: 4.0,
        },
    ];
    // All targets are ELONGATED (w != h). A square (w==h) target has an
    // isotropic Gaussian, so theta is unidentifiable and the KLD loss is
    // genuinely theta-insensitive there (the well-known square-object
    // degeneracy of Gaussian OBB losses) — we keep every target non-square so
    // all five parameters are recoverable.
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

    let l0 = loss_of(&w);
    // Adam: the bounded KLD form has heterogeneous per-parameter curvature
    // (centre offset vs log-size vs angle), so a per-parameter adaptive step
    // converges where plain SGD stalls near the floor.
    let lr = 0.05f32;
    let inv_n = 1.0 / n_nodes as f32;
    let mut st_w = AdamState::new(w.w.len());
    let mut st_b = AdamState::new(w.b.len());
    for _ in 0..3000 {
        let raw = linear_forward(&w, &feats);
        let mut grad_raw = vec![0.0f32; n_nodes * out_dim];
        for k in 0..n_nodes {
            let r: [f32; 5] = raw[k * out_dim..k * out_dim + out_dim].try_into().unwrap();
            let boxp = decode_forward(&r, &anchors[k]);
            let (_, cache) = gaussian_kld_forward(&boxp, &targets[k], tau);
            let mut gbox = gaussian_kld_backward(&cache, &boxp, &targets[k]);
            for v in &mut gbox {
                *v *= inv_n; // mean over nodes
            }
            let graw = decode_backward(&gbox, &boxp);
            grad_raw[k * out_dim..k * out_dim + out_dim].copy_from_slice(&graw);
        }
        let (_gx, gw) = linear_backward(&w, &feats, &grad_raw);
        adam_step(&mut w.w, &gw.w, &mut st_w, lr);
        adam_step(&mut w.b, &gw.b, &mut st_b, lr);
    }
    let l1 = loss_of(&w);

    assert!(l1 < 0.05 * l0, "KLD head did not converge: {l0} -> {l1}");
    assert!(l1 < 0.02, "KLD head did not reach the floor: {l1}");
    // The recovered box matches the target's GAUSSIAN, which is the invariant the
    // loss actually constrains. The (w,h,theta) parameterisation carries two
    // inherent ambiguities the Gaussian model deliberately quotients out (a
    // documented feature, cf. Yang et al.): pi-periodicity and the axis swap
    // (w,h,theta) == (h,w,theta +/- pi/2). So assert on the centre and the
    // covariance Sigma = R diag((w/2)^2,(h/2)^2) R^T, not on raw w/h/theta.
    let sigma = |b: &[f32; 5]| -> [f32; 3] {
        let (a, bb) = ((b[2] * 0.5).powi(2), (b[3] * 0.5).powi(2));
        let (s, c) = b[4].sin_cos();
        [
            a * c * c + bb * s * s,
            (a - bb) * c * s,
            a * s * s + bb * c * c,
        ]
    };
    for (b, t) in boxes_of(&w).iter().zip(&targets) {
        assert!(
            (b[0] - t[0]).abs() < 0.2 && (b[1] - t[1]).abs() < 0.2,
            "centre {b:?} vs {t:?}"
        );
        let (sb, st) = (sigma(b), sigma(t));
        for k in 0..3 {
            assert!(
                (sb[k] - st[k]).abs() < 0.15,
                "Sigma[{k}] {} vs {} ({b:?} vs {t:?})",
                sb[k],
                st[k]
            );
        }
    }
}
