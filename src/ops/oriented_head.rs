//! Oriented-bbox head glue: anchor-relative decode (differentiable) + a
//! structural node->object assignment (no backward — the `cpml_tier`
//! discipline). The head composes existing ops: pooled node features
//! (`node_pool` (+) `oriented_descriptor`) -> `linear`/`kan` -> a per-node raw
//! 5-vector -> [`decode_forward`] -> an oriented box -> [`gaussian_kld`] loss.
//!
//! Only two things are new here and both are thin:
//! - [`decode_forward`]/[`decode_backward`]: map a raw 5-vector to an oriented
//!   box relative to the node's quadtree cell (anchor). `w,h` go through
//!   `exp(.)` so they are strictly positive (the `gaussian_kld` precondition).
//! - [`assign_nodes`]: a deterministic, gradient-free map from ground-truth
//!   objects to the leaf node whose cell contains each object's centre.

use crate::ops::quadtree::Quadtree;

/// The per-node anchor derived from its quadtree cell `[y0, x0, y1, x1)`:
/// centre `(cx, cy)` and reference size `(w, h)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Anchor {
    pub cx: f32,
    pub cy: f32,
    pub w: f32,
    pub h: f32,
}

/// Anchor of a leaf cell `[y0, x0, y1, x1)` (pixel coords, `x`=col, `y`=row).
pub fn anchor_of_cell(cell: &[usize; 4]) -> Anchor {
    let [y0, x0, y1, x1] = *cell;
    Anchor {
        cx: 0.5 * (x0 + x1) as f32,
        cy: 0.5 * (y0 + y1) as f32,
        w: (x1 - x0) as f32,
        h: (y1 - y0) as f32,
    }
}

/// Decode a raw 5-vector into an oriented box `[cx, cy, w, h, theta]` relative
/// to `anchor`:
/// `cx = a.cx + r0`, `cy = a.cy + r1`, `w = a.w*exp(r2)`, `h = a.h*exp(r3)`,
/// `theta = r4`. `w,h > 0` by construction.
pub fn decode_forward(raw: &[f32; 5], anchor: &Anchor) -> [f32; 5] {
    [
        anchor.cx + raw[0],
        anchor.cy + raw[1],
        anchor.w * raw[2].exp(),
        anchor.h * raw[3].exp(),
        raw[4],
    ]
}

/// Backward of [`decode_forward`]. Given `dL/dbox` and the decoded `box`,
/// returns `dL/draw`. Only `box.w`, `box.h` (indices 2,3) are needed for the
/// `exp` chain (`dw/dr2 = a.w*exp(r2) = box.w`).
pub fn decode_backward(grad_box: &[f32; 5], boxp: &[f32; 5]) -> [f32; 5] {
    [
        grad_box[0],
        grad_box[1],
        grad_box[2] * boxp[2],
        grad_box[3] * boxp[3],
        grad_box[4],
    ]
}

/// Deterministically assign ground-truth `objects` (`[cx, cy, w, h, theta]`, in
/// pixel coords) to leaf nodes: each object goes to the leaf whose cell contains
/// its centre (via the per-pixel `assign`). One object per node — the first
/// object claiming a node wins; later collisions and out-of-image centres are
/// skipped (structural, no backward).
///
/// # Preconditions
/// `qt.assign.len() == g*g`.
/// # Postconditions
/// Returned node indices are distinct and each `< qt.cells.len()`.
pub fn assign_nodes(objects: &[[f32; 5]], qt: &Quadtree, g: usize) -> Vec<(usize, [f32; 5])> {
    debug_assert_eq!(qt.assign.len(), g * g, "assign must be g*g");
    let mut taken = vec![false; qt.cells.len()];
    let mut out = Vec::new();
    for obj in objects {
        let col = obj[0].floor();
        let row = obj[1].floor();
        if col < 0.0 || row < 0.0 || col >= g as f32 || row >= g as f32 {
            continue; // centre outside the image
        }
        let node = qt.assign[row as usize * g + col as usize] as usize;
        if taken[node] {
            continue; // one object per node (deterministic: first wins)
        }
        taken[node] = true;
        out.push((node, *obj));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::quadtree::{quadtree_build, QuadtreeConfig};

    #[test]
    fn anchor_from_cell() {
        let a = anchor_of_cell(&[2, 4, 6, 10]); // y0,x0,y1,x1
        assert_eq!(
            a,
            Anchor {
                cx: 7.0,
                cy: 4.0,
                w: 6.0,
                h: 4.0
            }
        );
    }

    #[test]
    fn decode_backward_matches_fd() {
        let anchor = Anchor {
            cx: 3.0,
            cy: 5.0,
            w: 4.0,
            h: 2.0,
        };
        let raw = [0.3f32, -0.2, 0.4, -0.1, 0.6];
        let boxp = decode_forward(&raw, &anchor);
        // Arbitrary upstream gradient on the box.
        let grad_box = [0.7f32, -0.5, 0.9, 0.3, -0.4];
        let grad_raw = decode_backward(&grad_box, &boxp);
        // Scalar L = <grad_box, box> so dL/dbox = grad_box exactly.
        let dot = |b: &[f32; 5]| -> f32 { b.iter().zip(grad_box).map(|(&x, g)| x * g).sum() };
        let eps = 1e-3f32;
        for i in 0..5 {
            let mut rp = raw;
            rp[i] += eps;
            let mut rm = raw;
            rm[i] -= eps;
            let num = (dot(&decode_forward(&rp, &anchor)) - dot(&decode_forward(&rm, &anchor)))
                / (2.0 * eps);
            assert!(
                (grad_raw[i] - num).abs() < 1e-2,
                "grad_raw[{i}] {} vs {num}",
                grad_raw[i]
            );
        }
    }

    #[test]
    fn assign_maps_center_to_leaf_and_dedups() {
        // A tree that splits (high energy in one quadrant) so >1 leaf exists.
        let g = 8;
        let mut energy = vec![0.0f32; g * g];
        for i in 0..4 {
            for j in 0..4 {
                energy[i * g + j] = 1.0; // hot top-left -> subdivides
            }
        }
        let qt = quadtree_build(
            &energy,
            &QuadtreeConfig {
                g,
                max_depth: 3,
                min_side: 1,
                thresh: 0.1,
            },
        );
        // Object at pixel (col=1,row=1) must map to that pixel's leaf.
        let objs = [
            [1.0f32, 1.0, 2.0, 2.0, 0.0],
            [1.2f32, 1.1, 2.0, 2.0, 0.0],  // same leaf -> deduped
            [-1.0f32, 3.0, 1.0, 1.0, 0.0], // outside -> skipped
        ];
        let got = assign_nodes(&objs, &qt, g);
        assert_eq!(got.len(), 1, "dedup + out-of-image cull");
        assert_eq!(got[0].0, qt.assign[g + 1] as usize);
        // node indices distinct and in range.
        assert!(got.iter().all(|(n, _)| *n < qt.cells.len()));
    }
}
