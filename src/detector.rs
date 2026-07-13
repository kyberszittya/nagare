//! SBSH detector — a YOLO alternative whose detection grid is a **dynamic
//! quadtree** instead of a fixed `S x S` grid. Leaves subdivide only where
//! there is content (gradient energy), so each leaf is an adaptive "grid cell"
//! (the SBSH spatial-hypergraph node). This module composes the FD-clean ops
//! into a forward detector + an objectness/box training step. It introduces
//! **no new closed-form backward** — the trainable part is a single `linear`
//! head over fixed descriptor+geometry features, so the backward is exactly the
//! already-verified `oriented_head_learn` path plus a BCE branch.
//!
//! Pipeline (per image):
//! `grad-energy -> quadtree_build -> per-leaf [oriented_descriptor(crop) (+)
//! node_pool means (+) geometry] -> linear head -> (objectness logit, box raw5)
//! -> decode_forward(anchor)`.
//!
//! Losses: `bce_with_logits`(objectness, leaf-on-object) + `lambda`
//! `gaussian_kld`(box, assigned GT). Structural steps (quadtree, assignment)
//! carry no gradient (the `cpml_tier` discipline).

use crate::ops::gaussian_kld::{gaussian_kld_backward, gaussian_kld_forward, Obox};
use crate::ops::linear::{linear_backward, linear_forward, LinearLayer};
use crate::ops::loss::{bce_with_logits_backward, bce_with_logits_forward};
use crate::ops::oriented_descriptor::{oriented_descriptor_forward, oriented_dim};
use crate::ops::oriented_head::{anchor_of_cell, decode_backward, decode_forward, Anchor};
use crate::ops::quadtree::{node_pool_forward, quadtree_build, Quadtree, QuadtreeConfig};
use crate::ops::{adam::adam_step, adam::AdamState};

/// Detector configuration. `qt` drives the adaptive grid; `crop` is the square
/// descriptor window each leaf is resampled into; `bins` the orientation bins.
#[derive(Clone)]
pub struct DetectorConfig {
    pub g: usize,
    pub crop: usize,
    pub bins: usize,
    pub lambda_box: f32,
    pub obj_thresh: f32,
    /// Reference box size (px) for the anchor. The leaf supplies the *centre*;
    /// the *size* comes from this scale prior (not the grid-cell size, which is
    /// the tree resolution, orders smaller than an object) so the initial box is
    /// near the target and the bounded-KLD gradient does not vanish.
    pub anchor_scale: f32,
    /// Centre-sampling radius as a fraction of an object's short side: a leaf is
    /// positive iff its anchor centre is within `center_frac * min(w,h)` of an
    /// object centre.
    pub center_frac: f32,
    /// Centre-distance NMS radius (px): detections within this of a kept,
    /// higher-scoring detection are suppressed.
    pub nms_radius: f32,
    pub qt: QuadtreeConfig,
}

impl DetectorConfig {
    /// A sensible default for a `g x g` synthetic scene.
    pub fn new(g: usize) -> Self {
        DetectorConfig {
            g,
            crop: 12,
            bins: 8,
            lambda_box: 1.0,
            obj_thresh: 0.7,
            anchor_scale: 14.0,
            center_frac: 0.5,
            nms_radius: 12.0,
            qt: QuadtreeConfig {
                g,
                max_depth: 5,
                min_side: 2,
                thresh: 0.04,
            },
        }
    }
}

/// One node's prediction: which leaf, its anchor, the objectness **logit**, and
/// the decoded oriented box `[cx, cy, w, h, theta]`.
#[derive(Clone, Copy, Debug)]
pub struct NodePred {
    pub node: usize,
    pub anchor: Anchor,
    pub objectness: f32,
    pub bbox: [f32; 5],
}

impl NodePred {
    /// Objectness probability `sigmoid(logit)`.
    pub fn prob(&self) -> f32 {
        if self.objectness >= 0.0 {
            1.0 / (1.0 + (-self.objectness).exp())
        } else {
            let e = self.objectness.exp();
            e / (1.0 + e)
        }
    }
}

/// The detector: a single learned `linear` head over fixed per-leaf features.
pub struct SbshDetector {
    pub head: LinearLayer,
    pub cfg: DetectorConfig,
    st_w: AdamState,
    st_b: AdamState,
}

impl SbshDetector {
    /// Feature dim: `oriented_dim(bins)` (edge/orientation descriptor) + 2
    /// (node_pool gradient means) + 5 (geometry: log-area, mean-energy,
    /// **mean-intensity (DC)**, normalised centroid x,y). The mean-intensity
    /// term is what lets objectness separate a flat object *interior* leaf
    /// (bright, zero-gradient) from flat background — the edge descriptor alone
    /// cannot, since both are isotropic.
    fn feat_dim_for(cfg: &DetectorConfig) -> usize {
        oriented_dim(cfg.bins) + 2 + 5
    }
    pub fn feat_dim(&self) -> usize {
        Self::feat_dim_for(&self.cfg)
    }

    /// New detector with a Glorot-init head (`feat_dim -> 6`: 1 objectness + 5 box).
    pub fn new(cfg: DetectorConfig, seed: u64) -> Self {
        let fd = Self::feat_dim_for(&cfg);
        let head = LinearLayer::new(fd, 6, seed);
        let (nw, nb) = (head.w.len(), head.b.len());
        SbshDetector {
            head,
            cfg,
            st_w: AdamState::new(nw),
            st_b: AdamState::new(nb),
        }
    }

    /// Build the quadtree and extract the fixed per-leaf feature matrix
    /// `(n_leaves, feat_dim)` plus each leaf's anchor. No gradient flows here.
    pub fn node_features(&self, img: &[f32]) -> (Quadtree, Vec<f32>, Vec<Anchor>) {
        let g = self.cfg.g;
        let field = grad_field(img, g);
        let energy = energy_map(&field, g);
        let qt = quadtree_build(&energy, &self.cfg.qt);
        let n = qt.cells.len();
        let (crop, bins) = (self.cfg.crop, self.cfg.bins);

        // Batched descriptor field: n leaves, each a `crop x crop` (gx,gy) field.
        let mut dfield = vec![0.0f32; n * crop * crop * 2];
        for (ci, cell) in qt.cells.iter().enumerate() {
            let cimg = crop_resample(img, g, cell, crop);
            for r in 0..crop {
                for c in 0..crop {
                    let (gx, gy) = cdiff(&cimg, crop, r, c);
                    let base = (ci * crop * crop + r * crop + c) * 2;
                    dfield[base] = gx;
                    dfield[base + 1] = gy;
                }
            }
        }
        let desc = oriented_descriptor_forward(&dfield, n, crop, bins);
        let dd = oriented_dim(bins);
        let (npool, _counts) = node_pool_forward(&field, &qt.assign, n, 2);

        let fd = self.feat_dim();
        let mut feats = vec![0.0f32; n * fd];
        let mut anchors = Vec::with_capacity(n);
        for ci in 0..n {
            let cell = qt.cells[ci];
            let a0 = anchor_of_cell(&cell);
            // Centre from the leaf; size from the scale prior (see anchor_scale).
            let a = Anchor {
                cx: a0.cx,
                cy: a0.cy,
                w: self.cfg.anchor_scale,
                h: self.cfg.anchor_scale,
            };
            let off = ci * fd;
            feats[off..off + dd].copy_from_slice(&desc.feat[ci * dd..(ci + 1) * dd]);
            feats[off + dd] = npool[ci * 2];
            feats[off + dd + 1] = npool[ci * 2 + 1];
            feats[off + dd + 2] = (a.w * a.h + 1.0).ln();
            feats[off + dd + 3] = cell_mean_energy(&energy, g, &cell);
            feats[off + dd + 4] = cell_mean_intensity(img, g, &cell);
            feats[off + dd + 5] = a.cx / g as f32;
            feats[off + dd + 6] = a.cy / g as f32;
            anchors.push(a);
        }
        (qt, feats, anchors)
    }

    /// Forward: per-leaf objectness logit + decoded oriented box.
    pub fn forward(&self, img: &[f32]) -> (Quadtree, Vec<NodePred>) {
        let (qt, feats, anchors) = self.node_features(img);
        let out = linear_forward(&self.head, &feats);
        let preds = (0..qt.cells.len())
            .map(|ci| {
                let o = &out[ci * 6..ci * 6 + 6];
                let raw5: [f32; 5] = o[1..6].try_into().unwrap();
                NodePred {
                    node: ci,
                    anchor: anchors[ci],
                    objectness: o[0],
                    bbox: decode_forward(&raw5, &anchors[ci]),
                }
            })
            .collect();
        (qt, preds)
    }

    /// Detections above the objectness threshold, after centre-distance NMS
    /// (many centre-region leaves fire per object; NMS keeps the top-scoring
    /// one within `nms_radius`).
    pub fn detections(&self, img: &[f32]) -> Vec<NodePred> {
        let (_, preds) = self.forward(img);
        let mut cand: Vec<NodePred> = preds
            .into_iter()
            .filter(|p| p.prob() >= self.cfg.obj_thresh)
            .collect();
        cand.sort_by(|a, b| {
            b.prob()
                .partial_cmp(&a.prob())
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        let mut kept: Vec<NodePred> = Vec::new();
        for p in cand {
            let clash = kept.iter().any(|k| {
                let d = ((p.bbox[0] - k.bbox[0]).powi(2) + (p.bbox[1] - k.bbox[1]).powi(2)).sqrt();
                d < self.cfg.nms_radius
            });
            if !clash {
                kept.push(p);
            }
        }
        kept
    }

    /// One Adam training step on a single image with GT objects. Returns
    /// `(objectness_loss, box_loss)`. Objectness BCE is positive-weighted to
    /// counter the background-leaf majority.
    pub fn train_step(&mut self, img: &[f32], objs: &[Obox]) -> (f32, f32) {
        let (qt, feats, anchors) = self.node_features(img);
        let n = qt.cells.len();
        let out = linear_forward(&self.head, &feats);

        // Center-sampling (FCOS-style): a leaf is a positive iff its anchor
        // centre sits in the CENTRE region of some object; that SAME object is
        // its box target. Label == supervision (per the metric-integrity rule),
        // and box regression is only asked of leaves whose anchor is near the
        // object centre — where the offset is small and locally learnable.
        let tgt: Vec<Option<[f32; 5]>> = anchors
            .iter()
            .map(|a| leaf_center_object(objs, a, self.cfg.center_frac).map(|o| o.to_array()))
            .collect();
        let obj_t: Vec<f32> = tgt
            .iter()
            .map(|t| if t.is_some() { 1.0 } else { 0.0 })
            .collect();
        let npos = tgt.iter().filter(|t| t.is_some()).count().max(1);
        let w_pos = ((n - npos) as f32 / npos as f32).clamp(1.0, 20.0);

        let logits: Vec<f32> = (0..n).map(|i| out[i * 6]).collect();
        let obj_grad = bce_with_logits_backward(&logits, &obj_t);
        let obj_loss = bce_with_logits_forward(&logits, &obj_t);
        let inv_obj = self.cfg.lambda_box / npos as f32;

        // Assemble the head-output gradient (n, 6): col 0 objectness, cols 1..6 box.
        let mut grad_out = vec![0.0f32; n * 6];
        let mut box_loss = 0.0f32;
        for i in 0..n {
            let w = if obj_t[i] > 0.5 { w_pos } else { 1.0 };
            grad_out[i * 6] = obj_grad[i] * w;
            if let Some(target) = tgt[i] {
                let raw5: [f32; 5] = out[i * 6 + 1..i * 6 + 6].try_into().unwrap();
                let boxp = decode_forward(&raw5, &anchors[i]);
                let (l, cache) = gaussian_kld_forward(&boxp, &target, 1.0);
                box_loss += l;
                let mut gbox = gaussian_kld_backward(&cache, &boxp, &target);
                for v in &mut gbox {
                    *v *= inv_obj;
                }
                let graw = decode_backward(&gbox, &boxp);
                grad_out[i * 6 + 1..i * 6 + 6].copy_from_slice(&graw);
            }
        }
        let n_obj = npos;
        let (_gx, gw) = linear_backward(&self.head, &feats, &grad_out);
        adam_step(&mut self.head.w, &gw.w, &mut self.st_w, 0.02);
        adam_step(&mut self.head.b, &gw.b, &mut self.st_b, 0.02);
        (obj_loss, box_loss / n_obj as f32)
    }
}

// ---- fixed feature helpers (no gradient) ----

/// Per-pixel `(gx,gy)` central-difference field, interleaved `[g*g*2]`.
fn grad_field(img: &[f32], g: usize) -> Vec<f32> {
    let at = |i: i32, j: i32| -> f32 {
        let i = i.clamp(0, g as i32 - 1) as usize;
        let j = j.clamp(0, g as i32 - 1) as usize;
        img[i * g + j]
    };
    let mut f = vec![0.0f32; g * g * 2];
    for i in 0..g {
        for j in 0..g {
            let (ii, jj) = (i as i32, j as i32);
            f[(i * g + j) * 2] = at(ii, jj + 1) - at(ii, jj - 1);
            f[(i * g + j) * 2 + 1] = at(ii + 1, jj) - at(ii - 1, jj);
        }
    }
    f
}

/// Per-pixel gradient magnitude `[g*g]` — the quadtree split energy.
fn energy_map(field: &[f32], g: usize) -> Vec<f32> {
    (0..g * g)
        .map(|p| (field[p * 2].powi(2) + field[p * 2 + 1].powi(2)).sqrt())
        .collect()
}

/// Bilinear sample of `img` at fractional `(fy,fx)` with edge clamping.
fn bilinear(img: &[f32], g: usize, fy: f32, fx: f32) -> f32 {
    let y = fy.clamp(0.0, g as f32 - 1.0);
    let x = fx.clamp(0.0, g as f32 - 1.0);
    let (y0, x0) = (y.floor() as usize, x.floor() as usize);
    let (y1, x1) = ((y0 + 1).min(g - 1), (x0 + 1).min(g - 1));
    let (ty, tx) = (y - y0 as f32, x - x0 as f32);
    let a = img[y0 * g + x0];
    let b = img[y0 * g + x1];
    let c = img[y1 * g + x0];
    let d = img[y1 * g + x1];
    let top = a + (b - a) * tx;
    let bot = c + (d - c) * tx;
    top + (bot - top) * ty
}

/// Resample a leaf cell `[y0,x0,y1,x1)` into a `crop x crop` image (bilinear).
fn crop_resample(img: &[f32], g: usize, cell: &[usize; 4], crop: usize) -> Vec<f32> {
    let [y0, x0, y1, x1] = *cell;
    let (ch, cw) = ((y1 - y0) as f32, (x1 - x0) as f32);
    let mut out = vec![0.0f32; crop * crop];
    for r in 0..crop {
        for c in 0..crop {
            let fy = y0 as f32 + (r as f32 + 0.5) / crop as f32 * ch - 0.5;
            let fx = x0 as f32 + (c as f32 + 0.5) / crop as f32 * cw - 0.5;
            out[r * crop + c] = bilinear(img, g, fy, fx);
        }
    }
    out
}

/// Central difference on a `crop x crop` image (edge clamped).
fn cdiff(cimg: &[f32], crop: usize, r: usize, c: usize) -> (f32, f32) {
    let at = |a: i32, b: i32| -> f32 {
        let a = a.clamp(0, crop as i32 - 1) as usize;
        let b = b.clamp(0, crop as i32 - 1) as usize;
        cimg[a * crop + b]
    };
    let (ri, ci) = (r as i32, c as i32);
    (
        at(ri, ci + 1) - at(ri, ci - 1),
        at(ri + 1, ci) - at(ri - 1, ci),
    )
}

/// Mean energy over a leaf cell.
fn cell_mean_energy(energy: &[f32], g: usize, cell: &[usize; 4]) -> f32 {
    cell_mean(energy, g, cell)
}

/// Mean image intensity (DC / brightness) over a leaf cell — the term that
/// separates a flat object interior (bright) from flat background (dark).
fn cell_mean_intensity(img: &[f32], g: usize, cell: &[usize; 4]) -> f32 {
    cell_mean(img, g, cell)
}

/// Mean of a `g*g` scalar field over a leaf cell `[y0,x0,y1,x1)`.
fn cell_mean(field: &[f32], g: usize, cell: &[usize; 4]) -> f32 {
    let [y0, x0, y1, x1] = *cell;
    let mut e = 0.0f32;
    for i in y0..y1 {
        for j in x0..x1 {
            e += field[i * g + j];
        }
    }
    e / (((y1 - y0) * (x1 - x0)).max(1)) as f32
}

// ---- scene / ground truth ----

/// Is pixel `(row,col)` inside the oriented box?
pub fn obox_contains(o: &Obox, row: usize, col: usize) -> bool {
    let (dx, dy) = (col as f32 - o.cx, row as f32 - o.cy);
    let (c, s) = (o.theta.cos(), o.theta.sin());
    let rx = dx * c + dy * s;
    let ry = -dx * s + dy * c;
    rx.abs() <= o.w * 0.5 && ry.abs() <= o.h * 0.5
}

/// Synthetic scene: `k` filled oriented rects (value +1) on a flat background
/// (-1), plus the ground-truth boxes. Shared by the tests, the demo, and eval.
pub fn gen_scene<R: rand::Rng>(g: usize, k: usize, rng: &mut R) -> (Vec<f32>, Vec<Obox>) {
    let mut img = vec![-1.0f32; g * g];
    let mut objs = Vec::with_capacity(k);
    for _ in 0..k {
        let w: f32 = rng.random_range(12.0..24.0);
        let h: f32 = rng.random_range(8.0..16.0);
        let m = w.max(h) * 0.6;
        let o = Obox {
            cx: rng.random_range(m..(g as f32 - m)),
            cy: rng.random_range(m..(g as f32 - m)),
            w,
            h,
            theta: rng.random_range(0.0..std::f32::consts::PI),
        };
        for i in 0..g {
            for j in 0..g {
                if obox_contains(&o, i, j) {
                    img[i * g + j] = 1.0;
                }
            }
        }
        objs.push(o);
    }
    (img, objs)
}

/// Centre-sampling target: the object whose centre is nearest the leaf's anchor
/// centre AND within `frac * min(w,h)` of it, or `None`. Positive leaves are
/// the object-centre neighbourhood — where a local head can regress the box.
pub fn leaf_center_object(objs: &[Obox], anchor: &Anchor, frac: f32) -> Option<Obox> {
    let mut best: Option<(Obox, f32)> = None;
    for o in objs {
        let d = ((anchor.cx - o.cx).powi(2) + (anchor.cy - o.cy).powi(2)).sqrt();
        let radius = frac * o.w.min(o.h);
        if d < radius && best.is_none_or(|(_, bd)| d < bd) {
            best = Some((*o, d));
        }
    }
    best.map(|(o, _)| o)
}

/// Fraction of a leaf cell's pixels inside any object (a coverage helper).
pub fn leaf_on_object(objs: &[Obox], cell: &[usize; 4], _g: usize) -> f32 {
    let [y0, x0, y1, x1] = *cell;
    let (mut hit, mut tot) = (0usize, 0usize);
    for i in y0..y1 {
        for j in x0..x1 {
            tot += 1;
            if objs.iter().any(|o| obox_contains(o, i, j)) {
                hit += 1;
            }
        }
    }
    if tot == 0 {
        0.0
    } else {
        hit as f32 / tot as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn forward_shapes_and_positive_boxes() {
        let cfg = DetectorConfig::new(64);
        let det = SbshDetector::new(cfg, 1);
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let (img, _objs) = gen_scene(64, 2, &mut rng);
        let (qt, preds) = det.forward(&img);
        assert_eq!(preds.len(), qt.cells.len());
        assert!(!preds.is_empty());
        for p in &preds {
            assert!(p.bbox[2] > 0.0 && p.bbox[3] > 0.0, "positive box size");
            assert!(p.prob() >= 0.0 && p.prob() <= 1.0);
            assert!(p.node < qt.cells.len());
        }
    }

    #[test]
    fn feature_dim_matches() {
        let cfg = DetectorConfig::new(64);
        let det = SbshDetector::new(cfg, 2);
        let mut rng = rand::rngs::StdRng::seed_from_u64(3);
        let (img, _) = gen_scene(64, 1, &mut rng);
        let (qt, feats, anchors) = det.node_features(&img);
        assert_eq!(feats.len(), qt.cells.len() * det.feat_dim());
        assert_eq!(anchors.len(), qt.cells.len());
    }

    #[test]
    fn leaf_on_object_detects_coverage() {
        let o = Obox {
            cx: 10.0,
            cy: 10.0,
            w: 8.0,
            h: 8.0,
            theta: 0.0,
        };
        // A cell fully inside the object → fraction 1.
        assert!(leaf_on_object(&[o], &[8, 8, 12, 12], 64) > 0.99);
        // A cell far away → 0.
        assert!(leaf_on_object(&[o], &[40, 40, 44, 44], 64) < 0.01);
    }

    #[test]
    fn overfits_one_scene() {
        // On a FIXED scene, training must (a) separate centre-region objectness
        // from background and (b) reduce the box KLD (centre+orientation are
        // learnable; absolute size is NOT locally observable for a filled rect,
        // so we assert the loss DROPS, not that it reaches the floor).
        let g = 64;
        let cfg = DetectorConfig::new(g);
        let mut det = SbshDetector::new(cfg, 5);
        let mut rng = rand::rngs::StdRng::seed_from_u64(11);
        let (img, objs) = gen_scene(g, 3, &mut rng);

        let (obj0, box0) = det.train_step(&img, &objs);
        for _ in 0..400 {
            det.train_step(&img, &objs);
        }
        let (obj1, box1) = det.train_step(&img, &objs);
        assert!(
            obj1 < 0.5 * obj0,
            "objectness loss did not drop: {obj0} -> {obj1}"
        );
        assert!(box1 < box0, "box loss did not drop: {box0} -> {box1}");

        // Centre-region leaves must separate from the rest in probability.
        let (_qt, preds) = det.forward(&img);
        let mut pos = Vec::new();
        let mut neg = Vec::new();
        for p in &preds {
            if leaf_center_object(&objs, &p.anchor, det.cfg.center_frac).is_some() {
                pos.push(p.prob());
            } else {
                neg.push(p.prob());
            }
        }
        let mean = |v: &[f32]| {
            if v.is_empty() {
                0.0
            } else {
                v.iter().sum::<f32>() / v.len() as f32
            }
        };
        assert!(
            mean(&pos) > mean(&neg) + 0.3,
            "no obj/bg separation: pos {} vs neg {}",
            mean(&pos),
            mean(&neg)
        );
    }
}
