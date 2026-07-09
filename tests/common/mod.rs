//! Shared scaffolding for the HSiKAN mixed-arity entropy tests: HSiKAN as a fixed
//! feature extractor + an entropy-gated / constant linear readout, a mixed-arity toy,
//! and a linear-teacher label generator. Used by `hsikan_layer.rs` (single-seed) and
//! `hsikan_multiseed.rs` (median/IQR). Reuses `entropy2`/`softmax2`/`cross_entropy` (§6.1).
#![allow(dead_code)] // each test binary compiles this module and uses a subset.

pub mod vision;

use holonomy_learn::{
    cross_entropy, entropy2, hsikan_forward, softmax2, HsikanConfig, HsikanEdges, HsikanParams,
};
use rand::{Rng, SeedableRng};

/// One uniform-arity slice of the hypergraph (shares the model's parameters).
pub struct EdgeGroup {
    pub arity: usize,
    pub n_edges: usize,
    pub vertices: Vec<u32>,
    pub signs: Vec<i8>,
}

/// Fixed HSiKAN parameters used as a mixed-arity feature extractor.
pub struct FeatureExtractor {
    pub d: usize,
    s: usize,
    grid: usize,
    k: usize,
    x: Vec<f32>,
    inner_coef: Vec<f32>,
    outer_coef: Vec<f32>,
    gate_w: Vec<f32>,
    gate_b: Vec<f32>,
}

impl FeatureExtractor {
    pub fn new(n_nodes: usize, d: usize, s: usize, grid: usize, k: usize, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let mut fill = |n: usize, scale: f32| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
                .collect()
        };
        let branch = s * d * k;
        Self {
            d,
            s,
            grid,
            k,
            x: fill(n_nodes * d, 0.4),
            inner_coef: fill(branch, 0.3),
            outer_coef: fill(branch, 0.3),
            gate_w: fill(d * d, 0.2),
            gate_b: vec![-1.0; d],
        }
    }

    fn params(&self) -> HsikanParams<'_> {
        HsikanParams {
            inner_coef: &self.inner_coef,
            outer_coef: &self.outer_coef,
            gate_w: &self.gate_w,
            gate_b: &self.gate_b,
        }
    }

    /// Per-edge embeddings across all groups, flat `(T, d)` in group order.
    pub fn features(&self, groups: &[EdgeGroup]) -> Vec<f32> {
        let mut feats = Vec::new();
        for g in groups {
            let cfg =
                HsikanConfig::new(g.n_edges, g.arity, self.d, self.s, self.grid, self.k, true);
            let edges = HsikanEdges {
                vertices: &g.vertices,
                signs: &g.signs,
            };
            let (h_e, _) = hsikan_forward(self.params(), &self.x, edges, cfg);
            feats.extend_from_slice(&h_e);
        }
        feats
    }
}

/// Two-class linear readout trained by the entropy-gated / constant local delta rule.
pub struct Readout {
    d: usize,
    w: Vec<f32>,
    b: [f32; 2],
    entropy_gate: bool,
}

impl Readout {
    pub fn new(d: usize, seed: u64, entropy_gate: bool) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let w = (0..d * 2)
            .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.01)
            .collect();
        Self {
            d,
            w,
            b: [0.0, 0.0],
            entropy_gate,
        }
    }

    fn logits(&self, feat: &[f32]) -> (f32, f32) {
        let mut l = self.b;
        for (i, &v) in feat.iter().enumerate() {
            l[0] += v * self.w[2 * i];
            l[1] += v * self.w[2 * i + 1];
        }
        (l[0], l[1])
    }

    /// One pass of the delta rule over all samples; gate = 0.25+H(p) or 1.
    pub fn train_epoch(&mut self, feats: &[f32], labels: &[f32], lr: f32) {
        for (t, &label) in labels.iter().enumerate() {
            let feat = &feats[t * self.d..(t + 1) * self.d];
            let (l0, l1) = self.logits(feat);
            let (p0, p1) = softmax2(l0, l1);
            let gate = if self.entropy_gate {
                0.25 + entropy2(p0, p1)
            } else {
                1.0
            };
            let delta = [f32::from(label == 0.0) - p0, f32::from(label == 1.0) - p1];
            for (i, &v) in feat.iter().enumerate() {
                self.w[2 * i] += lr * gate * v * delta[0];
                self.w[2 * i + 1] += lr * gate * v * delta[1];
            }
            self.b[0] += lr * gate * delta[0];
            self.b[1] += lr * gate * delta[1];
        }
    }

    /// Cross-entropy loss and accuracy over the samples (reuses the crate metric).
    pub fn eval(&self, feats: &[f32], labels: &[f32]) -> (f32, f32) {
        let n = labels.len();
        let mut logits = vec![0.0f32; n * 2];
        let y: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
        for t in 0..n {
            let (l0, l1) = self.logits(&feats[t * self.d..(t + 1) * self.d]);
            logits[2 * t] = l0;
            logits[2 * t + 1] = l1;
        }
        let ce = cross_entropy(&logits, &y);
        (ce.loss, ce.acc)
    }
}

/// A fixed mixed-arity toy: 10 nodes, six arity-3 and six arity-4 edges.
pub fn toy() -> Vec<EdgeGroup> {
    vec![
        EdgeGroup {
            arity: 3,
            n_edges: 6,
            vertices: vec![0, 1, 2, 2, 3, 4, 4, 5, 6, 1, 6, 8, 3, 7, 9, 0, 5, 8],
            signs: vec![
                1, -1, 1, -1, 1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1, 1, 1,
            ],
        },
        EdgeGroup {
            arity: 4,
            n_edges: 6,
            vertices: vec![
                0, 1, 2, 3, 1, 2, 4, 5, 3, 4, 6, 7, 2, 5, 8, 9, 0, 6, 7, 9, 1, 3, 5, 8,
            ],
            signs: vec![
                1, -1, 1, -1, 1, 1, -1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1,
            ],
        },
    ]
}

/// Balanced, linearly-separable labels in the FIXED feature space: a random linear
/// teacher split at its median. Isolates the learning rule, not feature capacity.
pub fn teacher_labels(feats: &[f32], d: usize, seed: u64) -> Vec<f32> {
    let n = feats.len() / d;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let w_true: Vec<f32> = (0..d).map(|_| rng.random::<f32>() * 2.0 - 1.0).collect();
    let scores: Vec<f32> = (0..n)
        .map(|t| {
            feats[t * d..(t + 1) * d]
                .iter()
                .zip(&w_true)
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[n / 2];
    scores.iter().map(|&s| f32::from(s < med)).collect()
}

/// Train a readout in one gate mode; returns (initial loss, final loss, final acc).
pub fn train_mode(
    feats: &[f32],
    labels: &[f32],
    d: usize,
    entropy_gate: bool,
    seed: u64,
    epochs: usize,
) -> (f32, f32, f32) {
    let mut readout = Readout::new(d, seed, entropy_gate);
    let (initial, _) = readout.eval(feats, labels);
    for _ in 0..epochs {
        readout.train_epoch(feats, labels, 0.1);
    }
    let (final_loss, acc) = readout.eval(feats, labels);
    (initial, final_loss, acc)
}
