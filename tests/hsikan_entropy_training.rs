//! Phase 1c′ (wiring) — both entropy mechanisms in ONE HSiKAN training loop.
//!
//! The node-embedding matrix `x` (the matrix `EntropyRegulariser` canonically targets)
//! is trained on a mixed-arity signed-hypergraph task by:
//!   (1) an **entropy-gated local delta-rule readout** — Nagare's learning substrate
//!       (`gate = 0.25 + H(softmax)`), and the task error backprops through it and
//!       `hsikan_backward` into `x`; and
//!   (2) the **spectral-entropy regulariser** (`SpectralEntropyReg::step`) whose `∇_x`
//!       is summed into the `x` update.
//! HSiKAN spline/gate params are fixed (feature extractor); `x` and the readout learn.
//!
//! The gate: with the regulariser ON, (a) task loss still falls, and (b) `H_norm(x)`
//! is pulled toward the target τ — demonstrably closer than the reg-OFF control. That
//! is the regulariser doing its job wired into real training, not in isolation.

use holonomy_learn::{
    cross_entropy, entropy2, hsikan_backward, hsikan_forward, softmax2, HsikanCache, HsikanConfig,
    HsikanEdges, HsikanParams, SpectralEntropyConfig, SpectralEntropyReg,
};
use rand::{Rng, SeedableRng};

struct EdgeGroup {
    arity: usize,
    n_edges: usize,
    vertices: Vec<u32>,
    signs: Vec<i8>,
}

impl EdgeGroup {
    fn cfg(&self, d: usize, s: usize, grid: usize, k: usize) -> HsikanConfig {
        HsikanConfig::new(self.n_edges, self.arity, d, s, grid, k, true)
    }
    fn edges(&self) -> HsikanEdges<'_> {
        HsikanEdges {
            vertices: &self.vertices,
            signs: &self.signs,
        }
    }
}

struct Model {
    d: usize,
    s: usize,
    grid: usize,
    k: usize,
    n_nodes: usize,
    x: Vec<f32>,
    inner_coef: Vec<f32>,
    outer_coef: Vec<f32>,
    gate_w: Vec<f32>,
    gate_b: Vec<f32>,
    rw: Vec<f32>, // readout weights, flat (d, 2)
    rb: [f32; 2],
}

impl Model {
    fn new(n_nodes: usize, d: usize, s: usize, grid: usize, k: usize, seed: u64) -> Self {
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
            n_nodes,
            x: fill(n_nodes * d, 0.4),
            inner_coef: fill(branch, 0.3),
            outer_coef: fill(branch, 0.3),
            gate_w: fill(d * d, 0.2),
            gate_b: vec![-1.0; d],
            rw: fill(d * 2, 0.05),
            rb: [0.0, 0.0],
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

    fn forward(&self, groups: &[EdgeGroup]) -> (Vec<f32>, Vec<HsikanCache>) {
        let mut feats = Vec::new();
        let mut caches = Vec::with_capacity(groups.len());
        for g in groups {
            let (h_e, cache) = hsikan_forward(
                self.params(),
                &self.x,
                g.edges(),
                g.cfg(self.d, self.s, self.grid, self.k),
            );
            feats.extend_from_slice(&h_e);
            caches.push(cache);
        }
        (feats, caches)
    }

    fn logits_row(&self, feat: &[f32]) -> (f32, f32) {
        let mut l = self.rb;
        for (i, &v) in feat.iter().enumerate() {
            l[0] += v * self.rw[2 * i];
            l[1] += v * self.rw[2 * i + 1];
        }
        (l[0], l[1])
    }

    /// One step: entropy-gated readout update + task-backprop into `x`, optionally
    /// plus the spectral-entropy regulariser gradient. Returns `(task_loss, H_norm)`.
    fn step(
        &mut self,
        groups: &[EdgeGroup],
        labels: &[f32],
        lr: f32,
        reg: &mut SpectralEntropyReg,
        reg_on: bool,
    ) -> (f32, f32) {
        let (feats, caches) = self.forward(groups);
        let n = labels.len();

        // Per-sample: entropy-gated readout update + CE grad w.r.t. edge features.
        let mut grad_he = vec![0.0f32; feats.len()];
        for t in 0..n {
            let feat = &feats[t * self.d..(t + 1) * self.d];
            let (l0, l1) = self.logits_row(feat);
            let (p0, p1) = softmax2(l0, l1);
            let gate = 0.25 + entropy2(p0, p1);
            let (y0, y1) = (f32::from(labels[t] == 0.0), f32::from(labels[t] == 1.0));
            let dce = [p0 - y0, p1 - y1]; // ∂CE/∂logit
            for i in 0..self.d {
                // feature gradient uses pre-update weights.
                grad_he[t * self.d + i] = self.rw[2 * i] * dce[0] + self.rw[2 * i + 1] * dce[1];
                // entropy-gated local delta rule on the readout (Δw = lr·gate·φ·(y−p)).
                self.rw[2 * i] -= lr * gate * feat[i] * dce[0];
                self.rw[2 * i + 1] -= lr * gate * feat[i] * dce[1];
            }
            self.rb[0] -= lr * gate * dce[0];
            self.rb[1] -= lr * gate * dce[1];
        }

        // Backprop grad_he into x via hsikan_backward (shared params, both arities).
        let mut grad_x = vec![0.0f32; self.x.len()];
        let mut row = 0usize;
        for (g, cache) in groups.iter().zip(&caches) {
            let span = g.n_edges * self.d;
            let gh = &grad_he[row * self.d..row * self.d + span];
            row += g.n_edges;
            let bw = hsikan_backward(
                self.params(),
                g.edges(),
                cache,
                gh,
                g.cfg(self.d, self.s, self.grid, self.k),
            );
            for (gx, b) in grad_x.iter_mut().zip(&bw.grad_x) {
                *gx += b;
            }
        }

        // Spectral-entropy regulariser on the node-embedding matrix x (n_nodes × d).
        let (_reg, grad_x_reg) = reg.step(&self.x, self.n_nodes, self.d);
        if reg_on {
            for (gx, gr) in grad_x.iter_mut().zip(&grad_x_reg) {
                *gx += gr;
            }
        }
        for (xi, gx) in self.x.iter_mut().zip(&grad_x) {
            *xi -= lr * gx;
        }

        // Report the task loss at the (pre-update) logits.
        let mut logits = vec![0.0f32; n * 2];
        let y: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
        for t in 0..n {
            let (l0, l1) = self.logits_row(&feats[t * self.d..(t + 1) * self.d]);
            logits[2 * t] = l0;
            logits[2 * t + 1] = l1;
        }
        (cross_entropy(&logits, &y).loss, reg.last_h_norm)
    }
}

fn toy() -> (Vec<EdgeGroup>, Vec<f32>) {
    let groups = vec![
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
    ];
    let labels = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    (groups, labels)
}

fn reg_cfg() -> SpectralEntropyConfig {
    SpectralEntropyConfig {
        lam_0: 0.5,
        lam_a: 1.0,
        lam_b: 1.0,
        lam_kl: 0.0,
        target: 0.5,
        ..Default::default()
    }
}

/// Run the loop; returns (initial task loss, final task loss, final H_norm).
fn run(reg_on: bool) -> (f32, f32, f32) {
    let (groups, labels) = toy();
    let mut model = Model::new(10, 6, 2, 6, 4, 7);
    let mut reg = SpectralEntropyReg::new(reg_cfg());
    let (initial, _) = model.step(&groups, &labels, 0.15, &mut reg, reg_on);
    let mut last = (initial, 0.0);
    for _ in 0..400 {
        last = model.step(&groups, &labels, 0.15, &mut reg, reg_on);
    }
    (initial, last.0, last.1)
}

#[test]
fn spectral_regulariser_wired_into_training_shapes_spectrum() {
    let target = 0.5f32;
    let (on_init, on_loss, on_h) = run(true);
    let (_off_init, off_loss, off_h) = run(false);

    eprintln!("HSiKAN training + spectral-entropy regulariser (target H_norm={target}):");
    eprintln!("  reg ON : task {on_init:.4} -> {on_loss:.4}   final H_norm {on_h:.4}");
    eprintln!("  reg OFF:                          final H_norm {off_h:.4}");

    // (a) Task still learns with the regulariser on.
    assert!(
        on_loss < 0.5 * on_init,
        "reg-on task loss did not fall: {on_init:.4}->{on_loss:.4}"
    );
    assert!(off_loss.is_finite());
    // (b) The regulariser pulls H_norm toward τ — closer than the reg-off control.
    assert!(
        (on_h - target).abs() < (off_h - target).abs(),
        "regulariser did not move H_norm toward target: on |Δτ|={:.4} vs off |Δτ|={:.4}",
        (on_h - target).abs(),
        (off_h - target).abs()
    );
}
