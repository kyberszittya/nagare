//! `NagareRuntime` — the execution band.
//!
//! Wires the Nagare operator set into one training/inference unit:
//!
//! ```text
//!   clifford_fir_forward    → per_cycle  [n_c × d_in]
//!   scatter_mean_forward    → h           [n_v × d_in]
//!   linear_forward          → logits      [n_v × d_out]
//!   bce_with_logits_forward → scalar loss
//!
//!   ── backward (closed-form, no autograd) ──
//!   bce_with_logits_backward → grad_logits
//!   linear_backward          → (grad_h, grad_head)
//!   scatter_mean_backward    → grad_per_cycle
//!   clifford_fir_backward    → (_, grad_fir)
//!
//!   ── Adam (4 param groups) ──
//!   fir.a | fir.b | head.w | head.b
//! ```
//!
//! No allocation on the hot path beyond the forward activation buffers.
//! No Python, no autograd graph, no dynamic dispatch.
//!
//! Relation to the full HyMeKo vertical:
//! ```text
//!   HyMeKo lang  →  HIVE IR  →  [NAGARE]  →  HSMM  →  Zynq
//! ```
//! This module is the **programming model** tier: it takes a compiled
//! cycle pool (= HIVE's hypergraph structure) and runs it end-to-end.

use hymeko_graph::{clifford_fir_backward, clifford_fir_forward, CliffordFIR, TopKCyclesBatch};

use crate::ops::{
    adam::{adam_step, AdamState},
    linear::{linear_backward, linear_forward, LinearLayer},
    loss::{bce_with_logits_backward, bce_with_logits_forward},
    scatter::{scatter_mean_backward, scatter_mean_forward},
};

/// Complete NAGARE training/inference runtime for a signed-hypergraph
/// classification head.
///
/// Parameter groups:
///
/// | group     | shape         | description                              |
/// |-----------|---------------|------------------------------------------|
/// | `fir.a`   | `(k,)`        | Clifford scalar-grade (σ=+1 branch)      |
/// | `fir.b`   | `(k,)`        | Clifford pseudoscalar-grade (σ=−1 branch)|
/// | `head.w`  | `(d_in×d_out)`| Linear head weights                      |
/// | `head.b`  | `(d_out,)`    | Linear head bias                         |
///
/// The `step` method performs one complete forward + backward + Adam update
/// and returns the scalar BCE loss.
#[derive(Debug)]
pub struct NagareRuntime {
    /// Signed-cycle FIR filter (Clifford Cl(0,1) parameterisation).
    pub fir: CliffordFIR,
    /// Linear classification head.
    pub head: LinearLayer,

    // Adam moment buffers (private — implementation detail).
    fir_adam_a: AdamState,
    fir_adam_b: AdamState,
    head_adam_w: AdamState,
    head_adam_b: AdamState,

    /// Adam learning rate.
    pub lr: f32,
    /// Per-vertex feature dimension (FIR input / scatter-mean output dim).
    pub d_in: usize,
    /// Logit dimension (1 for binary classification).
    pub d_out: usize,
}

impl NagareRuntime {
    /// Construct a new runtime with signed-mean FIR init and Glorot head.
    ///
    /// - `k`     : cycle length (filter length)
    /// - `d_in`  : per-vertex feature dimension
    /// - `d_out` : logit dimension (use 1 for binary BCE)
    /// - `lr`    : Adam learning rate
    /// - `seed`  : RNG seed for linear head initialisation
    pub fn new(k: usize, d_in: usize, d_out: usize, lr: f32, seed: u64) -> Self {
        let fir = CliffordFIR::signed_mean(k);
        let head = LinearLayer::new(d_in, d_out, seed);
        Self {
            fir_adam_a: AdamState::new(k),
            fir_adam_b: AdamState::new(k),
            head_adam_w: AdamState::new(head.w.len()),
            head_adam_b: AdamState::new(head.b.len()),
            fir,
            head,
            lr,
            d_in,
            d_out,
        }
    }

    /// Forward-only inference. Returns logits `(n_vertices × d_out)` flat.
    ///
    /// Does not update any parameters or Adam state.
    pub fn predict(
        &self,
        batch: &TopKCyclesBatch,
        features: &[f32],
        n_vertices: usize,
    ) -> Vec<f32> {
        let per_cycle = clifford_fir_forward(batch, features, n_vertices, self.d_in, &self.fir);
        let (h, _) =
            scatter_mean_forward(&batch.cycles, batch.k, &per_cycle, self.d_in, n_vertices);
        linear_forward(&self.head, &h)
    }

    /// One gradient step over a full cycle-pool batch.
    ///
    /// Returns the scalar BCE-with-logits loss *before* the parameter update.
    ///
    /// `targets` must be `(n_vertices × d_out)` flat with values in {0.0, 1.0}.
    /// For binary classification (`d_out == 1`) pass a `(n_vertices,)` slice.
    pub fn step(
        &mut self,
        batch: &TopKCyclesBatch,
        features: &[f32],
        n_vertices: usize,
        targets: &[f32],
    ) -> f32 {
        debug_assert_eq!(
            targets.len(),
            n_vertices * self.d_out,
            "targets must be (n_vertices × d_out)"
        );

        // ── Forward ────────────────────────────────────────────────────
        let per_cycle = clifford_fir_forward(batch, features, n_vertices, self.d_in, &self.fir);
        let (h, counts) =
            scatter_mean_forward(&batch.cycles, batch.k, &per_cycle, self.d_in, n_vertices);
        let logits = linear_forward(&self.head, &h);
        let loss = bce_with_logits_forward(&logits, targets);

        // ── Backward (all closed-form) ─────────────────────────────────
        let grad_logits = bce_with_logits_backward(&logits, targets);
        let (grad_h, grad_head) = linear_backward(&self.head, &h, &grad_logits);
        let grad_per_cycle = scatter_mean_backward(
            &batch.cycles,
            batch.k,
            &grad_h,
            self.d_in,
            &counts,
            n_vertices,
        );
        let (_grad_x, grad_fir) = clifford_fir_backward(
            batch,
            features,
            n_vertices,
            self.d_in,
            &self.fir,
            &grad_per_cycle,
        );

        // ── Adam (4 param groups) ──────────────────────────────────────
        adam_step(&mut self.fir.a, &grad_fir.a, &mut self.fir_adam_a, self.lr);
        adam_step(&mut self.fir.b, &grad_fir.b, &mut self.fir_adam_b, self.lr);
        adam_step(
            &mut self.head.w,
            &grad_head.w,
            &mut self.head_adam_w,
            self.lr,
        );
        adam_step(
            &mut self.head.b,
            &grad_head.b,
            &mut self.head_adam_b,
            self.lr,
        );

        loss
    }

    /// Binary classification accuracy: predicted class = (logit > 0.0).
    ///
    /// `logits` and `targets` must have equal length.
    pub fn accuracy(logits: &[f32], targets: &[f32]) -> f32 {
        assert_eq!(logits.len(), targets.len());
        if logits.is_empty() {
            return 0.0;
        }
        let correct = logits
            .iter()
            .zip(targets.iter())
            .filter(|&(&l, &t)| {
                let pred = if l > 0.0 { 1.0f32 } else { 0.0 };
                (pred - t).abs() < 0.5
            })
            .count();
        correct as f32 / logits.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hymeko_graph::TopKCyclesBatch;

    /// Helper: build a trivial cycle pool from flat arrays.
    fn make_batch(cycles: Vec<u32>, signs: Vec<i8>, k: usize) -> TopKCyclesBatch {
        let n = cycles.len() / k;
        TopKCyclesBatch {
            cycles,
            signs,
            scores: vec![0.0; n],
            k,
        }
    }

    #[test]
    fn predict_output_shape() {
        // 4 vertices, 2 triangles, d_in=3, d_out=1.
        let batch = make_batch(vec![0, 1, 2, 1, 2, 3], vec![1, -1, 1, 1, 1, -1], 3);
        let rt = NagareRuntime::new(3, 3, 1, 1e-3, 42);
        let features = vec![0.1f32; 4 * 3];
        let logits = rt.predict(&batch, &features, 4);
        assert_eq!(logits.len(), 4, "logit shape = n_vertices x d_out");
    }

    #[test]
    fn step_returns_finite_loss() {
        let batch = make_batch(vec![0, 1, 2, 1, 2, 3], vec![1, -1, 1, 1, 1, -1], 3);
        let mut rt = NagareRuntime::new(3, 4, 1, 1e-3, 7);
        let features: Vec<f32> = (0..4 * 4).map(|i| i as f32 * 0.1).collect();
        let targets = vec![1.0, 0.0, 1.0, 0.0];
        let loss = rt.step(&batch, &features, 4, &targets);
        assert!(
            loss.is_finite(),
            "step() must return finite loss, got {loss}"
        );
        assert!(
            loss > 0.0,
            "BCE loss must be positive for non-trivial targets"
        );
    }

    #[test]
    fn loss_decreases_after_training() {
        // Synthetic task: vertex 0 and 2 have features [1, 0], label 1.
        //                 vertex 1 and 3 have features [0, 1], label 0.
        // Cycles: (0,1,2) + and (1,2,3) mixed signs.
        // A Nagare model with d_in=2 should fit this in ≤100 steps.
        let batch = make_batch(
            vec![0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3],
            vec![1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1],
            3,
        );
        let features = vec![
            1.0f32, 0.0, // v0
            0.0, 1.0, // v1
            1.0, 0.0, // v2
            0.0, 1.0, // v3
        ];
        let targets = vec![1.0f32, 0.0, 1.0, 0.0];
        let mut rt = NagareRuntime::new(3, 2, 1, 5e-2, 99);
        let loss_0 = rt.step(&batch, &features, 4, &targets);
        for _ in 1..100 {
            rt.step(&batch, &features, 4, &targets);
        }
        let logits = rt.predict(&batch, &features, 4);
        let loss_100 = bce_with_logits_forward(&logits, &targets);
        assert!(
            loss_100 < loss_0,
            "loss should decrease after 100 steps: {loss_0:.4} → {loss_100:.4}"
        );
    }

    #[test]
    fn accuracy_helper() {
        let logits = vec![1.0f32, -1.0, 0.5, -0.5];
        let targets = vec![1.0f32, 0.0, 1.0, 0.0];
        let acc = NagareRuntime::accuracy(&logits, &targets);
        assert!(
            (acc - 1.0).abs() < 1e-6,
            "perfect prediction should be 1.0, got {acc}"
        );
    }
}
