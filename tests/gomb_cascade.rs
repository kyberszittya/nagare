//! Phase 2b — the two-shell Gömb cascade: outer FIR shell → middle HSiKAN → readout.
//!
//! Composes existing ops end-to-end (no new derivative):
//!   X (per-vertex) → `gomb_outer` (M FIR banks, per-cycle) → `scatter_mean` (back to
//!   per-vertex) → `hsikan` (over the cycles-as-hyperedges, per-cycle) → linear readout.
//! Trained by the composed closed-form backward (readout → hsikan → scatter → outer),
//! features frozen. This demonstrates the middle shell (the Phase-1 `hsikan` op)
//! composes after the outer shell and the whole cascade learns.
//!
//! Discriminating add: against a **nonlinear** target (from a fixed two-shell teacher —
//! the HSiKAN spline nonlinearity), a **two-shell** student (with the middle) vs a
//! **one-shell** student (outer + linear readout only). The middle should help where the
//! target is nonlinear in the outer features. Reported, not asserted (§3); single arity,
//! single seed, constructed target — mixed-arity + natural-task are follow-ups.

use holonomy_learn::{
    cross_entropy, gomb_outer_backward, gomb_outer_forward, hsikan_backward, hsikan_forward,
    linear_backward, linear_forward, scatter_mean_backward, scatter_mean_forward, softmax2,
    HsikanConfig, HsikanEdges, HsikanParams, LinearLayer,
};
use hymeko_graph::{CliffordFIR, TopKCyclesBatch};
use rand::{rngs::StdRng, Rng, SeedableRng};

const V: usize = 10;
const K: usize = 3;
const D_FEAT: usize = 2;
const M: usize = 2;
const HID: usize = M * D_FEAT; // outer output width = HSiKAN hidden dim
const S: usize = 2;
const GRID: usize = 6;
const CHEB: usize = 4;

fn make_batch(n_cycles: usize, rng: &mut StdRng) -> TopKCyclesBatch {
    TopKCyclesBatch {
        cycles: (0..n_cycles * K)
            .map(|_| rng.random_range(0..V as u32))
            .collect(),
        signs: (0..n_cycles * K)
            .map(|_| if rng.random::<bool>() { 1i8 } else { -1 })
            .collect(),
        scores: vec![0.0f64; n_cycles],
        k: K,
    }
}

fn rand_vec(n: usize, scale: f32, rng: &mut StdRng) -> Vec<f32> {
    (0..n)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
        .collect()
}

fn ce_grad(logits: &[f32], labels: &[f32]) -> Vec<f32> {
    let n = labels.len();
    let mut g = vec![0.0f32; n * 2];
    for t in 0..n {
        let (p0, p1) = softmax2(logits[2 * t], logits[2 * t + 1]);
        g[2 * t] = (p0 - f32::from(labels[t] == 0.0)) / n as f32;
        g[2 * t + 1] = (p1 - f32::from(labels[t] == 1.0)) / n as f32;
    }
    g
}

fn sgd(p: &mut [f32], g: &[f32], lr: f32) {
    for (pi, gi) in p.iter_mut().zip(g) {
        *pi -= lr * gi;
    }
}

/// The two-shell cascade's learnable parameters.
struct TwoShell {
    banks: Vec<CliffordFIR>,
    inner: Vec<f32>,
    outer: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    readout: LinearLayer,
}

impl TwoShell {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let banks = (0..M)
            .map(|_| CliffordFIR::new(rand_vec(K, 0.4, &mut rng), rand_vec(K, 0.4, &mut rng)))
            .collect();
        Self {
            banks,
            inner: rand_vec(S * HID * CHEB, 0.3, &mut rng),
            outer: rand_vec(S * HID * CHEB, 0.3, &mut rng),
            gw: rand_vec(HID * HID, 0.2, &mut rng),
            gb: vec![-1.0; HID],
            readout: LinearLayer::new(HID, 2, seed.wrapping_add(1)),
        }
    }
    fn params(&self) -> HsikanParams<'_> {
        HsikanParams {
            inner_coef: &self.inner,
            outer_coef: &self.outer,
            gate_w: &self.gw,
            gate_b: &self.gb,
        }
    }
    fn cfg(&self, nc: usize) -> HsikanConfig {
        HsikanConfig::new(nc, K, HID, S, GRID, CHEB, true)
    }

    /// Per-cycle logits `(n_cycles, 2)` (full cascade forward).
    fn logits(&self, batch: &TopKCyclesBatch, x: &[f32]) -> Vec<f32> {
        let y = gomb_outer_forward(batch, x, &self.banks, V, D_FEAT);
        let (x_outer, _counts) = scatter_mean_forward(&batch.cycles, K, &y, HID, V);
        let edges = HsikanEdges {
            vertices: &batch.cycles,
            signs: &batch.signs,
        };
        let (h, _) = hsikan_forward(self.params(), &x_outer, edges, self.cfg(batch.len()));
        linear_forward(&self.readout, &h)
    }

    /// One SGD step through the composed backward; returns the pre-update loss.
    fn step(
        &mut self,
        batch: &TopKCyclesBatch,
        x: &[f32],
        labels: &[f32],
        y_usize: &[usize],
        lr: f32,
    ) -> f32 {
        // Forward, keeping intermediates.
        let y = gomb_outer_forward(batch, x, &self.banks, V, D_FEAT);
        let (x_outer, counts) = scatter_mean_forward(&batch.cycles, K, &y, HID, V);
        let cfg = self.cfg(batch.len());
        let edges = HsikanEdges {
            vertices: &batch.cycles,
            signs: &batch.signs,
        };
        let (h, hcache) = hsikan_forward(self.params(), &x_outer, edges, cfg);
        let logits = linear_forward(&self.readout, &h);
        let loss = cross_entropy(&logits, y_usize).loss;

        // Backward: readout → hsikan → scatter → outer.
        let grad_logits = ce_grad(&logits, labels);
        let (grad_h, grad_readout) = linear_backward(&self.readout, &h, &grad_logits);
        let hb = hsikan_backward(self.params(), edges, &hcache, &grad_h, cfg);
        let grad_y = scatter_mean_backward(&batch.cycles, K, &hb.grad_x, HID, &counts, V);
        let (_grad_feat, grad_banks) =
            gomb_outer_backward(batch, x, &self.banks, &grad_y, V, D_FEAT);

        // Update (features frozen).
        for (bank, gbank) in self.banks.iter_mut().zip(&grad_banks) {
            sgd(&mut bank.a, &gbank.a, lr);
            sgd(&mut bank.b, &gbank.b, lr);
        }
        sgd(&mut self.inner, &hb.grad_inner_coef, lr);
        sgd(&mut self.outer, &hb.grad_outer_coef, lr);
        sgd(&mut self.gw, &hb.grad_gate_w, lr);
        sgd(&mut self.gb, &hb.grad_gate_b, lr);
        sgd(&mut self.readout.w, &grad_readout.w, lr);
        sgd(&mut self.readout.b, &grad_readout.b, lr);
        loss
    }
}

/// One-shell baseline: outer FIR shell + a linear readout (no middle).
struct OneShell {
    banks: Vec<CliffordFIR>,
    readout: LinearLayer,
}

impl OneShell {
    fn new(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let banks = (0..M)
            .map(|_| CliffordFIR::new(rand_vec(K, 0.4, &mut rng), rand_vec(K, 0.4, &mut rng)))
            .collect();
        Self {
            banks,
            readout: LinearLayer::new(HID, 2, seed.wrapping_add(1)),
        }
    }
    fn step(
        &mut self,
        batch: &TopKCyclesBatch,
        x: &[f32],
        labels: &[f32],
        y_usize: &[usize],
        lr: f32,
    ) -> f32 {
        let y = gomb_outer_forward(batch, x, &self.banks, V, D_FEAT);
        let logits = linear_forward(&self.readout, &y);
        let loss = cross_entropy(&logits, y_usize).loss;
        let grad_logits = ce_grad(&logits, labels);
        let (grad_y, grad_readout) = linear_backward(&self.readout, &y, &grad_logits);
        let (_gf, grad_banks) = gomb_outer_backward(batch, x, &self.banks, &grad_y, V, D_FEAT);
        for (bank, gbank) in self.banks.iter_mut().zip(&grad_banks) {
            sgd(&mut bank.a, &gbank.a, lr);
            sgd(&mut bank.b, &gbank.b, lr);
        }
        sgd(&mut self.readout.w, &grad_readout.w, lr);
        sgd(&mut self.readout.b, &grad_readout.b, lr);
        loss
    }
}

/// Nonlinear labels from a fixed two-shell teacher (HSiKAN nonlinearity), median-split.
fn teacher_labels(batch: &TopKCyclesBatch, x: &[f32], seed: u64) -> Vec<f32> {
    let t = TwoShell::new(seed);
    let y = gomb_outer_forward(batch, x, &t.banks, V, D_FEAT);
    let (x_outer, _c) = scatter_mean_forward(&batch.cycles, K, &y, HID, V);
    let edges = HsikanEdges {
        vertices: &batch.cycles,
        signs: &batch.signs,
    };
    let (h, _) = hsikan_forward(t.params(), &x_outer, edges, t.cfg(batch.len()));
    let nc = batch.len();
    let scores: Vec<f32> = (0..nc)
        .map(|c| (0..HID).map(|j| h[c * HID + j] * t.readout.w[j * 2]).sum())
        .collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[nc / 2];
    scores.iter().map(|&s| f32::from(s > med)).collect()
}

fn acc(logits: &[f32], labels: &[f32]) -> f32 {
    let n = labels.len();
    let correct = (0..n)
        .filter(|&t| ((logits[2 * t + 1] > logits[2 * t]) as usize as f32) == labels[t])
        .count();
    correct as f32 / n as f32
}

#[test]
fn two_shell_cascade_learns_and_vs_one_shell() {
    let mut rng = StdRng::seed_from_u64(5);
    let x = rand_vec(V * D_FEAT, 0.5, &mut rng);
    let batch = make_batch(24, &mut rng);
    let labels = teacher_labels(&batch, &x, 99);
    let y_usize: Vec<usize> = labels.iter().map(|&l| l as usize).collect();

    let mut two = TwoShell::new(3);
    let mut one = OneShell::new(3);
    let (mut two_init, mut two_last) = (0.0f32, 0.0f32);
    let (mut one_init, mut one_last) = (0.0f32, 0.0f32);
    for e in 0..500 {
        let t = two.step(&batch, &x, &labels, &y_usize, 0.1);
        let o = one.step(&batch, &x, &labels, &y_usize, 0.1);
        if e == 0 {
            two_init = t;
            one_init = o;
        }
        two_last = t;
        one_last = o;
    }
    let two_acc = acc(&two.logits(&batch, &x), &labels);

    eprintln!("Gömb two-shell cascade (outer FIR → middle HSiKAN → readout):");
    eprintln!("  two-shell: BCE {two_init:.4} -> {two_last:.4}  acc {two_acc:.3}");
    eprintln!("  one-shell: BCE {one_init:.4} -> {one_last:.4}  (outer + linear readout)");
    eprintln!(
        "  verdict: middle {} (Δ {:.4})",
        if two_last < one_last {
            "helps"
        } else {
            "does NOT help"
        },
        one_last - two_last
    );

    // The composed cascade must learn end-to-end; the vs-one-shell verdict is reported.
    assert!(
        two_last < 0.6 * two_init,
        "two-shell cascade did not learn: {two_init:.4}->{two_last:.4}"
    );
    assert!(two_last.is_finite() && one_last.is_finite());
}
