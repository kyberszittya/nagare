//! Phase 2a discriminating test — does the Gömb outer **shell** (M banks) beat **flat**
//! pooling (M=1) on mixed arity? (the handoff's Phase-2 question).
//!
//! Setup: FIXED random node features over a signed cycle pool with BOTH arity-3 and
//! arity-4 cycles (per-arity banks, since a CliffordFIR's length = its cycle arity).
//! Labels come from a teacher with **2-filter structure** (a fixed 2-bank shell + linear
//! readout, median-split) — a target that genuinely needs more than one filter. Two
//! students learn it by closed-form backward (features frozen; only banks + readout
//! train): a **shell** (M=2 banks/arity) and a **flat** baseline (M=1). We *report* the
//! verdict — the pass condition only requires both to learn (§3). This is a constructed
//! target (isolates shell capacity); the natural-task comparison (real signed-link
//! cycles) is a follow-up.

use holonomy_learn::{
    cross_entropy, gomb_outer_backward, gomb_outer_forward, linear_backward, linear_forward,
    softmax2, LinearLayer,
};
use hymeko_graph::{CliffordFIR, TopKCyclesBatch};
use rand::{rngs::StdRng, Rng, SeedableRng};

const V: usize = 12;
const D: usize = 4;

fn make_batch(n_cycles: usize, k: usize, rng: &mut StdRng) -> TopKCyclesBatch {
    let cycles = (0..n_cycles * k)
        .map(|_| rng.random_range(0..V as u32))
        .collect();
    let signs = (0..n_cycles * k)
        .map(|_| if rng.random::<bool>() { 1i8 } else { -1 })
        .collect();
    TopKCyclesBatch {
        cycles,
        signs,
        scores: vec![0.0f64; n_cycles],
        k,
    }
}

fn random_banks(m: usize, k: usize, rng: &mut StdRng) -> Vec<CliffordFIR> {
    (0..m)
        .map(|_| {
            let a = (0..k)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.4)
                .collect();
            let b = (0..k)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.4)
                .collect();
            CliffordFIR::new(a, b)
        })
        .collect()
}

/// Concatenate the per-group outer-shell outputs into one `(total_cycles, M·d)` matrix.
fn shell_features(groups: &[TopKCyclesBatch], x: &[f32], banks: &[Vec<CliffordFIR>]) -> Vec<f32> {
    let mut y = Vec::new();
    for (g, gb) in groups.iter().zip(banks) {
        y.extend_from_slice(&gomb_outer_forward(g, x, gb, V, D));
    }
    y
}

/// Labels from a fixed 2-bank teacher shell + linear readout, median-split (balanced).
fn teacher_labels(groups: &[TopKCyclesBatch], x: &[f32], rng: &mut StdRng) -> Vec<f32> {
    let banks: Vec<Vec<CliffordFIR>> = groups.iter().map(|g| random_banks(2, g.k, rng)).collect();
    let w: Vec<f32> = (0..2 * D)
        .map(|_| rng.random::<f32>() * 2.0 - 1.0)
        .collect();
    let feats = shell_features(groups, x, &banks); // (total, 2*D)
    let n = feats.len() / (2 * D);
    let scores: Vec<f32> = (0..n)
        .map(|c| {
            feats[c * 2 * D..(c + 1) * 2 * D]
                .iter()
                .zip(&w)
                .map(|(f, wi)| f * wi)
                .sum()
        })
        .collect();
    let mut sorted = scores.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let med = sorted[n / 2];
    scores.iter().map(|&s| f32::from(s > med)).collect()
}

/// Train a student with `n_banks` banks/arity; features frozen. Returns (init, final, acc) BCE.
fn train_student(
    groups: &[TopKCyclesBatch],
    x: &[f32],
    labels: &[f32],
    n_banks: usize,
    seed: u64,
    epochs: usize,
) -> (f32, f32, f32) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut banks: Vec<Vec<CliffordFIR>> = groups
        .iter()
        .map(|g| random_banks(n_banks, g.k, &mut rng))
        .collect();
    let mut readout = LinearLayer::new(n_banks * D, 2, seed.wrapping_add(1));
    let y_usize: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
    let lr = 0.1;

    let eval = |banks: &[Vec<CliffordFIR>], readout: &LinearLayer| -> (f32, f32) {
        let feats = shell_features(groups, x, banks);
        let logits = linear_forward(readout, &feats);
        let ce = cross_entropy(&logits, &y_usize);
        (ce.loss, ce.acc)
    };
    let (initial, _) = eval(&banks, &readout);

    for _ in 0..epochs {
        let feats = shell_features(groups, x, &banks); // (total, n_banks*D)
        let logits = linear_forward(&readout, &feats);
        // CE-softmax gradient on logits (mean over samples).
        let n = labels.len();
        let mut grad_logits = vec![0.0f32; n * 2];
        for t in 0..n {
            let (p0, p1) = softmax2(logits[2 * t], logits[2 * t + 1]);
            grad_logits[2 * t] = (p0 - f32::from(labels[t] == 0.0)) / n as f32;
            grad_logits[2 * t + 1] = (p1 - f32::from(labels[t] == 1.0)) / n as f32;
        }
        let (grad_feats, grad_readout) = linear_backward(&readout, &feats, &grad_logits);

        // Split grad_feats per group, backprop into each group's banks, SGD-update.
        let width = n_banks * D;
        let mut row = 0usize;
        for (g, gb) in groups.iter().zip(banks.iter_mut()) {
            let nc = g.len();
            let gy = &grad_feats[row * width..(row + nc) * width];
            row += nc;
            let (_, grad_banks) = gomb_outer_backward(g, x, gb, gy, V, D);
            for (bank, gbank) in gb.iter_mut().zip(&grad_banks) {
                for (a, ga) in bank.a.iter_mut().zip(&gbank.a) {
                    *a -= lr * ga;
                }
                for (b, gb2) in bank.b.iter_mut().zip(&gbank.b) {
                    *b -= lr * gb2;
                }
            }
        }
        for (w, gw) in readout.w.iter_mut().zip(&grad_readout.w) {
            *w -= lr * gw;
        }
        for (b, gb) in readout.b.iter_mut().zip(&grad_readout.b) {
            *b -= lr * gb;
        }
    }

    let (final_loss, acc) = eval(&banks, &readout);
    (initial, final_loss, acc)
}

#[test]
fn shell_vs_flat_on_mixed_arity() {
    let mut rng = StdRng::seed_from_u64(7);
    let x: Vec<f32> = (0..V * D)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.5)
        .collect();
    let groups = vec![make_batch(10, 3, &mut rng), make_batch(10, 4, &mut rng)];
    let labels = teacher_labels(&groups, &x, &mut rng);
    assert_eq!(labels.len(), 20, "10 arity-3 + 10 arity-4 cycles");

    let (s_init, s_loss, s_acc) = train_student(&groups, &x, &labels, 2, 100, 400);
    let (f_init, f_loss, f_acc) = train_student(&groups, &x, &labels, 1, 100, 400);
    eprintln!("Gömb outer shell vs flat on mixed-arity (2-filter target):");
    eprintln!("  shell (M=2): BCE {s_init:.4} -> {s_loss:.4}  acc {s_acc:.3}");
    eprintln!("  flat  (M=1): BCE {f_init:.4} -> {f_loss:.4}  acc {f_acc:.3}");
    eprintln!(
        "  verdict: shell {} flat (Δ {:.4})",
        if s_loss < f_loss {
            "beats"
        } else {
            "does NOT beat"
        },
        f_loss - s_loss
    );

    // Both must learn; the winner is reported, not asserted (§3).
    assert!(
        s_loss < 0.9 * s_init,
        "shell did not learn: {s_init:.4}->{s_loss:.4}"
    );
    assert!(
        f_loss < 0.95 * f_init,
        "flat did not learn: {f_init:.4}->{f_loss:.4}"
    );
    assert!(s_acc.is_finite() && f_acc.is_finite());
}
