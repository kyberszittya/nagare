//! Integration test: end-to-end training convergence.
//!
//! Verifies that `NagareRuntime::step` produces a gradient signal that
//! actually reduces BCE loss on a small, analytically-checkable synthetic
//! task. This is the cheapest correctness gate for the full operator
//! composition (FIR → scatter-mean → linear → BCE → backward → Adam).

use holonomy_learn::ops::loss::bce_with_logits_forward;
use holonomy_learn::NagareRuntime;
use hymeko_graph::TopKCyclesBatch;

fn make_batch(cycles: Vec<u32>, signs: Vec<i8>, k: usize) -> TopKCyclesBatch {
    let n = cycles.len() / k;
    TopKCyclesBatch {
        cycles,
        signs,
        scores: vec![0.0; n],
        k,
    }
}

/// Trivial separable task: vertices 0, 2 → label 1; vertices 1, 3 → label 0.
///
/// Feature encoding:
///   v0, v2 : [1, 0]
///   v1, v3 : [0, 1]
///
/// Any model that picks up the first feature component should classify
/// perfectly. The cycle pool reinforces the signal via sign-weighted
/// aggregation.
#[test]
fn loss_decreases_on_separable_task() {
    let batch = make_batch(
        vec![0, 1, 2, 1, 2, 3, 0, 2, 3, 0, 1, 3],
        vec![1, -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, -1],
        3,
    );
    let features = vec![
        1.0f32, 0.0, // v0 — label 1
        0.0, 1.0, // v1 — label 0
        1.0, 0.0, // v2 — label 1
        0.0, 1.0, // v3 — label 0
    ];
    let targets = vec![1.0f32, 0.0, 1.0, 0.0];

    let mut rt = NagareRuntime::new(3, 2, 1, 5e-2, 99);
    let loss_0 = rt.step(&batch, &features, 4, &targets);
    assert!(loss_0.is_finite(), "initial loss must be finite");

    for _ in 1..200 {
        rt.step(&batch, &features, 4, &targets);
    }

    let logits = rt.predict(&batch, &features, 4);
    let loss_final = bce_with_logits_forward(&logits, &targets);
    let acc = NagareRuntime::accuracy(&logits, &targets);

    assert!(
        loss_final < loss_0,
        "loss did not decrease: {loss_0:.4} → {loss_final:.4}"
    );
    assert!(
        acc >= 0.75,
        "accuracy below 0.75 after 200 steps ({acc:.2}); gradient may be broken"
    );
}

/// Sanity: a runtime with d_out=2 (multi-label) must also produce
/// finite loss and non-NaN logits.
#[test]
fn multi_label_head_is_finite() {
    let batch = make_batch(vec![0, 1, 2, 1, 2, 3], vec![1, -1, 1, 1, 1, -1], 3);
    let features = vec![0.1f32; 4 * 4];
    // targets: 4 vertices × 2 logits each
    let targets = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
    let mut rt = NagareRuntime::new(3, 4, 2, 1e-3, 17);
    let loss = rt.step(&batch, &features, 4, &targets);
    assert!(loss.is_finite());
    let logits = rt.predict(&batch, &features, 4);
    assert_eq!(logits.len(), 4 * 2);
    assert!(logits.iter().all(|x| x.is_finite()));
}

/// Gradient-flow sanity: two identical steps must not produce NaN
/// parameters (checks Adam bias-correction doesn't divide by zero).
#[test]
fn repeated_steps_stay_finite() {
    let batch = make_batch(
        vec![0, 1, 2, 2, 3, 4, 0, 3, 5],
        vec![1, 1, -1, -1, 1, 1, 1, -1, 1],
        3,
    );
    let n_v = 6;
    let features: Vec<f32> = (0..n_v * 8).map(|i| (i as f32) * 0.05 - 1.0).collect();
    let targets: Vec<f32> = (0..n_v)
        .map(|v| if v % 2 == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut rt = NagareRuntime::new(3, 8, 1, 1e-2, 5);
    for step in 0..50 {
        let loss = rt.step(&batch, &features, n_v, &targets);
        assert!(
            loss.is_finite(),
            "loss became non-finite at step {step}: {loss}"
        );
    }
    // Parameters must be finite after 50 steps.
    assert!(
        rt.fir.a.iter().all(|x| x.is_finite()),
        "fir.a contains NaN/Inf"
    );
    assert!(
        rt.fir.b.iter().all(|x| x.is_finite()),
        "fir.b contains NaN/Inf"
    );
    assert!(
        rt.head.w.iter().all(|x| x.is_finite()),
        "head.w contains NaN/Inf"
    );
}
