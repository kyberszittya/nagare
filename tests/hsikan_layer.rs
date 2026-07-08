//! Phase 1c — Nagare entropy-gated local learning on mixed-arity HSiKAN features.
//!
//! HSiKAN (`src/ops/hsikan.rs`) is a **fixed** feature extractor over a signed
//! hypergraph with BOTH arity-3 and arity-4 hyperedges sharing ONE parameter set (the
//! mixed-arity claim); a linear readout is trained on the per-edge embeddings by
//! Nagare's entropy-gated local delta rule (`gate = 0.25 + H(softmax)`) or a constant
//! gate. Entropy-vs-constant is *reported*, not asserted (the winner is the
//! measurement; the pass condition is only that the local update drives learning).
//! The multi-seed median/IQR is in `hsikan_multiseed.rs`. Scaffolding: `common`.

mod common;
use common::{teacher_labels, toy, train_mode, FeatureExtractor};

#[test]
fn entropy_gated_local_learning_on_mixed_arity() {
    let d = 6;
    let groups = toy();
    let extractor = FeatureExtractor::new(10, d, 2, 6, 4, 7);
    let feats = extractor.features(&groups);
    let labels = teacher_labels(&feats, d, 11);
    assert_eq!(labels.len(), 12, "6 arity-3 + 6 arity-4 edges");
    assert!(feats.iter().all(|v| v.is_finite()));

    let (e_init, e_loss, e_acc) = train_mode(&feats, &labels, d, true, 99, 300);
    let (c_init, c_loss, c_acc) = train_mode(&feats, &labels, d, false, 99, 300);
    eprintln!("HSiKAN mixed-arity entropy-gated local learning:");
    eprintln!("  entropy : BCE {e_init:.4} -> {e_loss:.4}  acc {e_acc:.3}");
    eprintln!("  constant: BCE {c_init:.4} -> {c_loss:.4}  acc {c_acc:.3}");

    for (name, init, loss, acc) in [
        ("entropy", e_init, e_loss, e_acc),
        ("constant", c_init, c_loss, c_acc),
    ] {
        assert!(
            loss < 0.5 * init,
            "{name}: loss did not fall: {init:.4}->{loss:.4}"
        );
        assert!(
            acc >= 0.9,
            "{name}: separable task not learned: acc {acc:.3}"
        );
    }
}

#[test]
fn shared_params_features_finite_both_arities() {
    let d = 6;
    let groups = toy();
    let extractor = FeatureExtractor::new(10, d, 2, 6, 4, 3);
    let feats = extractor.features(&groups);
    assert_eq!(feats.len(), 12 * d); // 12 edges across both arities
    assert!(feats.iter().all(|v| v.is_finite()));
}
