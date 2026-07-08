//! Phase 1c (multi-seed) — entropy-vs-constant on mixed-arity HSiKAN features across
//! N independent seeds, reporting median/IQR (§3) so the 1c single-seed lead becomes a
//! real distributional claim rather than a point estimate. Each seed draws fresh HSiKAN
//! feature params and a fresh linear-teacher label split; both gate modes are trained
//! on the identical (features, labels) instance. The winner is *reported* — the pass
//! condition only requires that both modes learn robustly across seeds.

mod common;
use common::{teacher_labels, toy, train_mode, FeatureExtractor};

/// (median, q1, q3) of a sample (nearest-rank).
fn median_iqr(mut v: Vec<f32>) -> (f32, f32, f32) {
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let q = |p: f32| v[(((n - 1) as f32) * p).round() as usize];
    (q(0.5), q(0.25), q(0.75))
}

#[test]
fn entropy_vs_constant_multiseed() {
    let d = 6;
    let n_seeds = 15u64;
    let (mut e_losses, mut c_losses) = (Vec::new(), Vec::new());
    let mut e_wins = 0;
    for s in 0..n_seeds {
        let groups = toy();
        let feats = FeatureExtractor::new(10, d, 2, 6, 4, s).features(&groups);
        let labels = teacher_labels(&feats, d, 1000 + s);
        let (_, e_loss, _) = train_mode(&feats, &labels, d, true, 2000 + s, 300);
        let (_, c_loss, _) = train_mode(&feats, &labels, d, false, 2000 + s, 300);
        if e_loss < c_loss {
            e_wins += 1;
        }
        eprintln!("SEED {s} entropy {e_loss:.6} constant {c_loss:.6}");
        e_losses.push(e_loss);
        c_losses.push(c_loss);
    }

    let (em, eq1, eq3) = median_iqr(e_losses.clone());
    let (cm, cq1, cq3) = median_iqr(c_losses.clone());
    eprintln!("HSiKAN mixed-arity entropy-vs-constant, {n_seeds} seeds (final BCE):");
    eprintln!("  entropy : median {em:.4}  IQR [{eq1:.4}, {eq3:.4}]");
    eprintln!("  constant: median {cm:.4}  IQR [{cq1:.4}, {cq3:.4}]");
    eprintln!("  entropy < constant in {e_wins}/{n_seeds} seeds");

    // Report-only science: both modes must learn robustly; the winner is reported,
    // not asserted (do not bake a hoped-for direction into the pass condition).
    assert!(em.is_finite() && cm.is_finite());
    assert!(
        em < 0.45 && cm < 0.45,
        "a gate mode failed to learn across seeds: entropy median {em:.4}, constant median {cm:.4}"
    );
}
