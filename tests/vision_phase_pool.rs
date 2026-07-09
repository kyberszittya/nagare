//! Nagare CV — **quaternion-phase pooling + entropy feedback** (the user's fix for the failed
//! rotor-pool), on the shape task.
//!
//! The vector rotor-pool failed because it rotated *non-equivariant feature channels* (scramble).
//! This pools the **rotor phase** instead: each patch's dominant gradient is a z-rotor whose phase
//! is its orientation `θ_p` (`e^{iθ}` = a unit quaternion). Pooling the magnitude-weighted phases
//! is an orientation histogram `h` that circularly shifts under image rotation — so the
//! rotation-**invariant** summaries are `|DFT(h)|` and the **phase entropy** `H(h)`. No feature
//! vector is ever rotated — only phases are pooled. (Histogram + features + trainer live in
//! `common::vision`, shared with `vision_texture_entropy`.)
//!
//! Three arms (linear classifier on a fixed per-image feature), 4 seeds, randomly-rotated shapes:
//! raw histogram (covariant floor) vs phase-pool `|DFT|` (invariant) vs `|DFT|`+entropy.

use rand::{rngs::StdRng, SeedableRng};

mod common;
use common::vision::{make_set, phase_features, phase_histogram, train_linear, PhaseFeature};

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn phase_pool_entropy_vs_raw_histogram() {
    let (mut raw, mut phase, mut phent) = (Vec::new(), Vec::new(), Vec::new());
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let (x_tr, y_tr) = make_set(400, &mut rng);
        let (x_te, y_te) = make_set(160, &mut rng);
        let (htr, hte) = (phase_histogram(&x_tr, 400), phase_histogram(&x_te, 160));
        let ev = |mode: PhaseFeature| {
            let (ftr, dim) = phase_features(&htr, 400, mode);
            let (fte, _) = phase_features(&hte, 160, mode);
            train_linear(&ftr, dim, &y_tr, &fte, &y_te, seed + 1)
        };
        let (r, p, pe) = (
            ev(PhaseFeature::Raw),
            ev(PhaseFeature::Dft),
            ev(PhaseFeature::DftEntropy),
        );
        eprintln!("seed {seed}: raw-hist {r:.3}  phase-pool {p:.3}  phase+entropy {pe:.3}");
        raw.push(r);
        phase.push(p);
        phent.push(pe);
    }
    let (rm, pm, pem) = (median(raw), median(phase), median(phent));
    eprintln!("Nagare CV — quaternion-phase pooling + entropy feedback (rotated shapes):");
    eprintln!("  raw histogram {rm:.3}   phase-pool |DFT| {pm:.3}   phase-pool + entropy {pem:.3}");
    eprintln!(
        "  verdict: phase-pool {} raw (Δ {:+.3}); entropy {} phase-pool alone (Δ {:+.3})",
        if pm > rm + 0.01 { "beats" } else { "matches" },
        pm - rm,
        if pem > pm + 0.01 {
            "helps"
        } else if pem < pm - 0.01 {
            "hurts"
        } else {
            "neutral"
        },
        pem - pm
    );
    assert!(
        pm > 0.35 && pem > 0.35,
        "phase-pool failed to learn: {pm:.3}/{pem:.3}"
    );
}
