//! Nagare CV — **entropy feedback on a headroom task**. The shape task saturated at 0.97, hiding
//! whether phase **entropy** adds over `|DFT|`. Here the classes are separated by *orientation
//! disorder itself*: class `c` is a texture built from `n_ori(c)` oriented gratings at **random**
//! angles (+noise). More orientations → more spread → higher phase entropy. The random angles make
//! the task inherently rotation-invariant (a raw histogram can't do it; the shift-invariant
//! `|DFT|` can), and — crucially — a *linear* classifier cannot compute entropy (`−Σp log p`, a
//! nonlinear function of the histogram) from `|DFT|` magnitudes, so the **explicit** entropy
//! feature is a genuinely new invariant it otherwise lacks.
//!
//! Ablation (reported, §3), 4 seeds: raw histogram vs phase-pool `|DFT|` vs `|DFT|`+entropy.
//! Does entropy help where the task has headroom?

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f32::consts::PI;

mod common;
use common::vision::{phase_features, phase_histogram, train_linear, PhaseFeature, G, K};

/// Texture with `n_ori` gratings at random orientations (+noise) → flat `G*G` in ~[-1,1].
fn render_texture(n_ori: usize, rng: &mut StdRng) -> Vec<f32> {
    let freq = 6.0f32;
    let oris: Vec<(f32, f32)> = (0..n_ori)
        .map(|_| {
            let a = rng.random::<f32>() * PI; // random orientation
            (a, rng.random::<f32>() * 2.0 * PI) // (angle, phase)
        })
        .collect();
    let mut img = vec![0.0f32; G * G];
    for i in 0..G {
        for j in 0..G {
            let cy = (i as f32 + 0.5) / G as f32 * 2.0 - 1.0;
            let cx = (j as f32 + 0.5) / G as f32 * 2.0 - 1.0;
            let mut v = 0.0f32;
            for &(a, ph) in &oris {
                v += (freq * (cx * a.cos() + cy * a.sin()) + ph).sin();
            }
            v = v / n_ori as f32 + 0.35 * (rng.random::<f32>() * 2.0 - 1.0);
            img[i * G + j] = v.clamp(-1.0, 1.0);
        }
    }
    img
}

/// `n` textures; class `c ∈ 0..K` uses `N_ORI[c]` orientations (increasing disorder).
fn make_texture_set(n: usize, rng: &mut StdRng) -> (Vec<f32>, Vec<usize>) {
    const N_ORI: [usize; 4] = [1, 2, 3, 6]; // class → orientation count (K=4)
    let mut x = vec![0.0f32; n * G * G];
    let mut y = vec![0usize; n];
    for s in 0..n {
        let c = rng.random_range(0..K);
        y[s] = c;
        x[s * G * G..(s + 1) * G * G].copy_from_slice(&render_texture(N_ORI[c], rng));
    }
    (x, y)
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

#[test]
fn entropy_feedback_vs_dft_on_orientation_disorder() {
    let (mut raw, mut phase, mut phent) = (Vec::new(), Vec::new(), Vec::new());
    for seed in 0..4u64 {
        let mut rng = StdRng::seed_from_u64(seed + 100);
        let (x_tr, y_tr) = make_texture_set(400, &mut rng);
        let (x_te, y_te) = make_texture_set(200, &mut rng);
        let (htr, hte) = (phase_histogram(&x_tr, 400), phase_histogram(&x_te, 200));
        let ev = |mode: PhaseFeature| {
            let (ftr, dim) = phase_features(&htr, 400, mode);
            let (fte, _) = phase_features(&hte, 200, mode);
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
    eprintln!("Nagare CV — entropy feedback on orientation-disorder textures (headroom task):");
    eprintln!("  raw histogram {rm:.3}   phase-pool |DFT| {pm:.3}   phase-pool + entropy {pem:.3}");
    eprintln!(
        "  verdict: |DFT| leaves headroom ({pm:.3} < ~1.0); entropy feedback {} — Δ over |DFT| {:+.3}",
        if pem > pm + 0.01 {
            "HELPS"
        } else if pem < pm - 0.01 {
            "hurts"
        } else {
            "neutral"
        },
        pem - pm
    );
    // Gate: the invariant arms learn above chance; the entropy-vs-|DFT| delta is the measurement.
    assert!(
        pm > 0.3 && pem > 0.3,
        "phase-pool failed to learn textures: {pm:.3}/{pem:.3}"
    );
}
