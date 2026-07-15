//! Neocognitron N2b — the harder task that FORCES a learned oriented S-cell, no
//! autograd. Detect a low-contrast oriented BAR buried in **energy-matched**
//! isotropic noise (label 1) vs noise only (label 0). Energy is matched by
//! construction, so an isotropic/energy filter is at chance — only a *learned
//! oriented* filter that integrates along the coherent line separates them, and
//! the `group_pool` C-cell makes that detection rotation-invariant.
//!
//! Stack (LEARNABLE this time): `conv2d (S, learned) → oriented vectors →
//! group_pool (C-cell) → mean → linear → BCE`. A/B: the C-cell's group is C₈
//! (rotation-invariant) vs C₁ (`--c1`, orientation-specific). Train on a SINGLE
//! orientation, test AUROC on the HELD-OUT C₈ orbit.
//!
//! MEASURED FINDING (honest): the task DOES force oriented features — the
//! energy-only baseline is at chance (~0.47) while the learned oriented S-cell
//! reaches ~0.73 AUROC. BUT C₈ ≈ C₁: with a *learnable* conv the network
//! discovers an orientation-AGNOSTIC line/coherence detector whose global-mean
//! readout does not engage the group-pool orientation machinery, so the explicit
//! C-cell rotation-invariance is redundant. This CONFIRMS the N2 design choice
//! (fix the S-cell to Sobel to isolate the C-cell): the explicit invariance is
//! load-bearing only when the representation is CONSTRAINED to be oriented.
//!
//! Run: `cargo run --release --example neocognitron_aniso -- [--c1] [out.json]`

use holonomy_learn::{
    adam_step, auroc, conv2d_backward, conv2d_forward, group_pool_backward, group_pool_forward,
    linear_backward, linear_forward, AdamState, ConvLayer, ConvShape, DihedralGroup, LinearLayer,
};
use std::f32::consts::PI;
use std::io::Write;

const G: usize = 20;
const GG: usize = G * G;
const CBAR: f32 = 0.5; // bar contrast (weak)
const SIG: f32 = 0.6; // base noise std
const NBAR: f32 = 15.0; // approx bar pixel count (for energy matching)

fn flag(name: &str) -> bool {
    std::env::args().any(|a| a == name)
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
    fn gauss(&mut self) -> f32 {
        // Box–Muller.
        let (u1, u2) = (self.f().max(1e-7), self.f());
        (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos()
    }
}

/// A sample: bar+noise (label 1) or noise-only (label 0), energy-matched.
fn sample(theta: f32, bar: bool, rng: &mut Rng) -> Vec<f32> {
    // noise-only variance is boosted so E[energy] matches the bar class.
    let sig_b = (SIG * SIG + NBAR * CBAR * CBAR / GG as f32).sqrt();
    let sig = if bar { SIG } else { sig_b };
    let mut img: Vec<f32> = (0..GG).map(|_| sig * rng.gauss()).collect();
    if bar {
        let cc = (G - 1) as f32 / 2.0;
        let (ct, st) = (theta.cos(), theta.sin());
        for r in 0..G {
            for col in 0..G {
                let (dx, dy) = (col as f32 - cc, r as f32 - cc);
                if (-dx * st + dy * ct).abs() <= 1.0 && (dx * ct + dy * st).abs() <= 7.0 {
                    img[r * G + col] += CBAR;
                }
            }
        }
    }
    img
}

fn to_vectors(y: &[f32]) -> Vec<f32> {
    let mut v = vec![0.0f32; GG * 3];
    for p in 0..GG {
        v[p * 3] = y[p];
        v[p * 3 + 1] = y[GG + p];
    }
    v
}

fn main() {
    let c1 = flag("--c1");
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/neocognitron-aniso.json".into());
    let group = if c1 {
        DihedralGroup::new(1, false)
    } else {
        DihedralGroup::new(8, false)
    };
    let tau = 0.3f32;
    let s = ConvShape {
        c_in: 1,
        h: G,
        w: G,
        pad: 1,
    };

    // Train on a SINGLE orientation: C_1 (orientation-specific) then cannot
    // interpolate to unseen orientations, while C_8 pools over the orbit → the
    // held-out group-element orientations map onto the trained one (invariant).
    let train_ang: Vec<f32> = [0.0f32].iter().map(|d| d * PI / 180.0).collect();
    let test_ang: Vec<f32> = [45.0f32, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0]
        .iter()
        .map(|d| d * PI / 180.0)
        .collect();

    let mut conv = ConvLayer::new(1, 2, 3, 3, 5);
    let mut filt = vec![0.5f32, -0.2, 0.1];
    let mut head = LinearLayer::new(1, 1, 9);
    let (mut scw, mut scb) = (AdamState::new(conv.w.len()), AdamState::new(conv.b.len()));
    let mut sfilt = AdamState::new(3);
    let (mut shw, mut shb) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));

    let feat_of = |conv: &ConvLayer,
                   filt: &[f32],
                   img: &[f32]|
     -> (f32, ConvLayer, holonomy_learn::GroupPoolOut, Vec<f32>) {
        let (y, _, _) = conv2d_forward(conv, img, s);
        let v = to_vectors(&y);
        let gp = group_pool_forward(&v, group, filt, tau);
        (gp.resp.iter().sum::<f32>() / GG as f32, conv.clone(), gp, y)
    };

    let mut rng = Rng(999);
    for _ in 0..1500 {
        // one bar + one noise sample per step (balanced), random train orientation.
        for bar in [true, false] {
            let th = train_ang[(rng.f() * train_ang.len() as f32) as usize % train_ang.len()];
            let img = sample(th, bar, &mut rng);
            let (f, _c, gp, _y) = feat_of(&conv, &filt, &img);
            let logit = linear_forward(&head, &[f])[0];
            let p = 1.0 / (1.0 + (-logit).exp());
            let gl = vec![p - if bar { 1.0 } else { 0.0 }];
            let (gfeat, ghead) = linear_backward(&head, &[f], &gl);
            let gresp = vec![gfeat[0] / GG as f32; GG];
            let (gv, gfilt) = group_pool_backward(&gp, group, &filt, tau, &gresp);
            let mut gy = vec![0.0f32; 2 * GG];
            for pp in 0..GG {
                gy[pp] = gv[pp * 3];
                gy[GG + pp] = gv[pp * 3 + 1];
            }
            let (_gx, gconv) = conv2d_backward(&conv, &img, s, &gy);
            adam_step(&mut conv.w, &gconv.w, &mut scw, 0.02);
            adam_step(&mut conv.b, &gconv.b, &mut scb, 0.02);
            adam_step(&mut filt, &gfilt, &mut sfilt, 0.02);
            adam_step(&mut head.w, &ghead.w, &mut shw, 0.02);
            adam_step(&mut head.b, &ghead.b, &mut shb, 0.02);
        }
    }

    // Evaluate AUROC (bar vs noise) on train orientations and held-out orientations.
    let eval = |angs: &[f32], rng: &mut Rng| -> f64 {
        let (mut sc, mut lb) = (Vec::new(), Vec::new());
        for &th in angs {
            for _ in 0..40 {
                for bar in [true, false] {
                    let img = sample(th, bar, rng);
                    let (f, _, _, _) = feat_of(&conv, &filt, &img);
                    sc.push(linear_forward(&head, &[f])[0]);
                    lb.push(bar as u8);
                }
            }
        }
        auroc(&sc, &lb)
    };
    let train_auc = eval(&train_ang, &mut rng);
    let test_auc = eval(&test_ang, &mut rng);
    // energy baseline: does mean image energy separate the classes? (should be ~0.5)
    let mut ergn = Rng(7);
    let (mut es, mut el) = (Vec::new(), Vec::new());
    for &th in &test_ang {
        for _ in 0..40 {
            for bar in [true, false] {
                let img = sample(th, bar, &mut ergn);
                es.push(img.iter().map(|x| x * x).sum::<f32>());
                el.push(bar as u8);
            }
        }
    }
    let energy_auc = auroc(&es, &el);

    println!(
        "neocognitron aniso ({}): train AUROC {train_auc:.3}  HELD-OUT-rotation AUROC {test_auc:.3}  (energy-only baseline {energy_auc:.3})",
        if c1 { "C_1 (orientation-specific)" } else { "C_8 (rotation-invariant)" }
    );
    let json = format!(
        "{{\n  \"c1\": {c1},\n  \"train_auc\": {train_auc:.4},\n  \"test_auc\": {test_auc:.4},\n  \"energy_auc\": {energy_auc:.4}\n}}\n"
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
