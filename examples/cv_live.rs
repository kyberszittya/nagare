//! Nagare CV — **live demo**: watch three models classify a real image as it rotates through 360°.
//! Trained on upright images; then for one sample per class we rotate it frame-by-frame and record
//! each model's prediction. The spatial arm (pixel-linear) flips as the digit turns; the
//! rotation-invariant phase-pool holds; the spatial phase map (the lift) sits between.
//!
//! Dumps `frames.bin` (u8 rotated images) + `meta.txt` (labels + per-frame predictions) into `--out`,
//! which `scripts/dev/render_cv_live.py` turns into an animated GIF.
//!
//! Run: `cargo run --release --example cv_live -- --dataset mnist --data ~/nagare_data/mnist --out /tmp/cv_live`

use std::io::Write;
use std::path::Path;

use holonomy_learn::{
    accuracy_k, cross_entropy_k_backward, linear_forward, rotate_image, spatial_phase_features,
    LinearLayer, PhaseFeature,
};

fn arg(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}

const FRAMES: usize = 24; // 15° steps
const B: usize = 18;

/// Standardised linear classifier on fixed features: (layer, mu, sd).
struct Clf {
    layer: LinearLayer,
    mu: Vec<f32>,
    sd: Vec<f32>,
    dim: usize,
}
impl Clf {
    fn fit(feat: &[f32], dim: usize, k: usize, y: &[usize]) -> Clf {
        let n = y.len();
        let (mut mu, mut sd) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        for r in feat.chunks(dim) {
            for j in 0..dim {
                mu[j] += r[j] / n as f32;
            }
        }
        for r in feat.chunks(dim) {
            for j in 0..dim {
                sd[j] += (r[j] - mu[j]).powi(2) / n as f32;
            }
        }
        for s in &mut sd {
            *s = s.sqrt() + 1e-6;
        }
        let ftr: Vec<f32> = feat
            .chunks(dim)
            .flat_map(|r| (0..dim).map(|j| (r[j] - mu[j]) / sd[j]))
            .collect();
        let mut layer = LinearLayer::new(dim, k, 7);
        for _ in 0..250 {
            let gl = cross_entropy_k_backward(&linear_forward(&layer, &ftr), y, n, k);
            let (_g, grad) = holonomy_learn::linear_backward(&layer, &ftr, &gl);
            for (w, g) in layer.w.iter_mut().zip(&grad.w) {
                *w -= 0.5 * g;
            }
            for (b, g) in layer.b.iter_mut().zip(&grad.b) {
                *b -= 0.5 * g;
            }
        }
        Clf { layer, mu, sd, dim }
    }
    /// Predict one sample's class from its raw (un-standardised) feature vector.
    fn predict(&self, feat: &[f32]) -> usize {
        let z: Vec<f32> = (0..self.dim)
            .map(|j| (feat[j] - self.mu[j]) / self.sd[j])
            .collect();
        let l = linear_forward(&self.layer, &z);
        (0..l.len()).max_by(|&a, &b| l[a].total_cmp(&l[b])).unwrap()
    }
}

fn load(dir: &Path, ds: &str, cap: usize, train: bool) -> (Vec<f32>, Vec<usize>, usize) {
    let (imf, laf) = if ds == "mnist" {
        if train {
            ("train-images-idx3-ubyte", "train-labels-idx1-ubyte")
        } else {
            ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte")
        }
    } else if train {
        ("train-images.bin", "train-labels.bin")
    } else {
        ("test-images.bin", "test-labels.bin")
    };
    let b = std::fs::read(dir.join(imf)).expect("images");
    let (n, g, off) = if ds == "mnist" {
        (
            u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize,
            u32::from_be_bytes([b[8], b[9], b[10], b[11]]) as usize,
            16,
        )
    } else {
        let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
        (rd(0), rd(4), 12)
    };
    let n = n.min(cap);
    let x = b[off..off + n * g * g]
        .iter()
        .map(|&p| p as f32 / 255.0 * 2.0 - 1.0)
        .collect();
    let lb = std::fs::read(dir.join(laf)).expect("labels");
    let lo = if ds == "mnist" { 8 } else { 4 };
    let y = lb[lo..lo + n].iter().map(|&l| l as usize).collect();
    (x, y, g)
}

fn main() {
    let ds = arg("--dataset").unwrap_or_else(|| "mnist".into());
    let dir = arg("--data").expect("--data");
    let out = arg("--out").unwrap_or_else(|| "/tmp/cv_live".into());
    let d = Path::new(&dir);
    let (x_tr, y_tr, g) = load(d, &ds, 8000, true);
    let (x_te, y_te, _) = load(d, &ds, 4000, false);
    let k = y_tr.iter().chain(&y_te).copied().max().unwrap() + 1;
    let nt = y_tr.len();
    let rr = if g.is_multiple_of(7) { 7 } else { 8 }; // spatial-phase grid for the "lift" arm

    // Three models: pixel-linear (spatial), phase-pool R=1 (invariant), spatial-phase R (lift).
    let names = ["pixel", "phase-R1", &format!("sphase-R{rr}")];
    let sp =
        |x: &[f32], n: usize, r: usize| spatial_phase_features(x, n, g, r, B, PhaseFeature::Dft);
    let clf_px = Clf::fit(&x_tr, g * g, k, &y_tr);
    let (p1, d1) = sp(&x_tr, nt, 1);
    let clf_p1 = Clf::fit(&p1, d1, k, &y_tr);
    let (pr, dr) = sp(&x_tr, nt, rr);
    let clf_pr = Clf::fit(&pr, dr, k, &y_tr);

    // One sample per class from the test set.
    let mut samples: Vec<usize> = Vec::new();
    for cls in 0..k {
        if let Some(i) = (0..y_te.len()).find(|&i| y_te[i] == cls) {
            samples.push(i);
        }
    }
    let m = samples.len();

    std::fs::create_dir_all(&out).unwrap();
    let mut frames = Vec::<u8>::new();
    let mut preds = String::new(); // "s f p_px p_p1 p_pr"
    for (si, &idx) in samples.iter().enumerate() {
        let img = &x_te[idx * g * g..(idx + 1) * g * g];
        for f in 0..FRAMES {
            let theta = f as f32 / FRAMES as f32 * std::f32::consts::TAU;
            let rot = rotate_image(img, g, theta);
            frames.extend(
                rot.iter()
                    .map(|&v| (((v + 1.0) * 0.5).clamp(0.0, 1.0) * 255.0) as u8),
            );
            let (pp1, _) = sp(&rot, 1, 1);
            let (ppr, _) = sp(&rot, 1, rr);
            let (a, b, c) = (
                clf_px.predict(&rot),
                clf_p1.predict(&pp1),
                clf_pr.predict(&ppr),
            );
            preds.push_str(&format!("{si} {f} {a} {b} {c}\n"));
        }
    }
    std::fs::write(Path::new(&out).join("frames.bin"), &frames).unwrap();
    let mut meta = std::fs::File::create(Path::new(&out).join("meta.txt")).unwrap();
    writeln!(meta, "{m} {FRAMES} {g} {}", names.len()).unwrap();
    writeln!(meta, "{}", names.join(",")).unwrap();
    writeln!(
        meta,
        "{}",
        samples
            .iter()
            .map(|&i| y_te[i].to_string())
            .collect::<Vec<_>>()
            .join(",")
    )
    .unwrap();
    meta.write_all(preds.as_bytes()).unwrap();

    // Sanity: report upright accuracy of the three models (context in the demo caption).
    let up = |c: &Clf, feat: &[f32], dim: usize| {
        accuracy_k(
            &linear_forward(
                &c.layer,
                &feat
                    .chunks(dim)
                    .flat_map(|r| (0..dim).map(|j| (r[j] - c.mu[j]) / c.sd[j]))
                    .collect::<Vec<_>>(),
            ),
            &y_te,
            y_te.len(),
            k,
        )
    };
    let (fp1, _) = sp(&x_te, y_te.len(), 1);
    let (fpr, _) = sp(&x_te, y_te.len(), rr);
    println!(
        "{ds}: {m} samples × {FRAMES} frames → {out}. upright acc: pixel {:.3} / phase-R1 {:.3} / sphase-R{rr} {:.3}",
        up(&clf_px, &x_te, g * g), up(&clf_p1, &fp1, d1), up(&clf_pr, &fpr, dr)
    );
}
