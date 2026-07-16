//! Deep-holonomy dissociation — the empirical crux of "one step to deep-representation
//! learning". No autograd; the whole pipeline is closed-form.
//!
//! Pipeline: `RotorMeshNet` (deep learned rotors + mesh mix) → readout → logistic.
//! Two readouts on the SAME learned net:
//!   ENTROPY — normalized spectral eigen-entropy of the output field (arrangement-
//!             sensitive; `spectral_reg_value_grad`, HSiKAN's entropy op).
//!   MEAN    — the mean 3-vector of the output field (arrangement-blind).
//! Trained by the closed-form chain (BCE → logistic → readout backward →
//! `RotorMeshNet::backward` → bivectors); no autograd tape.
//!
//! Task (zero-mean by construction, so a *raw* mean is chance): a ring mesh; class 0 =
//! a COHERENT twist (v_i = R(θ·i)·u, anisotropic/structured covariance), class 1 =
//! ISOTROPIC random directions. Reading the coherence needs the arrangement, not the
//! mean.
//!
//! 2×2: {deep L=3, shallow L=1} × {entropy, mean} readout, multi-seed → held-out AUROC.
//! A hard FD gate on the end-to-end gradient runs first (a wrong gradient = a phantom).
//!
//! Run: `cargo run --release --example holonomy_deep_dissociation [-- --fdcheck]`

use holonomy_learn::{spectral_reg_value_grad, MeshTopology, RotorMeshNet, SpectralEntropyConfig};

const N: usize = 12; // ring nodes
const D: usize = 3;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32) / 4294967296.0
    }
    fn g(&mut self) -> f32 {
        (-2.0 * self.f().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * self.f()).cos()
    }
}

/// Ring mesh: N nodes, N triangular hyperedges {i, i+1, i+2} (mod N), unit signs,
/// degree-normalized scale.
fn ring_mesh() -> MeshTopology {
    let (mut cycles, mut signs) = (Vec::new(), Vec::new());
    for i in 0..N {
        for j in 0..3 {
            cycles.push(((i + j) % N) as u32);
            signs.push(1.0);
        }
    }
    let scale = vec![1.0 / 3.0f32.sqrt(); N]; // each node in 3 edges
    MeshTopology::new(cycles, signs, scale, N, 3)
}

/// One sample field `(N,3)`, centered to zero mean. label 0 = coherent twist, 1 = isotropic.
fn gen_sample(rng: &mut Rng, label: u8) -> Vec<f32> {
    let mut v = vec![0.0f32; N * D];
    if label == 0 {
        // coherent twist about a random axis: v_i = R(theta*i) u
        let axis = {
            let a = [rng.g(), rng.g(), rng.g()];
            let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-6);
            [a[0] / n, a[1] / n, a[2] / n]
        };
        let u = [rng.g(), rng.g(), rng.g()];
        let theta = 0.4 + 0.3 * rng.f();
        for i in 0..N {
            let ang = theta * i as f32;
            let r = rodrigues(axis, ang, u);
            for c in 0..D {
                v[i * D + c] = r[c] + 0.15 * rng.g();
            }
        }
    } else {
        for x in v.iter_mut() {
            *x = rng.g();
        }
    }
    // center to zero mean per channel
    for c in 0..D {
        let m: f32 = (0..N).map(|i| v[i * D + c]).sum::<f32>() / N as f32;
        for i in 0..N {
            v[i * D + c] -= m;
        }
    }
    v
}

/// Rotate `v` about unit `axis` by `ang` (Rodrigues).
fn rodrigues(axis: [f32; 3], ang: f32, v: [f32; 3]) -> [f32; 3] {
    let (c, s) = (ang.cos(), ang.sin());
    let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    [
        v[0] * c + cross[0] * s + axis[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + axis[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + axis[2] * dot * (1.0 - c),
    ]
}

fn entropy_cfg() -> SpectralEntropyConfig {
    // reg = lam_eff·H_norm (pass-through entropy readout): lam_a=0, lam_b=1, lam_kl=0
    SpectralEntropyConfig {
        lam_0: 1.0,
        lam_a: 0.0,
        lam_b: 1.0,
        lam_kl: 0.0,
        ..SpectralEntropyConfig::default()
    }
}

/// Readout features from the output field. entropy: [H_norm]; mean: [x̄0,x̄1,x̄2].
fn readout(field: &[f32], use_entropy: bool, cfg: &SpectralEntropyConfig) -> Vec<f32> {
    if use_entropy {
        let (h, _g, _) = spectral_reg_value_grad(field, N, D, cfg, 1.0);
        vec![h]
    } else {
        (0..D)
            .map(|c| (0..N).map(|i| field[i * D + c]).sum::<f32>() / N as f32)
            .collect()
    }
}

/// Gradient of the readout w.r.t. the field, given upstream grad on the features.
fn readout_backward(
    field: &[f32],
    use_entropy: bool,
    cfg: &SpectralEntropyConfig,
    grad_feat: &[f32],
) -> Vec<f32> {
    if use_entropy {
        let (_h, g, _) = spectral_reg_value_grad(field, N, D, cfg, 1.0);
        g.iter().map(|x| x * grad_feat[0]).collect() // scalar feature
    } else {
        let mut gf = vec![0.0f32; N * D];
        for c in 0..D {
            for i in 0..N {
                gf[i * D + c] = grad_feat[c] / N as f32;
            }
        }
        gf
    }
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Held-out AUROC (rank-based, ties averaged).
fn auroc(scores: &[f32], labels: &[u8]) -> f64 {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank = vec![0.0f64; n];
    let (mut i, mut r) = (0usize, 0usize);
    while i < n {
        let mut j = i;
        while j + 1 < n && scores[idx[j + 1]] == scores[idx[i]] {
            j += 1;
        }
        let avg = (r + (j - i) + r) as f64 / 2.0 + 1.0; // average 1-based rank
        for k in i..=j {
            rank[idx[k]] = avg;
        }
        r += j - i + 1;
        i = j + 1;
    }
    let (np, nn) = (
        labels.iter().filter(|&&l| l == 1).count(),
        labels.iter().filter(|&&l| l == 0).count(),
    );
    if np == 0 || nn == 0 {
        return 0.5;
    }
    let sum_pos: f64 = (0..n).filter(|&k| labels[k] == 1).map(|k| rank[k]).sum();
    (sum_pos - (np * (np + 1)) as f64 / 2.0) / (np * nn) as f64
}

/// Train one (depth, readout) config; return (final held-out AUROC, [(epoch, AUROC)]
/// at checkpoints — the convergence curve).
fn run_config(
    depth: usize,
    use_entropy: bool,
    seed: u64,
    cfg: &SpectralEntropyConfig,
    lr: f32,
    epochs: usize,
) -> (f64, Vec<(usize, f64)>) {
    let topo = ring_mesh();
    let mut rng = Rng(101 + seed);
    // data
    let gen_set = |rng: &mut Rng, n: usize| -> (Vec<Vec<f32>>, Vec<u8>) {
        let mut xs = Vec::new();
        let mut ys = Vec::new();
        for k in 0..n {
            let lab = (k % 2) as u8;
            xs.push(gen_sample(rng, lab));
            ys.push(lab);
        }
        (xs, ys)
    };
    let (xtr, ytr) = gen_set(&mut rng, 120);
    let (xte, yte) = gen_set(&mut rng, 120);

    // params: net bivecs + classifier (w, b)
    let mut bivecs: Vec<Vec<f32>> = (0..depth)
        .map(|_| (0..N * D).map(|_| 0.1 * rng.g()).collect())
        .collect();
    let n_feat = if use_entropy { 1 } else { D };
    let mut w = vec![0.0f32; n_feat];
    let mut b = 0.0f32;

    let eval = |bivecs: &[Vec<f32>], w: &[f32], b: f32| -> f64 {
        let mut scores = Vec::new();
        for x in &xte {
            let net = RotorMeshNet::new(&topo, bivecs.to_vec());
            let (out, _) = net.forward(x);
            let feat = readout(&out, use_entropy, cfg);
            scores.push(feat.iter().zip(w).map(|(f, wi)| f * wi).sum::<f32>() + b);
        }
        let a = auroc(&scores, &yte);
        a.max(1.0 - a) // symmetric under label flip; report separability
    };
    let checkpoints = [1usize, 2, 3, 5, 10, 20, 50, 100, 200];
    let mut curve = Vec::new();

    for ep in 0..epochs {
        let mut gb: Vec<Vec<f32>> = bivecs.iter().map(|v| vec![0.0f32; v.len()]).collect();
        let mut gw = vec![0.0f32; n_feat];
        let mut gbias = 0.0f32;
        let mut loss = 0.0f32;
        for (x, &y) in xtr.iter().zip(&ytr) {
            let net = RotorMeshNet::new(&topo, bivecs.clone());
            let (out, cache) = net.forward(x);
            let feat = readout(&out, use_entropy, cfg);
            let logit = feat.iter().zip(&w).map(|(f, wi)| f * wi).sum::<f32>() + b;
            let p = sigmoid(logit);
            loss -= (y as f32) * p.max(1e-7).ln() + (1.0 - y as f32) * (1.0 - p).max(1e-7).ln();
            let dlogit = p - y as f32; // BCE grad
            for c in 0..n_feat {
                gw[c] += dlogit * feat[c];
            }
            gbias += dlogit;
            let grad_feat: Vec<f32> = w.iter().map(|wi| dlogit * wi).collect();
            let grad_field = readout_backward(&out, use_entropy, cfg, &grad_feat);
            let (gb_net, _gv0) = net.backward(&cache, &grad_field);
            for l in 0..depth {
                for i in 0..N * D {
                    gb[l][i] += gb_net[l][i];
                }
            }
        }
        let m = xtr.len() as f32;
        for l in 0..depth {
            for i in 0..N * D {
                bivecs[l][i] -= lr * gb[l][i] / m;
            }
        }
        for c in 0..n_feat {
            w[c] -= lr * gw[c] / m;
        }
        b -= lr * gbias / m;
        let _ = loss;
        if checkpoints.contains(&(ep + 1)) {
            curve.push((ep + 1, eval(&bivecs, &w, b)));
        }
    }
    (eval(&bivecs, &w, b), curve)
}

/// Hard FD gate: analytic end-to-end grad w.r.t. a bivec entry == finite difference.
fn fd_gate(cfg: &SpectralEntropyConfig) {
    let topo = ring_mesh();
    let mut rng = Rng(3);
    for &use_entropy in &[true, false] {
        let depth = 2;
        let bivecs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..N * D).map(|_| 0.2 * rng.g()).collect())
            .collect();
        let x = gen_sample(&mut rng, 0);
        let (w, b) = (vec![0.7f32; if use_entropy { 1 } else { D }], 0.1f32);
        let y = 1.0f32;
        let loss = |bv: &[Vec<f32>]| -> f32 {
            let net = RotorMeshNet::new(&topo, bv.to_vec());
            let (out, _) = net.forward(&x);
            let feat = readout(&out, use_entropy, cfg);
            let logit = feat.iter().zip(&w).map(|(f, wi)| f * wi).sum::<f32>() + b;
            let p = sigmoid(logit);
            -(y * p.max(1e-7).ln() + (1.0 - y) * (1.0 - p).max(1e-7).ln())
        };
        // analytic
        let net = RotorMeshNet::new(&topo, bivecs.clone());
        let (out, cache) = net.forward(&x);
        let feat = readout(&out, use_entropy, cfg);
        let logit = feat.iter().zip(&w).map(|(f, wi)| f * wi).sum::<f32>() + b;
        let dlogit = sigmoid(logit) - y;
        let grad_feat: Vec<f32> = w.iter().map(|wi| dlogit * wi).collect();
        let grad_field = readout_backward(&out, use_entropy, cfg, &grad_feat);
        let (gb, _) = net.backward(&cache, &grad_field);
        // FD a few entries
        let eps = 1e-3f32;
        let mut max_err = 0.0f32;
        for l in 0..depth {
            for i in [0usize, 5, 11, 20, 33] {
                let (mut bp, mut bm) = (bivecs.clone(), bivecs.clone());
                bp[l][i] += eps;
                bm[l][i] -= eps;
                let fd = (loss(&bp) - loss(&bm)) / (2.0 * eps);
                max_err = max_err.max((fd - gb[l][i]).abs());
            }
        }
        println!(
            "  FD gate [{}]: max |analytic - fd| = {:.2e}  {}",
            if use_entropy { "entropy" } else { "mean" },
            max_err,
            if max_err < 5e-2 { "PASS" } else { "FAIL" }
        );
        assert!(
            max_err < 5e-2,
            "gradient FD gate failed — training numbers would be a phantom"
        );
    }
}

fn main() {
    let cfg = entropy_cfg();
    let fdcheck = std::env::args().any(|a| a == "--fdcheck");
    println!("== FD gate (end-to-end closed-form gradient) ==");
    fd_gate(&cfg);
    if fdcheck {
        return;
    }
    println!("\n== 2x2 dissociation (5 seeds, lr=0.05 x 200 ep, held-out AUROC median) ==");
    println!("  {:<16} {:>10} {:>10}", "", "entropy", "mean");
    for (name, depth) in [("deep (L=3)", 3usize), ("shallow (L=1)", 1usize)] {
        let med = |ent: bool| -> f64 {
            let mut v: Vec<f64> = (0..5)
                .map(|s| run_config(depth, ent, s, &cfg, 0.05, 200).0)
                .collect();
            v.sort_by(|a, b| a.total_cmp(b));
            v[2]
        };
        println!("  {:<16} {:>10.3} {:>10.3}", name, med(true), med(false));
    }

    // "how instantaneous?" — held-out AUROC vs number of passes (deep+entropy), a few LRs.
    println!("\n== convergence: deep+entropy held-out AUROC vs #passes (seed 0) ==");
    for lr in [0.05f32, 0.5, 2.0] {
        let (_final, curve) = run_config(3, true, 0, &cfg, lr, 200);
        let pts: Vec<String> = curve.iter().map(|(e, a)| format!("{e}:{a:.3}")).collect();
        println!("  lr={lr:<4}  {}", pts.join("  "));
    }
}
