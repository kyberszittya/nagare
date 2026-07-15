//! E1 — the evolvent on STATIC datasets (generated + real, classification +
//! regression). No autograd. On a fixed random-Fourier-feature (RFF) basis, the
//! evolvent head is exact online ridge in ONE PASS (no lr/epoch tuning); we ask
//! whether it MATCHES multi-epoch backprop at a fraction of the training effort.
//!
//! Three arms, same RFF basis for the two linear ones:
//!   A EVOLVENT   — RFF + EvolventHead (RLS, lambda=1): ONE pass, closed-form.
//!   B ONLINE-SGD — RFF + linear + Adam: ONE pass (isolates RLS vs SGD).
//!   C BACKPROP   — MLP (raw features -> tanh -> out) + Adam: EPOCHS passes.
//! Regression: R^2 / RMSE. Classification: one-hot least-squares -> argmax accuracy.
//!
//! Run: `cargo run --release --example evolvent_bench -- [--seed=N]`

use holonomy_learn::{
    adam_step, linear_backward, linear_forward, load_csv, load_csv_regression, r2_score, AdamState,
    EvolventHead, LinearLayer, Tabular, TabularReg,
};
use std::f32::consts::TAU;
use std::io::Write;
use std::time::Instant;

const M: usize = 256; // RFF features
const H: usize = 64; // MLP hidden
const EPOCHS: usize = 200; // backprop epochs (the "slow" arm)

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 33) as f32) / (u32::MAX as f32)
    }
    fn u(&mut self) -> f32 {
        2.0 * self.f() - 1.0
    }
    fn gauss(&mut self) -> f32 {
        (-2.0 * self.f().max(1e-7).ln()).sqrt() * (TAU * self.f()).cos()
    }
}

/// Fixed RFF map phi(x) = sqrt(2/M) cos(Wx+b), approximating an RBF kernel.
struct Rff {
    w: Vec<f32>,
    b: Vec<f32>,
    d: usize,
}
impl Rff {
    fn new(d: usize, gamma: f32, rng: &mut Rng) -> Self {
        Rff {
            w: (0..M * d).map(|_| gamma * rng.gauss()).collect(),
            b: (0..M).map(|_| TAU * rng.f()).collect(),
            d,
        }
    }
    fn phi(&self, x: &[f32]) -> Vec<f32> {
        let s = (2.0 / M as f32).sqrt();
        (0..M)
            .map(|j| {
                let z: f32 = (0..self.d)
                    .map(|k| self.w[j * self.d + k] * x[k])
                    .sum::<f32>()
                    + self.b[j];
                s * z.cos()
            })
            .collect()
    }
}

fn shuffle(n: usize, rng: &mut Rng) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.f() * (i + 1) as f32) as usize % (i + 1);
        idx.swap(i, j);
    }
    idx
}

/// Regression bench on `(x,y)` rows (features flat n*d). Returns test R2 [A,B,C].
fn bench_reg(name: &str, x: &[f32], y: &[f32], n: usize, d: usize, seed: u64) -> [f64; 3] {
    let mut rng = Rng(1 + seed);
    let ntr = n * 3 / 4;
    let perm = shuffle(n, &mut rng);
    let rff = Rff::new(d, 1.0 / (d as f32).sqrt(), &mut rng);

    // A: evolvent (one pass)
    let mut head = EvolventHead::new(M, 1, 1.0, 1.0);
    let t = Instant::now();
    for &i in perm.iter().take(ntr) {
        head.update(&rff.phi(&x[i * d..i * d + d]), &[y[i]]);
    }
    let ta = t.elapsed().as_secs_f64() * 1e3;

    // B: online-SGD (one pass, same basis)
    let mut lin = LinearLayer::new(M, 1, 21 + seed);
    let (mut aw, mut ab) = (AdamState::new(lin.w.len()), AdamState::new(lin.b.len()));
    let t = Instant::now();
    for &i in perm.iter().take(ntr) {
        let phi = rff.phi(&x[i * d..i * d + d]);
        let p = linear_forward(&lin, &phi)[0];
        let (_g, gl) = linear_backward(&lin, &phi, &[2.0 * (p - y[i])]);
        adam_step(&mut lin.w, &gl.w, &mut aw, 0.01);
        adam_step(&mut lin.b, &gl.b, &mut ab, 0.01);
    }
    let tb = t.elapsed().as_secs_f64() * 1e3;

    // C: backprop MLP (EPOCHS passes on raw features)
    let mut l1 = LinearLayer::new(d, H, 31 + seed);
    let mut l2 = LinearLayer::new(H, 1, 41 + seed);
    let (mut c1w, mut c1b, mut c2w, mut c2b) = (
        AdamState::new(l1.w.len()),
        AdamState::new(l1.b.len()),
        AdamState::new(l2.w.len()),
        AdamState::new(l2.b.len()),
    );
    let t = Instant::now();
    for _ in 0..EPOCHS {
        for &i in perm.iter().take(ntr) {
            let z1 = linear_forward(&l1, &x[i * d..i * d + d]);
            let a1: Vec<f32> = z1.iter().map(|v| v.tanh()).collect();
            let p = linear_forward(&l2, &a1)[0];
            let (ga1, gl2) = linear_backward(&l2, &a1, &[2.0 * (p - y[i])]);
            let gz1: Vec<f32> = ga1
                .iter()
                .zip(&z1)
                .map(|(&g, &z)| g * (1.0 - z.tanh().powi(2)))
                .collect();
            let (_g, gl1) = linear_backward(&l1, &x[i * d..i * d + d], &gz1);
            adam_step(&mut l1.w, &gl1.w, &mut c1w, 0.01);
            adam_step(&mut l1.b, &gl1.b, &mut c1b, 0.01);
            adam_step(&mut l2.w, &gl2.w, &mut c2w, 0.01);
            adam_step(&mut l2.b, &gl2.b, &mut c2b, 0.01);
        }
    }
    let tc = t.elapsed().as_secs_f64() * 1e3;

    // eval on held-out
    let (mut pa, mut pb, mut pc, mut yt) = (vec![], vec![], vec![], vec![]);
    for &i in perm.iter().skip(ntr) {
        let phi = rff.phi(&x[i * d..i * d + d]);
        pa.push(head.predict(&phi)[0]);
        pb.push(linear_forward(&lin, &phi)[0]);
        let a1: Vec<f32> = linear_forward(&l1, &x[i * d..i * d + d])
            .iter()
            .map(|v| v.tanh())
            .collect();
        pc.push(linear_forward(&l2, &a1)[0]);
        yt.push(y[i]);
    }
    let rmse = |p: &[f32]| {
        (p.iter()
            .zip(&yt)
            .map(|(&a, &b)| (a - b).powi(2))
            .sum::<f32>()
            / p.len() as f32)
            .sqrt()
    };
    let (ra, rb, rc) = (r2_score(&pa, &yt), r2_score(&pb, &yt), r2_score(&pc, &yt));
    println!(
        "  {name:<16} R2  A(evolvent) {ra:.3}  B(sgd) {rb:.3}  C(mlp) {rc:.3}  | RMSE {:.3}/{:.3}/{:.3}  | 1pass {ta:.0}/{tb:.0}ms vs {EPOCHS}ep {tc:.0}ms",
        rmse(&pa), rmse(&pb), rmse(&pc)
    );
    [ra as f64, rb as f64, rc as f64]
}

/// Classification bench (one-hot least-squares -> argmax). Returns test accuracy [A,B,C].
fn bench_cls(
    name: &str,
    x: &[f32],
    y: &[usize],
    n: usize,
    d: usize,
    k: usize,
    seed: u64,
) -> [f64; 3] {
    let mut rng = Rng(2 + seed);
    let ntr = n * 3 / 4;
    let perm = shuffle(n, &mut rng);
    let rff = Rff::new(d, 1.0 / (d as f32).sqrt(), &mut rng);
    let onehot = |c: usize| {
        (0..k)
            .map(|j| if j == c { 1.0 } else { 0.0 })
            .collect::<Vec<f32>>()
    };

    // A: evolvent one-hot RLS (one pass)
    let mut head = EvolventHead::new(M, k, 1.0, 1.0);
    let t = Instant::now();
    for &i in perm.iter().take(ntr) {
        head.update(&rff.phi(&x[i * d..i * d + d]), &onehot(y[i]));
    }
    let ta = t.elapsed().as_secs_f64() * 1e3;

    // B: online-SGD one-hot (one pass, same basis)
    let mut lin = LinearLayer::new(M, k, 21 + seed);
    let (mut aw, mut ab) = (AdamState::new(lin.w.len()), AdamState::new(lin.b.len()));
    let t = Instant::now();
    for &i in perm.iter().take(ntr) {
        let phi = rff.phi(&x[i * d..i * d + d]);
        let p = linear_forward(&lin, &phi);
        let oh = onehot(y[i]);
        let g: Vec<f32> = p.iter().zip(&oh).map(|(&pp, &o)| 2.0 * (pp - o)).collect();
        let (_gg, gl) = linear_backward(&lin, &phi, &g);
        adam_step(&mut lin.w, &gl.w, &mut aw, 0.01);
        adam_step(&mut lin.b, &gl.b, &mut ab, 0.01);
    }
    let tb = t.elapsed().as_secs_f64() * 1e3;

    // C: backprop MLP (EPOCHS)
    let mut l1 = LinearLayer::new(d, H, 31 + seed);
    let mut l2 = LinearLayer::new(H, k, 41 + seed);
    let (mut c1w, mut c1b, mut c2w, mut c2b) = (
        AdamState::new(l1.w.len()),
        AdamState::new(l1.b.len()),
        AdamState::new(l2.w.len()),
        AdamState::new(l2.b.len()),
    );
    let t = Instant::now();
    for _ in 0..EPOCHS {
        for &i in perm.iter().take(ntr) {
            let z1 = linear_forward(&l1, &x[i * d..i * d + d]);
            let a1: Vec<f32> = z1.iter().map(|v| v.tanh()).collect();
            let p = linear_forward(&l2, &a1);
            let oh = onehot(y[i]);
            let g2: Vec<f32> = p.iter().zip(&oh).map(|(&pp, &o)| 2.0 * (pp - o)).collect();
            let (ga1, gl2) = linear_backward(&l2, &a1, &g2);
            let gz1: Vec<f32> = ga1
                .iter()
                .zip(&z1)
                .map(|(&g, &z)| g * (1.0 - z.tanh().powi(2)))
                .collect();
            let (_g, gl1) = linear_backward(&l1, &x[i * d..i * d + d], &gz1);
            adam_step(&mut l1.w, &gl1.w, &mut c1w, 0.01);
            adam_step(&mut l1.b, &gl1.b, &mut c1b, 0.01);
            adam_step(&mut l2.w, &gl2.w, &mut c2w, 0.01);
            adam_step(&mut l2.b, &gl2.b, &mut c2b, 0.01);
        }
    }
    let tc = t.elapsed().as_secs_f64() * 1e3;

    let argmax = |v: &[f32]| {
        (0..k)
            .max_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap())
            .unwrap()
    };
    let (mut ca, mut cb, mut cc, mut tot) = (0usize, 0usize, 0usize, 0usize);
    for &i in perm.iter().skip(ntr) {
        let phi = rff.phi(&x[i * d..i * d + d]);
        if head.predict_class(&phi) == y[i] {
            ca += 1;
        }
        if argmax(&linear_forward(&lin, &phi)) == y[i] {
            cb += 1;
        }
        let a1: Vec<f32> = linear_forward(&l1, &x[i * d..i * d + d])
            .iter()
            .map(|v| v.tanh())
            .collect();
        if argmax(&linear_forward(&l2, &a1)) == y[i] {
            cc += 1;
        }
        tot += 1;
    }
    let acc = |c: usize| c as f64 / tot as f64;
    println!(
        "  {name:<16} ACC A(evolvent) {:.3}  B(sgd) {:.3}  C(mlp) {:.3}  | 1pass {ta:.0}/{tb:.0}ms vs {EPOCHS}ep {tc:.0}ms",
        acc(ca), acc(cb), acc(cc)
    );
    [acc(ca), acc(cb), acc(cc)]
}

fn main() {
    let seed: u64 = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--seed=")
                .map(|s| s.parse::<u64>().unwrap_or(0))
        })
        .unwrap_or(0);
    println!("evolvent bench (seed {seed}) — A one-pass RLS · B one-pass SGD · C {EPOCHS}-epoch MLP\nREGRESSION:");
    let mut rows: Vec<(&str, &str, [f64; 3])> = Vec::new();

    // generated regression: nonlinear function of 5 inputs
    {
        let (n, d) = (2000usize, 5usize);
        let mut rng = Rng(100 + seed);
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0.0f32; n];
        for i in 0..n {
            for k in 0..d {
                x[i * d + k] = rng.u();
            }
            let r = &x[i * d..i * d + d];
            y[i] = (2.0 * r[0]).sin() + r[1] * r[2] + 0.5 * r[3] * r[3] - r[4] + 0.1 * rng.gauss();
        }
        rows.push(("gen_reg", "R2", bench_reg("generated", &x, &y, n, d, seed)));
    }
    // real regression: California housing
    if let Ok(txt) = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/california.csv"
    )) {
        let data: TabularReg = load_csv_regression(&txt);
        rows.push((
            "california_reg",
            "R2",
            bench_reg("california", &data.x, &data.target, data.n, data.d, seed),
        ));
    }

    println!("CLASSIFICATION:");
    // generated classification: 3 gaussian blobs in 4-D
    {
        let (n, d, k) = (1500usize, 4usize, 3usize);
        let mut rng = Rng(200 + seed);
        let centers: Vec<Vec<f32>> = (0..k)
            .map(|_| (0..d).map(|_| rng.u() * 2.0).collect())
            .collect();
        let mut x = vec![0.0f32; n * d];
        let mut y = vec![0usize; n];
        for i in 0..n {
            let c = i % k;
            y[i] = c;
            for j in 0..d {
                x[i * d + j] = centers[c][j] + 0.7 * rng.gauss();
            }
        }
        rows.push((
            "gen_cls",
            "ACC",
            bench_cls("generated", &x, &y, n, d, k, seed),
        ));
    }
    // real classification: Iris
    if let Ok(txt) = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/iris.csv"
    )) {
        let data: Tabular = load_csv(&txt);
        rows.push((
            "iris_cls",
            "ACC",
            bench_cls(
                "iris",
                &data.x,
                &data.y,
                data.n,
                data.d,
                data.n_classes,
                seed,
            ),
        ));
    }

    // machine-readable summary
    let body = rows
        .iter()
        .map(|(name, metric, v)| format!("  {{\"dataset\": \"{name}\", \"metric\": \"{metric}\", \"evolvent\": {:.4}, \"sgd\": {:.4}, \"mlp\": {:.4}}}", v[0], v[1], v[2]))
        .collect::<Vec<_>>()
        .join(",\n");
    let out = format!("reports/figures/evolvent_bench_seed{seed}.json");
    if let Some(p) = std::path::Path::new(&out).parent() {
        std::fs::create_dir_all(p).ok();
    }
    std::fs::File::create(&out)
        .unwrap()
        .write_all(format!("{{\n \"seed\": {seed}, \"m_rff\": {M}, \"mlp_epochs\": {EPOCHS},\n \"results\": [\n{body}\n ]\n}}\n").as_bytes())
        .unwrap();
}
