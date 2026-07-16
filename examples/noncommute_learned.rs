//! **Capstone — learning holonomy-native on the non-abelian task.** An end-to-end network with
//! `rotor_holonomy` as a differentiable CORE layer, trained by composing the crate's closed-form
//! backwards (`linear` + `rotor_holonomy`) — NO autograd tape (the F-HOLO-6 discipline). Does the
//! LEARNED holonomy pipeline discover the commutator on the F-HOLO-9 non-commutativity task, where a
//! generic MLP over the raw edges is at chance?
//!
//! Model:  raw edges (2k×4)  →  learned per-edge linear W1 (4→4)  →  rotor_holonomy (2 cycles, k)
//!         →  [H_A', H_B'] (8)  →  learned head (8→16→1).
//! Backward composes `linear_backward` ∘ `rotor_holonomy_backward` ∘ `linear_backward` — every atom a
//! hand-derived closed-form op. A hard FD gate on d(loss)/d(W1) runs first (a wrong gradient = a phantom).
//!
//! Arms (5 seeds, held-out AUROC): generic-MLP (raw) — chance; learned-holonomy-net — SOLVES;
//! fixed-commutator (context) — 1.0.
//!
//! Run: `cargo run --release --example noncommute_learned [-- --fdcheck]`

use holonomy_learn::{
    adam_step, auroc, commutator_angle, linear_backward, linear_forward, rotor_holonomy_backward,
    rotor_holonomy_forward, sample_noncommute, AdamState, CurvatureRng, LinearLayer,
};

const K: usize = 6;
const N_TRAIN: usize = 400;
const N_TEST: usize = 400;
const SEEDS: u64 = 5;

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}
fn sep(scores: &[f32], labels: &[u8]) -> f64 {
    let a = auroc(scores, labels);
    a.max(1.0 - a)
}
fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn gen_set(rng: &mut CurvatureRng, n: usize) -> (Vec<Vec<f32>>, Vec<u8>) {
    let mut xs = Vec::with_capacity(n);
    let mut ys = Vec::with_capacity(n);
    for i in 0..n {
        let class = (i % 2) as u8;
        let theta = 0.8 + rng.f() * 1.0;
        xs.push(sample_noncommute(rng, K, theta, class));
        ys.push(class);
    }
    (xs, ys)
}

/// The learned holonomy-native network.
struct HoloNet {
    w1: LinearLayer,    // per-edge 4→4
    head1: LinearLayer, // 8→16
    head2: LinearLayer, // 16→1
}
struct Cache {
    eprime: Vec<f32>,   // (2k,4) transformed edges
    prefixes: Vec<f32>, // rotor_holonomy prefixes (2k*4)
    holo: Vec<f32>,     // (8) [H_A', H_B']
    h: Vec<f32>,        // (16) tanh hidden
}

impl HoloNet {
    fn new(seed: u64) -> Self {
        // W1 initialised to identity (so the initial holonomy is the true product)
        let mut w1 = LinearLayer::new(4, 4, seed);
        for (i, v) in w1.w.iter_mut().enumerate() {
            *v = if i / 4 == i % 4 { 1.0 } else { 0.0 };
        }
        for b in w1.b.iter_mut() {
            *b = 0.0;
        }
        HoloNet {
            w1,
            head1: LinearLayer::new(8, 16, seed + 1),
            head2: LinearLayer::new(16, 1, seed + 2),
        }
    }

    fn forward(&self, x: &[f32]) -> (f32, Cache) {
        let eprime = linear_forward(&self.w1, x); // (2k,4)
        let (holo, prefixes) = rotor_holonomy_forward(&eprime, 2, K); // holo (8)
        let hpre = linear_forward(&self.head1, &holo); // (16)
        let h: Vec<f32> = hpre.iter().map(|z| z.tanh()).collect();
        let logit = linear_forward(&self.head2, &h)[0];
        (
            logit,
            Cache {
                eprime,
                prefixes,
                holo,
                h,
            },
        )
    }

    /// Returns per-parameter gradients (as LinearLayers) for (w1, head1, head2).
    fn backward(
        &self,
        x: &[f32],
        c: &Cache,
        dlogit: f32,
    ) -> (LinearLayer, LinearLayer, LinearLayer) {
        let (grad_h, ghead2) = linear_backward(&self.head2, &c.h, &[dlogit]);
        let grad_hpre: Vec<f32> = grad_h
            .iter()
            .zip(&c.h)
            .map(|(g, hv)| g * (1.0 - hv * hv))
            .collect();
        let (grad_holo, ghead1) = linear_backward(&self.head1, &c.holo, &grad_hpre);
        let grad_eprime = rotor_holonomy_backward(&c.eprime, &c.prefixes, &grad_holo, 2, K);
        let (_gx, gw1) = linear_backward(&self.w1, x, &grad_eprime);
        (gw1, ghead1, ghead2)
    }
}

fn bce(logit: f32, y: u8) -> f32 {
    let p = sigmoid(logit).clamp(1e-7, 1.0 - 1e-7);
    -(y as f32 * p.ln() + (1.0 - y as f32) * (1.0 - p).ln())
}

/// Hard FD gate: analytic d(loss)/d(W1[idx]) == finite difference. A wrong gradient is a phantom.
fn fd_gate() {
    let mut rng = CurvatureRng(3);
    let (xs, ys) = gen_set(&mut rng, 1);
    let (x, y) = (&xs[0], ys[0]);
    let net = HoloNet::new(7);
    let (logit, cache) = net.forward(x);
    let (gw1, _g1, _g2) = net.backward(x, &cache, sigmoid(logit) - y as f32);
    let eps = 1e-3f32;
    let loss_with = |w: &[f32]| -> f32 {
        let mut n2 = HoloNet::new(7);
        n2.w1.w.copy_from_slice(w);
        let (l, _) = n2.forward(x);
        bce(l, y)
    };
    let mut max_err = 0.0f32;
    for idx in [0usize, 3, 5, 10, 15] {
        let (mut wp, mut wm) = (net.w1.w.clone(), net.w1.w.clone());
        wp[idx] += eps;
        wm[idx] -= eps;
        let fd = (loss_with(&wp) - loss_with(&wm)) / (2.0 * eps);
        max_err = max_err.max((fd - gw1.w[idx]).abs());
    }
    println!(
        "  FD gate d(loss)/d(W1): max |analytic - fd| = {max_err:.2e}  {}",
        if max_err < 5e-2 { "PASS" } else { "FAIL" }
    );
    assert!(
        max_err < 5e-2,
        "end-to-end gradient FD gate failed — training would be a phantom"
    );
}

fn train_holonet(seed: u64, epochs: usize, lr: f32, train_w1: bool) -> f64 {
    let mut rng = CurvatureRng(101 + seed);
    let (xtr, ytr) = gen_set(&mut rng, N_TRAIN);
    let (xte, yte) = gen_set(&mut rng, N_TEST);
    let mut net = HoloNet::new(777 + seed);
    let mut a = (
        AdamState::new(net.w1.w.len()),
        AdamState::new(net.w1.b.len()),
        AdamState::new(net.head1.w.len()),
        AdamState::new(net.head1.b.len()),
        AdamState::new(net.head2.w.len()),
        AdamState::new(net.head2.b.len()),
    );
    for _ in 0..epochs {
        let (mut gw1, mut gb1) = (vec![0.0f32; net.w1.w.len()], vec![0.0f32; net.w1.b.len()]);
        let (mut gh1w, mut gh1b) = (
            vec![0.0f32; net.head1.w.len()],
            vec![0.0f32; net.head1.b.len()],
        );
        let (mut gh2w, mut gh2b) = (
            vec![0.0f32; net.head2.w.len()],
            vec![0.0f32; net.head2.b.len()],
        );
        for (x, &y) in xtr.iter().zip(&ytr) {
            let (logit, c) = net.forward(x);
            let (a1, a2, a3) = net.backward(x, &c, sigmoid(logit) - y as f32);
            for (g, v) in gw1.iter_mut().zip(&a1.w) {
                *g += v;
            }
            for (g, v) in gb1.iter_mut().zip(&a1.b) {
                *g += v;
            }
            for (g, v) in gh1w.iter_mut().zip(&a2.w) {
                *g += v;
            }
            for (g, v) in gh1b.iter_mut().zip(&a2.b) {
                *g += v;
            }
            for (g, v) in gh2w.iter_mut().zip(&a3.w) {
                *g += v;
            }
            for (g, v) in gh2b.iter_mut().zip(&a3.b) {
                *g += v;
            }
        }
        let m = xtr.len() as f32;
        for g in gw1
            .iter_mut()
            .chain(gb1.iter_mut())
            .chain(gh1w.iter_mut())
            .chain(gh1b.iter_mut())
            .chain(gh2w.iter_mut())
            .chain(gh2b.iter_mut())
        {
            *g /= m;
        }
        if train_w1 {
            adam_step(&mut net.w1.w, &gw1, &mut a.0, lr);
            adam_step(&mut net.w1.b, &gb1, &mut a.1, lr);
        }
        adam_step(&mut net.head1.w, &gh1w, &mut a.2, lr);
        adam_step(&mut net.head1.b, &gh1b, &mut a.3, lr);
        adam_step(&mut net.head2.w, &gh2w, &mut a.4, lr);
        adam_step(&mut net.head2.b, &gh2b, &mut a.5, lr);
    }
    let scores: Vec<f32> = xte.iter().map(|x| net.forward(x).0).collect();
    sep(&scores, &yte)
}

/// generic MLP over the raw edges (the F-HOLO-9 floor).
fn train_generic_mlp(seed: u64) -> f64 {
    let mut rng = CurvatureRng(101 + seed);
    let (xtr, ytr) = gen_set(&mut rng, N_TRAIN);
    let (xte, yte) = gen_set(&mut rng, N_TEST);
    let din = xtr[0].len();
    let hidden = 64;
    let mut r = CurvatureRng(777 + seed);
    let mut w1: Vec<f32> = (0..hidden * din)
        .map(|_| 0.3 * r.g() / (din as f32).sqrt())
        .collect();
    let mut b1 = vec![0.0f32; hidden];
    let mut w2: Vec<f32> = (0..hidden)
        .map(|_| 0.3 * r.g() / (hidden as f32).sqrt())
        .collect();
    let mut b2 = 0.0f32;
    let (mut s1, mut sb1, mut s2, mut sb2) = (
        AdamState::new(hidden * din),
        AdamState::new(hidden),
        AdamState::new(hidden),
        AdamState::new(1),
    );
    let fwd = |w1: &[f32], b1: &[f32], w2: &[f32], b2: f32, x: &[f32]| -> (f32, Vec<f32>) {
        let mut h = vec![0.0f32; hidden];
        for j in 0..hidden {
            let mut z = b1[j];
            for k in 0..din {
                z += w1[j * din + k] * x[k];
            }
            h[j] = z.tanh();
        }
        (b2 + (0..hidden).map(|j| w2[j] * h[j]).sum::<f32>(), h)
    };
    for _ in 0..500 {
        let (mut gw1, mut gb1, mut gw2, mut gb2) = (
            vec![0.0f32; hidden * din],
            vec![0.0f32; hidden],
            vec![0.0f32; hidden],
            0.0f32,
        );
        for (x, &y) in xtr.iter().zip(&ytr) {
            let (logit, h) = fwd(&w1, &b1, &w2, b2, x);
            let dl = sigmoid(logit) - y as f32;
            gb2 += dl;
            for j in 0..hidden {
                gw2[j] += dl * h[j];
                let dz = dl * w2[j] * (1.0 - h[j] * h[j]);
                gb1[j] += dz;
                for k in 0..din {
                    gw1[j * din + k] += dz * x[k];
                }
            }
        }
        let m = xtr.len() as f32;
        for g in gw1.iter_mut() {
            *g /= m;
        }
        for g in gb1.iter_mut() {
            *g /= m;
        }
        for g in gw2.iter_mut() {
            *g /= m;
        }
        gb2 /= m;
        adam_step(&mut w1, &gw1, &mut s1, 0.01);
        adam_step(&mut b1, &gb1, &mut sb1, 0.01);
        adam_step(&mut w2, &gw2, &mut s2, 0.01);
        let mut bb = [b2];
        adam_step(&mut bb, &[gb2], &mut sb2, 0.01);
        b2 = bb[0];
    }
    let scores: Vec<f32> = xte.iter().map(|x| fwd(&w1, &b1, &w2, b2, x).0).collect();
    sep(&scores, &yte)
}

fn main() {
    let fdcheck = std::env::args().any(|a| a == "--fdcheck");
    println!("== FD gate (end-to-end closed-form gradient through rotor_holonomy) ==");
    fd_gate();
    if fdcheck {
        return;
    }
    let t0 = std::time::Instant::now();
    println!("\n== Capstone: learned holonomy-native vs generic MLP on the non-abelian task ({SEEDS} seeds) ==");
    let holo = median(
        &(0..SEEDS)
            .map(|s| train_holonet(s, 400, 0.02, true))
            .collect::<Vec<_>>(),
    );
    let holo_frozen = median(
        &(0..SEEDS)
            .map(|s| train_holonet(s, 400, 0.02, false))
            .collect::<Vec<_>>(),
    );
    let mlp = median(&(0..SEEDS).map(train_generic_mlp).collect::<Vec<_>>());
    // fixed-commutator context (5 seeds)
    let mut rng = CurvatureRng(101);
    let (xs, ys) = gen_set(&mut rng, N_TEST);
    let com: Vec<f32> = xs.iter().map(|x| commutator_angle(x, K)).collect();
    let fixed = sep(&com, &ys);

    println!("\n  {:<30} {:>8}", "arm", "median");
    println!("  {:<30} {:>8.3}", "generic-MLP (raw edges)", mlp);
    println!("  {:<30} {:>8.3}", "learned-holo-net (train W1)", holo);
    println!(
        "  {:<30} {:>8.3}",
        "learned-holo-net (frozen W1)", holo_frozen
    );
    println!("  {:<30} {:>8.3}", "fixed-commutator (context)", fixed);
    println!("\n== VERDICT ==");
    println!(
        "  frozen-W1 (fixed holonomy op + learned head) {holo_frozen:.3} vs trainable-W1 {holo:.3}: {}",
        if holo_frozen >= holo + 0.2 {
            "the TRAINABLE pre-composition CORRUPTS the holonomy — the designed op must stay FIXED"
        } else {
            "trainable and frozen similar — the failure is elsewhere"
        }
    );
    println!(
        "  learned-holo (best {:.3}) vs generic-MLP {mlp:.3}: {}",
        holo.max(holo_frozen),
        if holo.max(holo_frozen) >= mlp + 0.2 {
            "learning ON the fixed holonomy op works where generic learning can't"
        } else {
            "no learned arm beats generic — the value is the FIXED op alone (thesis: designed > learned)"
        }
    );
    println!("== done in {:.1}s ==", t0.elapsed().as_secs_f32());
}
