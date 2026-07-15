//! E0 — the evolvent hypothesis probe (NON-CV): incremental online learning vs
//! slow backprop on a NON-STATIONARY regression stream. No autograd.
//!
//! One drifting teacher `y = sin(<a_t,x>) + <c_t,x> + noise` is streamed one
//! sample at a time; `a_t`,`c_t` drift slowly and flip abruptly at the midpoint.
//! Three learners see the SAME stream, predict-then-update (prequential):
//!   A EVOLVENT   — fixed RFF basis + `EvolventHead` (forgetting-RLS): one-pass,
//!                  closed-form, NO backward sweep.
//!   B ONLINE-BP  — same RFF basis + linear head by Adam SGD (isolates RLS vs SGD).
//!   C BACKPROP   — a small MLP whose FEATURES are learned by full online backprop
//!                  (linear->tanh->linear, the sweep backprop pays for).
//!
//! Reported: windowed prequential MSE (adaptation + post-drift recovery),
//! throughput (us/update), steady-state MSE. The evolvent win only counts if A
//! MATCHES C's steady accuracy while adapting faster (honesty guard).
//!
//! Run: `cargo run --release --example evolvent_stream -- [--seed=N] [out.json]`

use holonomy_learn::{
    adam_step, linear_backward, linear_forward, AdamState, EvolventHead, LinearLayer,
};
use std::f32::consts::TAU;
use std::io::Write;
use std::time::Instant;

const DX: usize = 4; // input dim
const M: usize = 96; // RFF features (kept << forgetting window for a well-conditioned RLS)
const H: usize = 64; // MLP hidden
const N: usize = 8000; // stream length
const WINDOWS: usize = 16;
const LAMBDA: f32 = 0.999; // forgetting factor (effective window ~1000 >> M)
const TRACE_CAP: f32 = 1.0e3; // tight covariance bound → windup-stable tail

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
        let (u1, u2) = (self.f().max(1e-7), self.f());
        (-2.0 * u1.ln()).sqrt() * (TAU * u2).cos()
    }
}

/// Fixed random Fourier features approximating an RBF kernel: phi_j(x) = sqrt(2/M) cos(w_j·x + b_j).
struct Rff {
    w: Vec<f32>,
    b: Vec<f32>,
}
impl Rff {
    fn new(rng: &mut Rng, gamma: f32) -> Self {
        let w = (0..M * DX).map(|_| gamma * rng.gauss()).collect();
        let b = (0..M).map(|_| TAU * rng.f()).collect();
        Rff { w, b }
    }
    fn phi(&self, x: &[f32]) -> Vec<f32> {
        let scale = (2.0 / M as f32).sqrt();
        (0..M)
            .map(|j| {
                let dot: f32 = (0..DX).map(|k| self.w[j * DX + k] * x[k]).sum::<f32>() + self.b[j];
                scale * dot.cos()
            })
            .collect()
    }
}

fn main() {
    let seed_base: u64 = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--seed=")
                .map(|s| s.parse::<u64>().unwrap_or(0))
        })
        .unwrap_or(0);
    let out_path = std::env::args()
        .filter(|a| !a.starts_with("--"))
        .nth(1)
        .unwrap_or_else(|| "reports/figures/evolvent-stream.json".into());
    let mut rng = Rng(1234 + seed_base.wrapping_mul(7919));
    let rff = Rff::new(&mut rng, 1.5);

    // A: evolvent forgetting-RLS head over the RFF basis
    let mut head = EvolventHead::new(M, 1.0, LAMBDA).with_trace_cap(TRACE_CAP);
    // B: RFF + linear head by Adam
    let mut lin_b = LinearLayer::new(M, 1, 21 + seed_base);
    let (mut b_w, mut b_b) = (AdamState::new(lin_b.w.len()), AdamState::new(lin_b.b.len()));
    // C: MLP (linear->tanh->linear) by full backprop + Adam
    let mut l1 = LinearLayer::new(DX, H, 31 + seed_base);
    let mut l2 = LinearLayer::new(H, 1, 41 + seed_base);
    let (mut c1w, mut c1b) = (AdamState::new(l1.w.len()), AdamState::new(l1.b.len()));
    let (mut c2w, mut c2b) = (AdamState::new(l2.w.len()), AdamState::new(l2.b.len()));

    // drifting teacher
    let mut a_t: Vec<f32> = (0..DX).map(|_| rng.u()).collect();
    let mut c_t: Vec<f32> = (0..DX).map(|_| rng.u()).collect();
    let teach = |x: &[f32], a: &[f32], c: &[f32], rng: &mut Rng| -> f32 {
        let za: f32 = (0..DX).map(|k| a[k] * x[k]).sum();
        let zc: f32 = (0..DX).map(|k| c[k] * x[k]).sum();
        (2.0 * za).sin() + zc + 0.05 * rng.gauss()
    };

    let mut preq = [(0.0f64, 0.0f64, 0.0f64); WINDOWS];
    let mut count = [0u32; WINDOWS];
    let (mut ta, mut tb, mut tc) = (0u128, 0u128, 0u128);

    for t in 0..N {
        // slow continuous drift + abrupt flip at midpoint
        for k in 0..DX {
            a_t[k] += 0.0008 * rng.u();
            c_t[k] += 0.0008 * rng.u();
        }
        if t == N / 2 {
            for v in a_t.iter_mut() {
                *v = -*v;
            }
        }
        let x: Vec<f32> = (0..DX).map(|_| rng.u()).collect();
        let y = teach(&x, &a_t, &c_t, &mut rng);
        let phi = rff.phi(&x);
        let win = (t * WINDOWS / N).min(WINDOWS - 1);

        // A: predict-then-update (evolvent)
        let t0 = Instant::now();
        let pa = head.predict(&phi);
        head.update(&phi, y);
        ta += t0.elapsed().as_nanos();

        // B: predict-then-update (RFF + linear, Adam)
        let t0 = Instant::now();
        let pb = linear_forward(&lin_b, &phi)[0];
        let (_gx, glb) = linear_backward(&lin_b, &phi, &[2.0 * (pb - y)]);
        adam_step(&mut lin_b.w, &glb.w, &mut b_w, 0.01);
        adam_step(&mut lin_b.b, &glb.b, &mut b_b, 0.01);
        tb += t0.elapsed().as_nanos();

        // C: predict-then-update (MLP, full backprop, Adam)
        let t0 = Instant::now();
        let z1 = linear_forward(&l1, &x);
        let a1: Vec<f32> = z1.iter().map(|v| v.tanh()).collect();
        let pc = linear_forward(&l2, &a1)[0];
        let (g_a1, gl2) = linear_backward(&l2, &a1, &[2.0 * (pc - y)]);
        let g_z1: Vec<f32> = g_a1
            .iter()
            .zip(&z1)
            .map(|(&g, &z)| g * (1.0 - z.tanh().powi(2)))
            .collect();
        let (_gx, gl1) = linear_backward(&l1, &x, &g_z1);
        adam_step(&mut l1.w, &gl1.w, &mut c1w, 0.01);
        adam_step(&mut l1.b, &gl1.b, &mut c1b, 0.01);
        adam_step(&mut l2.w, &gl2.w, &mut c2w, 0.01);
        adam_step(&mut l2.b, &gl2.b, &mut c2b, 0.01);
        tc += t0.elapsed().as_nanos();

        preq[win].0 += ((pa - y) as f64).powi(2);
        preq[win].1 += ((pb - y) as f64).powi(2);
        preq[win].2 += ((pc - y) as f64).powi(2);
        count[win] += 1;
    }

    // windowed prequential RMSE
    let rmse = |s: f64, n: u32| (s / n as f64).sqrt();
    let curve: Vec<[f64; 3]> = (0..WINDOWS)
        .map(|w| {
            [
                rmse(preq[w].0, count[w]),
                rmse(preq[w].1, count[w]),
                rmse(preq[w].2, count[w]),
            ]
        })
        .collect();
    // steady-state = last window (stationary tail after the drift settles)
    let steady = curve[WINDOWS - 1];
    let us = |ns: u128| ns as f64 / N as f64 / 1000.0;

    println!("evolvent stream (seed {seed_base}):");
    println!(
        "  cold-start RMSE (window 0) A {:.4}  B {:.4}  C {:.4}",
        curve[0][0], curve[0][1], curve[0][2]
    );
    println!(
        "  steady RMSE  A(evolvent) {:.4}  B(online-bp) {:.4}  C(backprop-mlp) {:.4}",
        steady[0], steady[1], steady[2]
    );
    println!(
        "  us/update    A {:.2} ({:.0}/s)  B {:.2}  C {:.2}",
        us(ta),
        1e9 / (ta as f64 / N as f64),
        us(tb),
        us(tc)
    );
    // post-drift recovery: window right after the midpoint flip (window 8)
    let dw = WINDOWS / 2;
    println!(
        "  post-drift RMSE (window {dw})  A {:.4}  B {:.4}  C {:.4}",
        curve[dw][0], curve[dw][1], curve[dw][2]
    );

    let json = format!(
        "{{\n  \"seed\": {seed_base},\n  \"steady\": [{:.4},{:.4},{:.4}],\n  \"us_per_update\": [{:.3},{:.3},{:.3}],\n  \"curve\": [{}]\n}}\n",
        steady[0], steady[1], steady[2], us(ta), us(tb), us(tc),
        curve.iter().map(|w| format!("[{:.4},{:.4},{:.4}]", w[0], w[1], w[2])).collect::<Vec<_>>().join(",")
    );
    if let Some(par) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(par).ok();
    }
    std::fs::File::create(&out_path)
        .unwrap()
        .write_all(json.as_bytes())
        .unwrap();
}
