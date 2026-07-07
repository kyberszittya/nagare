//! Forward-throughput scaling benchmark for the entropy-feedback net.
//!
//! Parameterized over batch / points / input-dim / hidden so Nagare's
//! closed-form parallel forward can be stress-tested at scale and high
//! dimensionality against a matched PyTorch net (see
//! `scripts/dev/scaling_bench_torch.py`). Random generated data — this measures
//! throughput + memory, not accuracy.
//!
//! Uses the shipped ikj kernels (`linear_forward`, `fused_entropy_update`) and a
//! batch-parallel pool (bit-identical; parallel pays off at hidden >= ~16).
//!
//! Run: `cargo run --release --example scaling_bench -- \
//!         --batch 1024 --points 64 --input-dim 64 --hidden 128 --reps 50`

use std::time::Instant;

use holonomy_learn::{
    fused_entropy_update_forward, linear_forward, FusedEntropyUpdateShape, LinearLayer,
};
use rayon::prelude::*;

fn arg(name: &str, default: usize) -> usize {
    std::env::args()
        .skip_while(|a| a != name)
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

/// mean/std/max global pool over points -> (batch, 3*hidden), parallel over
/// batch (each set independent -> bit-identical to serial).
fn global_pool(input: &[f32], batch: usize, points: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * hidden * 3];
    let inv_points = 1.0 / points as f32;
    out.par_chunks_mut(hidden * 3)
        .enumerate()
        .for_each(|(b, chunk)| {
            for h in 0..hidden {
                let mut sum = 0.0;
                let mut max_val = f32::NEG_INFINITY;
                for p in 0..points {
                    let v = input[(b * points + p) * hidden + h];
                    sum += v;
                    max_val = max_val.max(v);
                }
                let mu = sum * inv_points;
                let mut var = 0.0;
                for p in 0..points {
                    let d = input[(b * points + p) * hidden + h] - mu;
                    var += d * d;
                }
                chunk[h] = mu;
                chunk[hidden + h] = (var * inv_points).sqrt();
                chunk[2 * hidden + h] = max_val;
            }
        });
    out
}

fn entropy_feature(logits: &[f32]) -> Vec<f32> {
    logits
        .par_chunks(2)
        .map(|pair| {
            let (a, c) = (pair[0], pair[1]);
            let m = a.max(c);
            let (ea, ec) = ((a - m).exp(), (c - m).exp());
            let (p0, p1) = (ea / (ea + ec), ec / (ea + ec));
            -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2
        })
        .collect()
}

fn pseudo(len: usize, salt: u32) -> Vec<f32> {
    (0..len)
        .map(|i| {
            let h = (i as u32)
                .wrapping_mul(2654435761)
                .wrapping_add(salt.wrapping_mul(40503));
            (h >> 8) as f32 / (1u32 << 24) as f32 - 0.5
        })
        .collect()
}

fn main() {
    let batch = arg("--batch", 1024);
    let points = arg("--points", 64);
    let input_dim = arg("--input-dim", 64);
    let hidden = arg("--hidden", 128);
    let reps = arg("--reps", 50);

    let x = pseudo(batch * points * input_dim, 1);
    let embed = LinearLayer::new(input_dim, hidden, 11);
    let first = LinearLayer::new(3 * hidden, 2, 12);
    let update = LinearLayer::new(4 * hidden + 1, hidden, 13);
    let head = LinearLayer::new(3 * hidden, 2, 14);
    let shape = FusedEntropyUpdateShape {
        batch,
        points,
        hidden,
    };

    let forward = || {
        let embed_pre = linear_forward(&embed, &x);
        let h = relu(&embed_pre);
        let pooled = global_pool(&h, batch, points, hidden);
        let logits_first = linear_forward(&first, &pooled);
        let entropy = entropy_feature(&logits_first);
        let update_pre = fused_entropy_update_forward(&update, &h, &pooled, &entropy, shape);
        let h2 = relu(&update_pre);
        let pooled2 = global_pool(&h2, batch, points, hidden);
        linear_forward(&head, &pooled2)
    };

    for _ in 0..5 {
        let _ = forward();
    }
    let mut times = Vec::with_capacity(reps);
    for _ in 0..reps {
        let s = Instant::now();
        let out = forward();
        times.push(s.elapsed().as_secs_f64());
        std::hint::black_box(out);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    let median_s = times[times.len() / 2];
    let rows = (batch * points) as f64;
    println!(
        "batch={batch} points={points} input_dim={input_dim} hidden={hidden} threads={} | \
         median_ms={:.3} us_per_sample={:.3} Mrows_per_s={:.1}",
        rayon::current_num_threads(),
        median_s * 1e3,
        median_s * 1e6 / batch as f64,
        rows / median_s / 1e6,
    );
}
