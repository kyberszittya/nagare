//! Per-stage timing breakdown of the entropy-feedback parity forward.
//!
//! kato15 has no `perf`/`cargo flamegraph`, so this is the profiler-free
//! substitute: it runs the exact op sequence of the PyTorch-parity forward
//! (embed -> pool -> first -> entropy -> fused-update -> pool -> head) at the
//! parity shapes and reports each stage's share of the wall time. Use it to
//! locate the bottleneck *before* optimizing any kernel (CLAUDE.md S3: no
//! micro-optimization without a profile showing the hot spot).
//!
//! Run: `cargo run --release --example forward_profile [-- --reps 300]`

use std::time::Instant;

use holonomy_learn::{
    fused_entropy_update_forward, linear_forward, FusedEntropyUpdateShape, LinearLayer,
};

const BATCH: usize = 96;
const POINTS: usize = 48;
const HIDDEN: usize = 32;

fn relu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|v| v.max(0.0)).collect()
}

/// mean/std/max global pool over points -> (batch, 3*hidden). Serial, matching
/// the parity harness.
fn global_pool(input: &[f32], batch: usize, points: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0; batch * hidden * 3];
    let inv_points = 1.0 / points as f32;
    for b in 0..batch {
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
            out[b * hidden * 3 + h] = mu;
            out[b * hidden * 3 + hidden + h] = (var * inv_points).sqrt();
            out[b * hidden * 3 + 2 * hidden + h] = max_val;
        }
    }
    out
}

fn entropy_feature(logits: &[f32]) -> Vec<f32> {
    let batch = logits.len() / 2;
    let mut out = vec![0.0; batch];
    for b in 0..batch {
        let (a, c) = (logits[2 * b], logits[2 * b + 1]);
        let m = a.max(c);
        let (ea, ec) = ((a - m).exp(), (c - m).exp());
        let (p0, p1) = (ea / (ea + ec), ec / (ea + ec));
        out[b] = -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2;
    }
    out
}

/// Deterministic pseudo-random f32 in [-0.5, 0.5] (no rng dep needed here).
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
    let reps: usize = std::env::args()
        .skip_while(|a| a != "--reps")
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(300);

    let x = pseudo(BATCH * POINTS * 2, 1);
    let embed = LinearLayer::new(2, HIDDEN, 11);
    let first = LinearLayer::new(3 * HIDDEN, 2, 12);
    let update = LinearLayer::new(4 * HIDDEN + 1, HIDDEN, 13);
    let head = LinearLayer::new(3 * HIDDEN, 2, 14);
    let shape = FusedEntropyUpdateShape {
        batch: BATCH,
        points: POINTS,
        hidden: HIDDEN,
    };

    // Accumulated nanos per stage.
    let mut t = [0u128; 7];
    let labels = [
        "embed (linear 2->32, 4608 rows)",
        "pool1 (serial mean/std/max)",
        "first (linear 96->2)",
        "entropy (exp/ln, 96)",
        "fused_update (129->32, 4608 rows)",
        "pool2 (serial mean/std/max)",
        "head (linear 96->2)",
    ];

    for rep in 0..(reps + 20) {
        let s = Instant::now();
        let embed_pre = linear_forward(&embed, &x);
        let d0 = s.elapsed().as_nanos();
        let h = relu(&embed_pre);

        let s = Instant::now();
        let pooled = global_pool(&h, BATCH, POINTS, HIDDEN);
        let d1 = s.elapsed().as_nanos();

        let s = Instant::now();
        let logits_first = linear_forward(&first, &pooled);
        let d2 = s.elapsed().as_nanos();

        let s = Instant::now();
        let entropy = entropy_feature(&logits_first);
        let d3 = s.elapsed().as_nanos();

        let s = Instant::now();
        let update_pre = fused_entropy_update_forward(&update, &h, &pooled, &entropy, shape);
        let d4 = s.elapsed().as_nanos();
        let h2 = relu(&update_pre);

        let s = Instant::now();
        let pooled2 = global_pool(&h2, BATCH, POINTS, HIDDEN);
        let d5 = s.elapsed().as_nanos();

        let s = Instant::now();
        let _logits = linear_forward(&head, &pooled2);
        let d6 = s.elapsed().as_nanos();

        if rep >= 20 {
            for (acc, d) in t.iter_mut().zip([d0, d1, d2, d3, d4, d5, d6]) {
                *acc += d;
            }
        }
    }

    let total: u128 = t.iter().sum();
    let per_sample_us = |ns: u128| ns as f64 / reps as f64 / 1000.0 / BATCH as f64;
    println!(
        "profile: batch={BATCH} points={POINTS} hidden={HIDDEN} reps={reps} threads={}",
        rayon::current_num_threads()
    );
    println!("{:<38} {:>10} {:>8}", "stage", "us/sample", "% total");
    for (label, &ns) in labels.iter().zip(t.iter()) {
        println!(
            "{:<38} {:>10.3} {:>7.1}%",
            label,
            per_sample_us(ns),
            100.0 * ns as f64 / total as f64
        );
    }
    println!(
        "{:<38} {:>10.3}",
        "TOTAL (timed stages)",
        per_sample_us(total)
    );
}
