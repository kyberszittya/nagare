//! Phase-pool latency (criterion, §10) — both axes of the differentiable global orientation pool.
//!
//! Forward (deploy: field → invariant feature) and the full train step (forward + backward, the
//! grad that flows into an upstream learned front-end). Measured at a realistic CV size
//! `n=64, g=32, b=16` (nk=9). Run: `cargo bench --bench phase_pool_bench`.

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holonomy_learn::{phase_pool_backward, phase_pool_forward};

/// Deterministic LCG — no external RNG dep in the bench binary.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

fn bench(c: &mut Criterion) {
    let (n, g, b) = (64usize, 32usize, 16usize);
    let mut st = 0x0f1e_2d3c_4b5a_6978u64;
    let field: Vec<f32> = (0..n * g * g * 2).map(|_| lcg(&mut st)).collect();

    c.bench_function("phase_pool_forward_n64_g32_b16", |bn| {
        bn.iter(|| black_box(phase_pool_forward(&field, n, g, b)))
    });
    c.bench_function("phase_pool_train_step_n64_g32_b16", |bn| {
        bn.iter(|| {
            let out = phase_pool_forward(&field, n, g, b);
            let grad_feat = vec![1.0f32; out.feat.len()];
            black_box(phase_pool_backward(&field, &out, &grad_feat, n, g, b))
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
