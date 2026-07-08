//! Phase 1d (latency) — HSiKAN forward + training-step throughput (criterion, §10).
//!
//! Measures both deploy axes: forward at B=1 (single-edge latency) and batched
//! (T=1000), plus the full train step (forward + closed-form backward at T=1000).
//! Run: `cargo bench --bench hsikan_bench`

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use holonomy_learn::{hsikan_backward, hsikan_forward, HsikanConfig, HsikanEdges, HsikanParams};

/// Deterministic LCG — no external RNG dep in the bench binary.
fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

struct Setup {
    x: Vec<f32>,
    inner: Vec<f32>,
    outer: Vec<f32>,
    gate_w: Vec<f32>,
    gate_b: Vec<f32>,
    vertices: Vec<u32>,
    signs: Vec<i8>,
    cfg: HsikanConfig,
}

impl Setup {
    fn new(t: usize) -> Self {
        let (k, d, s, grid, cheb_k) = (4usize, 32usize, 2usize, 6usize, 4usize);
        let n_nodes = (t * k / 2).max(8);
        let mut st = 0x1234_5678_9abc_def0u64;
        let mut fill = |n: usize| (0..n).map(|_| lcg(&mut st)).collect::<Vec<f32>>();
        let vertices = (0..t * k)
            .map(|i| (i.wrapping_mul(2_654_435_761) % n_nodes) as u32)
            .collect();
        let signs = (0..t * k)
            .map(|i| if i % 2 == 0 { 1 } else { -1 })
            .collect();
        Self {
            x: fill(n_nodes * d),
            inner: fill(s * d * cheb_k),
            outer: fill(s * d * cheb_k),
            gate_w: fill(d * d),
            gate_b: vec![-1.0; d],
            vertices,
            signs,
            cfg: HsikanConfig::new(t, k, d, s, grid, cheb_k, true),
        }
    }
    fn params(&self) -> HsikanParams<'_> {
        HsikanParams {
            inner_coef: &self.inner,
            outer_coef: &self.outer,
            gate_w: &self.gate_w,
            gate_b: &self.gate_b,
        }
    }
    fn edges(&self) -> HsikanEdges<'_> {
        HsikanEdges {
            vertices: &self.vertices,
            signs: &self.signs,
        }
    }
}

fn bench(c: &mut Criterion) {
    let b1 = Setup::new(1);
    let big = Setup::new(1000);

    c.bench_function("hsikan_forward_b1", |b| {
        b.iter(|| black_box(hsikan_forward(b1.params(), &b1.x, b1.edges(), b1.cfg)))
    });
    c.bench_function("hsikan_forward_t1000", |b| {
        b.iter(|| black_box(hsikan_forward(big.params(), &big.x, big.edges(), big.cfg)))
    });
    c.bench_function("hsikan_train_step_t1000", |b| {
        b.iter(|| {
            let (h_e, cache) = hsikan_forward(big.params(), &big.x, big.edges(), big.cfg);
            let grad_he = vec![1.0f32; h_e.len()];
            black_box(hsikan_backward(
                big.params(),
                big.edges(),
                &cache,
                &grad_he,
                big.cfg,
            ))
        })
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
