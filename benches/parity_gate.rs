//! Parity-gate benchmark: NAGARE-CPU throughput on a synthetic
//! Slashdot-class signed cycle pool.
//!
//! This is prerequisite #4 from the frozen NAGARE plan:
//!   "Slashdot 5-seed AUC parity — Nagare-CPU vs PyTorch+Triton"
//!
//! Until the real Slashdot dataset loader is wired in, this measures
//! FIR + scatter-mean forward throughput (cycles/sec) on an 82k-node
//! synthetic graph — the order-of-magnitude proxy for Slashdot.
//!
//! Run: `cargo bench --bench parity_gate`
//! Compare: PyTorch+Triton baseline figure goes here once recorded.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use holonomy_learn::ops::scatter::scatter_mean_forward;
use holonomy_learn::NagareRuntime;
use hymeko_graph::{clifford_fir_forward, CliffordFIR, TopKCyclesBatch};

// ── Synthetic data helpers ────────────────────────────────────────────────────

/// Deterministic LCG — no external RNG dep in bench binary.
fn lcg_next(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *state
}

/// Build a synthetic `TopKCyclesBatch` with `n_cycles` k-cycles over
/// `n_vertices` vertices. Vertex indices and signs are drawn from the LCG.
fn make_synthetic_batch(n_vertices: usize, n_cycles: usize, k: usize) -> TopKCyclesBatch {
    let mut state: u64 = 0xdead_beef_cafe_babe;
    let mut cycles = Vec::with_capacity(n_cycles * k);
    let mut signs = Vec::with_capacity(n_cycles * k);
    for _ in 0..n_cycles * k {
        let r = lcg_next(&mut state);
        cycles.push(((r >> 33) as u32) % n_vertices as u32);
        signs.push(if (r >> 63) == 0 { 1i8 } else { -1i8 });
    }
    TopKCyclesBatch {
        cycles,
        signs,
        scores: vec![0.0f64; n_cycles],
        k,
    }
}

fn make_features(n_vertices: usize, d: usize) -> Vec<f32> {
    let total = n_vertices * d;
    (0..total).map(|i| i as f32 / total as f32).collect()
}

// ── Benchmarks ────────────────────────────────────────────────────────────────

/// CliffordFIR forward pass — sweeps over number of cycles.
///
/// This is the innermost hot kernel: measures how fast the sign-aware
/// FIR can aggregate per-vertex features into per-cycle features.
fn bench_fir_forward(c: &mut Criterion) {
    // Slashdot order-of-magnitude: ~82 k nodes, k=4 cycles.
    const N_VERTICES: usize = 82_140;
    const D: usize = 32;
    const K: usize = 4;

    let features = make_features(N_VERTICES, D);
    let fir = CliffordFIR::signed_mean(K);

    let mut group = c.benchmark_group("fir_forward/n_cycles");
    for &n_cycles in &[1_000usize, 10_000, 100_000, 500_000] {
        let batch = make_synthetic_batch(N_VERTICES, n_cycles, K);
        group.bench_with_input(BenchmarkId::from_parameter(n_cycles), &n_cycles, |b, _| {
            b.iter(|| {
                clifford_fir_forward(
                    black_box(&batch),
                    black_box(&features),
                    black_box(N_VERTICES),
                    black_box(D),
                    black_box(&fir),
                )
            });
        });
    }
    group.finish();
}

/// Scatter-mean forward — 100 k cycles, Slashdot-class.
fn bench_scatter_mean(c: &mut Criterion) {
    const N_VERTICES: usize = 82_140;
    const D: usize = 32;
    const K: usize = 4;
    const N_CYCLES: usize = 100_000;

    let batch = make_synthetic_batch(N_VERTICES, N_CYCLES, K);
    let per_cycle = vec![0.1f32; N_CYCLES * D];

    c.bench_function("scatter_mean/100k_cycles", |b| {
        b.iter(|| {
            scatter_mean_forward(
                black_box(&batch.cycles),
                black_box(K),
                black_box(&per_cycle),
                black_box(D),
                black_box(N_VERTICES),
            )
        });
    });
}

/// Full NagareRuntime forward pass (FIR + scatter + linear).
/// This is the latency number to compare against the ≤10 ms reflex budget.
fn bench_runtime_predict(c: &mut Criterion) {
    const N_VERTICES: usize = 82_140;
    const D: usize = 32;
    const K: usize = 4;
    const N_CYCLES: usize = 100_000;

    let batch = make_synthetic_batch(N_VERTICES, N_CYCLES, K);
    let features = make_features(N_VERTICES, D);
    let rt = NagareRuntime::new(K, D, 1, 1e-3, 42);

    c.bench_function("runtime_predict/100k_cycles", |b| {
        b.iter(|| {
            rt.predict(
                black_box(&batch),
                black_box(&features),
                black_box(N_VERTICES),
            )
        });
    });
}

/// Full NagareRuntime::step() — forward + backward + Adam.
/// This is the number to watch: it must stay under the ≤10 ms reflex budget
/// at Slashdot scale and must reach AUC parity with PyTorch+Triton.
fn bench_runtime_step(c: &mut Criterion) {
    const N_VERTICES: usize = 82_140;
    const D: usize = 32;
    const K: usize = 4;
    const N_CYCLES: usize = 100_000;

    let batch = make_synthetic_batch(N_VERTICES, N_CYCLES, K);
    let features = make_features(N_VERTICES, D);
    // Binary targets: even vertices → 1, odd → 0.
    let targets: Vec<f32> = (0..N_VERTICES)
        .map(|v| if v % 2 == 0 { 1.0 } else { 0.0 })
        .collect();

    let mut rt = NagareRuntime::new(K, D, 1, 1e-3, 42);

    c.bench_function("runtime_step/100k_cycles", |b| {
        b.iter(|| {
            rt.step(
                black_box(&batch),
                black_box(&features),
                black_box(N_VERTICES),
                black_box(&targets),
            )
        });
    });
}

criterion_group!(
    benches,
    bench_fir_forward,
    bench_scatter_mean,
    bench_runtime_predict,
    bench_runtime_step
);
criterion_main!(benches);
