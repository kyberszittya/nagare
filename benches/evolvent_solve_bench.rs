//! E5-follow-up (criterion, §10) — does the multifrontal solve's analytic O(d·w³)
//! flop win show up in WALL-CLOCK vs the dense O(d³) solve?
//!
//! Both solvers are the real ones in the crate and give the identical answer (E5
//! `branching_tree_equals_dense_at_bounded_width`): `JunctionTreeCholesky::solve`
//! (multifrontal Cholesky on the hypergraph clique tree) vs `InfoEvolventHead::solve`
//! (dense Gaussian elimination on the full assembled `J`). Measured on the E5
//! branching binary clique tree at depths 4–7 (d = 105 … 889).
//!
//! Note: the dense baseline is Gaussian elimination (~2× the flops of a pure dense
//! Cholesky), so the measured ratio is a mild over-estimate of the pure-algorithmic
//! win — it is the honest comparison of the two solvers that actually exist.
//!
//! Run: `cargo bench --bench evolvent_solve_bench`.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use holonomy_learn::{balanced_binary_tree, InfoEvolventHead, JunctionTreeCholesky};

fn lcg(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
}

/// Prime a multifrontal solver and a dense solver with the SAME measurements
/// (homed at cliques for the tree; as global vectors for the dense head).
fn setup(depth: usize) -> (JunctionTreeCholesky, InfoEvolventHead, usize) {
    let (cliques, d) = balanced_binary_tree(depth, 2, 3);
    let mut jt = JunctionTreeCholesky::new(cliques.clone(), 1.0, d);
    let mut dense = InfoEvolventHead::new(d, 1.0);
    let mut st = 0x1234_5678_9abc_def0u64;
    for (c, cl) in cliques.iter().enumerate() {
        let m = cl.vars.len();
        for _ in 0..20 {
            let local: Vec<f32> = (0..m).map(|_| lcg(&mut st)).collect();
            let y = lcg(&mut st);
            jt.update(c, &local, y);
            let mut global = vec![0.0f32; d];
            for (i, &v) in cl.vars.iter().enumerate() {
                global[v] = local[i];
            }
            dense.update(&global, y);
        }
    }
    (jt, dense, d)
}

fn bench(c: &mut Criterion) {
    let mut group = c.benchmark_group("evolvent_solve");
    group.sample_size(30);
    for &depth in &[4usize, 5, 6, 7] {
        let (mut jt, dense, d) = setup(depth);
        group.bench_function(BenchmarkId::new("multifrontal", d), |b| {
            b.iter(|| black_box(jt.solve()))
        });
        group.bench_function(BenchmarkId::new("dense_gauss", d), |b| {
            b.iter(|| black_box(dense.solve()))
        });
    }
    group.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
