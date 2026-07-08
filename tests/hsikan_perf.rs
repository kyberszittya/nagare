//! Phase 1d (memory) — the `chunk_t` streaming cap holds peak heap bounded.
//!
//! The naive forward materialises `(T, k, S, d)` intermediates (~327 MB at
//! Bitcoin-Alpha scale). `hsikan_forward_chunked` streams T so peak stays
//! `O(chunk · k · S · d)`. A std-only tracking global allocator measures peak live
//! bytes; at T=5000 the chunked forward's peak is a large factor below the naive
//! forward's. (Latency is in `benches/hsikan_bench.rs` — criterion, §10.)

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use holonomy_learn::{
    hsikan_forward, hsikan_forward_chunked, HsikanConfig, HsikanEdges, HsikanParams,
};

struct Track;
static LIVE: AtomicUsize = AtomicUsize::new(0);
static PEAK: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for Track {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let p = System.alloc(layout);
        if !p.is_null() {
            let now = LIVE.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            PEAK.fetch_max(now, Ordering::Relaxed);
        }
        p
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size(), Ordering::Relaxed);
        System.dealloc(ptr, layout);
    }
}

#[global_allocator]
static ALLOC: Track = Track;

fn reset_peak() {
    PEAK.store(LIVE.load(Ordering::Relaxed), Ordering::Relaxed);
}
fn peak() -> usize {
    PEAK.load(Ordering::Relaxed)
}

fn deterministic(n: usize, scale: f32, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| scale * ((i as f32 * 0.7 + phase).sin()))
        .collect()
}

#[test]
fn chunked_forward_is_memory_bounded() {
    let (t, k, d, s, grid, cheb_k) = (5000usize, 4usize, 32usize, 2usize, 6usize, 4usize);
    let n_nodes = 1000usize;
    let x = deterministic(n_nodes * d, 0.3, 0.1);
    let inner = deterministic(s * d * cheb_k, 0.2, 0.3);
    let outer = deterministic(s * d * cheb_k, 0.2, 0.9);
    let gate_w = deterministic(d * d, 0.1, 0.5);
    let gate_b = vec![-1.0f32; d];
    let vertices: Vec<u32> = (0..t * k)
        .map(|i| (i.wrapping_mul(2_654_435_761) % n_nodes) as u32)
        .collect();
    let signs: Vec<i8> = (0..t * k)
        .map(|i| if i % 2 == 0 { 1 } else { -1 })
        .collect();

    let params = HsikanParams {
        inner_coef: &inner,
        outer_coef: &outer,
        gate_w: &gate_w,
        gate_b: &gate_b,
    };
    let edges = HsikanEdges {
        vertices: &vertices,
        signs: &signs,
    };
    let cfg = HsikanConfig::new(t, k, d, s, grid, cheb_k, true);

    // Naive forward (retains the full (T,k,S,d) cache).
    reset_peak();
    {
        let (h, _c) = hsikan_forward(params, &x, edges, cfg);
        std::hint::black_box(&h);
    }
    let naive = peak();

    // Chunked forward-only (each chunk's cache dropped).
    let chunk = 256;
    reset_peak();
    {
        let h = hsikan_forward_chunked(params, &x, edges, cfg, chunk);
        std::hint::black_box(&h);
    }
    let chunked = peak();

    eprintln!(
        "peak heap over baseline: naive={} KiB  chunked({chunk})={} KiB  ({:.1}× smaller)",
        naive / 1024,
        chunked / 1024,
        naive as f64 / chunked.max(1) as f64
    );
    // Streaming must bound peak well below the full-materialisation cost.
    assert!(
        chunked * 3 < naive,
        "chunk_t cap did not bound peak: naive={naive} chunked={chunked}"
    );
    // And the chunked output must equal the naive h_e (correctness under load).
    let (naive_he, _) = hsikan_forward(params, &x, edges, cfg);
    let chunked_he = hsikan_forward_chunked(params, &x, edges, cfg, chunk);
    assert_eq!(naive_he.len(), chunked_he.len());
    assert!(naive_he
        .iter()
        .zip(&chunked_he)
        .all(|(a, b)| (a - b).abs() < 1e-6));
}
