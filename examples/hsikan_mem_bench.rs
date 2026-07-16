//! HSiKAN forward(+backward) at scale — time + (externally measured) peak RSS.
//! No autograd tape: the backward is closed-form, so peak memory is params + one
//! cache, not a growing computation graph. Compare to the PyTorch arm under
//! `/usr/bin/time -l` (macOS max RSS) / `/usr/bin/time -v` (Linux).
//!
//! Args: `--edges N` (hyperedges), `--dim D` (hidden dim), `--fwd-only`
//! (deploy/inference path via `hsikan_forward_chunked`, no backward).
//!
//! Run: `/usr/bin/time -l cargo run --release --example hsikan_mem_bench -- --edges 50000 --dim 16`

use holonomy_learn::{
    hsikan_backward, hsikan_forward, hsikan_forward_chunked, HsikanConfig, HsikanEdges,
    HsikanParams,
};
use std::time::Instant;

const CB: usize = 6; // Chebyshev order
const SC: usize = 2; // sign branches
const ITERS: usize = 100;

fn main() {
    let mut args = std::env::args();
    let (mut t, mut d, mut fwd_only) = (50_000usize, 16usize, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--edges" => t = args.next().and_then(|s| s.parse().ok()).unwrap_or(t),
            "--dim" => d = args.next().and_then(|s| s.parse().ok()).unwrap_or(d),
            "--fwd-only" => fwd_only = true,
            _ => {}
        }
    }
    let n = (t / 2).max(64); // vertices

    let mut s = 0x1234_5678u64;
    let mut nx = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0 // [-1,1]
    };
    let x: Vec<f32> = (0..n * d).map(|_| nx()).collect();
    let vertices: Vec<u32> = (0..t * 3)
        .map(|_| (nx().abs() * n as f32) as u32 % n as u32)
        .collect();
    let signs: Vec<i8> = (0..t * 3)
        .map(|_| if nx() > 0.0 { 1 } else { -1 })
        .collect();
    let inner: Vec<f32> = (0..SC * d * CB).map(|_| 0.3 * nx()).collect();
    let outer: Vec<f32> = (0..SC * d * CB).map(|_| 0.3 * nx()).collect();
    let gw: Vec<f32> = (0..d * d).map(|_| 0.2 * nx()).collect();
    let gb: Vec<f32> = vec![-1.0f32; d];

    let cfg = HsikanConfig::new(t, 3, d, SC, 8, CB, true);
    let edges = HsikanEdges {
        vertices: &vertices,
        signs: &signs,
    };
    let params = HsikanParams {
        inner_coef: &inner,
        outer_coef: &outer,
        gate_w: &gw,
        gate_b: &gb,
    };
    let grad_he: Vec<f32> = (0..t * d).map(|_| 0.01 * nx()).collect();

    let time0 = Instant::now();
    let mut checksum = 0.0f64;
    for _ in 0..ITERS {
        if fwd_only {
            // Deploy / inference path: bounded-memory streaming forward, no cache.
            let h = hsikan_forward_chunked(params, &x, edges, cfg, t.div_ceil(64).max(1));
            checksum += h[0] as f64;
        } else {
            let (h, cache) = hsikan_forward(params, &x, edges, cfg);
            let bwd = hsikan_backward(params, edges, &cache, &grad_he, cfg);
            checksum += h[0] as f64 + bwd.grad_inner_coef[0] as f64;
        }
    }
    let ms = time0.elapsed().as_secs_f64() * 1e3 / ITERS as f64;
    let mode = if fwd_only { "fwd-only" } else { "fwd+bwd " };
    println!("Nagare HSiKAN  edges={t}  d={d}  {mode} {ms:.3} ms/iter  ({ITERS} iters)  [checksum {checksum:.4}]");
}
