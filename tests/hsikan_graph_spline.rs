//! Testing HSiKAN on a real signed graph with **both** spline bases — the "test it" half
//! of the spline-pluggable ask (CR-Chebyshev vs Kochanek-Bartels).
//!
//! Model (HSiKAN is the *only* nonlinearity, so the spline basis is the sole varying factor):
//!   Iris features `x (n,4)` → build a leakage-free kNN signed graph, enumerate signed
//!   triangles → `hsikan` over the triangles-as-hyperedges → `scatter_mean (cycle→sample)`
//!   → linear → softmax₃, transductive (train on train labels, eval test).
//! The two arms differ **only** in `spline_kind`; same splits, same seeds, same everything
//! else. Multi-seed median (§3); the verdict is reported, not assumed.

use holonomy_learn::{
    accuracy_k, build_signed_cycle_pool, hsikan_backward, hsikan_forward, linear_backward,
    linear_forward, load_csv, scatter_mean_backward, scatter_mean_forward, softmax_k, GraphPool,
    HsikanConfig, HsikanEdges, HsikanParams, LinearLayer, SplineKind, Tabular,
};
use rand::{rngs::StdRng, Rng, SeedableRng};

const S: usize = 2; // sign branches
const GRID: usize = 5;
const CHEB: usize = 4;

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

fn test_acc(logits: &[f32], y: &[usize], idx: &[usize], k: usize) -> f32 {
    let sub: Vec<f32> = idx
        .iter()
        .flat_map(|&i| logits[i * k..i * k + k].to_vec())
        .collect();
    let sub_y: Vec<usize> = idx.iter().map(|&i| y[i]).collect();
    accuracy_k(&sub, &sub_y, idx.len(), k)
}

/// Transductive HSiKAN node classifier over the signed triangles, parameterised by basis.
struct HsikanGraph {
    inner: Vec<f32>,
    outer: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    readout: LinearLayer,
    d: usize,
    k: usize,
    kind: SplineKind,
}

impl HsikanGraph {
    fn new(d: usize, k: usize, kind: SplineKind, seed: u64) -> Self {
        // param_len is independent of n_edges; a placeholder edge count sizes the buffers.
        let plen = HsikanConfig::new(1, 3, d, S, GRID, CHEB, true)
            .with_spline_kind(kind)
            .param_len();
        let mut rng = StdRng::seed_from_u64(seed.wrapping_add(300));
        let mut small = |n: usize| -> Vec<f32> {
            (0..n)
                .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
                .collect()
        };
        Self {
            inner: small(plen),
            outer: small(plen),
            gw: small(d * d),
            gb: vec![-1.0; d],
            readout: LinearLayer::new(d, k, seed.wrapping_add(301)),
            d,
            k,
            kind,
        }
    }

    fn cfg(&self, nc: usize) -> HsikanConfig {
        HsikanConfig::new(nc, 3, self.d, S, GRID, CHEB, true).with_spline_kind(self.kind)
    }
    fn params(&self) -> HsikanParams<'_> {
        HsikanParams {
            inner_coef: &self.inner,
            outer_coef: &self.outer,
            gate_w: &self.gw,
            gate_b: &self.gb,
        }
    }
    fn param_count(&self) -> usize {
        self.inner.len() + self.outer.len() + self.gw.len() + self.gb.len() + self.readout.w.len()
    }

    /// Per-sample logits `(n, k)` from the full HSiKAN→scatter→linear forward.
    fn logits(&self, pool: &GraphPool, x: &[f32], n: usize) -> Vec<f32> {
        let edges = HsikanEdges {
            vertices: &pool.batch.cycles,
            signs: &pool.batch.signs,
        };
        let (h_e, _) = hsikan_forward(self.params(), x, edges, self.cfg(pool.n_cycles));
        let (xg, _c) = scatter_mean_forward(&pool.batch.cycles, 3, &h_e, self.d, n);
        linear_forward(&self.readout, &xg)
    }

    /// One transductive SGD step (masked CE over train rows, mean over train).
    fn step(&mut self, pool: &GraphPool, x: &[f32], y: &[usize], tr: &[usize], n: usize, lr: f32) {
        let (nc, d, k) = (pool.n_cycles, self.d, self.k);
        let cfg = self.cfg(nc);
        let edges = HsikanEdges {
            vertices: &pool.batch.cycles,
            signs: &pool.batch.signs,
        };
        let (h_e, hcache) = hsikan_forward(self.params(), x, edges, cfg);
        let (xg, counts) = scatter_mean_forward(&pool.batch.cycles, 3, &h_e, d, n);
        let logits = linear_forward(&self.readout, &xg);
        let probs = softmax_k(&logits, n, k);

        let inv = 1.0 / tr.len() as f32;
        let mut grad_logits = vec![0.0f32; n * k];
        for &r in tr {
            for j in 0..k {
                grad_logits[r * k + j] = (probs[r * k + j] - f32::from(j == y[r])) * inv;
            }
        }
        let (grad_xg, grad_readout) = linear_backward(&self.readout, &xg, &grad_logits);
        let grad_he = scatter_mean_backward(&pool.batch.cycles, 3, &grad_xg, d, &counts, n);
        let hb = hsikan_backward(self.params(), edges, &hcache, &grad_he, cfg);

        for (p, g) in self.inner.iter_mut().zip(&hb.grad_inner_coef) {
            *p -= lr * g;
        }
        for (p, g) in self.outer.iter_mut().zip(&hb.grad_outer_coef) {
            *p -= lr * g;
        }
        for (p, g) in self.gw.iter_mut().zip(&hb.grad_gate_w) {
            *p -= lr * g;
        }
        for (p, g) in self.gb.iter_mut().zip(&hb.grad_gate_b) {
            *p -= lr * g;
        }
        for (p, g) in self.readout.w.iter_mut().zip(&grad_readout.w) {
            *p -= lr * g;
        }
        for (p, g) in self.readout.b.iter_mut().zip(&grad_readout.b) {
            *p -= lr * g;
        }
    }
}

fn run(
    iris: &Tabular,
    pool: &GraphPool,
    tr: &[usize],
    te: &[usize],
    kind: SplineKind,
    seed: u64,
) -> f32 {
    let mut m = HsikanGraph::new(iris.d, iris.n_classes, kind, seed);
    for _ in 0..300 {
        m.step(pool, &iris.x, &iris.y, tr, iris.n, 0.1);
    }
    test_acc(
        &m.logits(pool, &iris.x, iris.n),
        &iris.y,
        te,
        iris.n_classes,
    )
}

#[test]
fn hsikan_graph_chebyshev_vs_kochanek_bartels_on_iris() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/iris.csv");
    let iris = load_csv(&std::fs::read_to_string(path).expect("iris fixture"));
    // One graph per split; both bases classify over the SAME signed triangles.
    let (mut cheb, mut kb) = (Vec::new(), Vec::new());
    let mut n_cycles = 0usize;
    for seed in 0..5u64 {
        let (tr, te) = iris.split(0.25, seed);
        // kNN=6 signed graph → a lean-but-ample triangle set (keeps the suite fast).
        let pool = build_signed_cycle_pool(&iris.x, iris.n, iris.d, 6, 3);
        n_cycles = pool.n_cycles;
        let c = run(&iris, &pool, &tr, &te, SplineKind::ChebyshevCr, seed);
        let k = run(&iris, &pool, &tr, &te, SplineKind::KochanekBartels, seed);
        eprintln!("seed {seed}: Cheb {c:.3}  KB {k:.3}");
        cheb.push(c);
        kb.push(k);
    }
    let (cm, km) = (median(cheb.clone()), median(kb.clone()));
    // Param-count context (KB is a denser parametrisation than Chebyshev).
    let np_c = HsikanGraph::new(iris.d, iris.n_classes, SplineKind::ChebyshevCr, 0).param_count();
    let np_k =
        HsikanGraph::new(iris.d, iris.n_classes, SplineKind::KochanekBartels, 0).param_count();
    eprintln!(
        "HSiKAN-on-Iris-graph ({n_cycles} signed triangles), median held-out acc over 5 seeds:"
    );
    eprintln!("  Chebyshev-CR   {cm:.3}   ({np_c} params)");
    eprintln!("  Kochanek-Bartels {km:.3}   ({np_k} params)");
    eprintln!(
        "  verdict: KB {} Chebyshev-CR (Δ {:.3})",
        match km.partial_cmp(&cm) {
            Some(std::cmp::Ordering::Greater) => "beats",
            Some(std::cmp::Ordering::Less) => "trails",
            _ => "ties",
        },
        km - cm
    );
    // Both bases must produce a working classifier; which wins is the measurement.
    assert!(
        cm >= 0.7 && km >= 0.7,
        "a spline basis failed to classify: Cheb {cm:.3}, KB {km:.3}"
    );
}
