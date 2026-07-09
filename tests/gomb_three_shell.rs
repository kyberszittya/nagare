//! Phase 2c — the full three-shell Gömb cascade with the CPML inner core.
//!
//! Composes all three role-distinct shells end-to-end (cascade.py order), closed-form:
//!   X → **outer** `gomb_outer` (M Clifford-FIR banks, V1) → scatter_mean →
//!       **middle** `hsikan` (signed CR-spline, V4) → scatter_mean →
//!       **inner** CPML tier core (degree-tier routing + concat readout, IT) → readout.
//! The inner core is the new piece (`cpml_tier`): vertices are stratified into degree tiers,
//! each tier aggregates only the cycles that touch it, and `concat(X_core, H₀…H_{L-1})` feeds
//! the head. Trained by the composed backward (readout → inner tiers → scatter → hsikan →
//! scatter → outer), node features frozen.
//!
//! Discriminating add: **does the inner tier-stratification earn its weight?** Labels come
//! from a fixed L=3 three-shell teacher; an **L=3** student (tiered) vs an **L=1** student
//! (flat, single aggregator, same everything else), over **5 seeds** (§3 — a single seed here
//! was a tiers-favorable draw). Reported, not asserted (constructed teacher target); only the
//! cascade-learns gate is asserted.

use holonomy_learn::{
    cross_entropy, cycle_incidence_degrees, gomb_outer_backward, gomb_outer_forward,
    hsikan_backward, hsikan_forward, linear_backward, linear_forward, scatter_mean_backward,
    scatter_mean_forward, softmax2, tier_cycle_indices, HsikanConfig, HsikanEdges, HsikanParams,
    LinearLayer, TierSpec,
};
use hymeko_graph::{CliffordFIR, TopKCyclesBatch};
use rand::{rngs::StdRng, Rng, SeedableRng};

const V: usize = 12;
const K: usize = 3;
const D_FEAT: usize = 2;
const M: usize = 2;
const HID: usize = M * D_FEAT; // outer width = hsikan hidden = inner core input width
const D_LAYER: usize = 3; // inner per-tier aggregator output
const S: usize = 2;
const GRID: usize = 6;
const CHEB: usize = 4;

fn rand_vec(n: usize, scale: f32, rng: &mut StdRng) -> Vec<f32> {
    (0..n)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
        .collect()
}

fn make_batch(n_cycles: usize, rng: &mut StdRng) -> TopKCyclesBatch {
    TopKCyclesBatch {
        cycles: (0..n_cycles * K)
            .map(|_| rng.random_range(0..V as u32))
            .collect(),
        signs: (0..n_cycles * K)
            .map(|_| if rng.random::<bool>() { 1i8 } else { -1 })
            .collect(),
        scores: vec![0.0f64; n_cycles],
        k: K,
    }
}

fn gather_rows(x: &[f32], idx: &[u32], width: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; idx.len() * width];
    for (j, &v) in idx.iter().enumerate() {
        out[j * width..j * width + width]
            .copy_from_slice(&x[v as usize * width..v as usize * width + width]);
    }
    out
}

/// Per-tier inner aggregator: gather corners → linear → mean over corners → scatter_mean.
struct TierCache {
    sub_cycles: Vec<u32>,
    corners: Vec<f32>,
    counts: Vec<u32>,
    m_sub: usize,
}

fn inner_tier_forward(
    x_core: &[f32],
    cycles: &[u32],
    tier_of: &[usize],
    ell: usize,
    lin: &LinearLayer,
) -> (Vec<f32>, TierCache) {
    let idx = tier_cycle_indices(cycles, K, tier_of, ell);
    let mut sub_cycles = vec![0u32; idx.len() * K];
    for (j, &c) in idx.iter().enumerate() {
        sub_cycles[j * K..j * K + K].copy_from_slice(&cycles[c * K..c * K + K]);
    }
    let m_sub = idx.len();
    if m_sub == 0 {
        return (
            vec![0.0f32; V * D_LAYER],
            TierCache {
                sub_cycles,
                corners: Vec::new(),
                counts: Vec::new(),
                m_sub,
            },
        );
    }
    let corners = gather_rows(x_core, &sub_cycles, HID); // (m_sub*K, HID)
    let lin_out = linear_forward(lin, &corners); // (m_sub*K, D_LAYER)
    let mut per_cycle = vec![0.0f32; m_sub * D_LAYER];
    for c in 0..m_sub {
        for j in 0..D_LAYER {
            let s: f32 = (0..K).map(|i| lin_out[(c * K + i) * D_LAYER + j]).sum();
            per_cycle[c * D_LAYER + j] = s / K as f32;
        }
    }
    let (h, counts) = scatter_mean_forward(&sub_cycles, K, &per_cycle, D_LAYER, V);
    (
        h,
        TierCache {
            sub_cycles,
            corners,
            counts,
            m_sub,
        },
    )
}

/// Backward for one tier → (grad w.r.t. x_core, grad w.r.t. the tier's linear).
fn inner_tier_backward(
    grad_h: &[f32],
    cache: &TierCache,
    lin: &LinearLayer,
) -> (Vec<f32>, holonomy_learn::LinearLayer) {
    let mut grad_x_core = vec![0.0f32; V * HID];
    if cache.m_sub == 0 {
        return (grad_x_core, lin.zero_grad());
    }
    let grad_per_cycle =
        scatter_mean_backward(&cache.sub_cycles, K, grad_h, D_LAYER, &cache.counts, V);
    // Broadcast per-cycle grad back to its K corners (mean → /K).
    let mut grad_lin_out = vec![0.0f32; cache.m_sub * K * D_LAYER];
    for c in 0..cache.m_sub {
        for i in 0..K {
            for j in 0..D_LAYER {
                grad_lin_out[(c * K + i) * D_LAYER + j] =
                    grad_per_cycle[c * D_LAYER + j] / K as f32;
            }
        }
    }
    let (grad_corners, grad_lin) = linear_backward(lin, &cache.corners, &grad_lin_out);
    for (corner, &v) in cache.sub_cycles.iter().enumerate() {
        for j in 0..HID {
            grad_x_core[v as usize * HID + j] += grad_corners[corner * HID + j];
        }
    }
    (grad_x_core, grad_lin)
}

/// The full three-shell cascade with a configurable inner tier count.
struct ThreeShell {
    banks: Vec<CliffordFIR>,
    inner: Vec<f32>,
    outer_c: Vec<f32>,
    gw: Vec<f32>,
    gb: Vec<f32>,
    tier_lins: Vec<LinearLayer>,
    readout: LinearLayer,
    n_tiers: usize,
}

impl ThreeShell {
    fn new(n_tiers: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let banks = (0..M)
            .map(|_| CliffordFIR::new(rand_vec(K, 0.4, &mut rng), rand_vec(K, 0.4, &mut rng)))
            .collect();
        let tier_lins = (0..n_tiers)
            .map(|t| LinearLayer::new(HID, D_LAYER, seed + 10 + t as u64))
            .collect();
        Self {
            banks,
            inner: rand_vec(S * HID * CHEB, 0.3, &mut rng),
            outer_c: rand_vec(S * HID * CHEB, 0.3, &mut rng),
            gw: rand_vec(HID * HID, 0.2, &mut rng),
            gb: vec![-1.0; HID],
            tier_lins,
            readout: LinearLayer::new(HID + n_tiers * D_LAYER, 2, seed.wrapping_add(1)),
            n_tiers,
        }
    }

    fn params(&self) -> HsikanParams<'_> {
        HsikanParams {
            inner_coef: &self.inner,
            outer_coef: &self.outer_c,
            gate_w: &self.gw,
            gate_b: &self.gb,
        }
    }
    fn cfg(&self, nc: usize) -> HsikanConfig {
        HsikanConfig::new(nc, K, HID, S, GRID, CHEB, true)
    }

    /// Full forward → per-vertex logits `(V, 2)`, plus every cache the backward needs.
    #[allow(clippy::type_complexity)]
    fn forward(
        &self,
        batch: &TopKCyclesBatch,
        x: &[f32],
        tier_of: &[usize],
    ) -> (
        Vec<u32>,
        Vec<u32>,
        Vec<f32>,
        holonomy_learn::HsikanCache,
        Vec<f32>,
        Vec<TierCache>,
        Vec<f32>,
    ) {
        let y_out = gomb_outer_forward(batch, x, &self.banks, V, D_FEAT);
        let (x_out, cnt_out) = scatter_mean_forward(&batch.cycles, K, &y_out, HID, V);
        let edges = HsikanEdges {
            vertices: &batch.cycles,
            signs: &batch.signs,
        };
        let (h_mid, hcache) = hsikan_forward(self.params(), &x_out, edges, self.cfg(batch.len()));
        let (x_core, cnt_mid) = scatter_mean_forward(&batch.cycles, K, &h_mid, HID, V);
        let mut x_final = vec![0.0f32; V * (HID + self.n_tiers * D_LAYER)];
        let stride = HID + self.n_tiers * D_LAYER;
        for v in 0..V {
            x_final[v * stride..v * stride + HID].copy_from_slice(&x_core[v * HID..v * HID + HID]);
        }
        let mut tier_caches = Vec::with_capacity(self.n_tiers);
        for (ell, lin) in self.tier_lins.iter().enumerate() {
            let (h_ell, cache) = inner_tier_forward(&x_core, &batch.cycles, tier_of, ell, lin);
            for v in 0..V {
                let base = v * stride + HID + ell * D_LAYER;
                x_final[base..base + D_LAYER]
                    .copy_from_slice(&h_ell[v * D_LAYER..v * D_LAYER + D_LAYER]);
            }
            tier_caches.push(cache);
        }
        let logits = linear_forward(&self.readout, &x_final);
        (
            cnt_out,
            cnt_mid,
            x_final,
            hcache,
            x_core,
            tier_caches,
            logits,
        )
    }

    /// One SGD step through the full composed backward; returns pre-update loss.
    fn step(
        &mut self,
        batch: &TopKCyclesBatch,
        x: &[f32],
        tier_of: &[usize],
        labels: &[f32],
        lr: f32,
    ) -> f32 {
        let (cnt_out, cnt_mid, x_final, hcache, _x_core, tier_caches, logits) =
            self.forward(batch, x, tier_of);
        let y_usize: Vec<usize> = labels.iter().map(|&l| l as usize).collect();
        let loss = cross_entropy(&logits, &y_usize).loss;
        // BCE gradient over vertices.
        let mut grad_logits = vec![0.0f32; V * 2];
        for v in 0..V {
            let (p0, p1) = softmax2(logits[2 * v], logits[2 * v + 1]);
            grad_logits[2 * v] = (p0 - f32::from(labels[v] == 0.0)) / V as f32;
            grad_logits[2 * v + 1] = (p1 - f32::from(labels[v] == 1.0)) / V as f32;
        }
        let (grad_x_final, grad_readout) = linear_backward(&self.readout, &x_final, &grad_logits);
        let stride = HID + self.n_tiers * D_LAYER;
        // Split grad_x_final → direct grad on x_core + per-tier grad_H.
        let mut grad_x_core = vec![0.0f32; V * HID];
        for v in 0..V {
            grad_x_core[v * HID..v * HID + HID]
                .copy_from_slice(&grad_x_final[v * stride..v * stride + HID]);
        }
        let mut grad_tier_lins: Vec<LinearLayer> = Vec::with_capacity(self.n_tiers);
        for (ell, lin) in self.tier_lins.iter().enumerate() {
            let mut grad_h = vec![0.0f32; V * D_LAYER];
            for v in 0..V {
                let base = v * stride + HID + ell * D_LAYER;
                grad_h[v * D_LAYER..v * D_LAYER + D_LAYER]
                    .copy_from_slice(&grad_x_final[base..base + D_LAYER]);
            }
            let (gx, glin) = inner_tier_backward(&grad_h, &tier_caches[ell], lin);
            for (a, b) in grad_x_core.iter_mut().zip(&gx) {
                *a += b;
            }
            grad_tier_lins.push(glin);
        }
        // grad_x_core is grad w.r.t. x_mid (scatter of h_mid).
        let grad_h_mid = scatter_mean_backward(&batch.cycles, K, &grad_x_core, HID, &cnt_mid, V);
        let edges = HsikanEdges {
            vertices: &batch.cycles,
            signs: &batch.signs,
        };
        let hb = hsikan_backward(
            self.params(),
            edges,
            &hcache,
            &grad_h_mid,
            self.cfg(batch.len()),
        );
        let grad_y_out = scatter_mean_backward(&batch.cycles, K, &hb.grad_x, HID, &cnt_out, V);
        let (_gf, grad_banks) = gomb_outer_backward(batch, x, &self.banks, &grad_y_out, V, D_FEAT);

        // Updates (features frozen).
        for (bank, gb) in self.banks.iter_mut().zip(&grad_banks) {
            for (a, ga) in bank.a.iter_mut().zip(&gb.a) {
                *a -= lr * ga;
            }
            for (b, gbb) in bank.b.iter_mut().zip(&gb.b) {
                *b -= lr * gbb;
            }
        }
        for (p, g) in self.inner.iter_mut().zip(&hb.grad_inner_coef) {
            *p -= lr * g;
        }
        for (p, g) in self.outer_c.iter_mut().zip(&hb.grad_outer_coef) {
            *p -= lr * g;
        }
        for (p, g) in self.gw.iter_mut().zip(&hb.grad_gate_w) {
            *p -= lr * g;
        }
        for (p, g) in self.gb.iter_mut().zip(&hb.grad_gate_b) {
            *p -= lr * g;
        }
        for (lin, glin) in self.tier_lins.iter_mut().zip(&grad_tier_lins) {
            for (w, gw) in lin.w.iter_mut().zip(&glin.w) {
                *w -= lr * gw;
            }
            for (b, gb) in lin.b.iter_mut().zip(&glin.b) {
                *b -= lr * gb;
            }
        }
        for (w, gw) in self.readout.w.iter_mut().zip(&grad_readout.w) {
            *w -= lr * gw;
        }
        for (b, gb) in self.readout.b.iter_mut().zip(&grad_readout.b) {
            *b -= lr * gb;
        }
        loss
    }

    fn logits_of(&self, batch: &TopKCyclesBatch, x: &[f32], tier_of: &[usize]) -> Vec<f32> {
        self.forward(batch, x, tier_of).6
    }
}

fn acc(logits: &[f32], labels: &[f32]) -> f32 {
    let correct = (0..labels.len())
        .filter(|&v| ((logits[2 * v + 1] > logits[2 * v]) as usize as f32) == labels[v])
        .count();
    correct as f32 / labels.len() as f32
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

/// One seed: fresh graph + features + L=3 teacher labels, then train an L=3 (tiered) and an
/// L=1 (flat) student. Returns `(full_init_bce, full_bce, flat_bce, full_acc, flat_acc)`.
fn run_seed(seed: u64) -> (f32, f32, f32, f32, f32) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x = rand_vec(V * D_FEAT, 0.5, &mut rng);
    let batch = make_batch(30, &mut rng);
    let degrees = cycle_incidence_degrees(&batch.cycles, V);
    let tier3 = TierSpec::uniform(3).assign(&degrees);
    let tier1 = TierSpec::uniform(1).assign(&degrees);

    // Labels from a fixed L=3 three-shell teacher (median split → balanced).
    let teacher = ThreeShell::new(3, seed.wrapping_add(99));
    let tl = teacher.logits_of(&batch, &x, &tier3);
    let margins: Vec<f32> = (0..V).map(|v| tl[2 * v + 1] - tl[2 * v]).collect();
    let med = median(margins.clone());
    let labels: Vec<f32> = margins.iter().map(|&m| f32::from(m > med)).collect();

    let mut full = ThreeShell::new(3, seed.wrapping_add(3)); // tiered inner (L=3)
    let mut flat = ThreeShell::new(1, seed.wrapping_add(3)); // flat inner (L=1), same seed
    let (mut f_init, mut f_last, mut l_last) = (0.0f32, 0.0f32, 0.0f32);
    for e in 0..600 {
        let f = full.step(&batch, &x, &tier3, &labels, 0.15);
        let l = flat.step(&batch, &x, &tier1, &labels, 0.15);
        if e == 0 {
            f_init = f;
        }
        f_last = f;
        l_last = l;
    }
    let f_acc = acc(&full.logits_of(&batch, &x, &tier3), &labels);
    let l_acc = acc(&flat.logits_of(&batch, &x, &tier1), &labels);
    (f_init, f_last, l_last, f_acc, l_acc)
}

#[test]
fn three_shell_cascade_learns_and_inner_ablation() {
    let mut f_init0 = 0.0f32;
    let (mut fs, mut ls, mut wins) = (Vec::new(), Vec::new(), 0);
    let (mut facc, mut lacc) = (Vec::new(), Vec::new());
    for seed in 0..5u64 {
        let (fi, fb, lb, fa, la) = run_seed(seed);
        if seed == 0 {
            f_init0 = fi;
        }
        eprintln!("  seed {seed}: L=3 BCE {fb:.4} (acc {fa:.3})   L=1 BCE {lb:.4} (acc {la:.3})");
        if fb < lb {
            wins += 1;
        }
        fs.push(fb);
        ls.push(lb);
        facc.push(fa);
        lacc.push(la);
    }
    let (fm, lm) = (median(fs.clone()), median(ls.clone()));
    eprintln!(
        "Gömb three-shell cascade (outer FIR → middle HSiKAN → inner CPML tier core), 5 seeds:"
    );
    eprintln!(
        "  L=3 (tiered): median BCE {fm:.4}  median acc {:.3}",
        median(facc)
    );
    eprintln!(
        "  L=1 (flat):   median BCE {lm:.4}  median acc {:.3}",
        median(lacc)
    );
    // "Helps" only if it wins a majority of seeds; a lower median from a minority of wins is a
    // tie, not a win (the single-seed trap this multi-seed pass exists to avoid).
    let verdict = if wins >= 3 {
        "helps"
    } else {
        "does NOT robustly help (median tie)"
    };
    eprintln!(
        "  verdict: inner tier-stratification {verdict} — L=3 lower BCE on {wins}/5 seeds (median ΔBCE {:.4})",
        lm - fm
    );

    // Gate: the full three-shell must learn end-to-end (seed 0); the tiered-vs-flat verdict is
    // the measurement, reported not asserted.
    assert!(
        fs[0] < 0.7 * f_init0,
        "three-shell cascade did not learn: {f_init0:.4}->{:.4}",
        fs[0]
    );
    assert!(fs.iter().chain(&ls).all(|v| v.is_finite()));
}
