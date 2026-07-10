//! Justification experiment for the Gömb CPML inner core: does **degree-tier stratification**
//! earn its weight on a **real, heavy-tailed-degree** signed graph (the regime the toy
//! 12-vertex 2c ablation lacked)?
//!
//! Pipeline (leakage-free, train edges only): per-vertex signed-degree features → enumerate
//! train triangles → CPML tier core (real degrees → tiers; per-tier restricted-triangle
//! aggregation `gather→linear→mean→scatter`; `concat(X₀, H₀…H_{L-1})` node embedding) → edge
//! head scores `sign(u,v)` from `[emb[u], emb[v]]`. Trained closed-form (Adam), test AUROC.
//!
//! Three inner mechanisms in one run, same features/triangles/edges:
//!   1. **L=1 flat** — single sign-agnostic aggregator (baseline).
//!   2. **L=3 tiered** — fixed degree-tier routing (the CPML core).
//!   3. **signed hypergraph conv** — learned one-round signed HGNN embedding
//!      (`vertex_proj → node→edge (σ,D^{-1/2}) → edge_lin → edge→node → concat(x0,·)`),
//!      built from the `hg_message` kernels. The learned counterpart to the fixed routing.
//!
//! Reports all three test AUROCs + verdicts vs the flat baseline.
//!
//! Run: `cargo run --release --example cpml_signed_link -- --data path.csv [--seed 0] [--max-tri 60000]`

use std::collections::HashMap;

use holonomy_learn::{
    adam_step, cycle_incidence_degrees, gomb_outer_backward, gomb_outer_forward,
    hg_edge_to_node_backward, hg_edge_to_node_forward, hg_node_to_edge_backward,
    hg_node_to_edge_forward, hsikan_backward, hsikan_forward, linear_backward, linear_forward,
    rotor_holonomy_backward, rotor_holonomy_forward, scatter_mean_backward, scatter_mean_forward,
    tier_cycle_indices, AdamState, HsikanConfig, HsikanEdges, HsikanParams, LinearLayer, TierSpec,
};
use hymeko_graph::{CliffordFIR, TopKCyclesBatch};
use rand::{rngs::StdRng, Rng, SeedableRng};

const F: usize = 4; // per-vertex feature dim
const D: usize = 4; // per-tier aggregator output dim
const DH: usize = 8; // HGConv hidden dim

fn arg_str(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}
fn arg_f(name: &str, d: f32) -> f32 {
    arg_str(name).and_then(|s| s.parse().ok()).unwrap_or(d)
}

fn auroc(scores: &[f32], labels: &[u8]) -> f64 {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    let mut rank = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && scores[idx[j]] == scores[idx[i]] {
            j += 1;
        }
        let avg = (i + 1 + j) as f64 / 2.0;
        for &k in &idx[i..j] {
            rank[k] = avg;
        }
        i = j;
    }
    let npos = labels.iter().filter(|&&y| y == 1).count();
    let nneg = n - npos;
    if npos == 0 || nneg == 0 {
        return 0.5;
    }
    let rp: f64 = (0..n).filter(|&k| labels[k] == 1).map(|k| rank[k]).sum();
    (rp - (npos * (npos + 1)) as f64 / 2.0) / (npos * nneg) as f64
}

/// The CPML tier core + edge head over a fixed set of triangles, ablatable by tier count.
struct CpmlLinkModel {
    tier_lins: Vec<LinearLayer>,
    edge_head: LinearLayer,
    n_tiers: usize,
    emb_dim: usize,
}

struct TierCache {
    sub: Vec<u32>,
    corners: Vec<f32>,
    counts: Vec<u32>,
    m: usize,
}

impl CpmlLinkModel {
    fn new(n_tiers: usize, seed: u64) -> Self {
        let emb_dim = F + n_tiers * D;
        Self {
            tier_lins: (0..n_tiers)
                .map(|t| LinearLayer::new(F, D, seed + 11 + t as u64))
                .collect(),
            edge_head: LinearLayer::new(2 * emb_dim, 1, seed + 3),
            n_tiers,
            emb_dim,
        }
    }

    /// Node embeddings `(V, emb_dim)` = concat(x0, H₀…) + per-tier caches for the backward.
    fn node_embed(
        &self,
        x0: &[f32],
        tris: &[u32],
        tier_of: &[usize],
        n: usize,
    ) -> (Vec<f32>, Vec<TierCache>) {
        let mut emb = vec![0.0f32; n * self.emb_dim];
        for v in 0..n {
            emb[v * self.emb_dim..v * self.emb_dim + F].copy_from_slice(&x0[v * F..v * F + F]);
        }
        let mut caches = Vec::with_capacity(self.n_tiers);
        for (ell, lin) in self.tier_lins.iter().enumerate() {
            let idx = tier_cycle_indices(tris, 3, tier_of, ell);
            let mut sub = vec![0u32; idx.len() * 3];
            for (j, &c) in idx.iter().enumerate() {
                sub[j * 3..j * 3 + 3].copy_from_slice(&tris[c * 3..c * 3 + 3]);
            }
            let m = idx.len();
            let (h, cache) = if m == 0 {
                (
                    vec![0.0f32; n * D],
                    TierCache {
                        sub,
                        corners: Vec::new(),
                        counts: Vec::new(),
                        m,
                    },
                )
            } else {
                let corners = gather(x0, &sub, F);
                let lo = linear_forward(lin, &corners);
                let mut per_cycle = vec![0.0f32; m * D];
                for c in 0..m {
                    for j in 0..D {
                        let s: f32 = (0..3).map(|i| lo[(c * 3 + i) * D + j]).sum();
                        per_cycle[c * D + j] = s / 3.0;
                    }
                }
                let (h, counts) = scatter_mean_forward(&sub, 3, &per_cycle, D, n);
                (
                    h,
                    TierCache {
                        sub,
                        corners,
                        counts,
                        m,
                    },
                )
            };
            for v in 0..n {
                let base = v * self.emb_dim + F + ell * D;
                emb[base..base + D].copy_from_slice(&h[v * D..v * D + D]);
            }
            caches.push(cache);
        }
        (emb, caches)
    }

    fn edge_logits(&self, emb: &[f32], edges: &[(u32, u32)]) -> (Vec<f32>, Vec<f32>) {
        let mut ein = vec![0.0f32; edges.len() * 2 * self.emb_dim];
        for (e, &(u, v)) in edges.iter().enumerate() {
            let base = e * 2 * self.emb_dim;
            ein[base..base + self.emb_dim].copy_from_slice(
                &emb[u as usize * self.emb_dim..u as usize * self.emb_dim + self.emb_dim],
            );
            ein[base + self.emb_dim..base + 2 * self.emb_dim].copy_from_slice(
                &emb[v as usize * self.emb_dim..v as usize * self.emb_dim + self.emb_dim],
            );
        }
        (linear_forward(&self.edge_head, &ein), ein)
    }
}

fn gather(x: &[f32], idx: &[u32], w: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; idx.len() * w];
    for (j, &v) in idx.iter().enumerate() {
        out[j * w..j * w + w].copy_from_slice(&x[v as usize * w..v as usize * w + w]);
    }
    out
}

/// Train an L-tier model; return test AUROC.
#[allow(clippy::too_many_arguments)]
fn run(
    n_tiers: usize,
    x0: &[f32],
    tris: &[u32],
    tier_of: &[usize],
    n: usize,
    tr_edges: &[(u32, u32)],
    tr_y: &[f32],
    te_edges: &[(u32, u32)],
    te_y: &[u8],
    seed: u64,
) -> f64 {
    let mut m = CpmlLinkModel::new(n_tiers, seed);
    let mut tier_states: Vec<AdamState> = (0..n_tiers)
        .map(|t| AdamState::new(m.tier_lins[t].w.len() + m.tier_lins[t].b.len()))
        .collect();
    let mut head_state = AdamState::new(m.edge_head.w.len() + m.edge_head.b.len());
    let ntr = tr_y.len() as f32;
    for _ in 0..250 {
        let (emb, caches) = m.node_embed(x0, tris, tier_of, n);
        let (logits, ein) = m.edge_logits(&emb, tr_edges);
        // BCE grad on edge logits.
        let mut grad_logits = vec![0.0f32; tr_edges.len()];
        for i in 0..tr_edges.len() {
            let p = 1.0 / (1.0 + (-logits[i]).exp());
            grad_logits[i] = (p - tr_y[i]) / ntr;
        }
        let (grad_ein, grad_head) = linear_backward(&m.edge_head, &ein, &grad_logits);
        // Scatter edge-endpoint grads into grad_emb.
        let mut grad_emb = vec![0.0f32; n * m.emb_dim];
        for (e, &(u, v)) in tr_edges.iter().enumerate() {
            let base = e * 2 * m.emb_dim;
            for j in 0..m.emb_dim {
                grad_emb[u as usize * m.emb_dim + j] += grad_ein[base + j];
                grad_emb[v as usize * m.emb_dim + j] += grad_ein[base + m.emb_dim + j];
            }
        }
        // Per-tier inner backward → tier linear grads (x0 fixed, its grad ignored).
        let mut tier_grads: Vec<LinearLayer> = Vec::with_capacity(n_tiers);
        for (ell, lin) in m.tier_lins.iter().enumerate() {
            let cache = &caches[ell];
            if cache.m == 0 {
                tier_grads.push(lin.zero_grad());
                continue;
            }
            let mut grad_h = vec![0.0f32; n * D];
            for v in 0..n {
                let base = v * m.emb_dim + F + ell * D;
                grad_h[v * D..v * D + D].copy_from_slice(&grad_emb[base..base + D]);
            }
            let gpc = scatter_mean_backward(&cache.sub, 3, &grad_h, D, &cache.counts, n);
            let mut glo = vec![0.0f32; cache.m * 3 * D];
            for c in 0..cache.m {
                for i in 0..3 {
                    for j in 0..D {
                        glo[(c * 3 + i) * D + j] = gpc[c * D + j] / 3.0;
                    }
                }
            }
            let (_gc, glin) = linear_backward(lin, &cache.corners, &glo);
            tier_grads.push(glin);
        }
        // Adam updates.
        for (t, lin) in m.tier_lins.iter_mut().enumerate() {
            let mut flat: Vec<f32> = lin.w.iter().chain(&lin.b).copied().collect();
            let g: Vec<f32> = tier_grads[t]
                .w
                .iter()
                .chain(&tier_grads[t].b)
                .copied()
                .collect();
            adam_step(&mut flat, &g, &mut tier_states[t], 0.02);
            let (w, b) = flat.split_at(lin.w.len());
            lin.w.copy_from_slice(w);
            lin.b.copy_from_slice(b);
        }
        let mut hf: Vec<f32> = m
            .edge_head
            .w
            .iter()
            .chain(&m.edge_head.b)
            .copied()
            .collect();
        let hg: Vec<f32> = grad_head.w.iter().chain(&grad_head.b).copied().collect();
        adam_step(&mut hf, &hg, &mut head_state, 0.02);
        let (w, b) = hf.split_at(m.edge_head.w.len());
        m.edge_head.w.copy_from_slice(w);
        m.edge_head.b.copy_from_slice(b);
    }
    let (emb, _) = m.node_embed(x0, tris, tier_of, n);
    let (sc, _) = m.edge_logits(&emb, te_edges);
    auroc(&sc, te_y)
}

/// Adam-flatten helper for a `LinearLayer` param+grad → one step.
fn adam_layer(layer: &mut LinearLayer, grad: &LinearLayer, st: &mut AdamState, lr: f32) {
    let mut flat: Vec<f32> = layer.w.iter().chain(&layer.b).copied().collect();
    let g: Vec<f32> = grad.w.iter().chain(&grad.b).copied().collect();
    adam_step(&mut flat, &g, st, lr);
    let (w, b) = flat.split_at(layer.w.len());
    layer.w.copy_from_slice(w);
    layer.b.copy_from_slice(b);
}

/// Build edge-head input `[emb[u], emb[v]]` per edge (emb width `ed`).
fn edge_in(emb: &[f32], edges: &[(u32, u32)], ed: usize) -> Vec<f32> {
    let mut ein = vec![0.0f32; edges.len() * 2 * ed];
    for (e, &(u, v)) in edges.iter().enumerate() {
        let b = e * 2 * ed;
        ein[b..b + ed].copy_from_slice(&emb[u as usize * ed..u as usize * ed + ed]);
        ein[b + ed..b + 2 * ed].copy_from_slice(&emb[v as usize * ed..v as usize * ed + ed]);
    }
    ein
}

/// Learned signed **hypergraph convolution** node embedding, ablation arm vs the fixed tier
/// core: `x0 → vertex_proj → node→edge (σ, D^{-1/2}) → edge_lin → edge→node (σ, D^{-1/2}/D)
/// → concat(x0, out) → edge head`. Same edges/AUROC as `run`. Returns test AUROC.
#[allow(clippy::too_many_arguments)]
fn run_hgconv(
    x0: &[f32],
    tris: &[u32],
    tsig: &[f32],
    s_n2e: &[f32],
    s_e2n: &[f32],
    n: usize,
    tr_e: &[(u32, u32)],
    tr_y: &[f32],
    te_e: &[(u32, u32)],
    te_y: &[u8],
    seed: u64,
) -> f64 {
    let n_tri = tris.len() / 3;
    let ed = F + DH; // embedding width
    let mut vproj = LinearLayer::new(F, DH, seed + 1);
    let mut elin = LinearLayer::new(DH, DH, seed + 2);
    let mut head = LinearLayer::new(2 * ed, 1, seed + 3);
    let (mut sv, mut se, mut sh) = (
        AdamState::new(vproj.w.len() + vproj.b.len()),
        AdamState::new(elin.w.len() + elin.b.len()),
        AdamState::new(head.w.len() + head.b.len()),
    );

    // Forward → (node embedding, per-edge h_e cache for the backward).
    let embed = |vproj: &LinearLayer, elin: &LinearLayer| -> (Vec<f32>, Vec<f32>) {
        let x_p = linear_forward(vproj, x0);
        let h_e = hg_node_to_edge_forward(&x_p, tris, tsig, s_n2e, n_tri, 3, DH);
        let h_e2 = linear_forward(elin, &h_e);
        let out = hg_edge_to_node_forward(&h_e2, tris, tsig, s_e2n, n, 3, DH);
        let mut emb = vec![0.0f32; n * ed];
        for v in 0..n {
            emb[v * ed..v * ed + F].copy_from_slice(&x0[v * F..v * F + F]);
            emb[v * ed + F..v * ed + ed].copy_from_slice(&out[v * DH..v * DH + DH]);
        }
        (emb, h_e)
    };

    let ntr = tr_y.len() as f32;
    for _ in 0..250 {
        let (emb, h_e) = embed(&vproj, &elin);
        let ein = edge_in(&emb, tr_e, ed);
        let logits = linear_forward(&head, &ein);
        let mut gl = vec![0.0f32; tr_e.len()];
        for i in 0..tr_e.len() {
            let p = 1.0 / (1.0 + (-logits[i]).exp());
            gl[i] = (p - tr_y[i]) / ntr;
        }
        let (grad_ein, grad_head) = linear_backward(&head, &ein, &gl);
        // Scatter edge-endpoint grads → grad_emb, take the `out` slice.
        let mut grad_out = vec![0.0f32; n * DH];
        for (e, &(u, v)) in tr_e.iter().enumerate() {
            let b = e * 2 * ed;
            for j in 0..DH {
                grad_out[u as usize * DH + j] += grad_ein[b + F + j];
                grad_out[v as usize * DH + j] += grad_ein[b + ed + F + j];
            }
        }
        // Backprop the two HGConv kernels + the linears.
        let grad_h_e2 = hg_edge_to_node_backward(tris, tsig, s_e2n, &grad_out, n_tri, 3, DH);
        let (grad_h_e, grad_elin) = linear_backward(&elin, &h_e, &grad_h_e2);
        let grad_x_p = hg_node_to_edge_backward(tris, tsig, s_n2e, &grad_h_e, n, 3, DH);
        let (_gx0, grad_vproj) = linear_backward(&vproj, x0, &grad_x_p);
        adam_layer(&mut vproj, &grad_vproj, &mut sv, 0.02);
        adam_layer(&mut elin, &grad_elin, &mut se, 0.02);
        adam_layer(&mut head, &grad_head, &mut sh, 0.02);
    }
    let (emb, _) = embed(&vproj, &elin);
    let ein = edge_in(&emb, te_e, ed);
    auroc(&linear_forward(&head, &ein), te_y)
}

// ---- Gömb-Soma Step-1 gate: the FULL three-shell cascade as a signed-link predictor ----
// x0 → outer Clifford-FIR (MB banks) → scatter → HSiKAN → scatter → inner CPML tiers → node emb
// → edge head. Same data/edges/AUROC/Adam budget as the inner-core arms; the question is whether
// the outer+middle shells beat the inner core alone. Transcribes the tested `gomb_three_shell`
// forward + composed backward at runtime size.
const MB: usize = 2; // outer Clifford-FIR banks
const HC: usize = MB * F; // cascade hidden width = MB · D_FEAT (D_FEAT = F)
const SC: usize = 2; // hsikan spline segments
const GC: usize = 6; // hsikan grid
const CB: usize = 4; // hsikan Chebyshev order
const DL: usize = D; // inner per-tier output width

fn rand_vec(n: usize, scale: f32, rng: &mut StdRng) -> Vec<f32> {
    (0..n)
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * scale)
        .collect()
}

/// Per-tier inner aggregator over `x_core` (width `HC`): gather corners → linear → mean → scatter.
/// Returns the per-vertex tier features `(n·DL)` and the cache the backward needs.
fn cascade_tier_fwd(
    x_core: &[f32],
    tris: &[u32],
    tier_of: &[usize],
    ell: usize,
    lin: &LinearLayer,
    n: usize,
) -> (Vec<f32>, TierCache) {
    let idx = tier_cycle_indices(tris, 3, tier_of, ell);
    let mut sub = vec![0u32; idx.len() * 3];
    for (j, &c) in idx.iter().enumerate() {
        sub[j * 3..j * 3 + 3].copy_from_slice(&tris[c * 3..c * 3 + 3]);
    }
    let m = idx.len();
    if m == 0 {
        return (
            vec![0.0f32; n * DL],
            TierCache {
                sub,
                corners: Vec::new(),
                counts: Vec::new(),
                m,
            },
        );
    }
    let corners = gather(x_core, &sub, HC);
    let lo = linear_forward(lin, &corners);
    let mut per_cycle = vec![0.0f32; m * DL];
    for c in 0..m {
        for j in 0..DL {
            per_cycle[c * DL + j] = (0..3).map(|i| lo[(c * 3 + i) * DL + j]).sum::<f32>() / 3.0;
        }
    }
    let (h, counts) = scatter_mean_forward(&sub, 3, &per_cycle, DL, n);
    (
        h,
        TierCache {
            sub,
            corners,
            counts,
            m,
        },
    )
}

/// Backward of `cascade_tier_fwd` → (grad w.r.t. x_core `(n·HC)`, grad w.r.t. the tier linear).
fn cascade_tier_bwd(
    grad_h: &[f32],
    cache: &TierCache,
    lin: &LinearLayer,
    n: usize,
) -> (Vec<f32>, LinearLayer) {
    let mut grad_x_core = vec![0.0f32; n * HC];
    if cache.m == 0 {
        return (grad_x_core, lin.zero_grad());
    }
    let gpc = scatter_mean_backward(&cache.sub, 3, grad_h, DL, &cache.counts, n);
    let mut glo = vec![0.0f32; cache.m * 3 * DL];
    for c in 0..cache.m {
        for i in 0..3 {
            for j in 0..DL {
                glo[(c * 3 + i) * DL + j] = gpc[c * DL + j] / 3.0;
            }
        }
    }
    let (grad_corners, glin) = linear_backward(lin, &cache.corners, &glo);
    for (corner, &v) in cache.sub.iter().enumerate() {
        for j in 0..HC {
            grad_x_core[v as usize * HC + j] += grad_corners[corner * HC + j];
        }
    }
    (grad_x_core, glin)
}

#[allow(clippy::too_many_arguments)]
fn run_cascade(
    x0: &[f32],
    tris: &[u32],
    tri_signs: &[f32],
    tier_of: &[usize],
    n_tiers: usize,
    n: usize,
    tr_e: &[(u32, u32)],
    tr_y: &[f32],
    te_e: &[(u32, u32)],
    te_y: &[u8],
    seed: u64,
) -> f64 {
    let n_tri = tris.len() / 3;
    let emb_dim = HC + n_tiers * DL;
    let batch = TopKCyclesBatch {
        cycles: tris.to_vec(),
        signs: tri_signs.iter().map(|&s| s.signum() as i8).collect(),
        scores: vec![0.0f64; n_tri],
        k: 3,
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut banks: Vec<CliffordFIR> = (0..MB)
        .map(|_| CliffordFIR::new(rand_vec(3, 0.4, &mut rng), rand_vec(3, 0.4, &mut rng)))
        .collect();
    let mut inner = rand_vec(SC * HC * CB, 0.3, &mut rng);
    let mut outer = rand_vec(SC * HC * CB, 0.3, &mut rng);
    let mut gw = rand_vec(HC * HC, 0.2, &mut rng);
    let mut gb = vec![-1.0f32; HC];
    let mut tier_lins: Vec<LinearLayer> = (0..n_tiers)
        .map(|t| LinearLayer::new(HC, DL, seed + 10 + t as u64))
        .collect();
    let mut head = LinearLayer::new(2 * emb_dim, 1, seed + 3);
    let cfg = HsikanConfig::new(n_tri, 3, HC, SC, GC, CB, true);
    let edges = HsikanEdges {
        vertices: &batch.cycles,
        signs: &batch.signs,
    };

    // Adam states (one per param group).
    let mut s_bank = AdamState::new(MB * 6);
    let mut s_inner = AdamState::new(inner.len());
    let mut s_outer = AdamState::new(outer.len());
    let mut s_gw = AdamState::new(gw.len());
    let mut s_gb = AdamState::new(gb.len());
    let mut s_tiers: Vec<AdamState> = tier_lins
        .iter()
        .map(|l| AdamState::new(l.w.len() + l.b.len()))
        .collect();
    let mut s_head = AdamState::new(head.w.len() + head.b.len());
    let ntr = tr_y.len() as f32;

    for it in 0..250 {
        // Forward.
        let y_out = gomb_outer_forward(&batch, x0, &banks, n, F);
        let (x_out, cnt_out) = scatter_mean_forward(&batch.cycles, 3, &y_out, HC, n);
        let params = HsikanParams {
            inner_coef: &inner,
            outer_coef: &outer,
            gate_w: &gw,
            gate_b: &gb,
        };
        let (h_mid, hcache) = hsikan_forward(params, &x_out, edges, cfg);
        let (x_core, cnt_mid) = scatter_mean_forward(&batch.cycles, 3, &h_mid, HC, n);
        let mut emb = vec![0.0f32; n * emb_dim];
        for v in 0..n {
            emb[v * emb_dim..v * emb_dim + HC].copy_from_slice(&x_core[v * HC..v * HC + HC]);
        }
        let mut tier_caches = Vec::with_capacity(n_tiers);
        for (ell, lin) in tier_lins.iter().enumerate() {
            let (h, cache) = cascade_tier_fwd(&x_core, &batch.cycles, tier_of, ell, lin, n);
            for v in 0..n {
                let base = v * emb_dim + HC + ell * DL;
                emb[base..base + DL].copy_from_slice(&h[v * DL..v * DL + DL]);
            }
            tier_caches.push(cache);
        }
        let ein = edge_in(&emb, tr_e, emb_dim);
        let logits = linear_forward(&head, &ein);

        // BCE grad.
        let mut gl = vec![0.0f32; tr_e.len()];
        let mut loss = 0.0f32;
        for i in 0..tr_e.len() {
            let p = 1.0 / (1.0 + (-logits[i]).exp());
            gl[i] = (p - tr_y[i]) / ntr;
            let pc = p.clamp(1e-7, 1.0 - 1e-7);
            loss -= (tr_y[i] * pc.ln() + (1.0 - tr_y[i]) * (1.0 - pc).ln()) / ntr;
        }
        if it % 50 == 0 || it == 249 {
            println!("    cascade L={n_tiers} it {it:3}/250  BCE {loss:.4}");
        }
        let (grad_ein, grad_head) = linear_backward(&head, &ein, &gl);
        let mut grad_emb = vec![0.0f32; n * emb_dim];
        for (e, &(u, v)) in tr_e.iter().enumerate() {
            let b = e * 2 * emb_dim;
            for j in 0..emb_dim {
                grad_emb[u as usize * emb_dim + j] += grad_ein[b + j];
                grad_emb[v as usize * emb_dim + j] += grad_ein[b + emb_dim + j];
            }
        }
        // Split → grad on x_core + per-tier.
        let mut grad_x_core = vec![0.0f32; n * HC];
        for v in 0..n {
            grad_x_core[v * HC..v * HC + HC]
                .copy_from_slice(&grad_emb[v * emb_dim..v * emb_dim + HC]);
        }
        let mut grad_tier_lins = Vec::with_capacity(n_tiers);
        for (ell, lin) in tier_lins.iter().enumerate() {
            let mut grad_h = vec![0.0f32; n * DL];
            for v in 0..n {
                let base = v * emb_dim + HC + ell * DL;
                grad_h[v * DL..v * DL + DL].copy_from_slice(&grad_emb[base..base + DL]);
            }
            let (gx, glin) = cascade_tier_bwd(&grad_h, &tier_caches[ell], lin, n);
            for (a, bb) in grad_x_core.iter_mut().zip(&gx) {
                *a += bb;
            }
            grad_tier_lins.push(glin);
        }
        // Back through the two scatters + hsikan + outer.
        let grad_h_mid = scatter_mean_backward(&batch.cycles, 3, &grad_x_core, HC, &cnt_mid, n);
        let params = HsikanParams {
            inner_coef: &inner,
            outer_coef: &outer,
            gate_w: &gw,
            gate_b: &gb,
        };
        let hb = hsikan_backward(params, edges, &hcache, &grad_h_mid, cfg);
        let grad_y_out = scatter_mean_backward(&batch.cycles, 3, &hb.grad_x, HC, &cnt_out, n);
        let (_gf, grad_banks) = gomb_outer_backward(&batch, x0, &banks, &grad_y_out, n, F);

        // Adam updates.
        let mut bank_flat: Vec<f32> = banks
            .iter()
            .flat_map(|b| b.a.iter().chain(&b.b).copied())
            .collect();
        let bank_grad: Vec<f32> = grad_banks
            .iter()
            .flat_map(|g| g.a.iter().chain(&g.b).copied())
            .collect();
        adam_step(&mut bank_flat, &bank_grad, &mut s_bank, 0.02);
        for (bk, chunk) in banks.iter_mut().zip(bank_flat.chunks(6)) {
            bk.a.copy_from_slice(&chunk[..3]);
            bk.b.copy_from_slice(&chunk[3..6]);
        }
        adam_step(&mut inner, &hb.grad_inner_coef, &mut s_inner, 0.02);
        adam_step(&mut outer, &hb.grad_outer_coef, &mut s_outer, 0.02);
        adam_step(&mut gw, &hb.grad_gate_w, &mut s_gw, 0.02);
        adam_step(&mut gb, &hb.grad_gate_b, &mut s_gb, 0.02);
        for (t, lin) in tier_lins.iter_mut().enumerate() {
            adam_layer(lin, &grad_tier_lins[t], &mut s_tiers[t], 0.02);
        }
        adam_layer(&mut head, &grad_head, &mut s_head, 0.02);
    }

    // Eval.
    let y_out = gomb_outer_forward(&batch, x0, &banks, n, F);
    let (x_out, _) = scatter_mean_forward(&batch.cycles, 3, &y_out, HC, n);
    let params = HsikanParams {
        inner_coef: &inner,
        outer_coef: &outer,
        gate_w: &gw,
        gate_b: &gb,
    };
    let (h_mid, _) = hsikan_forward(params, &x_out, edges, cfg);
    let (x_core, _) = scatter_mean_forward(&batch.cycles, 3, &h_mid, HC, n);
    let mut emb = vec![0.0f32; n * emb_dim];
    for v in 0..n {
        emb[v * emb_dim..v * emb_dim + HC].copy_from_slice(&x_core[v * HC..v * HC + HC]);
    }
    for (ell, lin) in tier_lins.iter().enumerate() {
        let (h, _) = cascade_tier_fwd(&x_core, &batch.cycles, tier_of, ell, lin, n);
        for v in 0..n {
            let base = v * emb_dim + HC + ell * DL;
            emb[base..base + DL].copy_from_slice(&h[v * DL..v * DL + DL]);
        }
    }
    let ein = edge_in(&emb, te_e, emb_dim);
    auroc(&linear_forward(&head, &ein), te_y)
}

/// Unit-normalize each 4-vector row (a rotor *is* a unit quaternion) → `(normalized, norms)`.
fn unit_rows(q: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; n * 4];
    let mut norms = vec![0.0f32; n];
    for i in 0..n {
        let s = q[i * 4..i * 4 + 4]
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt()
            + 1e-6;
        norms[i] = s;
        for j in 0..4 {
            out[i * 4 + j] = q[i * 4 + j] / s;
        }
    }
    (out, norms)
}

/// Backward of [`unit_rows`]: `∂/∂raw = (grad − (grad·q̂) q̂) / ‖raw‖` with `q̂` the normalized row.
fn unit_rows_backward(qn: &[f32], norms: &[f32], grad_qn: &[f32], n: usize) -> Vec<f32> {
    let mut g = vec![0.0f32; n * 4];
    for i in 0..n {
        let dot: f32 = (0..4).map(|j| grad_qn[i * 4 + j] * qn[i * 4 + j]).sum();
        for j in 0..4 {
            g[i * 4 + j] = (grad_qn[i * 4 + j] - dot * qn[i * 4 + j]) / norms[i];
        }
    }
    g
}

// ---- Holonomy channel: the inner CPML core + a rotor-holonomy feature per cycle ----
// Per triangle edge (a,b,sign): learned linear(2F+1 → 4) → per-edge quaternion; rotor_holonomy over
// the 3 edge-quats → per-cycle holonomy; scatter-mean to vertices → 4-dim per-vertex feature concat
// into the inner-core embedding. A/B vs the inner core alone (run(3)): does the order-sensitive rotor
// holonomy carry signal the tier-degree features miss? Reuses linear + rotor_holonomy + scatter_mean.
#[allow(clippy::too_many_arguments)]
fn run_holonomy(
    x0: &[f32],
    tris: &[u32],
    tri_signs: &[f32],
    tier_of: &[usize],
    n: usize,
    tr_e: &[(u32, u32)],
    tr_y: &[f32],
    te_e: &[(u32, u32)],
    te_y: &[u8],
    seed: u64,
) -> f64 {
    let n_tri = tris.len() / 3;
    let n_tiers = 3;
    let mut m = CpmlLinkModel::new(n_tiers, seed); // tiers reused; its edge_head is unused here
    let base_dim = m.emb_dim; // F + n_tiers·D
    let full_dim = base_dim + 4; // + holonomy(4)
    let mut holo_lin = LinearLayer::new(2 * F + 1, 4, seed + 21);
    let mut head = LinearLayer::new(2 * full_dim, 1, seed + 22);
    let mut tier_states: Vec<AdamState> = m
        .tier_lins
        .iter()
        .map(|l| AdamState::new(l.w.len() + l.b.len()))
        .collect();
    let mut s_holo = AdamState::new(holo_lin.w.len() + holo_lin.b.len());
    let mut s_head = AdamState::new(head.w.len() + head.b.len());

    // Fixed per-edge feature matrix [x0[a], x0[b], sign] (x0 frozen, signs fixed) — build once.
    let ew = 2 * F + 1;
    let mut edge_feat = vec![0.0f32; n_tri * 3 * ew];
    for c in 0..n_tri {
        let t = [tris[3 * c], tris[3 * c + 1], tris[3 * c + 2]];
        for j in 0..3 {
            let (a, b) = (t[j] as usize, t[(j + 1) % 3] as usize);
            let row = (c * 3 + j) * ew;
            edge_feat[row..row + F].copy_from_slice(&x0[a * F..a * F + F]);
            edge_feat[row + F..row + 2 * F].copy_from_slice(&x0[b * F..b * F + F]);
            edge_feat[row + 2 * F] = tri_signs[3 * c + j];
        }
    }

    let combine = |emb_tier: &[f32], holo_vert: &[f32]| -> Vec<f32> {
        let mut emb = vec![0.0f32; n * full_dim];
        for v in 0..n {
            emb[v * full_dim..v * full_dim + base_dim]
                .copy_from_slice(&emb_tier[v * base_dim..v * base_dim + base_dim]);
            emb[v * full_dim + base_dim..v * full_dim + full_dim]
                .copy_from_slice(&holo_vert[v * 4..v * 4 + 4]);
        }
        emb
    };

    let ntr = tr_y.len() as f32;
    for it in 0..250 {
        let (emb_tier, caches) = m.node_embed(x0, tris, tier_of, n);
        let q_raw = linear_forward(&holo_lin, &edge_feat);
        let (q_edge, q_norms) = unit_rows(&q_raw, n_tri * 3); // rotors are unit quaternions
        let (holo, prefixes) = rotor_holonomy_forward(&q_edge, n_tri, 3);
        let (holo_vert, holo_counts) = scatter_mean_forward(tris, 3, &holo, 4, n);
        let emb = combine(&emb_tier, &holo_vert);
        let ein = edge_in(&emb, tr_e, full_dim);
        let logits = linear_forward(&head, &ein);

        let mut gl = vec![0.0f32; tr_e.len()];
        let mut loss = 0.0f32;
        for i in 0..tr_e.len() {
            let p = 1.0 / (1.0 + (-logits[i]).exp());
            gl[i] = (p - tr_y[i]) / ntr;
            let pc = p.clamp(1e-7, 1.0 - 1e-7);
            loss -= (tr_y[i] * pc.ln() + (1.0 - tr_y[i]) * (1.0 - pc).ln()) / ntr;
        }
        if it % 50 == 0 || it == 249 {
            println!("    holonomy it {it:3}/250  BCE {loss:.4}");
        }
        let (grad_ein, grad_head) = linear_backward(&head, &ein, &gl);
        let mut grad_emb = vec![0.0f32; n * full_dim];
        for (e, &(u, v)) in tr_e.iter().enumerate() {
            let b = e * 2 * full_dim;
            for j in 0..full_dim {
                grad_emb[u as usize * full_dim + j] += grad_ein[b + j];
                grad_emb[v as usize * full_dim + j] += grad_ein[b + full_dim + j];
            }
        }
        // Split → tier part + holonomy part.
        let mut grad_emb_tier = vec![0.0f32; n * base_dim];
        let mut grad_holo_vert = vec![0.0f32; n * 4];
        for v in 0..n {
            grad_emb_tier[v * base_dim..v * base_dim + base_dim]
                .copy_from_slice(&grad_emb[v * full_dim..v * full_dim + base_dim]);
            grad_holo_vert[v * 4..v * 4 + 4]
                .copy_from_slice(&grad_emb[v * full_dim + base_dim..v * full_dim + full_dim]);
        }
        // Tier backward (mirror `run`).
        let mut tier_grads: Vec<LinearLayer> = Vec::with_capacity(n_tiers);
        for (ell, lin) in m.tier_lins.iter().enumerate() {
            let cache = &caches[ell];
            if cache.m == 0 {
                tier_grads.push(lin.zero_grad());
                continue;
            }
            let mut grad_h = vec![0.0f32; n * D];
            for v in 0..n {
                let base = v * base_dim + F + ell * D;
                grad_h[v * D..v * D + D].copy_from_slice(&grad_emb_tier[base..base + D]);
            }
            let gpc = scatter_mean_backward(&cache.sub, 3, &grad_h, D, &cache.counts, n);
            let mut glo = vec![0.0f32; cache.m * 3 * D];
            for c in 0..cache.m {
                for i in 0..3 {
                    for j in 0..D {
                        glo[(c * 3 + i) * D + j] = gpc[c * D + j] / 3.0;
                    }
                }
            }
            let (_gc, glin) = linear_backward(lin, &cache.corners, &glo);
            tier_grads.push(glin);
        }
        // Holonomy backward: scatter → rotor_holonomy → linear.
        let grad_holo = scatter_mean_backward(tris, 3, &grad_holo_vert, 4, &holo_counts, n);
        let grad_q_edge = rotor_holonomy_backward(&q_edge, &prefixes, &grad_holo, n_tri, 3);
        let grad_q_raw = unit_rows_backward(&q_edge, &q_norms, &grad_q_edge, n_tri * 3);
        let (_ge, grad_holo_lin) = linear_backward(&holo_lin, &edge_feat, &grad_q_raw);

        for (t, lin) in m.tier_lins.iter_mut().enumerate() {
            adam_layer(lin, &tier_grads[t], &mut tier_states[t], 0.02);
        }
        adam_layer(&mut holo_lin, &grad_holo_lin, &mut s_holo, 0.02);
        adam_layer(&mut head, &grad_head, &mut s_head, 0.02);
    }

    let (emb_tier, _) = m.node_embed(x0, tris, tier_of, n);
    let q_raw = linear_forward(&holo_lin, &edge_feat);
    let (q_edge, _) = unit_rows(&q_raw, n_tri * 3);
    let (holo, _) = rotor_holonomy_forward(&q_edge, n_tri, 3);
    let (holo_vert, _) = scatter_mean_forward(tris, 3, &holo, 4, n);
    let emb = combine(&emb_tier, &holo_vert);
    let ein = edge_in(&emb, te_e, full_dim);
    auroc(&linear_forward(&head, &ein), te_y)
}

fn main() {
    let path = arg_str("--data").expect("--data <edgelist.csv>");
    let seed = arg_f("--seed", 0.0) as u64;
    let max_tri = arg_f("--max-tri", 60000.0) as usize;

    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    let mut relabel: HashMap<usize, u32> = HashMap::new();
    for line in std::fs::read_to_string(&path).expect("read data").lines() {
        if line.starts_with('#') {
            continue;
        }
        let p: Vec<&str> = line
            .split([',', ' ', '\t'])
            .filter(|s| !s.is_empty())
            .collect();
        if p.len() < 3 {
            continue;
        }
        let (u, v, r): (usize, usize, f32) = (
            p[0].parse().unwrap(),
            p[1].parse().unwrap(),
            p[2].parse().unwrap(),
        );
        let nx = relabel.len() as u32;
        let ru = *relabel.entry(u).or_insert(nx);
        let nx = relabel.len() as u32;
        let rv = *relabel.entry(v).or_insert(nx);
        edges.push((ru as usize, rv as usize, r));
    }
    let n = relabel.len();

    // Deterministic shuffle + 80/20 split.
    let mut order: Vec<usize> = (0..edges.len()).collect();
    let mut st = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    for i in (1..order.len()).rev() {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        order.swap(i, (st >> 33) as usize % (i + 1));
    }
    let cut = order.len() * 4 / 5;
    let (tr_i, te_i) = order.split_at(cut);

    // Train adjacency (undirected) + signed-degree tallies + per-edge signs.
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut esign: HashMap<(u32, u32), f32> = HashMap::new();
    let mut pos = vec![0.0f32; n];
    let mut neg = vec![0.0f32; n];
    let ekey = |a: u32, b: u32| if a < b { (a, b) } else { (b, a) };
    for &e in tr_i {
        let (u, v, r) = edges[e];
        adj.entry(u as u32).or_default().push(v as u32);
        adj.entry(v as u32).or_default().push(u as u32);
        esign.insert(ekey(u as u32, v as u32), r.signum());
        let s = if r > 0.0 { &mut pos } else { &mut neg };
        s[u] += 1.0;
        s[v] += 1.0;
    }
    // Per-vertex features x0 (leakage-free: train graph only), then standardise.
    let mut x0 = vec![0.0f32; n * F];
    for v in 0..n {
        let (p, ng) = (pos[v], neg[v]);
        x0[v * F] = (p + 1.0).ln();
        x0[v * F + 1] = (ng + 1.0).ln();
        x0[v * F + 2] = (p + ng + 1.0).ln();
        x0[v * F + 3] = (p + 1.0) / (p + ng + 2.0);
    }
    for j in 0..F {
        let mu: f32 = (0..n).map(|v| x0[v * F + j]).sum::<f32>() / n as f32;
        let sd: f32 =
            ((0..n).map(|v| (x0[v * F + j] - mu).powi(2)).sum::<f32>() / n as f32).sqrt() + 1e-6;
        for v in 0..n {
            x0[v * F + j] = (x0[v * F + j] - mu) / sd;
        }
    }

    // Enumerate train triangles (u<v<w, all three edges present), capped.
    let mut nbr: Vec<std::collections::HashSet<u32>> = vec![Default::default(); n];
    for (&u, vs) in &adj {
        for &v in vs {
            nbr[u as usize].insert(v);
        }
    }
    let mut tris: Vec<u32> = Vec::new();
    let mut tri_signs: Vec<f32> = Vec::new(); // per-corner: sign of the outgoing boundary edge
    'outer: for u in 0..n as u32 {
        for &v in &nbr[u as usize] {
            if v <= u {
                continue;
            }
            for &w in &nbr[u as usize] {
                if w <= v {
                    continue;
                }
                if nbr[v as usize].contains(&w) {
                    tris.extend_from_slice(&[u, v, w]);
                    // corner u → edge (u,v), corner v → edge (v,w), corner w → edge (w,u).
                    tri_signs.push(esign[&ekey(u, v)]);
                    tri_signs.push(esign[&ekey(v, w)]);
                    tri_signs.push(esign[&ekey(u, w)]);
                    if tris.len() / 3 >= max_tri {
                        break 'outer;
                    }
                }
            }
        }
    }
    let n_tri = tris.len() / 3;

    // Real degrees → tiers (heavy-tailed on these graphs) + D_v^{-1/2} scales for HGConv.
    let degrees = cycle_incidence_degrees(&tris, n);
    let dv: Vec<f32> = degrees.iter().map(|&d| d.max(1.0)).collect();
    let dv_inv_sqrt: Vec<f32> = dv.iter().map(|&d| d.powf(-0.5)).collect();
    let scale_e2n: Vec<f32> = dv.iter().zip(&dv_inv_sqrt).map(|(&d, &s)| s / d).collect();
    let tier3 = TierSpec::uniform(3).assign(&degrees);
    let tier1 = TierSpec::uniform(1).assign(&degrees);
    let sizes: Vec<usize> = (0..3)
        .map(|t| tier3.iter().filter(|&&x| x == t).count())
        .collect();

    let mk = |idx: &[usize]| -> (Vec<(u32, u32)>, Vec<f32>, Vec<u8>) {
        let mut e = Vec::with_capacity(idx.len());
        let mut yf = Vec::with_capacity(idx.len());
        let mut yu = Vec::with_capacity(idx.len());
        for &i in idx {
            let (u, v, r) = edges[i];
            e.push((u as u32, v as u32));
            yf.push((r > 0.0) as u32 as f32);
            yu.push((r > 0.0) as u8);
        }
        (e, yf, yu)
    };
    let (tr_e, tr_y, _) = mk(tr_i);
    let (te_e, _, te_y) = mk(te_i);
    let sd = seed.wrapping_add(1);

    let a1 = run(1, &x0, &tris, &tier1, n, &tr_e, &tr_y, &te_e, &te_y, sd);
    let a3 = run(3, &x0, &tris, &tier3, n, &tr_e, &tr_y, &te_e, &te_y, sd);
    let ah = run_hgconv(
        &x0,
        &tris,
        &tri_signs,
        &dv_inv_sqrt,
        &scale_e2n,
        n,
        &tr_e,
        &tr_y,
        &te_e,
        &te_y,
        sd,
    );
    let t_cascade = std::time::Instant::now();
    let ac = run_cascade(
        &x0, &tris, &tri_signs, &tier3, 3, n, &tr_e, &tr_y, &te_e, &te_y, sd,
    );
    let cascade_secs = t_cascade.elapsed().as_secs_f64();
    let aholo = run_holonomy(
        &x0, &tris, &tri_signs, &tier3, n, &tr_e, &tr_y, &te_e, &te_y, sd,
    );

    let name = path.rsplit(['/', '\\']).next().unwrap_or(&path);
    println!(
        "{name}  V={n} edges={} tri={n_tri} tiers(L=3)={sizes:?}",
        edges.len()
    );
    println!("  L=1 flat  inner (fixed):      test AUROC {a1:.4}");
    println!("  L=3 tiered inner (fixed):     test AUROC {a3:.4}");
    println!("  signed hypergraph conv (learned): test AUROC {ah:.4}");
    println!(
        "  FULL cascade L=3 (outer FIR→HSiKAN→inner): test AUROC {ac:.4}  ({cascade_secs:.1}s)"
    );
    println!("  inner core L=3 + rotor-holonomy channel:   test AUROC {aholo:.4}");
    let verdict = |name: &str, val: f64, base: f64| {
        let tag = if val > base + 0.003 {
            "HELPS"
        } else if val < base - 0.003 {
            "HURTS"
        } else {
            "ties"
        };
        println!(
            "  {name} vs flat baseline: {tag} (ΔAUROC {:+.4})",
            val - base
        );
    };
    verdict("tier-stratification", a3, a1);
    verdict("hypergraph conv    ", ah, a1);
    verdict("FULL cascade vs inner core", ac, a3);
    verdict("holonomy channel vs inner core", aholo, a3);
}
