//! Justification experiment for the Gömb CPML inner core: does **degree-tier stratification**
//! earn its weight on a **real, heavy-tailed-degree** signed graph (the regime the toy
//! 12-vertex 2c ablation lacked)?
//!
//! Pipeline (leakage-free, train edges only): per-vertex signed-degree features → enumerate
//! train triangles → CPML tier core (real degrees → tiers; per-tier restricted-triangle
//! aggregation `gather→linear→mean→scatter`; `concat(X₀, H₀…H_{L-1})` node embedding) → edge
//! head scores `sign(u,v)` from `[emb[u], emb[v]]`. Trained closed-form (Adam), test AUROC.
//!
//! Ablation in one run: **L=3 tiered** vs **L=1 flat** inner, same features/triangles/edges,
//! `tier_of` (hence routing) the only difference. Reports both AUROCs + the verdict.
//!
//! Run: `cargo run --release --example cpml_signed_link -- --data path.csv [--seed 0] [--max-tri 60000]`

use std::collections::HashMap;

use holonomy_learn::{
    adam_step, cycle_incidence_degrees, linear_backward, linear_forward, scatter_mean_backward,
    scatter_mean_forward, tier_cycle_indices, AdamState, LinearLayer, TierSpec,
};

const F: usize = 4; // per-vertex feature dim
const D: usize = 4; // per-tier aggregator output dim

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

    // Train adjacency (undirected) + signed-degree tallies.
    let mut adj: HashMap<u32, Vec<u32>> = HashMap::new();
    let mut pos = vec![0.0f32; n];
    let mut neg = vec![0.0f32; n];
    for &e in tr_i {
        let (u, v, r) = edges[e];
        adj.entry(u as u32).or_default().push(v as u32);
        adj.entry(v as u32).or_default().push(u as u32);
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
                    if tris.len() / 3 >= max_tri {
                        break 'outer;
                    }
                }
            }
        }
    }
    let n_tri = tris.len() / 3;

    // Real degrees → tiers (heavy-tailed on these graphs).
    let degrees = cycle_incidence_degrees(&tris, n);
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

    let a3 = run(
        3,
        &x0,
        &tris,
        &tier3,
        n,
        &tr_e,
        &tr_y,
        &te_e,
        &te_y,
        seed.wrapping_add(1),
    );
    let a1 = run(
        1,
        &x0,
        &tris,
        &tier1,
        n,
        &tr_e,
        &tr_y,
        &te_e,
        &te_y,
        seed.wrapping_add(1),
    );

    let name = path.rsplit(['/', '\\']).next().unwrap_or(&path);
    println!(
        "{name}  V={n} edges={} tri={n_tri} tiers(L=3)={sizes:?}",
        edges.len()
    );
    println!("  L=3 tiered inner: test AUROC {a3:.4}");
    println!("  L=1 flat  inner: test AUROC {a1:.4}");
    println!(
        "  verdict: tier-stratification {} on a real heavy-tailed graph (ΔAUROC {:+.4})",
        if a3 > a1 + 0.003 {
            "HELPS"
        } else if a3 < a1 - 0.003 {
            "HURTS"
        } else {
            "ties (does not earn its weight)"
        },
        a3 - a1
    );
}
