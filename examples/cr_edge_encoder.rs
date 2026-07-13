//! Standalone A/B — does a LEARNABLE Chebyshev-CR encoding of the real edge
//! weight beat a fixed binary/tanh encoding for signed-link sign prediction?
//!
//! A minimal, leakage-free (train-only features), end-to-end differentiable
//! edge-sign predictor:
//!   encode:  w_e = enc(r_e)                     enc ∈ {binary, tanh, cr}
//!   node:    net[v]=Σ_inc w_e, absum[v]=Σ_inc|w_e|
//!            avg[v]=net/(absum+ε) ∈[-1,1]  (reputation), d[v]=tanh(absum/scale)
//!   edge:    logit(u,v) = linear([avg_u,d_u,avg_v,d_v])
//!   loss:    BCE(logit, sign)
//! For `cr`, the Chebyshev-CR coefficients train end-to-end via the composed
//! backward (bce → linear → node aggregation → chebyshev_cr_backward → coef).
//! Binary is the |w|=1 special case, so `cr` can recover it or exploit magnitude.
//!
//! Run: `cargo run --release --example cr_edge_encoder -- --data <edgelist> [--seed 0] [--iters 400]`

use holonomy_learn::{
    adam_step, bce_with_logits_backward, chebyshev_cr_backward, chebyshev_cr_forward,
    linear_backward, linear_forward, AdamState, LinearLayer,
};
use std::collections::HashMap;

fn arg_str(name: &str) -> Option<String> {
    std::env::args().skip_while(|a| a != name).nth(1)
}
fn arg_f(name: &str, d: f32) -> f32 {
    arg_str(name).and_then(|s| s.parse().ok()).unwrap_or(d)
}

/// Mann–Whitney AUROC (ties ignored; scores continuous).
fn auroc(scores: &[f32], labels: &[u8]) -> f64 {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut rank_sum, mut n_pos) = (0.0f64, 0u64);
    for (rank, &i) in idx.iter().enumerate() {
        if labels[i] == 1 {
            rank_sum += (rank + 1) as f64;
            n_pos += 1;
        }
    }
    let n_neg = scores.len() as u64 - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    (rank_sum - (n_pos * (n_pos + 1) / 2) as f64) / (n_pos * n_neg) as f64
}

const EPS: f32 = 1e-3;
const K: usize = 6; // Chebyshev coefficients
const GRID: usize = 8; // CR control points

#[derive(Clone, Copy, PartialEq)]
enum Enc {
    Binary,
    Tanh,
    Cr,
}

/// Encode all edge weights under the current mode / coef.
/// Returns (w, optional CR (cache, control, basis) for the backward).
#[allow(clippy::type_complexity)]
fn encode(
    enc: Enc,
    rnorm: &[f32],
    raw: &[f32],
    mean_abs: f32,
    coef: &[f32],
) -> (
    Vec<f32>,
    Option<(holonomy_learn::CatmullRomCache, Vec<f32>, Vec<f32>)>,
) {
    match enc {
        Enc::Binary => (raw.iter().map(|&r| r.signum()).collect(), None),
        Enc::Tanh => (raw.iter().map(|&r| (r / mean_abs).tanh()).collect(), None),
        Enc::Cr => {
            let (y, cache, control, basis) =
                chebyshev_cr_forward(coef, rnorm, rnorm.len(), 1, GRID, K);
            (y, Some((cache, control, basis)))
        }
    }
}

/// net[v], absum[v] from train edges under weights `w`.
fn aggregate(tr: &[(usize, usize)], w: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    let (mut net, mut absum) = (vec![0.0f32; n], vec![0.0f32; n]);
    for (e, &(u, v)) in tr.iter().enumerate() {
        net[u] += w[e];
        net[v] += w[e];
        absum[u] += w[e].abs();
        absum[v] += w[e].abs();
    }
    (net, absum)
}

/// Per-node features [avg, d] flat (n,2) from net/absum.
fn node_feats(net: &[f32], absum: &[f32], scale: f32, n: usize) -> Vec<f32> {
    let mut x = vec![0.0f32; n * 2];
    for v in 0..n {
        x[v * 2] = net[v] / (absum[v] + EPS);
        x[v * 2 + 1] = (absum[v] / scale).tanh();
    }
    x
}

/// Edge input rows [avg_u,d_u,avg_v,d_v] flat (m,4).
fn edge_in(feats: &[f32], edges: &[(usize, usize)]) -> Vec<f32> {
    let mut ein = vec![0.0f32; edges.len() * 4];
    for (e, &(u, v)) in edges.iter().enumerate() {
        ein[e * 4] = feats[u * 2];
        ein[e * 4 + 1] = feats[u * 2 + 1];
        ein[e * 4 + 2] = feats[v * 2];
        ein[e * 4 + 3] = feats[v * 2 + 1];
    }
    ein
}

/// Shared split + normalisation context (bundled to keep the fn signatures lean).
struct Ctx<'a> {
    tr: &'a [(usize, usize)],
    tr_y: &'a [f32],
    rnorm: &'a [f32],
    raw: &'a [f32],
    mean_abs: f32,
    scale: f32,
    n: usize,
    te: &'a [(usize, usize)],
    te_y: &'a [u8],
}

/// Test AUROC of a trained (head, coef, enc) on held-out edges — features built
/// from TRAIN edges only (leakage-free).
fn eval(enc: Enc, head: &LinearLayer, coef: &[f32], c: &Ctx) -> f64 {
    let (w, _) = encode(enc, c.rnorm, c.raw, c.mean_abs, coef);
    let (net, absum) = aggregate(c.tr, &w, c.n);
    let feats = node_feats(&net, &absum, c.scale, c.n);
    let ein = edge_in(&feats, c.te);
    let logit = linear_forward(head, &ein);
    auroc(&logit, c.te_y)
}

fn train(enc: Enc, c: &Ctx, iters: usize, seed: u64) -> (f64, f64, Vec<f32>) {
    let (n, m) = (c.n, c.tr.len());
    let mut head = LinearLayer::new(4, 1, seed);
    // CR coef init ≈ identity (T_1 term = 1): the spline starts as w = rnorm.
    let mut coef = vec![0.0f32; K];
    if enc == Enc::Cr {
        coef[1] = 1.0;
    }
    let (mut st_w, mut st_b) = (AdamState::new(head.w.len()), AdamState::new(head.b.len()));
    let mut st_c = AdamState::new(K);
    let init = eval(enc, &head, &coef, c);
    for it in 0..iters {
        let (w, cr) = encode(enc, c.rnorm, c.raw, c.mean_abs, &coef);
        let (net, absum) = aggregate(c.tr, &w, n);
        let feats = node_feats(&net, &absum, c.scale, n);
        let ein = edge_in(&feats, c.tr);
        let logit = linear_forward(&head, &ein);
        let gl = bce_with_logits_backward(&logit, c.tr_y);
        let (grad_ein, ghead) = linear_backward(&head, &ein, &gl);

        // grad_ein rows [g_avg_u,g_d_u,g_avg_v,g_d_v] -> per-node grad_avg,grad_d.
        let (mut g_avg, mut g_d) = (vec![0.0f32; n], vec![0.0f32; n]);
        for (e, &(u, v)) in c.tr.iter().enumerate() {
            g_avg[u] += grad_ein[e * 4];
            g_d[u] += grad_ein[e * 4 + 1];
            g_avg[v] += grad_ein[e * 4 + 2];
            g_d[v] += grad_ein[e * 4 + 3];
        }
        // node -> grad_net, grad_absum.
        let (mut g_net, mut g_abs) = (vec![0.0f32; n], vec![0.0f32; n]);
        for v in 0..n {
            let denom = absum[v] + EPS;
            g_net[v] = g_avg[v] / denom;
            g_abs[v] = -g_avg[v] * net[v] / (denom * denom);
            let dv = (absum[v] / c.scale).tanh();
            g_abs[v] += g_d[v] * (1.0 - dv * dv) / c.scale;
        }
        // -> grad_w per edge, then encoder backward (CR only). Warm-start: keep
        // the spline frozen at identity for the first third so the head becomes
        // good BEFORE the spline moves — otherwise head+spline co-diverge into a
        // degenerate basin on ~1/5 seeds (measured 0.62 collapse).
        if enc == Enc::Cr && it >= iters / 3 {
            let mut grad_w = vec![0.0f32; m];
            for (e, &(u, v)) in c.tr.iter().enumerate() {
                grad_w[e] = g_net[u] + g_net[v] + (g_abs[u] + g_abs[v]) * w[e].signum();
            }
            let (cache, control, basis) = cr.unwrap();
            let back = chebyshev_cr_backward(&control, &basis, &cache, &grad_w, K);
            // Conservative coef step keeps the spline near identity (avoids the
            // degenerate basin that would otherwise collapse a seed).
            adam_step(&mut coef, &back.grad_coef, &mut st_c, 0.005);
        }
        adam_step(&mut head.w, &ghead.w, &mut st_w, 0.02);
        adam_step(&mut head.b, &ghead.b, &mut st_b, 0.02);
    }
    let fin = eval(enc, &head, &coef, c);
    (init, fin, coef)
}

fn main() {
    let path = arg_str("--data").expect("--data <edgelist>");
    let seed = arg_f("--seed", 0.0) as u64;
    let iters = arg_f("--iters", 400.0) as usize;

    // Load (u,v,r), relabel nodes.
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
    let mut relabel: HashMap<usize, usize> = HashMap::new();
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
        let nu = relabel.len();
        let ru = *relabel.entry(u).or_insert(nu);
        let nv = relabel.len();
        let rv = *relabel.entry(v).or_insert(nv);
        edges.push((ru, rv, r));
    }
    let n = relabel.len();
    let max_abs = edges.iter().map(|e| e.2.abs()).fold(1e-6f32, f32::max);
    let mean_abs = (edges.iter().map(|e| e.2.abs()).sum::<f32>() / edges.len() as f32).max(1e-6);

    // Deterministic 80/20 split.
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
    let tr: Vec<(usize, usize)> = tr_i.iter().map(|&i| (edges[i].0, edges[i].1)).collect();
    let te: Vec<(usize, usize)> = te_i.iter().map(|&i| (edges[i].0, edges[i].1)).collect();
    let tr_y: Vec<f32> = tr_i
        .iter()
        .map(|&i| (edges[i].2 > 0.0) as u32 as f32)
        .collect();
    let te_y: Vec<u8> = te_i.iter().map(|&i| (edges[i].2 > 0.0) as u8).collect();
    // CR input domain: raw weight normalised to [-1,1].
    let rnorm_tr: Vec<f32> = tr_i
        .iter()
        .map(|&i| (edges[i].2 / max_abs).clamp(-1.0, 1.0))
        .collect();
    let raw_tr: Vec<f32> = tr_i.iter().map(|&i| edges[i].2).collect();
    let scale = 2.0 * tr.len() as f32 / n as f32; // ≈ avg degree
    let ctx = Ctx {
        tr: &tr,
        tr_y: &tr_y,
        rnorm: &rnorm_tr,
        raw: &raw_tr,
        mean_abs,
        scale,
        n,
        te: &te,
        te_y: &te_y,
    };

    let name = path.rsplit(['/', '\\']).next().unwrap_or(&path);
    for (label, enc) in [
        ("binary", Enc::Binary),
        ("tanh", Enc::Tanh),
        ("cr", Enc::Cr),
    ] {
        let (init, fin, coef) = train(enc, &ctx, iters, seed.wrapping_add(1));
        let extra = if enc == Enc::Cr {
            format!(
                "  coef=[{}]",
                coef.iter()
                    .map(|c| format!("{c:.3}"))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        } else {
            String::new()
        };
        println!(
            "{name} seed={seed} {label:>6}: test AUROC init {init:.4} -> final {fin:.4}{extra}"
        );
    }
}
