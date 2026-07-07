//! Pure-Nagare closed-form signed-link prediction on a real signed graph.
//!
//! Loads an edge list `u,v,rating`, splits edges 80/20, computes leakage-free
//! holonomy features (signed degrees + triad holonomy $\sum_w w(u,w)\,w(w,v)$
//! over common neighbours), and trains a closed-form logistic with Nagare's
//! *local* update rule (analytic gradient, no autograd tape) using the shipped
//! `linear_forward` kernel. Reports test AUROC. With `--weighted`, edge values
//! are the real rating in [-1,1] instead of the sign.
//!
//! Run: `cargo run --release --example signed_link -- --data path.csv --scale 10 [--weighted]`

use std::collections::HashMap;

use holonomy_learn::{linear_forward, LinearLayer};

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
    let rp: f64 = (0..n).filter(|&k| labels[k] == 1).map(|k| rank[k]).sum();
    (rp - (npos * (npos + 1)) as f64 / 2.0) / (npos * nneg) as f64
}

fn main() {
    let path = arg_str("--data").expect("--data <edgelist.csv>");
    let scale = arg_f("--scale", 1.0);
    let weighted = std::env::args().any(|a| a == "--weighted");
    let seed = arg_f("--seed", 0.0) as u64;

    // load (u, v, rating)
    let mut edges: Vec<(usize, usize, f32)> = Vec::new();
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
        edges.push((
            p[0].parse().unwrap(),
            p[1].parse().unwrap(),
            p[2].parse().unwrap(),
        ));
    }
    // deterministic shuffle (LCG on seed) + 80/20 split
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

    // train adjacency (leakage-free) with edge value = sign or weighted rating
    let mut adj: HashMap<usize, HashMap<usize, f32>> = HashMap::new();
    let mut pos: HashMap<usize, f32> = HashMap::new();
    let mut neg: HashMap<usize, f32> = HashMap::new();
    for &e in tr_i {
        let (u, v, r) = edges[e];
        let val = if weighted {
            (r / scale).clamp(-1.0, 1.0)
        } else {
            r.signum()
        };
        adj.entry(u).or_default().insert(v, val);
        adj.entry(v).or_default().insert(u, val);
        *pos.entry(u).or_default() += (r > 0.0) as u32 as f32;
        *neg.entry(u).or_default() += (r < 0.0) as u32 as f32;
        *pos.entry(v).or_default() += (r > 0.0) as u32 as f32;
        *neg.entry(v).or_default() += (r < 0.0) as u32 as f32;
    }
    let deg = |m: &HashMap<usize, f32>, k: usize| m.get(&k).copied().unwrap_or(0.0);
    // features per edge: [logdeg(4), triad holonomy, log support, bias handled by layer]
    const NF: usize = 6;
    let feats = |u: usize, v: usize| -> [f32; NF] {
        let (mut holo, mut supp) = (0.0f32, 0.0f32);
        if let (Some(nu), Some(nv)) = (adj.get(&u), adj.get(&v)) {
            let (small, big) = if nu.len() < nv.len() {
                (nu, nv)
            } else {
                (nv, nu)
            };
            for (w, &wa) in small {
                if let Some(&wb) = big.get(w) {
                    holo += wa * wb;
                    supp += 1.0;
                }
            }
        }
        [
            (deg(&pos, u) + 1.0).ln(),
            (deg(&neg, u) + 1.0).ln(),
            (deg(&pos, v) + 1.0).ln(),
            (deg(&neg, v) + 1.0).ln(),
            holo.signum() * (holo.abs() + 1.0).ln(),
            (supp + 1.0).ln(),
        ]
    };
    let build = |idx: &[usize]| -> (Vec<f32>, Vec<u8>) {
        let mut x = Vec::with_capacity(idx.len() * NF);
        let mut y = Vec::with_capacity(idx.len());
        for &e in idx {
            let (u, v, r) = edges[e];
            x.extend_from_slice(&feats(u, v));
            y.push((r > 0.0) as u8);
        }
        (x, y)
    };
    let (mut xtr, ytr) = build(tr_i);
    let (mut xte, yte) = build(te_i);
    // standardise by train stats
    let mut mu = [0.0f32; NF];
    let mut sd = [0.0f32; NF];
    let ntr = ytr.len();
    for r in xtr.chunks(NF) {
        for j in 0..NF {
            mu[j] += r[j] / ntr as f32;
        }
    }
    for r in xtr.chunks(NF) {
        for j in 0..NF {
            sd[j] += (r[j] - mu[j]).powi(2) / ntr as f32;
        }
    }
    for s in &mut sd {
        *s = s.sqrt() + 1e-6;
    }
    for buf in [&mut xtr, &mut xte] {
        for r in buf.chunks_mut(NF) {
            for j in 0..NF {
                r[j] = (r[j] - mu[j]) / sd[j];
            }
        }
    }
    // closed-form logistic via Nagare local update: W += -lr * (p - y) * phi
    let mut layer = LinearLayer::new(NF, 1, 7);
    let lr = 0.3;
    for _ in 0..500 {
        let logits = linear_forward(&layer, &xtr); // Nagare kernel forward
        let mut gw = [0.0f32; NF];
        let mut gb = 0.0f32;
        for (i, row) in xtr.chunks(NF).enumerate() {
            let p = 1.0 / (1.0 + (-logits[i]).exp());
            let d = p - ytr[i] as f32;
            for j in 0..NF {
                gw[j] += d * row[j] / ntr as f32;
            }
            gb += d / ntr as f32;
        }
        for (w, &g) in layer.w.iter_mut().zip(gw.iter()) {
            *w -= lr * (g + 1e-3 * *w);
        }
        layer.b[0] -= lr * gb;
    }
    let sc = linear_forward(&layer, &xte);
    let a = auroc(&sc, &yte);
    println!(
        "{}  edges={} train={} test={} weighted={} | pure-Nagare closed-form AUROC={:.4}",
        path.rsplit(['/', '\\']).next().unwrap_or(&path),
        edges.len(),
        ntr,
        yte.len(),
        weighted,
        a
    );
}
