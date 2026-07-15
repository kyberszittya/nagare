//! E3 — the pairwise item vs the hypergraph tensor. No autograd.
//!
//! Data has a genuine bounded-width hypergraph coupling: HE hyperedges of 3
//! nodes, target y = sum_e [ strong 3-way term x_a x_b x_c + pairwise + linear ].
//! Two node layouts: DISJOINT hyperedges (feature blocks uncorrelated -> block
//! precision is EXACT) and an OVERLAPPING chain (adjacent edges share a node ->
//! block precision drops the separator coupling -> approximate).
//!
//! Three learners, one online pass each:
//!   A DENSE-PAIRWISE — RLS over nodes + within-edge PAIR products (NO 3-way).
//!                      The "pairwise item": provably cannot represent x_a x_b x_c.
//!   B DENSE-HYPEREDGE — RLS over per-edge {lin,pair,3-way} features, DENSE O(d^2).
//!   C BLOCK-HYPEREDGE — same features, BLOCK precision (one 7x7 per edge), O(d*w).
//!
//! Reports test R^2 and the precision storage (nnz) each arm carries.
//! Run: `cargo run --release --example evolvent_hypergraph -- [--seed=N]`

use holonomy_learn::{r2_score, BlockEvolventHead, EvolventHead};

const HE: usize = 20; // hyperedges
const FPE: usize = 7; // per-edge features: xa xb xc, ab ac bc, abc
const N: usize = 6000;

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        // top 32 bits / 2^32 -> uniform [0,1). (Earlier `>>33 / u32::MAX` gave
        // [0,0.5) -> biased inputs; fixed so E[x]=0 and the orthogonality holds.)
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32) / 4294967296.0
    }
    fn u(&mut self) -> f32 {
        2.0 * self.f() - 1.0
    }
    fn g(&mut self) -> f32 {
        (-2.0 * self.f().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * self.f()).cos()
    }
}

/// per-edge node triples for a layout; returns (edges, n_nodes)
fn edges(overlap: bool) -> (Vec<[usize; 3]>, usize) {
    let mut e = Vec::with_capacity(HE);
    if overlap {
        for i in 0..HE {
            e.push([2 * i, 2 * i + 1, 2 * i + 2]);
        }
        (e, 2 * HE + 1)
    } else {
        for i in 0..HE {
            e.push([3 * i, 3 * i + 1, 3 * i + 2]);
        }
        (e, 3 * HE)
    }
}

/// per-edge {a,b,c, ab,ac,bc, abc} features (grouped by edge -> contiguous blocks)
fn hyper_feats(x: &[f32], e: &[[usize; 3]]) -> Vec<f32> {
    let mut f = Vec::with_capacity(HE * FPE);
    for t in e {
        let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
        f.extend_from_slice(&[a, b, c, a * b, a * c, b * c, a * b * c]);
    }
    f
}

/// nodes + within-edge PAIR products (NO 3-way) — the pairwise item
fn pair_feats(x: &[f32], e: &[[usize; 3]], n: usize) -> Vec<f32> {
    let mut f = x[..n].to_vec();
    for t in e {
        let (a, b, c) = (x[t[0]], x[t[1]], x[t[2]]);
        f.extend_from_slice(&[a * b, a * c, b * c]);
    }
    f
}

fn main() {
    let seed: u64 = std::env::args()
        .find_map(|a| {
            a.strip_prefix("--seed=")
                .map(|s| s.parse::<u64>().unwrap_or(0))
        })
        .unwrap_or(0);

    for overlap in [false, true] {
        let (e, n) = edges(overlap);
        let mut rng = Rng(100 + seed + if overlap { 1 } else { 0 });
        // random target coefficients per edge; 3-way term deliberately strong
        let coefs: Vec<[f32; 7]> = (0..HE)
            .map(|_| {
                let mut c = [0.0f32; 7];
                for v in c.iter_mut().take(6) {
                    *v = 0.3 * rng.u(); // small linear + pairwise
                }
                c[6] = 2.6 * (0.6 + 0.4 * rng.f()); // DOMINANT 3-way term
                c
            })
            .collect();

        // stream: build both feature sets + target
        let mut xs = Vec::with_capacity(N);
        let mut hy = Vec::with_capacity(N);
        let mut pw = Vec::with_capacity(N);
        let mut ys = Vec::with_capacity(N);
        for _ in 0..N {
            let x: Vec<f32> = (0..n).map(|_| rng.u()).collect();
            let hf = hyper_feats(&x, &e);
            let mut y = 0.0f32;
            for (ei, c) in coefs.iter().enumerate() {
                for j in 0..FPE {
                    y += c[j] * hf[ei * FPE + j];
                }
            }
            y += 0.1 * rng.g();
            hy.push(hf);
            pw.push(pair_feats(&x, &e, n));
            ys.push(y);
            xs.push(x);
        }
        let ntr = N * 3 / 4;
        let d_h = HE * FPE;
        let d_p = pw[0].len();

        // A: dense-pairwise RLS (no 3-way)
        let mut a = EvolventHead::new(d_p, 1, 1.0, 1.0);
        for i in 0..ntr {
            a.update(&pw[i], &[ys[i]]);
        }
        // B: dense-hyperedge RLS
        let mut b = EvolventHead::new(d_h, 1, 1.0, 1.0);
        for i in 0..ntr {
            b.update(&hy[i], &[ys[i]]);
        }
        // C: block-hyperedge RLS (one FPE-block per hyperedge)
        let mut c = BlockEvolventHead::new(&[FPE; HE], 1.0, 1.0);
        for i in 0..ntr {
            c.update(&hy[i], ys[i]);
        }

        // eval R2
        let (mut pa, mut pb, mut pc, mut yt) = (vec![], vec![], vec![], vec![]);
        for i in ntr..N {
            pa.push(a.predict(&pw[i])[0]);
            pb.push(b.predict(&hy[i])[0]);
            pc.push(c.predict(&hy[i]));
            yt.push(ys[i]);
        }
        let nnz_dense = (d_h * d_h) as f64;
        let nnz_block = c.precision_nnz() as f64;
        let layout = if overlap {
            "OVERLAP chain"
        } else {
            "DISJOINT   "
        };
        println!(
            "{layout}  R2  A pairwise {:.3}  |  B dense-hyper {:.3}  C block-hyper {:.3}  |  precision nnz: dense {} block {} ({:.0}x)",
            r2_score(&pa, &yt),
            r2_score(&pb, &yt),
            r2_score(&pc, &yt),
            nnz_dense as u64,
            nnz_block as u64,
            nnz_dense / nnz_block
        );
    }
    println!("(A pairwise cannot represent the 3-way term; B/C can. C matches B when disjoint, approximates on overlap.)");
}
