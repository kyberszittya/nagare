//! E8 — the width/representability boundary: a non-additive (product) cross-clique
//! term. No autograd. Runs on CPU (Mac or anywhere).
//!
//! Each "triple" k contributes an explicit NON-ADDITIVE feature `prod_k = rk0·rk1`
//! (a product of its two residual vars). The target is
//!   y = Σ_k [ w0·rk0 + w1·rk1 + β·prod_k ] + w_h·h + noise,
//! so `β` dials the strength of the non-additive component.
//!
//! The product feature is representable ONLY by a clique wide enough to contain
//! both `rk0` and `rk1`. Three arms:
//!   MF-WIDE  — multifrontal on a width-4 star clique tree `{rk0, rk1, prod_k, h}`:
//!              hosts the product, EXACT (= dense-wide) at O(d·w³).
//!   DENSE-WIDE — InfoEvolventHead over all features incl. products: the ceiling.
//!   NARROW   — InfoEvolventHead over LINEAR features only (products omitted): the
//!              too-narrow model. Structurally cannot fit the β·prod_k term.
//!
//! As β grows, NARROW drops while MF-WIDE == DENSE-WIDE hold — the required
//! treewidth = the target's interaction order; the SBSH width certificate is the
//! check. This is a REPRESENTABILITY limit (β omitted regardless of data), distinct
//! from the estimation gap of E6/E7.
//!
//! Run: `cargo run --release --example evolvent_width -- [--k=N] [--beta10=N] [--seed=N]`

use holonomy_learn::{r2_score, Clique, InfoEvolventHead, JunctionTreeCholesky};

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32) / 4294967296.0
    }
    fn g(&mut self) -> f32 {
        (-2.0 * self.f().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * self.f()).cos()
    }
}

fn main() {
    let arg = |k: &str, d: u64| -> u64 {
        std::env::args()
            .find_map(|a| a.strip_prefix(k).map(|s| s.parse::<u64>().unwrap_or(d)))
            .unwrap_or(d)
    };
    let seed = arg("--seed=", 0);
    let kk = arg("--k=", 12) as usize; // number of triples
    let beta = arg("--beta10=", 10) as f32 / 10.0; // product (non-additive) weight
    let per = 40usize; // rich regime: this is about representability, not scarcity
    let mut rng = Rng(23 + seed);

    // WIDE layout: h=0; triple k -> rk0=1+3k, rk1=2+3k, prod_k=3+3k. D = 1+3K.
    let d = 1 + 3 * kk;
    let (rk0, rk1, prodk) = (
        |k: usize| 1 + 3 * k,
        |k: usize| 2 + 3 * k,
        |k: usize| 3 + 3 * k,
    );
    // NARROW linear-only layout: h=0; rk0->1+2k, rk1->2+2k. D_lin = 1+2K.
    let d_lin = 1 + 2 * kk;

    // true weights
    let w_h = rng.g();
    let w0: Vec<f32> = (0..kk).map(|_| rng.g()).collect();
    let w1: Vec<f32> = (0..kk).map(|_| rng.g()).collect();

    // WIDE star clique tree, width 4, hub h shared: clique k = {rk0, rk1, prod_k, h}
    let mut cliques = Vec::with_capacity(kk);
    cliques.push(Clique {
        vars: vec![rk0(0), rk1(0), prodk(0), 0], // root eliminates h too
        n_res: 4,
        parent: None,
    });
    for k in 1..kk {
        cliques.push(Clique {
            vars: vec![rk0(k), rk1(k), prodk(k), 0],
            n_res: 3, // h is the separator
            parent: Some(0),
        });
    }
    let mut jt = JunctionTreeCholesky::new(cliques, 1.0, d);
    let mut dense = InfoEvolventHead::new(d, 1.0);
    let mut narrow = InfoEvolventHead::new(d_lin, 1.0);

    let gen = |rng: &mut Rng, k: usize| -> (Vec<f32>, Vec<f32>, Vec<f32>, f32) {
        let (a, b, hv) = (rng.g(), rng.g(), rng.g());
        let prod = a * b;
        let y = w0[k] * a + w1[k] * b + beta * prod + w_h * hv + 0.05 * rng.g();
        // clique-local (order = clique vars [rk0, rk1, prod, h])
        let local = vec![a, b, prod, hv];
        // dense-wide global
        let mut gw = vec![0.0f32; d];
        gw[rk0(k)] = a;
        gw[rk1(k)] = b;
        gw[prodk(k)] = prod;
        gw[0] = hv;
        // narrow global (linear only, NO product)
        let mut gn = vec![0.0f32; d_lin];
        gn[0] = hv;
        gn[1 + 2 * k] = a;
        gn[2 + 2 * k] = b;
        (local, gw, gn, y)
    };

    for k in 0..kk {
        for _ in 0..per {
            let (local, gw, gn, y) = gen(&mut rng, k);
            jt.update(k, &local, y);
            dense.update(&gw, y);
            narrow.update(&gn, y);
        }
    }
    let w_mf = jt.solve();
    let w_dense = dense.solve();
    let w_narrow = narrow.solve();

    let (mut pm, mut pd, mut pn, mut yt) = (vec![], vec![], vec![], vec![]);
    for _ in 0..(40 * kk) {
        let k = (rng.f() * kk as f32) as usize % kk;
        let (_l, gw, gn, y) = gen(&mut rng, k);
        pm.push(InfoEvolventHead::predict(&gw, &w_mf));
        pd.push(InfoEvolventHead::predict(&gw, &w_dense));
        pn.push(InfoEvolventHead::predict(&gn, &w_narrow));
        yt.push(y);
    }

    println!(
        "k {kk} beta {beta:.1} seed {seed}  d {d} (narrow {d_lin})  |  R2  MF-WIDE {:.4}  DENSE-WIDE {:.4}  NARROW {:.4}  |  storage MF {} dense {} ({:.1}%)",
        r2_score(&pm, &yt),
        r2_score(&pd, &yt),
        r2_score(&pn, &yt),
        jt.factor_storage(),
        d * d,
        100.0 * jt.factor_storage() as f64 / (d * d) as f64,
    );
}
