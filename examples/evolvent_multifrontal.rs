//! E5 — multifrontal (clique-tree) Cholesky over a BRANCHING bounded-width
//! hypergraph. No autograd. The sparse solve-time realization of the E4
//! information form: the hypergraph clique tree drives the factorization, the
//! separator Schur complements are the messages.
//!
//! Branching binary clique tree (`balanced_binary_tree`): each clique is a
//! hyperedge, adjacent cliques share a `sep`-variable separator. Local
//! measurements are homed at cliques. Four things measured:
//! 1. R² — MULTIFRONTAL == DENSE exactly (both solve the same J); BLOCK
//!    (separator-dropping) trails — here the coupling is load-bearing.
//! 2. STORAGE — multifrontal frontals `Σ|C|²` vs dense `d²`.
//! 3. FLOPS — factorization `Σ|C|³` vs dense `d³/6`.
//! 4. LOCALITY — an online update touches only its clique's path to the root
//!    (mean path length vs #cliques) → incremental re-fire is O(depth·w³).
//!
//! Run: `cargo run --release --example evolvent_multifrontal -- [--depth=N] [--seed=N]`

use holonomy_learn::{
    balanced_binary_tree, r2_score, star_clique_tree, InfoEvolventHead, JunctionTreeCholesky,
};

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
    fn idx(&mut self, n: usize) -> usize {
        (self.f() * n as f32) as usize % n
    }
}

fn main() {
    let arg = |k: &str, d: u64| -> u64 {
        std::env::args()
            .find_map(|a| a.strip_prefix(k).map(|s| s.parse::<u64>().unwrap_or(d)))
            .unwrap_or(d)
    };
    let seed = arg("--seed=", 0);
    let depth = arg("--depth=", 5) as usize;
    // measurements homed at each clique. Low `per` (<= clique arity) is the
    // DATA-SCARCE regime where no clique determines its own vars, so the separator
    // coupling MF keeps (and BLOCK drops) becomes load-bearing.
    let per = arg("--per=", 60) as usize;
    // fanout>0 switches to a STAR topology: one separator shared across `fanout`
    // children (separator-sharing axis). fanout=0 => balanced binary tree.
    let fanout = arg("--fanout=", 0) as usize;
    let (sep, res) = (if fanout > 0 { 3 } else { 2 }, 3usize);
    let mut rng = Rng(11 + seed);

    let (cliques, d) = if fanout > 0 {
        star_clique_tree(fanout, sep, res, res)
    } else {
        balanced_binary_tree(depth, sep, res)
    };
    let nc = cliques.len();
    // true weights over all d features
    let w_true: Vec<f32> = (0..d).map(|_| rng.g()).collect();

    let mut jt = JunctionTreeCholesky::new(cliques.clone(), 1.0, d);
    let mut dense = InfoEvolventHead::new(d, 1.0);

    let gen = |rng: &mut Rng, c: usize| -> (Vec<f32>, Vec<f32>, f32) {
        // local phi over clique c's vars; y = phi . w_true(restricted); global phi too
        let vars = &cliques[c].vars;
        let m = vars.len();
        let phi_local: Vec<f32> = (0..m).map(|_| rng.g()).collect();
        let mut phi_global = vec![0.0f32; d];
        let mut y = 0.0f32;
        for i in 0..m {
            phi_global[vars[i]] = phi_local[i];
            y += phi_local[i] * w_true[vars[i]];
        }
        y += 0.05 * rng.g();
        (phi_local, phi_global, y)
    };

    // train
    for c in 0..nc {
        for _ in 0..per {
            let (pl, pg, y) = gen(&mut rng, c);
            jt.update(c, &pl, y);
            dense.update(&pg, y);
        }
    }
    let w_mf = jt.solve();
    let w_dense = dense.solve();
    let w_blk = jt.solve_block_diagonal();

    // weight-recovery RMSE vs the true weights (the estimation gap: BLOCK's
    // separator vars are estimated from one clique, MF pools across the tree)
    let wrmse = |w: &[f32]| -> f32 {
        (w.iter()
            .zip(&w_true)
            .map(|(&a, &b)| (a - b) * (a - b))
            .sum::<f32>()
            / d as f32)
            .sqrt()
    };
    let (wr_mf, wr_blk) = (wrmse(&w_mf), wrmse(&w_blk));

    // held-out test set
    let (mut pm, mut pd, mut pb, mut yt) = (vec![], vec![], vec![], vec![]);
    for _ in 0..(20 * nc) {
        let c = rng.idx(nc);
        let (_pl, pg, y) = gen(&mut rng, c);
        pm.push(InfoEvolventHead::predict(&pg, &w_mf));
        pd.push(InfoEvolventHead::predict(&pg, &w_dense));
        pb.push(InfoEvolventHead::predict(&pg, &w_blk));
        yt.push(y);
    }

    // locality: mean path-to-root over leaves vs #cliques
    let is_parent: Vec<bool> = {
        let mut v = vec![false; nc];
        for cl in &cliques {
            if let Some(p) = cl.parent {
                v[p] = true;
            }
        }
        v
    };
    let leaves: Vec<usize> = (0..nc).filter(|&c| !is_parent[c]).collect();
    let mean_path: f64 = leaves
        .iter()
        .map(|&c| jt.ancestors_inclusive(c).len() as f64)
        .sum::<f64>()
        / leaves.len() as f64;

    let dense_store = d * d;
    let dense_flops = (d as u64).pow(3) / 6;
    println!(
        "depth {depth} fanout {fanout} seed {seed} per {per}  d {d} cliques {nc}  |  R2 MF {:.4} DENSE {:.4} BLOCK {:.4}  |  wRMSE MF {:.4} BLOCK {:.4}  |  storage MF {} dense {} ({:.1}%)  flops MF {} dense~{} ({:.0}x)  |  locality: mean path {:.1} / {} cliques",
        r2_score(&pm, &yt),
        r2_score(&pd, &yt),
        r2_score(&pb, &yt),
        wr_mf,
        wr_blk,
        jt.factor_storage(),
        dense_store,
        100.0 * jt.factor_storage() as f64 / dense_store as f64,
        jt.factor_flops(),
        dense_flops,
        dense_flops as f64 / jt.factor_flops() as f64,
        mean_path,
        nc,
    );
}
