//! E4 — the junction-tree (information-form) evolvent vs the block-diagonal
//! approximation. No autograd.
//!
//! A path of N features; each sample is a LOCAL measurement over a window of W
//! consecutive features whose values are CORRELATED (a short random walk), so
//! neighbours -- including across block boundaries -- are coupled. The target is
//! linear in the window. Because measurements cross block boundaries, the
//! cross-block (separator) coupling is load-bearing.
//!
//! Three arms:
//!   DENSE  — EvolventHead over all N features: exact, O(N^2) precision.
//!   BLOCK  — BlockEvolventHead (contiguous B-blocks): drops cross-block coupling.
//!   INFO   — InfoEvolventHead (junction-tree information form): EXACT (= dense),
//!            but the information matrix is SPARSE (banded) -> O(nnz)=O(N*W).
//!
//! Reports test R^2 and the information-matrix sparsity.
//! Run: `cargo run --release --example evolvent_junction -- [--seed=N]`

use holonomy_learn::{r2_score, BlockEvolventHead, EvolventHead, InfoEvolventHead};

const W: usize = 6; // measurement window
const B: usize = 6; // block size -> N/B blocks
const NS: usize = 8000; // samples

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32) / 4294967296.0 // uniform [0,1)
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
    let n = arg("--n=", 48) as usize; // path length (features), multiple of B
    let mut rng = Rng(7 + seed);
    // smooth true weights over the path
    let mut c = vec![0.0f32; n];
    c[0] = rng.g();
    for i in 1..n {
        c[i] = 0.8 * c[i - 1] + 0.6 * rng.g();
    }

    // stream of local correlated windows + target
    let mut xs = Vec::with_capacity(NS);
    let mut ys = Vec::with_capacity(NS);
    for _ in 0..NS {
        let s = (rng.f() * (n - W + 1) as f32) as usize % (n - W + 1);
        let mut x = vec![0.0f32; n];
        let mut v = rng.g();
        let mut y = 0.0f32;
        for k in 0..W {
            v = 0.7 * v + 0.7 * rng.g(); // correlated window values
            x[s + k] = v;
            y += c[s + k] * v;
        }
        y += 0.05 * rng.g();
        xs.push(x);
        ys.push(y);
    }
    let ntr = NS * 3 / 4;

    // DENSE
    let mut dense = EvolventHead::new(n, 1, 1.0, 1.0);
    for i in 0..ntr {
        dense.update(&xs[i], &[ys[i]]);
    }
    // BLOCK (contiguous B-blocks)
    let nblk = n / B;
    let mut block = BlockEvolventHead::new(&vec![B; nblk], 1.0, 1.0);
    for i in 0..ntr {
        block.update(&xs[i], ys[i]);
    }
    // INFO (junction-tree information form)
    let mut info = InfoEvolventHead::new(n, 1.0);
    for i in 0..ntr {
        info.update(&xs[i], ys[i]);
    }
    let w_info = info.solve();

    let (mut pd, mut pb, mut pi, mut yt) = (vec![], vec![], vec![], vec![]);
    for i in ntr..NS {
        pd.push(dense.predict(&xs[i])[0]);
        pb.push(block.predict(&xs[i]));
        pi.push(InfoEvolventHead::predict(&xs[i], &w_info));
        yt.push(ys[i]);
    }
    let dense_nnz = n * n;
    println!(
        "n {n}  seed {seed}  R2  DENSE {:.4}  BLOCK {:.4}  INFO {:.4}  |  precision/info nnz: dense {} info {} ({:.1}% of dense)",
        r2_score(&pd, &yt),
        r2_score(&pb, &yt),
        r2_score(&pi, &yt),
        dense_nnz,
        info.nnz(),
        100.0 * info.nnz() as f64 / dense_nnz as f64
    );
}
