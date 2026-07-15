//! Multifrontal (clique-tree) Cholesky for a bounded-width SPD information matrix.
//! No autograd — this is an exact linear solver, the sparse realization of the
//! E4 information form (`InfoEvolventHead`).
//!
//! A **clique tree** with the running-intersection property is the numerical face
//! of a bounded-width signed hypergraph: each clique is a hyperedge (a small
//! tensor block), and adjacent cliques share a **separator** (the coupling the
//! block-diagonal `BlockEvolventHead` drops and this solver keeps).
//!
//! Each clique owns its variables ordered RESIDUAL-first then SEPARATOR, a frontal
//! matrix over those variables, and a frontal right-hand side. Factorization
//! sweeps **leaves → root**: eliminate each clique's residual by a small dense
//! Cholesky, then send the **Schur complement over its separator** up to the
//! parent. That message `U = F_SS − F_SRᵀ F_RR⁻¹ F_RS` is a tensor contraction
//! over the eliminated interior indices onto the separator boundary — structurally
//! the `hg_message` edge→node incidence contraction. Back-substitution sweeps
//! **root → leaves**: `x_R = yr − W_RS · x_S`.
//!
//! # Contract
//! - **Preconditions:** the cliques form a tree (each non-root has exactly one
//!   parent) with running intersection; a clique's separator variables are a
//!   subset of its parent's variables; the assembled `J` is SPD.
//! - **Postconditions:** [`JunctionTreeCholesky::solve`] returns the exact
//!   solution of `J x = b` (identical to a dense Cholesky of the assembled `J`).
//! - **Complexity:** storage `O(Σ_c |C|²)`, factorization `O(Σ_c |C|³)` — both
//!   `O(d·w²)` / `O(d·w³)` for width `w`, vs dense `O(d²)` / `O(d³)`.

/// One clique (hyperedge) of the tree. `vars` are global feature indices ordered
/// residual-first: `vars[..n_res]` are eliminated at this clique, `vars[n_res..]`
/// are the separator shared with `parent`.
#[derive(Clone, Debug)]
pub struct Clique {
    pub vars: Vec<usize>,
    pub n_res: usize,
    pub parent: Option<usize>,
}

/// Multifrontal Cholesky over a clique tree. Measurements accumulate into per-
/// clique frontal blocks; [`solve`](Self::solve) factorizes and back-substitutes.
///
/// Storage is **contiguous**: every per-clique block (frontals, rhs, and the
/// Cholesky factors) lives in a single flat arena addressed by precomputed
/// offsets, and the parent-local assembly map is precomputed once. A `solve`
/// therefore clones two flat `Vec<f32>`s and runs on reused scratch — no per-clique
/// `Vec` allocation and no `position()` search in the hot path.
#[derive(Clone, Debug)]
pub struct JunctionTreeCholesky {
    cliques: Vec<Clique>,
    ridge: f32,
    d: usize,
    order: Vec<usize>, // post-order: children before parents (leaves..root)
    // contiguous per-clique geometry (precomputed once in `new`):
    m: Vec<usize>,         // clique size |C|
    nres: Vec<usize>,      // residual count (eliminated here)
    nsep: Vec<usize>,      // separator count (shared with parent)
    foff: Vec<usize>,      // offset of the |C|*|C| frontal in the flat arenas
    boff: Vec<usize>,      // offset of the |C| rhs
    loff: Vec<usize>,      // offset of the r*r Cholesky factor
    woff: Vec<usize>,      // offset of the r*s W_RS block
    yoff: Vec<usize>,      // offset of the r yr vector
    sploc_off: Vec<usize>, // offset into `sploc` of this clique's separator map
    sploc: Vec<usize>,     // concatenated parent-local separator positions (assembly)
    // persistent accumulated measurements (flat; never overwritten by a factorize):
    acc_front: Vec<f32>,
    acc_rhs: Vec<f32>,
    // factor storage (flat; rebuilt each factorize), for back-substitution:
    l_rr: Vec<f32>,
    w_rs: Vec<f32>,
    yr: Vec<f32>,
}

impl JunctionTreeCholesky {
    /// Build a solver over `cliques` (a valid clique tree) with prior precision
    /// `ridge > 0` on the diagonal of every variable (added once, at the variable's
    /// eliminating/residual clique).
    ///
    /// # Panics
    /// If `ridge <= 0`, the tree has no unique root, or a separator is not a subset
    /// of its parent's variables.
    pub fn new(cliques: Vec<Clique>, ridge: f32, d: usize) -> Self {
        assert!(ridge > 0.0, "ridge must be > 0");
        // validate parents + separator-subset-of-parent (contract precondition)
        let mut n_roots = 0;
        for (i, c) in cliques.iter().enumerate() {
            match c.parent {
                None => n_roots += 1,
                Some(p) => {
                    assert!(p < cliques.len() && p != i, "bad parent index");
                    let pv = &cliques[p].vars;
                    for s in &c.vars[c.n_res..] {
                        assert!(pv.contains(s), "separator var {s} not in parent {p}");
                    }
                }
            }
        }
        assert_eq!(n_roots, 1, "clique tree must have exactly one root");
        let order = post_order(&cliques);
        let n = cliques.len();

        // precompute per-clique geometry + flat-arena offsets
        let (mut m, mut nres, mut nsep) = (vec![0; n], vec![0; n], vec![0; n]);
        let (mut foff, mut boff) = (vec![0; n], vec![0; n]);
        let (mut loff, mut woff, mut yoff) = (vec![0; n], vec![0; n], vec![0; n]);
        let (mut ft, mut bt, mut lt, mut wt, mut yt) = (0, 0, 0, 0, 0);
        for (c, cl) in cliques.iter().enumerate() {
            let (mc, rc) = (cl.vars.len(), cl.n_res);
            let sc = mc - rc;
            m[c] = mc;
            nres[c] = rc;
            nsep[c] = sc;
            foff[c] = ft;
            ft += mc * mc;
            boff[c] = bt;
            bt += mc;
            loff[c] = lt;
            lt += rc * rc;
            woff[c] = wt;
            wt += rc * sc;
            yoff[c] = yt;
            yt += rc;
        }
        // assembly map: parent-local index of each separator var (searched once)
        let mut sploc_off = vec![0; n];
        let mut sploc: Vec<usize> = Vec::new();
        for (c, cl) in cliques.iter().enumerate() {
            sploc_off[c] = sploc.len();
            if let Some(p) = cl.parent {
                for g in &cl.vars[cl.n_res..] {
                    let pos = cliques[p]
                        .vars
                        .iter()
                        .position(|v| v == g)
                        .expect("separator var in parent");
                    sploc.push(pos);
                }
            }
        }

        JunctionTreeCholesky {
            cliques,
            ridge,
            d,
            order,
            m,
            nres,
            nsep,
            foff,
            boff,
            loff,
            woff,
            yoff,
            sploc_off,
            sploc,
            acc_front: vec![0.0; ft],
            acc_rhs: vec![0.0; bt],
            l_rr: vec![0.0; lt],
            w_rs: vec![0.0; wt],
            yr: vec![0.0; yt],
        }
    }

    /// Number of cliques.
    pub fn n_cliques(&self) -> usize {
        self.cliques.len()
    }

    /// Local information update homed at `clique`: `F += φφᵀ`, `b += φ·y`, where
    /// `phi_local` is in the clique's local variable order. `O(|C|²)`.
    ///
    /// # Panics
    /// If `phi_local.len()` differs from the clique's variable count.
    pub fn update(&mut self, clique: usize, phi_local: &[f32], y: f32) {
        let m = self.m[clique];
        assert_eq!(phi_local.len(), m, "phi_local must match clique arity");
        let (fo, bo) = (self.foff[clique], self.boff[clique]);
        for i in 0..m {
            let pi = phi_local[i];
            if pi == 0.0 {
                continue;
            }
            self.acc_rhs[bo + i] += pi * y;
            let row = fo + i * m;
            for (j, &pj) in phi_local.iter().enumerate() {
                self.acc_front[row + j] += pi * pj;
            }
        }
    }

    /// Total frontal/factor storage (nonzero-block footprint) vs a dense `d²`.
    pub fn factor_storage(&self) -> usize {
        self.cliques
            .iter()
            .map(|c| c.vars.len() * c.vars.len())
            .sum()
    }

    /// Factorization flop estimate `Σ_c |C|³` vs a dense `d³/6`.
    pub fn factor_flops(&self) -> u64 {
        self.cliques
            .iter()
            .map(|c| (c.vars.len() as u64).pow(3))
            .sum()
    }

    /// Path from clique `c` up to the root (inclusive), leaf-first. An online
    /// update homed at `c` can only change the factors of these cliques (the
    /// Schur messages flow strictly upward); off-path subtrees are untouched.
    pub fn ancestors_inclusive(&self, c: usize) -> Vec<usize> {
        let mut path = vec![c];
        let mut cur = c;
        while let Some(p) = self.cliques[cur].parent {
            path.push(p);
            cur = p;
        }
        path
    }

    /// Cheap signature of clique `c`'s stored Cholesky factor (`Σ|L_RR|`), valid
    /// after a [`solve`](Self::solve). Detects which factors an update changed.
    pub fn factor_checksum(&self, c: usize) -> f64 {
        let (lo, r) = (self.loff[c], self.nres[c]);
        self.l_rr[lo..lo + r * r]
            .iter()
            .map(|v| v.abs() as f64)
            .sum()
    }

    /// Factorize (multifrontal Cholesky) and back-substitute; returns the exact
    /// global solution `x` of `J x = b` (length `d`).
    pub fn solve(&mut self) -> Vec<f32> {
        self.factorize();
        self.back_substitute()
    }

    /// Block-diagonal (separator-dropping) baseline — the E3 `BlockEvolventHead`
    /// approximation on the clique tree: solve each clique's own frontal in
    /// isolation (no Schur messages up or down) and read each variable from the
    /// clique where it is residual. Discards exactly the cross-clique coupling
    /// [`solve`](Self::solve) keeps, so it measures what that coupling is worth.
    pub fn solve_block_diagonal(&self) -> Vec<f32> {
        let mut x = vec![0.0f32; self.d];
        let mw = self.m.iter().copied().max().unwrap_or(0);
        let (mut a, mut l, mut yb, mut xl) = (
            vec![0.0f32; mw * mw],
            vec![0.0f32; mw * mw],
            vec![0.0f32; mw],
            vec![0.0f32; mw],
        );
        for (c, cl) in self.cliques.iter().enumerate() {
            let (m, fo, bo) = (self.m[c], self.foff[c], self.boff[c]);
            a[..m * m].copy_from_slice(&self.acc_front[fo..fo + m * m]);
            for i in 0..m {
                a[i * m + i] += self.ridge; // prior on every local var (separators too, else singular)
            }
            cholesky_lower_into(&a[..m * m], m, &mut l[..m * m]);
            forward_solve_into(&l[..m * m], &self.acc_rhs[bo..bo + m], m, &mut yb[..m]);
            back_solve_into(&l[..m * m], &yb[..m], m, &mut xl[..m]);
            for r in 0..cl.n_res {
                x[cl.vars[r]] = xl[r]; // each var read from its residual (eliminating) clique
            }
        }
        x
    }

    /// Leaves→root elimination. Builds working frontals from the accumulated
    /// measurements + ridge, eliminates each clique's residual, and assembles the
    /// Schur-complement message into the parent.
    fn factorize(&mut self) {
        // working frontals = accumulated measurements + ridge on residual diagonal
        let mut front = self.acc_front.clone();
        let mut rhs = self.acc_rhs.clone();
        for c in 0..self.cliques.len() {
            let (m, fo) = (self.m[c], self.foff[c]);
            for i in 0..self.nres[c] {
                front[fo + i * m + i] += self.ridge;
            }
        }
        // scratch reused across cliques — no per-clique allocation in the hot loop
        let mw = self.m.iter().copied().max().unwrap_or(0);
        let mut frr = vec![0.0f32; mw * mw];
        let mut zb = vec![0.0f32; mw * mw]; // Z, indexed i*s + col
        let mut colb = vec![0.0f32; mw];
        let mut solb = vec![0.0f32; mw];
        let mut yzb = vec![0.0f32; mw];

        for oi in 0..self.order.len() {
            let c = self.order[oi];
            let (m, r, s) = (self.m[c], self.nres[c], self.nsep[c]);
            let (fo, bo, lo, wo, yo) = (
                self.foff[c],
                self.boff[c],
                self.loff[c],
                self.woff[c],
                self.yoff[c],
            );
            // F_RR = r*r top-left of the working frontal (row stride m) -> Cholesky
            for i in 0..r {
                for j in 0..r {
                    frr[i * r + j] = front[fo + i * m + j];
                }
            }
            cholesky_lower_into(&frr[..r * r], r, &mut self.l_rr[lo..lo + r * r]);
            // Z[:,col] = L^-1 F_RS[:,col]
            for col in 0..s {
                for i in 0..r {
                    colb[i] = front[fo + i * m + (r + col)];
                }
                forward_solve_into(&self.l_rr[lo..lo + r * r], &colb[..r], r, &mut solb[..r]);
                for (i, &v) in solb[..r].iter().enumerate() {
                    zb[i * s + col] = v;
                }
            }
            // yz = L^-1 b_R
            colb[..r].copy_from_slice(&rhs[bo..bo + r]);
            forward_solve_into(&self.l_rr[lo..lo + r * r], &colb[..r], r, &mut yzb[..r]);
            // W_RS[:,col] = L^-T Z[:,col]  (stored for back-substitution)
            for col in 0..s {
                for i in 0..r {
                    colb[i] = zb[i * s + col];
                }
                back_solve_into(&self.l_rr[lo..lo + r * r], &colb[..r], r, &mut solb[..r]);
                for (i, &v) in solb[..r].iter().enumerate() {
                    self.w_rs[wo + i * s + col] = v;
                }
            }
            // yr = L^-T yz
            back_solve_into(
                &self.l_rr[lo..lo + r * r],
                &yzb[..r],
                r,
                &mut self.yr[yo..yo + r],
            );

            // Schur message to parent: U = F_SS - Z^T Z; m_b = b_S - Z^T yz
            if let Some(p) = self.cliques[c].parent {
                let (mp, fpo, bpo) = (self.m[p], self.foff[p], self.boff[p]);
                let sp = self.sploc_off[c]; // precomputed parent-local separator indices
                for a in 0..s {
                    let pa = self.sploc[sp + a];
                    let mut zy = 0.0f32;
                    for i in 0..r {
                        zy += zb[i * s + a] * yzb[i];
                    }
                    let mb = rhs[bo + r + a] - zy;
                    rhs[bpo + pa] += mb;
                    for b in 0..s {
                        let pb = self.sploc[sp + b];
                        let mut zz = 0.0f32;
                        for i in 0..r {
                            zz += zb[i * s + a] * zb[i * s + b];
                        }
                        let u = front[fo + (r + a) * m + (r + b)] - zz;
                        front[fpo + pa * mp + pb] += u;
                    }
                }
            }
        }
    }

    /// Root→leaves back-substitution using the stored factors. `x_R = yr − W_RS x_S`.
    fn back_substitute(&self) -> Vec<f32> {
        let mut x = vec![0.0f32; self.d];
        for &c in self.order.iter().rev() {
            let (r, s) = (self.nres[c], self.nsep[c]);
            let (wo, yo) = (self.woff[c], self.yoff[c]);
            let vars = &self.cliques[c].vars;
            for i in 0..r {
                // x_R = yr − W_RS x_S  (x_S already solved in an ancestor)
                let mut v = self.yr[yo + i];
                for a in 0..s {
                    v -= self.w_rs[wo + i * s + a] * x[vars[r + a]];
                }
                x[vars[i]] = v;
            }
        }
        x
    }
}

/// Build a balanced **binary** bounded-width clique tree: the root introduces
/// `res + 2*sep` fresh variables; every clique spawns up to two children, each
/// borrowing a distinct `sep`-variable separator from the parent's fresh vars and
/// introducing `res + 2*sep` fresh residual vars of its own. Returns the cliques
/// and the total variable count `d`. Width is `res + 3*sep` (residual + own
/// separator); the tree forks, so the multifrontal factorization is a genuine
/// tree, not a band.
///
/// # Preconditions
/// `depth >= 1`, `sep >= 1`, `res >= 1`.
pub fn balanced_binary_tree(depth: usize, sep: usize, res: usize) -> (Vec<Clique>, usize) {
    assert!(depth >= 1 && sep >= 1 && res >= 1);
    let mut cliques: Vec<Clique> = Vec::new();
    let mut next = 0usize;
    let fresh_per_clique = res + 2 * sep; // enough to seed two children with distinct separators
    let take = |n: usize, next: &mut usize| -> Vec<usize> {
        (0..n)
            .map(|_| {
                let v = *next;
                *next += 1;
                v
            })
            .collect()
    };
    let root_vars = take(fresh_per_clique, &mut next);
    cliques.push(Clique {
        vars: root_vars,
        n_res: fresh_per_clique,
        parent: None,
    });
    let mut frontier = vec![0usize];
    for _ in 1..depth {
        let mut nextf = Vec::new();
        for &p in &frontier {
            for child in 0..2 {
                let pv = cliques[p].vars.clone();
                let base = child * sep;
                let sep_vars: Vec<usize> = (0..sep).map(|i| pv[base + i]).collect();
                let mut vars = take(fresh_per_clique, &mut next);
                let n_res = vars.len();
                vars.extend_from_slice(&sep_vars);
                let id = cliques.len();
                cliques.push(Clique {
                    vars,
                    n_res,
                    parent: Some(p),
                });
                nextf.push(id);
            }
        }
        frontier = nextf;
    }
    (cliques, next)
}

/// Build a **star** clique tree: one root and `fanout` children that ALL share the
/// SAME `sep`-variable separator (the root's first `sep` vars). The shared
/// separator therefore appears in `fanout + 1` cliques — the separator-sharing
/// axis the binary tree cannot express (there each separator serves one child).
/// Root introduces `sep + res_root` residual vars; each child introduces
/// `res_child` residual vars plus the shared separator. Child arity `res_child +
/// sep` is independent of `fanout`, so data-scarcity is held fixed as sharing
/// grows. Returns the cliques and total variable count `d`.
///
/// # Preconditions
/// `fanout >= 1`, `sep >= 1`, `res_root >= 0`, `res_child >= 1`.
pub fn star_clique_tree(
    fanout: usize,
    sep: usize,
    res_root: usize,
    res_child: usize,
) -> (Vec<Clique>, usize) {
    assert!(fanout >= 1 && sep >= 1 && res_child >= 1);
    let mut next = 0usize;
    let take = |n: usize, next: &mut usize| -> Vec<usize> {
        (0..n)
            .map(|_| {
                let v = *next;
                *next += 1;
                v
            })
            .collect()
    };
    let sep_vars = take(sep, &mut next); // shared separator (residual at root)
    let mut root_vars = sep_vars.clone();
    root_vars.extend(take(res_root, &mut next));
    let n_res_root = root_vars.len();
    let mut cliques = vec![Clique {
        vars: root_vars,
        n_res: n_res_root,
        parent: None,
    }];
    for _ in 0..fanout {
        let mut vars = take(res_child, &mut next);
        let n_res = vars.len();
        vars.extend_from_slice(&sep_vars); // separator = the shared vars
        cliques.push(Clique {
            vars,
            n_res,
            parent: Some(0),
        });
    }
    (cliques, next)
}

/// Post-order (children before parents) over the clique forest rooted at the
/// unique root. Iterative to avoid recursion depth limits.
fn post_order(cliques: &[Clique]) -> Vec<usize> {
    let n = cliques.len();
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut root = 0;
    for (i, c) in cliques.iter().enumerate() {
        match c.parent {
            Some(p) => children[p].push(i),
            None => root = i,
        }
    }
    let mut order = Vec::with_capacity(n);
    let mut stack = vec![(root, false)];
    while let Some((node, expanded)) = stack.pop() {
        if expanded {
            order.push(node);
        } else {
            stack.push((node, true));
            for &ch in &children[node] {
                stack.push((ch, false));
            }
        }
    }
    order
}

/// Cholesky `A = L Lᵀ` for a small SPD `n×n` matrix (row-major) into `out` (lower
/// `L`). Only the lower triangle + diagonal of `out` is written (and only those are
/// read by the solves), so `out` may be reused scratch with stale upper entries.
///
/// # Panics
/// If `A` is not positive definite (a non-positive pivot appears).
fn cholesky_lower_into(a: &[f32], n: usize, out: &mut [f32]) {
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= out[i * n + k] * out[j * n + k];
            }
            if i == j {
                assert!(sum > 0.0, "frontal not positive definite (pivot {sum})");
                out[i * n + j] = sum.sqrt();
            } else {
                out[i * n + j] = sum / out[j * n + j];
            }
        }
    }
}

/// Solve `L z = b` (lower-triangular forward substitution) into `out`.
fn forward_solve_into(l: &[f32], b: &[f32], n: usize, out: &mut [f32]) {
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * out[k];
        }
        out[i] = s / l[i * n + i];
    }
}

/// Solve `Lᵀ x = z` (upper-triangular back substitution on `Lᵀ`) into `out`.
fn back_solve_into(l: &[f32], z: &[f32], n: usize, out: &mut [f32]) {
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in i + 1..n {
            s -= l[k * n + i] * out[k];
        }
        out[i] = s / l[i * n + i];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut xs = seed;
        move || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 32) as u32 as f32) / 4294967296.0 - 0.5
        }
    }

    /// Dense reference: assemble J from the clique frontals (+ ridge on residual
    /// diagonals) and solve J x = b by dense Gaussian elimination.
    fn dense_reference(jt: &JunctionTreeCholesky) -> Vec<f32> {
        let d = jt.d;
        let mut j = vec![0.0f32; d * d];
        let mut b = vec![0.0f32; d];
        for (c, cl) in jt.cliques.iter().enumerate() {
            let (m, fo, bo) = (jt.m[c], jt.foff[c], jt.boff[c]);
            for i in 0..m {
                b[cl.vars[i]] += jt.acc_rhs[bo + i];
                for k in 0..m {
                    j[cl.vars[i] * d + cl.vars[k]] += jt.acc_front[fo + i * m + k];
                }
            }
            for r in 0..cl.n_res {
                j[cl.vars[r] * d + cl.vars[r]] += jt.ridge;
            }
        }
        gauss_solve(&mut j, &mut b, d)
    }

    fn gauss_solve(a: &mut [f32], b: &mut [f32], d: usize) -> Vec<f32> {
        for col in 0..d {
            let mut piv = col;
            for r in col + 1..d {
                if a[r * d + col].abs() > a[piv * d + col].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for c in 0..d {
                    a.swap(col * d + c, piv * d + c);
                }
                b.swap(col, piv);
            }
            let diag = a[col * d + col];
            for r in 0..d {
                if r == col {
                    continue;
                }
                let f = a[r * d + col] / diag;
                if f != 0.0 {
                    for c in col..d {
                        a[r * d + c] -= f * a[col * d + c];
                    }
                    b[r] -= f * b[col];
                }
            }
        }
        (0..d).map(|i| b[i] / a[i * d + i]).collect()
    }

    #[test]
    fn single_clique_equals_dense() {
        let (mut nx, d) = (lcg(1), 5usize);
        let cliques = vec![Clique {
            vars: (0..d).collect(),
            n_res: d,
            parent: None,
        }];
        let mut jt = JunctionTreeCholesky::new(cliques, 1.0, d);
        for _ in 0..40 {
            let phi: Vec<f32> = (0..d).map(|_| nx()).collect();
            jt.update(0, &phi, nx());
        }
        let want = dense_reference(&jt);
        let got = jt.solve();
        for i in 0..d {
            assert!(
                (want[i] - got[i]).abs() < 1e-4,
                "x[{i}] dense {} vs jt {}",
                want[i],
                got[i]
            );
        }
        // with a single clique there are no separators to drop, so the
        // block-diagonal baseline must coincide with the exact solve
        let blk = jt.solve_block_diagonal();
        for i in 0..d {
            assert!(
                (got[i] - blk[i]).abs() < 1e-4,
                "block[{i}] {} vs solve {}",
                blk[i],
                got[i]
            );
        }
    }

    #[test]
    fn branching_tree_equals_dense_at_bounded_width() {
        let (cliques, d) = balanced_binary_tree(4, 2, 3); // depth 4 binary tree, sep=2, res=3
        let mut nx = lcg(7);
        let mut jt = JunctionTreeCholesky::new(cliques, 1.0, d);
        // home each measurement at a random clique, local support
        let nc = jt.n_cliques();
        for _ in 0..(60 * nc) {
            let c = ((nx() + 0.5) * nc as f32) as usize % nc;
            let m = jt.cliques[c].vars.len();
            let phi: Vec<f32> = (0..m).map(|_| nx()).collect();
            jt.update(c, &phi, nx());
        }
        let want = dense_reference(&jt);
        let got = jt.solve();
        let mut max_err = 0.0f32;
        for i in 0..d {
            max_err = max_err.max((want[i] - got[i]).abs());
        }
        assert!(max_err < 1e-3, "max |dense - jt| = {max_err}");
        // bounded-width storage: frontals << dense d^2
        assert!(
            jt.factor_storage() < d * d,
            "frontal storage {} !< d^2 {}",
            jt.factor_storage(),
            d * d
        );
    }

    /// An online update homed at a leaf clique changes only the Cholesky factors
    /// on that leaf's path to the root; every off-path subtree is untouched. This
    /// is the locality that makes an incremental re-fire O(depth·w³), not O(N·w³).
    #[test]
    fn star_tree_equals_dense_with_shared_separator() {
        // one shared separator across many children — exercises the assembly of
        // multiple Schur messages into the SAME parent positions
        let (cliques, d) = star_clique_tree(6, 3, 3, 3);
        let mut nx = lcg(5);
        let mut jt = JunctionTreeCholesky::new(cliques, 1.0, d);
        let nc = jt.n_cliques();
        for _ in 0..(50 * nc) {
            let c = ((nx() + 0.5) * nc as f32) as usize % nc;
            let m = jt.cliques[c].vars.len();
            let phi: Vec<f32> = (0..m).map(|_| nx()).collect();
            jt.update(c, &phi, nx());
        }
        let want = dense_reference(&jt);
        let got = jt.solve();
        let max_err = (0..d)
            .map(|i| (want[i] - got[i]).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 1e-3, "star max |dense - jt| = {max_err}");
    }

    #[test]
    fn online_update_only_perturbs_path_to_root() {
        let (cliques, _d) = balanced_binary_tree(4, 2, 3);
        let mut nx = lcg(3);
        let mut jt = JunctionTreeCholesky::new(cliques, 1.0, _d);
        let nc = jt.n_cliques();
        for _ in 0..(40 * nc) {
            let c = ((nx() + 0.5) * nc as f32) as usize % nc;
            let m = jt.cliques[c].vars.len();
            let phi: Vec<f32> = (0..m).map(|_| nx()).collect();
            jt.update(c, &phi, nx());
        }
        let _ = jt.solve();
        let before: Vec<f64> = (0..nc).map(|c| jt.factor_checksum(c)).collect();

        // pick a genuine leaf (a clique that is nobody's parent)
        let is_parent: Vec<bool> = {
            let mut v = vec![false; nc];
            for cl in &jt.cliques {
                if let Some(p) = cl.parent {
                    v[p] = true;
                }
            }
            v
        };
        let leaf = (0..nc).rev().find(|&c| !is_parent[c]).unwrap();
        let path: std::collections::HashSet<usize> =
            jt.ancestors_inclusive(leaf).into_iter().collect();

        let m = jt.cliques[leaf].vars.len();
        let phi: Vec<f32> = (0..m).map(|_| nx()).collect();
        jt.update(leaf, &phi, nx());
        let _ = jt.solve();
        let after: Vec<f64> = (0..nc).map(|c| jt.factor_checksum(c)).collect();

        for c in 0..nc {
            let changed = (before[c] - after[c]).abs() > 1e-9;
            assert_eq!(
                changed,
                path.contains(&c),
                "clique {c} changed={changed} but on_path={}",
                path.contains(&c)
            );
        }
        // the path is strictly shorter than the whole tree (locality is real)
        assert!(path.len() < nc, "path {} !< n_cliques {nc}", path.len());
    }
}
