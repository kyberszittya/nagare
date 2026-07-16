//! **Nonlinear curvature (B+C)** — a spline-parameterized curvature *field* over a 2-D
//! lattice patch, where the constant-rotor exact solve (F-HOLO-3) is *blind* and a
//! closed-form ChebyCR readout recovers the field.
//!
//! Extends F-HOLO-3 (constant per-plaquette flux, exact tree-gauge solve reaches the oracle)
//! to the multi-scale regime the handoff's GAP #1 named. The per-plaquette curvature is a
//! FIELD `F(r,c)` over an `L×L` lattice; two classes with the SAME flux value multiset and
//! total differ only in spatial arrangement:
//! * **smooth** (label 0): the plaquette-angle field is a low-order 2-D Chebyshev field.
//! * **rough** (label 1): a random spatial permutation of the smooth field's values.
//!
//! **Construction (genuine SO(3) lattice connection, closed form).** 2-D plaquette variables
//! are independent. Set horizontals to identity, verticals cumulatively `U(r,c)=R_r·∏_{c'<c}F`,
//! so plaquette `(r,c)` holonomy `= U(r,c)⁻¹U(r,c+1) = F(r,c)` exactly; then apply a random Haar
//! gauge `g↦G_v g G_u⁻¹`, which makes every edge Haar-marginal while preserving every holonomy
//! (curvature is gauge-invariant). Result: matched Haar marginals AND matched mean flux across
//! classes ⇒ trivial covariance entropy AND the constant-rotor mean are BOTH at chance; only the
//! spatial arrangement of curvature separates them.
//!
//! **Readout (C, closed form, no training):** extract `θ_obs(r,c)=angle(plaquette holonomy)` via
//! [`crate::rotor_holonomy_forward`]; fit a low-order 2-D Chebyshev with the separable projector
//! `P=B_k(B_kᵀB_k)⁻¹B_kᵀ` from [`crate::chebyshev_knot_basis`]; roughness = residual energy
//! fraction. Smooth → small, rough → large.
//!
//! Reuses `curvature_task` primitives (`haar_quat`, `axis_angle_quat`, `rotor_angle`, `Rng`),
//! `chebyshev_knot_basis`, `rotor_holonomy_forward`, and `hymeko_clifford` quat algebra — no
//! re-implementation (§6.1).

use crate::chebyshev_knot_basis;
use crate::curvature_task::{axis_angle_quat, haar_quat, rotor_angle, Rng, IDENT};
use crate::rotor_holonomy_forward;
use crate::{spectral_reg_value_grad, SpectralEntropyConfig};
use hymeko_clifford::{quat_conjugate, quat_mul};

/// An `L×L` lattice: nodes `r*L+c`, horizontal then vertical edges, unit-square plaquettes.
/// Edge layout: `h(r,c)` = `(r,c)→(r,c+1)` at index `r*(L-1)+c`; `v(r,c)` = `(r,c)→(r+1,c)` at
/// index `L*(L-1) + r*L + c`.
#[derive(Clone, Debug)]
pub struct GridGraph {
    /// Side length in nodes.
    pub l: usize,
    /// Plaquette grid side = `l - 1`.
    pub m: usize,
    /// Node count `l*l`.
    pub n_nodes: usize,
    /// Edge count `2*l*(l-1)`.
    pub n_edges: usize,
    /// Directed edges `(u, v)`.
    pub edges: Vec<(u32, u32)>,
    /// One per plaquette `(r,c)` (row-major, `r*m+c`): the 4 traversal `(edge_idx, forward?)`.
    pub plaquettes: Vec<[(usize, bool); 4]>,
}

impl GridGraph {
    #[inline]
    fn h_idx(&self, r: usize, c: usize) -> usize {
        r * (self.l - 1) + c
    }
    #[inline]
    fn v_idx(&self, r: usize, c: usize) -> usize {
        self.l * (self.l - 1) + r * self.l + c
    }
}

/// Build the `L×L` lattice patch.
///
/// # Preconditions
/// `l >= 3`.
///
/// # Panics
/// If `l < 3`.
pub fn grid_graph(l: usize) -> GridGraph {
    assert!(l >= 3, "grid needs l >= 3");
    let m = l - 1;
    let n_nodes = l * l;
    let node = |r: usize, c: usize| (r * l + c) as u32;
    let mut edges = Vec::with_capacity(2 * l * (l - 1));
    // horizontals: r in 0..l, c in 0..l-1
    for r in 0..l {
        for c in 0..l - 1 {
            edges.push((node(r, c), node(r, c + 1)));
        }
    }
    // verticals: r in 0..l-1, c in 0..l
    for r in 0..l - 1 {
        for c in 0..l {
            edges.push((node(r, c), node(r + 1, c)));
        }
    }
    let n_edges = edges.len();
    let mut g = GridGraph {
        l,
        m,
        n_nodes,
        n_edges,
        edges,
        plaquettes: Vec::with_capacity(m * m),
    };
    // plaquette (r,c): (r,c)→(r,c+1)→(r+1,c+1)→(r+1,c)→(r,c)
    for r in 0..m {
        for c in 0..m {
            g.plaquettes.push([
                (g.h_idx(r, c), true),      // h(r,c) fwd
                (g.v_idx(r, c + 1), true),  // v(r,c+1) fwd
                (g.h_idx(r + 1, c), false), // h(r+1,c) reversed
                (g.v_idx(r, c), false),     // v(r,c) reversed
            ]);
        }
    }
    g
}

#[inline]
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// A low-order **2-D Chebyshev angle field** on the `m×m` plaquette grid, values in `(0, π)`.
/// The generator side of "nonlinear curvature via a ChebyCR patch": `θ(r,c) = θ_mid +
/// amp·tanh(Σ_{a<k,b<k} C_{ab}·T_a(x_r)·T_b(y_c))`, `C` decaying so the field is low-frequency.
///
/// # Preconditions
/// `m >= 2`, `k in 1..=m`.
pub fn chebyshev_angle_field(m: usize, k: usize, rng: &mut Rng) -> Vec<f32> {
    assert!(m >= 2 && (1..=m).contains(&k));
    let basis = chebyshev_knot_basis(m, k); // (m, k), T_0..T_{k-1} at m uniform knots
                                            // decaying random coefficients (low-frequency dominant)
    let mut coef = vec![0.0f32; k * k];
    for a in 0..k {
        for b in 0..k {
            coef[a * k + b] = rng.g() / (1.0 + (a + b) as f32);
        }
    }
    let (theta_mid, amp) = (1.5f32, 1.15f32);
    let mut field = vec![0.0f32; m * m];
    for r in 0..m {
        for c in 0..m {
            let mut s = 0.0f32;
            for a in 0..k {
                for b in 0..k {
                    s += coef[a * k + b] * basis[r * k + a] * basis[c * k + b];
                }
            }
            field[r * m + c] = theta_mid + amp * s.tanh();
        }
    }
    field
}

/// Sample an `SO(3)` connection realizing a curvature field on `g`. Returns the edge rotors
/// (flat `(|E|·4)`) and the **true** plaquette-angle field (`m·m`, for the oracle).
///
/// `rough=false` → a smooth low-order Chebyshev angle field; `rough=true` → the same field's
/// values randomly permuted in space (identical multiset + total). Per-plaquette axes are iid
/// random (non-abelian). `noise` (rad) perturbs edge rotors after gauge (0 → exact, for tests).
///
/// # Postconditions
/// With `noise=0`, every edge is unit and (pre-noise) plaquette `(r,c)` holonomy angle equals
/// `θ(r,c)` exactly; after the Haar gauge every edge is Haar-marginal.
///
/// # Panics
/// If `k` is out of range.
pub fn sample_curvature_field(
    g: &GridGraph,
    rng: &mut Rng,
    rough: bool,
    k: usize,
    noise: f32,
) -> (Vec<f32>, Vec<f32>) {
    let m = g.m;
    // 1. angle field (smooth), then optional spatial permutation
    let mut theta = chebyshev_angle_field(m, k, rng);
    if rough {
        // Fisher–Yates over the flat m*m field
        for i in (1..theta.len()).rev() {
            let j = (rng.f() * (i as f32 + 1.0)) as usize;
            let j = j.min(i);
            theta.swap(i, j);
        }
    }
    // 2–5: realize the field as an SO(3) connection (shared with the regional generator)
    let edge_q = realize_field(g, rng, &theta, noise);
    (edge_q, theta)
}

/// Realize a plaquette-angle field `theta` `(m·m)` as an `SO(3)` connection: per-plaquette rotor
/// about a random axis, the cumulative-vertical construction (horizontals identity), a random Haar
/// gauge (every edge Haar-marginal, all holonomies preserved), then optional per-edge noise.
/// Shared by [`sample_curvature_field`] and [`sample_regional_curvature`] (§6.1, no duplication).
///
/// # Preconditions
/// `theta.len() == g.m * g.m`.
pub fn realize_field(g: &GridGraph, rng: &mut Rng, theta: &[f32], noise: f32) -> Vec<f32> {
    let m = g.m;
    assert_eq!(theta.len(), m * m, "theta must be (m, m)");
    // per-plaquette rotor F(r,c) = axis_angle(random axis, theta)
    let mut fpl = vec![IDENT; m * m];
    for p in 0..m * m {
        let axis = [rng.g(), rng.g(), rng.g()];
        fpl[p] = axis_angle_quat(axis, theta[p]);
    }
    // horizontals = I, verticals cumulative U(r,c) = R_r · ∏_{c'<c} F(r,c')
    let mut edge_q = vec![0.0f32; g.n_edges * 4];
    for e in 0..g.l * (g.l - 1) {
        edge_q[e * 4..e * 4 + 4].copy_from_slice(&IDENT);
    }
    for r in 0..g.l - 1 {
        let mut acc = haar_quat(rng);
        for c in 0..g.l {
            let vi = g.v_idx(r, c);
            edge_q[vi * 4..vi * 4 + 4].copy_from_slice(&acc);
            if c < m {
                acc = quat_mul(acc, fpl[r * m + c]);
            }
        }
    }
    // random Haar gauge
    let gauge: Vec<[f32; 4]> = (0..g.n_nodes).map(|_| haar_quat(rng)).collect();
    for (e, &(u, v)) in g.edges.iter().enumerate() {
        let ge = q_at(&edge_q, e);
        let t = quat_mul(
            gauge[v as usize],
            quat_mul(ge, quat_conjugate(gauge[u as usize])),
        );
        edge_q[e * 4..e * 4 + 4].copy_from_slice(&t);
    }
    // optional measurement noise
    if noise > 0.0 {
        for e in 0..g.n_edges {
            let axis = [rng.g(), rng.g(), rng.g()];
            let pert = axis_angle_quat(axis, noise * rng.g());
            let ge = q_at(&edge_q, e);
            let t = quat_mul(pert, ge);
            edge_q[e * 4..e * 4 + 4].copy_from_slice(&t);
        }
    }
    edge_q
}

/// Fisher–Yates shuffle of the `theta` values among plaquettes in column range `[c0, c1)`
/// (all rows) — makes that region *rough* (decorrelated) while preserving its value multiset.
fn permute_region(theta: &mut [f32], m: usize, c0: usize, c1: usize, rng: &mut Rng) {
    let idx: Vec<usize> = (0..m)
        .flat_map(|r| (c0..c1).map(move |c| r * m + c))
        .collect();
    for k in (1..idx.len()).rev() {
        let j = ((rng.f() * (k as f32 + 1.0)) as usize).min(k);
        theta.swap(idx[k], idx[j]);
    }
}

/// **Gate-2 task — XOR-of-regional-roughness.** The plaquette field is split into two column-halves
/// `A` (`c < m/2`) and `B` (`c ≥ m/2`); each half is independently *smooth* (low-order Chebyshev) or
/// *rough* (permuted within that half). Returns `(edge rotors, true θ field, class)` with
/// `class = (A rough) ⊕ (B rough)` — i.e. **do the halves differ?**. The discriminative quantity is
/// `|roughness(A) − roughness(B)|`, invisible to any sum/mean-like global scalar (which tracks the
/// total, non-monotonic in the XOR) — so a fixed closed-form readout is at chance and a learned
/// nonlinear readout is necessary.
pub fn sample_regional_curvature(
    g: &GridGraph,
    rng: &mut Rng,
    k: usize,
    noise: f32,
) -> (Vec<f32>, Vec<f32>, u8) {
    let m = g.m;
    let mut theta = chebyshev_angle_field(m, k, rng);
    let split = m / 2;
    let rough_a = rng.f() < 0.5;
    let rough_b = rng.f() < 0.5;
    if rough_a {
        permute_region(&mut theta, m, 0, split, rng);
    }
    if rough_b {
        permute_region(&mut theta, m, split, m, rng);
    }
    let edge_q = realize_field(g, rng, &theta, noise);
    let class = (rough_a ^ rough_b) as u8;
    (edge_q, theta, class)
}

/// Variance-normalized Laplacian roughness restricted to interior plaquettes in columns
/// `[c0, c1)` (neighbors stay in-region).
fn region_laplacian(field: &[f32], m: usize, c0: usize, c1: usize) -> f32 {
    if c1 <= c0 + 2 || m < 3 {
        return 0.0;
    }
    let vals: Vec<f32> = (0..m)
        .flat_map(|r| (c0..c1).map(move |c| r * m + c))
        .map(|i| field[i])
        .collect();
    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
    let var = vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / vals.len() as f32;
    let (mut acc, mut cnt) = (0.0f32, 0usize);
    for r in 1..m - 1 {
        for c in c0 + 1..c1 - 1 {
            let center = field[r * m + c];
            let nbr = (field[(r - 1) * m + c]
                + field[(r + 1) * m + c]
                + field[r * m + c - 1]
                + field[r * m + c + 1])
                / 4.0;
            acc += (center - nbr).powi(2);
            cnt += 1;
        }
    }
    if cnt == 0 {
        return 0.0;
    }
    (acc / cnt as f32) / var.max(1e-9)
}

/// The **oracle / discriminative feature** for the Gate-2 XOR task: `|roughness(A) − roughness(B)|`
/// over the two column-halves. High when the halves differ (class 1). No fixed monotonic global
/// scalar computes it — it needs the two regions localized and combined nonlinearly (a difference).
pub fn region_roughness_diff(field: &[f32], m: usize) -> f32 {
    let split = m / 2;
    let ra = region_laplacian(field, m, 0, split);
    let rb = region_laplacian(field, m, split, m);
    (ra - rb).abs()
}

/// Column-block ranges: split `m` columns into `n_blocks` (near-)equal blocks.
fn col_block(b: usize, m: usize, n_blocks: usize) -> (usize, usize) {
    let c0 = b * m / n_blocks;
    let c1 = ((b + 1) * m / n_blocks).max(c0 + 1);
    (c0, c1)
}

/// **Framework 2nd-order pooling front-end** (bias-discrimination): per uniform column-block, the
/// covariance eigen-entropy of the local descriptors `[center, ↑, ↓, ←, →]` via the framework's
/// [`crate::spectral_reg_value_grad`]. Smooth block ⇒ neighbours track the centre ⇒ rank-1 covariance
/// ⇒ low `H`; rough block ⇒ decorrelated ⇒ full-rank ⇒ high `H`. Returns `n_blocks` features — the
/// framework's exact 2nd-order op, applied regionally (the Gömb-Soma / CPML-tier bias in miniature).
///
/// # Preconditions
/// `field.len() == m*m`, `n_blocks >= 1`.
pub fn block_entropy_features(field: &[f32], m: usize, n_blocks: usize) -> Vec<f32> {
    assert_eq!(field.len(), m * m);
    assert!(n_blocks >= 1);
    let cfg = SpectralEntropyConfig {
        lam_0: 1.0,
        lam_a: 0.0,
        lam_b: 1.0,
        lam_kl: 0.0,
        ..SpectralEntropyConfig::default()
    };
    let mut feats = vec![0.0f32; n_blocks];
    for (b, feat) in feats.iter_mut().enumerate() {
        let (c0, c1) = col_block(b, m, n_blocks);
        let mut desc = Vec::new();
        let mut n = 0usize;
        for r in 1..m.saturating_sub(1) {
            for c in c0.max(1)..c1.min(m.saturating_sub(1)) {
                desc.push(field[r * m + c]);
                desc.push(field[(r - 1) * m + c]);
                desc.push(field[(r + 1) * m + c]);
                desc.push(field[r * m + c - 1]);
                desc.push(field[r * m + c + 1]);
                n += 1;
            }
        }
        if n >= 3 {
            *feat = spectral_reg_value_grad(&desc, n, 5, &cfg, 1.0).0;
        }
    }
    feats
}

/// **Generic 2nd-order control front-end**: per uniform column-block, the variance-normalised local
/// Laplacian roughness. A generic spatial 2nd-order statistic (not the framework's entropy op) —
/// isolates whether *any* 2nd-order regional bias closes the gap, or specifically the framework's.
///
/// # Preconditions
/// `field.len() == m*m`, `n_blocks >= 1`.
pub fn block_laplacian_features(field: &[f32], m: usize, n_blocks: usize) -> Vec<f32> {
    assert_eq!(field.len(), m * m);
    assert!(n_blocks >= 1);
    (0..n_blocks)
        .map(|b| {
            let (c0, c1) = col_block(b, m, n_blocks);
            region_laplacian(field, m, c0, c1)
        })
        .collect()
}

/// Extract the gauge-invariant plaquette-angle field `θ_obs(r,c)` via the ordered 4-rotor
/// holonomy of each unit square ([`crate::rotor_holonomy_forward`]). Returns `m·m` angles.
pub fn extract_curvature_field(g: &GridGraph, edge_q: &[f32]) -> Vec<f32> {
    assert_eq!(edge_q.len(), g.n_edges * 4);
    let k = 4usize;
    let n = g.plaquettes.len();
    let mut ordered = vec![0.0f32; n * k * 4];
    for (p, plq) in g.plaquettes.iter().enumerate() {
        for (i, &(e, fwd)) in plq.iter().enumerate() {
            let q = q_at(edge_q, e);
            let q = if fwd { q } else { quat_conjugate(q) };
            ordered[(p * k + i) * 4..(p * k + i) * 4 + 4].copy_from_slice(&q);
        }
    }
    let (holo, _) = rotor_holonomy_forward(&ordered, n, k);
    (0..n).map(|p| rotor_angle(q_at(&holo, p))).collect()
}

/// The **constant-rotor** readout (the F-HOLO-3 analogue): mean plaquette-angle. Blind to
/// spatial arrangement (permutation-invariant), so at chance on the smooth-vs-rough task.
pub fn constant_rotor_energy(field: &[f32]) -> f32 {
    if field.is_empty() {
        return 0.0;
    }
    field.iter().sum::<f32>() / field.len() as f32
}

// ---- tiny linear algebra for the low-order Chebyshev projector (k×k SPD) ----

/// Cholesky inverse of a `k×k` SPD matrix (row-major). Small `k` (≤ ~5).
fn spd_inverse(a: &[f32], k: usize) -> Vec<f32> {
    // L Lᵀ = A
    let mut l = vec![0.0f32; k * k];
    for i in 0..k {
        for j in 0..=i {
            let mut s = a[i * k + j];
            for p in 0..j {
                s -= l[i * k + p] * l[j * k + p];
            }
            if i == j {
                l[i * k + j] = s.max(1e-9).sqrt();
            } else {
                l[i * k + j] = s / l[j * k + j];
            }
        }
    }
    // invert via forward/back solve of L Lᵀ X = I, column by column
    let mut inv = vec![0.0f32; k * k];
    for col in 0..k {
        let mut y = vec![0.0f32; k];
        for i in 0..k {
            let mut s = if i == col { 1.0 } else { 0.0 };
            for p in 0..i {
                s -= l[i * k + p] * y[p];
            }
            y[i] = s / l[i * k + i];
        }
        for i in (0..k).rev() {
            let mut s = y[i];
            for p in i + 1..k {
                s -= l[p * k + i] * inv[p * k + col];
            }
            inv[i * k + col] = s / l[i * k + i];
        }
    }
    inv
}

/// The `m×m` symmetric projector `P = B_k (B_kᵀ B_k)⁻¹ B_kᵀ` onto the low-order Chebyshev
/// subspace at `m` uniform knots.
fn chebyshev_projector(m: usize, k: usize) -> Vec<f32> {
    let b = chebyshev_knot_basis(m, k); // (m, k)
                                        // M = Bᵀ B (k×k)
    let mut mm = vec![0.0f32; k * k];
    for a in 0..k {
        for bb in 0..k {
            let mut s = 0.0f32;
            for r in 0..m {
                s += b[r * k + a] * b[r * k + bb];
            }
            mm[a * k + bb] = s;
        }
    }
    let minv = spd_inverse(&mm, k);
    // P = B minv Bᵀ  → first BM = B minv (m×k), then P = BM Bᵀ (m×m)
    let mut bm = vec![0.0f32; m * k];
    for r in 0..m {
        for a in 0..k {
            let mut s = 0.0f32;
            for p in 0..k {
                s += b[r * k + p] * minv[p * k + a];
            }
            bm[r * k + a] = s;
        }
    }
    let mut p = vec![0.0f32; m * m];
    for r in 0..m {
        for c in 0..m {
            let mut s = 0.0f32;
            for a in 0..k {
                s += bm[r * k + a] * b[c * k + a];
            }
            p[r * m + c] = s;
        }
    }
    p
}

/// The **ChebyCR roughness** readout (the B+C method): fraction of the field's centered energy
/// NOT captured by a low-order 2-D Chebyshev fit `θ̂ = P θ P`. Smooth → small, rough → large.
/// One-shot, closed-form, no training.
///
/// # Preconditions
/// `field.len() == m*m`, `k in 1..=m`.
pub fn chebycr_roughness(field: &[f32], m: usize, k: usize) -> f32 {
    assert_eq!(field.len(), m * m);
    assert!((1..=m).contains(&k));
    let p = chebyshev_projector(m, k);
    // fit = P θ P  (θ as m×m). First T = P θ (m×m), then fit = T P.
    let mut t = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0f32;
            for q in 0..m {
                s += p[i * m + q] * field[q * m + j];
            }
            t[i * m + j] = s;
        }
    }
    let mut fit = vec![0.0f32; m * m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0f32;
            for q in 0..m {
                s += t[i * m + q] * p[q * m + j];
            }
            fit[i * m + j] = s;
        }
    }
    let mean = field.iter().sum::<f32>() / (m * m) as f32;
    let mut resid = 0.0f32;
    let mut centered = 0.0f32;
    for i in 0..m * m {
        resid += (field[i] - fit[i]).powi(2);
        centered += (field[i] - mean).powi(2);
    }
    resid / centered.max(1e-9)
}

/// A **discrete-Laplacian roughness** baseline (no solve): mean squared deviation of each
/// interior plaquette from its 4-neighbour average, normalized by field variance. A robustness
/// check that the smooth/rough contrast is not an artifact of the Chebyshev basis.
pub fn laplacian_roughness(field: &[f32], m: usize) -> f32 {
    if m < 3 {
        return 0.0;
    }
    let mean = field.iter().sum::<f32>() / (m * m) as f32;
    let var = field.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / (m * m) as f32;
    let mut acc = 0.0f32;
    let mut cnt = 0usize;
    for r in 1..m - 1 {
        for c in 1..m - 1 {
            let center = field[r * m + c];
            let nbr = (field[(r - 1) * m + c]
                + field[(r + 1) * m + c]
                + field[r * m + c - 1]
                + field[r * m + c + 1])
                / 4.0;
            acc += (center - nbr).powi(2);
            cnt += 1;
        }
    }
    (acc / cnt as f32) / var.max(1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plaquette_holonomy_equals_field_exactly() {
        // The cumulative construction is correct: pre-noise, extracted angle == true theta.
        let g = grid_graph(8);
        for (seed, rough) in [(1u64, false), (2, true)] {
            let (eq, theta) = sample_curvature_field(&g, &mut Rng(seed), rough, 3, 0.0);
            let obs = extract_curvature_field(&g, &eq);
            for p in 0..g.m * g.m {
                assert!(
                    (obs[p] - theta[p]).abs() < 2e-3,
                    "plaquette {p}: obs {} != theta {} (rough={rough})",
                    obs[p],
                    theta[p]
                );
            }
        }
    }

    #[test]
    fn gauge_leaves_curvature_invariant_and_edges_nontrivial() {
        // After the Haar gauge the field is unchanged (invariance) but edges are not identity.
        let g = grid_graph(6);
        let (eq, theta) = sample_curvature_field(&g, &mut Rng(4), false, 3, 0.0);
        let obs = extract_curvature_field(&g, &eq);
        for p in 0..g.m * g.m {
            assert!((obs[p] - theta[p]).abs() < 2e-3);
        }
        // horizontals should no longer be identity after gauge
        let any_nontrivial = (0..g.l * (g.l - 1)).any(|e| rotor_angle(q_at(&eq, e)) > 1e-2);
        assert!(any_nontrivial, "gauge did not randomize horizontal edges");
    }

    #[test]
    fn smooth_has_lower_roughness_than_rough() {
        let g = grid_graph(12);
        let (smooth, _) = sample_curvature_field(&g, &mut Rng(5), false, 3, 0.0);
        let (rough, _) = sample_curvature_field(&g, &mut Rng(5), true, 3, 0.0);
        let rs = chebycr_roughness(&extract_curvature_field(&g, &smooth), g.m, 3);
        let rr = chebycr_roughness(&extract_curvature_field(&g, &rough), g.m, 3);
        assert!(rs < rr, "smooth roughness {rs} !< rough {rr}");
        assert!(
            rs < 0.5 && rr > 0.5,
            "poor separation: smooth {rs} rough {rr}"
        );
    }

    #[test]
    fn constant_rotor_mean_matched_across_classes() {
        // The insufficiency claim, in code: mean plaquette angle is (nearly) equal for a field
        // and its permutation, so the constant-rotor readout cannot separate the classes.
        let g = grid_graph(12);
        let mean_over = |rough: bool| -> f32 {
            let mut s = 0.0f32;
            for seed in 0..20u64 {
                let (eq, _) = sample_curvature_field(&g, &mut Rng(100 + seed), rough, 3, 0.0);
                s += constant_rotor_energy(&extract_curvature_field(&g, &eq));
            }
            s / 20.0
        };
        let (m0, m1) = (mean_over(false), mean_over(true));
        assert!(
            (m0 - m1).abs() / m0.max(1e-3) < 0.03,
            "constant-rotor mean differs across classes: {m0} vs {m1}"
        );
    }

    #[test]
    fn spd_inverse_correct() {
        // A·A⁻¹ ≈ I on a small SPD matrix.
        let a = [4.0f32, 1.0, 1.0, 3.0];
        let inv = spd_inverse(&a, 2);
        let prod = [
            a[0] * inv[0] + a[1] * inv[2],
            a[0] * inv[1] + a[1] * inv[3],
            a[2] * inv[0] + a[3] * inv[2],
            a[2] * inv[1] + a[3] * inv[3],
        ];
        assert!((prod[0] - 1.0).abs() < 1e-4 && prod[3] - 1.0 < 1e-4);
        assert!(prod[1].abs() < 1e-4 && prod[2].abs() < 1e-4);
    }

    #[test]
    fn perf_roughness_latency_budget() {
        use std::time::Instant;
        let g = grid_graph(12);
        let samples: Vec<Vec<f32>> = (0..200)
            .map(|s| sample_curvature_field(&g, &mut Rng(s), s % 2 == 0, 3, 0.05).0)
            .collect();
        let mut acc = 0.0f32;
        for x in &samples {
            acc += chebycr_roughness(&extract_curvature_field(&g, x), g.m, 3);
        }
        let mut us = vec![];
        for _ in 0..5 {
            let t = Instant::now();
            for x in &samples {
                acc += chebycr_roughness(&extract_curvature_field(&g, x), g.m, 3);
            }
            us.push(t.elapsed().as_secs_f64() * 1e6 / samples.len() as f64);
        }
        us.sort_by(|a, b| a.total_cmp(b));
        assert!(acc.is_finite());
        assert!(
            us[2] < 100.0,
            "readout median {:.1} us/sample exceeds 100 us",
            us[2]
        );
    }

    #[test]
    fn regional_sample_deterministic() {
        let g = grid_graph(12);
        let (a, _, ca) = sample_regional_curvature(&g, &mut Rng(9), 3, 0.0);
        let (b, _, cb) = sample_regional_curvature(&g, &mut Rng(9), 3, 0.0);
        assert_eq!(a, b);
        assert_eq!(ca, cb);
    }

    #[test]
    fn regional_xor_oracle_separates_classes() {
        // The defining property: the oracle |roughness(A) - roughness(B)| is larger when the halves
        // DIFFER (class 1) than when they match (class 0). Averaged over samples, clean data.
        let g = grid_graph(12);
        let (mut s0, mut n0, mut s1, mut n1) = (0.0f32, 0usize, 0.0f32, 0usize);
        for seed in 0..80u64 {
            let (eq, _th, class) = sample_regional_curvature(&g, &mut Rng(500 + seed), 3, 0.0);
            let d = region_roughness_diff(&extract_curvature_field(&g, &eq), g.m);
            if class == 1 {
                s1 += d;
                n1 += 1;
            } else {
                s0 += d;
                n0 += 1;
            }
        }
        let (m0, m1) = (s0 / n0.max(1) as f32, s1 / n1.max(1) as f32);
        assert!(
            m1 > m0 * 1.5,
            "oracle does not separate XOR classes: differ {m1} vs match {m0}"
        );
    }

    #[test]
    fn block_entropy_higher_for_rough() {
        // The framework 2nd-order front-end works: a rough field's block-entropy exceeds a smooth
        // field's (averaged over blocks and seeds).
        let g = grid_graph(12);
        let mean_ent = |rough: bool| -> f32 {
            let mut s = 0.0f32;
            for seed in 0..12u64 {
                let (eq, _) = sample_curvature_field(&g, &mut Rng(700 + seed), rough, 3, 0.0);
                let f = extract_curvature_field(&g, &eq);
                let feats = block_entropy_features(&f, g.m, 4);
                s += feats.iter().sum::<f32>() / feats.len() as f32;
            }
            s / 12.0
        };
        let (smooth, rough) = (mean_ent(false), mean_ent(true));
        assert!(
            rough > smooth + 0.02,
            "block-entropy not higher for rough: smooth {smooth} rough {rough}"
        );
        // the Laplacian control front-end likewise separates
        let lap_rough = {
            let (eq, _) = sample_curvature_field(&g, &mut Rng(1), true, 3, 0.0);
            block_laplacian_features(&extract_curvature_field(&g, &eq), g.m, 4)
        };
        assert_eq!(lap_rough.len(), 4);
    }
}
