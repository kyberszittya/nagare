//! **The curvature-discriminating task** — where trivial entropy is at chance but
//! holonomy succeeds (auto-holonomy Step 1; F-HOLO-2 binding lesson).
//!
//! A graph carries an `SO(3)` connection: each directed edge `u→v` holds a unit
//! quaternion rotor `g` (frame transport `frame_v = g · frame_u`). Two classes with
//! **identical edge marginals** differing *only* in loop-product (holonomy) structure:
//!
//! * **flat** (label 0): node frames `R_i ~ Haar SO(3)`, edge rotor `g_{u→v}=R_v R_u⁻¹`.
//!   Every fundamental-cycle holonomy is the identity — a pure gauge, "un-windable".
//! * **curved** (label 1): the same, then a flux `F_p = exp(½ θ_p n̂_p)` is injected on
//!   the one cotree edge of each plaquette, so that plaquette's holonomy is `F_p ≠ I`.
//!
//! **Why the marginals match (the metric-integrity crux).** A product of independent Haar
//! rotors is Haar; left/right multiplying a Haar rotor by an independent rotor is Haar.
//! So in *both* classes every edge rotor is marginally Haar `SO(3)` — the multiset of edge
//! log-rotors has the same isotropic covariance, and any first/second-order statistic of it
//! (mean, covariance eigen-entropy) is at chance. The classes separate **only** through the
//! ordered loop products. This is the continuous generalization of the `Z₂` balance /
//! frustration task, and the task on which holonomy's value can finally be *measured*.
//!
//! Topology: a **wheel** — hub node `0`, rim nodes `1..=n_rim` in a cycle. The star (spokes)
//! is a spanning tree; the rim edges are the cotree, one per triangular plaquette
//! `{0, i, i+1}`. Fundamental cycles = `n_rim`.
//!
//! Reuses `hymeko_clifford::{quat_mul, quat_conjugate}` — quaternion algebra is never
//! re-implemented here (§6.1). The small angle/log/Haar helpers are plain numerical glue.

use hymeko_clifford::{quat_conjugate, quat_mul};

/// Identity quaternion `(w, x, y, z) = (1, 0, 0, 0)`.
pub const IDENT: [f32; 4] = [1.0, 0.0, 0.0, 0.0];

/// Deterministic LCG with uniform (`f`) and standard-normal (`g`, Box–Muller) draws.
/// Same generator idiom as the crate's examples — reproducible in seed.
pub struct Rng(pub u64);
impl Rng {
    /// Uniform in `[0, 1)`.
    pub fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32) / 4294967296.0
    }
    /// Standard normal.
    pub fn g(&mut self) -> f32 {
        (-2.0 * self.f().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * self.f()).cos()
    }
}

/// A uniform (Haar) `SO(3)` rotor: normalize a 4D Gaussian → uniform point on `S³`,
/// canonicalized to `w ≥ 0` (the rotor double cover; either sign is the same rotation).
///
/// # Postconditions
/// Returns a unit quaternion with `w ≥ 0`.
pub fn haar_quat(rng: &mut Rng) -> [f32; 4] {
    let mut q = [rng.g(), rng.g(), rng.g(), rng.g()];
    let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
        .sqrt()
        .max(1e-12);
    for x in &mut q {
        *x /= n;
    }
    if q[0] < 0.0 {
        for x in &mut q {
            *x = -*x;
        }
    }
    q
}

/// Rotor from axis–angle: `q = (cos(θ/2), sin(θ/2)·n̂)` for a unit axis `n̂`.
///
/// # Preconditions
/// `axis` should be (near) unit; it is renormalized defensively.
pub fn axis_angle_quat(axis: [f32; 3], angle: f32) -> [f32; 4] {
    let n = (axis[0] * axis[0] + axis[1] * axis[1] + axis[2] * axis[2])
        .sqrt()
        .max(1e-12);
    let (s, c) = ((angle * 0.5).sin(), (angle * 0.5).cos());
    [c, s * axis[0] / n, s * axis[1] / n, s * axis[2] / n]
}

/// Rotation angle of a unit rotor, folded to `[0, π]`: `θ = 2·atan2(‖vec‖, |w|)`.
/// Numerically stable near the identity and near `π` (uses `atan2`, not `acos`).
pub fn rotor_angle(q: [f32; 4]) -> f32 {
    let vnorm = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    2.0 * vnorm.atan2(q[0].abs())
}

/// Log-map of a unit rotor to its bivector `θ·n̂` (a 3-vector), canonicalized to `w ≥ 0`.
/// Near the identity the axis is ill-defined; returns the small-angle limit `2·vec`.
pub fn rotor_log(q: [f32; 4]) -> [f32; 3] {
    let q = if q[0] < 0.0 {
        [-q[0], -q[1], -q[2], -q[3]]
    } else {
        q
    };
    let vnorm = (q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
    if vnorm < 1e-6 {
        return [2.0 * q[1], 2.0 * q[2], 2.0 * q[3]];
    }
    let theta = 2.0 * vnorm.atan2(q[0]);
    [
        theta * q[1] / vnorm,
        theta * q[2] / vnorm,
        theta * q[3] / vnorm,
    ]
}

#[inline]
fn q_at(buf: &[f32], i: usize) -> [f32; 4] {
    [buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]
}

/// The wheel connection graph: fixed topology plus the tree / cotree / plaquette structure
/// the estimator and oracle need. Edge layout is `[spokes (n_rim), rim (n_rim)]`.
#[derive(Clone, Debug)]
pub struct ConnGraph {
    /// Total nodes: hub `0` + rim `1..=n_rim`.
    pub n_nodes: usize,
    /// Rim size = number of triangular plaquettes = number of cotree edges.
    pub n_rim: usize,
    /// Directed edges `(u, v)`; a rotor transports `frame_u → frame_v`.
    pub edges: Vec<(u32, u32)>,
    /// Spanning-tree edge indices (the spokes).
    pub tree: Vec<usize>,
    /// Cotree edge indices (the rim edges), one per fundamental cycle.
    pub cotree: Vec<usize>,
    /// Per non-root node in BFS order: `(node, parent_edge_index, forward?)`. `forward`
    /// means the stored edge points parent→node (transport rotor applies directly);
    /// otherwise the edge points node→parent and its inverse (conjugate) transports.
    pub tree_transport: Vec<(usize, usize, bool)>,
    /// Fundamental cycles as ordered `[edge_index; 3]` with traversal directions, for the
    /// oracle (`rotor_holonomy_forward`). Cycle `p` = `0→(1+p) via spoke, rim, →0 via spoke`.
    pub cycles: Vec<[(usize, bool); 3]>,
}

/// Build the wheel graph with `n_rim` triangular plaquettes.
///
/// # Preconditions
/// `n_rim >= 3`.
///
/// # Panics
/// If `n_rim < 3`.
pub fn wheel_graph(n_rim: usize) -> ConnGraph {
    assert!(n_rim >= 3, "wheel needs >= 3 rim nodes");
    let n_nodes = n_rim + 1;
    let mut edges = Vec::with_capacity(2 * n_rim);
    // spokes: edge i = hub(0) -> rim node (1+i)
    for i in 0..n_rim {
        edges.push((0u32, (1 + i) as u32));
    }
    // rim: edge i = rim node (1+i) -> rim node (1 + (i+1)%n_rim)
    for i in 0..n_rim {
        let a = (1 + i) as u32;
        let b = (1 + (i + 1) % n_rim) as u32;
        edges.push((a, b));
    }
    let tree: Vec<usize> = (0..n_rim).collect(); // spokes
    let cotree: Vec<usize> = (n_rim..2 * n_rim).collect(); // rim edges
                                                           // star tree: every rim node's parent is the hub via its spoke (spoke points hub→rim,
                                                           // i.e. parent→node, so forward = true).
    let tree_transport: Vec<(usize, usize, bool)> = (0..n_rim).map(|i| (1 + i, i, true)).collect();
    // fundamental cycle p: spoke_p (0→1+p, fwd), rim_p (fwd), spoke_{(p+1)%n} reversed
    // (1+(p+1)%n → 0, so the stored spoke is used backward).
    let cycles: Vec<[(usize, bool); 3]> = (0..n_rim)
        .map(|p| {
            [
                (p, true),                // spoke_p forward: 0 → 1+p
                (n_rim + p, true),        // rim_p forward: 1+p → 1+(p+1)%n
                ((p + 1) % n_rim, false), // spoke_{(p+1)%n} reversed: 1+(p+1)%n → 0
            ]
        })
        .collect();
    ConnGraph {
        n_nodes,
        n_rim,
        edges,
        tree,
        cotree,
        tree_transport,
        cycles,
    }
}

/// Sample an `SO(3)` connection on `g`. Returns the edge rotors, flat `(|E|·4)`, in the
/// `[spokes, rim]` layout. `curved=false` → a pure gauge (flat); `curved=true` → each rim
/// edge carries an independent flux of angle `∈ [theta_min, π]`.
///
/// # Preconditions
/// `theta_min ∈ [0, π]`.
///
/// # Postconditions
/// Every returned rotor is unit length; the per-edge marginal is Haar `SO(3)` in *both*
/// class settings (see module docs). On `curved=false` every cycle holonomy is the identity.
///
/// # Panics
/// If `theta_min` is out of range.
pub fn sample_connection(g: &ConnGraph, rng: &mut Rng, curved: bool, theta_min: f32) -> Vec<f32> {
    assert!(
        (0.0..=std::f32::consts::PI).contains(&theta_min),
        "theta_min out of range"
    );
    // node frames R_i ~ Haar
    let frames: Vec<[f32; 4]> = (0..g.n_nodes).map(|_| haar_quat(rng)).collect();
    let mut edge_q = vec![0.0f32; g.edges.len() * 4];
    for (e, &(u, v)) in g.edges.iter().enumerate() {
        // base flat rotor g_{u→v} = R_v · R_u⁻¹
        let base = quat_mul(frames[v as usize], quat_conjugate(frames[u as usize]));
        edge_q[e * 4..e * 4 + 4].copy_from_slice(&base);
    }
    if curved {
        // inject flux on each cotree (rim) edge: g := F · g_base (left-mult keeps Haar marginal)
        for &e in &g.cotree {
            let axis = [rng.g(), rng.g(), rng.g()];
            let theta = theta_min + (std::f32::consts::PI - theta_min) * rng.f();
            let flux = axis_angle_quat(axis, theta);
            let base = q_at(&edge_q, e);
            let g_new = quat_mul(flux, base);
            edge_q[e * 4..e * 4 + 4].copy_from_slice(&g_new);
        }
    }
    edge_q
}

/// The `(|E|, 3)` field of edge log-rotors — the input the **trivial** covariance-entropy
/// baseline pools. Isotropic (hence class-blind) in both classes by construction.
pub fn edge_log_field(edge_q: &[f32]) -> Vec<f32> {
    let n = edge_q.len() / 4;
    let mut out = vec![0.0f32; n * 3];
    for e in 0..n {
        let l = rotor_log(q_at(edge_q, e));
        out[e * 3..e * 3 + 3].copy_from_slice(&l);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn haar_is_unit_and_canonical() {
        let mut rng = Rng(1);
        for _ in 0..200 {
            let q = haar_quat(&mut rng);
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((n - 1.0).abs() < 1e-5, "not unit: {n}");
            assert!(q[0] >= 0.0, "not canonicalized");
        }
    }

    #[test]
    fn rotor_angle_and_log_roundtrip() {
        // identity
        assert!(rotor_angle(IDENT) < 1e-6);
        assert!(rotor_log(IDENT).iter().all(|x| x.abs() < 1e-6));
        // known angle about x
        let q = axis_angle_quat([1.0, 0.0, 0.0], 1.2);
        assert!((rotor_angle(q) - 1.2).abs() < 1e-4);
        let l = rotor_log(q);
        assert!((l[0] - 1.2).abs() < 1e-4 && l[1].abs() < 1e-5 && l[2].abs() < 1e-5);
        // near pi (stability)
        let qp = axis_angle_quat([0.0, 1.0, 0.0], 3.10);
        assert!((rotor_angle(qp) - 3.10).abs() < 1e-3);
    }

    #[test]
    fn sample_is_deterministic_and_unit() {
        let g = wheel_graph(8);
        let a = sample_connection(&g, &mut Rng(42), false, 0.5);
        let b = sample_connection(&g, &mut Rng(42), false, 0.5);
        assert_eq!(a, b, "not deterministic in seed");
        for e in 0..a.len() / 4 {
            let q = q_at(&a, e);
            let n = (q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3]).sqrt();
            assert!((n - 1.0).abs() < 1e-4, "edge {e} not unit");
        }
    }

    // helper: wrong arg order guard — `sample_connection(g, rng, ...)`
    fn sample(g: &ConnGraph, rng: &mut Rng, curved: bool, tmin: f32) -> Vec<f32> {
        sample_connection(g, rng, curved, tmin)
    }

    #[test]
    fn flat_sample_has_identity_holonomy() {
        // The defining invariant: a flat connection's every plaquette holonomy is I.
        let g = wheel_graph(10);
        let mut rng = Rng(7);
        let eq = sample(&g, &mut rng, false, 0.5);
        for cyc in &g.cycles {
            let mut h = IDENT;
            for &(e, fwd) in cyc {
                let q = q_at(&eq, e);
                let q = if fwd { q } else { quat_conjugate(q) };
                h = quat_mul(q, h);
            }
            assert!(
                rotor_angle(h) < 1e-3,
                "flat holonomy not identity: {}",
                rotor_angle(h)
            );
        }
    }

    #[test]
    fn curved_sample_has_bounded_flux_holonomy() {
        // Curved plaquettes carry a holonomy angle in [theta_min, pi] (conjugation preserves angle).
        let g = wheel_graph(10);
        let mut rng = Rng(11);
        let tmin = 0.8;
        let eq = sample(&g, &mut rng, true, tmin);
        for cyc in &g.cycles {
            let mut h = IDENT;
            for &(e, fwd) in cyc {
                let q = q_at(&eq, e);
                let q = if fwd { q } else { quat_conjugate(q) };
                h = quat_mul(q, h);
            }
            let ang = rotor_angle(h);
            assert!(
                ang >= tmin - 1e-2 && ang <= std::f32::consts::PI + 1e-2,
                "curved holonomy {ang} out of [{tmin}, pi]"
            );
        }
    }

    #[test]
    fn edge_marginals_match_between_classes() {
        // The metric-integrity claim, in code: pooled edge log-rotors have ~zero mean and
        // ~isotropic covariance in BOTH classes (so a covariance statistic is class-blind).
        let g = wheel_graph(24);
        let stats = |curved: bool| -> ([f32; 3], [f32; 3]) {
            let mut mean = [0.0f32; 3];
            let mut var = [0.0f32; 3];
            let mut cnt = 0usize;
            for s in 0..40u64 {
                let mut rng = Rng(1000 + s);
                let eq = sample(&g, &mut rng, curved, 0.8);
                let f = edge_log_field(&eq);
                for e in 0..f.len() / 3 {
                    for c in 0..3 {
                        mean[c] += f[e * 3 + c];
                        var[c] += f[e * 3 + c] * f[e * 3 + c];
                        if c == 0 {
                            cnt += 1;
                        }
                    }
                }
            }
            for c in 0..3 {
                mean[c] /= cnt as f32;
                var[c] = var[c] / cnt as f32 - mean[c] * mean[c];
            }
            (mean, var)
        };
        let (m0, v0) = stats(false);
        let (m1, v1) = stats(true);
        // means near zero, variances near-equal across classes and across axes (isotropy)
        for c in 0..3 {
            assert!(
                m0[c].abs() < 0.15 && m1[c].abs() < 0.15,
                "mean not ~0: {m0:?} {m1:?}"
            );
            assert!(
                (v0[c] - v1[c]).abs() / v0[c].max(1e-3) < 0.15,
                "per-axis variance differs across classes: {v0:?} vs {v1:?}"
            );
        }
        // isotropy within each class
        let iso =
            |v: [f32; 3]| (v[0].max(v[1]).max(v[2]) - v[0].min(v[1]).min(v[2])) / v[0].max(1e-3);
        assert!(
            iso(v0) < 0.2 && iso(v1) < 0.2,
            "covariance not isotropic: {v0:?} {v1:?}"
        );
    }
}
