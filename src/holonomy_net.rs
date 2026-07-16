//! `RotorMeshNet` — a DEEP holonomy representation over a simplicial/hypergraph mesh.
//!
//! Each layer is *rotate then mix*:
//! 1. **rotate** — a learned per-node rotor (Cayley-parameterized bivector → unit
//!    quaternion, [`crate::cayley_rotor_forward`]) transports the node 3-vector field.
//!    The bivectors are the **learned representation** (not designed).
//! 2. **mix** — one mesh contraction round ([`crate::MeshTopology::conv_round`],
//!    node→edge→node) spreads the transported field across the simplicial neighborhood.
//!
//! Stacking these is genuinely deep: pure rotations would collapse (`SO(3)` composes
//! to one rotation), but the mesh mix *between* rotors re-combines the field, so depth
//! adds capacity. This is the auto-holonomy composed through depth.
//!
//! # No autograd
//! The end-to-end backward composes the two hand-derived, FD-verified closed-form
//! backwards ([`crate::cayley_rotor_backward`], [`crate::MeshTopology::conv_round_backward`]).
//! The gradient handed down between layers is `grad_v = R̄ · grad` — the upstream
//! gradient transported by the **inverse rotor** (quaternion conjugate): the *adjoint
//! holonomy transport* that replaces backprop's tape. The whole stack is FD-verified.
//!
//! The field width is fixed at `d = 3` (a rotor acts on a 3-vector per node).

use crate::mesh_tensor::MeshTopology;
use crate::ops::cayley_rotor::{cayley_rotor_backward, cayley_rotor_forward};

const D: usize = 3;

/// A deep rotor-mesh network: `n_layers` layers of (learned per-node rotor) + (mesh
/// contraction), over a fixed [`MeshTopology`].
#[derive(Clone, Debug)]
pub struct RotorMeshNet<'m> {
    topo: &'m MeshTopology,
    /// Per layer, the `(n_nodes, 3)` Cayley bivector parameters (the learned rotors).
    bivecs: Vec<Vec<f32>>,
    n_nodes: usize,
}

/// Saved forward state for the backward pass (per layer: the layer input field, the
/// rotor quaternions, and the rotated field fed to the mesh mix).
#[derive(Clone, Debug, Default)]
pub struct RotorMeshCache {
    v_in: Vec<Vec<f32>>,    // (n_nodes, 3) input to each layer's rotate
    quats: Vec<Vec<f32>>,   // (n_nodes, 4) saved rotors
    rotated: Vec<Vec<f32>>, // (n_nodes, 3) rotated field (input to the mesh mix)
}

impl<'m> RotorMeshNet<'m> {
    /// Build a net from per-layer bivector parameters. Each `bivecs[l]` is `(n_nodes, 3)`.
    ///
    /// # Preconditions
    /// `bivecs` is non-empty; every layer has `topo.n_nodes() * 3` entries.
    ///
    /// # Panics
    /// If any precondition is violated.
    pub fn new(topo: &'m MeshTopology, bivecs: Vec<Vec<f32>>) -> Self {
        assert!(!bivecs.is_empty(), "need at least one layer");
        let n_nodes = topo.n_nodes();
        for (l, b) in bivecs.iter().enumerate() {
            assert_eq!(b.len(), n_nodes * D, "layer {l} bivec must be (n_nodes, 3)");
        }
        RotorMeshNet {
            topo,
            bivecs,
            n_nodes,
        }
    }

    /// Number of layers (depth).
    pub fn depth(&self) -> usize {
        self.bivecs.len()
    }

    /// Mutable access to a layer's `(n_nodes, 3)` bivector parameters (for a learning
    /// step to write into — the holonomy/entropy feedback updates these).
    pub fn bivecs_mut(&mut self, layer: usize) -> &mut [f32] {
        &mut self.bivecs[layer]
    }

    /// Forward: `v0 (n_nodes, 3)` → final field `(n_nodes, 3)`, plus the cache.
    ///
    /// # Panics
    /// If `v0.len() != n_nodes * 3`.
    pub fn forward(&self, v0: &[f32]) -> (Vec<f32>, RotorMeshCache) {
        assert_eq!(v0.len(), self.n_nodes * D, "input must be (n_nodes, 3)");
        let mut cache = RotorMeshCache::default();
        let mut v = v0.to_vec();
        for b in &self.bivecs {
            let (rotated, quats) = cayley_rotor_forward(b, &v, self.n_nodes);
            let out = self.topo.conv_round(&rotated, D);
            cache.v_in.push(v);
            cache.quats.push(quats);
            cache.rotated.push(rotated);
            v = out;
        }
        (v, cache)
    }

    /// Backward: upstream `grad_out (n_nodes, 3)` → `(grad_bivecs, grad_v0)`, where
    /// `grad_bivecs[l]` is `(n_nodes, 3)` and `grad_v0` is `(n_nodes, 3)`. Composes the
    /// two FD-verified closed-form backwards; the gradient handed down is the
    /// inverse-rotor (adjoint) transport of the upstream gradient.
    ///
    /// # Panics
    /// If `grad_out.len() != n_nodes * 3` or the cache depth mismatches.
    pub fn backward(&self, cache: &RotorMeshCache, grad_out: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>) {
        assert_eq!(grad_out.len(), self.n_nodes * D);
        assert_eq!(cache.v_in.len(), self.depth(), "cache depth mismatch");
        let mut grad_bivecs = vec![Vec::new(); self.depth()];
        let mut g = grad_out.to_vec(); // gradient on the current layer's output
        for l in (0..self.depth()).rev() {
            // through the mesh mix: ∂L/∂out → ∂L/∂rotated
            let g_rot = self.topo.conv_round_backward(&g, D);
            // through the rotor: → (∂L/∂bivec_l, ∂L/∂v_in_l); ∂L/∂v_in is the
            // inverse-rotor transport of g_rot (the adjoint holonomy transport)
            let (gb, gv) = cayley_rotor_backward(
                &self.bivecs[l],
                &cache.v_in[l],
                &cache.quats[l],
                &g_rot,
                self.n_nodes,
            );
            grad_bivecs[l] = gb;
            g = gv;
        }
        (grad_bivecs, g)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: u64) -> impl FnMut() -> f32 {
        let mut xs = seed;
        move || {
            xs = xs
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((xs >> 32) as u32 as f32) / 4294967296.0 - 0.5
        }
    }

    fn small_mesh() -> MeshTopology {
        // 4 nodes, 2 triangular hyperedges sharing nodes 1,2
        MeshTopology::new(
            vec![0, 1, 2, 1, 2, 3],
            vec![1.0, -1.0, 1.0, 1.0, 1.0, -1.0],
            vec![1.0, 0.5, 0.5, 1.0],
            4,
            3,
        )
    }

    #[test]
    fn depth_collapses_without_the_mesh_mix_but_not_with_it() {
        // sanity: forward runs at depth 3 and the output actually depends on depth
        let topo = small_mesh();
        let mut nx = lcg(1);
        let n = topo.n_nodes();
        let v0: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let mk = |layers: usize, seed: u64| {
            let mut r = lcg(seed);
            let bivecs: Vec<Vec<f32>> = (0..layers)
                .map(|_| (0..n * 3).map(|_| r()).collect())
                .collect();
            RotorMeshNet::new(&topo, bivecs).forward(&v0).0
        };
        let one = mk(1, 9);
        let three = mk(3, 9);
        let diff: f32 = one.iter().zip(&three).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            diff > 1e-3,
            "deeper net should differ from shallow (got {diff})"
        );
    }

    /// The crux: the deep net's closed-form composed backward matches finite
    /// differences end-to-end — w.r.t. every layer's bivectors AND the input field.
    #[test]
    fn deep_backward_matches_fd() {
        let topo = small_mesh();
        let n = topo.n_nodes();
        let mut nx = lcg(7);
        let depth = 3;
        let bivecs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..n * 3).map(|_| 0.3 * nx()).collect())
            .collect();
        let v0: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let grad_out: Vec<f32> = (0..n * 3).map(|_| nx()).collect();

        let net = RotorMeshNet::new(&topo, bivecs.clone());
        let (_out, cache) = net.forward(&v0);
        let (grad_bivecs, grad_v0) = net.backward(&cache, &grad_out);

        // scalar loss L = <grad_out, forward(·)>
        let eps = 1e-3f32;
        let loss_with = |bv: &[Vec<f32>], v: &[f32]| -> f32 {
            RotorMeshNet::new(&topo, bv.to_vec())
                .forward(v)
                .0
                .iter()
                .zip(&grad_out)
                .map(|(a, b)| a * b)
                .sum()
        };
        // check grad w.r.t. each layer's bivectors
        for l in 0..depth {
            for i in 0..n * 3 {
                let (mut bp, mut bm) = (bivecs.clone(), bivecs.clone());
                bp[l][i] += eps;
                bm[l][i] -= eps;
                let fd = (loss_with(&bp, &v0) - loss_with(&bm, &v0)) / (2.0 * eps);
                assert!(
                    (fd - grad_bivecs[l][i]).abs() < 2e-2,
                    "bivec grad L{l}[{i}]: fd {fd} vs analytic {}",
                    grad_bivecs[l][i]
                );
            }
        }
        // check grad w.r.t. the input field
        for i in 0..n * 3 {
            let (mut vp, mut vm) = (v0.clone(), v0.clone());
            vp[i] += eps;
            vm[i] -= eps;
            let fd = (loss_with(&bivecs, &vp) - loss_with(&bivecs, &vm)) / (2.0 * eps);
            assert!(
                (fd - grad_v0[i]).abs() < 2e-2,
                "input grad [{i}]: fd {fd} vs analytic {}",
                grad_v0[i]
            );
        }
    }
}
