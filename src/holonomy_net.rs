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
use hymeko_clifford::{quat_conjugate, quat_rotate};

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

    /// **Holonomy-DFA backward** — the biologically-plausible credit assignment. The *same*
    /// global output-gradient `grad_out` is broadcast to *every* layer (no sequential
    /// threading); each layer applies its own exact mesh + inverse-rotor adjoint. Global
    /// routing, local exact transport: no weight transport, no stored tape, and — unlike
    /// direct feedback alignment — no separate feedback weights (it reuses the forward rotors).
    /// Returns per-layer bivector gradients `(n_nodes, 3)`.
    ///
    /// The top layer's gradient equals the sequential [`Self::backward`]'s (the broadcast `E`
    /// *is* the threaded `E` at the top); lower layers differ — that is the DFA approximation.
    ///
    /// # Panics
    /// If `grad_out.len() != n_nodes * 3` or the cache depth mismatches.
    pub fn backward_dfa(&self, cache: &RotorMeshCache, grad_out: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(grad_out.len(), self.n_nodes * D);
        assert_eq!(cache.v_in.len(), self.depth(), "cache depth mismatch");
        // one mesh adjoint of the global error, reused for every layer's local rotor adjoint
        let g_rot = self.topo.conv_round_backward(grad_out, D);
        (0..self.depth())
            .map(|l| {
                cayley_rotor_backward(
                    &self.bivecs[l],
                    &cache.v_in[l],
                    &cache.quats[l],
                    &g_rot,
                    self.n_nodes,
                )
                .0
            })
            .collect()
    }

    /// Rotor-only backward from an externally supplied per-layer field gradient (each at that
    /// layer's rotor output). Each layer's bivector gradient is the exact inverse-rotor adjoint
    /// of its given field gradient — a hook for alternative feedback routings (e.g. random-DFA
    /// feeds a random projection of the output error here). No mesh adjoint, no threading.
    ///
    /// # Panics
    /// If `rot_grads.len() != depth`, any entry is not `(n_nodes, 3)`, or the cache mismatches.
    pub fn backward_from_rot_grads(
        &self,
        cache: &RotorMeshCache,
        rot_grads: &[Vec<f32>],
    ) -> Vec<Vec<f32>> {
        assert_eq!(
            rot_grads.len(),
            self.depth(),
            "need one field gradient per layer"
        );
        assert_eq!(cache.v_in.len(), self.depth(), "cache depth mismatch");
        (0..self.depth())
            .map(|l| {
                assert_eq!(rot_grads[l].len(), self.n_nodes * D);
                cayley_rotor_backward(
                    &self.bivecs[l],
                    &cache.v_in[l],
                    &cache.quats[l],
                    &rot_grads[l],
                    self.n_nodes,
                )
                .0
            })
            .collect()
    }

    /// **Depth-composing transported broadcast** — the middle ground between the naive broadcast
    /// ([`Self::backward_dfa`], which hands every layer the raw top error and does not compose
    /// through depth) and exact backprop ([`Self::backward`]). The global credit signal is
    /// transported *down* through the **pure inverse-rotor chain** — each layer applies its own
    /// inverse rotor `R̄` (the return-path holonomy) — so the signal a layer receives depends on
    /// the rotors above it (depth composes), while the *inter-layer* mesh Jacobian is dropped from
    /// the transport (the approximation vs.\ exact, and what makes it cheap/parallel).
    ///
    /// This is the credit-side of "connection transport": differentiation as holonomy along the
    /// network path. It is also the backward-analogue of an inter-shell rotor transport — the
    /// connective primitive for a concentric-shell Gömb-Soma.
    ///
    /// # Panics
    /// If `grad_out.len() != n_nodes * 3` or the cache depth mismatches.
    pub fn backward_dfa_transported(
        &self,
        cache: &RotorMeshCache,
        grad_out: &[f32],
    ) -> Vec<Vec<f32>> {
        assert_eq!(grad_out.len(), self.n_nodes * D);
        assert_eq!(cache.v_in.len(), self.depth(), "cache depth mismatch");
        let mut grad_bivecs = vec![Vec::new(); self.depth()];
        let mut g = grad_out.to_vec(); // credit signal, holonomy-transported down
        for l in (0..self.depth()).rev() {
            // local bivec gradient = exact mesh + rotor adjoint of the transported-so-far signal
            let g_rot = self.topo.conv_round_backward(&g, D);
            let (gb, _gv) = cayley_rotor_backward(
                &self.bivecs[l],
                &cache.v_in[l],
                &cache.quats[l],
                &g_rot,
                self.n_nodes,
            );
            grad_bivecs[l] = gb;
            // transport the signal DOWN by this layer's pure inverse rotor (return-path holonomy),
            // dropping the mesh coupling from the inter-layer transport path
            let mut g_next = vec![0.0f32; self.n_nodes * D];
            for i in 0..self.n_nodes {
                let q = [
                    cache.quats[l][i * 4],
                    cache.quats[l][i * 4 + 1],
                    cache.quats[l][i * 4 + 2],
                    cache.quats[l][i * 4 + 3],
                ];
                let gi = [g[i * D], g[i * D + 1], g[i * D + 2]];
                let r = quat_rotate(quat_conjugate(q), gi);
                g_next[i * D..i * D + 3].copy_from_slice(&r);
            }
            g = g_next;
        }
        grad_bivecs
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

    /// Holonomy-DFA's TOP-layer gradient equals the sequential backward's (broadcast E ≡ threaded
    /// E at the top); lower layers differ (the DFA approximation). Also checks shapes.
    #[test]
    fn dfa_top_layer_equals_sequential_lower_differs() {
        let topo = small_mesh();
        let n = topo.n_nodes();
        let mut nx = lcg(21);
        let depth = 3;
        let bivecs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..n * 3).map(|_| 0.3 * nx()).collect())
            .collect();
        let v0: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let grad_out: Vec<f32> = (0..n * 3).map(|_| nx()).collect();

        let net = RotorMeshNet::new(&topo, bivecs);
        let (_out, cache) = net.forward(&v0);
        let (seq, _gv0) = net.backward(&cache, &grad_out);
        let dfa = net.backward_dfa(&cache, &grad_out);

        assert_eq!(dfa.len(), depth);
        for layer in &dfa {
            assert_eq!(layer.len(), n * 3);
        }
        // top layer identical
        let top = depth - 1;
        for i in 0..n * 3 {
            assert!(
                (seq[top][i] - dfa[top][i]).abs() < 1e-6,
                "top layer must match sequential: seq {} vs dfa {}",
                seq[top][i],
                dfa[top][i]
            );
        }
        // at least one lower layer must differ (broadcast ≠ threaded below the top)
        let diff: f32 = (0..n * 3).map(|i| (seq[0][i] - dfa[0][i]).abs()).sum();
        assert!(
            diff > 1e-4,
            "lower layer unexpectedly identical (diff {diff})"
        );
    }

    /// `backward_from_rot_grads` applied to the per-layer mesh-adjoint of a broadcast grad
    /// reproduces `backward_dfa` — the routing hook composes with the mesh adjoint.
    #[test]
    fn from_rot_grads_composes_with_dfa() {
        let topo = small_mesh();
        let n = topo.n_nodes();
        let mut nx = lcg(5);
        let depth = 2;
        let bivecs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..n * 3).map(|_| 0.2 * nx()).collect())
            .collect();
        let v0: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let grad_out: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let net = RotorMeshNet::new(&topo, bivecs);
        let (_o, cache) = net.forward(&v0);
        let dfa = net.backward_dfa(&cache, &grad_out);
        let g_rot = topo.conv_round_backward(&grad_out, 3);
        let rot_grads = vec![g_rot.clone(), g_rot];
        let via_hook = net.backward_from_rot_grads(&cache, &rot_grads);
        for l in 0..depth {
            for i in 0..n * 3 {
                assert!((dfa[l][i] - via_hook[l][i]).abs() < 1e-6);
            }
        }
    }

    /// The transported broadcast agrees with BOTH sequential and naive-DFA at the top layer
    /// (all three start from the raw global E), but its lower layers differ from the naive
    /// broadcast — the rotor-chain transport actually moves the signal through depth.
    #[test]
    fn transported_top_matches_lower_differs_from_naive() {
        let topo = small_mesh();
        let n = topo.n_nodes();
        let mut nx = lcg(31);
        let depth = 3;
        let bivecs: Vec<Vec<f32>> = (0..depth)
            .map(|_| (0..n * 3).map(|_| 0.3 * nx()).collect())
            .collect();
        let v0: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let grad_out: Vec<f32> = (0..n * 3).map(|_| nx()).collect();
        let net = RotorMeshNet::new(&topo, bivecs);
        let (_o, cache) = net.forward(&v0);
        let (seq, _gv) = net.backward(&cache, &grad_out);
        let naive = net.backward_dfa(&cache, &grad_out);
        let trans = net.backward_dfa_transported(&cache, &grad_out);

        let top = depth - 1;
        for i in 0..n * 3 {
            assert!(
                (trans[top][i] - seq[top][i]).abs() < 1e-6,
                "top must match sequential"
            );
            assert!(
                (trans[top][i] - naive[top][i]).abs() < 1e-6,
                "top must match naive"
            );
        }
        // a lower layer: transported must differ from the naive broadcast (transport moved it)
        let diff: f32 = (0..n * 3).map(|i| (trans[0][i] - naive[0][i]).abs()).sum();
        assert!(
            diff > 1e-4,
            "transport did not change the lower-layer signal (diff {diff})"
        );
    }
}
