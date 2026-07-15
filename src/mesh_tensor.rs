//! `MeshTensor` — the hypergraph latent space AS a mesh tensor.
//!
//! The signed-hypergraph incidence (`cycles`/`signs`/`scale`) is the **mesh**; the
//! latent is a **tensor field on the mesh** — a `d`-vector on every node and every
//! hyperedge. All computation is **contraction along the mesh** (node↔edge),
//! delegating to the FD-verified `hg_message` kernels — never a matrix inversion.
//!
//! This is the hypergraph⇄tensor conjugation done as a *contraction*, not a solve:
//! the substrate a holonomy embedding (HSiKAN / Gomb-Soma) and holonomy feedback act
//! on. [`MeshTopology`] is the mesh-as-operator (the fixed geometry); [`MeshTensor`]
//! is the latent field that rides on it and is evolved by contraction rounds.
//!
//! No autograd: [`MeshTopology::conv_round`] pairs with the hand-derived
//! [`MeshTopology::conv_round_backward`], FD-verified in the tests.

use crate::ops::hg_message::{
    hg_edge_to_node_backward, hg_edge_to_node_forward, hg_node_to_edge_backward,
    hg_node_to_edge_forward,
};

/// The fixed mesh: a signed hypergraph as an incidence structure. Each hyperedge is
/// `k` node corners with signs `σ ∈ {±1}`; `scale[v]` is the per-node structural
/// scale (e.g. `D_v^{-1/2}`). Treated as constant geometry — no gradient flows
/// through it. This is the **operator** side of the mesh tensor: it knows how to
/// contract a field, it holds no field itself.
#[derive(Clone, Debug)]
pub struct MeshTopology {
    cycles: Vec<u32>, // (n_edges, k) node indices — the incidence
    signs: Vec<f32>,  // (n_edges, k) corner signs σ
    scale: Vec<f32>,  // (n_nodes,) structural scale s_v
    n_nodes: usize,
    n_edges: usize,
    k: usize,
}

impl MeshTopology {
    /// Build a mesh from its incidence. `n_edges` is inferred as `cycles.len() / k`.
    ///
    /// # Preconditions
    /// `k >= 1`; `cycles.len()` is a multiple of `k`; `signs.len() == cycles.len()`;
    /// `scale.len() == n_nodes`; every corner index is `< n_nodes`.
    ///
    /// # Panics
    /// If any precondition is violated.
    pub fn new(
        cycles: Vec<u32>,
        signs: Vec<f32>,
        scale: Vec<f32>,
        n_nodes: usize,
        k: usize,
    ) -> Self {
        assert!(k >= 1, "k must be >= 1");
        assert_eq!(cycles.len() % k, 0, "cycles.len() must be a multiple of k");
        assert_eq!(signs.len(), cycles.len(), "signs must match cycles");
        assert_eq!(scale.len(), n_nodes, "scale must have one entry per node");
        assert!(
            cycles.iter().all(|&v| (v as usize) < n_nodes),
            "corner index out of range"
        );
        let n_edges = cycles.len() / k;
        MeshTopology {
            cycles,
            signs,
            scale,
            n_nodes,
            n_edges,
            k,
        }
    }

    /// Node count of the mesh.
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }
    /// Hyperedge count of the mesh.
    pub fn n_edges(&self) -> usize {
        self.n_edges
    }
    /// Corner arity `k` of each hyperedge.
    pub fn arity(&self) -> usize {
        self.k
    }

    /// Contract a node field onto the hyperedges (node→edge): signed scaled mean of
    /// each hyperedge's corners. `nodes` is `(n_nodes, d)`, returns `(n_edges, d)`.
    ///
    /// # Panics
    /// If `nodes.len() != n_nodes * d`.
    pub fn node_to_edge(&self, nodes: &[f32], d: usize) -> Vec<f32> {
        assert_eq!(
            nodes.len(),
            self.n_nodes * d,
            "node field must be (n_nodes, d)"
        );
        hg_node_to_edge_forward(
            nodes,
            &self.cycles,
            &self.signs,
            &self.scale,
            self.n_edges,
            self.k,
            d,
        )
    }

    /// Contract an edge field back onto the nodes (edge→node): signed sum of incident
    /// edges, per-node scaled. `edges` is `(n_edges, d)`, returns `(n_nodes, d)`.
    ///
    /// # Panics
    /// If `edges.len() != n_edges * d`.
    pub fn edge_to_node(&self, edges: &[f32], d: usize) -> Vec<f32> {
        assert_eq!(
            edges.len(),
            self.n_edges * d,
            "edge field must be (n_edges, d)"
        );
        hg_edge_to_node_forward(
            edges,
            &self.cycles,
            &self.signs,
            &self.scale,
            self.n_nodes,
            self.k,
            d,
        )
    }

    /// One signed-HGNN contraction round `node → edge → node` — the core mesh
    /// contraction. `nodes` is `(n_nodes, d)`, returns the new `(n_nodes, d)` field.
    /// Pure (does not mutate any field).
    pub fn conv_round(&self, nodes: &[f32], d: usize) -> Vec<f32> {
        let edges = self.node_to_edge(nodes, d);
        self.edge_to_node(&edges, d)
    }

    /// Hand-derived backward of [`conv_round`](Self::conv_round): the upstream
    /// gradient on the output nodes `(n_nodes, d)` mapped to the gradient on the input
    /// nodes `(n_nodes, d)`. Composes the two `hg_message` backwards (both FD-verified);
    /// the composition itself is FD-verified in the tests.
    ///
    /// # Panics
    /// If `grad_out.len() != n_nodes * d`.
    pub fn conv_round_backward(&self, grad_out: &[f32], d: usize) -> Vec<f32> {
        assert_eq!(
            grad_out.len(),
            self.n_nodes * d,
            "grad_out must be (n_nodes, d)"
        );
        let grad_edges = hg_edge_to_node_backward(
            &self.cycles,
            &self.signs,
            &self.scale,
            grad_out,
            self.n_edges,
            self.k,
            d,
        );
        hg_node_to_edge_backward(
            &self.cycles,
            &self.signs,
            &self.scale,
            &grad_edges,
            self.n_nodes,
            self.k,
            d,
        )
    }
}

/// The latent as a mesh tensor: a `d`-vector field on the nodes and hyperedges of a
/// [`MeshTopology`], evolved by contraction. The **state** side of the mesh tensor.
#[derive(Clone, Debug)]
pub struct MeshTensor<'m> {
    topo: &'m MeshTopology,
    d: usize,
    nodes: Vec<f32>, // (n_nodes, d) tensor field on vertices
    edges: Vec<f32>, // (n_edges, d) tensor field on hyperedges
}

impl<'m> MeshTensor<'m> {
    /// A latent seeded on the nodes; the edge field starts at zero.
    ///
    /// # Panics
    /// If `nodes.len() != n_nodes * d`.
    pub fn on_nodes(topo: &'m MeshTopology, d: usize, nodes: Vec<f32>) -> Self {
        assert_eq!(
            nodes.len(),
            topo.n_nodes * d,
            "node field must be (n_nodes, d)"
        );
        let edges = vec![0.0f32; topo.n_edges * d];
        MeshTensor {
            topo,
            d,
            nodes,
            edges,
        }
    }

    /// The node tensor field `(n_nodes, d)`.
    pub fn nodes(&self) -> &[f32] {
        &self.nodes
    }
    /// The edge tensor field `(n_edges, d)`.
    pub fn edges(&self) -> &[f32] {
        &self.edges
    }
    /// Channel width `d`.
    pub fn width(&self) -> usize {
        self.d
    }

    /// Contract the node field onto the hyperedges (node→edge), updating the edge field.
    pub fn lift_to_edges(&mut self) {
        self.edges = self.topo.node_to_edge(&self.nodes, self.d);
    }

    /// Contract the edge field back onto the nodes (edge→node), updating the node field.
    pub fn project_to_nodes(&mut self) {
        self.nodes = self.topo.edge_to_node(&self.edges, self.d);
    }

    /// Evolve the latent by one contraction round `node → edge → node`, updating both
    /// fields (edges hold the intermediate).
    pub fn conv_round(&mut self) {
        self.edges = self.topo.node_to_edge(&self.nodes, self.d);
        self.nodes = self.topo.edge_to_node(&self.edges, self.d);
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

    /// A small signed mesh: 4 nodes, 2 triangular hyperedges sharing node 1.
    fn small_mesh() -> MeshTopology {
        // edge 0 = {0,1,2}, edge 1 = {1,2,3}; signs mixed; unit scale
        let cycles = vec![0, 1, 2, 1, 2, 3];
        let signs = vec![1.0, -1.0, 1.0, 1.0, 1.0, -1.0];
        let scale = vec![1.0, 0.5, 0.5, 1.0];
        MeshTopology::new(cycles, signs, scale, 4, 3)
    }

    #[test]
    #[should_panic(expected = "corner index out of range")]
    fn topology_rejects_out_of_range_corner() {
        MeshTopology::new(vec![0, 1, 9], vec![1.0, 1.0, 1.0], vec![1.0, 1.0], 2, 3);
    }

    #[test]
    fn conv_round_equals_manual_hg_composition() {
        let topo = small_mesh();
        let d = 3;
        let mut nx = lcg(1);
        let nodes: Vec<f32> = (0..topo.n_nodes() * d).map(|_| nx()).collect();
        let got = topo.conv_round(&nodes, d);
        let manual = topo.edge_to_node(&topo.node_to_edge(&nodes, d), d);
        assert_eq!(got, manual);
        // and the stateful MeshTensor agrees
        let mut mt = MeshTensor::on_nodes(&topo, d, nodes.clone());
        mt.conv_round();
        assert_eq!(mt.nodes(), &manual[..]);
    }

    #[test]
    fn conv_round_backward_matches_fd() {
        let topo = small_mesh();
        let d = 2;
        let mut nx = lcg(7);
        let nodes: Vec<f32> = (0..topo.n_nodes() * d).map(|_| nx()).collect();
        let grad_out: Vec<f32> = (0..topo.n_nodes() * d).map(|_| nx()).collect(); // upstream ∂L/∂out
        let grad_in = topo.conv_round_backward(&grad_out, d);
        // scalar loss L = <grad_out, conv_round(nodes)>; check ∂L/∂nodes by central FD
        let loss = |n: &[f32]| -> f32 {
            topo.conv_round(n, d)
                .iter()
                .zip(&grad_out)
                .map(|(&a, &b)| a * b)
                .sum()
        };
        let eps = 1e-3f32;
        for i in 0..nodes.len() {
            let (mut hp, mut hm) = (nodes.clone(), nodes.clone());
            hp[i] += eps;
            hm[i] -= eps;
            let fd = (loss(&hp) - loss(&hm)) / (2.0 * eps);
            assert!(
                (fd - grad_in[i]).abs() < 1e-2,
                "grad[{i}] fd {fd} vs analytic {}",
                grad_in[i]
            );
        }
    }

    #[test]
    fn isolated_node_stays_zero_after_projection() {
        // node 3 is in NO hyperedge -> projecting an edge field leaves it zero (mesh structure)
        let cycles = vec![0, 1, 2]; // one edge over {0,1,2}; node 3 isolated
        let topo = MeshTopology::new(cycles, vec![1.0, 1.0, 1.0], vec![1.0; 4], 4, 3);
        let d = 2;
        let edges = vec![1.0, 2.0]; // (1 edge, d)
        let nodes = topo.edge_to_node(&edges, d);
        assert_eq!(
            &nodes[3 * d..3 * d + d],
            &[0.0, 0.0],
            "isolated node must stay zero"
        );
    }
}
