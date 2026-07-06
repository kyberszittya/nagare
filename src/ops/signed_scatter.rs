//! Signed, gate-weighted scatter over routed sequence pairs.
//!
//! Each pair contributes an already-transported vector to its query lane:
//! `mixed[query, block] += gate[pair] * tanh(offset_sign[offset, block]) *
//! transported[pair, block]`. The backward accumulates value and gate gradients
//! per pair, and sign gradients per `(offset, block)` lane.

fn pair_block_base(pair: usize, block: usize, n_blocks: usize) -> usize {
    (pair * n_blocks + block) * 3
}

fn query_block_base(query: usize, block: usize, n_blocks: usize) -> usize {
    (query * n_blocks + block) * 3
}

/// Query and offset lanes for signed scatter pairs.
#[derive(Debug, Clone, Copy)]
pub struct SignedScatterLanes<'a> {
    /// Query index for each routed pair, flat `(n_pairs)`.
    pub query_of: &'a [u32],
    /// Relative offset for each routed pair, flat `(n_pairs)`.
    pub offset_of: &'a [u32],
}

impl<'a> SignedScatterLanes<'a> {
    /// Construct pair lanes.
    ///
    /// # Preconditions
    /// `query_of.len() == offset_of.len()`.
    ///
    /// # Postconditions
    /// Returns a lane view borrowing the provided buffers.
    ///
    /// # Panics
    /// Panics if the lane lengths differ.
    pub fn new(query_of: &'a [u32], offset_of: &'a [u32]) -> Self {
        assert_eq!(query_of.len(), offset_of.len());
        Self {
            query_of,
            offset_of,
        }
    }
}

/// Shape metadata for signed scatter.
#[derive(Debug, Clone, Copy)]
pub struct SignedScatterLayout {
    /// Number of query lanes.
    pub n_queries: usize,
    /// Number of relative-offset lanes.
    pub n_offsets: usize,
    /// Number of 3-vector blocks per pair/query.
    pub n_blocks: usize,
}

impl SignedScatterLayout {
    /// Construct signed-scatter shape metadata.
    ///
    /// # Preconditions
    /// All dimensions must be non-zero.
    ///
    /// # Postconditions
    /// Returns a copyable layout value.
    ///
    /// # Panics
    /// Panics if any dimension is zero.
    pub fn new(n_queries: usize, n_offsets: usize, n_blocks: usize) -> Self {
        assert!(n_queries > 0);
        assert!(n_offsets > 0);
        assert!(n_blocks > 0);
        Self {
            n_queries,
            n_offsets,
            n_blocks,
        }
    }
}

/// Forward signed scatter.
///
/// # Args
/// - `transported`: Flat `(n_pairs, n_blocks, 3)` transported vectors.
/// - `gate`: Flat `(n_pairs)` gate weights.
/// - `offset_sign`: Flat `(n_offsets, n_blocks)` raw sign logits; `tanh` is
///   applied inside the op.
/// - `lanes`: Query and offset lane indices, each flat `(n_pairs)`.
/// - `layout`: Query/offset/block dimensions.
///
/// # Preconditions
/// Buffer lengths must match the documented shapes. `query_of[p] < n_queries`
/// and `offset_of[p] < n_offsets` for every pair.
///
/// # Postconditions
/// Returns `(mixed, signs)` with shapes `(n_queries, n_blocks, 3)` and
/// `(n_offsets, n_blocks)`. `signs` is saved for the backward pass.
///
/// # Panics
/// Panics if input shapes or lane indices violate the preconditions.
pub fn signed_scatter_forward(
    transported: &[f32],
    gate: &[f32],
    offset_sign: &[f32],
    lanes: SignedScatterLanes<'_>,
    layout: SignedScatterLayout,
) -> (Vec<f32>, Vec<f32>) {
    let n_pairs = gate.len();
    assert_eq!(lanes.query_of.len(), n_pairs);
    assert_eq!(lanes.offset_of.len(), n_pairs);
    assert_eq!(transported.len(), n_pairs * layout.n_blocks * 3);
    assert_eq!(offset_sign.len(), layout.n_offsets * layout.n_blocks);

    let signs: Vec<f32> = offset_sign.iter().map(|x| x.tanh()).collect();
    let mut mixed = vec![0.0; layout.n_queries * layout.n_blocks * 3];

    for (pair, &gate_weight) in gate.iter().enumerate() {
        let query = lanes.query_of[pair] as usize;
        let offset = lanes.offset_of[pair] as usize;
        assert!(query < layout.n_queries);
        assert!(offset < layout.n_offsets);
        for block in 0..layout.n_blocks {
            let sign = signs[offset * layout.n_blocks + block];
            let weight = gate_weight * sign;
            let t_base = pair_block_base(pair, block, layout.n_blocks);
            let m_base = query_block_base(query, block, layout.n_blocks);
            for c in 0..3 {
                mixed[m_base + c] += weight * transported[t_base + c];
            }
        }
    }

    (mixed, signs)
}

/// Backward signed scatter.
///
/// # Args
/// - `transported`: Flat `(n_pairs, n_blocks, 3)` transported vectors.
/// - `gate`: Flat `(n_pairs)` gate weights.
/// - `signs`: Flat `(n_offsets, n_blocks)` saved `tanh(offset_sign)` values.
/// - `lanes`: Query and offset lane indices, each flat `(n_pairs)`.
/// - `grad_mixed`: Flat `(n_queries, n_blocks, 3)` incoming gradient.
/// - `layout`: Query/offset/block dimensions.
///
/// # Preconditions
/// Buffer lengths must match the documented shapes. `signs` must come from the
/// matching forward pass.
///
/// # Postconditions
/// Returns `(grad_transported, grad_gate, grad_offset_sign)` with shapes
/// `(n_pairs, n_blocks, 3)`, `(n_pairs)`, and `(n_offsets, n_blocks)`.
///
/// # Panics
/// Panics if input shapes or lane indices violate the preconditions.
pub fn signed_scatter_backward(
    transported: &[f32],
    gate: &[f32],
    signs: &[f32],
    lanes: SignedScatterLanes<'_>,
    grad_mixed: &[f32],
    layout: SignedScatterLayout,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let n_pairs = gate.len();
    assert_eq!(lanes.query_of.len(), n_pairs);
    assert_eq!(lanes.offset_of.len(), n_pairs);
    assert_eq!(transported.len(), n_pairs * layout.n_blocks * 3);
    assert_eq!(signs.len(), layout.n_offsets * layout.n_blocks);
    assert_eq!(grad_mixed.len(), layout.n_queries * layout.n_blocks * 3);

    let mut grad_transported = vec![0.0; transported.len()];
    let mut grad_gate = vec![0.0; n_pairs];
    let mut grad_offset_sign = vec![0.0; layout.n_offsets * layout.n_blocks];

    for (pair, &gate_weight) in gate.iter().enumerate() {
        let query = lanes.query_of[pair] as usize;
        let offset = lanes.offset_of[pair] as usize;
        assert!(query < layout.n_queries);
        assert!(offset < layout.n_offsets);
        for block in 0..layout.n_blocks {
            let sign_idx = offset * layout.n_blocks + block;
            let sign = signs[sign_idx];
            let t_base = pair_block_base(pair, block, layout.n_blocks);
            let m_base = query_block_base(query, block, layout.n_blocks);
            let mut grad_dot_transported = 0.0;
            for c in 0..3 {
                let gm = grad_mixed[m_base + c];
                let transported_value = transported[t_base + c];
                grad_transported[t_base + c] += gate_weight * sign * gm;
                grad_dot_transported += gm * transported_value;
            }
            grad_gate[pair] += sign * grad_dot_transported;
            grad_offset_sign[sign_idx] += gate_weight * (1.0 - sign * sign) * grad_dot_transported;
        }
    }

    (grad_transported, grad_gate, grad_offset_sign)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_matches_hand_sum() {
        let transported = vec![1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 0.25, -0.5, 1.0];
        let gate = vec![0.5, 2.0, -1.0];
        let offset_sign = vec![0.0, 1.0];
        let query_of = vec![0, 0, 1];
        let offset_of = vec![0, 1, 1];
        let (mixed, signs) = signed_scatter_forward(
            &transported,
            &gate,
            &offset_sign,
            SignedScatterLanes::new(&query_of, &offset_of),
            SignedScatterLayout::new(2, 2, 1),
        );
        let s0 = signs[0];
        let s1 = signs[1];
        assert!((mixed[0] - (0.5 * s0 * 1.0 - 2.0 * s1)).abs() < 1e-6);
        assert!((mixed[1] - (0.5 * s0 * 2.0 + 2.0 * s1 * 0.5)).abs() < 1e-6);
        assert!((mixed[2] - (0.5 * s0 * 3.0 + 2.0 * s1 * 2.0)).abs() < 1e-6);
        assert!((mixed[3] - (-s1 * 0.25)).abs() < 1e-6);
        assert!((mixed[4] - (s1 * 0.5)).abs() < 1e-6);
        assert!((mixed[5] - (-s1)).abs() < 1e-6);
    }

    #[test]
    fn backward_matches_numerical_for_all_inputs() {
        let transported = vec![
            1.0, 2.0, 3.0, -1.0, 0.5, 2.0, 0.25, -0.5, 1.0, 0.6, -0.2, 0.4,
        ];
        let gate = vec![0.5, 2.0];
        let offset_sign = vec![0.1, -0.2, 0.3, 0.4];
        let query_of = vec![0, 1];
        let offset_of = vec![0, 1];
        let n_queries = 2;
        let n_offsets = 2;
        let n_blocks = 2;
        let (mixed, signs) = signed_scatter_forward(
            &transported,
            &gate,
            &offset_sign,
            SignedScatterLanes::new(&query_of, &offset_of),
            SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
        );
        let grad_mixed = vec![1.0; mixed.len()];
        let (grad_t, grad_g, grad_s) = signed_scatter_backward(
            &transported,
            &gate,
            &signs,
            SignedScatterLanes::new(&query_of, &offset_of),
            &grad_mixed,
            SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
        );
        let eps = 1e-3;

        for idx in 0..transported.len() {
            let mut plus = transported.clone();
            plus[idx] += eps;
            let mut minus = transported.clone();
            minus[idx] -= eps;
            let loss_plus: f32 = signed_scatter_forward(
                &plus,
                &gate,
                &offset_sign,
                SignedScatterLanes::new(&query_of, &offset_of),
                SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
            )
            .0
            .iter()
            .sum();
            let loss_minus: f32 = signed_scatter_forward(
                &minus,
                &gate,
                &offset_sign,
                SignedScatterLanes::new(&query_of, &offset_of),
                SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
            )
            .0
            .iter()
            .sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!((grad_t[idx] - numeric).abs() < 1e-2);
        }

        for idx in 0..gate.len() {
            let mut plus = gate.clone();
            plus[idx] += eps;
            let mut minus = gate.clone();
            minus[idx] -= eps;
            let loss_plus: f32 = signed_scatter_forward(
                &transported,
                &plus,
                &offset_sign,
                SignedScatterLanes::new(&query_of, &offset_of),
                SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
            )
            .0
            .iter()
            .sum();
            let loss_minus: f32 = signed_scatter_forward(
                &transported,
                &minus,
                &offset_sign,
                SignedScatterLanes::new(&query_of, &offset_of),
                SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
            )
            .0
            .iter()
            .sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!((grad_g[idx] - numeric).abs() < 1e-2);
        }

        for idx in 0..offset_sign.len() {
            let mut plus = offset_sign.clone();
            plus[idx] += eps;
            let mut minus = offset_sign.clone();
            minus[idx] -= eps;
            let loss_plus: f32 = signed_scatter_forward(
                &transported,
                &gate,
                &plus,
                SignedScatterLanes::new(&query_of, &offset_of),
                SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
            )
            .0
            .iter()
            .sum();
            let loss_minus: f32 = signed_scatter_forward(
                &transported,
                &gate,
                &minus,
                SignedScatterLanes::new(&query_of, &offset_of),
                SignedScatterLayout::new(n_queries, n_offsets, n_blocks),
            )
            .0
            .iter()
            .sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!((grad_s[idx] - numeric).abs() < 1e-2);
        }
    }
}
