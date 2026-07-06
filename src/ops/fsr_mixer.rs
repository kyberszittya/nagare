//! Composed Fiber-Spike-Rotor sequence mixer.
//!
//! The mixer composes the existing linear op with the Cayley rotor and signed
//! scatter kernels. Top-k source indices and gate weights are explicit inputs;
//! the data-dependent selection is outside this closed-form kernel.

use crate::ops::cayley_rotor::{cayley_rotor_backward, cayley_rotor_forward};
use crate::ops::linear::{linear_backward, linear_forward, LinearLayer};
use crate::ops::signed_scatter::{
    signed_scatter_backward, signed_scatter_forward, SignedScatterLanes, SignedScatterLayout,
};

/// Routed top-k sequence pairs for an FSR mixer call.
#[derive(Debug, Clone, Copy)]
pub struct FsrRoute<'a> {
    /// Selected source indices, flat `(batch, seq_len, k)`.
    pub selected: &'a [u32],
    /// Gate weights for the selected sources, flat `(batch, seq_len, k)`.
    pub gate: &'a [f32],
    /// Batch size.
    pub batch: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Number of selected sources per query.
    pub k: usize,
}

impl<'a> FsrRoute<'a> {
    /// Construct a routed top-k view.
    ///
    /// # Preconditions
    /// `selected.len() == gate.len() == batch * seq_len * k`.
    ///
    /// # Postconditions
    /// Returns a route borrowing the provided buffers.
    ///
    /// # Panics
    /// Panics if the buffer lengths do not match the dimensions.
    pub fn new(
        selected: &'a [u32],
        gate: &'a [f32],
        batch: usize,
        seq_len: usize,
        k: usize,
    ) -> Self {
        assert_eq!(selected.len(), batch * seq_len * k);
        assert_eq!(gate.len(), batch * seq_len * k);
        Self {
            selected,
            gate,
            batch,
            seq_len,
            k,
        }
    }
}

/// Fiber-Spike-Rotor mixer parameters.
#[derive(Debug, Clone)]
pub struct FsrMixer {
    /// Raw offset bivectors, flat `(max_seq_len, n_blocks, 3)`.
    pub offset_bivec: Vec<f32>,
    /// Raw offset sign logits, flat `(max_seq_len, n_blocks)`.
    pub offset_sign: Vec<f32>,
    /// Source hidden to value-fiber projection.
    pub to_v: LinearLayer,
    /// Mixed fiber to output projection.
    pub out: LinearLayer,
    /// Number of 3-vector blocks.
    pub n_blocks: usize,
    /// Maximum supported sequence length.
    pub max_seq_len: usize,
}

impl FsrMixer {
    /// Construct a mixer with zero offsets/signs and seeded linears.
    ///
    /// # Preconditions
    /// `n_blocks > 0` and `max_seq_len > 0`.
    ///
    /// # Postconditions
    /// Returns a mixer with hidden dimension `3 * n_blocks`.
    ///
    /// # Panics
    /// Panics if either dimension is zero.
    pub fn new(n_blocks: usize, max_seq_len: usize, seed: u64) -> Self {
        assert!(n_blocks > 0);
        assert!(max_seq_len > 0);
        let d = 3 * n_blocks;
        Self {
            offset_bivec: vec![0.0; max_seq_len * n_blocks * 3],
            offset_sign: vec![0.0; max_seq_len * n_blocks],
            to_v: LinearLayer::new(d, d, seed),
            out: LinearLayer::new(d, d, seed.wrapping_add(1)),
            n_blocks,
            max_seq_len,
        }
    }

    /// Gradient-shaped zero mixer.
    ///
    /// # Preconditions
    /// The receiver must have internally consistent parameter shapes.
    ///
    /// # Postconditions
    /// Returns zero-valued buffers with the same shapes as the receiver.
    pub fn zero_grad(&self) -> Self {
        Self {
            offset_bivec: vec![0.0; self.offset_bivec.len()],
            offset_sign: vec![0.0; self.offset_sign.len()],
            to_v: self.to_v.zero_grad(),
            out: self.out.zero_grad(),
            n_blocks: self.n_blocks,
            max_seq_len: self.max_seq_len,
        }
    }

    /// Hidden dimension, equal to `3 * n_blocks`.
    ///
    /// # Preconditions
    /// None.
    ///
    /// # Postconditions
    /// Returns the flat per-token hidden width.
    pub fn hidden_dim(&self) -> usize {
        3 * self.n_blocks
    }

    /// Forward FSR mixer call.
    ///
    /// # Args
    /// - `h`: Flat `(batch, seq_len, 3 * n_blocks)` input hidden states.
    /// - `route`: Selected source indices and fixed gate weights.
    ///
    /// # Preconditions
    /// `seq_len <= max_seq_len`, `h.len() == batch * seq_len * 3 * n_blocks`,
    /// and every selected source index is less than `seq_len`.
    ///
    /// # Postconditions
    /// Returns `(out, cache)` where `out` is flat `(batch, seq_len, hidden_dim)`.
    ///
    /// # Panics
    /// Panics if shapes or source indices violate the preconditions.
    pub fn forward(&self, h: &[f32], route: FsrRoute<'_>) -> (Vec<f32>, FsrMixerCache) {
        assert!(route.seq_len <= self.max_seq_len);
        let d = self.hidden_dim();
        let n_tokens = route.batch * route.seq_len;
        assert_eq!(h.len(), n_tokens * d);
        assert_eq!(
            self.offset_bivec.len(),
            self.max_seq_len * self.n_blocks * 3
        );
        assert_eq!(self.offset_sign.len(), self.max_seq_len * self.n_blocks);

        let to_v_out = linear_forward(&self.to_v, h);
        let n_pairs = route.selected.len();
        let n_pair_blocks = n_pairs * self.n_blocks;
        let mut bivec_pairs = vec![0.0; n_pair_blocks * 3];
        let mut v_pairs = vec![0.0; n_pair_blocks * 3];
        let mut query_of = vec![0u32; n_pairs];
        let mut offset_of = vec![0u32; n_pairs];
        let mut source_of = vec![0u32; n_pairs];

        for batch in 0..route.batch {
            for query in 0..route.seq_len {
                for slot in 0..route.k {
                    let pair = (batch * route.seq_len + query) * route.k + slot;
                    let source = route.selected[pair] as usize;
                    assert!(source < route.seq_len);
                    let offset = query.saturating_sub(source);
                    assert!(offset < self.max_seq_len);
                    query_of[pair] = (batch * route.seq_len + query) as u32;
                    offset_of[pair] = offset as u32;
                    source_of[pair] = (batch * route.seq_len + source) as u32;

                    for block in 0..self.n_blocks {
                        let pair_base = (pair * self.n_blocks + block) * 3;
                        let offset_base = (offset * self.n_blocks + block) * 3;
                        let source_base = (batch * route.seq_len + source) * d + block * 3;
                        bivec_pairs[pair_base..pair_base + 3]
                            .copy_from_slice(&self.offset_bivec[offset_base..offset_base + 3]);
                        v_pairs[pair_base..pair_base + 3]
                            .copy_from_slice(&to_v_out[source_base..source_base + 3]);
                    }
                }
            }
        }

        let (transported, quats) = cayley_rotor_forward(&bivec_pairs, &v_pairs, n_pair_blocks);
        let (mixed, signs) = signed_scatter_forward(
            &transported,
            route.gate,
            &self.offset_sign,
            SignedScatterLanes::new(&query_of, &offset_of),
            SignedScatterLayout::new(n_tokens, self.max_seq_len, self.n_blocks),
        );
        let out = linear_forward(&self.out, &mixed);

        (
            out,
            FsrMixerCache {
                h: h.to_vec(),
                to_v_out,
                bivec_pairs,
                v_pairs,
                quats,
                transported,
                signs,
                mixed,
                query_of,
                offset_of,
                source_of,
                gate: route.gate.to_vec(),
                batch: route.batch,
                seq_len: route.seq_len,
                k: route.k,
                n_blocks: self.n_blocks,
                max_seq_len: self.max_seq_len,
            },
        )
    }

    /// Backward FSR mixer call.
    ///
    /// # Args
    /// - `cache`: Saved cache from the matching forward pass.
    /// - `grad_out`: Flat `(batch, seq_len, hidden_dim)` incoming output grad.
    ///
    /// # Preconditions
    /// `cache` must come from `self.forward` with the same parameters and route
    /// dimensions. `grad_out` must match the forward output shape.
    ///
    /// # Postconditions
    /// Returns gradients for the input hidden states, fixed gate weights, and
    /// all mixer parameters.
    ///
    /// # Panics
    /// Panics if gradient shape does not match the cache dimensions.
    pub fn backward(&self, cache: &FsrMixerCache, grad_out: &[f32]) -> FsrMixerBackward {
        let d = self.hidden_dim();
        let n_tokens = cache.batch * cache.seq_len;
        let n_pairs = cache.gate.len();
        let n_pair_blocks = n_pairs * self.n_blocks;
        assert_eq!(grad_out.len(), n_tokens * d);
        assert_eq!(cache.n_blocks, self.n_blocks);
        assert_eq!(cache.max_seq_len, self.max_seq_len);

        let (grad_mixed, grad_out_layer) = linear_backward(&self.out, &cache.mixed, grad_out);
        let (grad_transported, grad_gate, grad_offset_sign) = signed_scatter_backward(
            &cache.transported,
            &cache.gate,
            &cache.signs,
            SignedScatterLanes::new(&cache.query_of, &cache.offset_of),
            &grad_mixed,
            SignedScatterLayout::new(n_tokens, self.max_seq_len, self.n_blocks),
        );
        let (grad_bivec_pairs, grad_v_pairs) = cayley_rotor_backward(
            &cache.bivec_pairs,
            &cache.v_pairs,
            &cache.quats,
            &grad_transported,
            n_pair_blocks,
        );

        let mut grad_offset_bivec = vec![0.0; self.offset_bivec.len()];
        let mut grad_to_v_out = vec![0.0; cache.to_v_out.len()];
        for pair in 0..n_pairs {
            let offset = cache.offset_of[pair] as usize;
            let source_token = cache.source_of[pair] as usize;
            for block in 0..self.n_blocks {
                let pair_base = (pair * self.n_blocks + block) * 3;
                let offset_base = (offset * self.n_blocks + block) * 3;
                let source_base = source_token * d + block * 3;
                for c in 0..3 {
                    grad_offset_bivec[offset_base + c] += grad_bivec_pairs[pair_base + c];
                    grad_to_v_out[source_base + c] += grad_v_pairs[pair_base + c];
                }
            }
        }

        let (grad_h, grad_to_v) = linear_backward(&self.to_v, &cache.h, &grad_to_v_out);
        FsrMixerBackward {
            grad_h,
            grad_gate,
            grad_params: FsrMixer {
                offset_bivec: grad_offset_bivec,
                offset_sign: grad_offset_sign,
                to_v: grad_to_v,
                out: grad_out_layer,
                n_blocks: self.n_blocks,
                max_seq_len: self.max_seq_len,
            },
        }
    }
}

/// Saved intermediates for `FsrMixer::backward`.
#[derive(Debug, Clone)]
pub struct FsrMixerCache {
    /// Input hidden states, flat `(batch, seq_len, hidden_dim)`.
    pub h: Vec<f32>,
    /// Value projection output, flat `(batch, seq_len, hidden_dim)`.
    pub to_v_out: Vec<f32>,
    /// Per-pair/block copied bivectors, flat `(n_pairs, n_blocks, 3)`.
    pub bivec_pairs: Vec<f32>,
    /// Per-pair/block copied value vectors, flat `(n_pairs, n_blocks, 3)`.
    pub v_pairs: Vec<f32>,
    /// Saved quaternions, flat `(n_pairs, n_blocks, 4)`.
    pub quats: Vec<f32>,
    /// Transported vectors, flat `(n_pairs, n_blocks, 3)`.
    pub transported: Vec<f32>,
    /// Saved tanh signs, flat `(max_seq_len, n_blocks)`.
    pub signs: Vec<f32>,
    /// Mixed vectors before output linear, flat `(batch, seq_len, hidden_dim)`.
    pub mixed: Vec<f32>,
    /// Query token lane for each pair, flat `(n_pairs)`.
    pub query_of: Vec<u32>,
    /// Relative-offset lane for each pair, flat `(n_pairs)`.
    pub offset_of: Vec<u32>,
    /// Source token lane for each pair, flat `(n_pairs)`.
    pub source_of: Vec<u32>,
    /// Saved gate weights, flat `(n_pairs)`.
    pub gate: Vec<f32>,
    /// Batch size.
    pub batch: usize,
    /// Sequence length.
    pub seq_len: usize,
    /// Top-k route width.
    pub k: usize,
    /// Number of 3-vector blocks.
    pub n_blocks: usize,
    /// Maximum sequence length.
    pub max_seq_len: usize,
}

/// Gradients returned by `FsrMixer::backward`.
#[derive(Debug, Clone)]
pub struct FsrMixerBackward {
    /// Gradient with respect to input hidden states.
    pub grad_h: Vec<f32>,
    /// Gradient with respect to fixed gate weights.
    pub grad_gate: Vec<f32>,
    /// Gradients with the same shapes as the mixer parameters.
    pub grad_params: FsrMixer,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn seeded_mixer() -> FsrMixer {
        let mut mixer = FsrMixer::new(1, 3, 7);
        mixer.offset_bivec = vec![0.0, 0.0, 0.0, 0.2, -0.1, 0.3, -0.2, 0.1, 0.25];
        mixer.offset_sign = vec![0.1, -0.2, 0.3];
        mixer
    }

    #[test]
    fn forward_produces_expected_shape() {
        let mixer = seeded_mixer();
        let h = vec![0.2, -0.1, 0.3, 0.4, 0.5, -0.2, -0.3, 0.7, 0.1];
        let selected = vec![0, 0, 1, 0, 1, 2];
        let gate = vec![1.0, 0.0, 0.7, 0.3, 0.2, 0.8];
        let route = FsrRoute::new(&selected, &gate, 1, 3, 2);
        let (out, cache) = mixer.forward(&h, route);
        assert_eq!(out.len(), h.len());
        assert_eq!(cache.query_of, vec![0, 0, 1, 1, 2, 2]);
        assert_eq!(cache.offset_of, vec![0, 0, 0, 1, 1, 0]);
    }

    #[test]
    fn backward_matches_numerical_for_core_parameters() {
        let mixer = seeded_mixer();
        let h = vec![0.2, -0.1, 0.3, 0.4, 0.5, -0.2, -0.3, 0.7, 0.1];
        let selected = vec![0, 0, 1, 0, 1, 2];
        let gate = vec![1.0, 0.0, 0.7, 0.3, 0.2, 0.8];
        let route = FsrRoute::new(&selected, &gate, 1, 3, 2);
        let (out, cache) = mixer.forward(&h, route);
        let grad_out = vec![1.0; out.len()];
        let backward = mixer.backward(&cache, &grad_out);
        let eps = 1e-3;

        for idx in 0..mixer.offset_bivec.len() {
            let mut plus = mixer.clone();
            plus.offset_bivec[idx] += eps;
            let mut minus = mixer.clone();
            minus.offset_bivec[idx] -= eps;
            let loss_plus: f32 = plus.forward(&h, route).0.iter().sum();
            let loss_minus: f32 = minus.forward(&h, route).0.iter().sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (backward.grad_params.offset_bivec[idx] - numeric).abs() < 1e-2,
                "offset_bivec[{idx}]: analytic={} numeric={}",
                backward.grad_params.offset_bivec[idx],
                numeric
            );
        }

        for idx in 0..mixer.offset_sign.len() {
            let mut plus = mixer.clone();
            plus.offset_sign[idx] += eps;
            let mut minus = mixer.clone();
            minus.offset_sign[idx] -= eps;
            let loss_plus: f32 = plus.forward(&h, route).0.iter().sum();
            let loss_minus: f32 = minus.forward(&h, route).0.iter().sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (backward.grad_params.offset_sign[idx] - numeric).abs() < 1e-2,
                "offset_sign[{idx}]: analytic={} numeric={}",
                backward.grad_params.offset_sign[idx],
                numeric
            );
        }

        for idx in 0..h.len() {
            let mut plus = h.clone();
            plus[idx] += eps;
            let mut minus = h.clone();
            minus[idx] -= eps;
            let loss_plus: f32 = mixer.forward(&plus, route).0.iter().sum();
            let loss_minus: f32 = mixer.forward(&minus, route).0.iter().sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!((backward.grad_h[idx] - numeric).abs() < 1e-2);
        }
    }
}
