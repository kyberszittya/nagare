//! Holonomy-informed projection bases for the local learner's gate.
//!
//! A [`ProjectionBasis`] is a rank-limited orthonormal set of directions in
//! the pooled structural feature space. Two constructors are provided:
//! the fixed [`default_holonomy_basis`] built from the channel-group
//! structure of the quaternion lift, and the fitted
//! [`fit_class_mean_basis`] built from class means of training data
//! (the "fitted projection gate" of
//! `reports/2026-07-02-fitted-projection-gate-holonomy-ablation.md`).
//! Application goes through the generic
//! [`crate::ops::project_alpha_mix`] kernel.

use crate::features::{GEOMETRY_CHANNELS, HOLONOMY_CHANNELS, ROTOR_CHANNELS};
use crate::ops::project_alpha_mix::{project_alpha_mix_forward, ProjectAlphaMixShape};
use crate::{PROJECTION_RANK, STRUCTURAL_FEATURES, VERTEX_FEATURES};

/// A rank-limited orthonormalised projection basis (rows may be zero when
/// fewer than `rank` independent candidates survive).
#[derive(Clone, Debug)]
pub struct ProjectionBasis {
    dim: usize,
    rank: usize,
    vectors: Vec<f32>,
}

impl ProjectionBasis {
    /// Build a basis by (twice-through) Gram–Schmidt over candidate rows.
    ///
    /// # Preconditions
    /// * `candidates.len()` is a multiple of `dim`.
    ///
    /// # Postconditions
    /// * Result holds exactly `rank` rows of width `dim`; accepted rows are
    ///   unit-norm and mutually orthogonal (within f32); surplus rows are
    ///   zero. Candidates whose residual norm is `<= 1e-7` are dropped.
    pub fn orthonormalize(candidates: &[f32], dim: usize, rank: usize) -> Self {
        assert!(dim > 0, "dim must be positive");
        assert_eq!(
            candidates.len() % dim,
            0,
            "candidates must be rows of width dim"
        );
        let mut vectors = vec![0.0f32; rank * dim];
        let mut accepted = 0usize;
        for candidate in candidates.chunks(dim) {
            if accepted == rank {
                break;
            }
            let mut vector = candidate.to_vec();
            for _ in 0..2 {
                for axis in vectors.chunks(dim).take(accepted) {
                    remove_axis_component(&mut vector, axis);
                }
            }
            let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm <= 1.0e-7 {
                continue;
            }
            for (dst, value) in vectors[accepted * dim..(accepted + 1) * dim]
                .iter_mut()
                .zip(vector.iter())
            {
                *dst = *value / norm;
            }
            accepted += 1;
        }
        Self { dim, rank, vectors }
    }

    /// Feature dimensionality of each row.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Number of rows (including possible zero rows).
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Row-major basis storage, `rank * dim`.
    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// Apply the alpha-mixed projection to `x` in place.
    ///
    /// # Preconditions
    /// * `x.len() == self.dim()`.
    pub fn apply_alpha_mix(&self, x: &mut [f32], alpha: f32) {
        assert_eq!(x.len(), self.dim);
        let shape = ProjectAlphaMixShape {
            dim: self.dim,
            rank: self.rank,
        };
        let mixed = project_alpha_mix_forward(x, &self.vectors, alpha, shape);
        x.copy_from_slice(&mixed);
    }
}

/// The fixed channel-group basis over the pooled structural features:
/// geometry / rotor / holonomy groups across all pooled statistics, plus
/// the std, max, and sign-entropy statistic blocks.
pub fn default_holonomy_basis() -> ProjectionBasis {
    let mut candidates = vec![0.0f32; PROJECTION_RANK * STRUCTURAL_FEATURES];
    let ones = [1.0f32; STRUCTURAL_FEATURES];
    copy_channel_group(row_mut(&mut candidates, 0), &ones, &GEOMETRY_CHANNELS);
    copy_channel_group(row_mut(&mut candidates, 1), &ones, &ROTOR_CHANNELS);
    copy_channel_group(row_mut(&mut candidates, 2), &ones, &HOLONOMY_CHANNELS);
    for channel in 0..VERTEX_FEATURES {
        candidates[3 * STRUCTURAL_FEATURES + VERTEX_FEATURES + channel] = 1.0;
        candidates[4 * STRUCTURAL_FEATURES + 2 * VERTEX_FEATURES + channel] = 1.0;
        candidates[5 * STRUCTURAL_FEATURES + 3 * VERTEX_FEATURES + channel] = 1.0;
    }
    ProjectionBasis::orthonormalize(&candidates, STRUCTURAL_FEATURES, PROJECTION_RANK)
}

/// Fit a projection basis from per-class means of pooled training features
/// (class delta, class mean, per-group deltas, spread/sign-entropy sums,
/// and two default fallback axes).
///
/// # Preconditions
/// * `structural.len() == labels.len() * STRUCTURAL_FEATURES`.
pub fn fit_class_mean_basis(structural: &[f32], labels: &[usize]) -> ProjectionBasis {
    assert_eq!(structural.len(), labels.len() * STRUCTURAL_FEATURES);
    let mut class_sum = [[0.0f32; STRUCTURAL_FEATURES]; 2];
    let mut class_count = [0usize; 2];
    for (sample, &label) in labels.iter().enumerate() {
        let label = label.min(1);
        class_count[label] += 1;
        let base = sample * STRUCTURAL_FEATURES;
        for i in 0..STRUCTURAL_FEATURES {
            class_sum[label][i] += structural[base + i];
        }
    }
    for label in 0..2 {
        let scale = 1.0 / class_count[label].max(1) as f32;
        for value in &mut class_sum[label] {
            *value *= scale;
        }
    }
    let n_candidates = PROJECTION_RANK + 2;
    let mut candidates = vec![0.0f32; n_candidates * STRUCTURAL_FEATURES];
    for i in 0..STRUCTURAL_FEATURES {
        candidates[i] = class_sum[1][i] - class_sum[0][i];
        candidates[STRUCTURAL_FEATURES + i] = 0.5 * (class_sum[0][i] + class_sum[1][i]);
    }
    let class_delta: Vec<f32> = candidates[..STRUCTURAL_FEATURES].to_vec();
    copy_channel_group(
        row_mut(&mut candidates, 2),
        &class_delta,
        &GEOMETRY_CHANNELS,
    );
    copy_channel_group(row_mut(&mut candidates, 3), &class_delta, &ROTOR_CHANNELS);
    copy_channel_group(
        row_mut(&mut candidates, 4),
        &class_delta,
        &HOLONOMY_CHANNELS,
    );
    for channel in 0..VERTEX_FEATURES {
        candidates[5 * STRUCTURAL_FEATURES + VERTEX_FEATURES + channel] =
            class_sum[0][VERTEX_FEATURES + channel] + class_sum[1][VERTEX_FEATURES + channel];
        candidates[5 * STRUCTURAL_FEATURES + 3 * VERTEX_FEATURES + channel] = class_sum[0]
            [3 * VERTEX_FEATURES + channel]
            + class_sum[1][3 * VERTEX_FEATURES + channel];
    }
    let defaults = default_holonomy_basis();
    candidates[6 * STRUCTURAL_FEATURES..7 * STRUCTURAL_FEATURES]
        .copy_from_slice(&defaults.vectors()[2 * STRUCTURAL_FEATURES..3 * STRUCTURAL_FEATURES]);
    candidates[7 * STRUCTURAL_FEATURES..8 * STRUCTURAL_FEATURES]
        .copy_from_slice(&defaults.vectors()[4 * STRUCTURAL_FEATURES..5 * STRUCTURAL_FEATURES]);
    ProjectionBasis::orthonormalize(&candidates, STRUCTURAL_FEATURES, PROJECTION_RANK)
}

fn row_mut(candidates: &mut [f32], row: usize) -> &mut [f32] {
    &mut candidates[row * STRUCTURAL_FEATURES..(row + 1) * STRUCTURAL_FEATURES]
}

fn copy_channel_group(dst: &mut [f32], src: &[f32], channels: &[usize]) {
    for &channel in channels {
        dst[channel] = src[channel];
        dst[VERTEX_FEATURES + channel] = src[VERTEX_FEATURES + channel];
        dst[2 * VERTEX_FEATURES + channel] = src[2 * VERTEX_FEATURES + channel];
        dst[3 * VERTEX_FEATURES + channel] = src[3 * VERTEX_FEATURES + channel];
    }
}

fn remove_axis_component(vector: &mut [f32], axis: &[f32]) {
    let norm2 = axis.iter().map(|v| v * v).sum::<f32>();
    if norm2 <= 1.0e-12 {
        return;
    }
    let dot = vector
        .iter()
        .zip(axis.iter())
        .map(|(&a, &b)| a * b)
        .sum::<f32>();
    let scale = dot / norm2;
    for (value, &axis_value) in vector.iter_mut().zip(axis.iter()) {
        *value -= scale * axis_value;
    }
}
