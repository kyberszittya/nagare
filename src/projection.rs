use crate::{
    LOCAL_FEATURES, PROJECTION_ALPHA, PROJECTION_RANK, STRUCTURAL_FEATURES, VERTEX_FEATURES,
};

pub fn project_onto_holonomy_subspace(
    phi: &mut [f32],
    basis: &[[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK],
) {
    assert_eq!(phi.len(), LOCAL_FEATURES);
    let mut projected = [0.0f32; STRUCTURAL_FEATURES];
    for axis in basis {
        let norm2 = axis.iter().map(|v| v * v).sum::<f32>();
        if norm2 <= 1.0e-12 {
            continue;
        }
        let dot = phi[..STRUCTURAL_FEATURES]
            .iter()
            .zip(axis.iter())
            .map(|(&a, &b)| a * b)
            .sum::<f32>();
        let scale = dot / norm2;
        for (dst, &axis_value) in projected.iter_mut().zip(axis.iter()) {
            *dst += scale * axis_value;
        }
    }
    for i in 0..STRUCTURAL_FEATURES {
        phi[i] = PROJECTION_ALPHA * projected[i] + (1.0 - PROJECTION_ALPHA) * phi[i];
    }
}

pub fn learn_holonomy_projection_basis(
    structural: &[f32],
    labels: &[usize],
) -> [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK] {
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
    let mut candidates = [[0.0f32; STRUCTURAL_FEATURES]; PROJECTION_RANK + 2];
    for i in 0..STRUCTURAL_FEATURES {
        candidates[0][i] = class_sum[1][i] - class_sum[0][i];
        candidates[1][i] = 0.5 * (class_sum[0][i] + class_sum[1][i]);
    }
    let class_delta = candidates[0];
    copy_channel_group(&mut candidates[2], &class_delta, &[0, 1, 2]);
    copy_channel_group(&mut candidates[3], &class_delta, &[3, 4]);
    copy_channel_group(&mut candidates[4], &class_delta, &[5, 6]);
    for channel in 0..VERTEX_FEATURES {
        candidates[5][VERTEX_FEATURES + channel] =
            class_sum[0][VERTEX_FEATURES + channel] + class_sum[1][VERTEX_FEATURES + channel];
        candidates[5][3 * VERTEX_FEATURES + channel] = class_sum[0][3 * VERTEX_FEATURES + channel]
            + class_sum[1][3 * VERTEX_FEATURES + channel];
    }
    let defaults = default_holonomy_projection_basis();
    candidates[6] = defaults[2];
    candidates[7] = defaults[4];
    orthonormalize_candidates(&candidates)
}

pub fn default_holonomy_projection_basis() -> [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK] {
    let mut candidates = [[0.0f32; STRUCTURAL_FEATURES]; PROJECTION_RANK];
    copy_channel_group(&mut candidates[0], &[1.0; STRUCTURAL_FEATURES], &[0, 1, 2]);
    copy_channel_group(&mut candidates[1], &[1.0; STRUCTURAL_FEATURES], &[3, 4]);
    copy_channel_group(&mut candidates[2], &[1.0; STRUCTURAL_FEATURES], &[5, 6]);
    for channel in 0..VERTEX_FEATURES {
        candidates[3][VERTEX_FEATURES + channel] = 1.0;
        candidates[4][2 * VERTEX_FEATURES + channel] = 1.0;
        candidates[5][3 * VERTEX_FEATURES + channel] = 1.0;
    }
    orthonormalize_candidates(&candidates)
}

fn copy_channel_group(dst: &mut [f32; STRUCTURAL_FEATURES], src: &[f32], channels: &[usize]) {
    for &channel in channels {
        dst[channel] = src[channel];
        dst[VERTEX_FEATURES + channel] = src[VERTEX_FEATURES + channel];
        dst[2 * VERTEX_FEATURES + channel] = src[2 * VERTEX_FEATURES + channel];
        dst[3 * VERTEX_FEATURES + channel] = src[3 * VERTEX_FEATURES + channel];
    }
}

fn orthonormalize_candidates<const N: usize>(
    candidates: &[[f32; STRUCTURAL_FEATURES]; N],
) -> [[f32; STRUCTURAL_FEATURES]; PROJECTION_RANK] {
    let mut basis = [[0.0f32; STRUCTURAL_FEATURES]; PROJECTION_RANK];
    let mut rank = 0usize;
    for candidate in candidates {
        if rank == PROJECTION_RANK {
            break;
        }
        let mut vector = *candidate;
        for _ in 0..2 {
            for axis in basis.iter().take(rank) {
                remove_axis_component(&mut vector, axis);
            }
        }
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm <= 1.0e-7 {
            continue;
        }
        for (dst, value) in basis[rank].iter_mut().zip(vector.iter()) {
            *dst = *value / norm;
        }
        rank += 1;
    }
    basis
}

fn remove_axis_component(vector: &mut [f32; STRUCTURAL_FEATURES], axis: &[f32]) {
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
