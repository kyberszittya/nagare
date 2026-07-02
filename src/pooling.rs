use crate::{
    datasets::Dataset, features::quaternion_periodic_features, metrics::binary_entropy,
    STRUCTURAL_FEATURES, VERTEX_FEATURES,
};

/// Global mean/std/max/sign-entropy pooling over quaternion periodic features.
pub fn structural_pool_features(data: &Dataset) -> Vec<f32> {
    let lifted = quaternion_periodic_features(&data.x, data.samples, data.points);
    let mut out = vec![0.0; data.samples * STRUCTURAL_FEATURES];
    let inv_points = 1.0 / data.points as f32;
    for sample in 0..data.samples {
        for channel in 0..VERTEX_FEATURES {
            let mut sum = 0.0;
            let mut max_value = f32::NEG_INFINITY;
            let mut positive = 0usize;
            for point in 0..data.points {
                let value = lifted[(sample * data.points + point) * VERTEX_FEATURES + channel];
                sum += value;
                max_value = max_value.max(value);
                positive += usize::from(value >= 0.0);
            }
            let mean = sum * inv_points;
            let mut var = 0.0;
            for point in 0..data.points {
                let value = lifted[(sample * data.points + point) * VERTEX_FEATURES + channel];
                let d = value - mean;
                var += d * d;
            }
            let pos = positive as f32 * inv_points;
            let base = sample * STRUCTURAL_FEATURES;
            out[base + channel] = mean;
            out[base + VERTEX_FEATURES + channel] = (var * inv_points + 1.0e-6).sqrt();
            out[base + 2 * VERTEX_FEATURES + channel] = max_value;
            out[base + 3 * VERTEX_FEATURES + channel] = binary_entropy(pos);
        }
    }
    out
}
