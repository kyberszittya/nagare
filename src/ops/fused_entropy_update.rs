//! Fused global-pool entropy update.
//!
//! This operator computes the update linear for rows that conceptually have
//! input `[h_row, pooled_batch, entropy_batch]`, without materialising that
//! wide broadcast tensor.

use rayon::prelude::*;

use crate::ops::linear::LinearLayer;

/// Shape for a fused entropy-update call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FusedEntropyUpdateShape {
    /// Number of independent point sets.
    pub batch: usize,
    /// Number of points per set.
    pub points: usize,
    /// Hidden feature channels per point.
    pub hidden: usize,
}

/// Backward result for [`fused_entropy_update_backward`].
#[derive(Debug, Clone)]
pub struct FusedEntropyUpdateBackward {
    /// Gradient with respect to local hidden rows, shape `(batch, points, hidden)`.
    pub grad_h: Vec<f32>,
    /// Gradient with respect to pooled context rows, shape `(batch, 3 * hidden)`.
    pub grad_pooled: Vec<f32>,
    /// Gradient with respect to entropy scalars, shape `(batch,)`.
    pub grad_entropy: Vec<f32>,
    /// Gradient with respect to the fused linear layer.
    pub grad_layer: LinearLayer,
}

/// Fused update forward.
///
/// `layer.in_dim` must equal `4 * hidden + 1`. The implicit row layout is
/// `[h[hidden], pooled[3 * hidden], entropy[1]]`.
pub fn fused_entropy_update_forward(
    layer: &LinearLayer,
    h: &[f32],
    pooled: &[f32],
    entropy: &[f32],
    shape: FusedEntropyUpdateShape,
) -> Vec<f32> {
    let FusedEntropyUpdateShape {
        batch,
        points,
        hidden,
    } = shape;
    assert_eq!(layer.in_dim, 4 * hidden + 1);
    assert_eq!(h.len(), batch * points * hidden);
    assert_eq!(pooled.len(), batch * 3 * hidden);
    assert_eq!(entropy.len(), batch);
    let mut out = vec![0.0; batch * points * layer.out_dim];
    // Parallel over rows like `linear_forward` (each output row is
    // independent, so the per-row accumulation order — and hence the
    // result — is bit-identical to the serial loop).
    let out_dim = layer.out_dim;
    out.par_chunks_mut(out_dim)
        .enumerate()
        .for_each(|(row, out_row)| {
            let b = row / points;
            let pooled_row = &pooled[b * 3 * hidden..(b + 1) * 3 * hidden];
            let ent = entropy[b];
            let h_row = &h[row * hidden..(row + 1) * hidden];
            // `ikj` (SAXPY) accumulation: for each input i, broadcast x[i] and add a
            // *contiguous* W-row into the *contiguous* out_row. The inner j-loop is
            // element-wise into distinct slots (no reduction), so it autovectorizes.
            // For a fixed j the additions still run in i-order, so the result is
            // bit-identical to the scalar-accumulate form.
            //
            // Row layout of the implicit input: [h(hidden) | pooled(3*hidden) | ent(1)].
            out_row.copy_from_slice(&layer.b);
            let mut w_base = 0usize;
            for &v in h_row {
                saxpy(out_row, &layer.w[w_base..w_base + out_dim], v);
                w_base += out_dim;
            }
            for &v in pooled_row {
                saxpy(out_row, &layer.w[w_base..w_base + out_dim], v);
                w_base += out_dim;
            }
            saxpy(out_row, &layer.w[w_base..w_base + out_dim], ent);
        });
    out
}

/// `y += a * x` over contiguous slices (autovectorizes to a broadcast-FMA loop).
#[inline(always)]
fn saxpy(y: &mut [f32], x: &[f32], a: f32) {
    for (yi, &xi) in y.iter_mut().zip(x.iter()) {
        *yi += a * xi;
    }
}

/// Fused update backward.
///
/// Returns gradients for each non-materialised input part and for `layer`.
pub fn fused_entropy_update_backward(
    layer: &LinearLayer,
    h: &[f32],
    pooled: &[f32],
    entropy: &[f32],
    grad_y: &[f32],
    shape: FusedEntropyUpdateShape,
) -> FusedEntropyUpdateBackward {
    let FusedEntropyUpdateShape {
        batch,
        points,
        hidden,
    } = shape;
    assert_eq!(layer.in_dim, 4 * hidden + 1);
    assert_eq!(h.len(), batch * points * hidden);
    assert_eq!(pooled.len(), batch * 3 * hidden);
    assert_eq!(entropy.len(), batch);
    assert_eq!(grad_y.len(), batch * points * layer.out_dim);
    let mut grad_h = vec![0.0; h.len()];
    let mut grad_pooled = vec![0.0; pooled.len()];
    let mut grad_entropy = vec![0.0; batch];
    let mut grad_w = vec![0.0; layer.w.len()];
    let mut grad_b = vec![0.0; layer.b.len()];
    for b in 0..batch {
        let pooled_row = &pooled[b * 3 * hidden..(b + 1) * 3 * hidden];
        let ent = entropy[b];
        for p in 0..points {
            let row = b * points + p;
            let h_row = &h[row * hidden..(row + 1) * hidden];
            let gy_row = &grad_y[row * layer.out_dim..(row + 1) * layer.out_dim];
            for (j, &gy) in gy_row.iter().enumerate() {
                grad_b[j] += gy;
                for (i, &v) in h_row.iter().enumerate() {
                    grad_h[row * hidden + i] += gy * layer.w[i * layer.out_dim + j];
                    grad_w[i * layer.out_dim + j] += v * gy;
                }
                for (i, &v) in pooled_row.iter().enumerate() {
                    let w_idx = (hidden + i) * layer.out_dim + j;
                    grad_pooled[b * 3 * hidden + i] += gy * layer.w[w_idx];
                    grad_w[w_idx] += v * gy;
                }
                let ent_w_idx = (4 * hidden) * layer.out_dim + j;
                grad_entropy[b] += gy * layer.w[ent_w_idx];
                grad_w[ent_w_idx] += ent * gy;
            }
        }
    }
    FusedEntropyUpdateBackward {
        grad_h,
        grad_pooled,
        grad_entropy,
        grad_layer: LinearLayer {
            w: grad_w,
            b: grad_b,
            in_dim: layer.in_dim,
            out_dim: layer.out_dim,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::linear::{linear_backward, linear_forward};

    fn materialize(
        h: &[f32],
        pooled: &[f32],
        entropy: &[f32],
        batch: usize,
        points: usize,
        hidden: usize,
    ) -> Vec<f32> {
        let in_dim = 4 * hidden + 1;
        let mut out = vec![0.0; batch * points * in_dim];
        for b in 0..batch {
            for p in 0..points {
                let dst = (b * points + p) * in_dim;
                let src_h = (b * points + p) * hidden;
                out[dst..dst + hidden].copy_from_slice(&h[src_h..src_h + hidden]);
                out[dst + hidden..dst + 4 * hidden]
                    .copy_from_slice(&pooled[b * 3 * hidden..(b + 1) * 3 * hidden]);
                out[dst + 4 * hidden] = entropy[b];
            }
        }
        out
    }

    #[test]
    fn fused_forward_matches_materialized_linear() {
        let batch = 2;
        let points = 3;
        let hidden = 4;
        let layer = LinearLayer::new(4 * hidden + 1, hidden, 7);
        let h: Vec<f32> = (0..batch * points * hidden)
            .map(|i| i as f32 * 0.03 - 0.2)
            .collect();
        let pooled: Vec<f32> = (0..batch * 3 * hidden)
            .map(|i| i as f32 * -0.02 + 0.4)
            .collect();
        let entropy = vec![0.25, 0.75];
        let shape = FusedEntropyUpdateShape {
            batch,
            points,
            hidden,
        };
        let fused = fused_entropy_update_forward(&layer, &h, &pooled, &entropy, shape);
        let x = materialize(&h, &pooled, &entropy, batch, points, hidden);
        let reference = linear_forward(&layer, &x);
        for (a, b) in fused.iter().zip(reference.iter()) {
            assert!((a - b).abs() < 1e-6, "a={a} b={b}");
        }
    }

    #[test]
    fn fused_backward_matches_materialized_linear_backward() {
        let batch = 2;
        let points = 2;
        let hidden = 3;
        let layer = LinearLayer::new(4 * hidden + 1, hidden, 11);
        let h: Vec<f32> = (0..batch * points * hidden)
            .map(|i| i as f32 * 0.04 - 0.3)
            .collect();
        let pooled: Vec<f32> = (0..batch * 3 * hidden)
            .map(|i| i as f32 * 0.01 + 0.1)
            .collect();
        let entropy = vec![0.2, 0.8];
        let grad_y: Vec<f32> = (0..batch * points * hidden)
            .map(|i| i as f32 * -0.05 + 0.6)
            .collect();
        let shape = FusedEntropyUpdateShape {
            batch,
            points,
            hidden,
        };
        let fused = fused_entropy_update_backward(&layer, &h, &pooled, &entropy, &grad_y, shape);
        let x = materialize(&h, &pooled, &entropy, batch, points, hidden);
        let (grad_x, grad_layer) = linear_backward(&layer, &x, &grad_y);
        for b in 0..batch {
            for p in 0..points {
                let src = (b * points + p) * (4 * hidden + 1);
                let dst = (b * points + p) * hidden;
                for i in 0..hidden {
                    assert!((fused.grad_h[dst + i] - grad_x[src + i]).abs() < 1e-6);
                }
                for i in 0..3 * hidden {
                    assert!(
                        (fused.grad_pooled[b * 3 * hidden + i]
                            - (0..points)
                                .map(|pp| grad_x[(b * points + pp) * (4 * hidden + 1) + hidden + i])
                                .sum::<f32>())
                        .abs()
                            < 1e-6
                    );
                }
                assert!(
                    (fused.grad_entropy[b]
                        - (0..points)
                            .map(|pp| grad_x[(b * points + pp) * (4 * hidden + 1) + 4 * hidden])
                            .sum::<f32>())
                    .abs()
                        < 1e-6
                );
            }
        }
        for (a, b) in fused.grad_layer.w.iter().zip(grad_layer.w.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
        for (a, b) in fused.grad_layer.b.iter().zip(grad_layer.b.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }
}
