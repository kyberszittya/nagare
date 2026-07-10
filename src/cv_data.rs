//! Shared CV dataset loading — MNIST IDX and little-endian `raw` splits, plus deterministic
//! random-rotation of a test split. Lifted from `examples/cv_bench` so `cv_bench`, `cv_live`, and
//! the learned-vs-fixed experiment share one loader (§6.1) instead of three copies.

use std::f32::consts::TAU;
use std::path::Path;

use crate::rotate_image;

/// A loaded split: images flat `(n, g*g)` in `[-1,1]`, labels `(n)`, grid side `g`.
pub struct Split {
    /// Pixels, row-major per image, scaled to `[-1, 1]`.
    pub x: Vec<f32>,
    /// Class labels.
    pub y: Vec<usize>,
    /// Image side length (images are square `g×g`).
    pub g: usize,
}

/// MNIST IDX (big-endian 16-byte image header / 8-byte label header).
///
/// # Preconditions
/// `dir/images` and `dir/labels` are IDX3/IDX1 files with square images.
pub fn load_idx(dir: &Path, images: &str, labels: &str, cap: usize) -> Split {
    let b = std::fs::read(dir.join(images)).expect("images");
    let n = (u32::from_be_bytes([b[4], b[5], b[6], b[7]]) as usize).min(cap);
    let g = u32::from_be_bytes([b[8], b[9], b[10], b[11]]) as usize;
    let x = b[16..16 + n * g * g]
        .iter()
        .map(|&p| p as f32 / 255.0 * 2.0 - 1.0)
        .collect();
    let lb = std::fs::read(dir.join(labels)).expect("labels");
    let y = lb[8..8 + n].iter().map(|&l| l as usize).collect();
    Split { x, y, g }
}

/// Little-endian raw: `n,h,w:u32` then `n*h*w` u8; labels `n:u32` then `n` u8.
///
/// # Preconditions
/// Images are square (`h == w`); asserted.
pub fn load_raw(dir: &Path, images: &str, labels: &str, cap: usize) -> Split {
    let b = std::fs::read(dir.join(images)).expect("images");
    let rd = |o: usize| u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]]) as usize;
    let (n0, h, w) = (rd(0), rd(4), rd(8));
    let (n, g) = (n0.min(cap), h);
    assert_eq!(h, w, "expected square images");
    let x = b[12..12 + n * g * g]
        .iter()
        .map(|&p| p as f32 / 255.0 * 2.0 - 1.0)
        .collect();
    let lb = std::fs::read(dir.join(labels)).expect("labels");
    let y = lb[4..4 + n].iter().map(|&l| l as usize).collect();
    Split { x, y, g }
}

/// Load a `{train|test}` split for `dataset` (`"mnist"` → IDX, else → `raw`) from `dir`, capped at
/// `cap` images.
pub fn load_split(dataset: &str, dir: &Path, train: bool, cap: usize) -> Split {
    match (dataset, train) {
        ("mnist", true) => load_idx(
            dir,
            "train-images-idx3-ubyte",
            "train-labels-idx1-ubyte",
            cap,
        ),
        ("mnist", false) => load_idx(dir, "t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte", cap),
        (_, true) => load_raw(dir, "train-images.bin", "train-labels.bin", cap),
        (_, false) => load_raw(dir, "test-images.bin", "test-labels.bin", cap),
    }
}

/// Randomly-rotated copy of a split's images (deterministic per-image angle, edge-clamp rotation).
///
/// # Preconditions
/// `x.len() == n * g * g`.
pub fn rot_all(x: &[f32], n: usize, g: usize) -> Vec<f32> {
    assert_eq!(x.len(), n * g * g);
    let mut out = vec![0.0f32; n * g * g];
    let mut st = 0x2545_f491_4f6c_dd1du64;
    for s in 0..n {
        st = st
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        let theta = (st >> 40) as f32 / (1u64 << 24) as f32 * TAU;
        out[s * g * g..(s + 1) * g * g].copy_from_slice(&rotate_image(
            &x[s * g * g..(s + 1) * g * g],
            g,
            theta,
        ));
    }
    out
}

/// Per-feature mean and standard deviation (`+1e-6`) over `f` viewed as `(_, dim)` rows.
///
/// # Preconditions
/// `f.len()` is a multiple of `dim`, `dim >= 1`.
pub fn feature_stats(f: &[f32], dim: usize) -> (Vec<f32>, Vec<f32>) {
    let n = (f.len() / dim).max(1);
    let (mut mu, mut sd) = (vec![0.0f32; dim], vec![0.0f32; dim]);
    for r in f.chunks(dim) {
        for j in 0..dim {
            mu[j] += r[j] / n as f32;
        }
    }
    for r in f.chunks(dim) {
        for j in 0..dim {
            sd[j] += (r[j] - mu[j]).powi(2) / n as f32;
        }
    }
    for s in &mut sd {
        *s = s.sqrt() + 1e-6;
    }
    (mu, sd)
}

/// Standardize `buf` (viewed as `(_, dim)` rows) in place with a fixed `(mu, sd)` preconditioner.
pub fn standardize_with(buf: &mut [f32], mu: &[f32], sd: &[f32], dim: usize) {
    for r in buf.chunks_mut(dim) {
        for j in 0..dim {
            r[j] = (r[j] - mu[j]) / sd[j];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feature_stats_then_standardize_is_zero_mean_unit_var() {
        let dim = 3;
        let f: Vec<f32> = (0..30)
            .map(|i| (i as f32 * 0.4).sin() + i as f32 * 0.01)
            .collect();
        let (mu, sd) = feature_stats(&f, dim);
        let mut g = f.clone();
        standardize_with(&mut g, &mu, &sd, dim);
        let (mu2, sd2) = feature_stats(&g, dim);
        for j in 0..dim {
            assert!(mu2[j].abs() < 1e-4, "col {j} mean {}", mu2[j]);
            assert!((sd2[j] - 1.0).abs() < 1e-3, "col {j} sd {}", sd2[j]);
        }
    }

    #[test]
    fn rot_all_preserves_shape_and_is_deterministic() {
        let (n, g) = (3, 6);
        let x: Vec<f32> = (0..n * g * g).map(|i| (i as f32 * 0.3).sin()).collect();
        let a = rot_all(&x, n, g);
        let b = rot_all(&x, n, g);
        assert_eq!(a.len(), n * g * g);
        assert_eq!(a, b, "rotation must be deterministic across calls");
    }
}
