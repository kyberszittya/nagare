use std::str::FromStr;

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Task {
    Moons,
    Spiral,
    Xor,
}

impl Task {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Moons => "moons",
            Self::Spiral => "spiral",
            Self::Xor => "xor",
        }
    }
}

impl FromStr for Task {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "moons" => Ok(Self::Moons),
            "spiral" => Ok(Self::Spiral),
            "xor" => Ok(Self::Xor),
            other => Err(format!("unknown task '{other}'")),
        }
    }
}

#[derive(Clone, Debug)]
pub struct Dataset {
    pub x: Vec<f32>,
    pub y: Vec<usize>,
    pub samples: usize,
    pub points: usize,
}

pub fn make_dataset(task: Task, samples: usize, points: usize, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut x = Vec::with_capacity(samples * points * 2);
    let mut y = Vec::with_capacity(samples);
    for sample in 0..samples {
        let label = sample % 2;
        y.push(label);
        match task {
            Task::Moons => sample_moons(label, points, &mut rng, &mut x),
            Task::Spiral => sample_spiral(label, points, &mut rng, &mut x),
            Task::Xor => sample_xor(label, points, &mut rng, &mut x),
        }
    }
    Dataset {
        x,
        y,
        samples,
        points,
    }
}

pub fn corrupt_dataset(data: &Dataset, noise_std: f32, missing_rate: f32, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = data.clone();
    for sample in 0..out.samples {
        for point in 0..out.points {
            let base = (sample * out.points + point) * 2;
            if rng.random::<f32>() < missing_rate {
                out.x[base] = 0.0;
                out.x[base + 1] = 0.0;
            } else if noise_std > 0.0 {
                out.x[base] += normalish(&mut rng) * noise_std;
                out.x[base + 1] += normalish(&mut rng) * noise_std;
            }
        }
    }
    out
}

/// Independently permute the point order within each sample.
///
/// This is the holonomy-isolating knife of the order-shuffle ablation: the
/// per-point geometry and rotor channels are permutation-invariant once
/// pooled, so shuffling perturbs only the order-sensitive holonomy channels
/// (the running quaternion product) while leaving the point cloud of every
/// sample unchanged.
///
/// # Postconditions
/// * Shape and labels are preserved; each sample's multiset of points is
///   unchanged (a genuine per-sample permutation). Deterministic in `seed`.
pub fn shuffle_point_order(data: &Dataset, seed: u64) -> Dataset {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = data.clone();
    let mut order: Vec<usize> = (0..data.points).collect();
    for sample in 0..data.samples {
        order.shuffle(&mut rng);
        let sample_base = sample * data.points * 2;
        for (dst_point, &src_point) in order.iter().enumerate() {
            let dst = sample_base + dst_point * 2;
            let src = sample_base + src_point * 2;
            out.x[dst] = data.x[src];
            out.x[dst + 1] = data.x[src + 1];
        }
    }
    out
}

/// Gather a minibatch of point sets by sample indices.
///
/// # Preconditions
/// * Every index in `indices` is `< data.samples`.
///
/// # Postconditions
/// * Returns `(x, y)` with `x.len() == indices.len() * data.points * 2` and
///   `y.len() == indices.len()`, in the order given by `indices`.
pub fn gather_batch(data: &Dataset, indices: &[usize]) -> (Vec<f32>, Vec<usize>) {
    let row = data.points * 2;
    let mut x = Vec::with_capacity(indices.len() * row);
    let mut y = Vec::with_capacity(indices.len());
    for &idx in indices {
        x.extend_from_slice(&data.x[idx * row..(idx + 1) * row]);
        y.push(data.y[idx]);
    }
    (x, y)
}

fn sample_moons(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    for _ in 0..points {
        let theta = rng.random::<f32>() * std::f32::consts::PI;
        let (x, y) = if label == 0 {
            (theta.cos(), theta.sin())
        } else {
            (1.0 - theta.cos(), 0.45 - theta.sin())
        };
        out.push(x + normalish(rng) * 0.055);
        out.push(y + normalish(rng) * 0.055);
    }
}

fn sample_spiral(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    for point in 0..points {
        let t =
            point as f32 / points as f32 * 3.5 * std::f32::consts::PI + rng.random::<f32>() * 0.15;
        let phase = if label == 0 {
            0.0
        } else {
            std::f32::consts::PI
        };
        let radius = 0.12 + 0.08 * t;
        out.push(radius * (t + phase).cos() + normalish(rng) * 0.04);
        out.push(radius * (t + phase).sin() + normalish(rng) * 0.04);
    }
}

fn sample_xor(label: usize, points: usize, rng: &mut StdRng, out: &mut Vec<f32>) {
    let centers = if label == 0 {
        [(-0.75, -0.75), (0.75, 0.75)]
    } else {
        [(-0.75, 0.75), (0.75, -0.75)]
    };
    for _ in 0..points {
        let (cx, cy) = centers[rng.random_range(0..2)];
        out.push(cx + normalish(rng) * 0.12);
        out.push(cy + normalish(rng) * 0.12);
    }
}

pub(crate) fn normalish(rng: &mut StdRng) -> f32 {
    let mut sum = 0.0;
    for _ in 0..6 {
        sum += rng.random::<f32>() - 0.5;
    }
    sum * 0.816_496_6
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sorted_sample_points(data: &Dataset, sample: usize) -> Vec<[u32; 2]> {
        let base = sample * data.points * 2;
        let mut pts: Vec<[u32; 2]> = (0..data.points)
            .map(|p| {
                [
                    data.x[base + p * 2].to_bits(),
                    data.x[base + p * 2 + 1].to_bits(),
                ]
            })
            .collect();
        pts.sort_unstable();
        pts
    }

    #[test]
    fn shuffle_preserves_shape_labels_and_multiset() {
        let data = make_dataset(Task::Spiral, 16, 24, 7);
        let shuffled = shuffle_point_order(&data, 99);
        assert_eq!(shuffled.samples, data.samples);
        assert_eq!(shuffled.points, data.points);
        assert_eq!(shuffled.x.len(), data.x.len());
        assert_eq!(shuffled.y, data.y, "labels must be untouched");
        for sample in 0..data.samples {
            assert_eq!(
                sorted_sample_points(&shuffled, sample),
                sorted_sample_points(&data, sample),
                "sample {sample} must be a permutation of the same points"
            );
        }
    }

    #[test]
    fn shuffle_is_deterministic_in_seed_and_actually_reorders() {
        let data = make_dataset(Task::Moons, 8, 32, 3);
        let a = shuffle_point_order(&data, 41);
        let b = shuffle_point_order(&data, 41);
        assert_eq!(a.x, b.x, "same seed must give identical output");
        let c = shuffle_point_order(&data, 42);
        assert_ne!(a.x, c.x, "different seed must reorder differently");
        // With 32 points a permutation almost surely moves at least one point.
        assert_ne!(a.x, data.x, "shuffle must actually reorder");
    }

    #[test]
    fn gather_batch_selects_rows_in_order() {
        let data = make_dataset(Task::Xor, 6, 4, 1);
        let (x, y) = gather_batch(&data, &[2, 0]);
        let row = data.points * 2;
        assert_eq!(x.len(), 2 * row);
        assert_eq!(y, vec![data.y[2], data.y[0]]);
        assert_eq!(&x[..row], &data.x[2 * row..3 * row]);
        assert_eq!(&x[row..], &data.x[0..row]);
    }
}
