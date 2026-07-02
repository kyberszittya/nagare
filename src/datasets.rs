use std::str::FromStr;

use rand::{rngs::StdRng, Rng, SeedableRng};

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
