use std::time::Instant;

use rand::{rngs::StdRng, seq::SliceRandom, Rng, SeedableRng};

use crate::{
    datasets::{corrupt_dataset, make_dataset, shuffle_point_order, Dataset, Task},
    metrics::{clifford_probability_error, cross_entropy, entropy2, softmax2, Metrics},
    pooling::structural_pool_features,
    projection::{default_holonomy_basis, fit_class_mean_basis, ProjectionBasis},
    LOCAL_FEATURES, PROJECTION_ALPHA, PROJECTION_RANK, STRUCTURAL_FEATURES,
};

#[derive(Clone, Debug)]
pub struct Config {
    pub tasks: Vec<Task>,
    pub n_train: usize,
    pub n_test: usize,
    pub n_points: usize,
    pub epochs: usize,
    pub batch_size: usize,
    pub lr: f32,
    pub seed: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            tasks: vec![Task::Moons, Task::Spiral, Task::Xor],
            n_train: 192,
            n_test: 96,
            n_points: 32,
            epochs: 50,
            batch_size: 32,
            lr: 0.05,
            seed: 53,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Timing {
    pub median_us_per_sample: f64,
    pub mean_us_per_sample: f64,
    pub max_us_per_sample: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GateMode {
    Entropy,
    Constant,
    Projection,
}

impl GateMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Entropy => "entropy",
            Self::Constant => "constant",
            Self::Projection => "projection",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StressKind {
    Clean,
    Noisy,
    Missing,
    FewShotNoisyMissing,
    /// Within-sample point-order shuffle (no value corruption): the
    /// holonomy-isolating knife. Perturbs only the order-sensitive holonomy
    /// channels, leaving every per-point (geometry/rotor) pooled statistic
    /// invariant. Applied to both train and test.
    Shuffled,
    /// The hard few-shot/noisy/missing regime *with* the point-order shuffle
    /// applied on top. This is the regime where the projection gate actually
    /// differentiates from the constant gate, so comparing it against
    /// `FewShotNoisyMissing` isolates whether the gate's advantage rides on
    /// order-sensitive holonomy structure.
    ShuffledFewShot,
}

impl StressKind {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Clean => "clean",
            Self::Noisy => "noisy",
            Self::Missing => "missing",
            Self::FewShotNoisyMissing => "fewshot_noisy_missing",
            Self::Shuffled => "shuffled",
            Self::ShuffledFewShot => "shuffled_fewshot",
        }
    }

    /// Whether this stress applies a within-sample point-order shuffle.
    fn shuffles(self) -> bool {
        matches!(self, Self::Shuffled | Self::ShuffledFewShot)
    }

    /// Whether this stress uses the reduced few-shot training budget with
    /// value noise and dropout.
    fn is_few_shot(self) -> bool {
        matches!(self, Self::FewShotNoisyMissing | Self::ShuffledFewShot)
    }

    fn train_samples(self, cfg: &Config) -> usize {
        if self.is_few_shot() {
            cfg.n_train.min(32)
        } else {
            cfg.n_train
        }
    }

    fn noise_std(self) -> f32 {
        match self {
            Self::Noisy => 0.18,
            _ if self.is_few_shot() => 0.22,
            _ => 0.0,
        }
    }

    fn missing_rate(self) -> f32 {
        match self {
            Self::Missing => 0.35,
            _ if self.is_few_shot() => 0.45,
            _ => 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct StressRow {
    pub task: Task,
    pub stress: StressKind,
    pub entropy_metrics: Metrics,
    pub constant_metrics: Metrics,
    pub projection_metrics: Metrics,
    pub entropy_timing: Timing,
    pub constant_timing: Timing,
    pub projection_timing: Timing,
    pub entropy_params: usize,
    pub constant_params: usize,
    pub projection_params: usize,
}

#[derive(Clone, Debug)]
pub struct EntropyPoolLocalLearner {
    w: Vec<f32>,
    b: [f32; 2],
    gate_mode: GateMode,
    projection_basis: ProjectionBasis,
}

impl EntropyPoolLocalLearner {
    pub fn new(seed: u64) -> Self {
        Self::new_with_gate(seed, GateMode::Entropy)
    }

    pub fn new_with_gate(seed: u64, gate_mode: GateMode) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut w = vec![0.0; LOCAL_FEATURES * 2];
        for value in &mut w {
            *value = (rng.random::<f32>() * 2.0 - 1.0) * 0.01;
        }
        Self {
            w,
            b: [0.0, 0.0],
            gate_mode,
            projection_basis: default_holonomy_basis(),
        }
    }

    pub fn logits_one(&self, phi: &[f32]) -> [f32; 2] {
        assert_eq!(phi.len(), LOCAL_FEATURES);
        let mut out = self.b;
        for (i, &v) in phi.iter().enumerate() {
            out[0] += v * self.w[2 * i];
            out[1] += v * self.w[2 * i + 1];
        }
        out
    }

    pub fn predict_dataset(&self, data: &Dataset) -> Vec<f32> {
        let structural = structural_pool_features(data);
        let mut logits = vec![0.0; data.samples * 2];
        for sample in 0..data.samples {
            let phi = self.phi_from_structural(
                &structural[sample * STRUCTURAL_FEATURES..(sample + 1) * STRUCTURAL_FEATURES],
            );
            let row = self.logits_one(&phi);
            logits[2 * sample] = row[0];
            logits[2 * sample + 1] = row[1];
        }
        logits
    }

    pub fn train(&mut self, data: &Dataset, epochs: usize, batch_size: usize, lr: f32, seed: u64) {
        let structural = structural_pool_features(data);
        if self.gate_mode == GateMode::Projection {
            self.projection_basis = fit_class_mean_basis(&structural, &data.y);
        }
        let mut rng = StdRng::seed_from_u64(seed);
        let mut indices: Vec<usize> = (0..data.samples).collect();
        for _ in 0..epochs {
            indices.shuffle(&mut rng);
            for chunk in indices.chunks(batch_size) {
                for &sample in chunk {
                    let base = sample * STRUCTURAL_FEATURES;
                    let phi =
                        self.phi_from_structural(&structural[base..base + STRUCTURAL_FEATURES]);
                    let logits = self.logits_one(&phi);
                    let (p0, p1) = softmax2(logits[0], logits[1]);
                    let y0 = f32::from(data.y[sample] == 0);
                    let y1 = f32::from(data.y[sample] == 1);
                    let gate = match self.gate_mode {
                        GateMode::Entropy => 0.25 + entropy2(p0, p1),
                        GateMode::Constant | GateMode::Projection => 1.0,
                    };
                    let delta = [y0 - p0, y1 - p1];
                    for (i, &value) in phi.iter().enumerate() {
                        self.w[2 * i] += lr * gate * value * delta[0];
                        self.w[2 * i + 1] += lr * gate * value * delta[1];
                    }
                    self.b[0] += lr * gate * delta[0];
                    self.b[1] += lr * gate * delta[1];
                }
            }
        }
    }

    pub fn n_params(&self) -> usize {
        let projection_params = if self.gate_mode == GateMode::Projection {
            STRUCTURAL_FEATURES * PROJECTION_RANK
        } else {
            0
        };
        self.w.len() + self.b.len() + projection_params
    }

    fn phi_from_structural(&self, structural: &[f32]) -> Vec<f32> {
        let mut warm = vec![0.0; LOCAL_FEATURES];
        warm[..STRUCTURAL_FEATURES].copy_from_slice(structural);
        warm[STRUCTURAL_FEATURES] = 1.0;
        let logits = self.logits_one(&warm);
        let (p0, p1) = softmax2(logits[0], logits[1]);
        warm[STRUCTURAL_FEATURES] = match self.gate_mode {
            GateMode::Entropy => entropy2(p0, p1),
            GateMode::Constant | GateMode::Projection => 1.0,
        };
        if self.gate_mode == GateMode::Projection {
            self.projection_basis
                .apply_alpha_mix(&mut warm[..STRUCTURAL_FEATURES], PROJECTION_ALPHA);
        }
        warm
    }
}

pub fn evaluate_local(model: &EntropyPoolLocalLearner, data: &Dataset) -> Metrics {
    let logits = model.predict_dataset(data);
    let ce = cross_entropy(&logits, &data.y);
    Metrics {
        acc: ce.acc,
        loss: ce.loss,
        entropy: ce.entropy,
        clifford_error: clifford_probability_error(&logits, &data.y),
    }
}

pub fn run_stress_ablation(cfg: &Config) -> Vec<StressRow> {
    let stresses = [
        StressKind::Clean,
        StressKind::Noisy,
        StressKind::Missing,
        StressKind::FewShotNoisyMissing,
        StressKind::Shuffled,
        StressKind::ShuffledFewShot,
    ];
    let mut rows = Vec::new();
    for (task_idx, &task) in cfg.tasks.iter().enumerate() {
        for (stress_idx, &stress) in stresses.iter().enumerate() {
            let train = make_dataset(
                task,
                stress.train_samples(cfg),
                cfg.n_points,
                cfg.seed + task_idx as u64 * 100 + stress_idx as u64 * 1_000,
            );
            let test = make_dataset(
                task,
                cfg.n_test,
                cfg.n_points,
                cfg.seed + task_idx as u64 * 100 + stress_idx as u64 * 1_000 + 1,
            );
            let train = corrupt_dataset(
                &train,
                stress.noise_std(),
                stress.missing_rate(),
                cfg.seed + 20_000 + stress_idx as u64,
            );
            let test = corrupt_dataset(
                &test,
                stress.noise_std(),
                stress.missing_rate(),
                cfg.seed + 30_000 + stress_idx as u64,
            );
            let (train, test) = if stress.shuffles() {
                (
                    shuffle_point_order(&train, cfg.seed + 40_000 + stress_idx as u64),
                    shuffle_point_order(&test, cfg.seed + 50_000 + stress_idx as u64),
                )
            } else {
                (train, test)
            };
            let mut entropy_model = EntropyPoolLocalLearner::new_with_gate(
                cfg.seed + task_idx as u64 * 17 + stress_idx as u64,
                GateMode::Entropy,
            );
            let mut constant_model = EntropyPoolLocalLearner::new_with_gate(
                cfg.seed + task_idx as u64 * 17 + stress_idx as u64,
                GateMode::Constant,
            );
            let mut projection_model = EntropyPoolLocalLearner::new_with_gate(
                cfg.seed + task_idx as u64 * 17 + stress_idx as u64,
                GateMode::Projection,
            );
            entropy_model.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + stress_idx as u64,
            );
            constant_model.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + stress_idx as u64,
            );
            projection_model.train(
                &train,
                cfg.epochs,
                cfg.batch_size,
                cfg.lr,
                cfg.seed + stress_idx as u64,
            );
            rows.push(StressRow {
                task,
                stress,
                entropy_metrics: evaluate_local(&entropy_model, &test),
                constant_metrics: evaluate_local(&constant_model, &test),
                projection_metrics: evaluate_local(&projection_model, &test),
                entropy_timing: forward_timing(&entropy_model, &test, 80),
                constant_timing: forward_timing(&constant_model, &test, 80),
                projection_timing: forward_timing(&projection_model, &test, 80),
                entropy_params: entropy_model.n_params(),
                constant_params: constant_model.n_params(),
                projection_params: projection_model.n_params(),
            });
        }
    }
    rows
}

pub fn forward_timing(model: &EntropyPoolLocalLearner, data: &Dataset, repeats: usize) -> Timing {
    for _ in 0..20 {
        let _ = model.predict_dataset(data);
    }
    let mut values = Vec::with_capacity(repeats);
    for _ in 0..repeats {
        let start = Instant::now();
        let _ = model.predict_dataset(data);
        values.push(start.elapsed().as_secs_f64() * 1.0e6 / data.samples as f64);
    }
    values.sort_by(|a, b| a.total_cmp(b));
    Timing {
        median_us_per_sample: values[values.len() / 2],
        mean_us_per_sample: values.iter().sum::<f64>() / values.len() as f64,
        max_us_per_sample: values[values.len() - 1],
    }
}
