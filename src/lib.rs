//! Experimental local holonomy learning with projection-gated global pooling.
//!
//! This crate extracts the Nagare holonomy toy learner into a small standalone
//! research surface. It intentionally keeps the implementation plain Rust so
//! benchmark behavior is easy to inspect.

pub mod datasets;
pub mod features;
pub mod learner;
pub mod metrics;
pub mod pooling;
pub mod projection;

pub const VERTEX_FEATURES: usize = 7;
pub const STRUCTURAL_FEATURES: usize = 4 * VERTEX_FEATURES;
pub const LOCAL_FEATURES: usize = STRUCTURAL_FEATURES + 1;
pub const PROJECTION_RANK: usize = 6;
pub const PROJECTION_ALPHA: f32 = 0.72;

pub use datasets::{corrupt_dataset, make_dataset, Dataset, Task};
pub use learner::{
    evaluate_local, forward_timing, run_stress_ablation, Config, EntropyPoolLocalLearner, GateMode,
    StressKind, StressRow, Timing,
};
pub use metrics::{clifford_probability_error, cross_entropy, entropy2, softmax2, Metrics};
pub use pooling::structural_pool_features;
pub use projection::{
    default_holonomy_projection_basis, learn_holonomy_projection_basis,
    project_onto_holonomy_subspace,
};
