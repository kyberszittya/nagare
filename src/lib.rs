//! Experimental local holonomy learning with projection-gated global pooling.
//!
//! This crate extracts the Nagare holonomy toy learner into a small standalone
//! research surface. It intentionally keeps the implementation plain Rust so
//! benchmark behavior is easy to inspect.

pub mod datasets;
pub mod features;
pub mod learner;
pub mod metrics;
pub mod ops;
pub mod optimizer;
pub mod pooling;
pub mod projection;
pub mod runtime;
pub mod tabular;
pub mod tabular_graph;

pub const VERTEX_FEATURES: usize = 7;
pub const STRUCTURAL_FEATURES: usize = 4 * VERTEX_FEATURES;
pub const LOCAL_FEATURES: usize = STRUCTURAL_FEATURES + 1;
pub const PROJECTION_RANK: usize = 6;
pub const PROJECTION_ALPHA: f32 = 0.72;

pub use datasets::{
    corrupt_dataset, gather_batch, make_dataset, shuffle_point_order, Dataset, Task,
};
pub use learner::{
    evaluate_local, forward_timing, run_stress_ablation, Config, EntropyPoolLocalLearner, GateMode,
    StressKind, StressRow, Timing,
};
pub use metrics::{clifford_probability_error, cross_entropy, entropy2, softmax2, Metrics};
pub use ops::adam::{adam_step, AdamState};
pub use ops::catmull_rom::{
    catmull_rom_backward, catmull_rom_forward, chebyshev_control_points, chebyshev_cr_backward,
    chebyshev_cr_forward, chebyshev_deploy_backward, chebyshev_deploy_forward,
    chebyshev_knot_basis, CatmullRomBackward, CatmullRomCache, ChebyshevCrBackward,
};
pub use ops::cayley_rotor::{cayley_rotor_backward, cayley_rotor_forward};
pub use ops::clifford_fir::{clifford_fir_backward, clifford_fir_forward, CliffordFIR};
pub use ops::fsr_mixer::{FsrMixer, FsrMixerBackward, FsrMixerCache, FsrRoute};
pub use ops::fused_entropy_update::{
    fused_entropy_update_backward, fused_entropy_update_forward, FusedEntropyUpdateBackward,
    FusedEntropyUpdateShape,
};
pub use ops::gomb_shell::{gomb_outer_backward, gomb_outer_forward};
pub use ops::hsikan::{
    hsikan_backward, hsikan_forward, hsikan_forward_chunked, HsikanBackward, HsikanCache,
    HsikanConfig, HsikanEdges, HsikanParams,
};
pub use ops::kan::{kan_backward, kan_forward, KanCache, KanConfig};
pub use ops::kochanek_bartels::{kb_backward, kb_forward, KbBackward, KbCache};
pub use ops::linear::{linear_backward, linear_forward, LinearLayer};
pub use ops::loss::{bce_with_logits_backward, bce_with_logits_forward};
pub use ops::mse::{mse_backward, mse_forward, r2_score};
pub use ops::project_alpha_mix::{
    project_alpha_mix_backward, project_alpha_mix_forward, ProjectAlphaMixBackward,
    ProjectAlphaMixShape,
};
pub use ops::scatter::{scatter_mean_backward, scatter_mean_forward};
pub use ops::signed_scatter::{
    signed_scatter_backward, signed_scatter_forward, SignedScatterLanes, SignedScatterLayout,
};
pub use ops::softmax_k::{
    accuracy_k, cross_entropy_k_backward, cross_entropy_k_forward, softmax_k,
};
pub use ops::spectral_entropy::{
    jacobi_eigh, spectral_reg_value_grad, SpectralEntropyConfig, SpectralEntropyReg,
};
pub use pooling::structural_pool_features;
pub use projection::{default_holonomy_basis, fit_class_mean_basis, ProjectionBasis};
pub use runtime::NagareRuntime;
pub use tabular::{load_csv, load_csv_regression, shuffle_split, Tabular, TabularReg};
pub use tabular_graph::{build_signed_cycle_pool, GraphPool};
