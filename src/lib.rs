//! Experimental local holonomy learning with projection-gated global pooling.
//!
//! This crate extracts the Nagare holonomy toy learner into a small standalone
//! research surface. It intentionally keeps the implementation plain Rust so
//! benchmark behavior is easy to inspect.

pub mod cv_data;
pub mod datasets;
pub mod detector;
pub mod features;
pub mod junction_tree;
pub mod learner;
pub mod metrics;
pub mod online;
pub mod ops;
pub mod optimizer;
pub mod pooling;
pub mod projection;
pub mod runtime;
pub mod tabular;
pub mod tabular_graph;
pub mod vision;

pub const VERTEX_FEATURES: usize = 7;
pub const STRUCTURAL_FEATURES: usize = 4 * VERTEX_FEATURES;
pub const LOCAL_FEATURES: usize = STRUCTURAL_FEATURES + 1;
pub const PROJECTION_RANK: usize = 6;
pub const PROJECTION_ALPHA: f32 = 0.72;

pub use cv_data::{
    feature_stats, load_idx, load_raw, load_split, rot_all, standardize_with, Split,
};
pub use datasets::{
    corrupt_dataset, gather_batch, make_dataset, shuffle_point_order, Dataset, Task,
};
pub use detector::{
    gen_scene, leaf_center_object, leaf_on_object, obox_contains, DetectorConfig, NodePred,
    SbshDetector,
};
pub use junction_tree::{balanced_binary_tree, Clique, JunctionTreeCholesky};
pub use learner::{
    evaluate_local, forward_timing, run_stress_ablation, Config, EntropyPoolLocalLearner, GateMode,
    StressKind, StressRow, Timing,
};
pub use metrics::{auroc, clifford_probability_error, cross_entropy, entropy2, softmax2, Metrics};
pub use online::{BlockEvolventHead, EvolventHead, InfoEvolventHead};
pub use ops::adam::{adam_step, AdamState};
pub use ops::catmull_rom::{
    catmull_rom_backward, catmull_rom_forward, chebyshev_control_points, chebyshev_cr_backward,
    chebyshev_cr_forward, chebyshev_deploy_backward, chebyshev_deploy_forward,
    chebyshev_knot_basis, CatmullRomBackward, CatmullRomCache, ChebyshevCrBackward,
};
pub use ops::cayley_rotor::{cayley_rotor_backward, cayley_rotor_forward};
pub use ops::clifford_fir::{clifford_fir_backward, clifford_fir_forward, CliffordFIR};
pub use ops::conv2d::{conv2d_backward, conv2d_forward, ConvLayer, ConvShape};
pub use ops::cpml_tier::{cycle_incidence_degrees, tier_cycle_indices, TierSpec};
pub use ops::dihedral::{dihedral_steer_backward, dihedral_steer_forward, DihedralGroup};
pub use ops::fsr_mixer::{FsrMixer, FsrMixerBackward, FsrMixerCache, FsrRoute};
pub use ops::fused_entropy_update::{
    fused_entropy_update_backward, fused_entropy_update_forward, FusedEntropyUpdateBackward,
    FusedEntropyUpdateShape,
};
pub use ops::gaussian_kld::{gaussian_kld_backward, gaussian_kld_forward, KldCache, Obox};
pub use ops::global_entropy_pool::{
    global_entropy_pool_backward, global_entropy_pool_forward, GlobalEntropyPoolOut,
    FEATS_PER_CHANNEL,
};
pub use ops::gomb_shell::{gomb_outer_backward, gomb_outer_forward};
pub use ops::group_pool::{group_pool_backward, group_pool_forward, GroupPoolOut};
pub use ops::hg_message::{
    hg_edge_to_node_backward, hg_edge_to_node_forward, hg_edge_to_node_sign_grad,
    hg_node_to_edge_backward, hg_node_to_edge_forward, hg_node_to_edge_sign_grad,
};
pub use ops::hsikan::{
    hsikan_backward, hsikan_forward, hsikan_forward_chunked, HsikanBackward, HsikanCache,
    HsikanConfig, HsikanEdges, HsikanParams, SplineKind,
};
pub use ops::kan::{kan_backward, kan_forward, KanCache, KanConfig};
pub use ops::kochanek_bartels::{kb_backward, kb_forward, KbBackward, KbCache};
pub use ops::linear::{linear_backward, linear_forward, LinearLayer};
pub use ops::loss::{bce_with_logits_backward, bce_with_logits_forward};
pub use ops::mse::{mse_backward, mse_forward, r2_score};
pub use ops::oriented_descriptor::{
    oriented_descriptor_backward, oriented_descriptor_forward, oriented_dim, OrientedOut,
};
pub use ops::oriented_head::{
    anchor_of_cell, assign_nodes, decode_backward, decode_forward, Anchor,
};
pub use ops::patch_projection::{
    patch_project_backward, patch_project_forward, PatchCache, PatchConfig,
};
pub use ops::phase_pool::{phase_pool_backward, phase_pool_dim, phase_pool_forward, PhasePoolOut};
pub use ops::project_alpha_mix::{
    project_alpha_mix_backward, project_alpha_mix_forward, ProjectAlphaMixBackward,
    ProjectAlphaMixShape,
};
pub use ops::quadtree::{
    node_pool_backward, node_pool_forward, quadtree_build, Quadtree, QuadtreeConfig,
};
pub use ops::rotor_holonomy::{rotor_holonomy_backward, rotor_holonomy_forward};
pub use ops::rotor_spike::{
    rotor_spike_backward, rotor_spike_dim, rotor_spike_forward, RotorSpikeOut,
};
pub use ops::sc_block::{
    oriented_sobel_bank, sc_block_backward, sc_block_forward, ScBlock, ScBlockCache, ScBlockGrad,
};
pub use ops::scatter::{scatter_mean_backward, scatter_mean_forward};
pub use ops::signed_scatter::{
    signed_scatter_backward, signed_scatter_forward, SignedScatterLanes, SignedScatterLayout,
};
pub use ops::soft_argmax::{soft_argmax_backward, soft_argmax_forward, SoftArgmaxOut};
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
pub use vision::{
    orientation_histogram, phase_feature_dim, phase_features, rotate_image, spatial_phase_features,
    PhaseFeature,
};
