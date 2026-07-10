//! Operator catalogue for Nagare.
//!
//! Each operator is a forward + backward pair of plain Rust functions
//! over SoA buffers (`&[f32]`, `&[u32]`, etc.). There is **no Op
//! trait**; we deliberately avoid an autograd graph. Operators
//! compose by direct function call + intermediate `Vec<f32>` storage,
//! and the training loop in `training.rs` orchestrates them.
//!
//! Per-operator gradients are derived once analytically and hand-coded
//! (see plan.tex Section "Per-primitive forward + backward kernels"
//! for the closed-form expressions for FIR, scatter-mean, linear,
//! BCE-with-logits).

pub mod adam;
pub mod catmull_rom;
pub mod cayley_rotor;
pub mod clifford_fir;
pub mod cpml_tier;
pub mod dihedral;
pub mod fsr_mixer;
pub mod fused_entropy_update;
pub mod gomb_shell;
pub mod hg_message;
pub mod hsikan;
pub mod kan;
pub mod kochanek_bartels;
pub mod linear;
pub mod loss;
pub mod mse;
pub mod patch_projection;
pub mod phase_pool;
pub mod project_alpha_mix;
pub mod rotor_holonomy;
pub mod scatter;
pub mod signed_scatter;
pub mod softmax_k;
pub mod spectral_entropy;
