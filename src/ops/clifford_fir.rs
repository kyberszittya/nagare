//! Clifford-derivative signed-cycle FIR (re-export from `hymeko_graph::spine`).
//!
//! The Clifford-FIR is the entry-point operator for Nagare: it
//! consumes a SoA cycle pool + per-vertex features and produces
//! per-cycle aggregated features. Backward is closed-form via the
//! Clifford derivative ∇ = ∂/∂a + i ∂/∂b on the Cl(0,1) ≅ ℂ unified
//! filter (one multivector parameter unifies the σ=+1 and σ=−1
//! sign branches).
//!
//! This module re-exports the implementation from `hymeko_graph::spine`
//! so the operator lives in the cycle-pool crate (where the SoA
//! types live) and Nagare composes it.

pub use hymeko_graph::{clifford_fir_backward, clifford_fir_forward, CliffordFIR};
