//! Gömb outer FIR shell — `M` parallel Clifford-FIR banks over a signed cycle pool.
//!
//! Port of `hymeko_neuro/models/hymeko_gomb/shells.py::OuterFIRShell` (the V1-analogue
//! "volume" of Clifford-FIR filter banks). Each bank is an independent
//! [`CliffordFIR`] with its own `Cl(0,1)` `(a, b)` coefficients; the shell runs all `M`
//! banks over the same cycle pool and concatenates their per-cycle outputs:
//!
//! ```text
//!   y_m = clifford_fir_forward(batch, X, N, d, bank_m)   ∈ ℝ^{N_c × d}
//!   Y   = [ y_1 ‖ … ‖ y_M ]                              ∈ ℝ^{N_c × M·d}
//! ```
//!
//! Pure orchestration over the existing `clifford_fir` op — no new derivative is
//! hand-rolled. The backward slices `∂L/∂Y` per bank, feeds each to
//! [`clifford_fir_backward`], accumulates the feature gradient across banks, and keeps
//! each bank's Clifford filter gradient. All cycles in one call share arity `k`
//! (`batch.k`); mixed arity is handled by the caller running one batch (and one bank
//! set) per arity, exactly as `hsikan` groups by arity. The flat-pooling baseline is a
//! single [`CliffordFIR::signed_mean`] bank (`M = 1`).
//!
//! # Preconditions
//! Every bank's filter length equals `batch.k`; `features.len() == n_vertices · d`.

use hymeko_graph::{clifford_fir_backward, clifford_fir_forward, CliffordFIR, TopKCyclesBatch};

/// Forward outer shell: `M` banks concatenated → flat `(n_cycles, M·d)`.
///
/// # Preconditions
/// `!banks.is_empty()`; each `bank.k() == batch.k`; `features.len() == n_vertices·d`.
///
/// # Postconditions
/// Returns `Y` flat `(n_cycles, banks.len()·d)`; column block `m` is bank `m`'s output.
///
/// # Panics
/// Panics (via `clifford_fir_forward`) if a bank's length ≠ `batch.k` or the feature
/// length is wrong.
pub fn gomb_outer_forward(
    batch: &TopKCyclesBatch,
    features: &[f32],
    banks: &[CliffordFIR],
    n_vertices: usize,
    d: usize,
) -> Vec<f32> {
    assert!(!banks.is_empty(), "outer shell needs ≥ 1 bank");
    let n_cycles = batch.len();
    let m = banks.len();
    let mut out = vec![0.0f32; n_cycles * m * d];
    for (bank_idx, fir) in banks.iter().enumerate() {
        let y = clifford_fir_forward(batch, features, n_vertices, d, fir); // (n_cycles, d)
        for c in 0..n_cycles {
            let base = c * m * d + bank_idx * d;
            out[base..base + d].copy_from_slice(&y[c * d..c * d + d]);
        }
    }
    out
}

/// Backward outer shell → `(grad_features, grad_banks)`.
///
/// `grad_y` is `∂L/∂Y` flat `(n_cycles, M·d)`. The feature gradient sums the banks'
/// contributions; each bank keeps its own Clifford filter gradient.
///
/// # Preconditions
/// `grad_y.len() == n_cycles · banks.len() · d`; `banks`/`features`/`batch` match the
/// forward call.
///
/// # Postconditions
/// `grad_features` is `(n_vertices, d)`; `grad_banks[m]` matches `banks[m]`'s shape.
///
/// # Panics
/// Panics if `grad_y` has the wrong length.
pub fn gomb_outer_backward(
    batch: &TopKCyclesBatch,
    features: &[f32],
    banks: &[CliffordFIR],
    grad_y: &[f32],
    n_vertices: usize,
    d: usize,
) -> (Vec<f32>, Vec<CliffordFIR>) {
    let n_cycles = batch.len();
    let m = banks.len();
    assert_eq!(grad_y.len(), n_cycles * m * d);
    let mut grad_features = vec![0.0f32; n_vertices * d];
    let mut grad_banks = Vec::with_capacity(m);
    for (bank_idx, fir) in banks.iter().enumerate() {
        // Slice out this bank's column block of ∂L/∂Y → (n_cycles, d).
        let mut gy = vec![0.0f32; n_cycles * d];
        for c in 0..n_cycles {
            let base = c * m * d + bank_idx * d;
            gy[c * d..c * d + d].copy_from_slice(&grad_y[base..base + d]);
        }
        let (gf, gfir) = clifford_fir_backward(batch, features, n_vertices, d, fir, &gy);
        for (acc, g) in grad_features.iter_mut().zip(&gf) {
            *acc += g;
        }
        grad_banks.push(gfir);
    }
    (grad_features, grad_banks)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 6 vertices, 4 signed 3-cycles.
    fn small_batch() -> (TopKCyclesBatch, usize, usize) {
        let cycles = vec![0, 1, 2, 2, 3, 4, 4, 5, 0, 1, 3, 5];
        let signs = vec![1i8, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1, 1];
        let batch = TopKCyclesBatch {
            cycles,
            signs,
            scores: vec![0.0f64; 4],
            k: 3,
        };
        (batch, 6, 3) // n_vertices, d
    }

    fn make_banks(m: usize, k: usize) -> Vec<CliffordFIR> {
        (0..m)
            .map(|bi| {
                let a = (0..k)
                    .map(|i| 0.2 * ((bi * k + i) as f32 * 0.7).sin())
                    .collect();
                let b = (0..k)
                    .map(|i| 0.2 * ((bi * k + i) as f32 * 1.1).cos())
                    .collect();
                CliffordFIR::new(a, b)
            })
            .collect()
    }

    fn features(nv: usize, d: usize) -> Vec<f32> {
        (0..nv * d)
            .map(|i| 0.1 * ((i as f32 * 1.3 + 0.4).sin()))
            .collect()
    }

    #[test]
    fn m1_equals_bare_clifford_fir() {
        // The flat baseline (M=1) must be exactly a bare clifford_fir forward.
        let (batch, nv, d) = small_batch();
        let x = features(nv, d);
        let bank = make_banks(1, batch.k);
        let y_shell = gomb_outer_forward(&batch, &x, &bank, nv, d);
        let y_bare = clifford_fir_forward(&batch, &x, nv, d, &bank[0]);
        assert_eq!(y_shell.len(), y_bare.len());
        assert!(y_shell
            .iter()
            .zip(&y_bare)
            .all(|(a, b)| (a - b).abs() < 1e-6));
    }

    #[test]
    fn forward_concat_shape_and_layout() {
        let (batch, nv, d) = small_batch();
        let x = features(nv, d);
        let m = 3;
        let bk = make_banks(m, batch.k);
        let y = gomb_outer_forward(&batch, &x, &bk, nv, d);
        assert_eq!(y.len(), batch.len() * m * d);
        // Column block m of cycle c equals bank m's bare output for cycle c.
        for (bank_idx, fir) in bk.iter().enumerate() {
            let bare = clifford_fir_forward(&batch, &x, nv, d, fir);
            for c in 0..batch.len() {
                let base = c * m * d + bank_idx * d;
                assert!(y[base..base + d]
                    .iter()
                    .zip(&bare[c * d..c * d + d])
                    .all(|(a, b)| (a - b).abs() < 1e-6));
            }
        }
    }

    fn perturb(
        banks: &[CliffordFIR],
        bank: usize,
        is_a: bool,
        coef: usize,
        eps: f32,
    ) -> Vec<CliffordFIR> {
        banks
            .iter()
            .enumerate()
            .map(|(i, f)| {
                let mut a = f.a.clone();
                let mut b = f.b.clone();
                if i == bank {
                    if is_a {
                        a[coef] += eps;
                    } else {
                        b[coef] += eps;
                    }
                }
                CliffordFIR::new(a, b)
            })
            .collect()
    }

    #[test]
    fn backward_matches_finite_difference() {
        let (batch, nv, d) = small_batch();
        let m = 2;
        let x = features(nv, d);
        let bk = make_banks(m, batch.k);
        let y = gomb_outer_forward(&batch, &x, &bk, nv, d);
        let grad_y = vec![1.0f32; y.len()]; // L = Σ Y
        let (gf, gbanks) = gomb_outer_backward(&batch, &x, &bk, &grad_y, nv, d);
        let eps = 1e-3;
        let sum_fwd = |xf: &[f32], bf: &[CliffordFIR]| -> f32 {
            gomb_outer_forward(&batch, xf, bf, nv, d).iter().sum()
        };

        for (idx, &g) in gf.iter().enumerate() {
            let mut xp = x.clone();
            xp[idx] += eps;
            let mut xm = x.clone();
            xm[idx] -= eps;
            let num = (sum_fwd(&xp, &bk) - sum_fwd(&xm, &bk)) / (2.0 * eps);
            assert!((g - num).abs() < 1e-2, "grad_feat[{idx}] {g} vs {num}");
        }
        for (bi, gbank) in gbanks.iter().enumerate() {
            for coef in 0..batch.k {
                for is_a in [true, false] {
                    let bp = perturb(&bk, bi, is_a, coef, eps);
                    let bm = perturb(&bk, bi, is_a, coef, -eps);
                    let num = (sum_fwd(&x, &bp) - sum_fwd(&x, &bm)) / (2.0 * eps);
                    let ana = if is_a { gbank.a[coef] } else { gbank.b[coef] };
                    assert!(
                        (ana - num).abs() < 1e-2,
                        "bank{bi}.{}[{coef}] {ana} vs {num}",
                        if is_a { "a" } else { "b" }
                    );
                }
            }
        }
    }
}
