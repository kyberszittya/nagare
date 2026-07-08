# HSiKAN → Nagare, Phase 1c′ — spectral-entropy regulariser as a closed-form op

Date: 2026-07-08 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-08-spectral-entropy-reg/` (4 artifacts) · Follows 1c

## Summary

The second entropy mechanism (user chose "both"): HSiKAN's **refined spectral-entropy
regulariser** (`hyperedge/entropy_reg.py::EntropyRegulariser`) ported to Nagare as a
**closed-form op** — a symmetric **Jacobi eigensolver** + a hand-derived
**spectral-entropy gradient** (no autograd). For a matrix `A ∈ ℝ^{n×d}` (the HSiKAN
node-embedding matrix):

```
G = AᵀA → (λ,U)=eigh(G) → p=max(λ,0)/Σ → H_norm=−Σp log₂p / log₂rank
reg = λ_eff·( a(H_norm−τ)² + b·H_norm + κ(1−H_norm) )        (λ_eff = detached Lyapunov schedule)
∇_A reg = 2·A·M,   M = U·diag(w)·Uᵀ,   w_j = ∂reg/∂λ_j
```

The gradient through the eigendecomposition (`dλ_j/dA = 2·A·u_j u_jᵀ`) is the risk;
`M` is well-defined even under eigenvalue degeneracy (`w_j` depends only on `λ_j`).

## Validation — three independent gates (all green, both machines)

| gate | result |
|---|---|
| **eigensolver** (`jacobi_reconstructs_and_is_orthonormal`) | `G = U diag(λ) Uᵀ` and `UᵀU = I` to 1e-5/1e-4 |
| **FD gradient** (`grad_matches_finite_difference`) | closed-form `∇_A` == central-diff (ε=1e-3), tol 1e-2 |
| **torch parity** (`matches_pytorch_entropy_regulariser`) | vs the **real** `EntropyRegulariser` autograd (float64): **reg \|Δ\| = 0.0, grad max\|Δ\| = 4.47e-8** |

The parity is the strong one: my f32 op with a **Jacobi** eigensolver reproduces the
PyTorch autograd gradient (which uses `eigvalsh`) to f32 machine precision — confirming
both the derivation *and* that the port matches the real regulariser's formula, not
just my own FD. (FD alone only checks internal consistency; parity checks faithfulness.)

## Files touched

| file | change | lines |
|---|---|---|
| `src/ops/spectral_entropy.rs` | **new** — `jacobi_eigh`, config, pure `spectral_reg_value_grad`, stateful `SpectralEntropyReg::step`, + 3 tests | 372 |
| `src/ops/mod.rs` / `src/lib.rs` | +1 mod / +re-export | +4 |
| `scripts/dev/spectral_entropy_fixture.py` | **new** — parity fixture generator | 100 |
| `tests/spectral_entropy_parity.rs` + `tests/fixtures/spectral_entropy.txt` | **new** — parity test + frozen fixture | 86 |

## CORE.YAML / deps

**None.** No dependency added — the eigensolver is hand-written (std + existing
`rayon` only); std-only fixture parser (no serde).

## Test results (both machines)

- Full suite **55 / 0** on Mac (arm64) and kato15 (x86_64) — +4 over 1c (3 unit + 1 parity).
- `clippy --all-targets -- -D warnings` clean, `fmt --check` clean on both.

## Design-by-contract / anti-patterns

Preconditions documented + asserted (`a.len()==n·d`, `gram.len()==m·m`). Decomposed
into small helpers (`offdiag_sq`/`apply_rotation`/`gram_ata`/`gram_aat`/
`spectral_distribution`/`spectral_lambda_grad`/`build_m`) — each under the complexity
ceiling. The stateful schedule is a struct threaded explicitly (no globals, §6.5 #11).

## Open / follow-up

1. **Wire into HSiKAN training** (the remaining plan step): add `SpectralEntropyReg::step`
   on the node-embedding matrix, its `∇_A` summed into `grad_x` (from `hsikan_backward`),
   and show (a) task loss still falls, (b) `H_norm` moves toward `τ`. This closes the
   "both mechanisms" loop (entropy-gated update [1c] + spectral regulariser [1c′]).
2. **1d** — forward/train latency + peak RSS + the `chunk_t` cap.
3. **Multi-seed** entropy-vs-constant (from 1c).
4. `CoefEntropyRegulariser` (per-coef-tensor) is a thin loop over this op if needed.

## Provenance

- Repo `github.com/kyberszittya/nagare`, base `a8ca716` (working tree dirty).
- Parity reference: torch 2.11.0+cu128 (kato15), float64, seed 53, n=5 d=3,
  reg=0.950993, H_norm=0.76332, lam_eff=1.0. Rust 1.96.1 both boxes.
- Not committed yet — awaiting user's go + kato15→GitHub deploy key.
