# Nagare — HSiKAN spline-pluggable (Chebyshev-CR ↔ Kochanek-Bartels)

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Made the HSiKAN op's inner + outer univariate splines **pluggable** between two bases —
`SplineKind::ChebyshevCr` (default, the PyTorch-parity path) and `SplineKind::KochanekBartels`
(TCB tangents) — without disturbing the FD- and parity-verified Chebyshev path. This mirrors
the PyTorch `SignedKANLayer.spline_kind` and wires the already-landed `kochanek_bartels` op
(`02d01c8`) into the signed-hyperedge core.

## Design — packed params, one dispatch point

The two bases carry **different learnable parametrisations**, so rather than add fields to the
public `HsikanParams` (which would break every call site), the KB params are **packed into the
existing `inner_coef`/`outer_coef` buffers**, interpreted per `spline_kind`:

| basis | per-branch packed layout | `branch_len` |
|---|---|---|
| `ChebyshevCr` | `(d, cheb_k)` Chebyshev coeffs | `d·cheb_k` |
| `KochanekBartels` | `(d, grid)` control points ++ `(d, grid, 3)` raw TCB tangents | `d·grid·4` |

`grad_inner_coef`/`grad_outer_coef` come back in the **same packed layout**. The Cheb/KB choice
lives in exactly **two helpers** — `branch_spline_forward` / `branch_spline_backward` — so the
four spline call sites (inner/outer × fwd/bwd) dispatch through one place (no repeated `match`,
§6.5 #9). A per-branch `enum BranchSpline { Cheb{cache,control}, Kb{cache,control,tcb} }` in the
cache holds exactly what each basis's backward re-reads.

- `HsikanParams`, `HsikanEdges`, and every existing caller are **untouched**.
- New surface: `SplineKind` enum + `HsikanConfig::with_spline_kind(..)` builder. `new()` still
  yields `ChebyshevCr`, so all existing configs are unchanged.

## Verification

**The gate — Chebyshev is byte-identical:**
- `hsikan_torch_parity` — still **max|Δ| = 1.19e-7** over 2 arities (unchanged).
- `backward_matches_finite_difference` (Cheb) — green; the fixture generates the *same* 24
  coeffs at the *same* indices as before.

**New KB path:**
- `kb_backward_matches_finite_difference` — every packed grad (`grad_inner_coef` = control ++
  TCB, `grad_outer_coef` likewise), plus `grad_x`, `grad_gate_w`, `grad_gate_b`, matches central
  difference (tol 1e-2, eps 1e-3) on the KB-packed fixture (`d·grid·4 = 60`/branch, 120 total).
- The FD sweep was refactored into `fd_sweep(build)` and is now driven by both a Cheb and a KB
  fixture builder — one body, two bases (no duplicated assert blocks).

## Files touched

| file | change |
|---|---|
| `src/ops/hsikan.rs` | `SplineKind` enum, `with_spline_kind`, packed `branch_len`, `BranchSpline` cache enum, `branch_spline_{forward,backward}` dispatch, KB FD test + `fd_sweep` refactor |
| `src/lib.rs` | re-export `SplineKind` |

## CORE / deps

**None.** `src/ops/hsikan.rs` is not a `CORE.YAML` item; no dependency change.

## Test results

- Full suite **84 / 0** on Mac (arm64); clippy `-D warnings` + fmt clean. kato15 mirror pending.

## Open / next

- **CR-vs-KB comparison on a real graph task** (HSiKAN as the middle shell of the Iris signed
  graph) — the "test it" half of the ask; multi-seed, plotted, honest verdict. Next step.
- Gömb 2c (inner CPML + full 3-shell) still open.

## Provenance

Repo `github.com/kyberszittya/nagare`. Rust 1.96.1. Chebyshev parity fixture unchanged.
