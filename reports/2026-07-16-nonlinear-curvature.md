---
title: "Nonlinear curvature (B+C) — a spline curvature field where the exact scalar solve is blind and a closed-form ChebyCR readout recovers it"
date: 2026-07-16
author: Aiko (Opus 4.8)
plan: docs/plans/2026-07-16-nonlinear-curvature/
status: complete
tags: [auto-holonomy, nonlinear-curvature, chebyshev, catmull-rom, clifford, rotor, SO3, gauge, nagare, positive-result, F-HOLO-4]
---

# Nonlinear curvature — the field the constant-rotor solve can't see, and the ChebyCR readout can

**Created-at:** 2026-07-16 23:44 JST · **Plan:** [docs/plans/2026-07-16-nonlinear-curvature/](../docs/plans/2026-07-16-nonlinear-curvature/) · **Extends:** F-HOLO-3 (`reports/2026-07-16-auto-holonomy.md`)

## Summary

Following the user's steer — *"nonlinear curvature? Chebyshev or CR-based patches? that can help with
closed form solution anyway"* — extended F-HOLO-3 from a **constant** per-plaquette flux (where the
exact tree-gauge solve *reaches* the oracle) to a **spline-parameterized curvature field** (where it
does **not**). This is the multi-scale regime the handoff's GAP #1 named, built on the real Nagare
spine (ChebyCR + `rotor_holonomy`), staying closed-form.

Two classes on a 2-D `SO(3)` lattice patch (`11×11` plaquettes), constructed with the **same flux
value multiset and total flux**, differing only in the spatial arrangement of curvature:
**smooth** (a low-order 2-D Chebyshev angle field) vs **rough** (a spatial permutation of the same
values). A random Haar gauge makes every edge Haar-marginal while preserving every holonomy.

**Headline (5-seed median held-out separability AUROC):**

| arm | AUROC | note |
|---|:-:|---|
| trivial-entropy (edge covariance) | **0.520** | at chance — matched marginals |
| **constant-rotor** (mean plaquette angle, the F-HOLO-3 solve) | **0.520** | **at chance — the exact scalar solve is BLIND here** |
| MLP over raw edges (trained) | 0.537 | learned baseline ≈ chance |
| Laplacian roughness (no-solve) | 0.999 | robustness check — not a Chebyshev artifact |
| **ChebyCR roughness (one-shot, no training)** | **1.000** | **reaches the oracle** |
| oracle (ChebyCR of the true field) | 1.000 | ceiling |

The sharper gate holds: trivial entropy **and** the constant-rotor mean are **both** at chance
(0.520 ≤ 0.60), the oracle is perfect. This is the point the user anticipated — **a curvature that is
nonlinear (spatially structured) exceeds what the constant-rotor exact solve of F-HOLO-3 can express**,
and the closed-form ChebyCR spline readout recovers it in one pass. The `--json` contrast sweep shows
ChebyCR holds 0.99–1.00 across generating orders `k_gen ∈ {2..8}` (a permutation destroys *all* spatial
autocorrelation, so any residual low-order structure suffices), while both blind arms stay at chance.

## Why the closed form (the user's "helps the closed form anyway")

A ChebyCR patch is **linear in its control points** with a closed-form, FD-verified derivative
(`catmull_rom`/`chebyshev_cr`, "all gradients are explicit closed-form buffers"). So:
- **Generation** of the smooth field is a closed-form low-order Chebyshev evaluation
  (`chebyshev_knot_basis`).
- **Extraction** of the gauge-invariant curvature field is a closed-form ordered 4-rotor holonomy per
  plaquette (`rotor_holonomy_forward`).
- **The readout** is a closed-form separable projection `θ̂ = P θ P`, `P = B_k(B_kᵀB_k)⁻¹B_kᵀ`
  (a tiny `k×k` SPD solve, `k=3`) — no gradient descent. Roughness = residual energy fraction.

The spline is the closed-form *nonlinear* representation: fitting the curvature field is a linear
(least-squares) problem in the control points, exactly as the user expected.

## Construction (why the metric is honest)

2-D lattice plaquette variables are independent (`#plaquettes = (L-1)² = |E|-(|V|-1)`). Set horizontals
to identity, verticals cumulatively `U(r,c)=R_r·∏_{c'<c}F(r,c')`, so plaquette `(r,c)` holonomy
`= U(r,c)⁻¹U(r,c+1) = F(r,c)` **exactly** (tested: `plaquette_holonomy_equals_field_exactly`). Apply a
random Haar gauge `g↦G_v g G_u⁻¹` — every holonomy preserved (gauge invariance, tested:
`gauge_leaves_curvature_invariant`), every edge now Haar. Hence in *both* classes: edge marginals
match (trivial entropy blind) **and** the mean plaquette angle matches (constant-rotor blind, tested:
`constant_rotor_mean_matched_across_classes`). Only the spatial arrangement separates them.

**Relation to F-HOLO-3.** There the curvature was a single constant per plaquette and the mean angle
(the constant-rotor readout) *was* the signal. Here the mean is matched by construction, so that exact
solve is insufficient — the nonlinear spatial structure needs a nonlinear readout. This is the on-ramp
to GAP #2 (a learned/spline readout beating the scalar solve), realized closed-form.

## Files touched (all new; `nagare_github`, no `CORE.YAML`)

| file | lines | role |
|---|--:|---|
| `src/curvature_field.rs` | 471 | grid graph, spline curvature-field generator (cumulative + gauge), field extraction, ChebyCR/Laplacian roughness, tiny SPD solve |
| `examples/curvature_field_dissociation.rs` | 365 | 3-way gate + arms + contrast sweep + JSON |
| `src/lib.rs` | +8 | `pub use` (append-only) |
| `reports/figures/plot_curvature_field.py` | 78 | figures |

**Reused (§6.1):** `rotor_holonomy_forward`, `chebyshev_knot_basis`, `spectral_reg_value_grad`,
`adam_step`, `metrics::auroc`, `hymeko_clifford::{quat_mul, quat_conjugate}`, and the F-HOLO-3
`curvature_task` primitives (`haar_quat`, `axis_angle_quat`, `rotor_angle`).

**Explicit non-goal (from the plan):** nonlinear *along a 1-D edge* — the path-ordered exponential
collapses to the net rotor at the endpoints (unidentifiable from graph holonomies). Not built.

## CORE.YAML items touched

**None.** No new dependency.

## Test results

`cargo test --release --lib` — **163 passed / 0 failed** (6 new in `curvature_field`). Layers:
- **Unit / correctness** — `plaquette_holonomy_equals_field_exactly` (cumulative construction),
  `gauge_leaves_curvature_invariant_and_edges_nontrivial` (gauge invariance + edges randomized),
  `spd_inverse_correct`, `smooth_has_lower_roughness_than_rough`.
- **Metric-integrity guard** — `constant_rotor_mean_matched_across_classes` (the insufficiency claim:
  the constant-rotor mean cannot separate the classes; if a change made it separable, this fails).
- **Performance** — `perf_roughness_latency_budget` (< 100 µs/sample; measured ~30 µs on the grid-12).

Static analysis: `clippy --all-targets -D warnings` **clean**; `fmt --check` **clean**. No `#[allow]`,
no `unwrap`/`expect` in non-test code, no §6.5 anti-patterns.

## Performance

| | wall | peak RSS | budget |
|---|--:|--:|--:|
| full example (5 seeds main + MLP + 5-level contrast sweep) | **2.3 s** | **5.2 MB** | < 30 s / < 2 GB ✅ |
| ChebyCR readout (extract + fit, grid-12, inference) | **~30 µs/sample** | — | < 100 µs ✅ |

CPU-only (Apple M5 Pro). Live per-seed progress to flushed stdout.

## New / removed dependencies

None.

## Experiment provenance

- **Git SHA:** `9432103` (branch `main`, `nagare_github`). Working tree dirty; added files as listed;
  **uncommitted**.
- **Env:** rustc 1.96.1; `hymeko_clifford` vendored; `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. Synthetic + deterministic in seed. Quaternion `(w,x,y,z)`, Hamilton product; noise 0.05 rad.
- **Data:** `reports/figures/curvature_field_dissociation.json` (main + contrast sweep).

## Open issues / follow-ups

- **Genuinely hard worst case.** ChebyCR is robust across `k_gen` (a permutation always fully
  decorrelates); the real ceiling-collapse case is a field with *no* spatial autocorrelation to begin
  with (near-white generation), where smooth ≈ rough — worth a `bandwidth→0` amplitude/correlation sweep.
- **Anisotropic / directional curvature.** The readout reads the angle field; an axis-structured field
  (curvature direction varying smoothly) would exercise the non-abelian content, not just the magnitude.
- **Learned ChebyCR patch (GAP #2 proper).** This readout is a *fixed* low-order fit; the learned
  version (Clifford-FIR over the cycle pool + entropy feedback, closed-form no-tape) attaching to this
  metric is the next rung — now with a task where the exact scalar solve is provably insufficient.
- **Clifford-FIR readout.** Only ChebyCR + Laplacian used; `clifford_fir` over the plaquette cycle pool
  is the natural signed-cycle aggregation to compare (the "CR-based patch" fully on-spine).

## Graphical output (§9)

- **Numerical:** `reports/figures/curvature_field_dissociation.json`.
- **Plotted:** `reports/figures/curvature_field_arms.png` (per-arm bars, gate line),
  `reports/figures/curvature_field_contrast_sweep.png` (AUROC vs `k_gen`).
- **Animated:** N/A — static graph inference (as F-HOLO-3 / B1b). A GIF applies once a curvature field
  is animated over a trajectory. Stated, not skipped.
