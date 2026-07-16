---
title: "Auto-holonomy Step 1+2 — the curvature-discriminating task and the closed Clifford one-shot estimator"
date: 2026-07-16
author: Aiko (Opus 4.8)
plan: docs/plans/2026-07-16-auto-holonomy/
status: complete
tags: [auto-holonomy, holonomy, curvature, clifford, rotor, SO3, gauge, nagare, positive-result, F-HOLO-3]
---

# Auto-holonomy — the task where holonomy is necessary, and the closed Clifford solve

**Created-at:** 2026-07-16 23:04 JST · **Plan:** [docs/plans/2026-07-16-auto-holonomy/](../docs/plans/2026-07-16-auto-holonomy/) · **Resumes:** `reports/2026-07-16-nagare-handoff-hsikan-to-autoholonomy.md`

## Summary

Resumed the auto-holonomy frontier on the correct task, per the handoff's binding order
(**discriminating task first**, F-HOLO-2). Built a controlled `SO(3)`-connection classification
where the two classes have **provably identical edge marginals** and differ *only* in loop-product
(holonomy) structure — **flat/integrable** (a pure gauge, `g_ij=R_iR_j⁻¹`, zero curvature) vs
**curved/frustrated** (nonzero plaquette flux). Then built the **closed Clifford approximation** —
a one-shot, closed-form curvature estimator (spanning-tree gauge-fix + cotree loop-closure
residual, no gradient descent, topology-only) — and A/B-tested it against trivial covariance
entropy, a trained MLP, and the oracle.

**Headline (5-seed median held-out separability AUROC):**

| arm | AUROC | note |
|---|:-:|---|
| trivial-entropy (covariance eigen-entropy) | **0.557** | at chance — the Step-1 gate |
| trivial-mean | 0.509 | sanity → chance |
| MLP h16 (trained) | 0.553 | learned baseline barely leaves chance |
| MLP h64, 3× data, 400 ep (strong, trained) | **0.626** | strong learner still ≈ chance |
| oracle (`rotor_holonomy` over true plaquettes) | **1.000** | ceiling |
| **closed-Clifford (one-shot, no training)** | **1.000** | **reaches the oracle** |

Three results, all measured:

1. **The task discriminates (Step-1 GATE PASS).** Trivial covariance eigen-entropy is at chance
   (0.557 ≤ 0.60), the oracle is perfect (1.000). Unlike the previous synthetic task where trivial
   entropy *won* (F-HOLO-2, 0.944 > 0.894), here holonomy's value is finally *measurable* — the
   metric is not inflatable by a second-order shortcut, because the classes are constructed with
   matched Haar marginals so only loop products separate them.
2. **The closed Clifford estimate reaches the oracle, one-shot, without training.** The
   tree-gauge + cotree-residual solve recovers the exact plaquette curvature (a tested
   identity: estimator ≡ oracle) from topology alone, at **~2 µs/sample** on the wheel-24.
3. **A strong learned baseline cannot.** A capacity-matched MLP (0.553) and even a strong MLP
   (hidden 64, 3× data, 400 epochs → 0.626) barely leave chance: the non-commutative loop-product
   structure is not accessible to a generic learner at this data budget, but is *exact* in one
   closed-form Clifford pass. **Flux sweep:** the closed form holds AUROC = 1.000 down to
   θ_min = 0.10 (weak flux) while the MLP crawls 0.55 → 0.62 and trivial stays at chance.

This is the auto-holonomy thesis in miniature — *differentiation as connection transport, exact and
closed-form* — realized as the instantaneous rotor rule flagged as open GAP #2 in
`feedback-nagare-closed-form-thesis-not-gaps` (closed-form Procrustes/gauge un-twist, not GD).

## The construction (why the metric is honest)

`SO(3)` connection on a **wheel** (hub 0 + rim `1..=n_rim`; star = spanning tree, rim = cotree, one
triangular plaquette per rim edge). Both classes draw node frames `R_i ~ Haar` and set spokes
`g_{0→i}=R_iR_0⁻¹`; **flat** sets rim `g=R_bR_a⁻¹` (every cycle holonomy = I); **curved** left-multiplies
each rim edge by an independent flux `F_p=exp(½θ_p n̂_p)`, `θ_p∈[θ_min,π]` (that plaquette's holonomy is
a conjugate of `F_p`, angle `θ_p`). Because a Haar rotor left/right-multiplied by an independent rotor
is Haar, **every edge is marginally Haar in both classes** — the pooled edge log-rotor field has the
same isotropic covariance, so any first/second-order statistic is class-blind. This is verified in code
(`edge_marginals_match_between_classes`: per-axis variance within 15 % across classes, isotropy < 20 %).
The classes separate only through ordered loop products — the continuous generalization of the `Z₂`
balance/frustration task (B1a/B2).

## The closed Clifford estimator (Step 2)

Given only topology + connection (not the flux): (1) gauge-fix by tree transport `R̂_root=I`,
`R̂_node=g·R̂_parent`; (2) cotree residual `ρ_e=R̂_b⁻¹·g_{a→b}·R̂_a` (identity iff flat); (3) curvature
energy `mean_e‖log ρ_e‖`. One pass, `O(|E|)`, closed-form, no descent. On this topology it recovers the
exact plaquette holonomy the oracle reads via `rotor_holonomy_forward` — a tested identity
(`estimator_equals_oracle`, |Δ| < 1e-3). Reuses `hymeko_clifford::{quat_mul, quat_conjugate}` and
`rotor_holonomy_forward` — no quaternion algebra re-implemented (§6.1).

**Relation to `RotorMeshNet`.** The iterative deep holonomy net (F-HOLO-1) is a *node-field* op; curvature
lives in the *cotree edges* (a cycle quantity), architecturally invisible to any gauge-natural node
encoding. So the appropriate learned baseline for a connection task is the edge-rotor MLP — and it fails,
while the closed-form cotree solve succeeds. The learned arm is the MLP by that design reasoning.

## Files touched (all new; `nagare_github`, no `CORE.YAML` here)

| file | lines | role |
|---|--:|---|
| `src/curvature_task.rs` | 380 | flat/curved wheel generator, Haar/axis-angle/log helpers, marginal-match tests |
| `src/holonomy_estimator.rs` | 210 | closed Clifford one-shot: tree gauge-fix + cotree residual + energy; oracle |
| `examples/auto_holonomy_dissociation.rs` | 353 | Step-1 gate + Step-2 A/B; flux sweep; live progress; JSON |
| `reports/figures/plot_auto_holonomy.py` | 75 | the two result figures |
| `src/lib.rs` | +14 | `pub mod` / `pub use` (append-only) |

**Reused (no edit):** `rotor_holonomy_forward`, `spectral_reg_value_grad` (trivial entropy),
`adam_step`/`AdamState` (MLP), `metrics::auroc`, `hymeko_clifford::{quat_mul, quat_conjugate}`.

## CORE.YAML items touched

**None.** `nagare_github` has no `CORE.YAML` (verified). No new dependency.

## Test results

`cargo test --release --lib` — **157 passed / 0 failed** (10 new: 6 in `curvature_task`, 4 in
`holonomy_estimator`). Layers:
- **Unit** — Haar unit+canonical; angle/log roundtrip incl. near-π stability; generator determinism;
  **flat → all holonomies identity**; **curved → bounded flux holonomy**; tree-gauge root-identity + unit;
  **estimator ≡ oracle** (the correctness identity); flat-energy≈0 / curved-energy>0 separation.
- **Metric-integrity (the F-HOLO-2 guard, in code)** — `edge_marginals_match_between_classes`: pooled
  edge log-rotor mean ≈ 0 and covariance isotropic + class-matched in both classes.
- **Integration/gate** — the example enforces the Step-1 gate at runtime (trivial ≤ 0.60 AND oracle ≥ 0.90).
- **Performance** — `perf_estimator_latency_budget` asserts median < 50 µs/sample (measured ~2 µs).

Static analysis: `cargo clippy --all-targets -- -D warnings` **clean**; `cargo fmt --check` **clean**. No
`#[allow]`, no `unwrap`/`expect` in non-test code, no §6.5 anti-patterns.

## Performance

| | wall | peak RSS | budget |
|---|--:|--:|--:|
| full example (main table 5 seeds + strong MLP + 6-level flux sweep) | **13.8 s** | **3.1 MB** | < 30 s / < 2 GB ✅ |
| closed Clifford estimate (wheel-24, inference) | **~2 µs/sample** | — | < 50 µs ✅ |

CPU-only (Apple M5 Pro, macOS 26.5.2). Live per-seed progress (per-arm AUROC, cell/s, wall) to flushed
stdout. *Budget note:* the plan declared < 30 s for a single-MLP design; adding the strong-MLP arm (to
preempt an undertraining objection) raised wall to 13.8 s after making the flux sweep light — still well
under budget.

## New / removed dependencies

None.

## Experiment provenance

- **Git SHA:** `9432103f5feee18120deb459514fc9168ff2b270` (branch `main`, `nagare_github`). Working tree
  dirty; the added files are those listed; **uncommitted** (no commit requested).
- **Env:** rustc 1.96.1; deps `hymeko_clifford` (vendored), `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. Data synthetic + deterministic in seed; quaternion `(w,x,y,z)`, Hamilton product,
  Haar via normalized 4D-Gaussian, canonicalized `w ≥ 0`.
- **Data:** `reports/figures/auto_holonomy_dissociation.json` (main + flux sweep).

## Open issues / follow-ups

- **Topology where estimator ≠ oracle.** On the wheel the fundamental-cycle basis coincides with the
  planted plaquettes, so the closed form recovers the oracle exactly. A grid/torus with flux on a
  random subset of plaquettes would make estimator and oracle diverge in SNR (the estimator aggregates
  all cotree residuals without knowing where flux was planted) — the next rung.
- **Harder worst case.** The flux sweep floors θ_min but keeps θ_max = π; a narrow-band flux
  `[θ_min, θ_min+ε]` would stress the estimator's angle resolution.
- **Compositional / learned rotor un-twist (GAP #2 proper).** This estimator is the *exact* gauge solve;
  the open frontier remains a *learned* holonomy-native rotor rule that composes through depth and, on a
  task where even the tree-gauge solve is insufficient (per-node-varying multi-scale twist), beats it —
  the `RotorMeshNet` mechanism (F-HOLO-1) attached to this metric.
- **Port target reciprocity.** This is the Rust/nagare counterpart of the PyTorch B1b rotor-sync
  benchmark; the two now bracket the op (outlier localization in PyTorch, curvature classification here).

## Graphical output (§9)

- **Numerical:** `reports/figures/auto_holonomy_dissociation.json` (per-arm median + flux sweep).
- **Plotted:** `reports/figures/auto_holonomy_arms.png` (per-arm bars, gate line),
  `reports/figures/auto_holonomy_flux_sweep.png` (AUROC vs θ_min).
- **Animated:** N/A — static graph inference (as B1b noted; a GIF applies once a *pose trajectory* is
  animated, the full SE(n) rung). Stated, not skipped.
