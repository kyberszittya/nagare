---
title: "Non-commutativity — holonomy finally earns its specificity: only rotor_holonomy captures a genuinely non-abelian signal"
date: 2026-07-17
author: Aiko (Opus 4.8)
plan: docs/plans/2026-07-17-noncommutativity/
status: complete
tags: [auto-holonomy, non-abelian, holonomy-specificity, rotor-holonomy, commutator, nagare, positive-result, F-HOLO-9]
---

# Non-commutativity — the framework's specific op is finally *necessary*

**Created-at:** 2026-07-17 03:28 JST · **Plan:** [docs/plans/2026-07-17-noncommutativity/](../docs/plans/2026-07-17-noncommutativity/) · **Closes the open question of** F-HOLO-8

## Summary

Across F-HOLO-2, the real-spine signed-link run, and F-HOLO-8, the framework's *principles* validated
while its *specific* holonomy/entropy ops kept tying or losing to generic 2nd-order baselines. The
missing test: **a task whose signal is genuinely non-abelian** — order-of-composition dependent — where a
scalar pooler / conv / mean is structurally blind but `rotor_holonomy` is built for it. This is that
test, and it is the arc's first clean **holonomy-specificity** result.

Task: two regional holonomies `H_A = R(a_A, θ)`, `H_B = R(a_B, θ)` with **matched angle θ**, built as
ordered edge-rotor loop products; the class is whether they **commute** (parallel axes, `[H_A,H_B]=I`)
or not (perpendicular axes). Axis marginals are Haar in both classes; the *only* difference is the axis
correlation, carried entirely by the commutator — an ordered, non-abelian quantity.

**Result (5-seed median, held-out separability AUROC; input = raw edge rotors unless noted):**

| arm | what it computes | AUROC |
|---|---|:-:|
| trivial-entropy | covariance eigen-entropy of edge rotors | 0.509 |
| generic-MLP (raw) | learned over the `2k×4` raw edges | 0.541 |
| abelian-angle | `θ_A + θ_B` via `rotor_holonomy` | 0.520 |
| MLP-on-holonomies | learned over `(H_A, H_B)` (holonomies **given**, 8-dim) | 0.994 |
| **framework-commutator** | `∠([H_A,H_B])` via `rotor_holonomy`, fixed | **1.000** |

**Verdict: HOLONOMY-SPECIFIC.** Every generic and abelian method on the raw edges is at chance
(0.51–0.54); only the non-abelian `rotor_holonomy` commutator solves it (1.000). This is the first time
in the arc the framework's *specific* op is **necessary** and a generic learner **cannot substitute**.

**The honest localization (the nuance arm).** `MLP-on-holonomies` = 0.994: *given* the two holonomies
`(H_A, H_B)` as 8-dim vectors, a small MLP learns the commutator easily. So the specificity is in the
**loop extraction** — composing the ordered, non-commutative product `H = q_{k-1}⋯q_0` from the raw
edges — which generic raw-edge methods cannot do (consistent with F-HOLO-3's MLP failing at loop
products, 0.55). It is *not* that the commutator is un-learnable; it is that **you cannot get to the
commutator without the ordered holonomy composition**, and that composition is exactly what
`rotor_holonomy` provides and a scalar pooler / conv / mean structurally cannot.

## The arc's characterization (the payoff)

Tonight's five results now give a *precise* boundary for when the framework's specific machinery earns
its keep:

- **Scalar / 2nd-order signals** (roughness, curvature magnitude, spatial texture): generic methods
  match or beat the framework's specific ops — F-HOLO-2 (trivial entropy > deep holonomy net),
  real-spine (spline core > Gömb-Soma cascade), F-HOLO-8 (Laplacian > entropy op). Use the simple thing.
- **Genuinely non-abelian signals** (ordered loop composition, commutators, path-holonomy): the
  framework's `rotor_holonomy` is **necessary**; generic methods are structurally blind — **F-HOLO-9**.

That is the honest, useful answer the whole arc was for: **the holonomy machinery is justified exactly
when the task's signal is order-dependent / non-commutative.** It is not a general-purpose win; it is a
*specific* one, on the class of problems it was designed for (pose-graph consistency, SE(n)
synchronization, gauge/frustration with non-abelian structure).

## Files touched (new/append; no `CORE.YAML`)

| file | change |
|---|---|
| `src/noncommute.rs` | `sample_noncommute`, `region_holonomy`, `commutator_angle`, `regional_angle_sum`; 4 tests |
| `examples/noncommute_specificity.rs` | the 5-arm A/B + JSON |
| `src/lib.rs` | module + 4 exports |

**Reused (§6.1):** `rotor_holonomy_forward`, `curvature_task` (`haar_quat`/`axis_angle_quat`/
`rotor_angle`/`Rng`), `hymeko_clifford::{quat_mul, quat_conjugate}`, `spectral_reg_value_grad`,
`edge_log_field`, `adam_step`, `auroc`.

## CORE.YAML items touched

**None.** No new dependency.

## Test results

`cargo test --release --lib` — **173 passed / 0 failed** (4 new): `loop_product_equals_target`
(construction correct), `commute_class_has_zero_commutator_noncommute_positive` (the signal exists),
`regional_angle_matched_across_classes` (the abelian-blindness, proved in code), determinism. Static:
`clippy --all-targets -D warnings` **clean**; `fmt --check` **clean**.

## Performance

Full 5-arm run: **3.1 s**, RSS negligible. CPU (Apple M5 Pro).

## Experiment provenance

- **Git SHA:** `9432103` (branch `main`), dirty; added files uncommitted.
- **Env:** rustc 1.96.1; `hymeko_clifford` vendored; `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. `reports/figures/noncommute.json`.

## Open issues / follow-ups

- **The generative signal = the framework readout (by construction).** `framework-commutator` reaches
  1.000 because the commutator *is* the planted signal — this demonstrates *necessity* (generic methods
  blind), not a learned win. The honest capability claim is exactly "the ordered non-abelian composition
  is necessary," which the generic arms at chance establish.
- **A learned holonomy-native model on this task** (`RotorMeshNet` / the transported-DFA rule, F-HOLO-6)
  — does the *learned* rotor machinery discover the commutator where a generic MLP-on-raw-edges can't?
  The natural next rung: it would show the *learning* (not just the fixed op) is holonomy-native.
- **Real non-abelian data.** Pose-graph SLAM loop-closure consistency *is* this signal (SE(3) holonomy
  around loops); F-HOLO-9 is the synthetic proof-of-necessity, the g2o benchmarks are the real one — now
  justified, where the scalar/curvature tasks were not.

## Graphical output (§9)

- **Numerical:** `reports/figures/noncommute.json`.
- **Plotted:** `reports/figures/noncommute.png` (5-arm bars).
- **Animated:** N/A.
