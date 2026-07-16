# Assimilation — auto-holonomy (Step 1 + Step 2)

**Date:** 2026-07-16 · **Host:** Apple M5 Pro, macOS 26.5.2, CPU · **Finding:** F-HOLO-3 · **Component:** ClosedCliffordCurvature v1.0

## 1. Experiment
Resume the auto-holonomy frontier on the correct task (handoff `2026-07-16-nagare-handoff-hsikan-to-autoholonomy.md`): Step 1 = a task where trivial entropy fails but holonomy succeeds; Step 2 = the closed Clifford one-shot estimator.

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed median held-out separability AUROC): trivial-entropy 0.557, trivial-mean 0.509, MLP h16 0.553, strong-MLP h64/3×/400ep 0.626, oracle 1.000, closed-Clifford 1.000. Flux sweep: closed form 1.000 at θ_min∈[0.10,1.40]; MLP 0.55→0.62; trivial ≈ chance throughout.
- **Interpretation:** the loop-product (curvature) signal is real and exact for the closed form, and not accessible to a generic learner at this budget.
- **Hypothesis (untested):** a grid/torus with partial flux would open an estimator↔oracle SNR gap; a learned rotor rule might exploit it.

## 3. Novelty classification
NEW_CANONICAL_CAPABILITY (F-HOLO-3) + NEW_REUSABLE_COMPONENT (ClosedCliffordCurvature) + NEW_EVALUATION_REQUIREMENT (matched-marginal discriminating task as the auto-holonomy ceiling). On-mission (closed-form gauge un-twist = GAP #2), not the exact-solve/Cholesky devolution.

## 4. Smallest reusable unit extracted
`holonomy_estimator` (`closed_clifford_curvature` + `tree_gauge_frames` + `cotree_residuals`) and `curvature_task` (`wheel_graph` + `sample_connection`), both in the crate lib and imported by the example (not left in the driver). Reuses `rotor_holonomy_forward`, `spectral_reg_value_grad`, `metrics::auroc`, `hymeko_clifford` quat ops.

## 5. Regression protection
10 new tests (see JSON). The metric-integrity guard `edge_marginals_match_between_classes` fails if a future generator change leaks a marginal difference (which would silently invalidate the gate). `estimator_equals_oracle` locks the correctness identity. `perf_estimator_latency_budget` locks the one-shot cost.

## 6. Anti-bloat check
Queried the registry + tree first. No existing module owned "curvature estimation over an SO(3) connection." `rotor_holonomy` is the loop-product op (reused), `RotorMeshNet` is a node-field learner (architecturally mismatched to a cotree-localized signal — documented in the report). B1b (`hymeko_neuro`, PyTorch) is a *different* task (outlier localization) in a *different* repo; this is its Rust/nagare curvature-classification counterpart, not a duplicate.

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-3, 26 total), `canonical_components.json` (+ClosedCliffordCurvature, 23 total), this assimilation `.md`/`.json`.

## 8. Default routing
`closed_clifford_curvature` is the default curvature readout; `oracle_curvature` is the ceiling check. No prior default changed.

## 9. Gates
Step-1 metric gate PASS; Step-2 A/B closed-Clifford reaches oracle one-shot; 157/0 tests; clippy/fmt clean; perf within budget.

## 10. Corrections to prior beliefs
None reversed. Reinforces F-HOLO-2 (measure the trivial ceiling first) by building a task that *survives* it.

## 11. Integrity note
Matched Haar marginals are proved (module docs) and verified in code; strong-MLP arm rules out undertraining; estimator≡oracle is tested. No inflatable metric.

## 12. Honest scope / limitations
Exact gauge solve, not a learned compositional rule (GAP #2 proper open). On the wheel estimator==oracle (cycle basis == plaquettes); a mismatched-tree topology is needed to strictly approximate. Weak-flux worst case (θ_max small) not fully stressed.

## 13. Next-experiment authorization
Authorized: (a) grid/torus curvature task with partial flux → estimator strictly approximates oracle (SNR gap) — cheap, still closed-form; then (b) the learned holonomy-native rotor rule (RotorMeshNet + entropy feedback) measured against that genuine gap. Do (a) before (b) so the learned method faces a non-saturated ceiling.
