# Assimilation — nonlinear curvature (B+C)

**Date:** 2026-07-16 · **Host:** Apple M5 Pro, macOS 26.5.2, CPU · **Finding:** F-HOLO-4 · **Component:** NonlinearCurvatureField v1.0 · **Extends:** F-HOLO-3

## 1. Experiment
User steer: nonlinear curvature via Chebyshev/CR patches, "helps the closed form anyway." Extend F-HOLO-3's constant-flux exact solve to a spline-parameterized curvature *field*.

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed median held-out separability): trivial 0.520, constant-rotor 0.520, MLP 0.537, Laplacian 0.999, ChebyCR 1.000, oracle 1.000. Contrast sweep k_gen 2..8: ChebyCR 0.99–1.00.
- **Interpretation:** the F-HOLO-3 constant-rotor exact solve is *insufficient* for spatially-structured curvature; a closed-form spline readout recovers it.
- **Hypothesis (untested):** near-white generation would collapse the ceiling; a learned patch would beat the fixed fit on a harder task.

## 3. Novelty classification
NEW_CANONICAL_CAPABILITY (F-HOLO-4) + NEW_REUSABLE_COMPONENT (NonlinearCurvatureField) + NEW_EVALUATION_REQUIREMENT (the 3-way gate: trivial AND constant-rotor both blind). On the real spine (ChebyCR + rotor_holonomy). Explicit NON-GOAL rejected: nonlinear-along-a-1-D-edge (unidentifiable).

## 4. Smallest reusable unit extracted
`curvature_field` module (`grid_graph`, `sample_curvature_field`, `extract_curvature_field`, `chebycr_roughness`), imported by the example. Reuses `rotor_holonomy_forward`, `chebyshev_knot_basis`, F-HOLO-3 primitives.

## 5. Regression protection
`constant_rotor_mean_matched_across_classes` guards the insufficiency claim; `plaquette_holonomy_equals_field_exactly` + `gauge_leaves_curvature_invariant` guard the construction; `perf_roughness_latency_budget` locks the cost. 6 new tests, 163/163 suite.

## 6. Anti-bloat check
Queried registry + tree; no module owned "curvature-field extraction / spline roughness." Reused ChebyCR (`chebyshev_knot_basis`) rather than a hand-rolled DCT; reused `rotor_holonomy` and F-HOLO-3 helpers. Companion to ClosedCliffordCurvature, not a duplicate (constant-flux exact solve vs nonlinear-field regime).

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-4, 27), `canonical_components.json` (+NonlinearCurvatureField, 24), this `.md`/`.json`.

## 8. Default routing
`chebycr_roughness` is the default nonlinear-curvature readout; `constant_rotor_energy` retained as the (now-insufficient) F-HOLO-3 baseline; `laplacian_roughness` as a no-solve check. No prior default changed.

## 9. Gates
3-way gate PASS; ChebyCR reaches oracle where constant-rotor is blind; 163/0; clippy/fmt clean; perf within budget.

## 10. Corrections to prior beliefs
Sharpens F-HOLO-3: the constant-rotor exact solve is not the end — it is *insufficient* for nonlinear curvature. No result reversed.

## 11. Integrity note
Insufficiency proved in code (matched mean flux), not asserted; matched Haar marginals keep trivial entropy blind; Laplacian arm rules out a basis artifact; plaquette=field and gauge-invariance are tested identities.

## 12. Honest scope / limitations
Robust across k_gen (permutation always decorrelates), so the contrast→0 worst case is unreached; near-white and anisotropic curvature untested. Fixed low-order fit, not a learned patch (GAP #2 proper open). clifford_fir readout not yet compared.

## 13. Next-experiment authorization
Authorized: the LEARNED ChebyCR/Clifford-FIR patch + entropy feedback (GAP #2 proper) on this metric, or the near-white/anisotropic worst-case stress first. Wire `clifford_fir` over the plaquette cycle pool as a second closed-form readout and measure the learned arm against the fixed-ChebyCR ceiling on a non-saturated contrast sweep.
