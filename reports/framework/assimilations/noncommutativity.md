# Assimilation — non-commutativity (F-HOLO-9)

**Date:** 2026-07-17 · **Host:** Apple M5 Pro, CPU · **Finding:** F-HOLO-9 (POSITIVE — holonomy specificity)

## 1. Experiment
The missing test: a genuinely non-abelian task where the framework's `rotor_holonomy` is necessary and generic/2nd-order methods are structurally blind. Task: do two matched-magnitude regional holonomies commute?

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed median AUROC): trivial 0.509, generic-MLP(raw) 0.541, abelian 0.520 (all chance); MLP-on-holonomies 0.994; framework-commutator 1.000.
- **Interpretation:** ordered non-abelian composition is necessary; generic raw-edge methods can't compose loops. The specificity is in the loop extraction, not the commutator (MLP learns it given the holonomies).
- **Hypothesis:** a learned holonomy-native model would discover the commutator from raw edges where a generic MLP can't.

## 3. Novelty classification
NEW_CANONICAL_CAPABILITY — first clean holonomy-specificity win. Closes the F-HOLO-8 open question.

## 4. Smallest reusable unit extracted
`noncommute` module: `sample_noncommute`, `region_holonomy`, `commutator_angle`, `regional_angle_sum` — the non-abelian discriminating task + readouts; imported by the example.

## 5. Regression protection
`loop_product_equals_target`, `commute_class_has_zero_commutator_noncommute_positive`, `regional_angle_matched_across_classes` (abelian-blindness), determinism. 173/173 suite.

## 6. Anti-bloat check
Reused `rotor_holonomy_forward`, `curvature_task` quat/RNG helpers, `hymeko_clifford`. New module only for the genuinely new non-abelian task; no op re-implemented.

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-9, 32), this `.md`/`.json`.

## 8. Default routing
No default changed; the module is a discriminating-task probe.

## 9. Gates
HOLONOMY-SPECIFIC: framework-commutator 1.000 vs generic/abelian max 0.541. 173/0; clippy/fmt clean; 3.1 s.

## 10. Corrections to prior beliefs
Completes the honest characterization the arc was for: the framework's *specific* ops are justified **only** on non-abelian / order-dependent signals (F-HOLO-9); on scalar/2nd-order signals, generic baselines win (F-HOLO-2 / real-spine / F-HOLO-8). Not a general-purpose win — a specific one.

## 11. Integrity note
framework-commutator = 1.0 is a *necessity* demo (the commutator is the planted signal), not a learned win — stated. The generic arms at chance (fair MLP attempt) establish structural blindness. Abelian-blindness proved in code.

## 12. Honest scope / limitations
Synthetic; the win is "ordered non-abelian composition is necessary." A learned holonomy-native model on this task, and real pose-graph data, are the next rungs.

## 13. Next-experiment authorization
(a) `RotorMeshNet` + transported-DFA (F-HOLO-6) on the non-commutativity task (raw-edge input): does the *learned* rotor machinery discover the commutator where a generic MLP can't? — the capstone (learning holonomy-native, on a task where it is necessary). (b) real pose-graph SLAM loop-closure consistency (SE(3) holonomy) — now justified by F-HOLO-9.
