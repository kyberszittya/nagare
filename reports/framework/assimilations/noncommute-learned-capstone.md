# Assimilation — capstone: learning through the holonomy op (F-HOLO-10)

**Date:** 2026-07-17 · **Host:** Apple M5 Pro, CPU · **Finding:** F-HOLO-10 (capstone, thesis-confirming)

## 1. Experiment
Does the LEARNED holonomy machinery discover the commutator end-to-end (rotor_holonomy as a differentiable core layer, no tape) on the F-HOLO-9 non-abelian task?

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed AUROC; FD gate 6.3e-5 PASS): generic-MLP 0.541; trainable-W1 0.513; frozen-W1 0.968; fixed-commutator 1.000.
- **Interpretation:** training THROUGH the holonomy op corrupts it (trainable pre-transform drifts off identity and destroys the ordered-product signal); freezing the fixed op + learning the readout works.
- **Hypothesis (localized by the frozen-W1 test):** the trainable pre-composition is the culprit, not the head or the gradient (FD-verified).

## 3. Novelty classification
NEW_MECHANISM + NEGATIVE_RESULT_REQUIRING_A_GUARD. Operationally confirms designed>learned; prescribes the deploy pattern.

## 4. Smallest reusable unit extracted
Demonstrated `rotor_holonomy` as a differentiable core layer (composed closed-form backward, FD-gated) — but the finding is it should be used FIXED. No new lib code; the model composes existing `linear` + `rotor_holonomy`.

## 5. Regression protection
Runtime FD gate (6.3e-5) in the example (F-HOLO-1 precedent). 173/173 lib suite.

## 6. Anti-bloat check
Composed existing ops (`linear`, `rotor_holonomy`, `adam`); reused `sample_noncommute`, `commutator_angle`. No new module.

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-10, 33), this `.md`/`.json`.

## 8. Default routing
Prescribed pattern for non-abelian tasks: fixed `rotor_holonomy` + learned readout (NOT end-to-end trained rotor stack).

## 9. Gates
Designed-op-fixed: frozen-W1 0.968 vs trainable-W1 0.513. FD gate 6.3e-5 PASS. 173/0; clippy/fmt clean.

## 10. Corrections to prior beliefs
Answers the capstone question honestly: the "learned holonomy-native" that works is a readout on the FIXED op; end-to-end training THROUGH the op is a liability. Vindicates the exact/designed>learned thesis operationally.

## 11. Integrity note
FD gate rules out a gradient bug (the failure is real optimization); frozen-W1 is the discriminating test. Not forced.

## 12. Honest scope / limitations
One non-abelian task; unconstrained-linear W1 (a rotor-manifold W1 untested); fixed-op-wins stands regardless.

## 13. Next-experiment authorization
Real pose-graph SLAM loop-closure consistency (SE(3) holonomy, g2o) with the fixed-op + learned-readout pattern — now justified AND prescribed by F-HOLO-9/10. Compute per-loop holonomy residuals (fixed), learn a lightweight consistency/outlier readout, vs classical chi-square.
