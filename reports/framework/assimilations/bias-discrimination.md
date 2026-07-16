# Assimilation — bias discrimination (F-HOLO-8)

**Date:** 2026-07-17 · **Host:** Apple M5 Pro, CPU · **Finding:** F-HOLO-8 (MIXED)

## 1. Experiment
On the Gate-2 task, does the framework's regional 2nd-order pooling close the generic-MLP→oracle gap, and is the win holonomy-specific?

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed median AUROC, same data): generic-MLP 0.661; block-Laplacian 0.929; block-entropy 0.858; oracle 0.935.
- **Interpretation:** regional 2nd-order pooling is the lever (both block arms ≫ MLP); the framework's *specific* entropy op is not (generic Laplacian beats it).
- **Hypothesis:** a genuinely non-abelian task (order-of-loop matters) would separate holonomy from any scalar 2nd-order pooler.

## 3. Novelty classification
NEW_MECHANISM (regional 2nd-order pooling closes the gap) + NEGATIVE_RESULT_REQUIRING_A_GUARD (not holonomy-specific — generic Laplacian wins).

## 4. Smallest reusable unit extracted
`block_entropy_features` (framework 2nd-order front-end) + `block_laplacian_features` (generic control) in `curvature_field`, imported by the example.

## 5. Regression protection
`block_entropy_higher_for_rough` (the front-end separates smooth/rough). 169/169 suite.

## 6. Anti-bloat check
Reused `region_laplacian`, `spectral_reg_value_grad`, the Gate-2 task/oracle. The trainer's 8 args folded into a `TrainCfg` struct (§6.5 #6, no allow). No new op.

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-8, 31), this `.md`/`.json`.

## 8. Default routing
No default changed; the block front-ends are opt-in probes.

## 9. Gates
Regional 2nd-order pooling closes the gap (0.66→0.93); NOT holonomy-specific (Laplacian 0.929 > entropy 0.858). 169/0; clippy/fmt clean.

## 10. Corrections to prior beliefs
Tempers the concentric-Gömb-Soma optimism: its value rests on regional 2nd-order pooling (validated) — but that is *not* holonomy-specific; a conv/Laplacian substitutes. Third consistent instance (F-HOLO-2, real-spine, F-HOLO-8) that framework *principles* validate while *specific-op* superiority does not.

## 11. Integrity note
Same data across arms; strong generic MLP; the two block arms share head hyperparameters (fair). The specificity NEGATIVE is the headline, not suppressed.

## 12. Honest scope / limitations
One task; conv baseline deferred; uniform column-block front-ends. SLAM/concentric remain gated on a task where the *specific* structure is necessary.

## 13. Next-experiment authorization
A **non-abelian discrimination task**: classify by the *non-commutativity* of regional holonomies (order of loop traversal changes the outcome) — a signal a scalar roughness/Laplacian/conv is blind to but `rotor_holonomy` captures. If the framework op wins there and generic 2nd-order cannot, holonomy-specificity is finally demonstrated — the missing justification for the concentric build.
