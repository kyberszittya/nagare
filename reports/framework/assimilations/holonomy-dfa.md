# Assimilation — holonomy-DFA (Gate 1)

**Date:** 2026-07-17 · **Host:** Apple M5 Pro, CPU · **Finding:** F-HOLO-5 (MIXED) · **Gate:** 1 of 2 before SLAM

## 1. Experiment
User gate: does the auto-holonomy learning rule hold up on small data before scaling to Visual-SLAM? Compare credit-assignment rules on one deep rotor net.

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed median AUROC, F-HOLO-1 substrate): sequential 0.894 (align 1.0); holonomy-DFA 0.831 (align +0.40); random-DFA 0.560 (align −0.005); trivial 0.944. Depth: holonomy-DFA L3 0.831 ≈ L1 0.815.
- **Interpretation:** exact inverse-rotor transport is the lever (beats random DFA, positive alignment); the global broadcast does not compose through depth (DFA ceiling).
- **Hypothesis:** a depth-composing broadcast (transport through the rotor chain) could recover depth-credit while staying global.

## 3. Novelty classification
NEW_MECHANISM (exact-rotor feedback path lifts DFA off chance) + NEGATIVE_RESULT_REQUIRING_A_GUARD (naive broadcast doesn't compose depth → don't scale it). Answers the small-data gate: learning doesn't collapse, but the frontier property is absent.

## 4. Smallest reusable unit extracted
`RotorMeshNet::backward_dfa` and `backward_from_rot_grads` — alternative credit-assignment rules on the existing type; the example imports them. Sequential exact `backward` remains the default.

## 5. Regression protection
`dfa_top_layer_equals_sequential_lower_differs` (anchors the method: broadcast ≡ threaded at the top) and `from_rot_grads_composes_with_dfa`. 165/165 suite.

## 6. Anti-bloat check
Methods added to the existing `RotorMeshNet` (no new type). Toy generator re-implemented in the example (small, acknowledged) rather than churning the frozen F-HOLO-1 example.

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-5, 28), this `.md`/`.json`. No new component (methods on an existing one).

## 8. Default routing
Sequential exact `backward` stays default; the DFA methods are opt-in research rules.

## 9. Gates
Gate 1 PARTIAL: holonomy-DFA beats random (YES), matches sequential (NO), uses depth (NO). 165/0 tests; clippy/fmt clean; 1.9 s.

## 10. Corrections to prior beliefs
Refines the auto-holonomy frontier: a *naive* global broadcast is insufficient for compositional-through-depth learning — the exact rotor helps but doesn't close it. The earlier optimistic framing ("global broadcast + local transport = the frontier") is narrowed to "exact-rotor feedback is necessary but not sufficient; depth-composition is the missing piece."

## 11. Integrity note
Rule-vs-rule on the same net (valid); not a task-necessity claim (F-HOLO-2 substrate; stated). The lever claim rests on two independent numbers (AUROC 0.831 vs 0.560 AND alignment +0.40 vs 0.00).

## 12. Honest scope / limitations
Naive broadcast, Adam only, trivial-solvable substrate. Depth-composing broadcast and a learning-necessary task are the open rungs. SLAM downstream of both.

## 13. Next-experiment authorization
Authorized: `backward_dfa_transported` (partial adjoint transport through the inverse-rotor chain) — A/B vs naive broadcast + sequential + alignment. If alignment → 1 and L3 > L1 returns, depth-composition is recovered → then Gate 2, then SLAM.
