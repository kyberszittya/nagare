# Assimilation — Gate 2 (learning-necessary task)

**Date:** 2026-07-17 · **Host:** Apple M5 Pro, CPU · **Finding:** F-HOLO-7 (NEGATIVE/informative)

## 1. Experiment
Build a task where learning is *necessary* (fixed closed-form fails, learned succeeds) so a learned-rule / concentric-architecture win is task-meaningful. Attempt: XOR-of-regional-roughness on the F-HOLO-4 lattice.

## 2. Result (measured / interpretation / hypothesis)
- **Measured** (5-seed median AUROC): fixed-global all ~0.52 (chance); strong MLP 0.661; oracle (fixed regional) 0.935.
- **Interpretation:** defeats fixed-global AND generic MLP, but a fixed *regional* closed-form solves it → bias-discriminating, not learning-necessary.
- **Hypothesis:** the holonomy/entropy model (regional 2nd-order pooling) may close the MLP→oracle gap a generic MLP cannot.

## 3. Novelty classification
NEGATIVE_RESULT_REQUIRING_A_GUARD. Guards against scaling on an unmet gate; reframes the next test.

## 4. Smallest reusable unit extracted
`sample_regional_curvature`, `region_roughness_diff`, and the shared `realize_field` (refactored out of `sample_curvature_field`) in `curvature_field`; imported by the example.

## 5. Regression protection
`regional_sample_deterministic`, `regional_xor_oracle_separates_classes`; the F-HOLO-4 tests still pass after the behavior-preserving `realize_field` refactor. 168/168 suite.

## 6. Anti-bloat check
`realize_field` refactored out (both samplers share it — removed a would-be duplication). Reused all F-HOLO-3/4 readouts. No new op.

## 7. Registries updated
`canonical_findings.json` (+F-HOLO-7, 30), this `.md`/`.json`.

## 8. Default routing
No default changed; the regional generator/oracle are opt-in probes.

## 9. Gates
Gate 2 (strict) FAIL — informative: fixed-global chance, MLP 0.661 insufficient, oracle 0.935 is a fixed regional closed-form.

## 10. Corrections to prior beliefs
Corrects the optimistic "XOR-of-regional-curvature will be learning-necessary" (from the F-HOLO-5/6 next-step): a known-region-split version is solved by a fixed regional closed-form, so it is not strictly learning-necessary. Constructing a strict one needs a latent partition (harder).

## 11. Integrity note
The MLP got a fair strong attempt (128/600/700; 0.542→0.661) before "generic learner insufficient" was concluded. Not forced.

## 12. Honest scope / limitations
Known-region-split only; strict (latent-partition) task deferred. SLAM/concentric remain gated.

## 13. Next-experiment authorization
Bias-discrimination test: regional-pooling holonomy/entropy readout (`global_entropy_pool` per `cpml_tier` region) vs generic MLP vs a conv baseline on `curvature_xor_gate2` — does the framework's regional 2nd-order bias close the 0.66→0.94 gap? A conv baseline controls "holonomy-specific" vs "any 2nd-order bias." This, if it wins, is the honest capability claim (short of learning-necessity) that would justify the concentric build.
