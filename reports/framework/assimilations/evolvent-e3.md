---
experiment_id: evolvent-e3
date: 2026-07-15
scope: BlockEvolventHead (hyperedge-clique precision) + pairwise-vs-tensor race; RNG bug fix + F-EVO-4 correction
---
# Assimilation — evolvent E3
1. Result: pairwise precision can't represent a 3-way interaction (R2 ~0.15); block hyperedge precision recovers it (0.997 == dense) at 20x less storage.
2. Evidence: reports/2026-07-15-evolvent-e3-hypergraph-precision.md, reports/figures/evolvent-hypergraph.png.
3. Novelty: F-EVO-5 NEW_CANONICAL_CAPABILITY; RNG BUG_FIX_WITH_ARCHITECTURAL_IMPACT.
4. Interpretation: the interaction unit is the hyperedge (tensor block), not the feature-pair (matrix entry); fixes both O(d^2) and second-order-only.
5. Framework: +BlockEvolventHead (EXPERIMENTAL); ties evolvent to signed-hypergraph substrate (hg_message, SBSH bounded width).
6. Source: src/online.rs BlockEvolventHead +2 tests; examples/evolvent_hypergraph.rs; RNG fix in stream/bench/hypergraph.
7-8. Components: BlockEvolventHead registered. Defaults unchanged.
9. Guards: RNG fix. CORRECTION: F-EVO-4 re-run fixed-RNG -> evolvent COMPETITIVE (not superior); decisive-on-hard retracted.
10. Superseded: F-EVO-4 pre-correction numbers.
11. Regression: single_block_equals_dense, block_matches_dense_when_separable. Suite 174/0.
12. Open: junction-tree block-tridiagonal precision via hg_message; SBSH width certificate as tractability guarantee; non-additive target to price the separator coupling.
13. Next authorized: junction-tree precision (block-tridiagonal) through hg_message.
