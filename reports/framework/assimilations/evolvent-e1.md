---
experiment_id: evolvent-e1
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
scope: multi-output EvolventHead + 50-seed static benchmark (cls + reg, generated + real) on kato15
---

# Assimilation — evolvent E1

## 1. Experiment result
50-seed, 6-dataset static benchmark (kato15 32-core). One-pass evolvent matches/beats a 200-epoch MLP on 5/6
datasets (decisive on hard high-d/multi-class), beats one-pass SGD on all 6, tightest IQR, ~1/10 training cost.

## 2. New evidence
`reports/2026-07-15-evolvent-e1-static-benchmark.md`, `reports/figures/evolvent-bench.png`,
`reports/figures/evolvent_bench_seeds/*.json` (50 seeds).

## 3. Novelty classification
F-EVO-4 `NEW_CANONICAL_CAPABILITY` (evolvent matches/beats backprop on static supervised, cls + reg).

## 4. Canonical interpretation
The evolvent's defining property — one-pass exact ridge on a fixed expressive basis — pays off on STATIC data
(no epochs, no tuning, tight variance, competitive-to-superior accuracy). Resolves the E0 tension: static win,
long-drift streaming still open (F-EVO-3).

## 5. Framework impact
- Interface: `EvolventHead` now MULTI-OUTPUT (`new(d,c,ridge,lambda)`, `predict -> Vec`, `update(phi, &[y])`,
  `predict_class`); precision P shared across outputs.
- Default: none flipped. Status EXPERIMENTAL -> **DEPLOYABLE** for static supervised readouts.
- Guard: F-EVO-1 windup guard retained; streaming still EXPERIMENTAL (F-EVO-3).

## 6. Source changes
`src/online.rs` (multi-output rewrite + `one_hot_classifies_blobs` test); `examples/evolvent_bench.rs` (6-dataset
3-arm benchmark + JSON); `scripts/dev/plot_evolvent_bench.py` (50-seed aggregation); `examples/evolvent_stream.rs`
(updated to the new API).

## 7. Components added or updated
`EvolventHead` v2.0 DEPLOYABLE (multi-output) in `canonical_components.json`.

## 8. Defaults changed
None runtime. Status promotion only, evidence-backed (F-EVO-4).

## 9. Negative findings and guards
None new. Honest caveats recorded: MLP edges evolvent where learned features matter (iris, california); the
50-seed wall was memory-bandwidth-bound at 32-way concurrency (per-model timing read from quiet-host runs, not the
contended aggregate).

## 10. Superseded paths
None. F-EVO-3 (streaming EXPERIMENTAL) stands alongside F-EVO-4 (static DEPLOYABLE).

## 11. Regression tests
`online::one_hot_classifies_blobs` added; `::converges_to_batch_ridge`, `::tracks_a_drift`,
`::windup_guard_keeps_it_bounded` retained. Suite 172/0, fmt+clippy clean.

## 12. Remaining open questions
Nagare op basis (HSiKAN/rotor) instead of RFF — lift on california where the MLP's learned features still edge it?
Directional-forgetting streaming variant (F-EVO-3).

## 13. Next-experiment authorization
**Authorized.** Options: (a) swap RFF -> HSiKAN/rotor closed-form basis under the evolvent head; (b) the
directional-forgetting streaming variant. Both reuse `EvolventHead`; (a) reuses `hsikan`/rotor ops.
