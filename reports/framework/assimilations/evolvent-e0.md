---
experiment_id: evolvent-e0
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
scope: first probe of the "evolvent incremental online learning vs slow backprop" hypothesis (NON-CV)
---

# Assimilation — evolvent E0

## 1. Experiment result
Drifting-stream regression; A evolvent (forgetting-RLS) vs B online-SGD vs C backprop-MLP. Mixed: A wins
cold-start sample-efficiency (5/5), plain RLS windup-limited long-run; strong hypothesis unsupported by plain RLS.

## 2. New evidence
`reports/2026-07-15-evolvent-e0-stream.md`, `reports/figures/evolvent-stream.{png,json}` (5 seeds).

## 3. Novelty classification
F-EVO-1 `BUG_FIX_WITH_ARCHITECTURAL_IMPACT` (covariance windup); F-EVO-2 `NEW_MECHANISM` (cold-start sample
efficiency, O(d^2) per-update); F-EVO-3 `NEGATIVE_RESULT_REQUIRING_A_GUARD` (plain RLS not competitive long-run).

## 4. Canonical interpretation
Incremental/online = **sample-fast, not FLOP-fast**. The evolvent readout needs fewer samples (beats online-SGD),
but plain forgetting-RLS trades tracking against windup stability and does not beat backprop-Adam on a long stream.

## 5. Framework impact
- Interface: new `src/online.rs` + `EvolventHead` re-export.
- Guard: covariance-trace windup protection on by default (F-EVO-1).
- Default: none changed. `EvolventHead` status EXPERIMENTAL (not a "beats backprop" default).

## 6. Source changes
`src/online.rs` (new: `EvolventHead` + windup guard + 3 tests); `src/lib.rs` re-export;
`examples/evolvent_stream.rs` (new 3-arm benchmark); `scripts/dev/plot_evolvent.py`.

## 7. Components added or updated
`EvolventHead` (EXPERIMENTAL) registered in `canonical_components.json`.

## 8. Defaults changed
None. The evolvent readout is explicitly non-default until the directional-forgetting variant proves it.

## 9. Negative findings and guards
F-EVO-1 → trace-cap guard + `online::windup_guard_keeps_it_bounded`. F-EVO-3 → EXPERIMENTAL status + closure note.

## 10. Superseded paths
None. First component in the online-learning line.

## 11. Regression tests
`online::converges_to_batch_ridge`, `::tracks_a_drift`, `::windup_guard_keeps_it_bounded`. Suite 171/0, fmt+clippy clean.

## 12. Remaining open questions
Directional (selective) forgetting for windup-free aggressive tracking; a rapid-drift task where sample-efficiency
is decisive; a deeper evolvent (local closed-form updates beyond the readout).

## 13. Next-experiment authorization
**Authorized (E1): directional-forgetting `EvolventHead`.** Reuses `EvolventHead`/`linear`/`mse`; constrained by
F-EVO-2/F-EVO-3; new capability = windup-free aggressive tracking; lives in `src/online.rs`; failure (still not
competitive) would be assimilated as a closure on plain-RLS-style evolvent learning, pushing toward a genuinely
local deep evolvent instead.
