# Assimilation — Evolvent E7 (separator-sharing axis; partial refutation)

Date: 2026-07-15 · lifecycle per `feedback-assimilation-lifecycle-protocol`

## 1. Experiment → evidence

E7 tests the axis I named after E6: does the block deficit grow with the *degree* of separator-sharing? A **star**
clique tree (`star_clique_tree`) shares one separator across `fanout` children; sweep `fanout` at fixed scarcity
(`per=4`), 5 seeds, kato15.

- Block deficit is large (~0.16–0.28 R² median) whenever a separator is shared.
- But it **jumps then plateaus** across fanout 2…32 — non-monotone, IQRs overlapping.
- `MF R² == DENSE` at every cell.

Evidence: `reports/2026-07-15-evolvent-e7-sharing.md`, `reports/figures/evolvent-sharing.png`,
`reports/figures/evolvent_e7_results.json`.

## 2. Novelty classification — a correction on record

`PARTIAL_REFUTATION` (F-EVO-9). After E6 I stated the gap would "widen further" with more sharing. **The multi-seed
data refutes monotone growth.** Sharing *matters* (confirmed — big jump when a separator is shared); *degree* of
sharing beyond 2 does not add much (refuted — plateau). The single-seed run (seed 0) looked monotone (0.02 → 0.26);
the 5-seed run does not. Recorded per "multi-seed is the verdict / don't force the convenient narrative."

## 3. Canonical decision

No component change. The finding **refines F-EVO-8**: the coupling's value is a **threshold** on sharing (turns on
when a separator is shared, bounded by the per-clique data budget), not a dose-response in the number of sharers.

## 4. Framework integration

New topology helper `star_clique_tree` (the binary tree cannot express a separator shared by >1 child). Extended
`evolvent_multifrontal` with a `--fanout` switch (one file, another mode — §6.5 #13). Honest knob; honest reporting
of the plateau and the small-`d` noise.

## 5. Regression protection

`star_tree_equals_dense_with_shared_separator` — MF == dense on a star, exercising the assembly of many Schur
messages into the *same* parent separator positions (the code path the binary-tree tests didn't cover). Full suite
**179/0**, fmt + clippy clean.

## 6. Source-of-truth update

- `canonical_findings.json` — F-EVO-9 added (refines F-EVO-8).
- Report + figure + results JSON on disk.
- Memory `project-nagare-evolvent-online-learning` updated with the correction.

## 7. Honest limitations carried forward

- Additive-in-features linear target; single scarcity (`per=4`) and separator width (`sep=3`).
- Small `d` at low fan-out is noisy (d=9 at fanout=1; one seed had BLOCK beat MF via over-regularization) — the
  plateau conclusion rests on fanout ≥ 2.
- The genuinely non-additive cross-clique feature is still the open discriminating knob.

## 8. Next (NOT yet authorized)

A non-additive cross-clique feature (explicit product spanning a separator); the joint scarcity × sharing sweep at
larger `d`; the E5 engineering follow-ups.
