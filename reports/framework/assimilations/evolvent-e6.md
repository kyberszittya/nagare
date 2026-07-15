# Assimilation — Evolvent E6 (data-scarcity discriminating test)

Date: 2026-07-15 · lifecycle per `feedback-assimilation-lifecycle-protocol`

## 1. Experiment → evidence

E6 answers the question E3–E5 left open: **what is the cross-clique (separator) coupling worth?** Sweeping
measurements-per-clique `per` on the branching clique tree (depth 6, d=441, arity ≈ 8), 5 seeds, kato15:

- R² gap MF − BLOCK rises 0.089 (per=2) → **peak 0.159 (per=6)** → 0.006 (per=40).
- `MF R² == DENSE R²` at every `(per, seed)`.
- weight-recovery RMSE worse for BLOCK throughout, deficit widening as data thins.

Evidence: `reports/2026-07-15-evolvent-e6-scarcity.md`, `reports/figures/evolvent-scarcity.png`,
`reports/figures/evolvent_e6_results.json`.

## 2. Novelty classification

`MEASUREMENT` (F-EVO-8) — no new component. A discriminating measurement over the existing
`JunctionTreeCholesky::solve` vs `solve_block_diagonal`. It **resolves** the open question carried by
F-EVO-5/6/7 ("a non-additive/data-scarce cross-clique target where BLOCK loses materially"): the answer is
data-scarcity, and block loses up to ~0.16 R².

## 3. Canonical decision

No component change. The finding delivers a **decision rule**: keep the coupling (info form / multifrontal) when
data is scarce per clique (< arity) — worth up to ~0.16 R²; block-diagonal suffices and is cheaper when data-rich.

## 4. Framework integration

Reused the existing solver and its block-diagonal baseline; extended `evolvent_multifrontal` with a `--per` axis
(one example, two sweep axes — no new file, §6.5 #4/#13) and a weight-recovery column. Honest knob: genuine
data-scarcity, not a manufactured non-additive coupling injected to force a gap.

## 5. Regression protection

Closed a §3 coverage gap: `solve_block_diagonal` (added in E5) had no direct unit test — `single_clique_equals_dense`
now also asserts it equals the exact solve on a single clique (where there are no separators to drop). Full suite
**178/0**, fmt + clippy clean.

## 6. Source-of-truth update

- `canonical_findings.json` — F-EVO-8 added; resolves the F-EVO-5/6/7 open question.
- Report + figure + results JSON on disk.
- Memory `project-nagare-evolvent-online-learning` updated.

## 7. Honest limitations carried forward

- Additive-in-features linear target; single depth (6). The scarcity is real data-scarcity.
- The gap is a `(structure, data-density)` property, not MF-alone — block-diagonal is nearly as good and cheaper in
  the rich regime.
- Separator-sharing axis (`sep`, depth) and a genuinely non-additive cross-clique feature untested.

## 8. Next (NOT yet authorized)

Separator-sharing sweep; a non-additive cross-clique feature; the E5 contiguous-storage + wall-clock follow-up.
