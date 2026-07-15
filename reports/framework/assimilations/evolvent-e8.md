# Assimilation — Evolvent E8 (width/representability boundary)

Date: 2026-07-15 · lifecycle per `feedback-assimilation-lifecycle-protocol` · **run on the Mac, away from kato15**

## 1. Experiment → evidence

E8 closes the last open discriminating knob: a **genuinely non-additive** cross-clique term (`prod_k = rk0·rk1`,
weight β). Three arms, 5 seeds, Mac:

- MF-WIDE (width-4 multifrontal hosting the product) `== DENSE-WIDE` exactly, flat ~0.998 for all β, at 14 %
  storage.
- NARROW (linear-only) collapses 0.999 → 0.060 as β grows — in the **data-rich** regime, so more data does not
  help.

Evidence: `reports/2026-07-15-evolvent-e8-width.md`, `reports/figures/evolvent-width.png`,
`reports/figures/evolvent_e8_results.json`.

## 2. Novelty classification

`NEW_CANONICAL_CAPABILITY` (F-EVO-10). Closes the "non-additive cross-clique" question carried since
F-EVO-5/6/7/9 — and the answer separates two mechanisms cleanly:

| axis | mechanism | closes with |
|---|---|---|
| estimation gap (E6) | pooling shared-var evidence | more **data** |
| sharing threshold (E7) | separator-sharing | — (saturates) |
| **representability (E8)** | **treewidth < interaction order** | wider **cliques** |

A non-additive term does *not* touch the E6/E7 estimation plateau; it is a different axis, governed by treewidth
vs interaction order, which the multifrontal solver handles **exactly at the right width**.

## 3. Canonical decision

No component change (reuses `JunctionTreeCholesky` + `InfoEvolventHead`). The finding establishes the **width
certificate as the validity precondition** for the whole E4–E7 exactness line: the multifrontal solve is exact at
`O(d·w³)` iff width ≥ interaction order.

## 4. Framework integration

New example only (`evolvent_width.rs`) — a genuinely different question (representability) from the solver-
characterization example. No `src/` change; the E5 exactness tests already guard the solver.

## 5. Regression protection

Solver exactness is covered by the E5 `junction_tree::*` tests (MF == dense, incl. the star/shared-separator path).
E8's contribution is measurement over the existing, already-guarded solver — the example is the integration
evidence. Full suite **179/0**, fmt + clippy clean.

## 6. Source-of-truth update

- `canonical_findings.json` — F-EVO-10 added; resolves the non-additive open question of F-EVO-5/6/7/9.
- Report + figure + results JSON on disk.
- Memory `project-nagare-evolvent-online-learning` updated.

## 7. Honest limitations carried forward

- Product is an explicit feature (linear-in-features; the tree affords + solves it, does not discover it).
- Degree-2 interaction, single arity; higher order (degree ≥ 3 → width ≥ order+1) untested.
- `β` is a controlled dial, not fit from real data.

## 8. Environment

Run entirely on **Hajdus-MacBook-Pro (arm64, CPU)**; `holonomy_learn` is CPU-only Rust and the sweeps are ms-scale,
so no kato15 needed. **kato15 not synced this session** (working away from it) — origin `main` is ahead, kato15
fast-forwards on next pull.

## 9. Next (NOT yet authorized)

Higher-order interactions (degree ≥ 3); wire the SBSH width certificate as an explicit precondition on
`JunctionTreeCholesky::new`; the E5 engineering follow-ups.
