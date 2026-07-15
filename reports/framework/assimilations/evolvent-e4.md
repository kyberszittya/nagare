# Assimilation — Evolvent E4 (junction-tree / information-form precision)

Date: 2026-07-15 · lifecycle per `feedback-assimilation-lifecycle-protocol`

## 1. Experiment → evidence

E4 (`examples/evolvent_junction.rs`, kato15, 5 seeds × 4 sizes) races three precisions on a **local-measurement
chain** (path of `n` features; each sample a window of `W=6` correlated features; target linear in the window):

- **DENSE** `EvolventHead` — `O(n²)` precision, exact.
- **BLOCK** `BlockEvolventHead` (E3) — block-diagonal `O(n·w)`, drops cross-block coupling.
- **INFO** `InfoEvolventHead` (new) — information form `J,b`, exact, `J` sparse → `O(nnz)=O(n·w)`.

Measured: `INFO R² == DENSE R²` at every `(n,seed)` cell (info form is exact); info-matrix storage **21.6% →
11.1% → 5.6% → 2.8%** of dense as `n` doubles (halves each step). BLOCK trails by ≤0.0003.

Evidence: `reports/2026-07-15-evolvent-e4-junction-tree.md`, `reports/figures/evolvent-junction.png`,
`reports/figures/evolvent_e4_results.json`.

## 2. Novelty classification

`NEW_CANONICAL_CAPABILITY` (F-EVO-6) — **but the algorithm is classical**, and I say so plainly: the information
filter / Gaussian-MRF precision accumulator / junction-tree form is textbook (Kalman information filter;
Lauritzen–Spiegelhalter 1988). No novelty is claimed for the method. The framework-level contribution is the
**exact + sparse evolvent head at O(d·w)** that completes the precision family and realises the hypergraph↔tensor
conjugation the user asked for. Per the "distinguish generic pattern from specific claim" rule: the generic
information form is old; its wiring as the exact junction-tree evolvent over the signed-hypergraph substrate is the
specific, non-novel-but-new-to-framework increment.

## 3. Canonical decision

- **Register** `InfoEvolventHead` — `DEPLOYABLE` **for local/bounded-width measurement structure** (sparse `J`).
  Explicitly *not* a win for dense/global measurements (`J` fills → same `O(d²)` as dense).
- Precision family now complete: dense `P` `O(d²)` exact → block-diagonal `O(d·w)` approximate → information `J`
  `O(d·w)` **exact**.

## 4. Framework integration

Already lives in `src/online.rs` (extends the evolvent module, no new file/crate — anti-bloat satisfied);
re-exported from `src/lib.rs`. No duplication: it is the third member of the existing `*EvolventHead` family, not
a parallel scaffold.

## 5. Regression protection

`online::info_form_equals_dense_and_is_sparse` — asserts `info.solve()` matches the dense RLS `w` within 1e-3
**and** `nnz(J) < d²` (guards both the exactness identity and the sparsity claim). Full suite **175/0**, fmt +
clippy clean.

## 6. Source-of-truth update

- `canonical_components.json` — `InfoEvolventHead` added; `BlockEvolventHead` note points to E4 as its exact form.
- `canonical_findings.json` — F-EVO-6 added.
- Report + figure + results JSON on disk.
- Memory `project-nagare-evolvent-online-learning` updated.

## 7. Honest limitations carried forward

- Sparsity is a property of the **data**, not the head.
- Current solve is **dense** `O(d³)`; the `O(d·w²)` block-tridiagonal Cholesky that exploits `nnz` is **not yet
  implemented** — E4 measures the storage + exactness win, not a solve-time win. `nnz()` accounts the sparsity
  that solve would exploit.
- Single-output, linear-in-features.

## 8. Next (NOT yet authorized)

Sparse block-tridiagonal Cholesky (the actual solve-time win) + a **non-additive** cross-hyperedge target — the
discriminating test where BLOCK should lose materially and INFO/dense hold; update `J` through `hg_message`; SBSH
width certificate as the tractability guarantee.
