# Assimilation — Evolvent E5 (multifrontal / clique-tree Cholesky)

Date: 2026-07-15 · lifecycle per `feedback-assimilation-lifecycle-protocol`

## 1. Experiment → evidence

E5 (`examples/evolvent_multifrontal.rs`, kato15, 5 seeds × 4 depths) builds the sparse solve E4 was missing:
**multifrontal Cholesky driven by the hypergraph clique tree** (`JunctionTreeCholesky`). On a branching binary
clique tree (a genuine fork, not a band):

- `MF R² == DENSE R²` at every `(depth, seed)` cell — exact.
- factorization flop speedup `Σ|C|³ / (d³/6)` = **18× → 77× → 314× → 1270×** as `d` 105→889 (grows `(d/w)²`);
  storage 10.7 % → 1.3 %.
- online update touches **exactly** its clique's path to the root (mean path = `log₂(N)`) → incremental re-fire
  `O(log N · w³)`.

Evidence: `reports/2026-07-15-evolvent-e5-multifrontal.md`, `reports/figures/evolvent-multifrontal.png`,
`reports/figures/evolvent_e5_results.json`.

## 2. Novelty classification

`NEW_CANONICAL_CAPABILITY` (F-EVO-7) — **classical method, stated plainly**: the multifrontal method (Duff & Reid
1983) and its identity with junction-tree / variable elimination are textbook. No algorithmic novelty is claimed.
The framework contribution is the wiring: the hypergraph clique tree *is* the assembly tree; the separator Schur
complements *are* the messages (`hg_message`-shaped); it is the exact sparse solver the E4 evolvent needed.

## 3. Canonical decision

Register `JunctionTreeCholesky` — `DEPLOYABLE` for bounded-width (junction-tree) SPD precisions. It completes the
evolvent precision family and delivers the F-EVO-6 open item.

## 4. Framework integration

- **Discovery pass (§6.1 / §6.5 #12) run first:** grepped `cholesky|junction|clique|multifrontal|frontal|schur`
  across `src/` and `examples/` — only doc-comment mentions, no existing solver. Confirmed `hg_message` operates on
  signed-hypergraph *feature* vectors, not Schur-complement blocks, so the contraction is implemented directly with
  the correspondence stated (not a forced `hg_message` call that wouldn't fit).
- **New module `src/junction_tree.rs`** — a sparse linear-algebra solver is a distinct concern from the online RLS
  heads, and `online.rs` was already 609 LOC (§6.5 #4). Not bolted into `online.rs`.
- `balanced_binary_tree` promoted to a public helper so the example and the test share one builder (no
  duplication).

## 5. Regression protection

Three tests pin the two load-bearing identities: `single_clique_equals_dense` and
`branching_tree_equals_dense_at_bounded_width` (MF == dense, frontals < d²), and
`online_update_only_perturbs_path_to_root` (changed factors == path-to-root, exactly — the locality claim).
Full suite **178/0**, fmt + clippy clean.

## 6. Source-of-truth update

- `canonical_components.json` — `JunctionTreeCholesky` added; `InfoEvolventHead` note points to E5 as its
  solve-time delivery.
- `canonical_findings.json` — F-EVO-7 added.
- Report + figure + results JSON on disk.
- Memory `project-nagare-evolvent-online-learning` updated.

## 7. Honest limitations carried forward

- Flop win is an **analytic count**, not wall-clock — a contiguous-storage rewrite + `criterion` at large `d` is
  the follow-up.
- Incremental re-fire is **proven-local** but the optimized `refactorize_path` is not built (measure-before-
  optimize, §3).
- Requires a valid clique tree; the SBSH width certificate is the tractability guarantee.
- The small BLOCK gap (~0.0002) is a **regime** property (additive + data-rich per clique), not a method limit;
  the non-additive / data-scarce discriminating target is still open.

## 8. Next (NOT yet authorized)

Contiguous-storage rewrite + wall-clock benchmark; optimized `refactorize_path`; the non-additive/data-scarce
cross-clique discriminating target; route the separator message literally through `hg_message`.
