---
title: "Evolvent E5 — multifrontal (clique-tree) Cholesky over a branching hypergraph: exact as dense, solve-time win grows as (d/w)², online update touches only log₂(N) cliques"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, cholesky, multifrontal, junction-tree, hypergraph, positive]
---

# Evolvent E5 — the sparse Cholesky the hypergraph gives you

Date: 2026-07-15 · Nagare `221b05d`+ (kato15, 32-core; CPU-only) · continues E4 (F-EVO-6)

## Summary

E4 gave the information form `J` (exact, sparse `O(d·w)` storage) but solved it **densely** `O(d³)`. E5 builds the
sparse solve: **multifrontal Cholesky driven by the hypergraph's clique tree** (`JunctionTreeCholesky`). Each
hyperedge is a clique (a small tensor block); the factorization sweeps leaves→root, eliminating each clique's
residual and sending the **Schur complement over its separator** up to the parent — the message
`U = F_SS − F_SRᵀ F_RR⁻¹ F_RS`, a tensor contraction over the eliminated interior onto the separator boundary
(structurally the `hg_message` edge→node incidence contraction). Back-substitution sweeps root→leaves.

On a **branching** binary clique tree (so the factorization is a genuine tree, not a band), 5 seeds × 4 depths,
kato15:

| depth | d | cliques | DENSE R² `O(d³)` | **MF R² `O(d·w³)`** | BLOCK R² | storage (% dense) | **flop speedup** | locality (path / N) |
|---|---|---|---|---|---|---|---|---|
| 4 | 105 | 15  | 0.9994 | **0.9994** | 0.9992 | 10.7 % | **18×**   | 4 / 15  |
| 5 | 217 | 31  | 0.9994 | **0.9994** | 0.9993 | 5.3 %  | **77×**   | 5 / 31  |
| 6 | 441 | 63  | 0.9994 | **0.9994** | 0.9992 | 2.6 %  | **314×**  | 6 / 63  |
| 7 | 889 | 127 | 0.9994 | **0.9994** | 0.9992 | 1.3 %  | **1270×** | 7 / 127 |

Figure: `reports/figures/evolvent-multifrontal.png`.

## What is measured

- **The multifrontal solve is EXACT.** `MF R² == DENSE R²` (dense `InfoEvolventHead`) at every one of the 20
  `(depth, seed)` cells. Same solution, `J x = b`, computed by eliminating on the clique tree instead of on the
  full matrix. The three module regression tests pin it to `< 1e-3` of the dense reference including on the
  branching tree.
- **The solve-time win grows as `(d/w)²`.** Factorization flops `Σ_c |C|³` vs dense `d³/6`: **18× → 77× → 314× →
  1270×** as the tree deepens. This is E4's *storage* win turned into a *compute* win — the piece that was still
  `O(d³)` in E4. Storage (frontals `Σ|C|²` vs `d²`) drops in step: 10.7 % → 1.3 %.
- **Locality: an online update touches only its path to the root.** Regression test
  `online_update_only_perturbs_path_to_root` proves it exactly — after a leaf update, the cliques whose Cholesky
  factor changed are **precisely** `ancestors_inclusive(leaf)`, no others. Measured mean path = the tree depth
  (`4,5,6,7`) vs `N` cliques (`15,31,63,127`): `depth = log₂(N)`. So an incremental re-fire is `O(log N · w³)`, not
  `O(N · w³)` — the evolvent form of the factorization.
- **Block-diagonal (separator-dropping) trails.** BLOCK R² 0.9992–0.9993 vs MF/DENSE 0.9994 — a consistent
  ~0.0002 gap. It is *small* here on purpose (see scope): the coupling is real, MF keeps it exactly, block drops
  it.

## Why the gap is small — and why that is honest, not a null

The target is additive-over-cliques and the regime is **data-rich per clique** (`per=60` measurements ≫ clique
arity ~7–9), so each clique's local system is heavily over-determined and block-diagonal recovers nearly the same
weights without the cross-clique messages. That is a property of the *data regime*, not a limit of the method: MF's
exactness is regime-**independent**. The regime where separators are load-bearing — data-scarce per clique, or a
genuinely **non-additive** cross-clique target — is exactly the flagged discriminating test (still open); E5 does
not manufacture it. What E5 proves is the *solver*: exact = dense at up to 1270× fewer flops, with log-depth online
locality.

## Honest scope

- **Classical method.** The multifrontal method (Duff & Reid 1983) and its equivalence to junction-tree /
  variable-elimination inference are textbook. No algorithmic novelty is claimed. The framework contribution is the
  wiring: the hypergraph clique tree *is* the assembly tree, the separator Schur complements *are* the messages,
  and it is the exact sparse solver the E4 evolvent needed.
- **The flop win is an analytic count**, not a wall-clock benchmark. At these sizes a wall-clock `criterion` run
  would be dominated by allocation and cache effects, not the asymptotics; the flop ratio `Σ|C|³ / (d³/6)` is the
  honest characterization of the factorization cost. A wall-clock benchmark belongs at larger `d` with a
  contiguous-storage rewrite — noted as follow-up.
- **The incremental re-fire is proven-local but not yet implemented as an optimized path.** E5 measures the
  locality (which factors change) that *makes* an `O(log N · w³)` update possible; the optimized
  `refactorize_path` that recomputes only the path is not built (that would be the optimization the measurement
  justifies — §3 order: measure first).
- **Requires a valid clique tree** (running intersection, separator ⊆ parent, SPD `J`) — checked at construction.
  A measurement coupling non-adjacent cliques breaks the tree structure; the SBSH bounded-width certificate is what
  guarantees the tree exists with width `w`.
- Single-output, linear-in-features (as the whole evolvent line).

## Tests / gates

| item | result |
|---|---|
| `junction_tree::single_clique_equals_dense` | pass (one clique ≡ dense Cholesky) |
| `junction_tree::branching_tree_equals_dense_at_bounded_width` | pass (MF == dense on a forked tree; frontals < d²) |
| `junction_tree::online_update_only_perturbs_path_to_root` | pass (changed factors == path-to-root, exactly) |
| `examples/evolvent_multifrontal` (5 seeds × 4 depths, kato15) | table above |
| full suite | **178 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/junction_tree.rs` | new module — `JunctionTreeCholesky` (multifrontal Cholesky), `Clique`, `balanced_binary_tree`, `solve`/`solve_block_diagonal`/factor accounting/locality accessors + 3 tests |
| `src/lib.rs` | module + re-exports |
| `examples/evolvent_multifrontal.rs` | new — MF vs dense vs block on a branching hypergraph; `--depth` sweep |
| `scripts/dev/plot_evolvent_multifrontal.py`, `reports/figures/evolvent-multifrontal.png`, `reports/figures/evolvent_e5_results.json` | figure + data |

## The precision family, complete

| form | storage | solve | coupling | exact? |
|---|---|---|---|---|
| dense `P` (E0/E1) | `O(d²)` | `O(d³)` | all | yes |
| block-diagonal (E3) | `O(d·w)` | `O(d·w²)` | within-block | only if separable |
| information `J` (E4) | `O(d·w)` | `O(d³)` (dense) | all | yes |
| **multifrontal `LLᵀ` (E5)** | **`O(d·w²)`** | **`O(d·w³)`** | **all** | **yes** |

E5 is the corner that was missing: exact **and** cheap to solve **and** cheap to update online.

## Provenance

- Nagare on kato15 (32-core, RTX6000; CPU-only), `source ~/.cargo/env`. Branching binary clique tree
  (`sep=2, res=3`, width ≈ 8), local measurements homed at cliques (`per=60`), `y = φ·w_true + 0.05 noise`.
  Depths 4–7 (d = 105–889), seeds 0–4. Reproduce:
  `cargo run --release --example evolvent_multifrontal -- --depth=D --seed=S`.

## Next

- **Contiguous-storage rewrite + wall-clock `criterion` benchmark** at large `d` to confirm the analytic flop win
  as measured time.
- **Optimized `refactorize_path`** (recompute only the leaf→root path) — the incremental online solver the
  locality result justifies.
- **A non-additive / data-scarce cross-clique target** — the discriminating test where block-diagonal loses
  materially and MF/dense hold (the block-vs-info question, now on the branching tree).
- Route the separator message literally through `hg_message` on a signed hypergraph.
