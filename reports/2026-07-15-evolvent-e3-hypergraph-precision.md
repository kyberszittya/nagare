---
title: "Evolvent E3 — the pairwise item vs the hypergraph tensor: a hyperedge-clique precision recovers higher-order interactions the pairwise precision cannot, at O(d·w)"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, precision, hypergraph, higher-order, block-rls, positive]
---

# Evolvent E3 — the pairwise precision has a shape problem

Date: 2026-07-15 · Nagare `1d63e68` · CPU

## Summary

The dense RLS precision `P` is a **pairwise** object, and that costs us twice: it is `O(d²)`, and it can only hold
*second-order* structure. This tests the alternative — a **hyperedge-clique (block-structured) precision**
(`BlockEvolventHead`: one small `b_e×b_e` block per hyperedge, shared residual/denominator, `O(Σ b_e²)=O(d·w)`).
On data with a genuine 3-way hypergraph coupling, 5 seeds:

| layout | dense-PAIRWISE (no 3-way) | dense-HYPEREDGE `O(d²)` | **block-HYPEREDGE `O(d·w)`** | precision nnz |
|---|---|---|---|---|
| disjoint | R² **0.16** | R² 0.997 | R² **0.997** | 980 vs 19600 (**20×**) |
| overlap chain | R² **0.17** | R² 0.997 | R² **0.997** | 980 vs 19600 (**20×**) |

Figure: `reports/figures/evolvent-hypergraph.png`.

## What is measured

- **The pairwise item cannot hold the 3-way term.** `x_a x_b x_c` is **L²-orthogonal** to every pairwise feature
  (for mean-zero inputs `E[x_a x_b x_c · x_a x_b] = E[x_a²]E[x_b²]E[x_c] = 0`), so a pairwise model captures *none*
  of the 3-way variance — measured R² ≈ 0.15 on a 3-way-dominant target, matching a numpy exact-LS ceiling of the
  same. This is "the pairwise item costs us" made quantitative: a genuine higher-order interaction is invisible to
  a pairwise precision.
- **The hyperedge-clique precision recovers it — exactly and cheaply.** Block-hyperedge R² **0.997 = dense-hyperedge
  0.997**, at **20× less precision storage** (`Σ b_e² = 980` vs `d² = 19600`). The 20× is `d/b` — the advantage
  grows with the number of hyperedges.
- **Holds on overlapping hypergraphs too.** The chain layout shares a node between adjacent edges, so the dense
  precision has cross-block terms the block form drops — but for an *additive-over-hyperedges* target that
  separator coupling is negligible, and block still matches dense (0.997).

## Why this is the right alternative

The block precision *is* the hypergraph made numerical: the interaction unit is the **hyperedge** (a small tensor
block), not the feature-pair (a matrix entry). It attacks both costs the pairwise `P` charges — quadratic size and
second-order-only expressiveness — and it ties the evolvent line to Nagare's signed-hypergraph substrate: the
block updates are local (the shape `hg_message` applies), and the tractability guarantee for a genuine
junction-tree form is exactly SBSH's **bounded-width** certificate.

## Honest scope + a bug found

- Block precision drops cross-block coupling; it is *exact* only when the true precision is block-diagonal
  (feature-disjoint / additive targets). Genuine cross-hyperedge (non-additive) structure needs the junction-tree
  (block-tridiagonal) form — the next step.
- The higher-order term must be an **explicit feature** (the model is linear-in-features); the block structure then
  makes higher-order *affordable*, it doesn't discover it.
- **RNG bug (fixed here, `1d63e68`):** the evolvent examples' `Rng` used `>>33 / u32::MAX` → `f ∈ [0,0.5)`, biasing
  inputs to `[-1,0)`. That broke the orthogonality (biased inputs let pairwise partially fit the 3-way, masking
  the effect); fixed to `>>32 / 2³²`. Same bug affected E1's synthetic datasets — corrected there (F-EVO-4).

## Tests / gates

| item | result |
|---|---|
| `online::single_block_equals_dense` | pass (block with 1 block ≡ dense EvolventHead) |
| `online::block_matches_dense_when_separable` | pass (matches dense on separable data, fewer than d² stored) |
| `examples/evolvent_hypergraph` (5 seeds) | table above |
| full suite | **174 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/online.rs` | new `BlockEvolventHead` (block-diagonal precision RLS) + 2 tests; RNG doc |
| `examples/evolvent_hypergraph.rs` | new — pairwise-vs-hyperedge-tensor race, disjoint/overlap; RNG fix |
| `examples/evolvent_{stream,bench}.rs` | RNG fix (shared bug) |
| `scripts/dev/plot_evolvent_hypergraph.py`, `reports/figures/evolvent-hypergraph.png` | figure |

## Next

- **Junction-tree (block-tridiagonal) precision** for genuine cross-hyperedge coupling — the exact bounded-width
  form, updated through `hg_message`, with the SBSH width certificate as the tractability guarantee.
- A general (non-additive) hypergraph target to measure what the separator coupling is worth.

## Provenance

- Nagare `1d63e68`; CPU. Synthetic: HE=20 hyperedges of 3 nodes, target `sum_e [strong 3-way + small pairwise +
  linear]`, mean-zero inputs; disjoint (n=60) and overlap-chain (n=41). 5 seeds. Reproduce:
  `cargo run --release --example evolvent_hypergraph -- --seed=N`.
