---
title: "Nagare SBSH Phase 1 — the dynamic quadtree as an FD-clean lib op"
date: 2026-07-11
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, sbsh, gomb-soma, object-detection, quadtree, closed-form-op]
---

# SBSH Phase 1 — `quadtree` op (structural build + differentiable node-pool)

Date: 2026-07-11 · Mac · Nagare at `83023d3`+ · CPU

## Summary

Promoted the SBSH dynamic quadtree (both hinges validated in the smoke) to a first-class Nagare op,
`src/ops/quadtree.rs`. Two pieces, split by the closed-form contract:

- **`quadtree_build(energy, cfg)` — structural, no backward** (the `cpml_tier` discipline): a per-pixel
  energy map → leaf cells + a per-pixel leaf `assign`. Splits a cell into 4 while its mean energy exceeds
  `thresh` (up to `max_depth`, min side `2·min_side`). The split is a fixed structural decision; no
  gradient flows through it.
- **`node_pool_forward` / `node_pool_backward` — differentiable, FD-verified**: per-cell mean-pool of a
  per-pixel feature field, so gradients flow from a per-node head/loss back to the (learned) field.
  `node[c] = mean_{p∈c} field[p]`; `∂L/∂field[p] = (1/N_{assign[p]})·∂L/∂node[assign[p]]` (the
  `scatter_mean` adjoint).

Single-image (`n=1`) — the tree is per-image. Plan bundle: `docs/plans/2026-07-11-sbsh-quadtree/`
(tex/pdf/tikz/mmd, gitignored).

## Tests (all layers)

| layer | test | result |
|---|---|---|
| structural (unit) | `concentrates_on_high_energy_region` | ok — high-energy quadrant gets finer cells than the flat region; `assign` covers every pixel once (`Σ N_c = g²`); all `assign < cells.len()` |
| unit (FD) | `node_pool_backward_matches_fd` | ok — directional-derivative check (4 dirs), abs+rel tol |
| unit | `uniform_field_pools_to_constant` | ok — constant field → constant nodes; grad distributes by `1/N_c` |
| integration | `learns_through_node_pool` (`tests/quadtree_learn.rs`) | ok — `linear → node_pool → MSE`, gradient descent through the pool converges to a reachable target (loss < 0.05·initial) |
| full suite | `cargo test --release` | **125 passed / 0 failed** (+4) |
| gate | `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Performance

`quadtree_build` is `O(g²)` amortised (each pixel visited `O(depth)` times); `node_pool` is `O(g²·d)`.
At `g=96, d=16` this is sub-millisecond and far below the plan's 2 ms budget — a structural partition plus
a single mean-pool, not a hot path. No dedicated criterion bench (proportionate: the op has no inner
numerical kernel to profile); complexity is the contract.

## What this unlocks / does NOT do

- **Unlocks:** a differentiable per-node feature (mean-pool) on a content-adaptive partition — the
  substrate the detector head sits on. The upstream feature field can now be a learned stem trained
  end-to-end through the pool.
- **Does not do (later phases):** the **canonical-aligned orientation descriptor** (Phase 2 — its own op
  with backward; the mean-pool here is a *generic* aggregator, not the rotation-robust descriptor from the
  H2 fix); the oriented-bbox head + KLD loss; batching (`n>1`). The smoke's local `phase_desc` (canonical
  alignment) is the Phase-2 op.

## Caveats

- Empty cells (`N_c=0`) are guarded (grad 0); cannot occur from `quadtree_build` (every leaf owns ≥ 1
  pixel) but `node_pool` is defensive for arbitrary `assign`.
- The 2nd-moment canonical estimator's near-square/symmetric ambiguity (from the smoke) is a **Phase-2**
  concern; this op is descriptor-agnostic.

## Files touched

| file | change |
|---|---|
| `src/ops/quadtree.rs` | new op — `quadtree_build`, `node_pool_forward/backward`, `Quadtree`, `QuadtreeConfig` + 3 unit tests |
| `tests/quadtree_learn.rs` | integration — learns through the node-pool |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export |

No new deps, no CORE.YAML (repo has none).

## Next

- **Phase 2** — the canonical-aligned orientation descriptor as a lib op (with FD backward) + a general
  canonical estimator that degrades gracefully on symmetric objects.
- **Phase 0** — 4-query novelty search before any external claim.
- Then the oriented-bbox head (surrogate KLD loss) + node→object assignment; no full detector until the
  descriptor op is FD-clean.

## Provenance

- Mac (Apple Silicon), Nagare `83023d3`+; CPU. No data, no GPU.
- Reproduce: `cargo test --release quadtree` and `cargo test --release --test quadtree_learn`.
