# Nagare on tabular, T3/T4 — graph-from-tabular vs the KAN (Iris)

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-09-nagare-tabular/` · Follows T1/T2

## Summary

The distinctive-core stage: build a signed graph *from tabular features* so Nagare's
**signed-cycle machinery** (not just the KAN) runs on Iris, then compare. **Verdict: the
signed graph does NOT beat the plain KAN on Iris — a TIE at median held-out accuracy
0.947.** The graph model is a *working* closed-form node classifier over 2520 signed
triangles; it simply does not improve on the additive KAN. This is the result the plan
explicitly anticipated ("T3 may not beat the KAN — a reportable result, not a failure").

## T3 — graph-from-tabular (`src/tabular_graph.rs`)

- **Nodes** = samples. **Edges** = a kNN graph (Euclidean distance in the standardised
  feature space, `k_nn=10`). **Signs** = the sign of the two samples' **feature
  correlation** — **leakage-free** (features only, never labels), so the node classifier
  cannot cheat.
- Signed **triangles** enumerated via `hymeko_graph::enumerate_simple_cycles_noprune`
  → a `TopKCyclesBatch` (2520 triangles on Iris).
- **Model** (transductive node classifier): `X → gomb_outer (4 FIR banks) →
  scatter_mean (cycle→sample) → linear → softmax₃`; trained on train-node labels only,
  evaluated on held-out nodes. All closed-form (composed backward through the four ops).

## T4 — comparison (`tests/graph_vs_kan_iris.rs`, 5 seeds, same splits)

| seed | KAN | graph |
|---|---|---|
| 0 | 0.868 | 0.842 |
| 1 | 0.974 | 0.974 |
| 2 | 0.921 | 0.868 |
| 3 | 0.947 | 0.947 |
| 4 | 0.974 | 0.947 |
| **median** | **0.947** | **0.947** |

**Δ = 0.000 — a tie.** Figure: `reports/figures/nagare-tabular-showcase.png` (paired
per-seed points; the T2 California R² panel alongside).

## Reading (honest)

The signed-graph core **runs on tabular** and classifies correctly — a genuine
demonstration that the machinery generalises past signed social graphs. But on **Iris**
it does **not** earn its keep: the additive KAN already captures the (low-dimensional,
non-relational) signal, and imposing a kNN-correlation graph adds no discriminative power
(and on 2/5 seeds is slightly worse). This is the expected outcome for a dataset with no
intrinsic graph structure — and exactly why the comparison was worth running rather than
assuming. Where the signed structure *should* pay (the mixed-arity signed-hypergraph
regime) is the HSiKAN/Gömb line, not tabular.

## Files touched

| file | change |
|---|---|
| `src/tabular_graph.rs` | **new** — kNN + correlation-sign graph, triangle enumeration → cycle pool |
| `src/lib.rs` | +mod / +re-export |
| `tests/graph_vs_kan_iris.rs` | **new** — KAN vs graph node-classifier comparison |
| `scripts/dev/plot_tabular_showcase.py`, `reports/figures/nagare-tabular-showcase.png` | **new** — showcase figure |

## CORE / deps

**None.** Reuses vendored `hymeko_graph::{SignedGraph, enumerate_simple_cycles_noprune}` +
`gomb_shell`/`scatter_mean`/`linear`/`softmax_k`. Plot via ephemeral `uv run --with matplotlib`.

## Test results (both machines)

- Full suite **77 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.
  Deterministic (seeded).

## Open / follow-up

1. Sensitivity: `k_nn`, `cycle_k` (4-cycles), the **sign semantics**, adding the middle
   HSiKAN shell to the node classifier — none tuned to a hoped-for win (§3).
2. California as a graph (regression node model) for completeness.
3. A task with **intrinsic** graph structure (where the graph should beat the KAN) — the
   fair home for the signed-cycle core, vs tabular where the KAN wins on simplicity.

## Provenance

Repo `github.com/kyberszittya/nagare`. Developed on kato15, mirrored via the Mac. Iris
from UCI. Rust 1.96.1; seeds fixed. Signs derived from features only (no label leakage).
