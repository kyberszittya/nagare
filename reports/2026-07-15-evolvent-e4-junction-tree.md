---
title: "Evolvent E4 — the junction-tree (information-form) precision: exact as the dense RLS, sparse as the hypergraph, at O(n·w)"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, precision, junction-tree, information-form, gmrf, positive]
---

# Evolvent E4 — the sparse-tensor conjugation of the precision

Date: 2026-07-15 · Nagare (kato15, 32-core; CPU-only) · continues E3 (F-EVO-5)

## Summary

E3 showed the dense RLS precision `P` is a *pairwise* object that costs twice — `O(d²)` storage and
second-order-only expressiveness — and that a **block-diagonal** (hyperedge-clique) precision recovers
higher-order structure at `O(d·w)` but **drops the cross-block (separator) coupling**. E4 closes that gap with
the **information form** (`InfoEvolventHead`): accumulate `J = ΦᵀΦ + ridge·I`, `b = Φᵀy` and solve `J w = b`.
Every update `J += φφᵀ` touches only `φ`'s support, so for **local (bounded-width) measurements** `J` stays
**sparse** — while the solution is **exact** (`= dense RLS`, keeping all coupling).

The user's ask — *"make P sparse and tensor-like — a real conjugation between the hypergraph and its tensor
interface"* — is exactly this: `J` **is** the hypergraph made numerical (the sparse information matrix = the
signed-hypergraph incidence structure), updated locally, solved exactly.

5 seeds × 4 sizes (`W=6`, `NS=8000`), kato15:

| n | DENSE R² `O(n²)` | **INFO R² `O(n·w)`** | BLOCK R² `O(n·w)` | info-matrix nnz | dense nnz | **info storage** |
|---|---|---|---|---|---|---|
| 48  | 0.9998 | **0.9998** | 0.9997 | 498  | 2 304   | **21.6 %** |
| 96  | 0.9998 | **0.9998** | 0.9998 | 1 026 | 9 216   | **11.1 %** |
| 192 | 0.9998 | **0.9998** | 0.9998 | 2 082 | 36 864  | **5.6 %** |
| 384 | 0.9998 | **0.9998** | 0.9996 | 4 194 | 147 456 | **2.8 %** |

Figure: `reports/figures/evolvent-junction.png`.

## What is measured

- **The information form is EXACT.** At *every* one of the 20 `(n, seed)` cells, `INFO R² == DENSE R²` to the
  printed precision. This is not an approximation that happens to be good — `J w = b` is the same normal-equation
  system the dense RLS `P` inverts; the info form just never forms `P = J⁻¹`. It keeps all cross-hyperedge
  coupling.
- **`J` is sparse, and the sparsity scales.** The info-matrix storage falls **21.6 % → 11.1 % → 5.6 % → 2.8 %**
  as `n` doubles — it **halves each doubling**, because `nnz(J) = O(n·w)` (banded: two features co-occur only when
  a measurement window covers both) while dense is `O(n²)`. The ratio is `O(w/n)` and the advantage *grows* with
  the model.
- **Block-diagonal drops the separator coupling — visibly, if slightly.** BLOCK trails INFO by ≤ 0.0003 (0.9996 vs
  0.9998 at n=384). On this *additive-over-windows* target the dropped coupling is worth little, so block is a fine
  approximation — but it is an approximation, and INFO is exact at the same storage order. When the cross-hyperedge
  structure is load-bearing (non-additive), that gap is where block loses and the junction-tree form is required.

## Why this is the right form

The three precisions are one family at increasing fidelity, all `O(n·w)` except dense:

| form | storage | coupling kept | exact? |
|---|---|---|---|
| dense `P` (E0/E1) | `O(n²)` | all | yes (but quadratic) |
| block-diagonal (E3) | `O(n·w)` | within-block only | only if `P` block-diagonal |
| **information `J` (E4)** | **`O(nnz)=O(n·w)`** | **all** | **yes** |

`J` is the natural tensor interface to the hypergraph: a nonzero `J[i,k]` is exactly an edge between features
`i,k` (a Gaussian-MRF conditional dependence), and a local measurement writes only into its own clique block. A
sparse (block-tridiagonal / junction-tree) Cholesky solves it in `O(n·w²)`; here a dense solve is used for
simplicity and `nnz()` reports the sparsity that solve would exploit — the SBSH **bounded-width** certificate is
the tractability guarantee for that step.

## Honest scope

- **`InfoEvolventHead` is a classical primitive** — the information filter / Gaussian-MRF precision accumulator /
  junction-tree form (Kalman information filter; Lauritzen–Spiegelhalter 1988). E4 does **not** claim the info
  form is novel. The framework contribution is narrow and concrete: an **exact + sparse evolvent head at O(n·w)**
  that realises the hypergraph↔tensor conjugation, extending the evolvent line past the pairwise/block limits.
- **The sparsity is a property of the DATA, not the head.** `J` is sparse only when measurements are local
  (bounded window). Dense (global) measurements fill `J` and the info form has no storage advantage over dense —
  it is then the same `O(n²)`. This is why status is DEPLOYABLE **for local/bounded-width structure**, not
  unconditionally.
- **Current solve is dense `O(n³)`.** The `nnz` accounting proves the sparsity is there; exploiting it needs the
  sparse Cholesky (block-tridiagonal), which is the next step. E4 measures the *storage* win and the *exactness*;
  it does not yet measure a *solve-time* win.
- **Linear-in-features.** As with all evolvent heads, higher-order terms must be explicit features; `J` makes them
  *affordable*, it does not discover them.

## Tests / gates

| item | result |
|---|---|
| `online::info_form_equals_dense_and_is_sparse` | pass (info.solve() == dense.w within 1e-3; nnz < d²) |
| `cargo test --release online` (kato15 + Mac) | 7 / 0 |
| full suite | **175 / 0** · fmt + clippy clean |
| E4 sweep (5 seeds × 4 n, kato15) | table above |

## Files touched

| file | change |
|---|---|
| `src/online.rs` | new `InfoEvolventHead` (information-form RLS: `new`/`update`/`nnz`/`solve`/`predict`) + 1 test |
| `src/lib.rs` | re-export `InfoEvolventHead` |
| `examples/evolvent_junction.rs` | new — dense / block / info race on local-measurement chain, `--n` sweep |
| `scripts/dev/plot_evolvent_junction.py`, `reports/figures/evolvent-junction.png` | figure |
| `reports/figures/evolvent_e4_results.json` | aggregated kato15 results |

## Provenance

- Nagare on kato15 (32-core, RTX6000; CPU-only run), `source ~/.cargo/env`. Data: path of `n` features, each
  sample a window of `W=6` consecutive features with *correlated* (random-walk) values, target linear in the
  window + 0.05 noise; `NS=8000`, 3:1 train/test. Seeds 0–4. Reproduce:
  `cargo run --release --example evolvent_junction -- --n=N --seed=S`.
- RNG: fixed `>>32 as u32 / 2³²` form (the E3 bias bug is not present here).

## Next

- **Sparse block-tridiagonal Cholesky** of `J` (the actual `O(n·w²)` solve) — turns the measured storage win into a
  measured solve-time win; validate the width bound with the SBSH certificate.
- A **non-additive** cross-hyperedge target to measure what the separator coupling is worth (where BLOCK loses
  materially and INFO/dense hold) — the discriminating test block-vs-info deserves.
- Update `J` **through `hg_message`** (higher-order tensor apply) so the info accumulation runs on the signed-
  hypergraph substrate directly.
