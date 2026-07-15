---
title: "Evolvent E8 — the width/representability boundary: a non-additive cross-clique term is fittable only by a clique wide enough to host it"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, junction-tree, treewidth, representability, non-additive, positive]
---

# Evolvent E8 — the last open knob: a genuinely non-additive cross-clique term

Date: 2026-07-15 · Nagare `a19e46e`+ · **run locally on the Mac (arm64, CPU), away from kato15**

## Summary

E6/E7 measured the *estimation* gap of dropping the separator coupling (a data-pooling effect). E8 tests the last
open knob — a **genuinely non-additive** term — and it is a different phenomenon entirely: a **representability**
limit. Each "triple" k carries an explicit product feature `prod_k = rk0·rk1`; the target is
`y = Σ_k [w0·rk0 + w1·rk1 + β·prod_k] + w_h·h + noise`, with `β` dialing the non-additive strength. Three arms,
5 seeds, Mac:

| β | MF-WIDE R² (width-4 tree) | DENSE-WIDE R² | NARROW R² (linear-only) |
|---|---|---|---|
| 0.0 | 0.9985 | 0.9985 | 0.999 |
| 0.5 | 0.9985 | 0.9985 | 0.926 |
| 1.0 | 0.9985 | 0.9985 | 0.748 |
| 2.0 | 0.9985 | 0.9985 | 0.376 |
| 4.0 | 0.9986 | 0.9986 | **0.060** |

MF storage = 14 % of dense (width 4). Figure: `reports/figures/evolvent-width.png`.

## What is measured

- **The multifrontal clique tree handles the non-additive term exactly — if the clique is wide enough to host
  it.** The product `rk0·rk1` lives in a clique that contains both `rk0` and `rk1` (width 4: `{rk0, rk1, prod_k,
  h}`). `MF-WIDE R² == DENSE-WIDE R²` at every `(β, seed)` cell, flat at ~0.998 regardless of β — the width-4
  multifrontal solve is exact for the non-additive target, at **14 % of the dense storage**. Non-additivity is not
  a problem for the junction-tree solver *per se*; it is a problem for **too-narrow** trees.
- **The narrow (linear-only) model omits the term — regardless of data.** NARROW cannot host `prod_k` (no clique
  contains both `rk0` and `rk1` in the width-2 world), so it structurally has no weight to fit β. Its R² collapses
  as β grows: 0.999 → 0.926 → 0.748 → 0.376 → **0.060**. This is **not** an estimation gap that more data would
  close (the E6/E7 mechanism); with `per=40` this is the data-rich regime and the narrow model still fails. The
  term is simply **not in its hypothesis space**.
- **The required treewidth = the target's interaction order.** A degree-2 interaction (`rk0·rk1`) needs a clique of
  width ≥ 3 (both operands + the interaction); the width-4 tree covers it. This is the concrete meaning of the SBSH
  **bounded-width certificate**: it certifies that the tree's width covers the interaction order of the data — the
  precondition under which the whole E4–E7 line (exactness at `O(d·w³)`) is valid.

## How this closes the arc

The block-vs-info question had two possible mechanisms; E6/E7/E8 separate them cleanly:

| phenomenon | mechanism | closes with |
|---|---|---|
| separator-**estimation** gap (E6) | pooling shared vars' evidence | more data (scarcity-dependent, F-EVO-8) |
| separator-**sharing** (E7) | threshold on sharing | — (saturates, F-EVO-9) |
| **non-additive** term (E8) | **representability** — width < interaction order | **wider cliques**, not more data (F-EVO-10) |

So "does a non-additive cross-clique feature break the plateau?" — it doesn't touch the plateau (an estimation
effect); it is a **different axis**: representability, governed by treewidth vs interaction order, and the
multifrontal solver handles it exactly at the right width.

## Honest scope

- **The product is an explicit feature.** As with the whole evolvent line, the model is linear-in-features; the
  clique tree makes the interaction *affordable and exactly solvable* at bounded width, it does not *discover* the
  interaction. NARROW's collapse is because the feature is absent from its basis, which is the honest content of
  "too narrow."
- **A degree-2 interaction, single arity.** Higher-order interactions need proportionally wider cliques
  (width ≥ order+1); untested but immediate.
- **`β` is a controlled dial, not fit from real data** — the point is the representability threshold, not a claim
  about any particular dataset's interaction strength.

## Tests / gates

| item | result |
|---|---|
| existing `junction_tree::*` (incl. star, single-clique-covers-block-diagonal) | pass |
| `examples/evolvent_width` (5 β × 5 seeds, Mac) | table above |
| full suite | **179 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `examples/evolvent_width.rs` | new — MF-wide / dense-wide / narrow race on a non-additive (product) target, `--beta10` sweep |
| `scripts/dev/plot_evolvent_width.py`, `reports/figures/evolvent-width.png`, `reports/figures/evolvent_e8_results.json` | figure + data |

Note: no `src/` change — E8 reuses `JunctionTreeCholesky` and `InfoEvolventHead` directly, so no new regression test
in `src/` (the example is the integration evidence; the solver's exactness is already guarded by the E5 tests).

## Provenance

- Nagare on Hajdus-MacBook-Pro (arm64, CPU-only), `cargo 1.96.1`. K=12 triples (d=37, narrow d=25), width-4 star
  clique tree hosting the product, `per=40` (data-rich), `β ∈ {0, 0.5, 1, 2, 4}`, `y = Σ[w0 rk0 + w1 rk1 + β
  rk0·rk1] + w_h h + 0.05 noise`, seeds 0–4. Reproduce:
  `cargo run --release --example evolvent_width -- --beta10=B --seed=S`.
- **kato15 not synced this session** (working away from it); origin `main` is ahead — kato15 will fast-forward on
  next pull.

## Next

- Higher-order interactions (degree ≥ 3) → width ≥ order+1; confirm the treewidth = interaction-order relation.
- The E5 engineering follow-ups (contiguous-storage rewrite + wall-clock `criterion`; optimized `refactorize_path`).
- Route the separator message literally through `hg_message`; wire the SBSH width certificate as an explicit
  precondition check on `JunctionTreeCholesky::new`.
