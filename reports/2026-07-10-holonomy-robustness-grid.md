---
title: "Nagare — holonomy channel robustness grid (data-seed × init-seed): the definitive verdict"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, cpml, signed-link, robustness, verdict]
---

# Holonomy channel — the (data-seed × init-seed) robustness grid

Date: 2026-07-10 · Mac (author box) · Nagare at `60caed6`+ · **5 data-seeds × 5 init-seeds = 50 cells**

## Why this run

The single-head holonomy result flip-flopped as the initialization seed changed (`96e29c4` said Alpha
robust / OTC marginal; the multi-head run said the opposite on OTC). The honest verdict needs the
**initialization varied independently of the data split** — a 2-D grid — because a few data-seeds at one
init samples only a sliver of a noisy surface. Added a `--grid --init I` fast path (inner core L=3 vs
+holonomy M=1, same model init, `ms = sd + I·7919`), swept **5 data-seeds × 5 init-seeds** on
Bitcoin-Alpha and Bitcoin-OTC (50 cells).

## Result — the grid REVERSES the single-init picture

| graph | median ΔAUROC | mean | IQR | helps / hurts / tie | min … max |
|---|---|---|---|---|---|
| **Bitcoin-Alpha** | **−0.0004** | +0.00001 | [−0.0045, +0.0037] | 10 / 12 / 3 | −0.0147 … +0.0129 |
| **Bitcoin-OTC** | **+0.0035** | +0.0029 | [+0.0006, +0.0048] | 19 / 6 / 0 | −0.0052 … +0.0158 |

(Δ = AUROC(inner + holonomy M=1) − AUROC(inner core), paired per cell.) Figure:
`reports/figures/holonomy-robustness-grid.png`.

- **Bitcoin-OTC — robustly POSITIVE.** median +0.0035, **19/25 cells help**, and the **IQR is entirely
  above 0** ([+0.0006, +0.0048]). The holonomy channel adds real, init-robust signal here.
- **Bitcoin-Alpha — a WASH.** median −0.0004, essentially a coin-flip (10 help / 12 hurt), IQR straddles
  0. No robust effect.

This is the **opposite** of the single-init reads (`96e29c4`: "Alpha robust +0.008, OTC marginal";
multi-head: "M=1 Alpha +0.008, OTC wash"). Both were unreliable draws. **Corrected, definitive verdict:
the rotor-holonomy channel robustly helps on Bitcoin-OTC and is a wash on Bitcoin-Alpha.**

## Why the earlier reads misled (mechanism)

The **inner core's own AUROC swings ±0.015 across init seeds** (Alpha 0.862–0.893 in the grid), while
the holonomy Δ is ±0.005. So the signal is *small relative to init variance*, and 3 data-seeds at one
init offset lands anywhere on that noisy surface — hence the flip-flop. Only the 2-D grid (which averages
over both nuisance axes) recovers the stable Δ. **Lesson, now firmly on record: for a small effect,
vary BOTH the data split and the initialization before any robustness claim — a few data-seeds is not
enough.** (Third instance of the favorable-draw trap this project: single_hsikan panic, option-msdm,
and now holonomy.)

## Why OTC and not Alpha (inferred)

OTC is the larger, denser graph (V=5881, ~25k triangles vs Alpha V=3783, ~17k). The holonomy is a
per-cycle feature, so more (and richer) triangles give it more signal to aggregate; on the sparser Alpha
it adds noise about as often as signal. Consistent with "the holonomy needs cycle density to pay off."

## Standing verdict on the whole holonomy arc

- The **reframe was correct** (Clifford-FIR as an order-sensitive rotor holonomy / transmission channel,
  not an outer compressor) and the **`rotor_holonomy` op is sound** (FD-verified).
- The **payoff is real but modest and graph-conditional**: a robust **+0.0035 median on OTC** (denser
  graph), a wash on Alpha. Not a universal win; a genuine, init-robust win where cycle density is high.
- Multi-head does not raise the ceiling (prior report). The `phase_pool` `|DFT|`-invariant variant is
  still the most principled untested lever.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `--grid --init I` fast path (data×init sweep, inner vs +holonomy M=1) |
| `scripts/dev/plot_holonomy_robustness.py`, `reports/figures/holonomy-robustness-grid.png` | Δ-distribution figure |

No new ops, no CORE.YAML, no new deps. fmt + clippy clean.

## Provenance

- Mac (Apple Silicon) + `~/hakiko_ai_ws/03_implementation/nagare_data/signed/` (SNAP Alpha/OTC).
- 50 cells: seeds 0–4 × init 0–4, `--max-tri 40000`, Adam lr 0.02, 250 iters, holonomy M=1.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <g.csv> --seed <s> --init <i> --max-tri 40000 --grid`.
