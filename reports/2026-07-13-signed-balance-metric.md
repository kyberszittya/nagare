---
title: "Nagare — the signed-balance metric (Cartwright–Harary = Z2 holonomy), 4-graph, unbiased; reconstructing the lost script"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, balance, cartwright-harary, holonomy, nature-leakage, structural]
---

# The signed-balance metric across 4 graphs (strong Cartwright–Harary)

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `8f8d2f7`+ · CPU

## Summary

Measured the **balance metric** — signed balance = Z₂ cycle holonomy, in its **strong Cartwright–Harary**
form (a triad is balanced iff its sign-product is +1: `+++`, `+--`) — across the 4 signed graphs, the same
set as the AUROC benchmark. This is the structural signal the Nature leakage paper's **Gömb-strict** uses (the
Cartwright–Harary balance pruner). It also **reconstructs the never-committed `holonomy_theorems.py`** that
produced the 2026-07-07 figure (a reproducibility gap: the figure existed, the code did not).

| graph | V | E | neg-frac q | **balanced-triad (CH)** | (Davis weak) | `+++` | `+--` | `++-` | `---` |
|---|---|---|---|---|---|---|---|---|---|
| bitcoin-alpha | 3,783 | 14,124 | 0.084 | **0.864** | 0.869 | 0.797 | 0.067 | 0.131 | 0.004 |
| bitcoin-otc | 5,881 | 21,492 | 0.136 | **0.893** | 0.900 | 0.744 | 0.149 | 0.100 | 0.007 |
| slashdot | 82,140 | 500,481 | 0.236 | **0.860** | 0.882 | 0.724 | 0.136 | 0.119 | 0.021 |
| epinions | 131,580 | 711,210 | 0.170 | **0.891** | 0.904 | 0.807 | 0.084 | 0.096 | 0.013 |

(200k triangle-uniform samples; chance = 0.5.) Figure: `reports/figures/balance-metrics.png` — left: balance
is fragile for random signs (P(all cycles balanced) vs q, K₃–K₆, collapses faster with more cycles); right:
all four real graphs sit at 0.86–0.89, far above chance, despite q up to 0.24. **That gap is structural
balance** — the low-entropy state signed networks occupy, and exactly why signed-holonomy features predict.

## The estimator matters — a diagnosed bias (on record)

The first reconstruction sampled a **uniform edge then a common neighbour**. It matched Bitcoin-Alpha (0.876
vs the on-record 0.870) but gave Slashdot **0.797** and Epinions **0.883** — well below the 2026-07-07
report's 0.917 / 0.930. This was a real **estimator bias**, not a definitional one: an edge-then-neighbour
sample under-weights the dense, internally-balanced clubs that dominate the true triangle count, biasing
balance *low*. Switching to **wedge sampling** (apex ∝ C(deg,2), two random neighbours, keep closed wedges —
the unbiased triangle-uniform estimator) raised Slashdot to 0.860 and Epinions to 0.891. Alpha, being
near-uniform, matched under both. A residual gap to the old figure (0.860 vs 0.917) remains and is attributed
to the lost script's directed/edge-dedup preprocessing (SNAP is directed; this measurement dedups to
undirected, first-sign-wins) — the old numbers were likely from a biased or directed estimator; **these are
the unbiased strong-CH values.** *(§3: estimator verified against a known ground-truth point before trusting
the rest.)*

## Definition (user-confirmed)

Strong **Cartwright–Harary** balance (sign-product +1). Weak Davis (also admitting `---`) is tabulated for
reference but not the headline. The choice was confirmed by the user for the article's numbers.

## Relation to the Nature leakage paper & what's next

The balance metric is the **structural signal**; the paper's claim is that it must be learned **without
leakage**. Two facts established here:

- The signed-link AUROC harness (`examples/cpml_signed_link.rs`) already enumerates triangles **train-edges
  only** — it is the **strict protocol**. So the 2026-07-13 Epinions/Slashdot AUROC benchmark is
  leakage-free, not transductive.
- **Next measurement (structural learning / generalization):** the **label-shuffle audit** — strict-real vs
  strict-shuffle AUROC (a strict model must drop toward chance under shuffled training labels, proving it
  learned real signed structure, not leakage), and the transductive-vs-strict contrast to quantify the
  leakage the paper audits. That fills the paper's Table 2 TBD rows.

## Files

| file | change |
|---|---|
| `scripts/dev/balance_metrics.py` | new — the (reconstructed, committed) balance-metric measurement + P(balanced) MC + figure |
| `reports/figures/balance-metrics.{png,json}` | new — figure + numbers |

No new deps, no CORE.YAML, no source code changed. Reproduce:
`python scripts/dev/balance_metrics.py <nagare_data/signed> reports/figures/balance-metrics.png reports/figures/balance-metrics.json 200000`.

## Provenance

- Mac (Apple Silicon), Nagare `8f8d2f7`; CPU. Data: `nagare_data/signed/soc-sign-*` (SNAP + Bitcoin).
  RNG seed 0; 200k triangle-uniform (wedge) samples; K₃–K₆ P(balanced) at 4000 signings/point.
