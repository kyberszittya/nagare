---
title: "Nagare — holonomy on Epinions: the density prediction fails; gain tracks headroom"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, cpml, signed-link, epinions, robustness, mechanism]
---

# Holonomy channel on Epinions — density prediction fails, headroom is the real driver

Date: 2026-07-10 · Mac (author box) · Nagare at `66ef72e`+ · 25-cell (data×init) grid

## The prediction being tested

The invariant-holonomy ablation left a mechanism claim: "the holonomy needs cycle density — OTC (denser)
robust, Alpha (sparser) modest; **Slashdot/Epinions denser → predict a robust gain.**" Epinions is the
largest, densest signed graph in the set (131,828 nodes, 841,372 edges). This tests that prediction with
the strongest config (**INV M=4**) on the same 5×5 (data-seed × init-seed) grid, `--max-tri 40000`.

## Result — the prediction FAILS

| graph | inner base AUROC | holonomy median Δ | helps | robust? |
|---|---|---|---|---|
| Bitcoin-Alpha (sparse) | 0.882 | +0.0028 | 14/25 | no (IQR straddles 0) |
| **Bitcoin-OTC (sweet spot)** | 0.904 | **+0.0067** | **23/25** | **yes (IQR>0)** |
| **Epinions (densest)** | **0.933** | **+0.0007** | 14/25 | **no** (IQR [−0.0005, +0.0022]) |

Epinions — the densest graph — shows the **weakest** gain (+0.0007 median, 14/25, not robust), *smaller*
than both OTC and Alpha. So "denser → more holonomy gain" is **false**. Figure:
`reports/figures/holonomy-headroom.png`.

## The corrected mechanism — gain tracks HEADROOM, not density

The gain is not monotonic in density; it tracks **headroom (1 − base AUROC)**:

- **Epinions' inner core is already 0.933** — near-ceiling. The signed-degree features + degree tiers
  nearly saturate the task, so there is almost nothing left for the holonomy to add, however dense the
  graph. Densest ≠ most gain when the base is already high.
- **OTC (base 0.904) is the sweet spot** — enough cycle density *and* enough headroom → robust +0.0067.
- **Alpha (base 0.882) has headroom but is sparse** → the holonomy has fewer/cruder cycles to aggregate
  → modest, non-robust +0.0028.

So the holonomy channel helps most at **moderate base + adequate cycle density** (OTC), and its value
shrinks toward the ceiling (Epinions) and toward sparsity (Alpha). This is a cleaner, testable law than
"denser is better," and it was worth falsifying the simpler one.

Secondary observation: Epinions' inner core is *more stable* across inits (swing 0.0053 vs Alpha/OTC's
±0.015) — the larger graph averages out init noise — so the small Δ is measured cleanly; it is genuinely
small, not just noisy.

## Standing verdict on the holonomy arc (updated)

- Reframe correct, `rotor_holonomy` op sound (FD-verified), invariant multi-head (INV M=4) is the best
  config. The payoff is **real, robust on the moderate-headroom OTC (+0.0067, 23/25)**, modest on Alpha,
  and **negligible where the base is already near-ceiling (Epinions +0.0007)**.
- Net: a genuine but **headroom-bounded** contribution — it buys accuracy where the inner core leaves
  room, not universally. Honest and bounded; not oversold.

## Files touched

| file | change |
|---|---|
| `scripts/dev/plot_holonomy_headroom.py`, `reports/figures/holonomy-headroom.png` | base-AUROC vs gain figure (3 graphs) |

No code change (reused `--grid --holo-invariant`), no new ops, no CORE.YAML, no new deps.

## Next

- The richer **holonomy-phase → per-vertex `|DFT|` histogram** invariant is the remaining lever, but the
  headroom law caps its payoff: expect gains where the base is moderate (OTC-like), not near-ceiling.
- Slashdot (base unknown) would be the last confirmation; the headroom law predicts its gain by its base
  AUROC, not its size.

## Provenance

- Mac + `~/hakiko_ai_ws/03_implementation/nagare_data/signed/soc-sign-epinions.txt` (SNAP, 131,828 nodes /
  841,372 edges). 25 cells (seeds 0–4 × init 0–4), `--max-tri 40000`, `--holo-invariant --holo-heads 4`,
  Adam lr 0.02, 250 iters. ~28 s/cell, peak RSS 0.66 GB.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data soc-sign-epinions.txt --seed <s> --init <i> --max-tri 40000 --grid --holo-invariant --holo-heads 4`.
