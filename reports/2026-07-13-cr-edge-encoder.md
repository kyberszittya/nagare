---
title: "Nagare — learnable Chebyshev-CR edge-weight encoder: real [-1,1] weights beat the ±1 indicator (standalone A/B)"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, chebyshev-cr, hsikan, real-weights, edge-encoder]
---

# Learnable Chebyshev-CR edge-weight encoder — does real magnitude help?

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `e252971`+ · CPU

## Summary

Tests the directive **"use [-1,1] real values, not only the ±1 indicator"** for signed-link sign prediction,
via a **learnable Chebyshev-CR** edge-weight encoder (the HSiKAN basis). Answer: **yes — magnitude is robustly
learnable, once the encoder is learnable (not fixed) and the optimisation is warm-started.**

The arc, three measured steps:

1. **Fixed real weights in the CPML core → tied.** `--real-weights` (`tanh(r/mean|r|)`) in
   `cpml_signed_link.rs` was 5-seed-tied with the binary indicator (OTC 0.9009 vs 0.9023). Reason (from
   reading the core): the inner core `run` does **not** use `tri_signs` — only `x0` signed-degree + triangle
   structure — so edge magnitude enters only via weighted degree, where it is neutral. A **fixed** squash
   extracts nothing.
2. **Standalone learnable CR → magnitude is learnable but unstable.** A minimal end-to-end predictor
   (`encode → reputation avg = net/absum → linear → BCE`, leakage-free) with a learnable Chebyshev-CR
   `enc(r)` beat binary at the median but **collapsed on ~1/5 seeds** (0.62 AUROC) — the head and the free
   spline co-diverge into a degenerate basin.
3. **Warm-start fixes it → robust win.** Freezing the spline at identity for the first third of training (so
   the head is good before the spline moves) removes the tail entirely.

## Result — 8-seed A/B (standalone predictor, test AUROC)

| graph | binary (±1) | tanh (fixed) | **Chebyshev-CR (learnable)** |
|---|---|---|---|
| bitcoin-otc | 0.9041 [.8829,.9124] | 0.9038 [.8924,.9134] | **0.9076 [.9023,.9203]** |
| bitcoin-alpha | 0.8814 [.8539,.9006] | 0.8805 [.8567,.9062] | **0.8870 [.8679,.9046]** |

The CR is highest by median on **both** graphs (+0.0035 OTC, +0.0056 Alpha over binary) and **more stable**
(its min, 0.9023 / 0.8679, is at or above the others' — no collapse). Modest but robust across 8 seeds.
Figure: `reports/figures/cr-edge-encoder-ab.png`. Only Bitcoin has magnitude; the ±1 graphs
(Slashdot/Epinions/Reddit) are binary-equivalent by construction and not shown.

## Why it works (and why the fixed versions don't)

- The reputation feature `avg[v] = net[v]/absum[v] ∈ [-1,1]` is magnitude-weighted; a **good** weight map
  (steep near 0 to keep small ratings meaningful, saturating the extremes) improves it. `r/max` (linear)
  drowns the ±1/±2 bulk → *worse*; `tanh` is a reasonable fixed guess → *tied*; the **learnable CR** finds a
  better shape than either (learned `coef` move well off identity, e.g. `[-0.34,0.58,-0.04,-0.29,0.20,0.09]`).
- Binary is the `|w|=1` special case, so the CR can only match-or-beat it given enough data — which it does,
  robustly, with warm-start.

## Method

`examples/cr_edge_encoder.rs` — a self-contained differentiable sign predictor with three swappable encoders
(binary / tanh / Chebyshev-CR). The CR path composes the crate's FD-verified ops end to end: `bce → linear →
node aggregation (net, absum) → chebyshev_cr_backward → coef`. All features are built from **training edges
only** (leakage-free, strict-protocol consistent). Warm-start = spline frozen (identity) for `iters/3`, then
trained with a conservative coef step (0.005).

## Files touched

| file | change |
|---|---|
| `examples/cr_edge_encoder.rs` | new — the standalone learnable-CR edge-encoder A/B |
| `scripts/dev/plot_cr_ab.py` | new — the A/B figure |
| `reports/figures/cr-edge-encoder-ab.png` | new |

Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` clean; full suite **145/0**. No new deps,
no CORE.YAML.

## Next

The standalone A/B validates the encoder. **Next: wire the warm-started Chebyshev-CR onto the CPML core's
balance/holonomy path** (`tri_signs` → holonomy rotor magnitude = graded balance coherence), where magnitude
actually enters the full model — the "CR target" fork we deferred until this isolated A/B confirmed the spline
trains and magnitude is learnable. Both now confirmed.

## Provenance

- Mac (Apple Silicon), Nagare `e252971`+; CPU. Data: `nagare_data/signed/soc-sign-bitcoin{otc,alpha}.csv`.
  8 seeds; 450 iters (warm-start iters/3); coef k=6, CR grid=8; leakage-free train-only features.
- Reproduce: `for s in 0..7; cargo run --release --example cr_edge_encoder -- --data <csv> --seed $s`.
