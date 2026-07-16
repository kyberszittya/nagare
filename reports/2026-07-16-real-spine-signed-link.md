---
title: "The real spine on real signed-link data — Gömb-Soma on bitcoin-alpha: the task is structural, but the full cascade underperforms the simple inner core"
date: 2026-07-16
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, hsikan, gomb-soma, signed-link, real-data, honest-baseline]
---

# The real spine on real signed-link data

Date: 2026-07-16 · Mac (arm64, CPU) · the actual spine (`examples/cpml_signed_link.rs`), no invention

## Summary

Stopped inventing wrappers and ran the **real** Nagare spine — the Gömb-Soma cascade (`x0 → Gömb (Clifford-FIR
banks) → HSiKAN → inner core → edge head`, Adam-trained via the shipped closed-form backwards) — on a **real**
signed graph (SNAP soc-sign-bitcoin-alpha, V=3783, 24186 signed edges). Signed-link prediction is the natural
"task where the sign structure matters" (raw covariance-entropy can't see it). 2 seeds + a sign-shuffle control:

| arm | AUROC s0 | AUROC s1 | shuffle | |
|---|---|---|---|---|
| **L=3 tiered inner (fixed)** | 0.882 | 0.890 | 0.503 | **best** |
| inner core + holonomy (M=1) | 0.877 | 0.890 | 0.504 | ties best |
| inner core + holonomy (M=4) | 0.875 | 0.885 | 0.493 | ~ties |
| L=1 flat inner (fixed) | 0.869 | 0.870 | 0.499 | baseline |
| signed hypergraph conv (learned) | 0.866 | 0.852 | 0.514 | ties baseline |
| **FULL Gömb-Soma cascade (L=3)** | 0.848 | 0.849 | **0.433** | **worst — HURTS −0.034** |

Figure: `reports/figures/real-spine-signed-link.png`.

## What is measured

- **Signed-link is a genuine structural task.** All arms reach ~0.85–0.89 AUROC, and under the **sign-shuffle
  control** (permute the training edges' signs) every arm **collapses to chance** (~0.50). So the ~0.88 is real
  learning of the sign / balance structure, not leakage — exactly the kind of task where a signed-hypergraph learner
  should have an edge, and where a raw-entropy baseline (F-HOLO-2) cannot compete.
- **But the full Gömb-Soma cascade underperforms the simple inner core** — 0.848 vs 0.882–0.890, consistently across
  both seeds (HURTS −0.034 vs the flat baseline). And under shuffle it drops **below chance** (0.433), i.e. it has
  enough capacity to fit noise structure that anti-correlates on the held-out set. The winning arm is the *simple*
  L=3 tiered inner core; holonomy-M1 ties it; the extra Gömb shell + HSiKAN cascade + holonomy banks do not earn
  their complexity here.
- **This is consistent with the project's own prior findings** ("topology INERT, the KAN spline nonlinearity is the
  only lever"; the HSiKAN-MLP hybrid audit). The fancy structural machinery repeatedly fails to beat a simpler
  spline core on these tasks.

## Honest read

Two things stand, both on the real spine and real data:
1. **A real task where structure is necessary exists** (signed-link, shuffle-verified) — this answers F-HOLO-2's
   open question at the *task* level: the sign structure is genuinely learnable and worth ~0.88.
2. **The deep cascade is not the winning way to exploit it** — the simple tiered inner core wins; the Gömb-Soma
   cascade is the worst arm. So even on the natural task, "more machinery" is not the answer here.

This is the honest baseline **before** wiring the global-instantaneous entropy feedback. It tempers expectations: the
cascade the feedback would attach to is already the underperforming arm, and a global-broadcast rule typically
trails backprop — so the feedback is worth testing as a *mechanism*, but is unlikely to make the cascade win on this
dataset. The real lever, per the evidence, is the spline core, not depth.

## Tests / gates

Full suite unchanged (**185 / 0**); this run added no library code (drives the existing spine + example). fmt +
clippy clean.

## Provenance

- Nagare `676f3e5`+ on Hajdus-MacBook-Pro (arm64, CPU-only). Dataset: SNAP soc-sign-bitcoin-alpha (download in
  `data/signed/README.md`; gitignored). Seeds 0–1, `--shuffle` control on seed 0. Reproduce:
  `cargo run --release --example cpml_signed_link -- --data data/signed/bitcoinalpha.csv --seed S [--shuffle]`.

## Next

- **Wire the global-instantaneous entropy feedback** into the cascade (broadcast the pooled entropy signal to every
  layer at once, vs the one-path backward) — measured against this baseline, honest expectation set.
- A task/dataset where the *cascade* (not just the inner core) is necessary — otherwise the honest conclusion is
  that the spline core, not the deep structural machinery, is the lever.
