---
title: "Nagare — Gömb-Soma Step-1 gate: full cascade vs inner CPML core on signed-link"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, gomb, cpml, signed-link, cascade, negative-result, gate]
---

# Gömb-Soma Step-1 gate — does the full Gömb cascade beat the inner CPML core?

Date: 2026-07-10 · Mac (author box) + kato15 data · Nagare at `896e116`+ · 3 seeds × 2 graphs

## Why this gate

Gömb-Soma (planned Phase 3) is a **compression + entropy-routing** layer whose discriminating test is
an efficiency claim: *does it hold the signed-link AUROC at lower compute?* That is only meaningful if
the full three-shell cascade is **worth its compute in the first place**. It was untested on real
signed-link: `cpml_signed_link` ran the **inner core only**; `gomb_three_shell` ran the full cascade on
a **toy classification** task. So Step 1 is a gate, not scaffolding:

> **Does the FULL cascade (outer Clifford-FIR → HSiKAN → inner CPML tiers) beat the INNER CPML core
> alone on real signed-link AUROC?** Beats ⇒ build Step 2. Ties/loses ⇒ report negative, do not build
> the routing layer on a falsified premise (anti-superstition).

## What was built

A `run_cascade` arm added to `examples/cpml_signed_link.rs`: same data / triangles / edges / AUROC /
Adam budget (250 iters) as the inner-core arms, differing only in the node-embedding pathway —
`x0 → gomb_outer (2 Clifford-FIR banks) → scatter → hsikan → scatter → inner CPML tiers → concat → edge
head`. Transcribes the tested `gomb_three_shell` forward + composed backward at runtime size. **Zero new
ops** (reuses `gomb_outer`, `hsikan`, `scatter_mean`, `cpml_tier`, `linear`, all FD-verified).

**Backward verified live:** the smoke BCE fell 0.688 → 0.145 monotonically — the composed backward
propagates correctly (my safety net for a transcription bug).

## Results — 3 seeds × 2 heavy-tailed graphs

| graph | inner CPML core (median) | full cascade (median) | Δ | cascade wall |
|---|---|---|---|---|
| Bitcoin-Alpha | **0.8818** [0.8817, 0.8899] | 0.8488 [0.8486, 0.8546] | **−0.0330** | ~4.7 s |
| Bitcoin-OTC | **0.9041** [0.8986, 0.9056] | 0.8916 [0.8846, 0.8941] | **−0.0125** | ~6.2 s |

The full cascade **loses on 6/6 runs** — never once beats the inner core — and costs **5–6 s** vs the
inner core's near-instant. Figure: `reports/figures/gomb-soma-cascade-gate.png`.

## Verdict — NEGATIVE; do not build Gömb-Soma Step 2 on this task

The outer FIR + HSiKAN pre-shells **degrade** signed-link AUROC. Since the expensive cascade is *worse*
than the cheap inner core, there is **nothing worth compressing** — Gömb-Soma's "hold AUROC at lower
compute" premise is moot here. Per the plan's explicit branch, Step 2 (compression + entropy routing) is
**not built**.

## Mechanism (inferred) and the session-wide pattern

The inner core reads the **interpretable signed-degree features `x0`** directly; the cascade replaces
`x0` with a learned FIR+HSiKAN transform (8-dim) that **washes out the clean signal**. Train BCE fell to
0.145 while test AUROC dropped — the added shell capacity **overfits** (lower train loss, worse test).
This is the *same signature* as the CV learned-vs-fixed negative (warmstart: lower train CE, worse
test). A consistent theme this session: **on these structured tasks, well-chosen hand-designed inputs
are near-optimal, and learned pre-transforms degrade them.** The value is in the invariant / the tier
routing, not in adding upstream learned shells.

## Caveats (threats to validity)

- Matched 250-iter Adam budget for both arms; the cascade's BCE was still falling slightly at 250, so
  it is *possibly* undertrained — but it fits train *better* than it generalises, so more iterations
  would likely widen the test gap (overfitting), not close it. Not exhaustively swept.
- Two graphs (Alpha, OTC) × 3 seeds; Slashdot/Epinions not run (the two Bitcoin graphs are the cleanest
  heavy-tailed regime where the inner core was justified). The negative is large-margin and unanimous.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | +`run_cascade` (full cascade arm + composed backward) + `cascade_tier_fwd/bwd`, `rand_vec`; 4th reported arm + wall time |
| `scripts/dev/plot_gomb_soma_gate.py`, `reports/figures/gomb-soma-cascade-gate.png` | gate figure |

No new ops, no CORE.YAML (repo has none), no new deps. Plan bundle:
`docs/plans/2026-07-10-gomb-soma/` (tex/pdf/tikz/mmd, gitignored). Full suite 116/0; fmt + clippy clean.

## Where this leaves the next step

The lever is **not** more upstream shells. Candidates that remain live (from the earlier menu):
richer invariant design (bispectrum vs `|DFT|`) on the CV side; or, on the flagship, pushing the
**inner core** itself (it is the winner) — e.g., its tier-assignment or the edge head — rather than
wrapping it in shells. Signed-graph link prediction (the inner CPML core) remains the flagship, now with
one more thing it is measured to be robust against: adding the outer/middle shells does not help.

## Provenance

- Mac (Apple Silicon) + data `~/hakiko_ai_ws/03_implementation/nagare_data/signed/` (SNAP
  Bitcoin-Alpha/OTC, repo-external; also on kato15 `/tmp/hajdu/signed/`).
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <graph.csv> --seed <s> --max-tri 40000`.
- Seeds 0–2; `--max-tri 40000`; Adam lr 0.02, 250 iters. Cascade: MB=2 banks, HID=8, HSiKAN S=2/grid=6/cheb=4, 3 tiers.
