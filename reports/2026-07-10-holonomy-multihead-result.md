---
title: "Nagare — multi-head rotor-holonomy + an honesty correction on the single-head positive"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, rotor, cpml, signed-link, multi-head, robustness, correction]
---

# Multi-head rotor-holonomy — and a robustness correction

Date: 2026-07-10 20:26 JST · Mac (author box) · Nagare at `96e29c4`+ · 3 seeds × 2 graphs

## What was tested

Generalized the `run_holonomy` arm to **M rotor-holonomy heads** (M independent `linear(2F+1→4)` maps
→ M per-edge unit quaternions → M `rotor_holonomy` products → 4M-dim per-vertex feature). Question:
does multi-head add over the single-head positive from `96e29c4`? Arms: inner core / +holonomy M=1 /
+holonomy M=4, 3 seeds × Bitcoin-Alpha + OTC. (Backward verified live: M=4 BCE 2.30 → 0.127.)

## Results — median AUROC (this run; edge-head init seed +90)

| | inner | +holonomy M=1 | +holonomy M=4 |
|---|---|---|---|
| Bitcoin-Alpha | 0.8818 | **0.8898** (+0.0080) | 0.8847 (+0.0029) |
| Bitcoin-OTC | 0.9041 | 0.9029 (**−0.0012**) | 0.9070 (+0.0029) |

Per-seed AUROC:

```
Alpha  inner  0.8818 0.8899 0.8817   M=1 0.8769 0.8898 0.8946   M=4 0.8752 0.8847 0.8889
OTC    inner  0.9056 0.9041 0.8986   M=1 0.9029 0.9060 0.9023   M=4 0.9082 0.9070 0.9019
```

Figure: `reports/figures/holonomy-multihead-signed-link.png`.

## Finding 1 — multi-head trades peak for consistency (does NOT raise the ceiling)

M=4 gives a **steady +0.003 on both graphs**; M=1 gives **+0.008 on Alpha but a wash on OTC**.
Averaging over heads reduces the single-head variance but does **not** beat single-head's Alpha peak —
on Alpha, M=4 (0.8847) is *below* M=1 (0.8898). So more heads = more consistent, not better. There is
no multi-head win to bank; single-head remains the stronger arm where the effect is real (Alpha).

## Finding 2 (correction) — the single-head positive is init-sensitive; my `96e29c4` "6/6 robust" over-claimed

The **only** difference between this M=1 arm and the committed `96e29c4` arm is the edge-head
initialization seed (`seed+22` → `seed+90`). That single change:

- left **Alpha's +0.008 intact** (0.8896 → 0.8898 median) — *robust* to init;
- **flipped OTC from +0.0031 to −0.0012** — *not* robust to init.

So the honest characterization of the rotor-holonomy channel is **robustly positive on Bitcoin-Alpha,
marginal / init-sensitive on Bitcoin-OTC** — not the clean "helps 6/6, never hurts" that `96e29c4`
reported. That report's OTC row was an init-favorable draw. This is the favorable-draw trap the project
has hit before (option-msdm POSITIVE → NOT_ROBUST on multi-seed); the discipline is: **a few data seeds
at one init is not a robustness verdict — vary the initialization too.**

**Standing claim, corrected:** the holonomy channel is a *small, real, but init/graph-dependent* effect
(≈0 to +0.008 median), robust on Alpha, marginal on OTC. Not a clean cross-graph win. The `phase_pool`
`|DFT|`-invariant variant remains untested and is the more principled lever (it discards the raw
holonomy components and keeps only the rotation-invariant magnitudes — potentially less init-sensitive).

## Why (inferred)

The per-edge rotor is a learned `linear(9→4)` whose init the edge head must co-adapt to; with only a
4-dim (M=1) or 16-dim (M=4) holonomy channel bolted onto a 16-dim tier embedding, the gain is small
enough to sit inside init variance on the easier graph (OTC, higher base AUROC 0.90). On Alpha (base
0.88, more headroom) the signal is above the noise. Multi-head's variance reduction helps OTC (M=4
+0.003 there) but its added capacity slightly overfits Alpha (M=4 < M=1).

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `run_holonomy` generalized to `n_heads` (+ `--holo-heads`), M=1 and M=4 arms + verdicts |
| `scripts/dev/plot_holonomy_multihead.py`, `reports/figures/holonomy-multihead-signed-link.png` | figure |

No new ops, no CORE.YAML, no new deps. fmt + clippy clean.

## Next (to actually settle it)

- A proper **(data-seed × init-seed) robustness grid** for M=1 — the honest verdict on the holonomy
  channel needs the initialization varied, not just the data split.
- The **holonomy-phase → `phase_pool` `|DFT|` invariant** variant (the CV↔graph unification), which may
  be less init-sensitive than concatenating raw holonomy components.
- Slashdot / Epinions confirmation (deferred per Hajdu).

## Provenance

- Mac + `~/hakiko_ai_ws/03_implementation/nagare_data/signed/` (SNAP Alpha/OTC). Seeds 0–2; `--max-tri 40000`;
  Adam lr 0.02, 250 iters; `--holo-heads 4`. Reproduce:
  `cargo run --release --example cpml_signed_link -- --data <g.csv> --seed <s> --max-tri 40000 --holo-heads 4`.
