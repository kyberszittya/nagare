---
title: "Nagare — holonomy richer invariant (per-vertex distribution): does not beat the mean"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, cpml, signed-link, invariant, negative-result]
---

# Richer holonomy invariant (per-vertex distribution) — no gain over the mean

Date: 2026-07-10 · Mac (author box) · Nagare at `efab394` · 25-cell (data×init) grids

## The lever

The best holonomy config so far pools the gauge-invariant magnitude `w = Re(H) = cos(θ/2)` per vertex as
its **mean** (INV M=4). The remaining lever: a **richer invariant** — the per-vertex *distribution* of
those magnitudes. (The SO(3) holonomy has no clean circular `|DFT|` phase, so the CV `phase_pool` doesn't
transfer directly; the tractable richer invariant is a per-vertex soft-**histogram** of `w`.) Added
`--holo-hist B`: soft-bin `w` into `B` bins and scatter-mean to vertices (feature `B` per head), vs the
scalar mean (`1` per head). `softbin` fwd/bwd (`∂bin/∂w = ±B/2`) threaded through `run_holonomy`.

## Result — the distribution does NOT beat the mean

| graph | INV M=4 **mean** | INV M=4 **hist=4** |
|---|---|---|
| Bitcoin-Alpha | median +0.0028, 14/25 | median **−0.0002**, 12/25 |
| Bitcoin-OTC | median +0.0067, 23/25, IQR [+0.0034,+0.0087] | median +0.0064, 23/25, IQR [+0.0029,+0.0091] |

- **OTC: a tie** (+0.0064 vs +0.0067, 23/25 both) — the distribution adds nothing over the mean.
- **Alpha: slightly worse** (−0.0002 vs +0.0028) — the 4× wider feature (B=4 bins × 4 heads) overfits on
  the sparse graph.

## Reading

The per-vertex **mean** of the holonomy magnitude already captures the signal the sign-prediction task
uses; the **distribution shape** (histogram) carries no additional discriminative information, and its
extra parameters cost on sparse graphs. So the richer invariant is an **honest negative** — the scalar
mean is sufficient. Combined with the earlier arc, this **settles the holonomy design**:

- raw covariant quaternion → init-sensitive (wash on Alpha);
- **invariant magnitude mean (INV M=4) → the best config** (robust +0.0067 on OTC, headroom-bounded);
- multi-head → only helps *because* the feature is invariant;
- invariant **distribution** (this test) → no better than the mean.

## Standing verdict (holonomy arc, final)

The reframe was correct and the `rotor_holonomy` op is sound; the deployable contribution is the
**gauge-invariant multi-head magnitude mean (INV M=4)** — a *real but modest, headroom-bounded* add to the
inner CPML core: robust on moderate-base graphs (OTC +0.0067, 23/25), negligible near the ceiling
(Epinions), non-robust on sparse graphs (Alpha). No further holonomy-feature variant tested improves it.
The inner CPML core remains the flagship; the holonomy is an optional OTC-regime booster.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `--holo-hist B` soft-histogram feature + `softbin` fwd/bwd — committed `efab394` |
| `reports/2026-07-10-holonomy-histogram.md` | this report |

No new ops, no CORE.YAML, no new deps. fmt + clippy clean.

## Provenance

- Mac + `~/hakiko_ai_ws/03_implementation/nagare_data/signed/` (SNAP Alpha/OTC). 2 grids × 25 cells
  (seeds 0–4 × init 0–4), `--max-tri 40000`, `--holo-invariant --holo-heads 4 --holo-hist 4`, Adam lr
  0.02, 250 iters. Mean baseline from the prior INV-M=4 grid (`b21ms3xi2`).
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <g.csv> --seed <s> --init <i> --max-tri 40000 --grid --holo-invariant --holo-heads 4 --holo-hist 4`.
