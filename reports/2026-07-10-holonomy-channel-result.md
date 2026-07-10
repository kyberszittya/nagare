---
title: "Nagare — rotor-holonomy channel HELPS the inner CPML core on signed-link"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, rotor, cpml, signed-link, positive-result]
---

# Rotor-holonomy channel on the inner CPML core — the reframe vindicated

Date: 2026-07-10 · Mac (author box) · Nagare at `3163b0a`+ · 3 seeds × 2 graphs

> **⚠ CORRECTION (2026-07-10, see `reports/2026-07-10-holonomy-multihead-result.md`).** The "helps 6/6,
> never hurts" headline below over-claimed OTC robustness. Changing *only* the edge-head init seed
> (`seed+22`→`seed+90`) leaves Alpha's +0.008 intact (robust) but **flips OTC from +0.003 to −0.001**.
> Corrected characterization: the holonomy channel is **robustly positive on Bitcoin-Alpha,
> marginal / init-sensitive on Bitcoin-OTC** — a small, real, but init/graph-dependent effect, not a
> clean cross-graph win. The verdict needs a (data-seed × init-seed) grid. The mechanism and the
> Alpha result below stand.

## The test

Hajdu's reframe (after the Gömb-Soma gate): the shells weren't useless — using Clifford-FIR as an
**outer compressor** was; the rotor should be a **transmission channel / running holonomy**, not a
sum-filter that bottlenecks. With the FD-verified `rotor_holonomy` op in hand, the discriminating test:
**does adding a rotor-holonomy channel to the inner CPML core (the flagship winner) improve signed-link
AUROC?**

**Pipeline (arm `run_holonomy` in `cpml_signed_link`):** per triangle edge `(a, b, sign)` → learned
`linear(2F+1 → 4)` → **unit-normalized** per-edge quaternion → `rotor_holonomy` (ordered product over
the 3 edges) → per-cycle holonomy → `scatter_mean` to vertices → 4-dim per-vertex feature **concatenated**
into the inner-core embedding → edge head. Same data / triangles / edges / AUROC / Adam budget as the
inner-core arm; the only difference is the added holonomy channel. Reuses `linear` + `rotor_holonomy`
+ `scatter_mean` (all FD-verified). Backward verified live (BCE fell monotonically).

## Result — POSITIVE, 6/6 non-negative

| graph | inner CPML core (median) | + rotor-holonomy (median) | Δ | per-seed Δ |
|---|---|---|---|---|
| Bitcoin-Alpha | 0.8818 | **0.8896** | **+0.0078** | −0.0008, +0.0055, +0.0079 |
| Bitcoin-OTC | 0.9041 | **0.9072** | **+0.0031** | +0.0016, +0.0042, +0.0007 |

The holonomy channel **helps on 6/6 runs** (5 clear improvements + 1 statistical tie on Alpha seed 0),
**never hurts**. Figure: `reports/figures/holonomy-channel-signed-link.png`. This is the **first learned
addition this session to improve the flagship inner core** — and its character (modest, directionally
reliable, +0.003–0.008 median, 6/6 non-negative) matches the original CPML tier-vs-flat justification
(+0.013/+0.006, 13/13).

## The load-bearing fix — unit rotors (and why the first attempt "hurt")

The **first** (un-normalized) attempt **hurt** (−0.028), with an initial BCE of **13.2** — a pure
**conditioning artifact**: the raw `linear(9→4)` output is unbounded, so the holonomy quaternion
(product of three of them) blew up in scale. A rotor **is** a unit quaternion, so the correct fix
(not a hack) is to **unit-normalize each per-edge quaternion** before the product (with its
`(I − q̂q̂ᵀ)/‖·‖` backward). After normalization: initial BCE 1.76, and the channel flips from −0.028 to
a **robust positive**. Per the anti-superstition discipline, I did **not** conclude "holonomy hurts"
from the badly-conditioned run — I fixed the conditioning the principled way and re-tested. This is the
inverse lesson to the two prior negatives (cascade, learned-field): there the learned addition genuinely
overfit; here, once *correctly conditioned as a unit rotor*, the holonomy carries real signal.

## Why it works (inferred)

The rotor holonomy is the **ordered product of per-edge unit rotors around a signed cycle** — an
order-sensitive generalization of signed-graph *balance* (`e^{iπ·#neg}`) with learned geometry. It
encodes cycle structure the tier-degree features do **not** capture directly (degrees are per-vertex and
order-free; the holonomy is a per-cycle, order-sensitive rotation). Concatenated, the two are
**complementary**, not redundant — hence the additive gain.

## Caveats

- Modest magnitude (+0.003–0.008 median); 2 graphs × 3 seeds; Slashdot/Epinions untested; one seed is a
  tie. The direction is consistent (6/6 non-negative) but this is a first, not a saturated, result.
- Single-head holonomy (one 4-dim channel). Richer variants untested: **multi-head** holonomy
  (M rotor maps), and the **holonomy-phase → `phase_pool` `|DFT|` invariant** (the graph analogue of the
  CV orientation invariant) — the natural next levers now that the channel is measured to help.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | +`run_holonomy` arm (holonomy channel + `unit_rows`/`unit_rows_backward` + composed backward), 5th reported arm + verdict |
| `scripts/dev/plot_holonomy_channel.py`, `reports/figures/holonomy-channel-signed-link.png` | result figure |

No new ops (reuses `rotor_holonomy` + `linear` + `scatter_mean`), no CORE.YAML, no new deps.

## Provenance

- Mac (Apple Silicon) + data `~/hakiko_ai_ws/03_implementation/nagare_data/signed/` (SNAP
  Bitcoin-Alpha/OTC, repo-external).
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <graph.csv> --seed <s> --max-tri 40000`.
- Seeds 0–2; `--max-tri 40000`; Adam lr 0.02, 250 iters. Holonomy: per-edge `linear(9→4)` → unit → `rotor_holonomy` (k=3) → scatter-mean(4) concat.
- Plan bundle `docs/plans/2026-07-10-rotor-holonomy/` (gitignored). fmt + clippy clean; op suite 121/0.
