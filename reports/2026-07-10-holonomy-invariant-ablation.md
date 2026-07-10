---
title: "Nagare — invariant holonomy feature + multi-head: 3-way robustness ablation"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, cpml, signed-link, invariance, robustness, ablation]
---

# Invariant holonomy + multi-head — the winning configuration

Date: 2026-07-10 · Mac (author box) · Nagare at `9f92bc8`+ · 3 grids × 50 cells (data×init)

## The idea

The robustness grid showed the **raw covariant** holonomy feature (the 4 quaternion components) is
init-sensitive (wash on Alpha). Hypothesis (the session's central thread, from the CV arc): make the
feature **gauge-invariant** and the init-sensitivity should drop. Cleanest invariant of a unit
quaternion `H`: its scalar part `w = Re(H) = cos(θ/2)` — the **rotation magnitude**, conjugation-invariant
(1 scalar/head instead of 4 covariant components). Added `--holo-invariant`; re-ran the full
5×5 (data-seed × init-seed) grid for **RAW M=1**, **INV M=1** (isolates invariance), **INV M=4**
(invariance + heads).

## Results — median ΔAUROC over 25 (data×init) cells, helps-fraction

| | RAW M=1 (covariant) | INV M=1 (invariant) | **INV M=4** |
|---|---|---|---|
| **Bitcoin-Alpha** | −0.0004  (10/25) | +0.0026  (13/25) | **+0.0028  (14/25)** |
| **Bitcoin-OTC** | +0.0035  (19/25) | +0.0023  (15/25) | **+0.0067  (23/25, IQR [+0.0034,+0.0087])** |

Figure: `reports/figures/holonomy-invariant-ablation.png`.

## The finding — multi-head helps ONLY when the feature is invariant

The clean ablation is more interesting than "invariance is the lever":

- **On Alpha, invariance does the work.** RAW M=1 → INV M=1 moves the median from a wash (−0.0004) to
  +0.0026 (rescued); adding heads (INV M=4) adds almost nothing (+0.0028). IQR still straddles 0, so
  Alpha is *improved but not fully robust*.
- **On OTC, the win needs invariance AND multi-head together.** Invariance *alone* slightly *hurts*
  (+0.0035 → +0.0023, and less robust: 19→15/25); but INV M=4 jumps to **+0.0067, 23/25, IQR entirely
  above 0**. Multi-head on the invariant feature is what drives it.
- **The key interaction:** with the **raw** covariant feature, multi-head did **not** help (prior
  report: M=4 raw ≤ M=1 raw — extra covariant heads overfit). With the **invariant** scalar, M heads
  give M *complementary rotation-magnitudes* (different learned rotor families), a low-variance rich
  descriptor that multi-head *can* exploit. **Invariance makes multi-head useful.**

## Verdict — strongest holonomy config is INV M=4

`INV M=4` (4 invariant rotation-magnitude heads) is the best configuration found: **robustly positive on
Bitcoin-OTC (+0.0067 median, 23/25, IQR>0)** and **modestly positive on Bitcoin-Alpha (+0.0028, 14/25 —
better than raw's wash, though IQR still straddles 0)**. This is the strongest signed-link result of the
holonomy arc, and it confirms — with a clean (data×init) grid and a 3-way ablation — that **the reframe
pays off when the holonomy is used as a gauge-invariant, multi-head magnitude descriptor**, not as raw
covariant components.

## Why (inferred)

`Re(H) = cos(θ/2)` strips the arbitrary global frame (gauge) of the learned rotors, which was the main
source of init variance in the raw feature. A single invariant scalar is low-dimensional, so on the
denser OTC (more triangles) it initially loses some of the covariant feature's information (INV M=1 <
RAW M=1 there) — but M complementary invariant heads recover and exceed it, because averaging invariant
magnitudes is low-variance where averaging covariant components is not.

## Honest caveats

- Alpha's IQR still includes 0 (14/25 help) — a *modest, not bulletproof* positive there; OTC is the
  robust win. Consistent with "holonomy needs cycle density" (OTC denser).
- Effect sizes are small (~+0.003 to +0.007 median) relative to the inner core's ±0.015 init swing —
  which is exactly why the 2-D grid (not a few data seeds) was required.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `--holo-invariant` (Re(H) scalar feature) — committed `9f92bc8` |
| `scripts/dev/plot_holonomy_invariant.py`, `reports/figures/holonomy-invariant-ablation.png` | 3-way figure |

No new ops, no CORE.YAML, no new deps.

## Next

- The full **holonomy-phase → per-vertex `|DFT|` histogram** invariant (a richer invariant than the
  single `Re(H)` scalar; needs a small differentiable scatter-phase-pool op) — could lift Alpha into
  robust territory.
- **Slashdot / Epinions** (denser than Alpha → the density mechanism predicts a robust holonomy gain).

## Provenance

- Mac + `~/hakiko_ai_ws/03_implementation/nagare_data/signed/` (SNAP Alpha/OTC). 3 grids × 50 cells
  (seeds 0–4 × init 0–4), `--max-tri 40000`, Adam lr 0.02, 250 iters.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <g.csv> --seed <s> --init <i> --max-tri 40000 --grid [--holo-invariant] [--holo-heads M]`.
