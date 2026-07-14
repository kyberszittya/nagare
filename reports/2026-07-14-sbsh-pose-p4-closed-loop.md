---
title: "SBSH Pose P4 — a closed kinematic loop is exactly where the shared-elin hg_conv breaks: the asymmetric loop closure is inexpressible, so the skeleton hurts (measured motivation for a per-edge transform)"
date: 2026-07-14
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, sbsh, pose, closed-loop, 4-bar, skeleton-hgconv, op-limit, negative-with-mechanism, no-autograd]
---

# SBSH Pose P4 — the closed loop that breaks the current op

Date: 2026-07-14 · Mac (Apple Silicon) · Nagare at `ca916a8`+ · CPU

## Summary

P3 showed the skeleton `hg_conv` wins when the recovery is a **translation** (a single-neighbour offset the shared
`elin` expresses exactly). P4 is the closed-loop archetype — a 4-bar **parallelogram** linkage with a true 4-cycle
skeleton — and it is the case that **breaks** the current op: the loop-closure recovery `C = B + D − A` is
*asymmetric* in the two neighbours, the shared `elin` cannot express it, and the skeleton **hurts**.

| clean coupler error (5 seeds) | value |
|---|---|
| backbone only | **7.26 px** median |
| + skeleton (4-cycle) | **10.69 px** median — *worse* on 4/5 seeds |
| midpoint-oracle (shared-`elin` ceiling) | 6.50 px |

Figure: `reports/figures/pose-loop.png`. This is a **negative-with-mechanism**, and it lands exactly where a
derivation predicted — the honest, discriminating outcome.

## The mechanism — derived, then measured

For the loop edges `[B,C]` and `[C,D]` (both signs `[+1,−1]`), the residual signed `hg_conv` with a **single
shared** `elin(x)=Mx+b` treats C's two neighbours symmetrically. Solving for the M that cancels the occluded
joint's own (garbage) estimate gives `M=−I/(2·scale²)`, and the recovered point collapses to the neighbour
**midpoint** `(B+D)/2` — *independent of the true closure*. For a parallelogram the true coupler is `C = B+D−A`,
which equals the midpoint only if `A=(B+D)/2` (never, for a moving crank). The measured **midpoint-oracle = 6.50 px**
confirms the ceiling: even perfectly hitting the midpoint is 6.5 px from the true coupler. The learned skeleton
lands at 10.69 px — at or beyond that ceiling, so it **degrades** the (correct) backbone prediction rather than
helping. The op boundary is now empirical, not just argued: **the signed `hg_conv` with one shared transform
expresses neighbour-average / single-neighbour-translation, but not the asymmetric affine closure a kinematic loop
needs.**

## A methodological note (honest)

The *occluded*-coupler numbers (~1.7 px, identical with and without the skeleton) are **confounded** and were not
used for the verdict: the occlusion box is drawn to span the visible B and D, so its shape leaks C's position (C
sits at a predictable corner of the hole) — the same box-shape artifact as P2's elbow. A joint bracketed by
visible neighbours cannot be cleanly hidden by a box. The **clean** coupler comparison (no occlusion) is the
unconfounded signal, and it is decisive: the skeleton hurts.

## Where this lands the arc

P2–P4 now map the skeleton conv's operating envelope completely:

| task | recovery needed | shared-`elin` hg_conv | verdict |
|---|---|---|---|
| P2 (2-link arm) | over-constrained middle joint | — (backbone already recovers) | **neutral** |
| **P3 (coupled arms)** | single-neighbour **translation** | expressible exactly | **decisive win** (2.6 vs 6.9px) |
| **P4 (4-bar loop)** | asymmetric **affine closure** `B+D−A` | **inexpressible** → midpoint | **hurts** (10.7 vs 7.3px) |

This is the same arc principle, sharpened: an explicit structural prior helps only where (a) the base mechanism
lacks the signal **and** (b) the prior's op can *express* the constraint. P3 satisfies both; P4 fails (b). The
result is a precise, measured specification for the next op: a **per-edge transform** (each hyperedge carries its
own learned map, so the loop closure's asymmetric neighbour weighting is representable) — the fork the user
flagged, now motivated by evidence rather than anticipation.

## Tests / gates

| item | result |
|---|---|
| `examples/pose_loop` (baseline + `--hg`, 5 seeds) | table above |
| full suite | **165 / 0** (reuses FD-verified `sc_block`/`conv2d`/`soft_argmax`/`hg_message`/`linear`) |
| `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

No new library op, no new deps, no CORE.YAML.

## Files touched

| file | change |
|---|---|
| `examples/pose_loop.rs` | new — 4-bar parallelogram loop, 4-cycle skeleton, coupler-occlusion A/B + midpoint-oracle |
| `scripts/dev/plot_pose_loop.py`, `reports/figures/pose-loop.png`, `reports/figures/pl*_*.json` | figure + 5-seed results |

## Next

- **Per-edge transform op** — extend the signed `hg_conv` so each hyperedge owns its learned map (not one shared
  `elin`). This is the minimal change that makes the asymmetric loop closure expressible; re-run P4 to confirm the
  loop becomes a win (and enables reflection symmetry, the other P3 follow-up).
- Then a general (non-parallelogram) 4-bar, whose closure is *nonlinear* (circle intersection) — the per-edge
  transform is linear, so this would test whether a small nonlinearity per edge is also needed.

## Provenance

- Mac (Apple Silicon), Nagare `ca916a8`+; CPU. Analytic data (parallelogram 4-bar, G=32, crank 8, ground A(9,22)–
  D(23,22), 1-DOF θ∈[−2.4,−0.7]). 5 seeds via `--seed=N`. Train 1800 configs, 55% random coupler occlusion; eval
  80 fresh configs.
- Reproduce: `cargo run --release --example pose_loop -- [--hg] [--seed=N]`.
