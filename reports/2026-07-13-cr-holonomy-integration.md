---
title: "Nagare — learnable Chebyshev-CR wired onto the CPML holonomy path: small but robust (paired) win"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, chebyshev-cr, holonomy, balance-coherence, hsikan]
---

# Chebyshev-CR on the CPML holonomy balance-coherence path

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `fdd9aa8`+ · CPU

## Summary

Wired the (warm-started) learnable **Chebyshev-CR** edge-weight encoder onto the CPML **holonomy** path — the
place the earlier core-read identified as where edge magnitude actually enters the full model (the inner core
ignores `tri_signs`; the holonomy uses them to build the per-edge quaternion rotors). The CR re-encodes the
holonomy sign feature each training step and trains end-to-end through the holonomy backward.

**Result (paired 5-seed A/B, `--cr-holo` vs baseline, holonomy M=4):** a **small but robust** win.

| graph | holo baseline | holo + CR | paired median Δ | seeds Δ>0 |
|---|---|---|---|---|
| bitcoin-otc | 0.9070 | 0.9077 | **+0.0010** | **5/5** |
| bitcoin-alpha | 0.8795 | 0.8785 | **+0.0015** | **4/5** |

The *unpaired* medians look tied (Alpha even inverts) because of seed-to-seed variance, but the **paired**
comparison — same seed, same data, same holonomy heads, only the CR encoder differs — is 9/10 seed-pairs
positive with a +0.001–0.0015 median. Small, consistent with the holonomy being a headroom-bounded channel
(the whole holonomy lift over the inner core is itself ~+0.001–0.007), but real and robust.

## How it wires in (gradient path already existed)

The holonomy builds per-edge quaternions from an edge-feature matrix whose last column is the edge sign
(`edge_feat[row + 2F] = tri_signs`). Reading the backward showed `linear_backward(holo_lin, edge_feat, …)`
already computes `grad_edge_feat` — it was **discarded** (`_ge`). So the integration is clean:

1. **Forward (each step):** `enc = chebyshev_cr(cr_coef, tri_signs)`; write `enc` into the sign column of
   `edge_feat`. The rotors are then built from the CR-graded weights (rotor magnitude = graded balance
   coherence rather than a hard ±1).
2. **Backward:** capture `grad_edge_feat` per head, accumulate its sign column across heads → `grad_enc` →
   `chebyshev_cr_backward` → `grad_coef` → Adam. **Warm-started** (spline frozen at identity for the first
   1/3) — the same fix that removed the standalone experiment's ~1/5-seed collapse.

Flag: `--cr-holo` (composes with `--real-weights`); default off (per §6.5 #19 — a measured, robust win but a
tiny one; leaving it opt-in until it's worth defaulting on a larger dataset sweep). All FD-verified ops; the
strict protocol is untouched (train-only features).

## The arc, closed

1. Fixed `tanh` in the inner core → tied (core ignores signs).
2. Standalone learnable CR → magnitude learnable but unstable → **warm-start** → robust win (OTC 0.9076 vs
   0.9041). *(`2026-07-13-cr-edge-encoder.md`)*
3. **This:** CR wired onto the holonomy path in the full model → small but robust paired win (9/10 seeds).

Magnitude, encoded by a learnable HSiKAN CR basis, carries a little genuine signal for signed-link — where it
enters the model (holonomy) and once optimised (warm-start). The user's "use real values, use CR" instinct is
validated, with the honest caveat that the effect is small.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `--cr-holo`: learnable Chebyshev-CR on the holonomy sign feature (fwd re-encode + bwd coef update, warm-started) |

Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` clean; full suite **145/0**. No new deps,
no CORE.YAML.

## Next

- A larger sweep (all 5 graphs, more seeds) to decide whether `--cr-holo` earns being defaulted, and whether
  `--real-weights --cr-holo` (CR over graded weights) beats `--cr-holo` alone (CR over ±1) — seed-0 hinted the
  latter is competitive.
- The same CR encoder could feed the `hg_message` (hgconv) arm, the other tri_signs consumer.

## Provenance

- Mac (Apple Silicon), Nagare `fdd9aa8`+; CPU. Data: `nagare_data/signed/soc-sign-bitcoin{otc,alpha}.csv`
  (only Bitcoin has magnitude). 5 seeds; `--grid --holo-heads 4 --max-tri 40000`; CR k=6, grid=8, warm 1/3.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <csv> --grid --holo-heads 4 --cr-holo`.
