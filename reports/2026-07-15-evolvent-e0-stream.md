---
title: "Evolvent E0 — incremental online learning (forgetting-RLS) vs backprop on a drifting stream: cold-start sample-efficiency confirmed, strong hypothesis not supported by plain RLS"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, online-learning, rls, non-stationary, non-cv, mixed-result]
---

# Evolvent E0 — the incremental-learning probe

Date: 2026-07-15 · Nagare `733f6ea`+ · CPU · NON-CV (streaming regression)

## Summary

First test of the hypothesis *"Nagare allows evolvent (incremental) online learning vs slow backprop."* On a
non-stationary regression stream (drifting teacher + an abrupt mid-stream flip), three learners predict-then-update
per sample: **A** evolvent = fixed RFF basis + `EvolventHead` (exact forgetting-RLS, one-pass, no backward sweep);
**B** same basis + online-SGD (Adam); **C** a small MLP with backprop-learned features. **A mixed result,
reported as such.**

| metric (5 seeds, median) | A evolvent | B online-SGD | C backprop-MLP |
|---|---|---|---|
| **cold-start RMSE** (window 0, sample efficiency) | **0.335** | 0.456 | 0.306 |
| **steady RMSE** (tail) | 1.60 | 0.371 | **0.255** |
| **µs / update** | 2.8 (O(d²)) | **0.4** | 0.95 |

Figure: `reports/figures/evolvent-stream.png`.

## What is measured

- **Sample efficiency confirmed vs online-SGD.** At cold-start the evolvent beats online-SGD on **all 5 seeds**
  (0.34 vs 0.46) and ties the backprop-MLP — the exact one-pass least-squares update needs fewer samples than
  iterative SGD to reach low error. This is the true "incremental learning is faster *in samples*" claim.
- **"Fast" is sample-fast, not FLOP-fast.** The RLS update is **O(d²)** (Sherman–Morrison over the d×d precision
  matrix) — ~2.8 µs vs backprop's O(d) ~0.4–0.95 µs. Per-update, the evolvent is *slower*, not faster. The
  hypothesis's "fast" must be read as data-efficiency, not compute.
- **Plain forgetting-RLS is windup-limited long-run.** With a forgetting factor λ<1 the precision matrix inflates
  in poorly-excited directions (covariance windup); low λ tracks drift but diverges (measured RMSE ~1e4 before the
  guard), while a stable config (high λ + tight covariance cap) under-tracks (steady 1.6 vs backprop 0.3).
  Backprop-Adam converges and stays lower. **The strong hypothesis — evolvent beats backprop outright — is NOT
  supported by plain RLS.**

## What is inferred / hypothesised (untested)

- The evolvent's proper regime is **rapid drift / few samples per regime**, where online-SGD never converges before
  the next change; E0's single flip over a long stream lets SGD converge, blunting the sample-efficiency edge.
- The windup instability is a **known, fixable** limitation: **directional (selective) forgetting** forgets only in
  excited directions, enabling aggressive tracking without blow-up. That is the identified next component (E1).

## Bug found and assimilated

The first run **diverged** (RMSE ~1e4): covariance windup. Fixed at the framework level — `EvolventHead` now
carries a **covariance-trace guard** (bounds `trace(P)` → bounds every eigenvalue → bounds the gain), on by
default; regression test `online::windup_guard_keeps_it_bounded` reproduces the many-feature drift and asserts the
weights stay finite and bounded (F-EVO-1).

## Tests / gates

| item | result |
|---|---|
| `online::converges_to_batch_ridge` | pass (closed-form verification vs batch ridge) |
| `online::tracks_a_drift`, `online::windup_guard_keeps_it_bounded` | pass |
| `examples/evolvent_stream` (5 seeds) | table above |
| full suite | **171 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/online.rs` | new module — `EvolventHead` (forgetting-RLS + windup guard), 3 tests |
| `src/lib.rs` | re-export `EvolventHead` |
| `examples/evolvent_stream.rs` | new — 3-arm drifting-stream benchmark |
| `scripts/dev/plot_evolvent.py`, `reports/figures/evolvent-stream.{png,json}` | figure + 5-seed results |

## Next (E1)

- **Directional-forgetting `EvolventHead`** (forget only excited directions) — the windup-free variant, so
  aggressive tracking is safe. Re-run on a **rapid-drift** stream (short regimes) where sample-efficiency should be
  decisive. The evolvent win is real only if it then **matches** backprop steady accuracy while adapting faster.
- A deeper evolvent (local closed-form updates in more than the readout) is the ambitious follow-on — but only
  after E1 establishes the readout case cleanly.

## Provenance

- Nagare `733f6ea`+; CPU. Synthetic drift stream (DX=4, M=96 RFF, N=8000, one mid-stream flip + slow random-walk
  drift). 5 seeds via `--seed=N`. λ=0.999, trace-cap 1e3 (stable config). Reproduce:
  `cargo run --release --example evolvent_stream -- --seed=N`.
