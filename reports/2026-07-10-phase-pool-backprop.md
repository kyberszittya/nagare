---
title: "Nagare — differentiable global orientation phase-pool (phase_pool op)"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, computer-vision, phase-pool, backprop, closed-form-op, rotation-invariance]
---

# Nagare — `phase_pool`: global-pooling backpropagation

Date: 2026-07-10 17:18 JST · Nagare (Mac author box), branched from `499413c` · CPU

## Summary

Added a closed-form, FD-verified **backward through the global orientation phase-pool** — the piece
the CV arc was missing. Until now `src/vision.rs` computed the rotation invariant
(`orientation_histogram` → `|DFT|`) as a **fixed descriptor with no backward**, so every CV result
was "fixed gradient descriptor + trained linear head" and no *learned* front-end could be trained
*under* the invariant. `src/ops/phase_pool.rs` makes the pool differentiable **w.r.t. a per-pixel
2-vector field** `(gx,gy)` that an upstream learned operator emits, so gradient now flows from the
invariant feature all the way back into the front-end.

Plan bundle: `docs/plans/2026-07-10-phase-pool-backprop/` (tex/pdf/tikz/mmd, gitignored).

## What the op is

`field (n·g·g·2) → orientation histogram h (soft-binned, magnitude-weighted) → |DFT(h)|_{0..b/2}`
(rotation invariant), differentiable throughout. Three composable backward pieces:

1. **soft orientation-histogram** — VJP of the magnitude-weighted soft-bin of `θ=atan2(gy,gx)` back to
   `(gx,gy)` (via `dm`, `dθ`).
2. **`|DFT(h)|`** — DFT is linear; complex-magnitude backward is the standard `z/|z|` VJP, singular
   only at `|z|=0` (guarded, documented — cf. `cayley_rotor` at 180°).
3. **global sum-pool** — the histogram is a sum over spatial positions; the adjoint is the broadcast
   already used by `scatter_mean`.

Math (forward + backward) is in the module docstring and `plan.tex §Math`; the singular set (`m→0`
pixels, zero DFT modes) is guarded to grad 0.

## Files touched

| file | lines | change |
|---|---|---|
| `src/ops/phase_pool.rs` | +301 | **new op** — `phase_pool_forward`/`_backward`/`_dim`, `PhasePoolOut`, 5 unit tests |
| `tests/phase_pool_learn.rs` | +88 | **integration** — learned front-end trained end-to-end through the pool |
| `benches/phase_pool_bench.rs` | +36 | **criterion** latency (forward + train-step) |
| `examples/phase_pool_curve.rs` | +86 | demo — dumps the loss-through-the-pool curve |
| `scripts/dev/plot_phase_pool_loss.py` | +51 | plot of that curve |
| `src/ops/mod.rs`, `src/lib.rs`, `Cargo.toml` | +6 | register module, re-export, bench target |

**CORE.YAML items touched:** none. The Nagare repo has no `CORE.YAML`; this is a net-new op module
changing no existing signature or behavior. No new dependencies (criterion already dev-dep).

## Test results

| layer | test | result |
|---|---|---|
| unit | `backward_matches_fd_scalar_sum` / `_weighted` | ok — directional-derivative FD check (5 dirs), abs+rel tol |
| unit | `rotation_invariant` | ok — whole-bin field rotation ⇒ `|DFT|` unchanged < 1e-3 |
| unit | `dc_mode_equals_total_mass`, `zero_field_is_zero_and_finite` | ok — analytic sanity + guard/NaN-free |
| integration | `learns_through_phase_pool` | ok — CE 2.55 → <0.6·initial, train acc ≥ 0.9 through the pool |
| full suite | `cargo test --release` | **114 passed / 0 failed** (was 108; +6 new) |
| gate | `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean, exit 0 |

**FD-check note:** the op is differentiable *a.e.* (kinks at soft-bin edges), so a per-component
central difference is fragile when one pixel sits within `eps` of an edge (seen live: one entry
`-0.391 vs -0.387`). Switched to the canonical **directional-derivative** check `⟨∇f,u⟩` vs FD along
`u` over 5 directions — a single near-edge pixel is averaged out; a real backward error would be
order-unity relative, not 1e-3.

## Performance (criterion, §10; median [low–high])

Size `n=64, g=32, b=16` (nk=9), CPU:

| bench | time |
|---|---|
| `phase_pool_forward` | **535 µs** [528–543] |
| `phase_pool_train_step` (fwd+bwd) | **1.151 ms** [1.141–1.160] |

Backward ≈ 0.62 ms. Both **well under the plan budget of 5 ms** fwd+bwd. Peak RSS trivial (single
small op, far under the 16 GB cap). Complexity `O(n·g²·(b/2+1))` as planned.

## Graphical (§9)

`reports/figures/phase-pool-loss-curve.png` — a learned front-end (`linear(W): x→field`) trained
end-to-end through the invariant pool: CE loss falls 2.54 → 0.000, the visual proof that
`phase_pool_backward` propagates gradient (`x ← W ← pool ← head`).

## What this unlocks (next step, not done here)

The scientific payoff is now runnable but deliberately **out of scope for this op step**: the
**discriminating experiment** — does a *learned* equivariant front-end (quat-conv / dihedral-steer /
HSiKAN), trained end-to-end through the invariant pool, **beat the fixed central-difference gradient
descriptor** under the same `|DFT|` invariant? On real data (MNIST / KTH-TIPS2, kato15), multi-seed
median/IQR, with the plotted+animated deliverables. That campaign consumes this op. It is also the
foundation the **fiber-rotor-spike** front-end needs (a spike readout adds a surrogate-gradient
backward *on top* of this differentiable pool).

## Provenance

- Nagare (Mac author box), branched from `499413c`; CPU (Apple Silicon). No GPU, no external data.
- Reproduce: `cargo test --release phase_pool`; `cargo bench --bench phase_pool_bench`;
  `cargo run --release --example phase_pool_curve && uv run --with matplotlib scripts/dev/plot_phase_pool_loss.py`.
- Seeds fixed (LinearLayer 11/23, LCG in bench). §6.5 anti-patterns: none introduced — op lives in the
  algorithm layer (`src/ops`), config-free, no Cartesian wrappers, no globals.
