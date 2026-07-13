---
title: "Nagare Neocognitron N1 — the rotation-equivariant C-cell (dihedral group-orbit attention pool), FD-clean"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, neocognitron, c-cell, rotation-equivariant, dihedral, quaternion-attention, group-pool, no-autograd]
---

# Neocognitron N1 — the rotation-equivariant C-cell

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `cc59cca`+ · CPU

## Summary

Built the **rotation-equivariant C-cell** (`src/ops/group_pool.rs`), the pooling half of the Neocognitron —
where Fukushima's C-cell built *shift* tolerance, this adds *rotation* tolerance. A **dihedral group-orbit
attention pool**: steer each geometric feature 3-vector to all `|G|` frames of a dihedral (dyadic) group,
score each with a shared oriented filter, and **soft-max-pool over the orbit** (a differentiable attention;
`τ→0` recovers the exact `D_n`-invariant group-max). Closed-form, FD-verified, no autograd. It **reuses**
`dihedral_steer` (+ its backward) — the dyadic group rotor.

## The op — group_pool

```text
steered = dihedral_steer(v)                       (|G| frames of each 3-vector)
score[g,i] = ⟨filt, steered[g,i]⟩                 (shared oriented filter)
p[g,i] = softmax_g(score[g,i]/τ)                  (rotor attention over the group)
resp[i]  = Σ_g p[g,i]·score[g,i]                   (soft group-max — rotation-INVARIANT)
orient[i]= atan2(Σ_g p sin α_g, Σ_g p cos α_g)     (dominant orientation — EQUIVARIANT)
```

**Backward** (hand-derived, FD-verified for `resp`): `∂resp/∂score[k,i] = p[k,i]·(1 + (score[k,i]−resp[i])/τ)`,
then through the filter and `dihedral_steer_backward`. Returns `(grad_v, grad_filt)`.

## Why this is the quaternion-attention / dihedral-rotor C-cell

- The **softmax over the group orbit is a rotor attention**: the group elements are rotors (each a planar
  z-rotation, i.e. a unit quaternion), so the pool attends over rotor-transformed features. The discrete
  dihedral group is the exact, singularity-free rotor set (the `dihedral` module note: Cayley is singular at
  α=π, so the discrete group uses the exact planar action). The **continuous** `cayley_rotor` variant of the
  attention is the follow-on.
- The pool is genuinely **rotation-equivariant**: `resp` is invariant (verified — rotating the input by any
  group element leaves the orbit unchanged, so `resp` is identical), and `orient` is equivariant (the dominant
  orientation rotates with the input). `rotor_spike` supplies the sharp orientation tuning upstream.

## Tests

| layer | test | result |
|---|---|---|
| unit (FD) | `group_pool::backward_matches_fd` | ok — grad-v + grad-filt FD-verified |
| unit | `response_is_group_invariant` | ok — rotating v by a group element leaves `resp` identical (< 1e-4) |
| unit | `small_tau_approaches_group_max` | ok — `τ=0.02` → `resp` ≈ the exact group-max |
| full suite | `cargo test --release` | **158 passed / 0 failed** (+3) |
| gate | `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Files touched

| file | change |
|---|---|
| `src/ops/group_pool.rs` | new op — `group_pool_forward/backward`, `GroupPoolOut` + 3 tests |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export |

No new deps, no CORE.YAML. Reuses `dihedral` (dyadic group rotor).

## The Neocognitron so far

- **S-cell** `conv2d` (N0) — learned spatial feature detectors.
- **C-cell** `group_pool` (N1) — rotation-equivariant orbit pool (this).
- Together: `conv2d → interpret oriented responses as geometric vectors → group_pool` is one shift-and-rotation-
  tolerant S/C block, all FD-verified closed-form ops, no autograd.

## Next

- **N2 — the stacked S/C hierarchy**: compose `conv2d → group_pool` blocks into a deep shift-and-rotation-
  tolerant feature stack, on a rendered scene with rotated instances; an A/B (with vs without the group pool)
  measuring rotation robustness. This is also the joint-discriminative feature stack the pose net (P1) needs —
  it unblocks the occlusion / multi-pose skeleton-benefit A/B.
- The **continuous quaternion-attention** variant (`cayley_rotor` rotor keys/queries) for sub-group-angle
  rotation tolerance.
- Multi-channel `group_pool` (a filter *bank* over the orbit) and an im2col `conv2d` perf path as the stack
  deepens.

## Prior art / positioning

Group-equivariant CNNs (Cohen & Welling 2016); steerable / dihedral CNNs (Weiler et al.); quaternion networks
(Parcollet et al.). **No novelty claimed.** The op is the closed-form, FD-verified C-cell; the line's
contribution is the no-autograd Neocognitron composition. Bounded search; a group-equivariant-CNN sweep
precedes any external claim.

## Provenance

- Mac (Apple Silicon), Nagare `cc59cca`+; CPU. No data (analytic vectors). Groups tested: C_8, D_4, D_6.
- Reproduce: `cargo test --release group_pool`.
