---
title: "Nagare Neocognitron N0 — the learned conv2d S-cell (FD-clean); rotation-equivariant C-cell framed"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, neocognitron, conv2d, s-cell, rotation-equivariant, quaternion-attention, dihedral-rotor, no-autograd]
---

# Neocognitron N0 — the conv2d S-cell

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `24183fe`+ · CPU

## Summary

Started the **Neocognitron** line (Fukushima's S-cell/C-cell hierarchy, made rotation-equivariant with the
crate's rotor ops) and landed its foundational primitive: a **learned 2-D convolution** (`src/ops/conv2d.rs`)
— the **S-cell** — closed-form, FD-verified, no autograd. This is the spatial-feature op the pose P1 report
identified as the missing piece (joint-discriminative features).

## The op — conv2d

Channel-first, multi-channel, zero-padded, stride-1: `x ∈ (C_i,H,W)`, `K ∈ (C_o,C_i,kh,kw)`, `b ∈ (C_o)`:

```text
y[o,h',w'] = b[o] + Σ_{i,a,c} K[o,i,a,c]·x[i, h'+a-p, w'+c-p]   (0 outside the image)
```

**Backward** (hand-derived, all three FD-verified, one bounds-checked pass): `b̄` = sum of `ȳ`; `K̄` =
`ȳ ⋆ x` (correlation); `x̄` = transposed correlation of `ȳ` with `K`. Interface mirrors `LinearLayer`
(`ConvLayer` doubles as the gradient buffer; `ConvShape { c_in,h,w,pad }`).

## The architecture — a real Neocognitron, rotation-equivariant

`docs/plans/2026-07-13-neocognitron/` frames the line: alternate **S-cells** (learned `conv2d` — feature
detectors) with **C-cells** (pooling that builds tolerance). The Nagare twist: the C-cells build **rotation**
tolerance, not just shift, using rotor ops already in the crate — all closed-form, FD-verified:

- **dihedral (dyadic) group rotor** (`dihedral`: `DihedralGroup` C_n/D_n + `dihedral_steer`) — pool over a
  discrete rotation/reflection **group orbit**.
- **quaternion attention** (`cayley_rotor` / `rotor_holonomy`) — pooling weights from a **rotor** inner
  product (continuous rotation tolerance).
- **`rotor_spike`** narrow tuning — the sharp V1-like orientation selectivity that makes the attention a
  genuine feature detector.

So the session's rotor thread (`rotor_spike`, `rotor_holonomy`, `dihedral`, `cayley_rotor`) becomes the
C-cell's rotation-equivariance engine; **`conv2d` (N0) is the missing S-cell**; N1 builds the C-cell.

## Tests

| layer | test | result |
|---|---|---|
| unit (FD) | `conv2d::grads_match_fd` | ok — grad-input, grad-kernel, grad-bias all FD-verified (passed first try) |
| unit | `identity_kernel_is_input` | ok — 1×1 kernel=1, pad 0 → output = input |
| unit | `shift_equivariance` | ok — shifting the input shifts the output |
| integration | `conv2d_learn::learns_a_known_filter` | ok — recovers a Sobel edge filter under MSE (loss < 1e-4, kernel matches) |
| full suite | `cargo test --release` | **155 passed / 0 failed** (+4) |
| gate | `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Files touched

| file | change |
|---|---|
| `src/ops/conv2d.rs` | new op — `ConvLayer`, `ConvShape`, `conv2d_forward/backward` + 3 tests |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export |
| `tests/conv2d_learn.rs` | new integration (learn a known filter) |

No new deps, no CORE.YAML. Plan bundle: `docs/plans/2026-07-13-neocognitron/` (gitignored, PDF built).

## Prior art / positioning

conv2d is universal; Neocognitron = Fukushima 1980; group-equivariant CNNs = Cohen & Welling 2016; steerable/
dihedral CNNs = Weiler et al.; quaternion nets = Parcollet et al. **No novelty claimed for conv2d.** The line's
contribution is the *composition*: a closed-form no-autograd Neocognitron whose rotation-equivariant C-cells
are built from the crate's rotor ops. Bounded search; a group-equivariant-CNN sweep precedes any external claim.

## Next

- **N1 — the rotation-equivariant C-cell**: quaternion attention (rotor keys/queries via `cayley_rotor`) +
  dihedral group-orbit pooling (`dihedral_steer`) + `rotor_spike` tuning, FD-verified, over `conv2d` feature
  maps. The rotation-tolerance mechanism.
- **N2 — the stacked S/C hierarchy** → a shift- and rotation-tolerant feature stack feeding the SBSH detector
  / pose soft-argmax; this also gives the pose net the joint-discriminative features P1 needs (unblocking the
  occlusion / multi-pose skeleton-benefit A/B).
- Performance: an im2col path for `conv2d` when the stack goes deep (N0 is direct-loop, correctness-first).

## Provenance

- Mac (Apple Silicon), Nagare `24183fe`+; CPU. No data. Seeds: conv init 5/7, target filter Sobel.
- Reproduce: `cargo test --release conv2d` and `cargo test --release --test conv2d_learn`.
