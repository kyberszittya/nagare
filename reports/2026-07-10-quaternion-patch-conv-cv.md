# Nagare CV — quaternion patch convolution (the rotor belongs in the conv, not the pool)

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

First **positive** for the CV expansion (signed-graph link prediction stays the flagship): a
**quaternion patch convolution** that canonicalises the equivariant gradient field with a
rotor, then convolves with a learned filter bank. On rotation-invariant shape ID it beats the
identical conv on raw gradients **4/4 seeds, +0.31 median** (canonical 0.600 vs raw 0.294 ≈
chance). The arc that got here — three rotor-*pool* failures — pinned exactly *why*.

## The arc (measured, honest)

| where the rotor lives | rotated quantity | vs baseline |
|---|---|---|
| pool, free-learned rotor | arbitrary token channels | **hurts** 0.52 vs 0.71 |
| pool, geometric-angle rotor | learned patch tokens | **hurts** 0.56 vs 0.71 |
| pool, on the gradient field | equivariant gradients | **ties** 0.45 vs 0.44 |
| **conv, on the gradient field** | equivariant gradients | **wins 0.60 vs 0.29 (4/4)** |

Two conditions had to both hold: (1) the rotor must act on an **equivariant** quantity — the
image gradient field co-rotates with the image, arbitrary learned channels do not (rotating them
scrambles → the two pool-fails); (2) the rotor must live in the **convolution**, which keeps the
discriminative local structure, not in an orderless pool that throws it away (the pool-on-
gradients tie → conv-on-gradients win).

## Design

```
image → gradient field (gx,gy per cell)                       [equivariant]
      → per patch: theta_p = atan2(sum dy, sum dx); rotor = z-rotation by -theta_p (cayley_rotor)
      → canonical gradient descriptor per patch (rotation-invariant)
      → learned filter bank (1x1 patch conv, M=12) + tanh -> feature map
      → mean-pool over patches -> readout -> softmax_4
```

Under a global image rotation phi, every theta_p -> theta_p+phi, so canonicalising by -theta_p
removes phi exactly (gradients co-rotate). The rotor is fixed data (no free params); `cayley_rotor`
does the work. Trained closed-form. Ablation: the SAME conv on raw (non-canonicalised) gradients.

## Result (test accuracy, 4 seeds, randomly-rotated shapes)

| | seed 0 | seed 1 | seed 2 | seed 3 | median |
|---|---|---|---|---|---|
| rotor-canonical | 0.663 | 0.556 | 0.600 | 0.594 | **0.600** |
| raw | 0.294 | 0.287 | 0.275 | 0.312 | 0.294 |

Raw sits at chance (0.25) — it cannot handle random rotation. Canonical lifts it to 0.60 on
**every** seed (Delta +0.31 median, 4/4). Plot: `reports/figures/quat-conv-cv.png`.

## Reading (measured / inferred)

- **Measured:** rotor canonicalization inside the conv, on the gradient field, robustly beats the
  raw conv under random rotation.
- **Inferred (mechanism, fully consistent across the arc):** the win is *rotation-invariance done
  right* — equivariant quantity x structure-preserving conv. The three earlier failures are the
  controls that prove each condition is necessary, not decoration.
- **Honest scope:** synthetic shapes, small grid, absolute acc ~0.60 (a 1x1 patch conv + pool is
  a weak backbone). This validates the *mechanism*, not a SOTA number. Real datasets (MNIST/CIFAR),
  a deeper conv, and generalising the rotor to hypergraph convolution + dihedral groups are next.

## Files touched

| file | change |
|---|---|
| `tests/vision_quat_conv.rs` | **new** — quaternion patch conv, rotor-canonical vs raw ablation |
| `scripts/dev/plot_quat_conv.py`, `reports/figures/quat-conv-cv.png` | **new** — arc + result plot |

## CORE / deps

**None.** Reuses `cayley_rotor` + `softmax_k`; no dependency change.

## Test results / provenance

- Full suite **97 / 0** on Mac (arm64); clippy `-D warnings` + fmt clean. (kato15 mirror detached
  per user — Mac-only from 2026-07-10.)
- Self-contained (synthetic, seeded); no external data. Repo `github.com/kyberszittya/nagare`.
