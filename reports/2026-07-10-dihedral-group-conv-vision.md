# Nagare CV — dihedral group-convolution vs single-θ canonicalization (measured)

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Wired the dihedral group-convolution (proven correct in `dihedral_hypergraph.rs`) into the vision
model and **measured** it against the single-θ canonicalization (the quat-conv winner) on rotated
shapes. **The C_8 group-conv beats the single-frame canonicalization — 0.650 vs 0.600 median (+0.05,
4 seeds), both crushing the rotation-blind raw floor (+0.31 / +0.36).** So the dihedral
generalization is not just equivariant-by-proof; its learned steerable capacity is empirically the
stronger of the two rotation-invariance strategies.

## Design — one group-max conv backbone, three arms

`|G|` steered frames → shared filter + tanh (1×1 patch conv) → **group-max over `|G|`** → mean-pool
over patches → readout. The arms differ only in how the geometric field is presented:

| arm | frames | how |
|---|---|---|
| **raw** | `|G|=1` | gradient field untouched (rotation-blind floor) |
| **single-θ canonical** | `|G|=1` | each patch's cells rotated by −θ_p (continuous, exact, one *data-dependent* frame — the quat-conv strategy) |
| **C_8 group-conv** | `|G|=8` | field steered to all 8 dihedral frames (`dihedral_steer`), group-max (discrete, *data-independent*, learned combination) |

The group-max routes the gradient to the argmax frame per (patch, filter) — the closed-form
backward through the group. `common::vision` now holds the shared rotated-shape rendering +
gradient-field extraction (§6.1) used by both vision tests.

## Result (test accuracy, 4 seeds, randomly-rotated shapes; C_8 = 8 frames)

| | seed 0 | seed 1 | seed 2 | seed 3 | median |
|---|---|---|---|---|---|
| raw | 0.294 | 0.287 | 0.275 | 0.312 | 0.294 |
| single-θ canonical | 0.663 | 0.556 | 0.600 | 0.594 | 0.600 |
| **C_8 group-conv** | 0.650 | 0.619 | 0.756 | 0.581 | **0.650** |

**Verdict: C_8 group-conv beats single-θ canonicalization by +0.05 median** (wins clearly on seeds
1–2, within noise on 0/3); both beat raw by ~+0.31/+0.36. Raw sits at chance — rotation-blind.

## Reading (measured / inferred)

- **Measured:** the learned discrete group-conv (8 frames + group-max) edges out the hand-designed
  continuous single-frame canonicalization, at ~8× the compute.
- **Inferred:** the group-conv's advantage is **learned steerable capacity** — it fits filters that
  respond across the group and combines them, rather than committing to one canonical frame. The
  cost is `|G|×` compute; the single-θ canonicalization is the cheaper, nearly-as-good option.
- **Budget sensitivity (honest):** at a reduced budget (`C_6`, 256 samples, 170 epochs) the two
  arms *match* (0.430 vs 0.438) and absolute acc drops — the group-conv's edge needs enough training
  to materialise. The `+0.05` is the full-budget (`C_8`, 360 samples, 280 epochs) result.

## Files touched

| file | change |
|---|---|
| `tests/common/vision.rs` | **new** — shared shape rendering + `patch_gradient_field` (§6.1) |
| `tests/common/mod.rs` | `+pub mod vision` |
| `tests/vision_dihedral_conv.rs` | **new** — raw / canonical / C_8 group-conv (group-max backbone) |

The measurement test is `#[ignore]`d (~140s, `|G|=8`); the **correctness** of the dihedral
hypergraph conv is gated by the fast always-run proofs in `dihedral_hypergraph.rs`. Run the
measurement with `cargo test --test vision_dihedral_conv -- --ignored`.

## CORE / deps

**None.** Reuses `dihedral_steer` + `softmax_k`; no dependency change.

## Test results

- Full suite **104 / 0** on Mac (arm64) with the heavy measurement ignored; clippy `-D warnings`
  + fmt clean. Mac-only (kato detached).

## Open / follow-ups

- **§6.1 debt:** `vision_quat_conv.rs` still carries its own copy of the shape/gradient helpers
  (predates `common::vision`); fold it onto the shared module (needs a 51s re-verify, deferred).
- Learnable per-*element* group filter (steerable-basis conv) rather than a shared filter + max;
  reflections (full `D_n`) for chirality; real datasets (MNIST/CIFAR).
