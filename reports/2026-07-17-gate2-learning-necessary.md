---
title: "Gate 2 — a strictly learning-necessary task is harder than it looks: XOR-of-regional-roughness is bias-discriminating, not learning-necessary"
date: 2026-07-17
author: Aiko (Opus 4.8)
plan: docs/plans/2026-07-17-gate2-learning-necessary/
status: complete
tags: [auto-holonomy, gate2, learning-necessary, inductive-bias, curvature, nagare, negative-result, F-HOLO-7]
---

# Gate 2 — the strict gate FAILS, informatively

**Created-at:** 2026-07-17 02:26 JST · **Plan:** [docs/plans/2026-07-17-gate2-learning-necessary/](../docs/plans/2026-07-17-gate2-learning-necessary/)

## Summary

Gate 2 aimed to build a task where learning is **necessary** — every fixed closed-form readout at
chance, only a learned model succeeding — so a learned-rule (or concentric-architecture) win is
task-meaningful. Built XOR-of-regional-roughness on the F-HOLO-4 lattice (class = do the two
column-halves **differ** in curvature roughness, `|r_A − r_B|`). The strict gate **FAILS**, and the
failure is the finding.

**Result (5-seed median, held-out separability AUROC):**

| arm | AUROC | role |
|---|:-:|---|
| trivial-entropy | 0.526 | fixed global → chance ✅ |
| constant-rotor | 0.528 | fixed global → chance ✅ |
| global-ChebyCR roughness | 0.512 | fixed global → chance ✅ |
| global-Laplacian roughness | 0.529 | fixed global → chance ✅ |
| linear-on-field | 0.526 | linear → chance ✅ |
| **learned-MLP** (128 hidden, 600 train, 700 ep) | **0.661** | generic learner → **cannot close the gap** |
| **oracle** `|r_A − r_B|` | **0.935** | fixed **regional** closed-form → solves it |

Two things are true at once, and both matter:

1. **The task defeats every fixed GLOBAL scalar and a generic MLP.** All global closed-form readouts —
   the exact ones that won F-HOLO-3/4 — are at chance (~0.52), because the discriminative quantity is a
   *difference* between regions, non-monotonic in any sum/mean-like global feature. A strong MLP over
   the raw extracted field reaches only **0.66**: the regional 2nd-order feature (roughness = local
   adjacent-similarity of a permuted-value field) is genuinely hard for a generic learner to discover.
2. **But the oracle is a FIXED regional closed-form** (`region_roughness_diff`, no learning) at **0.935**.
   So the task **has a closed-form solution** — it just has to be the regionally-aware one. **Therefore
   it is not strictly learning-necessary:** a hand-designed regional feature suffices.

**Conclusion:** a known-region-split XOR-of-roughness is **bias-discriminating, not learning-necessary**.
It shows that (a) the F-HOLO-3/4 *global* closed-form is insufficient here, and (b) a *generic* learner
is also insufficient — the right **inductive bias** (regional 2nd-order pooling) is what solves it. But
because a fixed regional closed-form solves it, the strict "learning is the only route" gate is not met.

## Why this is worth having (not a dead end)

- **It correctly keeps SLAM and the concentric build gated.** We do *not* yet have a task where a
  learned model beats every closed-form — so a learned-rule win cannot yet be claimed as necessary. The
  discipline holds: no scaling on an unmet gate.
- **It reframes the useful next test.** The gap that matters is now precise: **generic MLP 0.66 →
  oracle 0.94**. The oracle's feature is exactly *regional 2nd-order pooling* — which is what the
  holonomy/entropy machinery provides (`global_entropy_pool` = 2nd-order covariance eigen-entropy;
  `cpml_tier` = regional stratification; the concentric shells localize). So the well-posed capability
  test is: **does the holonomy/entropy MODEL close the MLP→oracle gap where a generic MLP cannot?** If
  it does, the framework's bias earns its keep on a task a generic learner fails — a meaningful claim,
  distinct from (and honestly weaker than) "learning is necessary."
- **The genuine open difficulty is named.** A *strictly* learning-necessary task (no closed-form
  solution, yet learnable at small scale) requires a **latent** structure the model must discover
  (e.g. a per-sample-random region partition) — which makes the oracle itself learning-dependent and
  the task much harder to learn. Constructing one without a wild-goose chase is real work, deferred.

## Integrity note

The MLP result is **not** an undertrained strawman: it was given 128 hidden units, 600 training
samples, and 700 epochs (a 6× data / 4× capacity increase over the first attempt, which scored 0.542 →
0.661 — it improves but plateaus far below the oracle). The claim "generic MLP insufficient" rests on
that fair attempt, not a weak one. All fixed-global arms are robustly at chance across 5 seeds.

## Files touched (new/append; no `CORE.YAML`)

| file | change |
|---|---|
| `src/curvature_field.rs` | refactor `realize_field` (shared); +`sample_regional_curvature`, +`region_roughness_diff`, +2 tests |
| `examples/curvature_xor_gate2.rs` | the 7-arm necessity probe + gate + JSON |
| `src/lib.rs` | +3 exports (append-only) |

**Reused (§6.1):** `grid_graph`, `chebyshev_angle_field`, `extract_curvature_field`,
`chebycr_roughness`, `laplacian_roughness`, `constant_rotor_energy`, `edge_log_field`,
`spectral_reg_value_grad`, `adam_step`, `auroc`. `realize_field` was refactored *out* of
`sample_curvature_field` — both callers share it; the existing 6 F-HOLO-4 tests still pass
(behavior-preserving refactor).

## CORE.YAML items touched

**None.** No new dependency.

## Test results

`cargo test --release --lib` — **168 passed / 0 failed** (2 new: `regional_sample_deterministic`,
`regional_xor_oracle_separates_classes`; the F-HOLO-4 tests still pass after the refactor). Static:
`clippy --all-targets -D warnings` **clean**; `fmt --check` **clean**.

## Performance

Full 7-arm run (5 seeds, incl. the strong MLP): **20.5 s**, RSS negligible. CPU (Apple M5 Pro).

## Experiment provenance

- **Git SHA:** `9432103` (branch `main`), working tree dirty; added files uncommitted.
- **Env:** rustc 1.96.1; `hymeko_clifford` vendored; `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. `reports/figures/gate2_xor.json`.

## Open issues / follow-ups

- **The well-posed next test (recommended):** the holonomy/entropy model (regional 2nd-order pooling —
  `cpml_tier` + `global_entropy_pool`) vs the generic MLP on this task. Does the framework's bias close
  the 0.66→0.94 gap? That is where the concentric Gömb-Soma's richness would first *earn* its complexity.
- **A strictly learning-necessary task (harder, deferred):** latent/per-sample-random region partition
  so no fixed closed-form works and the structure must be discovered — flagged as genuine difficulty,
  not to be rushed.
- **SLAM remains gated** on a met gate (either the bias-discrimination win above, or a strict one).

## Graphical output (§9)

- **Numerical:** `reports/figures/gate2_xor.json`.
- **Plotted:** `reports/figures/gate2_xor.png` (per-arm bars; global+MLP fail, regional oracle solves).
- **Animated:** N/A (static classification).
