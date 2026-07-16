---
title: "Bias discrimination — regional 2nd-order pooling closes the gap a generic learner can't, but it is NOT holonomy-specific"
date: 2026-07-17
author: Aiko (Opus 4.8)
plan: docs/plans/2026-07-17-bias-discrimination/
status: complete
tags: [auto-holonomy, inductive-bias, second-order-pooling, gomb-soma, nagare, mixed-result, F-HOLO-8]
---

# Bias discrimination — the bias is the lever; the framework's *specific* op is not

**Created-at:** 2026-07-17 03:02 JST · **Plan:** [docs/plans/2026-07-17-bias-discrimination/](../docs/plans/2026-07-17-bias-discrimination/) · **Follows:** F-HOLO-7 (Gate 2)

## Summary

Gate 2 (F-HOLO-7) left a precise gap on the XOR-of-regional-roughness task: a generic MLP over the raw
curvature field reaches only **0.66**, while a fixed regional closed-form oracle reaches **0.94**. This
test asks whether a model with the framework's **regional 2nd-order pooling** bias closes that gap —
and whether the win is *holonomy-specific* or would *any* 2nd-order bias do it. All arms train on the
**same** data (5 seeds, held-out AUROC); only the **input representation (the bias)** differs.

**Result (5-seed median, held-out separability AUROC):**

| arm | bias | AUROC |
|---|---|:-:|
| generic-MLP (raw field) | none (must discover) | 0.661 |
| **block-Laplacian + head** | generic 2nd-order (spatial) | **0.929** |
| **block-entropy + head** | framework 2nd-order (`spectral_reg_value_grad`) | **0.858** |
| oracle `\|r_A − r_B\|` | hand-designed regional | 0.935 |

Two findings, both honest:

1. **Regional 2nd-order pooling IS the lever.** Both block-pooled arms crush the generic MLP (0.66) —
   block-Laplacian (0.929) nearly reaches the oracle (0.935). Giving the model per-region 2nd-order
   features closes almost the entire gap a generic learner could not. This validates the framework's
   **architectural principle** — regional 2nd-order pooling, exactly what `cpml_tier` (regional
   stratification) + the entropy pool (2nd-order covariance) provide. The right *bias* solves what
   generic *learning* cannot.
2. **But it is NOT holonomy-specific.** The **generic Laplacian** control (0.929) **beats** the
   framework's own entropy op (0.858). So "any 2nd-order regional bias" suffices here — a plain
   Laplacian is in fact a *cleaner* match for this roughness signal (it measures adjacent-similarity
   directly), while the covariance eigen-entropy is a noisier proxy on the small per-block sample
   (~15 interior plaquettes for a 5×5 covariance). The framework's *specific* op is not the lever; the
   *pooling structure* is.

**Consequence.** The framework's **architectural bias** (regional 2nd-order pooling) earns its keep on
a task a generic learner fails — a real, honest capability claim. But the framework's **specific ops**
(holonomy/entropy) are *not* shown to beat generic 2nd-order primitives (conv/Laplacian) here; on this
task a plain Laplacian wins. So the concentric Gömb-Soma build must be justified by a task where the
framework's **specific** structure (non-abelian holonomy, hypergraph, signed cycles) is *necessary* —
not merely "2nd-order pooling," which a conv already does. That task is not yet in hand.

## Where this sits in the arc (honest ledger)

This is the third time the framework's *specific* machinery ties or loses to a simpler baseline while
its *principles* validate: F-HOLO-2 (trivial entropy > deep holonomy net), real-spine signed-link
(spline core > full Gömb-Soma cascade), and now F-HOLO-8 (Laplacian > entropy op). The pattern is
consistent and worth stating plainly: **the framework's designed representations keep needing a task
where their specific structure is load-bearing.** The bias/principles are validated; the specific-op
superiority is not — and inventing the task where it is, is the real open problem.

## Files touched (new/append; no `CORE.YAML`)

| file | change |
|---|---|
| `src/curvature_field.rs` | +`block_entropy_features` (framework 2nd-order), +`block_laplacian_features` (generic control), +test |
| `examples/bias_discrimination.rs` | the 4-arm A/B (config-struct trainer) + JSON |
| `src/lib.rs` | +2 exports |

**Reused (§6.1):** `sample_regional_curvature`, `extract_curvature_field`, `region_roughness_diff`,
`spectral_reg_value_grad`, the private `region_laplacian`, `adam_step`, `auroc`.

## CORE.YAML items touched

**None.** No new dependency.

## Test results

`cargo test --release --lib` — **169 passed / 0 failed** (1 new: `block_entropy_higher_for_rough`).
Static: `clippy --all-targets -D warnings` **clean** (the trainer's 8 args were folded into a
`TrainCfg` struct — §6.5 #6, no allow); `fmt --check` **clean**.

## Performance

Full 4-arm run (5 seeds, incl. the strong raw MLP): **22.3 s**, RSS negligible. CPU (Apple M5 Pro).

## Experiment provenance

- **Git SHA:** `9432103` (branch `main`), dirty; added files uncommitted.
- **Env:** rustc 1.96.1; `hymeko_clifford` vendored; `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. `reports/figures/bias_discrimination.json`.

## Open issues / follow-ups

- **A task where the framework's SPECIFIC structure is necessary.** The consistent finding across
  F-HOLO-2/real-spine/F-HOLO-8 is that generic baselines match the framework's specific ops. The real
  open problem: a task whose signal is genuinely non-abelian-holonomy / signed-hypergraph structured,
  where a conv/Laplacian provably cannot substitute. Until then, the concentric build is justified only
  as *architecture* (regional 2nd-order pooling), not as *holonomy-specific* capability — a weaker claim.
- **Conv baseline (stronger control).** A trained conv over the field (vs the fixed block front-ends)
  would tighten "any spatial bias" — expected to also close the gap; deferred.
- **SLAM remains gated.**

## Graphical output (§9)

- **Numerical:** `reports/figures/bias_discrimination.json`.
- **Plotted:** `reports/figures/bias_discrimination.png` (4-arm bars).
- **Animated:** N/A.
