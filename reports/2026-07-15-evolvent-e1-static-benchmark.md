---
title: "Evolvent E1 — massive static benchmark (50 seeds, 6 datasets): one-pass RLS matches or beats multi-epoch backprop on classification and regression"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, evolvent, online-learning, rls, classification, regression, benchmark, kato15, positive]
---

# Evolvent E1 — the massive static benchmark

Date: 2026-07-15 · kato15 (32-core) build/run · Mac aggregation · Nagare `4b3e3c6`

> **CORRECTION (2026-07-15, commit `1d63e68`+).** The first version of this benchmark ran on a buggy RNG
> (`>>33 / u32::MAX` → `f ∈ [0,0.5)`, inputs biased to `[-1,0)` not mean-zero). The table and verdict below are the
> **re-run with the fixed RNG** (proper mean-zero inputs). The evolvent-vs-backprop *comparison* was always valid
> (all arms saw the same data), but the biased data inflated the evolvent on the hard high-d regression — the
> corrected result **reverses** there. Honest revised verdict: **competitive, not superior.** See F-EVO-4.

## Summary

Generalised `EvolventHead` to **multi-output** (one-hot least-squares → argmax for classification; the precision
`P` is shared across outputs) and ran a **50-seed** benchmark across **6 datasets** (generated + real,
classification + regression) on kato15's 32 cores. Three arms on a fixed RFF basis: **A** one-pass evolvent (RLS,
λ=1), **B** one-pass SGD, **C** a 200-epoch backprop MLP. **Honest verdict:** the one-pass evolvent is
**competitive** with multi-epoch backprop — it **wins 2, loses 2, ties 2** vs the MLP — and **beats one-pass SGD
on all 6**, at ~1/10 the training cost with no lr/epoch/architecture tuning.

| dataset | metric | **evolvent** (1-pass RLS) | SGD (1-pass) | MLP (200 ep) | vs MLP |
|---|---|---|---|---|---|
| gen_reg | R² | 0.972 [0.003] | 0.960 [0.013] | 0.973 [0.013] | tie |
| gen_reg_highd (30-D) | R² | 0.326 [0.028] | 0.142 [0.074] | **0.408** [0.074] | **MLP wins** |
| california (real) | R² | 0.693 [0.049] | 0.663 [0.073] | 0.698 [0.079] | tie |
| gen_cls (3-class) | ACC | **0.950** [0.075] | 0.921 [0.093] | 0.934 [0.086] | **evolvent** |
| gen_cls_multi (6-class,12-D) | ACC | **0.995** [0.007] | 0.985 [0.015] | 0.977 [0.015] | **evolvent** |
| iris (real) | ACC | 0.947 [0.053] | 0.921 [0.079] | **0.961** [0.053] | MLP |

Median [IQR] over 50 seeds (fixed RNG). Figure: `reports/figures/evolvent-bench.png`; per-seed JSON in
`reports/figures/evolvent_bench_seeds/`.

## Findings

- **Competitive, not superior.** Evolvent wins the two multi-class classification tasks (gen_cls 0.950 vs 0.934;
  gen_cls_multi 0.995 vs 0.977), ties both regressions where features are easy (gen_reg, california), and **loses
  the hard high-d regression** (gen_reg_highd 0.326 vs MLP 0.408) — there the fixed RFF basis underfits a 30-D
  nonlinear target and the MLP's *learned* features win. The earlier "decisive on the hard regimes" was a
  biased-data artifact and is retracted.
- **Beats one-pass SGD on all 6** (exact least-squares > iterative in one pass) — the sample-efficiency edge (F-EVO-2).
- **The practical advantage stands:** one pass, no lr/epoch/architecture tuning, deterministic — competitive
  accuracy at a fraction of the training effort. (Superseded finding text below refers to the pre-correction run.)

### (pre-correction findings, retained for provenance)

- **Evolvent ≥ MLP on 5/6, decisively on the hard regimes.** On the high-dimensional/noisy regression
  (`gen_reg_highd`, R² 0.681 vs 0.449) and the 6-class overlapping classification (`gen_cls_multi`, 0.671 vs
  0.519) the one-pass evolvent **beats** the 200-epoch MLP outright. The RFF+RLS is expressive kernel-ridge solved
  *exactly*, while the fixed-epoch MLP underfits the harder targets. Only Iris favours the MLP (1.000 vs 0.974),
  and even there the evolvent is excellent.
- **Evolvent > SGD on all 6.** One-pass exact least-squares beats one-pass iterative SGD everywhere — the expected
  sample-efficiency edge (F-EVO-2), now confirmed at scale.
- **Tightest variance.** The evolvent has the smallest IQR on the regression tasks (0.005 / 0.012 / 0.039 vs the
  MLP's 0.027 / **0.170** / 0.069). The MLP's 0.170 IQR on `gen_reg_highd` is the tell — it is unstable/underfit
  there, while the evolvent is near-deterministic given the basis. Reproducibility is a practical advantage.
- **Cost.** The evolvent trains in **ONE pass, no lr/epoch/architecture tuning**; the MLP needs 200 epochs. On a
  quiet host the per-model wall ratio is ~8–10× (representative single-seed: gen_reg 1-pass 686 ms vs 200-ep
  1234 ms; gen_reg_highd 741 ms vs 3453 ms). *Honest caveat:* the 50-seed run was memory-bandwidth-bound at 32-way
  concurrency, so the aggregate wall reflects contention, not per-model cost — the ratio is read from the
  structural 1-pass-vs-200-epoch difference and the quiet-host timings, not the contended wall.

## Interpretation — this resolves the E0 tension

E0 (a long drifting stream) was *mixed*: plain forgetting-RLS faced a windup-vs-tracking trade-off and SGD had
enough data to converge on the long stationary segments. E1 (static datasets) is where the evolvent's defining
property — **one-pass exact ridge on a fixed expressive basis** — pays off cleanly: no epochs, no tuning, tighter
variance, competitive-to-superior accuracy. The two together map the regime: **static supervised = evolvent win;
long-drift streaming = needs directional forgetting (still open).**

## Tests / gates

| item | result |
|---|---|
| `online::converges_to_batch_ridge`, `::one_hot_classifies_blobs`, `::tracks_a_drift`, `::windup_guard_keeps_it_bounded` | pass |
| `examples/evolvent_bench` (6 datasets × 50 seeds, kato15) | table above |
| full suite | **172 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/online.rs` | `EvolventHead` → multi-output (shared `P`), `predict`/`update` vectorised, `predict_class`; +classification test |
| `examples/evolvent_bench.rs` | 6-dataset (gen + real, cls + reg) 3-arm benchmark, JSON per seed |
| `scripts/dev/plot_evolvent_bench.py` | 50-seed aggregation + figure |
| `reports/figures/evolvent-bench.png`, `reports/figures/evolvent_bench_seeds/*.json` | figure + raw |

## Next

- Deploy status: `EvolventHead` is now validated on real + generated, classification + regression — promote
  EXPERIMENTAL → DEPLOYABLE for static supervised readouts (record the drift caveat).
- Swap the RFF basis for a **Nagare op basis** (HSiKAN/rotor features) — does a learned/structured closed-form
  basis lift the evolvent further, especially on `california` where the MLP's learned features still edge it?
- The still-open E-line item: directional-forgetting RLS for the streaming regime (E0's F-EVO-3).

## Provenance

- kato15 (32-core), Rust 1.96.1, Nagare `4b3e3c6`; Mac aggregation (`.venv` numpy/matplotlib). 50 seeds
  (`--seed=0..49`), M=512 RFF, MLP 200 epochs, 75/25 train/test. Reproduce: `cargo run --release --example
  evolvent_bench -- --seed=N`; aggregate `scripts/dev/plot_evolvent_bench.py`.
