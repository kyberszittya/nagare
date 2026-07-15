---
title: "Nagare dataset snapshot (so far) — accuracy/R², compute time, AUROC"
date: 2026-07-16
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, benchmark, dataset, auroc, snapshot]
---

# Nagare dataset snapshot — so far

Date: 2026-07-16 · Mac (arm64, CPU) · `cargo 1.96.1`

## Summary

Ran the runnable dataset benchmarks and consolidated the three metrics asked for — **compute time, accuracy,
AUROC**. Two harnesses: `evolvent_bench` (6 datasets, R²/ACC + wall-clock, 3 seeds) and `auroc_eval` (the
**entropy-pool** closed-form learner, AUROC clean + hard). Fixtures `california.csv` + `iris.csv` are present, so
all datasets ran. Figure: `reports/figures/dataset-snapshot.png`.

**Two honesty flags up front:**
- The `evolvent_bench` line is the **frozen** evolvent arc (off-mission per the closed-form/holonomy re-anchor).
  The numbers are still valid dataset measurements, but this is not the holonomy-feedback direction.
- The **AUROC** table is the **on-mission** learner — the entropy-pool (`EntropyPoolLocalLearner`), the one that
  uses **entropy as the signal, not an error metric**.

## Accuracy / R² (median of 3 seeds) — arms: evolvent 1-pass RLS · 1-pass SGD · 200-epoch MLP

| dataset | metric | **evolvent** | SGD | MLP | note |
|---|---|---|---|---|---|
| gen_reg | R² | **0.973** | 0.967 | 0.966 | tie |
| gen_reg_highd | R² | 0.307 | 0.135 | **0.422** | **MLP wins** — learned features matter (F-EVO-4) |
| california | R² | 0.664 | 0.654 | **0.676** | ~tie |
| gen_cls | ACC | 0.881 | 0.863 | **0.884** | ~tie |
| gen_cls_multi | ACC | **0.986** | 0.957 | 0.935 | evolvent wins |
| iris | ACC | **0.947** | 0.895 | 0.947 | tie w/ MLP |

Consistent with F-EVO-4: one-pass evolvent is **competitive** (wins/ties 5/6), beats SGD 6/6, but the MLP takes the
one case where features must be learned (`gen_reg_highd`). No tuning on the evolvent side.

## Compute time (indicative single-shot wall-clock, ms) — evolvent 1-pass vs MLP 200-epoch

| dataset | evolvent (1 pass) | MLP (200 ep) | speedup |
|---|---|---|---|
| gen_reg | 724 | 1327 | 1.8× |
| gen_reg_highd | 765 | 3434 | 4.5× |
| california | 180 | 415 | 2.3× |
| gen_cls | 368 | 771 | 2.1× |
| gen_cls_multi | 560 | 1740 | 3.1× |
| iris | 14 | 26 | 1.9× |

One-pass is **1.8–4.5× faster** than the 200-epoch MLP. Caveat: RLS is `O(d²)`/sample with `M=512` RFF, so it is
slower than *one-pass SGD* (sample-fast, not FLOP-fast) — the win is one-pass + no tuning + competitive accuracy,
not raw FLOPs.

## AUROC — entropy-pool learner (on-mission), clean vs hard (noisy + missing + few-shot)

| task | AUROC clean | AUROC hard |
|---|---|---|
| moons | 1.000 | 1.000 |
| spiral | 1.000 | 0.959 |
| xor | 1.000 | 1.000 |

The entropy-signal closed-form learner is **perfect on clean data** and holds **0.96–1.0 under corruption** — the
holonomy-feedback direction, measured as a quality number, not just accuracy.

## Honest scope

- **Timings are single-shot `Instant`** (diagnostic, not a `criterion` §10 benchmark) — indicative, machine-specific
  ratios; the accuracy/R²/AUROC are the trustworthy numbers.
- **Small datasets** (iris, california, synthetic; toy 2-class tasks for AUROC) and **3 seeds** — a snapshot, not a
  scaled evaluation.
- The `MeshTensor` re-anchor has **no dataset result yet** (it is a representation; the nonlinear per-edge map +
  holonomy feedback are the next layers, not wired).

## Tests / gates

Full suite **183 / 0**, fmt + clippy clean (unchanged this run — benchmarks only, no code change).

## Provenance

- Nagare `23bbb41` on Hajdus-MacBook-Pro (arm64, CPU-only). `evolvent_bench` seeds 0–2 (M=512 RFF, 200-epoch MLP);
  `auroc_eval` (make_dataset / corrupt_dataset toy tasks). Fixtures `tests/fixtures/{california,iris}.csv`.
  Reproduce: `cargo run --release --example evolvent_bench -- --seed=S` and
  `cargo run --release --example auroc_eval`.
- Data: `reports/figures/dataset_snapshot_results.json`.

## Next (per the re-anchor)

- Not more evolvent/exact-solve. The on-mission thread: nonlinear per-edge map (HSiKAN / Gomb-Soma) on the
  `MeshTensor` edge field + entropy/holonomy feedback as the learning rule — then this same three-metric snapshot on
  *that*.
