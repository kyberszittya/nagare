# Nagare on tabular, T1 — closed-form Chebyshev-KAN on Iris

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-09-nagare-tabular/` (staged T1–T4)

## Summary

First stage of taking Nagare onto standard tabular benchmarks: a **closed-form
Chebyshev-KAN classifier** on **Iris** (150×4, 3-class). Model:
`x → KAN(4→3) + per-class bias → softmax₃ → CE`, trained purely by hand-derived closed-form
gradients (no autograd). **Median held-out accuracy = 0.947** over 5 seeds.

**Honest framing (from the plan):** Iris has no signed-hypergraph structure, so this uses
Nagare's *generic* closed-form ops as a KAN — a legitimate **generality** result, *not* the
signed-cycle core. That is the point of T1 (baseline); T3 will construct a graph and T4
will ask whether it earns its keep.

## New ops / infra (all FD-verified)

- `src/ops/softmax_k.rs` — K-class softmax + cross-entropy (fwd+bwd), generalising the
  2-class `metrics::softmax2`. **FD backward** matches central-diff; stable (max-subtraction).
- `src/ops/kan.rs` — the KAN layer `y_j = Σ_i φ_ij(x_i)`, each `φ` a Chebyshev-CR spline
  (reuses `chebyshev_cr` — no new spline). **FD backward** matches for `grad_x` and `grad_coef`.
- `src/tabular.rs` — a std-only label-last CSV loader that **min-max standardises features
  into [-1,1]** (the spline's trusted range) + a deterministic train/test split.

## Result (Iris, 25% held out, 5 seeds)

| seed | train acc | test acc |
|---|---|---|
| 0 | 1.000 | 0.868 |
| 1 | 0.964 | 0.974 |
| 2 | 0.982 | 0.921 |
| 3 | 0.964 | 0.947 |
| 4 | 0.955 | 0.974 |

**Median held-out accuracy = 0.947** — a standard Iris result from a pure closed-form KAN.

## Files touched

| file | change |
|---|---|
| `src/ops/softmax_k.rs`, `src/ops/kan.rs` | **new** ops (+ FD tests) |
| `src/tabular.rs` | **new** loader/standardise/split (+ tests) |
| `src/ops/mod.rs`, `src/lib.rs` | +mods / +re-exports |
| `tests/kan_iris.rs`, `tests/fixtures/iris.csv` | **new** — Iris classifier test + fixture (UCI, 4 KB, committed) |

## CORE / deps

**None.** KAN reuses `chebyshev_cr`; std-only CSV parse; no new dependency. Iris fixture
committed (tiny, public, standard) so the test is self-contained.

## Test results (both machines)

- Full suite **72 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.
  Deterministic (seeded).

## Open / follow-up

1. **T2** — MSE loss (fwd+bwd) + a regression head; **California housing** (fetch, RMSE/R²).
2. **T3** — graph-from-tabular (kNN → signed cycles → Gömb) on the same datasets.
3. **T4** — compare KAN vs graph: does the signed-graph structure beat the plain KAN?

## Provenance

Repo `github.com/kyberszittya/nagare`. Developed on kato15 (Katolab online), authored +
mirrored via the Mac. Iris from UCI. Rust 1.96.1. Seeds fixed (split + init).
