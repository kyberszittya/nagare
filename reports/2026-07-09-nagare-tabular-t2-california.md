# Nagare on tabular, T2 — closed-form KAN regression on California housing

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-09-nagare-tabular/` · Follows T1

## Summary

Regression stage: a **closed-form Chebyshev-KAN regressor** on **California housing**
(the standard 8-feature tabular regression benchmark → median house value). Model:
`x → KAN(8→1) + bias → scalar → MSE` (an additive spline model), trained by hand-derived
closed-form gradients (`kan_backward` + `mse_backward`). **Median held-out R² = 0.704**
over 5 seeds (RMSE ≈ $50 k). Still the *generic-KAN* generality track (no signed structure).

## New op / infra (FD-verified)

- `src/ops/mse.rs` — mean-squared-error loss (fwd+bwd, **FD backward** matches central-diff)
  + `r2_score` (scale-free regression score).
- `src/tabular.rs` — `load_csv_regression` (features → `[-1,1]`, target z-scored, keeps
  `target_mean/std` for RMSE in original units) + a shared `shuffle_split`.

## Result (California, 20% held out, 5 seeds, 2000-row subset)

| seed | test R² | RMSE ($) |
|---|---|---|
| 0 | 0.703 | 50 341 |
| 1 | 0.724 | 51 417 |
| 2 | 0.704 | 53 008 |
| 3 | 0.681 | 57 543 |
| 4 | 0.795 | 43 139 |

**Median held-out R² = 0.704.** For reference, sklearn's linear regression scores ≈0.6 on
this task — an additive closed-form KAN at 0.70 is a solid result, entirely without autograd.

## Data

Géron mirror of California housing; dropped the categorical `ocean_proximity` column and NA
rows (empty `total_bedrooms`), target = `median_house_value` (last). Full **20 433** clean
rows in the repo-external data dir; a **2000-row subset** committed as a self-contained test
fixture (120 KB). Reproduce both (+ Iris) with `scripts/dev/fetch_tabular_datasets.sh`.

## Files touched

| file | change |
|---|---|
| `src/ops/mse.rs` | **new** (MSE fwd+bwd + R², FD-tested) |
| `src/tabular.rs` | +`load_csv_regression`, `TabularReg`, `shuffle_split` |
| `src/ops/mod.rs`, `src/lib.rs` | +mod / +re-exports |
| `tests/kan_california.rs`, `tests/fixtures/california.csv` | **new** test + subset fixture |
| `scripts/dev/fetch_tabular_datasets.sh` | **new** — reproducible Iris + California fetch |

## CORE / deps

**None.** Reuses `kan`/`chebyshev_cr`; std-only CSV; no new dependency.

## Test results (both machines)

- Full suite **75 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.
  Deterministic (seeded). (The California test runs ~6 s — 5 seeds × 600 epochs on 1600 rows.)

## Open / follow-up

1. **T3** — graph-from-tabular (kNN → signed cycles → Gömb) on Iris + California.
2. **T4** — compare KAN vs graph: does the signed-graph structure beat the plain KAN?
3. Full 20 433-row California run (example) for a headline R² beyond the subset.

## Provenance

Repo `github.com/kyberszittya/nagare`. Developed on kato15, mirrored via the Mac.
California from the Géron mirror; Rust 1.96.1; seeds fixed.
