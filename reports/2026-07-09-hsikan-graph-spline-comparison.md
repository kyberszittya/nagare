# Nagare — HSiKAN on a signed graph: Chebyshev-CR vs Kochanek-Bartels

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

The "test it" half of the spline-pluggable ask: run HSiKAN on a **real signed graph** with
both spline bases and report which wins. **Verdict — they tie at the median (0.947 = 0.947);
KB does not beat Chebyshev-CR on Iris despite 3.7× the parameters.**

## Setup

A minimal transductive HSiKAN node classifier where **HSiKAN is the only nonlinearity**, so
`spline_kind` is the sole varying factor between the two arms:

```
Iris x (n,4) → leakage-free kNN(6) signed graph → signed triangles (754)
             → hsikan (triangles-as-hyperedges, d=4, S=2, grid=5, cheb_k=4, highway on)
             → scatter_mean (cycle→sample) → linear(4→3) → softmax₃
```

Transductive (train on train labels via masked CE, eval held-out), 5 seeds, same splits and
same graph for both arms. Only `HsikanConfig::with_spline_kind(..)` differs.

## Result (median held-out accuracy, 5 seeds)

| basis | median acc | params | per-seed (0..4) |
|---|---|---|---|
| Chebyshev-CR | **0.947** | 96 | 0.868, 0.974, 0.868, 0.947, 0.974 |
| Kochanek-Bartels | **0.947** | 352 | 0.868, 0.974, 0.895, 0.947, 0.947 |

**Verdict: tie (Δ 0.000).** Seed-level, KB edges ahead on 2/5 (seeds 1→already tied, 2), ties
2/5, trails on 1/5 (seed 4) — a mild, non-decisive lean with no median gain. Plot:
`reports/figures/hsikan-graph-spline-cheb-vs-kb.png`.

## Reading (measured / inferred / hypothesis)

- **Measured:** both bases train end-to-end through the composed closed-form backward and
  classify Iris at 0.947 median; KB carries 3.7× the params (control points + TCB tangents vs
  Chebyshev coeffs).
- **Inferred:** the extra KB capacity buys nothing at the median here because **Iris is
  saturated** — the whole tabular arc (KAN 0.947, graph-vs-KAN tie 0.947) sits at the same
  ceiling, so a denser spline basis has no headroom to exploit. This is the same honest
  negative shape as T4 (graph ties KAN): the machinery works, the task doesn't discriminate it.
- **Hypothesis (untested):** KB's tension/continuity/bias tangents would show a gap on a task
  with sharper local structure / extrapolation demand than Iris (KB's design advantage is
  local shape control, which a near-linearly-separable 3-class problem doesn't stress).

## Files touched

| file | change |
|---|---|
| `src/ops/hsikan.rs` | `+ HsikanConfig::param_len` (public buffer-sizing helper) |
| `tests/hsikan_graph_spline.rs` | **new** — HSiKAN-on-Iris-graph Cheb-vs-KB A/B (5 seeds) |
| `scripts/dev/plot_hsikan_spline.py` | **new** — grouped-bar + median plot |
| `reports/figures/hsikan-graph-spline-cheb-vs-kb.png` | **new** — the figure |

## CORE / deps

**None.** `param_len` is an additive public method on `HsikanConfig`; no dependency change.

## Test results

- Full suite **85 / 0** on Mac (arm64); clippy `-D warnings` + fmt clean. kato15 mirror pending.
- The comparison test runs in ~23 s (754 triangles × 2 bases × 5 seeds × 300 epochs).

## Open / next

- KB's hypothesised advantage on sharp-local-structure tasks is untested — a synthetic task
  with local kinks / an extrapolation split would discriminate the bases where Iris can't.
- Gömb 2c (inner CPML + full 3-shell) still open.

## Provenance

Repo `github.com/kyberszittya/nagare`. Rust 1.96.1. Iris fixture `tests/fixtures/iris.csv`
(min-max to [-1,1]); leakage-free graph = feature-correlation edge signs, no labels used in
graph construction.
