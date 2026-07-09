# Nagare — does Kochanek-Bartels beat Chebyshev-CR where its tangents *should* matter?

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

The HSiKAN-on-Iris spline A/B **tied** — but Iris is saturated, so a tie there doesn't tell
us whether KB's tangents are ever worth it. This runs the **discriminating** test directly:
fit univariate targets with sharp local structure, isolating the representational question.

**Finding (decisive):** KB is a *strict superset* of Catmull-Rom — at equal grid it fits
everything at least as well. But at a **matched parameter budget**, a *finer-grid Chebyshev*
**beats KB on every target**, and **~340× lower MSE on the steep step**. KB's tangent degrees
of freedom are a **worse use of the same parameters than grid refinement**, decisively so on
sharp targets. This mechanistically explains the Iris tie: KB offers no param-efficient edge.

## Method

Fit `y=f(x)` on `[-1,1]` (N=201) by Adam (4000 steps), median MSE over 4 init seeds. Targets:
a **smooth** sine (control), a **steep** `tanh(10(x−0.15))` step, and a **kink** `2|x−0.1|−1`
V-corner. Two lenses:

- **[1] matched grid** (both `grid=8`): Cheb-CR (8 params, fixed CR tangents) vs KB (8 control
  + 24 TCB = 32 params). Since KB reduces to Catmull-Rom at `t=c=b=0`, KB ⊇ CR.
- **[2] matched params (~32)**: a *finer* Chebyshev (`grid=cheb_k=32`, 32 params) vs the same
  KB (`grid=8`, 32 params). Same budget, spent on **more knots** (Cheb) vs **tangents** (KB).

## Results (median MSE / 4 seeds)

**[1] matched grid** — KB wins everywhere (confirms the superset, *not* a sharp-specific win):

| target | Cheb g8 (8p) | KB g8 (32p) | KB/Cheb |
|---|---|---|---|
| sine (smooth) | 1.05e-3 | 6.89e-7 | 0.00× |
| step (sharp) | 1.52e-2 | 2.26e-3 | 0.15× |
| kink (sharp) | 4.71e-4 | 3.37e-5 | 0.07× |

**[2] matched params (~32)** — finer-grid Chebyshev wins everywhere, crushingly on the step:

| target | Cheb g32 (32p) | KB g8 (32p) | KB/Cheb |
|---|---|---|---|
| sine (smooth) | 6.51e-7 | 6.89e-7 | 1.06× |
| **step (sharp)** | **6.65e-6** | **2.26e-3** | **339.97×** |
| kink (sharp) | 2.64e-5 | 3.37e-5 | 1.28× |

Plot: `reports/figures/spline-kb-vs-cheb-sharp.png` (2-panel, log-MSE).

## Reading (measured / inferred)

- **Measured:** KB ⊇ CR (matched grid, KB never worse). At matched params, finer Chebyshev is
  ≤ KB on all three targets, ~340× lower on the step.
- **Inferred (mechanism):** a width-~0.1 transition near `x=0.15` is resolved by **placing a
  knot there** (a 32-knot grid has one) — not by bending tangents on a coarse 8-knot grid,
  which cannot represent detail finer than its knot spacing (0.25) no matter the tangents. So
  the budget is better spent on spatial resolution than on tangent shape. The tangent DOF
  helps only when you're *already* grid-limited and can't add knots (the matched-grid case).
- **Consequence:** the Iris tie is not "the task was too easy to see KB's advantage" — it's
  that **KB has no advantage at a fair budget**. The pluggable KB option is a safe (never
  worse) drop-in at fixed grid, but not a param-efficient upgrade over Chebyshev-CR.

## Files touched

| file | change |
|---|---|
| `tests/spline_sharp_fit.rs` | **new** — trait-based Cheb/KB fit, matched-grid + matched-param A/B |
| `scripts/dev/plot_spline_sharp.py` | **new** — 2-panel log-MSE plot |
| `reports/figures/spline-kb-vs-cheb-sharp.png` | **new** — the figure |

## CORE / deps

**None.** Uses existing ops (`chebyshev_cr_*`, `kb_*`, `adam_step`); no new dependency.

## Test results

- Full suite **86 / 0** on Mac (arm64); clippy `-D warnings` + fmt clean. kato15 mirror pending.
- The fit test runs in ~11 s (3 targets × 2 lenses × 2 bases × 4 seeds × 4000 Adam steps).

## Open / next

- KB's pluggability is validated as *implemented + correct* but **not param-efficient** — the
  practical recommendation is Chebyshev-CR with grid tuned to the target detail. Closes the
  spline-basis question for now.
- Gömb 2c (inner CPML + full 3-shell) remains the open architecture thread.

## Provenance

Repo `github.com/kyberszittya/nagare`. Rust 1.96.1. Deterministic targets; seeds 0..3 (init).
