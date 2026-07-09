# Gömb → Nagare, Phase 2b — two-shell cascade (outer FIR → middle HSiKAN)

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba · Follows 2a

## Summary

Compose the middle shell (the Phase-1 `hsikan` op) after the outer FIR shell, forming
the two-shell Gömb cascade — **entirely from existing, individually FD-verified ops, with
no new derivative**. The whole cascade is trained by the composed closed-form backward.

## The cascade (data flow)

```
X (V, d_feat)  --gomb_outer (M FIR banks)-->  Y (N_c, M·d_feat)         [per-cycle]
               --scatter_mean (cycle→vertex)-> X_outer (V, M·d_feat)    [per-vertex]
               --hsikan (cycles-as-hyperedges)-> H (N_c, M·d_feat)      [per-cycle]
               --linear readout--------------> logits (N_c, 2)
```

The outer FIR aggregates per-vertex features over cycles; `scatter_mean` brings them back
to vertices (so the middle can consume per-vertex embeddings); HSiKAN adds the
nonlinearity the linear FIR lacks. **Backward** chains four op-backwards in reverse:
`linear_backward → hsikan_backward → scatter_mean_backward → gomb_outer_backward`. Each is
individually FD-verified, so the composition is correct by chain rule.

## Result

Single-arity (k=3) signed cycle pool, fixed node features, **nonlinear target** (from a
fixed two-shell teacher — the HSiKAN spline nonlinearity), 500 epochs. Two-shell student
(with the middle) vs one-shell student (outer + linear readout):

| student | BCE (init → final) | acc |
|---|---|---|
| **two-shell** (outer → HSiKAN → readout) | 0.7058 → **0.0115** | 1.000 |
| one-shell (outer → linear readout) | 0.6674 → **0.3124** | — |

**The composed cascade learns end-to-end** (BCE → 0.0115), and the **middle HSiKAN helps**
(Δ 0.30): the linear one-shell plateaus at 0.31 because it cannot fit a target that is
nonlinear in the outer features, while the two-shell's HSiKAN nonlinearity fits it.

## Honest caveats (§3)

- The primary claim is the **composition works** (4-op closed-form backward, cascade
  learns). The middle-helps verdict rides on a **constructed nonlinear target** (a
  two-shell teacher) — expected that a nonlinear model beats a linear one there; it is a
  capacity demonstration, not a natural-task win. Single arity, single seed.
- **Simplifications vs the full `OuterFIRShell`:** omitted the per-bank pre-projection
  (`Linear(d_in, d_layer)` before each FIR); single arity (mixed-arity cascade needs the
  per-vertex scatter combined across arities). Both are bounded follow-ups.

## Files touched

| file | change |
|---|---|
| `tests/gomb_cascade.rs` | **new** — two-shell cascade composition + vs one-shell |

## CORE / deps

**None.** Composes `gomb_outer` (2a), `scatter_mean`, `hsikan` (Phase 1), `linear`,
`cross_entropy` — all existing. No new op, no new dependency.

## Test results (both machines)

- Full suite **64 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.
  Deterministic (seeded).

## Open / follow-up

1. **2c** — the inner CPML shell + the full three-shell cascade.
2. **Mixed-arity** cascade (per-vertex scatter combined across arities) + the per-bank
   pre-projection, to be faithful to `OuterFIRShell`.
3. **Multi-seed + natural-task** (real signed cycles) to turn the middle-helps verdict
   into a claim rather than a nonlinear-capacity demonstration.

## Provenance

Repo `github.com/kyberszittya/nagare`. Developed on kato15 (Katolab online), authored +
mirrored via the Mac; Mac + kato15 + GitHub in sync. Rust 1.96.1. Seeds fixed.
