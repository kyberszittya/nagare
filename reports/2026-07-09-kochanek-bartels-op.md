# Nagare — Kochanek-Bartels (TCB) spline op

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Added a **Kochanek-Bartels (TCB) cubic spline** as a closed-form Nagare op — the first
step of "test HSiKAN with CR-Chebyshev *and* KB extrapolation". Port of
`hymeko_neuro/hyperedge/splines.py::_kb_eval`: a per-channel cubic Hermite spline on
`[-1,1]` whose endpoint tangents are shaped by learnable **tension / continuity / bias**
`(t,c,b) ∈ (-1,1)` (via `tanh`). At `t=c=b=0` it reduces **exactly** to Catmull-Rom; tension
is the extrapolation lever near the ±1 boundary. Parameters: control points `(C,G)` +
`tcb_raw (C,G,3)`.

## Verification (FD-green, both machines)

- `zero_tcb_equals_catmull_rom` — `t=c=b=0` output is bit-equal to `catmull_rom_forward`.
- `backward_matches_finite_difference` — the **hand-derived** backward matches central-diff
  for all three gradient buffers: control points, TCB params (through the tangent-weight
  products *and* the `tanh` map), and the input (via the segment coordinate).

## Files touched

| file | change |
|---|---|
| `src/ops/kochanek_bartels.rs` | **new** — `kb_forward`/`kb_backward` + FD/equivalence tests |
| `src/ops/mod.rs`, `src/lib.rs` | +mod / +re-export |

## CORE / deps

**None.** Standalone op; no new dependency.

## Test results

- Full suite **79 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.

## Next (the requested direction, staged)

1. **HSiKAN spline-pluggable** — thread a `spline_kind ∈ {ChebyshevCR, KochanekBartels}`
   through `hsikan_forward/backward` (mirrors the PyTorch `SignedKANLayer.spline_kind`),
   re-verifying FD + the PyTorch parity for the Chebyshev path (must stay 1.19e-7).
2. **HSiKAN-on-tabular-graph comparison** — HSiKAN as the middle shell of the Iris
   graph node classifier (2b cascade), CR-Chebyshev vs KB; report the basis effect.
3. **ViT-style patch-projection** — a separate track (needs an image data path Nagare
   lacks today); scoped when we get there.

## Provenance

Repo `github.com/kyberszittya/nagare`. Developed on kato15, mirrored via the Mac.
Rust 1.96.1; reference `_kb_eval` in `hymeko_neuro`.
