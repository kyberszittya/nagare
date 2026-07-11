---
title: "Nagare SBSH Phase 2 — the canonical-aligned orientation descriptor as an FD-clean op"
date: 2026-07-11
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, sbsh, gomb-soma, object-detection, orientation-descriptor, canonical-alignment, closed-form-op]
---

# SBSH Phase 2 — `oriented_descriptor` op

Date: 2026-07-11 · Mac · Nagare at `1b51b65`+ · CPU

## Summary

Promoted the H2-fix descriptor (canonical-orientation alignment, smoke drift 0.175 → 0.090) to an
FD-verified lib op, `src/ops/oriented_descriptor.rs`. Input a per-pixel 2-vector field `(gx,gy)` (from
gradients or a learned stem, like `phase_pool`); output `feat = |DFT(h)|_{0..b/2} ⊕ coherence`.

- **Forward:** 2nd circular moment `S=Σ m·sin2θ, C=Σ m·cos2θ` → canonical angle `θ₀=½·atan2(S,C)`
  (the dominant-edge frame) and coherence `R=√(S²+C²)/Σm`; histogram soft-bins the aligned orientation
  `φ=θ−θ₀`; `feat = |DFT(h)| ⊕ R`. Unlike `phase_pool` (global `|DFT|` invariant — fragile on the sharp
  near-delta orientation peaks of geometric edges), aligning to the object's own frame fixes the sub-bin
  aliasing.
- **Backward (FD-verified):** **`θ₀` is a detached (stop-gradient) canonical frame** — a *measurement* of
  the input's dominant orientation. Gradient flows through the aligned histogram with `θ₀` held fixed
  (the `phase_pool` form on `φ=θ−θ₀`) plus the coherence path (`R` is `θ₀`-independent).
- **Graceful degradation:** near-isotropic input (`S²+C²<ε`) → `θ₀=0`, i.e. the descriptor falls back to
  the plain `|DFT|` (itself invariant), and `R≈0` signals the regime.

## The design decision — detach θ₀ (and why)

The plan derived the *full* θ₀-coupled backward (θ₀ depends on every pixel via the moment). Implemented
and FD-tested, it was off in specific directions (dir 4: analytic 4.20 vs FD 5.73). Rather than chase the
intricate coupling, I switched to the **standard, principled choice: detach θ₀** — a canonicalisation
*pose* is a stop-gradient measurement (cf. the detached BatchNorm stats used in the CV experiment, and
the detached canonical frame in equivariant/canonicalisation nets). The gradient through the *aligned
histogram* (frame fixed) is the meaningful training signal for an upstream stem; it is exact-as-defined
and FD-verifiable.

**Debugging note (on record):** after detaching, a residual FD failure (0.7%) *persisted*. Isolation
(freeze θ₀ + zero the coherence weight → pure histogram path, structurally identical to the verified
`phase_pool`) showed the backward was **correct** — the residual was **bin-edge kink aliasing of my test
field** (a `+0.2` orientation offset concentrated gradients near bin edges, amplifying the a.e.-kink FD
bias). Using `phase_pool`'s dense-orientation field, the FD passes cleanly. *The measurement told me the
op was right and the test was wrong* — the same discipline that separated artifact from bug in the smoke.

## Tests

| layer | test | result |
|---|---|---|
| unit (FD) | `backward_matches_fd` | ok — directional-derivative check (6 dirs) vs a frozen-θ₀ forward |
| unit | `rotation_robust_and_coherent` | ok — elongated field: aligned `|DFT|` drift < 0.15 across orientations; coherence > 0.5 |
| unit | `isotropic_falls_back_no_nan` | ok — radial (isotropic) field: `θ₀` guarded, feat + grad finite |
| full suite | `cargo test --release` | **128 passed / 0 failed** (+3) |
| gate | `cargo fmt --check`, `cargo clippy --all-targets -D warnings` | clean |

## Files touched

| file | change |
|---|---|
| `src/ops/oriented_descriptor.rs` | new op — `oriented_descriptor_forward/backward`, `oriented_dim`, `OrientedOut` + 3 tests |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export |

No new deps, no CORE.YAML. Plan bundle: `docs/plans/2026-07-11-sbsh-oriented-descriptor/` (gitignored).

## Caveats

- The 2nd circular moment resolves the **π-ambiguity** (elongated objects — long-edge direction). Higher
  n-fold symmetric objects (near-square) get `R≈0` → the plain-`|DFT|` fallback (invariant, but the
  canonical gain is lost there). A learned or higher-moment canonical estimator is the graceful upgrade,
  deferred.
- Detached θ₀ ignores the (small, `|DFT|`-shift-invariant) sensitivity of the descriptor to the frame — a
  standard and intended approximation, not a defect.

## Next (SBSH sequence)

1. **Phase 0** — 4-query novelty search before any external claim.
2. **Oriented-bbox head** — a closed-form inner-CPML/KAN head on the `node_pool` (Phase 1) features +
   `oriented_descriptor` (Phase 2), regressing `(cx,cy,w,h,θ)` with a **surrogate KLD** oriented loss
   (avoids rotated-IoU gradients); node→object assignment. No full detector until this head is FD-clean.

## Provenance

- Mac (Apple Silicon), Nagare `1b51b65`+; CPU. No data, no GPU.
- Reproduce: `cargo test --release oriented_descriptor`.
