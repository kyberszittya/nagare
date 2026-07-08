# HSiKAN → Nagare, Phase 1a — closed-form op forward+backward (report)

Date: 2026-07-08 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-08-hsikan-nagare-phase1/` (on disk, not committed — public repo/IP)

## Summary

Ported the *single* `SignedKANLayer.forward` from
`hymeko_neuro/hyperedge/signedkan.py` into one closed-form Nagare op
`src/ops/hsikan.rs` — forward + hand-derived backward pair, no autograd. The two
ablation-critical pieces are preserved: the **Schmidhuber highway gate** on the
inner spline and the **sign-conditioned branches**. Both spline stages **reuse**
the Chebyshev–Catmull-Rom basis already in `ops/catmull_rom.rs`
(`chebyshev_cr_forward`/`_backward`) — no spline basis re-implemented (§6.1).

Forward: gather → inner Chebyshev spline (per sign branch) → highway gate →
sign-masked per-sign mean over the edge's vertices → diagonal outer Chebyshev
spline → sum over branches. Backward reverses this exactly, chaining the
Chebyshev-CR backward for both splines and hand-deriving the gate, mean, and
scatter gradients.

This is **Phase 1a** (op + FD test). 1b (PyTorch parity fixture), 1c (mixed-arity
toy + local-update training), 1d (perf/RSS characterisation) are the remaining
Phase-1 stages.

## Files touched

| file | change | lines |
|---|---|---|
| `src/ops/hsikan.rs` | **new** — op (config, params/edges views, forward, backward, 8 private helpers) + in-module tests | 663 (incl. ~150 test) |
| `src/ops/mod.rs` | +1 `pub mod hsikan;` | +1 |

Also on disk (uncommitted, not in git): `docs/plans/2026-07-08-hsikan-nagare-phase1/{plan.tex,plan.pdf,plan.tikz,plan.mmd}`.

## CORE.YAML items touched

**None.** All new files + one additive `mod.rs` line; `catmull_rom.rs` is called,
not modified. No dependency added (`std` + existing crate only).

## Test results (per layer)

Verified **on both machines** (Mac arm64 + kato15 x86_64), cargo 1.96.1:

| layer | test | result |
|---|---|---|
| unit (FD) | `backward_matches_finite_difference` — central-diff (ε=1e-3) vs analytic for **all 5 grad buffers** (`grad_x`, inner/outer coef, gate_w, gate_b), tol 1e-2 | ✅ pass |
| unit (shape) | `forward_shape_and_finite` — output `(T·d)`, all finite | ✅ pass |
| regression | `highway_off_ignores_gate_params` — gate params provably inert when `use_highway=false` (would have failed a leak) | ✅ pass |
| full suite | whole crate | **48 passed / 0 failed** (45 baseline + 3 new) |

Coverage: every new public and private fn is exercised — `hsikan_forward`/`hsikan_backward`
by all three tests; `gather`/`inner_forward`/`compute_gate`/`aggregate`/`accumulate_gated`/
`outer_forward` via forward; `outer_backward`/`inner_backward`/`distribute_branch`/
`gate_backward`/`scatter_grad`/`sign_value` via the FD test (highway on) and the
highway-off regression (gate-disabled branch).

Determinism: the FD test is a pure deterministic function of fixed fixture constants
(no RNG); order-independent.

## Static analysis (§6.3)

| gate | Mac | kato15 |
|---|---|---|
| `cargo clippy --all-targets -- -D warnings` | ✅ clean | ✅ clean¹ |
| `cargo fmt --check` | ✅ clean | ✅ clean |

¹ clippy/rustfmt components were **not installed** on kato15 (first CLIPPY_FAIL was a
missing-component error, *not* a lint failure — diagnosed, not assumed); added via
`rustup component add clippy rustfmt` (userspace, no sudo), then clean.

## Performance

Not formally benchmarked yet — that is **Phase 1d** (per the plan: forward B=1 +
batched latency, per-step train cost, peak RSS, ≥5 iters median/IQR). The toy FD
suite completes in <0.01 s; formal budgeted measurement is the 1d deliverable. The
plan's budgets (forward <300 µs @ T=10³, peak RSS <50 MiB) are declared there.

## §6.5 anti-patterns

None introduced. One **scoped** `#[allow(clippy::too_many_arguments)]` on
`HsikanConfig::new` (7 flat layout scalars) — the sanctioned exception (§6.5 #6:
the constructor *is* the flat layout surface). No Cartesian API surface (one op,
config-driven), no algorithm-behind-a-binding (pure algorithm crate), no
string-typed config (typed `HsikanConfig` + `bool`/`usize`), no globals.

## Design-by-contract (§8)

Preconditions documented in rustdoc (`# Preconditions`) and enforced by
`assert!`/`assert_eq!` (buffer-length + shape checks in `new` and `hsikan_forward`;
`grad_he` length in `hsikan_backward`). Trusted conditioning range documented:
`h_v, agg ∈ (-1,1)` (spline clamps to `[-1,1]`; grad zeroes outside, inherited
from `catmull_rom`).

## Open / follow-up

1. **1b — PyTorch parity fixture** (next): export a tiny `SignedKANLayer`'s weights
   + forward output to a frozen JSON on kato15 (torch env `~/envs/hymeko`); Rust
   loads and matches within ULP/relative tol. The discriminating "matches PyTorch" gate.
2. **1c** — mixed-arity toy (k∈{3,4}) + local-update loss-decreases (pattern from
   `tests/runtime_training.rs`).
3. **1d** — forward/train latency + peak-RSS characterisation vs the plan budgets.
4. **Highest-risk carryover (from plan):** the naive forward materialises `(T,k,S,d)`
   → ~327 MB at Bitcoin-Alpha scale. The op **must inherit a `chunk_t` streaming cap**
   before any 10⁵-row run — to be added with a peak-RSS test in 1c/1d, not assumed.

## Provenance

- Repo: `github.com/kyberszittya/nagare`, base `a8ca716` (working tree dirty:
  `src/ops/hsikan.rs` new, `src/ops/mod.rs` M, `docs/plans/` untracked).
- Toolchain: rustc/cargo **1.96.1** on both Mac (arm64) and kato15 (x86_64-linux);
  clippy/rustfmt 1.96-series.
- No new dependencies. No RNG seed (deterministic FD fixture).
- **Not committed** — awaiting user's go on commit + the kato15→GitHub deploy key
  for push-back. Mac authored; kato15 validated via 2-file rsync (temporary; git is
  the real sync bus once push auth is set).
