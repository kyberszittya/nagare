# NAGARE — reconcile the two trees onto the GitHub repo (full superset)

Created-at: 2026-07-06 17:56 JST
Plan: `docs/plans/2026-07-06-nagare-repo-reconcile/plan.{tex,pdf,tikz,mmd}` (gitignored)
Repos: `nagare_github` = `github.com/kyberszittya/nagare` (now authoritative) ← `hymeko_nagare` (framework crate, now frozen). **All changes local; commit/push held for the user.**

## Summary

Per the user's directive (2026-07-06: "fix the divergence here first; the final
should be the GitHub repo; in the future Nagare should be detached from here"),
converged the two Nagare trees onto `nagare_github` as the authoritative,
standalone-ready copy, and froze the framework crate with a pointer. Scope:
**Full superset** (the user's chosen option).

The GitHub repo was an earlier extraction (flat `src/`, older fixed-array
projection, subset of ops); the framework crate `hymeko_nagare` was ahead on the
FD-tested `project_alpha_mix` kernel, `ProjectionBasis`, the frozen seed-53
fixture, and the full op suite — but *behind* on the experiment harness
(`run_stress_ablation` + the 2026-07-06 order-shuffle ablation live only in
GitHub). So this was a **directed merge**, not a copy: GitHub's harness +
framework's tested kernel.

## What the divergence actually was (discovery)

- **Already synced** (July-5 WIP): runtime + `ops/{adam,linear,loss,scatter}`.
  `clifford_fir`/`runtime`/`optimizer` differed only by edition-2024 import
  ordering + one cosmetic unicode `×`→`x` — **left GitHub's** (cleaner).
- **metrics** differed by naming/organization (`cross_entropy`/`CeOut` vs
  `cross_entropy_eval`/`CrossEntropyEval`), not capability — **left GitHub's**.
- **Real gaps ported framework→GitHub:** 6 ops
  (`project_alpha_mix, catmull_rom, cayley_rotor, fsr_mixer,
  fused_entropy_update, signed_scatter`), the `ProjectionBasis` projection
  (replacing the fixed-array `project_onto_holonomy_subspace`), the frozen
  seed-53 fixture, and `gather_batch`.
- **Dependency direction:** `hymeko_nagare` is a leaf crate (only a workspace
  member; no dependents) → safe to freeze. `nagare_github`'s only framework
  ties are two path deps (`hymeko_clifford`, `hymeko_graph`) — the *future*
  detachment (vendoring), not this phase.

## Behaviour-preservation (the key check)

The projection rewire (fixed-array → `project_alpha_mix` kernel) is **exactly
behaviour-preserving**: re-running the order-shuffle ablation at seed 53 on the
reconciled repo reproduces the 2026-07-06 baseline **bit-identically** in every
spiral row, including the projection column:

| regime | projection loss (baseline) | projection loss (reconciled) |
|---|---:|---:|
| clean | 0.001586 | 0.001586 |
| few-shot noisy+missing | 0.250603 | 0.250603 |
| shuffled | 0.017641 | 0.017641 |
| shuffled + few-shot | 0.316421 | 0.316421 |

Same math (orthonormal basis makes the `/‖u‖²` a no-op), so identical results.
Per the **baselines-are-cached-facts** directive, this one-seed identity check is
the verification — **no benchmark grid re-run**. The frozen fixture test passing
independently confirms the generators are byte-identical across the two trees.

## Files touched

**`nagare_github` (local; +6 ops = 1936 LOC copied, +317 test LOC):**
- New ops (verbatim from framework, register in `ops/mod.rs`):
  `src/ops/{project_alpha_mix,catmull_rom,cayley_rotor,fsr_mixer,fused_entropy_update,signed_scatter}.rs`.
- `src/projection.rs` (rewired, ±227): fixed-array → `ProjectionBasis` +
  `project_alpha_mix`; `default_holonomy_basis`, `fit_class_mean_basis`.
- `src/learner.rs` (+63): projection field `[[f32;28];6]` → `ProjectionBasis`;
  fit/apply call sites rewired; harness + order-shuffle **kept**.
- `src/features.rs` (+7): channel-group consts (`GEOMETRY/ROTOR/HOLONOMY_CHANNELS`).
- `src/datasets.rs` (+108): `gather_batch` + its test (order-shuffle work from
  earlier today also here).
- `src/lib.rs` (+36): new-op + `ProjectionBasis` + `gather_batch` re-exports;
  projection API renamed.
- `tests/`: `holonomy_fixture.rs` + `project_alpha_mix_fd.rs` (ported, ASCII
  docs); `projection_basis.rs` rewritten for the new API;
  `tests/fixtures/moons_spiral_xor_seed53.txt` (copied).
- **Removed:** stale root `fixtures/moons_spiral_xor_seed53.txt` (July-2
  extraction duplicate, unreferenced, superseded by `tests/fixtures/`; §6.5 #13).
- `README.md` (+42): projection-backward-now-native correction, layout update,
  and the **multi-seed caveat** on the fitted-projection result.

**Framework (local):**
- `hymeko_nagare/README.md` (new): **FROZEN** marker + pointer to the GitHub
  repo; scheduled for deletion during detachment. No code touched.

**CORE.YAML items touched: none.** `nagare_github` has no CORE.YAML; framework
`hymeko_nagare` is a leaf crate; no workspace Cargo.toml edit. No new dependency.

## Test results

- `cargo test` (nagare_github): **45 passed, 0 failed, 1 ignored** (the
  deliberate fixture writer). Breakdown: 32 lib unit (incl. the 6 ported ops'
  in-module tests), 1 holonomy_fixture (fixture matches → generators
  byte-identical across trees), 2 local_learner, 1 order_shuffle mechanism, 4
  project_alpha_mix FD, 2 projection_basis (new API), 3 runtime_training.
- Gates: `cargo fmt --check` clean; `cargo clippy --all-targets --no-deps -D
  warnings` clean (fixed one unused-import). No `unwrap`/`unsafe`/suppressions.
- Coverage: `gather_batch` + new projection API driven by new/rewritten tests;
  ported ops carry their own FD + in-module tests; behaviour change (projection
  path) regression-checked by the bit-identical ablation re-run.
- Framework crate: untouched code → not re-tested (README-only addition).

## Performance / provenance

- CPU-only. **No GPU** (per user: GPU install in another thread). Peak RSS
  within the toy budget; ablation wall < 5 s/seed. Prior speed baselines
  (~23–25× forward, Chebyshev-deploy 1.7–1.9×) unchanged — not re-measured
  (cached facts).
- `nagare_github` HEAD `7534891` → dirty (this reconcile + the earlier
  order-shuffle + the pre-existing July-5 WIP). Framework HEAD `4320202`.
- rustc 1.93.1; seeds 53 (identity check). Ported ops byte-verbatim from the
  framework crate; fixture FNV hashes match.

## Open issues / follow-ups

1. **Commit/push (user's call).** GitHub working tree now holds: this reconcile,
   the order-shuffle work, and the pre-existing July-5 WIP (ops/runtime port +
   `nature-like-venue-readiness.md`). Suggest committing in that logical order.
2. **Detachment (future).** Vendor `hymeko_clifford` + `hymeko_graph` into the
   standalone repo, drop the two path deps, then delete the framework
   `hymeko_nagare` crate + its workspace membership.
3. **Reports location.** The Nagare reports currently live in the framework
   `reports/`; copies of the order-shuffle report were placed in
   `nagare_github/reports/` for self-containment. Decide the canonical home.
4. **Deferred science (unchanged):** `FixedProjection` gate (fitting-vs-fixed),
   multi-class 2→K.
