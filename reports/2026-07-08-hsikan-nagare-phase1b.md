# HSiKAN → Nagare, Phase 1b — PyTorch parity (report)

Date: 2026-07-08 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-08-hsikan-nagare-phase1/` · Follows: `reports/2026-07-08-hsikan-nagare-phase1a.md`

## Summary

Cross-implementation parity between the closed-form Rust op `src/ops/hsikan.rs`
and the **real** PyTorch `hymeko_neuro` `SignedKANLayer` (spline_kind=`catmull_rom`,
`inner_skip="highway"`, `outer_skip="none"`, S=2). On a fixed mixed-arity fixture
(arities 3 and 4), the f32 Rust forward matches the float64 PyTorch reference to

**max|Δ| = 1.19e-7** (tol 1e-3) — f32 machine precision.

This is the discriminating gate the plan/handoff called for: it confirms the
**structure** (gather → inner spline → highway gate → sign mask → per-sign mean →
diagonal outer → sum) is *exactly* the reference architecture, not merely
self-consistent. (The 1a FD test proves backward-vs-forward consistency but cannot
catch a systematic misreading of the architecture; this parity test does.)

## The exact mapping (why parity is meaningful)

The Rust op is parametrised in **Chebyshev coefficients** `(S,d,k)`; PyTorch's
`catmull_rom` activation in **CR control points** `(S,d,grid)`. Since
`chebyshev_cr(coef) ≡ catmull_rom(chebyshev_control_points(coef))` and all three CR
evaluators are bit-identical (`core/splines.catmull_rom` ==
`hyperedge/splines._catmull_rom_eval` == `src/ops/catmull_rom.rs`, verified by
inspection — same clamp, `u`, floor-clamp, CR weight polynomials, control-index
clamping), driving the PyTorch layer with `control_points = cheb_coef @ basis.T`
(basis = `chebyshev_knot_basis`) makes it compute the identical spline. So the
fixture stores Chebyshev coefs (what Rust consumes) and the PyTorch layer output.

## Files touched

| file | change | lines |
|---|---|---|
| `scripts/dev/hsikan_parity_fixture.py` | **new** — fixture generator (drives the real `SignedKANLayer`) | 148 |
| `tests/hsikan_torch_parity.rs` | **new** — std-only parser + parity assertion | 162 |
| `tests/fixtures/hsikan_torch_parity.txt` | **new** — frozen fixture (committed) | — |
| `src/lib.rs` | +1 `pub use ops::hsikan::{…}` re-export | +4 |

## CORE.YAML items touched

**None.** No dependency added — deliberately used a **std-only text fixture format
+ hand-parser** rather than `serde_json` (not a direct dep; adding it would be a
§1 core change).

## Two workarounds (honest notes, neither touches the layer math)

1. **serde avoided:** fixture is a line-based text format parsed with `std` only.
2. **sklearn import hook:** the kato15 torch env lacks `sklearn`, and importing
   `hymeko_neuro.hyperedge.signedkan` transitively runs `data.datasets.synth`
   (dataset *synthesis*, which imports sklearn/pandas/networkx) — a path the layer
   forward never touches. Rather than install a package (a dep change) or
   reimplement the layer (which would risk transcribing the same structural bug
   into both sides), the generator installs a `MetaPathFinder` that returns
   package-shaped stubs for those genuinely-absent deps. The **layer computation
   is fully real and untouched**; only the unused synthesis imports are shimmed.
   A real install always wins (guarded).

## Test results (both machines: Mac arm64 + kato15 x86_64)

| gate | Mac | kato15 |
|---|---|---|
| `matches_pytorch_signedkan_layer` (parity, max\|Δ\|) | ✅ **1.19e-7** | ✅ **1.19e-7** |
| `cargo clippy --all-targets -- -D warnings` | ✅ | ✅ |
| `cargo fmt --check` | ✅ | ✅ |
| full suite | **49 / 0** | **49 / 0** |

## Performance

Parity forward is instantaneous at this fixture size; formal latency/RSS budgeting
remains **Phase 1d**.

## Open / follow-up

1. **1c** — mixed-arity toy + closed-form local-update loss-decreases (learning works).
2. **1d** — forward/train latency + peak-RSS characterisation vs plan budgets; add
   the `chunk_t` streaming cap (highest-risk carryover) with a peak-RSS test.
3. Parity currently pins S=2, highway-on, arities {3,4}. A follow-up can widen the
   fixture (S=1, highway-off, more arities) if we want broader coverage — the
   generator parametrises trivially.

## Provenance

- Repo `github.com/kyberszittya/nagare`, base `a8ca716` (working tree dirty).
- Reference: torch **2.11.0+cu128** on kato15, CPU float64, seed **53**; fixture =
  2 arities (3,4), d=3, grid=6, cheb_k=4, S=2, 6 nodes.
- Rust toolchain 1.96.1 both boxes. No new deps. Fixture committed (frozen);
  regenerate only via `scripts/dev/hsikan_parity_fixture.py` after an intentional change.
- Not committed to git yet — awaiting user's go + kato15→GitHub deploy key.
