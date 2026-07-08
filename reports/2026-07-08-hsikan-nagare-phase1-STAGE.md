# STAGE REPORT — HSiKAN → Nagare, Phase 1 (2026-07-08)

Author: Aiko (agent) for Hajdu Csaba · Repo `github.com/kyberszittya/nagare` @ `a8ca716` (working tree dirty, uncommitted)

Self-contained ledger of the HSiKAN→Nagare port stage. Consolidates the three
sub-reports (1a/1b/1c) and the environment bring-up. Continues
[[project-nagare-holonomy-line]]; plan `docs/plans/2026-07-08-hsikan-nagare-phase1/`.

## Environment (established this session)

- **kato15** (`ssh kato15`, Katolab RTX 6000 Ada / 32c / 125 GiB) = **main dev box**;
  clone at `~/nagare`. **Mac** (`nagare_github`) = CPU runner + reference. Both synced
  to `a8ca716`; GitHub `origin/main` is the single source of truth. Details + gotchas
  in memory `project-nagare-kato15-mac-dev-setup`.
- **Push-back to GitHub is still blocked** on registering kato15's deploy key
  (write) — nothing committed yet.

## Stage ledger

| phase | deliverable | status | discriminating result |
|---|---|---|---|
| **1a** | HSiKAN core as closed-form op (`src/ops/hsikan.rs`), fwd+bwd | ✅ done | FD backward == central-diff on all 5 grad buffers (tol 1e-3) |
| **1b** | Parity vs the real PyTorch `SignedKANLayer` | ✅ done | **max\|Δ\| = 1.19e-7** (f32 vs float64, tol 1e-3) over arities {3,4} |
| **1c** | Nagare entropy-gated local learning (mixed-arity) | ✅ done | entropy **0.2857** vs constant **0.3104** BCE (both acc 1.0) — single-seed |
| **1c′** | HSiKAN spectral-entropy regulariser (closed-form op) | ⏳ next | needs Jacobi eigensolver + spectral-entropy backward, FD-tested |
| **1d** | Forward/train latency + peak RSS + `chunk_t` cap | ⏳ pending | budgets declared in plan; not yet measured |

## What Phase 1 established

1. **The HSiKAN core is a real Nagare op.** Forward (gather → inner Chebyshev spline
   → highway gate → sign-masked per-sign mean → diagonal outer spline → sum) + a
   hand-derived closed-form backward, reusing `catmull_rom.rs`'s Chebyshev (no basis
   re-implemented). The two ablation-critical pieces (highway gate, sign branches)
   are preserved.
2. **It matches the reference architecture exactly**, not just self-consistently.
   1b drives the *real* `SignedKANLayer` (via the Chebyshev↔CR-control-point mapping,
   both CR evaluators bit-identical) and the f32 Rust op reproduces its float64 output
   to f32 machine precision. This is the guarantee the 1a FD test *cannot* give (FD
   only proves backward-vs-forward consistency).
3. **It learns via the Nagare entropy-feedback substrate**, on mixed arity. HSiKAN as
   a fixed feature extractor + an entropy-gated local delta rule readout. On this
   single-seed mixed-arity toy the **entropy gate beat the constant gate** (0.2857 vs
   0.3104) — the *opposite* of the standing arity-2 result — which is a suggestive
   lead for the mixed-arity regime, **pending a multi-seed confirmation** (§3).

## Artifacts

- **Code:** `src/ops/hsikan.rs` (663 L), `src/ops/mod.rs` (+1), `src/lib.rs` (+re-export).
- **Tests (3 new):** `tests/hsikan_torch_parity.rs` (1b), `tests/hsikan_layer.rs` (1c),
  + in-module FD/regression tests in `hsikan.rs` (1a). Fixture
  `tests/fixtures/hsikan_torch_parity.txt`; generator `scripts/dev/hsikan_parity_fixture.py`.
- **Plan:** `docs/plans/2026-07-08-hsikan-nagare-phase1/` (tex/pdf/tikz/mmd; on disk, not committed — public repo/IP).
- **Reports:** `2026-07-08-hsikan-nagare-phase1{a,b,c}.md` + this stage report.

## Verification (both machines)

- **Full suite 51 / 0** on Mac (arm64) and kato15 (x86_64); +6 tests over the 45 baseline.
- `cargo clippy --all-targets -- -D warnings` clean, `cargo fmt --check` clean on both.
- CORE.YAML: **none touched.** New dependencies: **none** (std-only fixture parser;
  reused `catmull_rom`/`linear`/`loss`/`metrics` ops).

## Honest caveats / risk carryover

- **Single-seed entropy signal.** 1c's entropy-beats-constant is one seed on a
  teacher-separable toy. Not a ranking claim until multi-seed median/IQR.
- **`chunk_t` memory cap not yet implemented.** The naive forward materialises
  `(T,k,S,d)` → ~327 MB at Bitcoin-Alpha scale. Must inherit PyTorch's streaming cap
  before any 10⁵-row run (a 1d deliverable, gated by a peak-RSS test).
- **Nothing committed.** All work is uncommitted on both trees pending the user's go
  and the kato15→GitHub deploy key.
- **Two workarounds recorded** (1b): std-only fixture format (avoided a serde dep);
  sklearn import-hook to load the real layer without installing a package. Neither
  touches the layer math.

## Next

1. **1c′** — spectral-entropy regulariser op (eigensolver + spectral backward + FD test), then wire into HSiKAN training.
2. **1d** — perf/RSS characterisation + `chunk_t` cap.
3. **Multi-seed** entropy-vs-constant on mixed-arity (turn the lead into a claim).
4. Then **Phase 2 (Gömb)** — rotor/Clifford shells (`cayley_rotor`, `clifford_fir`).

## Provenance

Rust 1.96.1 (Mac arm64 + kato15 x86_64); torch 2.11.0+cu128 (kato15, 1b reference,
float64, seed 53). Deterministic seeds throughout. Base commit `a8ca716`; dirty files:
`src/{lib.rs,ops/mod.rs,ops/hsikan.rs}`, `tests/*`, `scripts/`, `docs/plans/`, `reports/*`.
