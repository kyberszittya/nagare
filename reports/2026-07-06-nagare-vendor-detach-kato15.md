# NAGARE — vendor the framework deps, detach, and run on kato15

Created-at: 2026-07-06 18:34 JST
Repo: `nagare_github` = `github.com/kyberszittya/nagare` (standalone). **All local; commit/push held.**
Remote host: `kato15` (RTX 6000 Ada box; used here as a 32-core Linux build/run host — Nagare is CPU-Rust, no GPU).

## Summary

Completed the detachment step: vendored the two framework path deps into the
standalone repo, verified it builds+tests with zero reference to the framework,
then proved detachment by building and running on **kato15**. Detachment proof:
the vendored repo at `/tmp/hajdu/nagare` builds resolving **only** from
`vendor/*` (grep-verified: no path references the framework) **even though a
framework copy is present on the box** at `~/hymeko_framework_rust` — the build
never touches it.

Correction (under-consulted `reference-katolab-gpu-kato15` first): kato15 is
**already provisioned** — cargo 1.96.1 lives at `~/.cargo` (my `bash -lc`
`command -v cargo` missed it because `~/.cargo/env` is sourced from tcsh's
`~/.cshrc`, not a bash profile), and the framework workspace is at
`~/hymeko_framework_rust`. I **redundantly** installed a second toolchain to
`/tmp`. It worked, but future runs should `source ~/.cargo/env` and reuse the
existing toolchain.

## Vendoring

- `vendor/hymeko_clifford/` (776 LOC, **zero deps**) and `vendor/hymeko_graph/`
  (10,880 LOC; deps `rayon`/`dashmap`/`rand`, all crates.io — no framework
  cascade), copied verbatim (minus `target/`, `Cargo.lock`).
- `Cargo.toml`: path deps repointed `../hymeko_framework_rust/*` →
  `vendor/*`.
- Verified self-contained: `grep` finds **no** path referencing the framework;
  clean `cargo build` + **45 tests pass** resolving entirely from `vendor/`.
- Neither crate is in CORE.YAML; framework untouched (read-only copy).
- Note: `hymeko_graph` is heavier than the narrow repo warrants (only
  `CliffordFIR` + `TopKCyclesBatch` + `clifford_fir_*` are used, by the
  cycle-pool runtime). Slimming it (feature-gate the runtime, or extract the
  FIR surface) is a follow-up; wholesale keeps behavior + all tests exact.

## kato15 run (detachment proof)

- Host: kato15, Linux 5.15 x86_64, 32 cores, 125 GB RAM; bare box (no Rust) →
  installed rustup **to `/tmp`** (`CARGO_HOME`/`RUSTUP_HOME` under
  `/tmp/hajdu`, ephemeral, NFS home untouched); rustc/cargo **1.96.1**.
- Transferred the working tree (tar-over-ssh, excl. `target/`/`.git`) to
  `/tmp/hajdu/nagare` (uncommitted changes → can't `git clone`).
- **Builds from the vendored tree; ablation reproduces the science
  bit-identically at the metric level** (spiral seed 53: clean 0.001586,
  few-shot 0.250603, shuffled 0.017641, shuffled+few-shot 0.316421 — matches
  the Windows baseline to 6 decimals). Initially 44/45 tests passed (one fixture
  failure, diagnosed below); **after the platform-aware fixture fix, 45/45 pass
  on kato15** (and 45/45 on Windows).

## One test fails on Linux — fully diagnosed, benign

`seed53_datasets_match_frozen_fixture` fails on Linux. **Discriminating test
run** (regenerate on kato15, diff vs the Windows-frozen fixture):

- **Label hashes (Y_FNV) match** across platforms (integers → platform-independent).
- **XOR float-hash matches** (its generator uses no transcendentals); only
  **moons/spiral** differ — the two generators calling `cos/sin/atan2`.
- Even the 8-value preview is identical; a ≤1-ULP difference appears deeper in
  the 12,288 floats.

**Cause (isolated, not asserted):** Windows MSVC libm vs Linux glibc libm differ
by ≤1 ULP on transcendental functions → different exact f32 bits → different
FNV hash, while every downstream metric rounds identically. This is **not** a
bug, RNG drift, or vendoring defect — it is the known limitation of an
exact-float-bit determinism guard across platforms. The fixture is working as
designed; "drift" simply includes benign platform libm noise it can't
distinguish from a real generator change.

## Resolved — platform-aware fixture guard (option 1, user-approved)

Applied the platform-aware guard (user: "we can be sure to run and compile for
specific platforms"):

- The fixture header now records `# platform: arch-os` (stamped by the
  regenerate writer; the current frozen file is `x86_64-windows`).
- `seed53_datasets_match_frozen_fixture` asserts **labels + structure**
  (task/split/samples/points/seed/`y_fnv`) bit-for-bit on **every** platform,
  and the **exact float-hash + preview** only when running on the recorded
  freeze platform. Off-platform it emits a one-line skip note naming both
  platforms.
- Regenerated on Windows: the diff vs the prior fixture is **only** the added
  `# platform: x86_64-windows` line — the frozen hashes are unchanged (no
  silent re-anchoring).
- Result: **strict on Windows (freeze platform), green on kato15** (labels
  verified, libm-dependent floats skipped). 45/45 both platforms.

A future move to a tolerance-based guard (store float values, compare within
~1e-5) would give cross-platform *float* strictness too; not needed now.

## Files touched (local)

- `nagare_github/Cargo.toml`: path deps → `vendor/*`.
- `nagare_github/vendor/{hymeko_clifford,hymeko_graph}/`: vendored crates (new).
- kato15: `/tmp/hajdu/{cargo,rustup,nagare}` (ephemeral scratch; fixture there
  was regenerated to Linux values during the discriminating test).

**CORE.YAML: none. No new dependency** (vendored deps were already transitive).

## Follow-ups

1. ~~Fixture guard decision~~ — done (platform-aware, above).
2. Slim `hymeko_graph` vendor (feature-gate the cycle-pool runtime — Nagare only
   uses `CliffordFIR`/`TopKCyclesBatch`/`clifford_fir_*`).
3. Commit order still held for the user (July-5 WIP → order-shuffle → reconcile →
   vendor/detach + fixture fix).
