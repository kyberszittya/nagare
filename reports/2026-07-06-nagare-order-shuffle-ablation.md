# NAGARE — point-order-shuffle holonomy ablation: does the projection gate learn, or ride on fixed features?

Created-at: 2026-07-06 16:40 JST
Plan: `docs/plans/2026-07-06-nagare-order-shuffle-ablation/plan.{tex,pdf,tikz,mmd}` (gitignored; created 16:22, ETA ≈17:20 — finished ~16:40, under estimate)
Repo under test: `nagare_github` = `github.com/kyberszittya/nagare` (the extracted, GitHub-synced Nagare). **All changes local; commit/push held for the user.**

## Summary

Ran the discriminating science test carried since 2026-07-02: *does the fitted
projection gate do learning work, or does it ride on strong fixed holonomy
features?* The knife is a **within-sample point-order shuffle**, added as two new
`StressKind` variants (`Shuffled`, `ShuffledFewShot`) on the existing
`run_stress_ablation` harness (extension, not a new loop — §6.1).

**Mechanism (why the knife is clean).** The pooled descriptor is
mean/std/max/sign-entropy over 7 per-point channels: geometry `(x,y,r)`, rotor
`(rx,ry)`, holonomy `(w,z)`. Geometry+rotor channels are *per-point*, so their
pooled statistics are **permutation-invariant**; only the holonomy channels (the
running quaternion product) are order-sensitive. Shuffling within-sample point
order therefore perturbs **only** the holonomy signal. A dedicated regression
test (`tests/order_shuffle.rs`) proves this on the pooled descriptor:
geometry/rotor deviation under shuffle `< 1e-3` (rounding), holonomy deviation
`> 1e-2`.

## Result (3 seeds — 53/54/55 — median; frozen 192/96/32, 50 epochs)

Figure: `reports/2026-07-06-nagare-order-shuffle.png` (aggregate:
`...-agg.json`; per-seed: `...-s{53,54,55}.json`). Accuracy saturates at 1.000
everywhere on clean data, so **test cross-entropy loss is the discriminator**.

### Finding 1 — the projection gate does **not** robustly beat the constant gate

Spiral (the only order-dependent task), median test loss by regime:

| regime | entropy | constant | **projection** |
|---|---:|---:|---:|
| clean | 0.00472 | **0.00168** | 0.00168 |
| few-shot noisy+missing | 0.2374 | **0.2168** | 0.2335 |
| shuffled | 0.02179 | **0.01607** | 0.01703 |
| shuffled + few-shot | 0.2859 | **0.2893** | 0.3164 |

The constant gate (plain linear readout of the same pooled features) is `≤` the
projection gate on **median in every regime**. The previously reported "first
positive gate result" — spiral few-shot **0.2506** — was **seed 53 only**; on
seeds 54/55 the projection gate *loses* to constant (0.2335, 0.1806 vs 0.2168,
0.1689). This is a textbook §3 "single-seed is a point estimate, not a verdict":
the projection gate's advantage does not survive seed variation.

### Finding 2 — holonomy order carries class signal, but only on spiral, and gate-independently

Production shuffle Δloss = median(shuffled) − median(clean):

| task | entropy | constant | projection | reading |
|---|---:|---:|---:|---|
| moons | +0.00014 | +0.00005 | +0.00005 | order-invariant (geometry) |
| **spiral** | **+0.0171** | **+0.0144** | **+0.0153** | order-sensitive (holonomy) — ~10× loss |
| xor | −0.00039 | −0.00017 | −0.00024 | order-invariant (geometry) |

Shuffling raises spiral loss ~10× (0.0017→0.017) and is a no-op (within seed
noise) on moons/xor. Because geometry pooled stats are permutation-invariant by
construction (Finding-1 mechanism, test-proven), the spiral signal it destroys
**lives in the holonomy channels**. But the degradation is essentially the same
for all three gates → it is a **feature** property, not a **gate** property.

### Finding 3 — where the gate leans hardest on holonomy, destroying it hurts most (but still no win)

In the hard few-shot regime the projection gate degrades most under shuffle
(spiral Δloss: projection **+0.083**, constant +0.072, entropy +0.049 — Panel
B, hatched). So the projection basis *does* lean on order-sensitive holonomy
structure — but that lean never converts into an advantage over the constant
gate (Finding 1).

### Answer to the science question

**The projection gate rides on the fixed features; it does not do robust
learning work.** On multi-seed median it fails to beat a plain linear readout of
the same pooled descriptor in every regime tested. The holonomy *features* are
genuinely load-bearing — but only on the rotationally-structured spiral task,
task-specifically and independently of the gate. The earlier positive gate
result was single-seed noise.

*Measured:* all four tables (3-seed medians). *Inferred:* "spiral signal lives
in holonomy" (from mechanism test + shuffle selectivity). *Still hypothesis:*
whether a *fitted* basis could beat constant with a different fit objective or on
a genuinely holonomy-rich task — untested here.

## Files touched (`nagare_github`, local only)

- `src/datasets.rs` (+78/−0): `shuffle_point_order` + 2 unit tests.
- `src/learner.rs` (+38 net): `StressKind::{Shuffled, ShuffledFewShot}`,
  `shuffles`/`is_few_shot` predicates, shuffle wiring in `run_stress_ablation`.
- `src/lib.rs` (+1): re-export `shuffle_point_order` (diff entangled with the
  pre-existing July-5 WIP re-exports; noted below).
- `tests/local_learner.rs` (1 line): smoke row count 4→6.
- `tests/order_shuffle.rs` (new, ~65): mechanism regression test.
- Artifacts (framework `reports/`): `2026-07-06-nagare-order-shuffle.png`,
  `-agg.json`, `-s{53,54,55}.json`.

**CORE.YAML items touched: none** (`nagare_github` has no CORE.YAML; framework
crate `hymeko_nagare` untouched). **No new dependency.**

## Test results

- `cargo test`: **22 passed, 0 failed** (14 lib incl. 2 new datasets tests +
  runtime; 2 local_learner incl. updated 6-row smoke; 1 new order_shuffle
  mechanism; 2 projection_basis; 3 runtime_training). Wall < 0.5 s.
- Coverage: every new function/variant driven by a test — `shuffle_point_order`
  by 2 datasets unit tests + the mechanism test; the new `StressKind` variants
  by the extended smoke (6 rows) and the experiment; behaviour-change regression
  = the 4→6 smoke assertion.
- Gates: `cargo fmt --check` clean; `cargo clippy --all-targets --no-deps -D
  warnings` clean. No `allow`/`unwrap`/`unsafe` introduced.
- §6.5 sweep: no anti-patterns — extended one enum + one harness, no new loop,
  no v2 files, no globals, no string-typed config.

## Performance vs budget

- Rust release ablation (3 tasks × 6 stresses × 3 gates): wall < 5 s/seed; peak
  RSS well under the 16 MiB toy budget (prior runs 9 MiB). CPU-only — **no GPU
  touched** (per user directive: GPU install in another thread). No perf
  regression: the added stresses are additive rows on the same forward path.
- No animation artifact: this is a static point-set classification ablation, not
  a policy/temporal/control task — §9's GIF clause does not apply; numerical +
  plotted forms are provided.

## Open issues / follow-ups

1. **`FixedProjection` gate** — add a gate that applies the *fixed* holonomy
   basis (no data fitting) to separate "does fitting help?" from "does the
   projection subspace help?". Needs a small `StressRow`/gate-list refactor
   (current 9 parallel gate fields are a mild §6.5 #1 smell — refactor to
   `Vec<GateOutcome>` when adding the 4th gate).
2. **Multi-class** — 2→K generalization of the learner (`b:[f32;2]`→`Vec`,
   `softmax2`→softmax-K) to test whether the binary result is a fixed-feature
   artifact.
3. **Memory correction** — `project-nagare-holonomy-line` calls the fitted
   projection gate "the first positive gate result (0.2506)"; this run shows
   that is single-seed. Memory updated alongside this report.
4. **Commit / divergence decision (user's)** — `nagare_github` carries
   uncommitted July-5 WIP (ops/runtime/optimizer port — verified green;
   `nature-like-venue-readiness.md` — aspirational). My changes sit on top,
   uncommitted. The framework crate `hymeko_nagare` is *ahead* on the projection
   kernel (`project_alpha_mix`, FD-tested, frozen fixture). Reconcile before any
   commit/push.

## Provenance

- `nagare_github` HEAD `7534891` (working tree dirty: my 5 files + pre-existing
  July-5 WIP `src/ops/`, `runtime.rs`, `optimizer.rs`, `benches/`,
  `tests/runtime_training.rs`, `Cargo.{toml,lock}`, `README.md`).
- Framework HEAD `4320202` (branch `hymeko-neuro-migration`).
- Host: Windows 11, rustc 1.93.1; matplotlib 3.11.0 via `uv run --no-project`.
- Seeds: 53, 54, 55 (top-level; sub-seeds derived deterministically in
  `run_stress_ablation`). Datasets generated in-process (seeded `StdRng`), no
  external fixture.
- GPU: not used.
