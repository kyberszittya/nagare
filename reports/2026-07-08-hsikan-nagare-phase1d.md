# HSiKAN → Nagare, Phase 1d — performance + the chunk_t memory cap

Date: 2026-07-08 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-08-hsikan-nagare-phase1/` (§Risk anticipation) · Follows 1a–1c′

## Summary

Characterise both deploy axes (forward latency + training-step cost) and close the
Phase-1 **highest risk**: the naive forward materialises `(T,k,S,d)` intermediates
(~327 MB at Bitcoin-Alpha scale). Added `hsikan_forward_chunked` — a forward-only
streaming variant that drops each chunk's cache, holding peak heap to
`O(chunk·k·S·d)`. Latency measured with criterion (§10).

## Memory — the chunk_t cap (the risk, closed)

`tests/hsikan_perf.rs` (std-only tracking global allocator, peak live bytes) at
**T=5000, k=4, d=32, S=2**:

| forward | peak heap over baseline |
|---|---|
| naive (`hsikan_forward`, full cache) | **38 401 KiB** (37.5 MiB) |
| chunked (`hsikan_forward_chunked`, chunk=256) | **2 816 KiB** (2.75 MiB) |
| ratio | **13.6× smaller** |

The chunked output is **bit-identical** to the naive `h_e` (asserted). Peak scales with
`chunk_t`, not `T` — so the 10⁵-row Bitcoin-Alpha forward is bounded by choosing
`chunk_t`, never materialising the full `(T,k,S,d)`. Well under the 16 GB cap and the
plan's 50 MiB toy budget. **Contract inherited** (the PyTorch `HSIKAN_CHUNK_T` cap), as
§2 required, and proven by a test rather than assumed.

Note: chunking is the **forward-only / feature-extraction** path (no backward cache).
Training that needs the backward cache at 10⁵ rows would use gradient-checkpointed
recompute-in-backward — a follow-up if a full-scale *trained* run is required; the
feature-extraction use (1c) is forward-only and fully covered now.

## Latency — criterion (median), Mac arm64

Config: k=4, d=32, S=2, grid=6, cheb_k=4, highway on.

| bench (median, 100 samples) | Mac arm64 | kato15 x86 | plan budget |
|---|---|---|---|
| `hsikan_forward_b1` (1 edge) | **4.10 µs** | 5.93 µs | < 3 µs (over ~1.4×) |
| `hsikan_forward_t1000` | **1.456 ms** | 5.195 ms | < 300 µs (over ~4.9×) |
| `hsikan_train_step_t1000` (fwd+bwd) | **3.50 ms** | 8.10 ms | < 1 ms (over ~3.5×) |

**kato15 is ~3.5× slower than the Mac** — expected, not a regression: the op is
**single-threaded scalar f32** (no rayon), so kato15's 32 cores are idle and Apple
Silicon's stronger per-core wins. If throughput matters, the op is embarrassingly
parallel over edges (rayon over the T-chunks) — a profiled follow-up, not this task.

**The measurements exceed the plan's budgets — reported, not buried.** Per §3 this is
analysed, not declared: the budgets were optimistic because they under-counted the
**highway gate**. The gate is a `Linear(d,d)` evaluated per *(edge, vertex)*, i.e.
`O(T·k·d²)` = 1000·4·32² ≈ **4.2 M** mults at d=32 — on par with the two Chebyshev-CR
spline stages (≈2.5 M CR evals). The measured ~1.46 ms is consistent with that FLOP
count on scalar f32 (~5 GFLOP/s effective with the CR gathers), so it is **inherent to
the architecture at these dims, not a defect** — no accidental O(n²) beyond the
intended d×d gate. The train step (3.50 ms) = forward (1.46) + backward (~2.0), a
sane ~1.4× backward ratio.

**No optimization applied** (§3 forbids it without a profiled hot spot + plan
justification; this task's goal was correctness + the memory cap). Known levers *if*
deploy latency later matters, as a profiled follow-up: (a) the highway gate
(`O(d²)`/row — the dominant term); (b) the inner spline currently evaluates **both**
sign branches for **every** row then masks (a ~2× redundancy, matching the PyTorch
batched spline) — a closed-form op could evaluate only each row's own branch.

## Files touched

| file | change | lines |
|---|---|---|
| `src/ops/hsikan.rs` | `hsikan_forward_chunked` + `chunked_matches_naive` test | +50 |
| `src/lib.rs` | re-export | +1 |
| `tests/hsikan_perf.rs` | **new** — tracking allocator + memory-bound test | 118 |
| `benches/hsikan_bench.rs` + `Cargo.toml` | **new** criterion bench + `[[bench]]` | 100 |

## CORE / deps

**None.** No dependency added (criterion already a dev-dep; tracking allocator is
std-only; `[[bench]]` is a build target, not a dependency).

## Test results (both machines)

- Full suite **58 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean on both. Memory cap 13.6× on both.

## Open / follow-up

1. **Multi-seed** entropy-vs-constant (1c signal).
2. Gradient-checkpointed chunked **training** (only if a full-scale trained run is needed).
3. **Phase 2 (Gömb)** — rotor/Clifford shells.

## Provenance

- Repo `github.com/kyberszittya/nagare`, base `927bad7`. Rust 1.96.1 both boxes.
- Host for latency: Mac (Apple Silicon arm64). Deterministic LCG data (no RNG dep in bench).
- Not committed yet — awaiting the run + user's go + the deploy key for push.
