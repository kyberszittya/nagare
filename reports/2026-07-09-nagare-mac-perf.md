# Nagare core — Mac (Apple M5 Pro) performance snapshot

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

First native-Mac run of the standing Nagare examples after the Mac-only switch, to
confirm the framework runs and to record the M5 baseline. `toy_compare` reproduced the
documented results (entropy-pool learner 60 params / 100% acc / sub-µs forward; entropy
gate negative on the arity-2 toys; projection gate best 0.2506 on `spiral
fewshot_noisy_missing` — matching prior reports to the digit). This report records the
two throughput/profile examples.

## Host

Apple **M5 Pro**, 18 cores (6 performance), 48 GiB, macOS 26.4; rustc/cargo 1.96.1;
release profile. Rayon auto-detected **18 threads**.

## `scaling_bench` (release)

| batch | points | input_dim | hidden | threads | median | µs/sample | Mrows/s |
|---|---|---|---|---|---|---|---|
| 1024 | 64 | 64 | 128 | 18 | 32.62 ms | **31.86** | **2.0** |

Single default config (the example runs one point, not the full sweep). 2.0 M
input-rows/s at hidden=128 on the M5.

## `forward_profile` (release) — stage breakdown

batch=96, points=48, hidden=32, reps=300, 18 threads. **Total 13.725 µs/sample.**

| stage | µs/sample | % total |
|---|---|---|
| `fused_update` (129→32, 4608 rows) | **5.021** | **36.6%** |
| embed (linear 2→32, 4608 rows) | 2.490 | 18.1% |
| head (linear 96→2) | 1.755 | 12.8% |
| first (linear 96→2) | 1.746 | 12.7% |
| pool2 (serial mean/std/max) | 1.364 | 9.9% |
| pool1 (serial mean/std/max) | 1.339 | 9.8% |
| entropy (exp/ln, 96) | 0.011 | 0.1% |

**Read:** the entropy-pool `fused_update` kernel is the dominant term (36.6%), then the
per-row embed (18%). The entropy computation itself is negligible (0.1%) — the "entropy
feedback" cost is in the fused local update, not the entropy math. The two serial pools
(~10% each) are the standing serial-vs-parallel lever noted in prior work (parallelising
them only pays above hidden ≈ 16 — the 7-channel learner pool is kept serial).

## Notes / caveats

- These are single-config example runs (diagnostic), not the multi-seed / criterion
  discipline used for the HSiKAN benches — recorded as an M5 baseline, not a headline claim.
- `signed_link` / `auroc_eval` were **not** run: they need the signed-link datasets that
  lived on kato15 (`/tmp/hajdu/signed/`), not present on the Mac. Regenerating/fetching
  that data on the Mac is a follow-up before the arity-2 signed-link line can run here.

## Provenance

Repo `github.com/kyberszittya/nagare` @ `73907b6` (clean). Examples: `toy_compare`,
`scaling_bench`, `forward_profile`. Deterministic (seeded). No new deps; no source change.
