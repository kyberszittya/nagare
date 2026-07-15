# Assimilation — Evolvent E9 (wall-clock validation; corrects F-EVO-7 magnitude)

Date: 2026-07-16 · lifecycle per `feedback-assimilation-lifecycle-protocol` · Mac · **criterion 0.5 (§10)**

## 1. Experiment → evidence

E9 closes the caveat E5–E8 repeated ("flop win is analytic, not wall-clock"). A `criterion` benchmark times the two
real solvers — `JunctionTreeCholesky::solve` (multifrontal) vs `InfoEvolventHead::solve` (dense Gaussian), identical
answer — on the E5 branching tree, d 105→889:

- **Measured speedup 5.9× → 16.4× → 63× → 255×** — grows exactly as `O(d·w³)` vs `O(d³)` predicts.
- **~5× below the analytic flop count** (18×→1270×), stable at 5.0× for large d.
- At d=889: **85 µs vs 21.8 ms** (online-viable).

Evidence: `reports/2026-07-16-evolvent-e9-wallclock.md`, `reports/figures/evolvent-wallclock.png`,
`reports/figures/evolvent_e9_results.json`.

## 2. Novelty classification

`VALIDATION_WITH_CORRECTION` (F-EVO-11). The *direction and growth* of F-EVO-7 are confirmed by the clock; the
*magnitude* is corrected downward ~5× (per-clique small-`Vec` allocation overhead + contiguous dense baseline). The
1270× headline was the **flop ceiling**; 255× is the **deployable** number.

## 3. Canonical decision

No new component. Correct the record: `JunctionTreeCholesky` note and F-EVO-7 now cite measured 5.9×–255× as
deployable and 18×–1270× as the ceiling. The ~5× gap is the contiguous-storage rewrite's measured target.

## 4. Framework integration

New `criterion` bench (`benches/evolvent_solve_bench.rs`) + `[[bench]]` registration. **No new dependency** —
`criterion 0.5` is already a dev-dependency (the §10-pinned benchmark tool, in `tools.yaml`), so this is not a §1
core change.

## 5. Regression protection

A benchmark, not a correctness test — solver exactness is already guarded by the E5 `junction_tree` tests. Full
suite **179/0**, fmt + clippy clean (including the bench target).

## 6. Source-of-truth update

- `canonical_findings.json` — F-EVO-11 added; F-EVO-7 note corrected.
- `canonical_components.json` — `JunctionTreeCholesky` known-limitations + note updated with the measured number.
- Report + figure + results JSON on disk.
- Memory updated.

## 7. Honest limitations carried forward

- Single machine (absolute µs machine-specific; ratios transfer).
- Dense baseline is Gaussian elimination (~2× the flops of pure Cholesky) → the ratio mildly over-estimates the
  pure-algorithmic win.
- Both solvers re-factorize from scratch each `solve`; the contiguous-storage rewrite that would close the ~5× gap
  is not built.

## 8. Environment

Run on the Mac (arm64, CPU). **kato15 not synced this session** — origin `main` ahead, fast-forwards on next pull.

## 9. Next (NOT yet authorized)

Contiguous-storage multifrontal (close the ~5× gap toward the flop ceiling); a dense-Cholesky baseline; higher-order
(degree ≥ 3) representability.
