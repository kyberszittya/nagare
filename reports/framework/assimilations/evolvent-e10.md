# Assimilation — Evolvent E10 (contiguous-storage rewrite)

Date: 2026-07-16 · lifecycle per `feedback-assimilation-lifecycle-protocol` · Mac · **criterion 0.5 (§10)**

## 1. Experiment → evidence

E10 does the rewrite E9 (F-EVO-11) diagnosed. `JunctionTreeCholesky`'s internals move from per-clique
`Vec<Vec<f32>>` to **flat arenas** (precomputed offsets for frontals/rhs/factors), a **precomputed parent-local
assembly map**, and **reused scratch** in `factorize` — no per-clique allocation, no `position()` search in the hot
path. Re-benchmarked (criterion, Mac):

- MF solve **2.6–2.8× faster** (d=889: 31.7 µs vs 85 µs).
- Measured speedup vs dense: **5.9×–255× → 16.7×–682×**.
- Flop-to-measured gap: **~5.0× → ~1.8×** (at d=105, 1.08× — at the flop ceiling).

Evidence: `reports/2026-07-16-evolvent-e10-contiguous.md`, `reports/figures/evolvent-contiguous.png`,
`reports/figures/evolvent_e10_results.json`.

## 2. Novelty classification

`OPTIMIZATION` (F-EVO-12). Resolves F-EVO-11's open "contiguous-storage rewrite" item and confirms E9's diagnosis
(the ~5× was allocation overhead) was correct — the measured 2.7× self-speedup is exactly that overhead removed.

## 3. Canonical decision

`JunctionTreeCholesky` **updated in place** — same public API, flat-arena internals. Deployable speedup is now
16.7×–682×; the component note and F-EVO-11's known-limitation are updated.

## 4. Framework integration

In-place component update, **not** a `JunctionTreeCholesky2` / `_contiguous` variant (§6.5 #13). The linear-algebra
helpers were consolidated into buffer-based `*_into` forms reused by both `factorize` and `solve_block_diagonal` —
one implementation, no duplication. Dead `Clique::n_sep` removed.

## 5. Regression protection

No new tests needed: correctness is a **layout-invariant** property here (arithmetic unchanged), and the four
existing exactness/locality tests (`== dense` on single / branching / star trees + update-locality) guard it —
they pass, and the E5–E8 example outputs are byte-identical to pre-rewrite. Full suite **179/0**, fmt + clippy
clean.

## 6. Source-of-truth update

- `canonical_findings.json` — F-EVO-12 added.
- `canonical_components.json` — `JunctionTreeCholesky` note + known-limitations updated (rewrite done, 16.7×–682×).
- Report + figure + results JSON on disk.
- Memory updated.

## 7. Honest limitations carried forward

- The 682× is vs a Gaussian dense baseline (~2× the flops of pure Cholesky) → the layout-independent number is the
  **2.7× self-speedup**; a Cholesky baseline would leave ~2× more residual.
- Single machine (ratios transfer).
- Remaining headroom below the ceiling (blocked/SIMD triangular kernels) not pursued — E9's ~5× caveat is closed.

## 8. Environment

Run on the Mac (arm64, CPU). **kato15 not synced this session** — origin `main` ahead, fast-forwards on next pull.

## 9. Next (NOT yet authorized)

Blocked/SIMD triangular kernels + a dense-Cholesky baseline (the residual below the ceiling); optimized
`refactorize_path` incremental online solver; higher-order (degree ≥ 3) representability.
