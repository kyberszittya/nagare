# HSiKAN mixed-arity — entropy vs constant gate, multi-seed (report)

Date: 2026-07-08 · Author: Aiko (agent) for Hajdu Csaba · Follows 1c

## Summary

Turn the 1c single-seed signal (entropy 0.2857 vs constant 0.3104) into a
distributional claim (§3): entropy-gated vs constant local-update readout on **fixed
mixed-arity HSiKAN features**, over **15 independent seeds** (fresh feature params +
fresh linear-teacher labels per seed; both gates trained on the identical instance).

## Result (15 seeds, final BCE — lower better)

| gate | median | IQR |
|---|---|---|
| **entropy** | **0.2726** | [0.2210, 0.3452] |
| constant | 0.2842 | [0.2309, 0.3620] |

**entropy < constant in 11/15 seeds.** Figure:
`reports/figures/hsikan-multiseed-entropy-vs-constant.png` (paired scatter + medians).

## Honest reading (measured / inferred / caveat)

- **Measured:** the entropy gate has a **small, consistent** advantage on this
  mixed-arity regime — lower median (Δ ≈ 0.012) and the majority of seeds (11/15).
- **Inferred:** the direction is the **opposite** of the standing arity-2 result (where
  the constant gate won all 12 stress rows) — consistent with the hypothesis that
  entropy feedback helps where higher-arity structure is present.
- **Caveat (do not overclaim):** the **IQRs overlap heavily** and 4/15 seeds favour
  constant — this is a *modest* effect, not a decisive win. The single-seed 1c gap
  (0.025) sat on the larger end; the multi-seed median gap (0.012) is smaller — the
  point estimate slightly overstated it, which is exactly why the multi-seed pass was
  worth running. No significance test is claimed on 15 seeds; a larger N or a paired
  test would be needed to call it significant.
- **Task caveat:** labels are linearly separable by construction in the feature space,
  so this isolates the *learning rule's* convergence quality, not a task-difficulty
  win. A non-separable / real mixed-arity dataset is the stronger follow-up.

## Files touched

| file | change |
|---|---|
| `tests/common/mod.rs` | **new** — shared extractor/readout/toy/teacher scaffolding (lifted from 1c, §6.1) |
| `tests/hsikan_layer.rs` | refactored to use `common` (no behaviour change) |
| `tests/hsikan_multiseed.rs` | **new** — 15-seed median/IQR experiment + per-seed output |
| `scripts/dev/plot_multiseed.py` | **new** — paired-scatter plot |
| `reports/figures/hsikan-multiseed-entropy-vs-constant.png` | **new** — the figure |

## CORE / deps

**None.** matplotlib was run via `uv run --with matplotlib` (ephemeral env, **not** a
project dependency). Refactoring 1c into `tests/common` removed duplication (the third
user of the scaffolding, §6.1) with no behaviour change.

## Test results

- **Mac (arm64) only** — per the user's switch to Mac-only development (kato15 out of
  the loop). Full suite **59 / 0**; clippy `-D warnings` + fmt clean. The experiment is
  deterministic (fixed seeds), so the Mac result is reproducible without kato15.

## Open / follow-up

1. Larger N + a paired significance test if the entropy edge is to be *claimed*, not
   just reported.
2. A **non-separable / real** mixed-arity signed-hypergraph task (Bitcoin-Alpha
   n-tuples) — the regime where the 0.934-vs-0.886 HSiKAN-over-SGCN result lives.
3. **Phase 2 (Gömb)** — rotor/Clifford shells.

## Provenance

- Repo `github.com/kyberszittya/nagare`, base `06a3c50`. Rust 1.96.1 (Mac arm64).
- Seeds: extractor `s`, teacher `1000+s`, readout `2000+s`, `s ∈ 0..15`; 300 epochs, lr 0.1.
- Plot via `uv run --with matplotlib --with numpy`. Not committed/pushed yet.
