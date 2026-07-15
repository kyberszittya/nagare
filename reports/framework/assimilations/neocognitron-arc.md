---
experiment_id: neocognitron-arc
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
scope: retroactive assimilation of the full Neocognitron arc (N0..N4/entropy + pose P1..P4)
---

# Assimilation — Neocognitron arc

First application of the assimilation lifecycle, run retroactively over the completed Neocognitron arc before the
next experiment (per §15).

## 1. Experiment result

Built a closed-form, FD-verified, no-autograd Neocognitron: `conv2d` S-cell (N0), `group_pool` C-cell (N1), the
`ScBlock` deep stack (N3), the `global_entropy_pool` group-invariant top (entropy-top; user hypothesis confirmed),
and a spatial-backbone pose net (P2) with a skeleton `hg_conv` characterized across P3 (win) and P4 (limit).

## 2. New evidence

Reports: `2026-07-13-neocognitron-n0-conv2d.md`, `-n1-ccell.md`, `2026-07-14-neocognitron-n2-stack.md`,
`-n2b-learned-scell.md`, `-n3-deep-stack.md`, `-entropy-top.md`, `2026-07-14-sbsh-pose-p2-backbone.md`,
`-p3-skeleton-win.md`, `-p4-closed-loop.md`. Figures + per-seed JSON under `reports/figures/`.

## 3. Novelty classification

- `NEW_CANONICAL_CAPABILITY` — F-ENT-1 (covariance eigen-entropy: global-invariant recognition + equivariant pose + real-time).
- `NEW_MECHANISM` — F-N3-1 (C-cell load-bearing when compositional), F-N3-2 (local != global invariance), F-P3-1 (skeleton decisive under redundant coupling).
- `NEW_EVALUATION_REQUIREMENT` — F-ARC-1 (prior pays iff base lacks signal AND op can express it), F-P4-2 (occlusion-box leak).
- `NEGATIVE_RESULT_REQUIRING_A_GUARD` — F-ENT-2 (entropy cold-start), F-P4-1 (shared-elin can't express loop closure).
- `NEGATIVE_RESULT_CLOSING_A_DIRECTION` — F-N2b-1 (C-cell redundant under learnable single block), F-P2-1 (skeleton neutral when over-constrained).

## 4. Canonical interpretation

The arc's throughline (F-ARC-1) is the canonical claim: structure is not free virtue; it pays exactly where the
base mechanism lacks the signal and the prior's op can express the constraint. The entropy pool and the P3
skeleton are the two clear wins; the C-cell-under-learnable-conv and the P4 loop are the two clear non-wins, each
for a named reason.

## 5. Framework impact

- Schema: none changed.
- Interfaces: `ScBlock` gains `new_oriented`; `metrics` gains `auroc`; both re-exported from `lib.rs`.
- Defaults: `global_entropy_pool` recorded as the recommended rotation-invariant top over the channel-mean.
- Reusable components: `oriented_sobel_bank`/`ScBlock::new_oriented` (extracted), `auroc` (consolidated).
- Guards: F-ENT-2 (cold-start warm-start), F-P4-1 (do not canonicalize shared-elin refine for loops), F-P4-2 (occlusion-box-leak eval rule).
- Closed/superseded: channel-mean top superseded by entropy pool for the invariant path; 4x `oriented_conv_init` and 2x `auroc` example copies removed.

## 6. Source changes

| file | change |
|---|---|
| `src/ops/sc_block.rs` | + `oriented_sobel_bank`, `ScBlock::new_oriented`, 2 tests (F-ENT-2 guard) |
| `src/metrics.rs` | + `auroc` (Mann–Whitney U) + test |
| `src/lib.rs` | re-export `oriented_sobel_bank`, `auroc` |
| `examples/neocognitron_entropy.rs` | import `oriented_sobel_bank`, `auroc` (drop 2 private copies) |
| `examples/neocognitron_deep.rs`, `examples/neocognitron_aniso.rs` | import `auroc` (drop copies) |
| `examples/pose_backbone.rs`, `pose_symmetry.rs`, `pose_loop.rs` | import `oriented_sobel_bank` (drop copies + unused `PI`) |
| `reports/framework/canonical_components.json`, `canonical_findings.json` | created |

## 7. Components added or updated

Added to registry: `sc_block`, `global_entropy_pool`, `oriented_sobel_bank/new_oriented`, `auroc` (CANONICAL);
`conv2d`, `group_pool`, `soft_argmax` (CANONICAL, pre-existing, recorded). `skeleton_hg_refine` recorded as
`MECHANISM_DEMO` with extraction blocked pending the per-edge transform.

## 8. Defaults changed

`global_entropy_pool` is the default/recommended rotation-invariant top (arrangement-blind channel-mean demoted).
No runtime default was silently flipped; the entropy pool wins on measured evidence (F-ENT-1).

## 9. Negative findings and guards

F-ENT-2 → `oriented_sobel_bank` guard + tests. F-P4-1 → `skeleton_hg_refine` kept non-canonical, per-edge
transform recorded as the fix. F-P2-1 / F-N2b-1 → closed-direction records. F-P4-2 → occlusion-box-leak eval rule.

## 10. Superseded paths

- Channel-mean rotation-invariant top → superseded by `global_entropy_pool` for the invariant path (historical mean-top remains available via `neocognitron_entropy --mean-top`).
- `oriented_conv_init` (×4), `fn auroc` (×3 of ~7) → superseded by framework symbols. Remaining `auroc` copies in `cpml_signed_link`/`signed_link`/`cr_edge_encoder` flagged for a follow-up consolidation (not this session's arc).

## 11. Regression tests

`sc_block::oriented_bank_is_structured_not_isotropic`, `::new_oriented_installs_the_bank`,
`metrics::auroc_tests::perfect_and_chance_and_degenerate` — added. Full suite **168 / 0**; fmt + clippy clean.

## 12. Remaining open questions

- Per-edge transform op → then re-run P4 to confirm the loop flips *hurts → wins* and to enable reflection symmetry.
- Multi-object / cluttered scenes: per-region entropy pools.
- General (non-parallelogram) 4-bar: nonlinear closure may need a per-edge nonlinearity.
- Follow-up: consolidate the remaining `auroc` copies; consider extracting the keypoint-backbone (`ScBlock → conv head → soft_argmax`) once a second consumer exists.

## 13. Next-experiment authorization

**Authorized.** The successful components are extracted and imported by their drivers; the negative lessons are
guards or closure records; the registries and ledger exist; regressions pass. The clearest next experiment — the
**per-edge transform op** — is genuinely new: it reuses `hg_message`/`ScBlock`/`global_entropy_pool`, is
constrained by F-P4-1/F-P3-1, would live in `src/ops/hg_message.rs` (or a new `hg_edge_transform` module) if
successful, and its failure (loop still hurts) would be assimilated as a stronger closure on the signed-hg_conv
expressivity limit.
