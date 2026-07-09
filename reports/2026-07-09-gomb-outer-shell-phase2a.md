# Gömb → Nagare, Phase 2a — outer FIR shell + shell-vs-flat

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-09-gomb-outer-shell/` (4 artifacts)

## Summary

First increment of the Gömb (spherical-shell) line. Gömb is a three-shell cortical
cascade (`models/hymeko_gomb/`): **outer** = M parallel Clifford-FIR banks, **middle** =
a SignedKAN layer, **inner** = CPML. Nagare already has two of the three pieces —
`clifford_fir` is the FIR bank and `hsikan` (Phase 1) is the middle shell — so Phase 2 is
mostly composition. 2a builds the **outer shell** op and answers the handoff's question:
*does M-bank shell aggregation beat flat (M=1) pooling on mixed arity?*

## The op (`src/ops/gomb_shell.rs`)

M parallel `CliffordFIR` banks over the signed cycle pool, concatenated →
`(n_cycles, M·d)`. **Pure orchestration over the existing `clifford_fir`** — the forward
is M `clifford_fir_forward` calls; the backward slices `∂L/∂Y` per bank, accumulates
`clifford_fir_backward`'s feature gradient across banks, and keeps each bank's Clifford
filter gradient. No new derivative hand-rolled. Mixed arity is handled by per-arity
groups (a `CliffordFIR`'s length = its cycle arity), like `hsikan`.

Verified: **FD gradient** (features + every bank's a,b vs central-diff), **M=1 ≡ bare
`clifford_fir`** (the flat baseline is exact), and the concat layout.

## Discriminating A/B (`tests/gomb_shell_vs_flat.rs`)

Mixed-arity signed cycle pool (10 arity-3 + 10 arity-4), FIXED random node features,
labels from a teacher with **2-filter structure** (a fixed 2-bank shell + linear readout,
median-split). Two students learn it by closed-form backward (features frozen; only banks
+ readout train):

| student | BCE (init → final) | acc |
|---|---|---|
| **shell (M=2)** | 0.6308 → **0.0672** | 1.000 |
| flat (M=1) | 0.7064 → **0.1102** | 1.000 |

**Verdict: shell beats flat (Δ 0.043 BCE).** On a target with multi-filter structure, the
shell's extra bank capacity is usable and pays off.

**Honest caveats (§3):** the target is **constructed** to need 2 filters, so the shell
winning is *expected*, not surprising — it demonstrates the shell's added capacity is
learnable, not that it wins on a natural task. Single seed. The test *reports* the verdict;
its pass condition only requires both to learn. The natural-task comparison (real
signed-link cycles, e.g. Bitcoin-Alpha) and multi-seed are follow-ups.

## Files touched

| file | change |
|---|---|
| `src/ops/gomb_shell.rs` | **new** — outer shell fwd+bwd + FD/equivalence/layout tests (`84eef36`) |
| `src/ops/mod.rs` / `src/lib.rs` | +1 mod / +re-export |
| `tests/gomb_shell_vs_flat.rs` | **new** — the shell-vs-flat A/B |

## CORE / deps

**None.** Reuses vendored `hymeko_graph::{clifford_fir_*, CliffordFIR, TopKCyclesBatch}`
(called, not modified) + `linear`/`cross_entropy` ops. No new dependency.

## Test results (both machines)

- Full suite **63 / 0** on Mac (arm64) + kato15 (x86_64); clippy `-D warnings` + fmt clean.
- Deterministic (seeded).

## Open / follow-up

1. **Multi-seed** + a **natural-task** shell-vs-flat (real signed cycles) to make the
   verdict a claim rather than a capacity demonstration.
2. **2b** — compose the middle shell (my `hsikan` op) after the outer: outer FIR → middle
   HSiKAN → readout (the two-shell cascade).
3. **2c** — the inner CPML shell + the full three-shell cascade.

## Provenance

Repo `github.com/kyberszittya/nagare`; op at `84eef36`, A/B this commit. Developed on
kato15 (Katolab back online), authored + mirrored via the Mac; both boxes + GitHub in sync.
Rust 1.96.1. Seeds fixed (features/teacher/students).
