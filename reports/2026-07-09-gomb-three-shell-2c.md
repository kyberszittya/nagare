# Nagare — Gömb Phase 2c: CPML inner core + full three-shell cascade

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Closes the Gömb architecture arc: ported the **inner CPML core** and wired the **full
three-shell cascade** (outer Clifford-FIR → middle HSiKAN → inner CPML) end-to-end, closed-form.
The cascade **learns** (BCE 0.70→0.03). The inner tier-stratification, ablated L=3-vs-L=1 over
5 seeds, **does not robustly help** on a tier-structured target — a median tie (L=3 lower on
2/5 seeds). Honest negative, consistent with the stack's own "membrane must earn its weight"
principle.

## Discovery / disambiguation (§6.1)

Two different objects share the acronym **CPML**:
- `hyperedge/cpml.py` = **Concentric-Pyramid Multi-Layer**: degree-percentile tier
  stratification + tier-restricted cycle routing + `concat(X₀,H₀…)` readout.
- `architecture/cognitive_stack/README.md` line 24 frames the inner core as "grade-preserving
  polynomial layers, grade-0 readout ⟨·⟩₀" — a *different* description.

The **implemented** `InnerCPMLCore` (`models/hymeko_gomb/shells.py`) wraps the tier-stratified
CPML (`CPMLConfig`→`CPML`), not a grade-polynomial layer. **Code is authoritative** — this port
mirrors the tier stratification. The README's grade-poly line is aspirational/divergent and is
flagged, not ported (the kind of doc-vs-code layer mismatch CLAUDE.md's quarantine warns of).

## What was built

**New op `src/ops/cpml_tier.rs`** — the genuinely new primitive (the aggregation reuses
`linear`/`scatter_mean`):
- `TierSpec::{new, uniform, assign}` — degree → percentile rank → tier bin (port of
  `TierSpec.assign`: tier 0 closed-left `[c₀,c₁]`, later tiers half-open `(cℓ,cℓ₊₁]`).
- `cycle_incidence_degrees` — unsigned corner-count degree proxy.
- `tier_cycle_indices` — cycles touching a tier-ℓ vertex (hard incidence routing; overlapping,
  not a partition).

Tier assignment is a **fixed structural routing** from graph degrees — no backward (like the
cycle enumeration); `tier_of` is a fixed input to the differentiable part.

**Cascade `tests/gomb_three_shell.rs`** — the full three-shell (cascade.py order):
```
X → gomb_outer (M FIR banks) → scatter_mean → hsikan (signed CR-spline) → scatter_mean
  → inner CPML: per tier ℓ  [gather corners → linear → mean → scatter_mean → H_ℓ]
  → concat(X_core, H₀…H_{L-1}) → readout
```
Trained by the composed closed-form backward (readout → inner tiers → scatter → hsikan →
scatter → outer), features frozen.

## Results (5 seeds, L=3 teacher target)

| inner | median final BCE | median acc | L=3 wins |
|---|---|---|---|
| L=3 tiered (CPML) | **0.0121** | 1.000 | — |
| L=1 flat | **0.0133** | 1.000 | 2/5 |

**Cascade-learns gate (seed 0):** BCE 0.70 → 0.03 (≈25×) — the three shells compose and the
whole stack trains. **Inner ablation verdict:** *does not robustly help* — median BCE is tied
(ΔBCE 0.0012, negligible), both at 100% median acc, and the flat L=1 inner wins 3/5 seeds.
Plot: `reports/figures/gomb-three-shell-inner-ablation.png`.

## Reading (measured / inferred)

- **Measured:** full three-shell closed-form forward+backward learns; L=3-vs-L=1 inner is a
  median tie over 5 seeds.
- **Process note (the single-seed trap):** the *first* single-seed run showed L=3 winning by
  ΔBCE 0.076 ("helps") — a tiers-favorable draw. Multi-seed flipped it to a tie. This is the
  §3 "single-seed is a point estimate, not a verdict" rule catching a false positive in real
  time; the verdict logic now labels "helps" only on a majority of seed wins.
- **Inferred:** on a small graph (12 vertices, uniform [4,4,4] tiers) both inners saturate the
  easy teacher target (100% acc); the tier concat adds params without a robust BCE edge — the
  same shape as this session's KB and graph-vs-KAN negatives. Whether tiers help on a large,
  genuinely degree-heterogeneous graph (the CPML paper's Slashdot/Epinions regime) is untested
  here and is the honest open question.

## Files touched

| file | change |
|---|---|
| `src/ops/cpml_tier.rs` | **new** — `TierSpec`, `cycle_incidence_degrees`, `tier_cycle_indices` + 5 unit tests |
| `src/ops/mod.rs`, `src/lib.rs` | +mod / +re-export |
| `tests/gomb_three_shell.rs` | **new** — full three-shell cascade + L=3-vs-L=1 inner ablation (5 seeds) |
| `scripts/dev/plot_three_shell.py`, `reports/figures/gomb-three-shell-inner-ablation.png` | **new** — plot |

## CORE / deps

**None.** Standalone op + composition test; no dependency change.

## Test results

- Full suite **92 / 0** on Mac (arm64); clippy `-D warnings` + fmt clean. kato15 mirror pending.
- `cpml_tier` unit tests: 5/5 (tier sizes, monotonicity, L=1 flat, routing coverage, degrees).
- Three-shell test runs in ~7 s (5 seeds × 2 students × 600 steps).

## Open / next

- Gömb architecture arc (2a outer, 2b middle, 2c inner + full cascade) is **complete**. The
  degree-tier benefit on a large heterogeneous-degree signed graph is the natural follow-up if
  the inner core is to be justified beyond "correct and composes".

## Provenance

Repo `github.com/kyberszittya/nagare`. Rust 1.96.1. Reference: `hymeko_neuro/hyperedge/cpml.py`,
`models/hymeko_gomb/{shells,cascade}.py`. Deterministic seeds 0..4.
