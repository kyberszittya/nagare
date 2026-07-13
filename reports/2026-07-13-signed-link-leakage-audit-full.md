---
title: "Nagare — the full 2×2 leakage audit: strict vs transductive under label shuffle (Table 2 complete, 4-graph)"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, leakage, label-shuffle, strict-protocol, transductive, nature, table2]
---

# Full leakage audit — {strict, transductive} × {real, shuffle}, 4 graphs

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `357f502`+ · CPU · 4 graphs × 4 conditions × 5 seeds

## Summary

Completes the Nature paper's audit: the honest **strict** protocol next to the leaky **transductive**
convention, each under real and label-shuffled training signs. The label-shuffle audit quantifies the
leakage directly.

| graph | strict / real | strict / shuffle | transd / real | transd / shuffle | **leakage** |
|---|---|---|---|---|---|
| bitcoin-alpha | 0.8818 | **0.5030** | 0.9409 | 0.8628 | **82%** |
| bitcoin-otc | 0.9023 | **0.4806** | 0.9519 | 0.8697 | **82%** |
| slashdot | 0.8922 | **0.4936** | 0.9283 | 0.8224 | **75%** |
| epinions | 0.9330 | **0.5284** | 0.9584 | 0.8877 | **85%** |

(Inner-core L=3 AUROC, median over 5 seeds, `--max-tri 40000`. Leakage = share of the transductive model's
*above-chance* score that survives shuffling: `(transd_shuffle − 0.5) / (transd_real − 0.5)`.) Figure:
`reports/figures/leakage-audit.png`.

## The two columns that make the point

- **strict / shuffle → chance (0.48–0.53).** With features enumerated over training edges only, shuffling the
  training signs leaves nothing to predict test signs from. The strict CPML core is honest: no test-edge
  leakage, and its real-label score (0.88–0.93) is *genuine structural learning* of the Cartwright–Harary
  balance signal.
- **transductive / shuffle → RETAINS 0.82–0.89.** With features enumerated over train+test edges, the *real*
  test-edge signs sit inside the triangle sign-products (and signed-degree tallies). So even when the training
  labels are shuffled, the model reads the answer out of its own features. **75–85% of the transductive
  score is leakage**, not learning.

This is the paper's mechanism, measured on the CPML core across all four graphs. It matches the paper's
headline (a transductive HSiKAN retains 0.997→0.992 = 99.5% leakage); the CPML core is a simpler model, so its
leakage is a slightly lower but still dominant 75–85%. The transductive *real* scores (0.93–0.96) are also
inflated above the strict scores (0.88–0.93) by the same leaky features — the inflation that a naive benchmark
would report as "improvement."

## Method

`examples/cpml_signed_link.rs` gained two audit flags (composable):
- `--shuffle` — permute the **signs of the training edges** (Fisher–Yates, seed-derived); test signs real.
- `--transductive` — enumerate all structural features (signed-degree tallies + triangle sign-products) over
  **train+test** edges (`feat_i = tr_i ∪ te_i`) instead of train-only. This is the leaky convention; without
  it the harness is the strict protocol (default).

Ran `--grid` (inner L=3 + holonomy) for all four `{strict,transductive}×{real,shuffle}` cells, 5 seeds, 4
graphs. The strict rows are reused from `2026-07-13-signed-link-shuffle-audit.md` (the `--transductive` edit
does not change the strict path — `feat_i = tr_i` — so those results are unchanged; verified by the identical
strict numbers).

## Relation to the balance metric

The structure the strict model learns is the **balance metric** (`2026-07-13-signed-balance-metric.md`): these
graphs sit at balanced-triad fraction 0.86–0.89 (strong Cartwright–Harary), and the strict CPML core's honest
0.88–0.93 AUROC is exactly that balance signal being exploited without leakage. The audit closes the loop:
balance is the real signal (measured), the strict protocol learns it (real→high, shuffle→chance), and the
transductive convention *contaminates* it (shuffle→retained).

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `--transductive` leaky-enumeration flag (features over train+test) |
| `scripts/dev/analyze_leakage_audit.py` | new — 2×2 audit table + leakage fraction + 4-bar figure |
| `reports/figures/leakage-audit.{png,json}` | new |

Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` clean; full suite **145/0**. No new deps,
no CORE.YAML.

## Status — the paper's Table 2 is now filled (CPML core)

Both sides are measured across 4 graphs: the honest strict protocol (real high, shuffle chance) and the leaky
transductive convention (real inflated, shuffle retained → 75–85% leakage). Reddit Hyperlinks (the 5th dataset
in the draft) is not in `nagare_data/signed/` — add that edgelist to extend the table to 5 graphs.

## Provenance

- Mac (Apple Silicon), Nagare `357f502`; CPU. Data: `nagare_data/signed/soc-sign-*` (SNAP + Bitcoin).
  Seeds 0–4; `--max-tri 40000`; shuffle seed-derived (test signs real); transductive = features over train+test.
- Reproduce: `bash /tmp/audit/run.sh` (strict) + `bash /tmp/audit/run_transd.sh` (transductive), then
  `python scripts/dev/analyze_leakage_audit.py`. Raw: `/tmp/audit/results{,_transd}.tsv`.
