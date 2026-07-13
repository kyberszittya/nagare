---
title: "Nagare — label-shuffle audit: the strict-protocol CPML core learns signed structure, not leakage (4-graph)"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, leakage, label-shuffle, strict-protocol, structural-learning, nature, balance]
---

# Label-shuffle audit — structural learning vs leakage, under the strict protocol

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `bac2ee7`+ · CPU · 4 graphs × {real, shuffle} × 5 seeds

## Summary

The **structural-learning / generalization test** for the Nature leakage paper: does the CPML inner core, run
under the **strict protocol** (triangle features enumerated over *training edges only*), learn genuine signed
**structure** — the Cartwright–Harary balance signal — or does it ride test-edge **leakage**? The
**label-shuffle audit** decides it: train on **shuffled training-edge signs** (test signs unchanged) and
re-measure AUROC. A leakage-free structural learner must collapse toward **chance (0.5)**; a transductive
(leaky) model would **retain** its score because test-edge signs sit inside its cycle features.

**Result — unanimous across all 4 graphs: STRUCTURAL, no leakage.**

| graph | real AUROC | shuffled-train AUROC | drop | verdict |
|---|---|---|---|---|
| bitcoin-alpha | 0.8818 [.8817,.8852] | **0.5030** [.4798,.5084] | +0.379 | structural |
| bitcoin-otc | 0.9023 [.9016,.9041] | **0.4806** [.4796,.5022] | +0.422 | structural |
| slashdot | 0.8922 [.8907,.8942] | **0.4936** [.4910,.4949] | +0.399 | structural |
| epinions | 0.9330 [.9322,.9342] | **0.5284** [.5258,.5312] | +0.405 | structural |

(Strict-protocol inner-core L=3 AUROC, median/IQR over 5 seeds, `--max-tri 40000`.) Figure:
`reports/figures/shuffle-audit.png`. Every real→shuffle drop is ~0.38–0.42, landing the shuffled model on
the chance line.

## Why this is the structural-learning result the article needs

This is the **double-dissociation**:

- **Real labels → high** (0.88–0.93): the model extracts real signed structure. That structure *is* the
  balance metric measured in `2026-07-13-signed-balance-metric.md` — these graphs sit at balanced-triad
  fraction 0.86–0.89, and the strict CPML core exploits exactly that Cartwright–Harary signal.
- **Shuffled train labels → chance** (0.48–0.53): destroying the training signs destroys the prediction.
  Because the protocol is **strict** (no test-edge sign in any feature), there is no leakage to fall back on
  — so the score collapses. A **transductive/leaky** model would instead retain a high AUROC here (the Nature
  paper's headline: a transductive HSiKAN keeps 0.997→0.992 under shuffle = 99.5% leakage). The strict CPML
  core does the opposite, which is the point.

So the strict protocol both (a) removes the leakage the paper audits and (b) leaves a model that genuinely
**generalizes from signed structure** — the two claims the article makes, now measured on the CPML core
across the full 4-graph set.

## Method

Added a `--shuffle` mode to `examples/cpml_signed_link.rs`: after the deterministic 80/20 edge split, permute
the **signs of the training edges** (Fisher–Yates, seed-derived), leaving test signs untouched. All
downstream strict features — signed-degree tallies, `x0`, the train-triangle enumeration and `tri_signs`, and
the training targets `tr_y` — are then built from the shuffled train graph; `te_y` stays real. Ran the fast
`--grid` path (inner L=3 core + holonomy) for real and shuffle, 5 seeds, 4 graphs.

This confirms the earlier note: the harness is **already the strict protocol** (train-only triangles), so the
2026-07-13 Epinions/Slashdot AUROC benchmark is leakage-free — and now demonstrably so (its scores vanish
under shuffle).

## What remains for the paper's Table 2

This fills the **strict-CPML** rows (real + shuffle). To complete the audit's contrast, the **transductive
(leaky) arm** is still needed: enumerate triangles over train+test edges and show the leaky model *retains*
its score under shuffle (quantifying the leakage). That is a separate harness mode (transductive enumeration)
and the natural next measurement; the strict side — the honest side — is done here.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | `--shuffle` label-shuffle audit mode (permute train-edge signs; test unchanged) |
| `scripts/dev/analyze_shuffle_audit.py` | new — audit table + grouped-bar figure |
| `reports/figures/shuffle-audit.{png,json}` | new |

Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` clean; full suite **145/0**. No new deps,
no CORE.YAML.

## Provenance

- Mac (Apple Silicon), Nagare `bac2ee7`; CPU. Data: `nagare_data/signed/soc-sign-*` (SNAP + Bitcoin).
  Seeds 0–4; `--max-tri 40000`; strict protocol (train-only triangles); shuffle seed-derived, test signs real.
- Reproduce: `bash /tmp/audit/run.sh` then `python scripts/dev/analyze_shuffle_audit.py`. Raw:
  `/tmp/audit/results.tsv`.
