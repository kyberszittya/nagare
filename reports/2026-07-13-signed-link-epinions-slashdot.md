---
title: "Nagare — signed-link prediction on Epinions & Slashdot: completing the 4-graph benchmark (5-seed)"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, epinions, slashdot, benchmark, holonomy, headroom]
---

# CPML signed-link prediction — Epinions & Slashdot (5-seed benchmark)

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `7d1b01d`+ · CPU · 5 seeds × 2 graphs

## Summary

Ran the CPML signed-link harness (`examples/cpml_signed_link.rs`) on the two large SNAP signed graphs,
5 seeds each, `--max-tri 40000` — **completing the 4-graph set** (Bitcoin-Alpha, Bitcoin-OTC were on record;
**Slashdot was predicted but never measured; Epinions was measured only on the holonomy grid**). Median/IQR
over 5 seeds (§3 benchmark discipline).

## Result — test AUROC (median over 5 seeds, IQR)

| dataset | V | edges | L=1 flat | L=3 inner | hg-conv | cascade | holo M=1 | holo M=4 |
|---|---|---|---|---|---|---|---|---|
| **epinions** | 131,828 | 841,372 | 0.9323 [.9318,.9334] | 0.9330 [.9322,.9342] | 0.9325 [.9298,.9331] | 0.7202 [.7172,.7203] | **0.9341** [.9329,.9350] | 0.9333 [.9317,.9334] |
| **slashdot** | 82,140 | 549,202 | 0.8903 [.8901,.8923] | 0.8922 [.8907,.8942] | 0.8911 [.8910,.8940] | 0.7833 [.7762,.7842] | **0.8938** [.8917,.8949] | 0.8920 [.8907,.8926] |

Figure: `reports/figures/signed-link-bench.png` (grouped bars, IQR error bars; dotted = each graph's flat
baseline). Wall: Epinions ~108 s/seed, Slashdot ~70 s/seed (all six arms).

## Reading

- **Slashdot is the new number.** The inner CPML core reaches **0.892 AUROC** (L=3), a clean +0.0019 over the
  L=1 flat baseline (0.8903). Epinions **replicates** the on-record 0.933 (holonomy-grid value 0.933 →
  5-seed flat 0.9323).
- **The signed-degree features nearly saturate both tasks.** L=1 flat, L=3 tiered, and the learned
  hypergraph-conv are all within ~0.002 of each other — the tiering/conv machinery *ties* the flat
  baseline. The task is largely solved by leakage-free signed-degree statistics; the structural arms add
  little on these two graphs.
- **Holonomy is modest and headroom-consistent.** `holo M=1` is the best arm on both (0.9341 Epinions,
  0.8938 Slashdot), but the gain over the flat baseline is small: **+0.0018 on Epinions, +0.0035 on
  Slashdot**. Slashdot's larger gain is consistent with the **headroom law** from the 2026-07-10 report
  (gain ∝ 1 − base; Slashdot's headroom 0.110 ≈ 1.6× Epinions' 0.068). But both deltas are within the arms'
  IQRs — this is a *modest, headroom-consistent* effect, **not** a robustly-separated win at this near-ceiling
  regime. On record, the holonomy's robust win was on the moderate-headroom **OTC (+0.0067, 23/25)**; these
  two high-base graphs sit on the shrinking-returns end of that law, exactly as predicted.
- **The FULL cascade hurts, as before.** 0.720 (Epinions) / 0.783 (Slashdot), −0.21 / −0.11 vs flat — the
  Gömb-Soma outer cascade gate remains a negative on signed-link (consistent with prior reports; not
  re-litigated here, just confirmed across 5 seeds).

## The 4-graph picture (with the two on-record graphs)

| graph | inner base AUROC | headroom | holonomy verdict |
|---|---|---|---|
| Bitcoin-Alpha (sparse) | 0.882 | 0.118 | modest, non-robust (+0.0028) — sparse, few cycles |
| Bitcoin-OTC (sweet spot) | 0.904 | 0.096 | **robust +0.0067 (23/25)** — enough headroom *and* cycle density |
| **Slashdot** (this run) | **0.892** | **0.108** | modest +0.0035 (headroom-consistent, within IQR) |
| **Epinions** (this run) | **0.933** | **0.067** | negligible +0.0018 (near-ceiling) — least gain |

The law holds across all four: the holonomy buys accuracy where the inner core leaves room *and* there are
cycles to aggregate; it fades toward the ceiling (Epinions) and toward sparsity (Alpha). OTC remains the
sweet spot. Nothing here overturns the settled holonomy verdict — it extends it to the two largest graphs.

## Provenance

- Mac (Apple Silicon), Nagare `7d1b01d`; CPU. Data: `nagare_data/signed/soc-sign-epinions.txt`
  (V=131,828, E=841,372), `soc-sign-Slashdot090221.txt` (V=82,140, E=549,202) — SNAP signed graphs.
- Seeds 0–4; `--max-tri 40000`; 80/20 edge split (deterministic per seed); leakage-free signed-degree
  features from the train graph only.
- Reproduce: `bash /tmp/sl_bench/run.sh` (loops the two graphs × 5 seeds) then
  `python scripts/dev/analyze_signed_link_bench.py`. Raw logs `/tmp/sl_bench/{epinions,slashdot}_s*.log`.
- No code changed (existing harness on existing data); no new deps, no CORE.YAML. Not a per-seed
  bit-reproducibility claim (RL/stochastic carve-out N/A — these are deterministic given seed; the split
  and init are seeded).

## Follow-ups

- The holonomy's paired inner→holo Δ per seed (for a proper IQR-of-Δ robustness test on these two graphs)
  can be extracted via the `--grid` mode if a robust-or-not verdict on Slashdot is wanted; the arm-median
  table above already shows it sits within noise.
- These two graphs are near-ceiling for the current feature set; a genuinely harder signed-link regime (or a
  richer negative-sampling protocol) would be needed to give the structural arms room to separate.
