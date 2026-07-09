# Nagare — justifying the CPML inner core on real heavy-tailed signed graphs

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

The 2c ablation tied on a toy 12-vertex uniform-degree graph — leaving open whether the CPML
inner core's **degree-tier stratification** ever earns its weight. This runs it on the regime
it was designed for: **real, heavy-tailed-degree** signed graphs (Bitcoin Alpha/OTC, Slashdot).

**Finding: it earns its weight.** L=3 tiered inner beats L=1 flat on **13/13 runs** (5 seeds ×
Alpha, 5 × OTC, 3 × Slashdot) — **never once negative** — with median ΔAUROC +0.013 / +0.006 /
+0.0025. The gain is modest and shrinks as the graph grows (flat catches up with more data), but
it is directionally reliable. This cleanly explains the 2c tie: **stratification only has value
when degrees are heterogeneous** — the toy graph had none.

## Method

`examples/cpml_signed_link.rs` — leakage-free signed link prediction, train edges only:
per-vertex signed-degree features `x₀` → enumerate train triangles → **CPML tier core** (real
degrees → `TierSpec.uniform(L)` tiers; per-tier restricted-triangle aggregation
`gather→linear→mean→scatter`; `concat(x₀, H₀…H_{L-1})` node embedding) → edge head scores
`sign(u,v)` from `[emb[u], emb[v]]`. Closed-form Adam, 250 steps, test AUROC (80/20 edge split).

Ablation in one run: **L=3 tiered** vs **L=1 flat** inner — same features, same triangles, same
edges; `tier_of` (hence which triangles route to which aggregator) is the **only** difference.

## Results (test AUROC)

| graph | V / edges / tri | L=3 median | L=1 median | median Δ | L=3 wins |
|---|---|---|---|---|---|
| Bitcoin Alpha | 3.8k / 24k / 17k | **0.8818** | 0.8700 | **+0.0132** | 5/5 |
| Bitcoin OTC | 5.9k / 36k / 25k | **0.9023** | 0.8971 | **+0.0057** | 5/5 |
| Slashdot | 82k / 549k / 60k* | **0.8923** | 0.8901 | **+0.0025** | 3/3 |

*triangle-capped at 60k. Every per-seed Δ is positive (13/13); plot:
`reports/figures/cpml-justify-signed-link.png`.

## Reading (measured / inferred)

- **Measured:** on three real heavy-tailed signed graphs, degree-tier stratification of the
  cycle aggregation improves signed-link AUROC on every seed — never hurts.
- **Inferred (mechanism):** heavy-tailed degrees mean hub-neighbourhoods and leaf-neighbourhoods
  carry different signal; routing each triangle to a tier-specific aggregator (then concatenating)
  lets the head weight them separately — a useful prior the flat single-aggregator can't express.
  On the toy uniform-degree graph (2c) there is nothing to stratify, so the tiers were only
  extra params → tie. **The two results agree**: the core's value is the degree heterogeneity.
- **Effect size, honestly:** the gain is small (~0.003–0.013 median AUROC) and **decreases with
  graph size** — Alpha (small, few train edges) gains most, Slashdot (large) least. This is the
  expected small-data-prior behaviour: the tier structure helps most when data is scarce and the
  flat model can't infer the hub/leaf split itself. It is a *consistent* edge, not a headline
  jump. Whether it survives against a strong signed-GNN baseline is untested (out of scope: this
  is a within-model L=3-vs-L=1 ablation, isolating the tier contribution).

## Relation to the session

Closes the honest-verdict loop: KB (tie, dominated by grid refinement), graph-vs-KAN (tie,
saturated), 2c inner on toy (tie, uniform degrees) — and now the **first robust positive**, the
CPML core on the heavy-tailed regime it targets. The negatives weren't "nothing works"; they
were the wrong regime for the mechanism. This is the right one.

## Files touched

| file | change |
|---|---|
| `examples/cpml_signed_link.rs` | **new** — CPML tier core signed-link predictor + L=3/L=1 ablation on real data |
| `scripts/dev/plot_cpml_justify.py`, `reports/figures/cpml-justify-signed-link.png` | **new** — plot |

## CORE / deps

**None.** Reuses `cpml_tier` + `linear`/`scatter_mean`/`adam_step`; no dependency change.

## Test results / provenance

- Example builds clean (`clippy -D warnings`, fmt). Full suite unchanged at **92/0**.
- Data (repo-external): `nagare_data/signed/soc-sign-{bitcoinalpha,bitcoinotc}.csv`,
  `soc-sign-Slashdot090221.txt` (SNAP). Measured on the Apple-Silicon Mac; Slashdot run ~12 s.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <file> --seed <s>`.
- Repo `github.com/kyberszittya/nagare`. Rust 1.96.1. Leakage-free: features + triangles from
  train edges only.
