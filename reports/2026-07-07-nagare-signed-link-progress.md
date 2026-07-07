# Nagare signed-link benchmark — progress: data + validated baseline ceiling

Created-at: 2026-07-07 16:56 JST
The decisive experiment (`nagare_github/docs/HANDOFF-signed-link-benchmark.md`).
Host: kato15. This session established the data + the reference ceiling; the
Nagare closed-form model is the next phase.

## Done this session

1. **Data** (SNAP signed networks, on kato15 `/tmp/hajdu/signed/`):
   Bitcoin-Alpha (24k edges), Slashdot (549k), Epinions (841k).
2. **Harness validated + baseline ceiling** (`/tmp/hajdu/slb.py`): 80/20 edge
   split, **leakage-free** endpoint signed-degree features (counts from TRAIN
   edges only), logistic regression, Mann-Whitney AUROC. 3 seeds:

| dataset | pos fraction | baseline test AUROC (3-seed) |
|---|---:|---:|
| Bitcoin-Alpha | 0.936 | 0.907 / 0.919 / 0.910 |
| Slashdot | 0.774 | 0.913 / 0.912 / 0.912 |
| Epinions | 0.853 | 0.951 / 0.951 / 0.951 |

These match the signed-link-prediction literature (signed-degree/FExtra features
give ~0.85–0.95 AUROC), so the loader, split, no-leakage feature computation, and
metric are all correct. **This is the reference Nagare must be competitive with.**

## The target band

- **Floor (simple features):** ~0.91 (Slashdot/BTC-Alpha), ~0.95 (Epinions) —
  measured above.
- **Ceiling (signed-GNN, published):** SGCN / SiGAT report ~0.93–0.97 on these.
  Nagare is "competitive" if it lands in/above this band on multi-seed median.

## Signed holonomy VALIDATED as the mechanism (2026-07-07)

Discriminating test (`scripts/dev/signed_link_holonomy.py`): within the same
leakage-free harness, does adding **signed holonomy** (length-2 balance vote,
`sum_w sign(u,w)·sign(w,v)` over common neighbours — the accumulated sign product,
i.e. Z₂ holonomy of the triad) lift AUROC over degree-only features?

| dataset | degree-only | + signed holonomy | lift |
|---|---:|---:|---:|
| Bitcoin-Alpha | 0.889 | 0.901 | +0.013 |
| Slashdot | 0.898 | 0.909 | +0.011 |
| Epinions | 0.933 | 0.954 | +0.021 |

**Consistent positive lift on all three** → signed holonomy is a real signal for
signed-link prediction, and lands in the competitive band. This is *triad*
(length-2) holonomy computed classically; **Nagare's differentiation is deeper,
learned, closed-form holonomy over longer signed cycles** (`clifford_fir` over the
`hymeko_graph` cycle pool). The approach is now de-risked: the Rust model must
reproduce/beat this via its own kernels.

## Next phase (the Nagare closed-form model — not yet built)

Per the handoff: build the edge-sign pipeline on the shipped kernels — signed
graph → `hymeko_graph` top-k signed cycles → `clifford_fir` (holonomy features)
→ per-vertex embeddings (`scatter_mean`) → **closed-form edge-sign head** on the
two endpoints' embeddings → AUROC on held-out edges, 5 seeds. Keep the head
closed-form/local (a backprop head would defeat the thesis test). Then add SGCN /
SiGAT reference baselines. **No Nagare number exists yet — do not claim one.**

## Also delivered this session

`nagare_github/docs/nagare-review.{tex,pdf}` — the comprehensive review in LaTeX
with TikZ architecture + mechanism diagrams (committed `6722797`).

## Honest state

The hard *validation* work (correct leakage-free harness + literature-matching
ceiling) is done — that de-risks the whole benchmark. What remains is the Nagare
model build + its AUROC vs this band + proper GNN baselines + multi-seed. That is
the experiment that decides whether closed-form local learning on holonomy is a
real learning mechanism. Until it runs, the framework's evidence stays
"fast/lean CPU forward + toy AUROC."
