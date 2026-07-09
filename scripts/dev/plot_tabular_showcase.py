#!/usr/bin/env python
"""Showcase figure for Nagare on tabular (T1/T2/T4).

Per-seed results are deterministic (seeded) from the committed tests:
  - tests/kan_iris.rs          (KAN on Iris)
  - tests/graph_vs_kan_iris.rs (KAN vs graph-from-tabular on Iris — the T4 verdict)
  - tests/kan_california.rs     (KAN regression on California)
Renders reports/figures/nagare-tabular-showcase.png. Run (no permanent install):
  uv run --with matplotlib --with numpy python scripts/dev/plot_tabular_showcase.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Iris held-out accuracy per seed (0..4).
iris_kan = np.array([0.868, 0.974, 0.921, 0.947, 0.974])
iris_graph = np.array([0.842, 0.974, 0.868, 0.947, 0.947])
# California held-out R² per seed (0..4), KAN regressor.
cal_r2 = np.array([0.703, 0.724, 0.704, 0.681, 0.795])

fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.5, 4.6))

# ── Panel A: T4 — KAN vs graph on Iris (paired) ──────────────────────
for e, g in zip(iris_kan, iris_graph):
    axa.plot([1, 2], [e, g], color="0.78", lw=0.9, zorder=1)
axa.scatter([1] * 5, iris_kan, color="#2aa76a", s=34, zorder=3,
            label=f"KAN (median {np.median(iris_kan):.3f})")
axa.scatter([2] * 5, iris_graph, color="#c85a33", s=34, zorder=3,
            label=f"graph (median {np.median(iris_graph):.3f})")
axa.plot([0.82, 1.18], [np.median(iris_kan)] * 2, color="#177a4a", lw=2.6)
axa.plot([1.82, 2.18], [np.median(iris_graph)] * 2, color="#9c3f1f", lw=2.6)
axa.set_xlim(0.6, 2.4)
axa.set_xticks([1, 2])
axa.set_xticklabels(["plain KAN", "graph-from-tabular"])
axa.set_ylabel("held-out accuracy")
axa.set_title("T4 — Iris: does the signed graph beat the KAN?\n"
              "verdict: TIE at 0.947 (graph runs, doesn't help)")
axa.legend(fontsize=8, loc="lower center")

# ── Panel B: T2 — California R² ──────────────────────────────────────
axb.scatter(np.arange(5), cal_r2, color="#3a6ea5", s=40, zorder=3)
axb.axhline(np.median(cal_r2), color="#274d73", lw=2.2,
            label=f"median R² = {np.median(cal_r2):.3f}")
axb.axhline(0.606, color="0.6", ls="--", lw=1.2, label="sklearn linear reg ≈ 0.61")
axb.set_xticks(np.arange(5))
axb.set_xlabel("seed")
axb.set_ylabel("held-out R²")
axb.set_ylim(0.4, 0.85)
axb.set_title("T2 — California housing regression\nclosed-form additive KAN")
axb.legend(fontsize=8, loc="lower right")

fig.suptitle("Nagare on standard tabular benchmarks (closed-form, no autograd)", fontsize=12)
fig.tight_layout(rect=(0, 0, 1, 0.96))
out = "reports/figures/nagare-tabular-showcase.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
