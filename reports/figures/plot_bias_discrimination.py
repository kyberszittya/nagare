#!/usr/bin/env python3
"""Bias-discrimination figure (§9): regional 2nd-order pooling closes the gap; not holonomy-specific."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "bias_discrimination.json").read_text())
arms = d["arms"]
names = [a["arm"] for a in arms]
vals = [a["median_auroc"] for a in arms]
colors = {"generic-MLP": "#d95f02", "block-Laplacian": "#66a61e",
          "block-entropy": "#1b9e77", "oracle |rA-rB|": "#3b7dd8"}
bar_colors = [colors.get(n, "#999999") for n in names]

fig, ax = plt.subplots(figsize=(8.5, 5), dpi=130)
bars = ax.bar(names, vals, color=bar_colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.set_ylim(0.5, 1.0)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title(
    "Bias discrimination: regional 2nd-order pooling CLOSES the generic-MLP→oracle gap\n"
    "(0.66→0.93) — but a generic Laplacian beats the framework entropy op (NOT holonomy-specific)"
)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.006, f"{v:.3f}", ha="center", fontsize=10)
ax.legend(loc="lower right", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9.5)
fig.tight_layout()
fig.savefig(HERE / "bias_discrimination.png")
print("wrote bias_discrimination.png")
