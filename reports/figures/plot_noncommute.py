#!/usr/bin/env python3
"""Non-commutativity figure (§9): only the non-abelian rotor_holonomy op captures the signal."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "noncommute.json").read_text())
arms = d["arms"]
names = [a["arm"] for a in arms]
vals = [a["median_auroc"] for a in arms]
colors = []
for n in names:
    if "commutator" in n:
        colors.append("#1b9e77")
    elif "holonomies" in n:
        colors.append("#3b7dd8")
    else:
        colors.append("#999999")

fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.axhline(0.60, ls=":", c="crimson", lw=1.0, label="chance band")
ax.set_ylim(0.45, 1.02)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title(
    "Non-commutativity: every generic/abelian method on the raw edges is at chance;\n"
    "only the non-abelian rotor_holonomy commutator solves it — HOLONOMY-SPECIFIC"
)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=10)
ax.legend(loc="center left", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=14, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "noncommute.png")
print("wrote noncommute.png")
