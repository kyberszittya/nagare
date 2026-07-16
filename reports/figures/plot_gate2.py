#!/usr/bin/env python3
"""Gate-2 figure (§9): every fixed global scalar + generic MLP fail; only the fixed regional oracle solves it."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "gate2_xor.json").read_text())
arms = d["arms"]
names = [a["arm"] for a in arms]
vals = [a["median_auroc"] for a in arms]
colors = []
for n in names:
    if "oracle" in n:
        colors.append("#3b7dd8")
    elif "learned" in n:
        colors.append("#d95f02")
    else:
        colors.append("#999999")

fig, ax = plt.subplots(figsize=(9.5, 5), dpi=130)
bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.axhline(0.60, ls=":", c="crimson", lw=1.0, label="gate band")
ax.set_ylim(0.4, 1.0)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title(
    "Gate 2 (XOR-of-regional-roughness): every fixed GLOBAL scalar AND a generic MLP fail;\n"
    "only the fixed REGIONAL closed-form (oracle) solves it → bias-discriminating, not learning-necessary"
)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
ax.legend(loc="upper left", fontsize=8.5)
plt.setp(ax.get_xticklabels(), rotation=18, ha="right", fontsize=8.5)
fig.tight_layout()
fig.savefig(HERE / "gate2_xor.png")
print("wrote gate2_xor.png")
