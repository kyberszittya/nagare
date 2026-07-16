#!/usr/bin/env python3
"""Capstone figure (§9): learning ON the fixed holonomy op works; training THROUGH it corrupts."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
HERE = Path(__file__).parent
d = json.loads((HERE / "noncommute_learned.json").read_text())
names = [a["arm"] for a in d["arms"]]
vals = [a["median_auroc"] for a in d["arms"]]
colors = ["#999999", "#d95f02", "#1b9e77", "#3b7dd8"]
fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.set_ylim(0.45, 1.02)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title("Capstone — the designed holonomy op must stay FIXED:\n"
             "learning a readout ON rotor_holonomy works (0.968); training THROUGH it corrupts (0.513)")
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, v+0.008, f"{v:.3f}", ha="center", fontsize=10)
ax.legend(loc="center left", fontsize=9)
plt.setp(ax.get_xticklabels(), rotation=14, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "noncommute_learned.png")
print("wrote noncommute_learned.png")
