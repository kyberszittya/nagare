#!/usr/bin/env python3
"""Deep-holonomy double dissociation: 2x2 held-out AUROC (depth x readout)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/holonomy_dissociation_results.json"))
a = d["auroc_median_5seed"]
grid = np.array([[a["deep_entropy"], a["deep_mean"]],
                 [a["shallow_entropy"], a["shallow_mean"]]])
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.15, 1]})

# heatmap
im = ax1.imshow(grid, cmap="RdYlGn", vmin=0.5, vmax=0.8, aspect="auto")
ax1.set_xticks([0, 1]); ax1.set_xticklabels(["entropy readout\n(arrangement)", "mean readout\n(arrangement-blind)"])
ax1.set_yticks([0, 1]); ax1.set_yticklabels(["deep (L=3)", "shallow (L=1)"])
for i in range(2):
    for j in range(2):
        ax1.text(j, i, f"{grid[i,j]:.3f}", ha="center", va="center", fontsize=15,
                 color="black", fontweight="bold" if (i==0 and j==0) else "normal")
ax1.set_title("2×2 held-out AUROC (median of 5 seeds)")
fig.colorbar(im, ax=ax1, fraction=0.046, label="AUROC")

# grouped bars w/ chance line
labels = ["deep\n+entropy", "deep\n+mean", "shallow\n+entropy", "shallow\n+mean"]
vals = [a["deep_entropy"], a["deep_mean"], a["shallow_entropy"], a["shallow_mean"]]
colors = ["#2a7de1", "#c0c0c0", "#c0c0c0", "#c0c0c0"]
ax2.bar(range(4), vals, color=colors)
ax2.axhline(0.5, color="#c0392b", ls="--", lw=1, label="chance (0.5)")
for i, v in enumerate(vals):
    ax2.text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
ax2.set_xticks(range(4)); ax2.set_xticklabels(labels, fontsize=8)
ax2.set_ylim(0.5, 0.8); ax2.set_ylabel("held-out AUROC (median/5)")
ax2.set_title("Remove depth OR entropy → chance")
ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

fig.suptitle("Deep holonomy net learns useful features THROUGH DEPTH — closed-form (FD-verified), no autograd\nDouble dissociation: depth AND the entropy (arrangement) readout both load-bearing",
             fontsize=11, y=1.04)
fig.tight_layout()
out = root / "reports/figures/holonomy-dissociation.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
