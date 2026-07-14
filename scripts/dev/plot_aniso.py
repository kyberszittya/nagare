#!/usr/bin/env python3
"""Plot the N2b aniso A/B: C_8 vs C_1 vs energy baseline (bar+noise detection)."""
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

c8 = json.load(open(sys.argv[1]))
c1 = json.load(open(sys.argv[2]))
out = sys.argv[3]

fig, ax = plt.subplots(figsize=(7.2, 4.6))
groups = ["train\norientation", "held-out\norbit"]
x = range(len(groups))
w = 0.28
ax.bar([i - w for i in x], [c8["train_auc"], c8["test_auc"]], w, label="C₈ (rotation-invariant)", color="#2a9d8f")
ax.bar([i for i in x], [c1["train_auc"], c1["test_auc"]], w, label="C₁ (orientation-specific)", color="#e76f51")
ax.bar([i + w for i in x], [c8["energy_auc"], c8["energy_auc"]], w, label="energy-only baseline", color="#adb5bd")
ax.axhline(0.5, ls="--", lw=1, color="k", alpha=0.5)
ax.text(1.42, 0.505, "chance", fontsize=8, color="k", alpha=0.6)
ax.set_xticks(list(x))
ax.set_xticklabels(groups)
ax.set_ylabel("AUROC  (oriented bar vs energy-matched noise)")
ax.set_ylim(0.4, 0.85)
ax.set_title("N2b — task forces oriented features (energy ≈ chance);\nlearnable S-cell makes explicit C-cell redundant (C₈ ≈ C₁)")
ax.legend(loc="upper left", fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
