#!/usr/bin/env python3
"""Plot the auto-holonomy dissociation results (§9 graphical output).

Two figures from `auto_holonomy_dissociation.json`:
  1. grouped bar of per-arm median separability AUROC (the Step-1 gate + Step-2 A/B).
  2. flux sweep: separability vs theta_min for trivial / MLP / closed-Clifford / oracle.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
data = json.loads((HERE / "auto_holonomy_dissociation.json").read_text())

# ---- fig 1: per-arm bars ----
arms = data["main"]
names = [a["arm"] for a in arms]
vals = [a["median_auroc"] for a in arms]
# color: closed-form holonomy arms green, learned/trivial gray, oracle blue
colors = []
for n in names:
    if "closed" in n:
        colors.append("#1b9e77")
    elif n == "oracle":
        colors.append("#3b7dd8")
    elif "trivial" in n:
        colors.append("#999999")
    else:
        colors.append("#d95f02")

fig, ax = plt.subplots(figsize=(8.5, 4.6), dpi=130)
bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.axhline(0.60, ls=":", c="crimson", lw=1.0, label="Step-1 gate (trivial ≤ 0.60)")
ax.set_ylim(0.4, 1.03)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title(
    "Auto-holonomy curvature task: trivial entropy is at chance,\n"
    "closed-Clifford one-shot reaches the oracle (no training)"
)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.legend(loc="center right", fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "auto_holonomy_arms.png")
print("wrote auto_holonomy_arms.png")

# ---- fig 2: flux sweep ----
sw = data["flux_sweep"]
th = [r["theta_min"] for r in sw]
fig2, ax2 = plt.subplots(figsize=(7.5, 4.6), dpi=130)
ax2.plot(th, [r["closed_clifford"] for r in sw], "o-", c="#1b9e77", lw=2, label="closed-Clifford (one-shot)")
ax2.plot(th, [r["oracle"] for r in sw], "s--", c="#3b7dd8", lw=1.2, label="oracle", alpha=0.7)
ax2.plot(th, [r["mlp"] for r in sw], "^-", c="#d95f02", lw=1.5, label="MLP h16 (trained)")
ax2.plot(th, [r["trivial_entropy"] for r in sw], "v-", c="#999999", lw=1.5, label="trivial entropy")
ax2.axhline(0.5, ls="--", c="k", lw=0.8)
ax2.set_ylim(0.45, 1.03)
ax2.set_xlabel("flux magnitude floor  θ_min  (rad)")
ax2.set_ylabel("held-out separability AUROC (median)")
ax2.set_title("Robustness to flux magnitude: closed-form holds at 1.0 even for weak flux")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(HERE / "auto_holonomy_flux_sweep.png")
print("wrote auto_holonomy_flux_sweep.png")

if len(sys.argv) > 1 and sys.argv[1] == "--show-sizes":
    for p in ("auto_holonomy_arms.png", "auto_holonomy_flux_sweep.png"):
        print(p, (HERE / p).stat().st_size, "bytes")
