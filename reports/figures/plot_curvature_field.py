#!/usr/bin/env python3
"""Plot the nonlinear-curvature (B+C) dissociation (§9 graphical output).

  1. per-arm bars: trivial + constant-rotor BOTH blind; ChebyCR reaches oracle.
  2. contrast sweep: separability vs generating order k_gen.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).parent
d = json.loads((HERE / "curvature_field_dissociation.json").read_text())

# ---- fig 1: per-arm bars ----
arms = d["main"]
names = [a["arm"] for a in arms]
vals = [a["median_auroc"] for a in arms]
colors = []
for n in names:
    if n == "ChebyCR":
        colors.append("#1b9e77")
    elif n == "oracle":
        colors.append("#3b7dd8")
    elif n == "Laplacian":
        colors.append("#66a61e")
    elif n in ("trivial-entropy", "constant-rotor"):
        colors.append("#999999")
    else:
        colors.append("#d95f02")

fig, ax = plt.subplots(figsize=(9, 4.7), dpi=130)
bars = ax.bar(names, vals, color=colors, edgecolor="black", linewidth=0.6)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.axhline(0.60, ls=":", c="crimson", lw=1.0, label="gate (blind arms ≤ 0.60)")
ax.set_ylim(0.4, 1.03)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title(
    "Nonlinear curvature (B+C): trivial entropy AND the constant-rotor exact solve\n"
    "are BOTH blind; the closed-form ChebyCR readout reaches the oracle (no training)"
)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9)
ax.legend(loc="center right", fontsize=8)
plt.setp(ax.get_xticklabels(), rotation=15, ha="right", fontsize=9)
fig.tight_layout()
fig.savefig(HERE / "curvature_field_arms.png")
print("wrote curvature_field_arms.png")

# ---- fig 2: contrast sweep ----
sw = d["contrast_sweep"]
kg = [r["k_gen"] for r in sw]
fig2, ax2 = plt.subplots(figsize=(7.5, 4.7), dpi=130)
ax2.plot(kg, [r["chebycr"] for r in sw], "o-", c="#1b9e77", lw=2, label="ChebyCR (one-shot)")
ax2.plot(kg, [r["oracle"] for r in sw], "s--", c="#3b7dd8", lw=1.2, label="oracle", alpha=0.7)
ax2.plot(kg, [r["constant_rotor"] for r in sw], "^-", c="#999999", lw=1.5, label="constant-rotor (blind)")
ax2.plot(kg, [r["trivial"] for r in sw], "v-", c="#bbbbbb", lw=1.5, label="trivial entropy (blind)")
ax2.axhline(0.5, ls="--", c="k", lw=0.8)
ax2.set_ylim(0.45, 1.03)
ax2.set_xlabel("generating field order  k_gen  (fixed fit order k_fit=3)")
ax2.set_ylabel("held-out separability AUROC (median)")
ax2.set_title("Robustness: ChebyCR holds; a permutation destroys all spatial structure")
ax2.legend(fontsize=9)
ax2.grid(alpha=0.25)
fig2.tight_layout()
fig2.savefig(HERE / "curvature_field_contrast_sweep.png")
print("wrote curvature_field_contrast_sweep.png")
