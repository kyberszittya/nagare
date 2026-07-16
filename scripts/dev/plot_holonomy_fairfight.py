#!/usr/bin/env python3
"""Competitiveness fair fight: holonomy net vs MLP vs trivial raw-entropy baseline."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/holonomy_dissociation_results.json"))
arms = d["competitiveness_fair_fight_5seed_median"]["arms"]
arms = sorted(arms, key=lambda a: -a["auroc"])
names = [a["name"] for a in arms]
vals = [a["auroc"] for a in arms]
colors = ["#c0392b" if "trivial: raw entropy" in n else ("#2a7de1" if "holonomy" in n else "#b0b0b0") for n in names]
fig, ax = plt.subplots(figsize=(9, 4.6))
bars = ax.barh(range(len(names)), vals, color=colors)
ax.set_yticks(range(len(names)))
ax.set_yticklabels([f"{n}\n(~{a['params']} params)" for n, a in zip(names, arms)], fontsize=8)
ax.invert_yaxis()
ax.axvline(0.5, color="#888", ls="--", lw=1, label="chance")
for i, v in enumerate(vals):
    ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=10, fontweight="bold" if i == 0 else "normal")
ax.set_xlim(0.5, 1.0); ax.set_xlabel("held-out AUROC (median of 5 seeds, same data)")
ax.set_title("Competitiveness — HONEST NEGATIVE: on this task the trivial raw-entropy baseline (2 params)\nBEATS the deep holonomy net (0.944 vs 0.894). The machinery does not earn its complexity here.")
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
out = root / "reports/figures/holonomy-fairfight.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
