#!/usr/bin/env python3
"""Optimizer comparison: deep+entropy AUROC vs #passes — GD vs Adam vs Nesterov."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/holonomy_dissociation_results.json"))
o = d["optimizers_deep_entropy_seed0"]
p = o["passes"]
fig, ax = plt.subplots(figsize=(7.8, 4.7))
ax.plot(p, o["adam_lr0p05"], "o-", color="#2a7de1", lw=2.4, label="Adam lr=0.05 (10 passes → 0.87; 0.947)")
ax.plot(p, o["adam_lr0p02"], "o--", color="#7aa8d8", lw=1.6, label="Adam lr=0.02")
ax.plot(p, o["nesterov_lr1p0"], "s-", color="#e08a2a", lw=2.0, label="Nesterov lr=1.0 (0.865)")
ax.plot(p, o["gd_lr2p0"], "^--", color="#999", lw=2.0, label="GD lr=2.0 (100 passes → 0.89; 0.902)")
ax.axhline(0.87, color="#2a9d5a", ls=":", lw=1)
ax.annotate("Adam hits 0.87 in 10 passes;\nGD needs ~100", (10, 0.878), fontsize=8, color="#2a7d4f")
ax.set_xscale("log"); ax.set_xticks(p); ax.set_xticklabels(p)
ax.set_xlabel("training passes (epochs)"); ax.set_ylabel("held-out AUROC (deep+entropy, seed 0)")
ax.set_title("Adaptive momentum (Adam) accelerates the rotor learning ~10× and raises the ceiling\n(step 1: plain Euclidean Adam over the Cayley chart — Clifford+entropy geometry next)")
ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out = root / "reports/figures/holonomy-optimizers.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
