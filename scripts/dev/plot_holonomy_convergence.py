#!/usr/bin/env python3
"""Deep+entropy held-out AUROC vs #passes — the 'how instantaneous' curve."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/holonomy_dissociation_results.json"))
c = d["convergence_deep_entropy_seed0"]
p = c["passes"]
fig, ax = plt.subplots(figsize=(7.4, 4.6))
ax.plot(p, c["lr_2p0"], "o-", color="#2a7de1", lw=2.2, label="lr=2.0 (trained → 0.90)")
ax.plot(p, c["lr_0p5"], "s-", color="#7aa8d8", lw=1.8, label="lr=0.5")
ax.plot(p, c["lr_0p05"], "^--", color="#999", lw=1.8, label="lr=0.05 (the committed 2×2 setting)")
ax.axhline(0.728, color="#2a9d5a", ls=":", lw=1.5)
ax.annotate("ONE pass = 0.728 (near-instantaneous:\narchitecture + linear probe over random deep rotors)",
            (1.2, 0.735), fontsize=8, color="#2a7d4f")
ax.axhline(0.561, color="#c0392b", ls=":", lw=1)
ax.annotate("shallow+entropy 0.561 (depth needed even at 1 pass)", (5, 0.567), fontsize=8, color="#c0392b")
ax.set_xscale("log"); ax.set_xticks(p); ax.set_xticklabels(p)
ax.set_xlabel("training passes (epochs)"); ax.set_ylabel("held-out AUROC (deep+entropy, seed 0)")
ax.set_title("How instantaneous? Deep-holonomy+entropy is discriminative in ONE pass (0.728);\nrotor-weight learning then refines it to 0.90 (iterative GD — the pure one-shot rule still open)")
ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.3, which="both")
fig.tight_layout()
out = root / "reports/figures/holonomy-convergence.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
