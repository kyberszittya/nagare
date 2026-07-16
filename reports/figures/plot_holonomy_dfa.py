#!/usr/bin/env python3
"""Gate-1 holonomy-DFA diagnostic figure (§9): rule AUROC (L3 vs L1) + gradient alignment."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
d = json.loads((HERE / "holonomy_dfa_gate1.json").read_text())
rules = d["rules"]
names = [r["rule"] for r in rules]
l3 = [r["auroc_l3"] for r in rules]
l1 = [r["auroc_l1"] for r in rules]
align = [r["align_l3"] for r in rules]
triv = d["trivial_floor"]

x = np.arange(len(names))
fig, ax = plt.subplots(figsize=(9, 5), dpi=130)
b3 = ax.bar(x - 0.19, l3, 0.36, label="depth 3", color="#1b6ec2", edgecolor="black", linewidth=0.5)
b1 = ax.bar(x + 0.19, l1, 0.36, label="depth 1 (shallow)", color="#9dc3e6", edgecolor="black", linewidth=0.5)
ax.axhline(0.5, ls="--", c="k", lw=0.8, label="chance")
ax.axhline(triv, ls=":", c="#2f8f5b", lw=1.2, label=f"trivial floor {triv:.3f} (F-HOLO-2)")
ax.set_ylim(0.45, 1.0)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel("held-out separability AUROC (median, 5 seeds)")
ax.set_title(
    "holonomy-DFA: naive broadcast is shallow (0.831, L3≈L1); rotor-chain TRANSPORT\n"
    "recovers depth-composition and matches exact backprop (0.894); random feedback = chance"
)
for b, v in zip(b3, l3):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}", ha="center", fontsize=9)
# annotate alignment under each rule
for i, a in enumerate(align):
    ax.text(x[i], 0.465, f"align {a:+.2f}", ha="center", fontsize=8.5, color="#b26a10", fontweight="bold")
ax.legend(loc="upper right", fontsize=8.5, ncol=2)
fig.tight_layout()
fig.savefig(HERE / "holonomy_dfa_gate1.png")
print("wrote holonomy_dfa_gate1.png")
