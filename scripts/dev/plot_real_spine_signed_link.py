#!/usr/bin/env python3
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/real_spine_signed_link.json"))
arms = d["arms_test_auroc_2seed"]
names = [a["arm"] for a in arms]
real = [ (a["s0"]+a["s1"])/2 for a in arms ]
shuf = [ a["shuffle_s0"] for a in arms ]
colors = ["#c0392b" if "FULL" in n else ("#2a9d5a" if "tiered" in n else "#7a86b8") for n in names]
y = np.arange(len(names)); h=0.38
fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.barh(y+h/2, real, h, color=colors, label="real (2-seed mean)")
ax.barh(y-h/2, shuf, h, color="#c9ced8", label="sign-shuffle control")
ax.axvline(0.5, color="#888", ls="--", lw=1)
ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5); ax.invert_yaxis()
for i,v in enumerate(real): ax.text(v+0.004, y[i]+h/2, f"{v:.3f}", va="center", fontsize=8.5,
    fontweight="bold" if "FULL" in names[i] or "tiered" in names[i] else "normal")
ax.set_xlim(0.4, 0.92); ax.set_xlabel("test AUROC")
ax.set_title("Real spine on real signed-link (bitcoin-alpha): structure is real (shuffle→chance),\nbut the FULL Gömb-Soma cascade underperforms the simple inner core")
ax.legend(fontsize=8, loc="lower right"); ax.grid(axis="x", alpha=0.3)
fig.tight_layout(); out = root/"reports/figures/real-spine-signed-link.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
