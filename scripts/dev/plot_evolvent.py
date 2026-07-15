#!/usr/bin/env python3
"""E0 evolvent stream: windowed prequential RMSE, evolvent (A) vs online-SGD (B) vs backprop-MLP (C)."""
import glob
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed_dir, out = sys.argv[1], sys.argv[2]
runs = [json.load(open(f)) for f in sorted(glob.glob(f"{seed_dir}/evolvent_*.json"))]
W = len(runs[0]["curve"])
# mean over seeds per window, per arm
mean = [[sum(r["curve"][w][a] for r in runs) / len(runs) for w in range(W)] for a in range(3)]

fig, ax = plt.subplots(figsize=(8, 4.7))
labels = ["A evolvent (forgetting-RLS)", "B online-SGD (same basis)", "C backprop-MLP (learned features)"]
cols = ["#e76f51", "#457b9d", "#2a9d8f"]
for a in range(3):
    ax.plot(range(W), mean[a], "o-", color=cols[a], label=labels[a], lw=1.8, ms=4)
ax.axvline(W / 2, ls="--", color="k", alpha=0.5)
ax.text(W / 2 + 0.1, ax.get_ylim()[1] * 0.6, "abrupt drift", fontsize=8, alpha=0.7)
ax.set_yscale("log")
ax.set_xlabel("stream window (→ time)")
ax.set_ylabel("prequential RMSE (log)")
ax.set_title(
    "E0 — evolvent vs backprop on a drifting stream\n"
    "A wins cold-start (fewer samples) but plain forgetting-RLS is windup-limited long-run;\n"
    "backprop converges & stays lower. Strong hypothesis needs directional forgetting."
)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
