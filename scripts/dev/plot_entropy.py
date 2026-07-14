#!/usr/bin/env python3
"""Entropy-top hypothesis test: recognition (invariant) + pose (equivariant) + speed."""
import glob
import json
import statistics as st
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed_dir, out = sys.argv[1], sys.argv[2]


def load(prefix, key):
    return [json.load(open(f))[key] for f in sorted(glob.glob(f"{seed_dir}/{prefix}_*.json"))]


ent_tr, ent_te = load("ent", "train_auc"), load("ent", "test_auc")
mean_tr, mean_te = load("entm", "train_auc"), load("entm", "test_auc")
pose = [p for p in load("ent", "pose_mae_deg") if p >= 0]
ups = load("ent", "updates_per_s")

fig, (ax, axp) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2, 1]})

groups = ["entropy-top\nTRAIN", "entropy-top\nHELD-OUT", "mean-top\nTRAIN", "mean-top\nHELD-OUT"]
data = [ent_tr, ent_te, mean_tr, mean_te]
cols = ["#2a9d8f", "#2a9d8f", "#adb5bd", "#adb5bd"]
for i, (vals, col) in enumerate(zip(data, cols)):
    med = st.median(vals)
    ax.bar(i, med, 0.62, color=col, edgecolor="k" if i < 2 else "none", lw=1.2)
    ax.errorbar(i, med, yerr=[[med - min(vals)], [max(vals) - med]], fmt="none", ecolor="k", capsize=4, lw=1)
    ax.text(i, med + 0.02, f"{med:.2f}", ha="center", fontsize=9)
ax.axhline(0.5, ls="--", lw=1, color="k", alpha=0.5)
ax.text(3.05, 0.52, "chance", fontsize=8, alpha=0.6)
ax.set_xticks(range(4))
ax.set_xticklabels(groups, fontsize=8.5)
ax.set_ylabel("recognition AUROC (corner vs length-matched bar)")
ax.set_ylim(0.35, 1.06)
ax.set_title("Recognition: entropy feed closes the held-out-rotation gap\n(1.00 vs mean-top ≈ chance)")

# pose + speed panel
axp.hist(pose, bins=6, color="#e76f51", alpha=0.85)
axp.axvline(st.median(pose), color="k", ls="--", lw=1)
axp.set_xlabel("pose MAE (deg, principal-axis vs true θ)")
axp.set_ylabel("seeds")
axp.set_title(f"Pose: median {st.median(pose):.1f}° MAE\nUpdate: {st.median(ups):.0f}/s (~{1e6/st.median(ups):.0f}µs) — real-time")

fig.suptitle(
    "Neocognitron entropy global-pool top — one feed: invariant Hs → recognition, equivariant angle → pose, fast update",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
