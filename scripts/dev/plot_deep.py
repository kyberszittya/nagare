#!/usr/bin/env python3
"""N3 deep-stack A/B: median ± spread over seeds, train vs held-out rotation."""
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


arms = [
    ("2-block · C₈", "2blk_c8", "#2a9d8f"),
    ("2-block · C₁", "2blk_c1", "#e76f51"),
    ("1-block · C₈", "1blk_c8", "#457b9d"),
]

fig, (axt, axh) = plt.subplots(1, 2, figsize=(10.5, 4.6), sharey=True)
for ax, key, title in [(axt, "train_auc", "TRAIN orientations"), (axh, "test_auc", "HELD-OUT rotations")]:
    for i, (label, prefix, col) in enumerate(arms):
        vals = load(prefix, key)
        med = st.median(vals)
        ax.bar(i, med, 0.6, color=col)
        ax.errorbar(i, med, yerr=[[med - min(vals)], [max(vals) - med]], fmt="none", ecolor="k", capsize=4, lw=1)
        ax.text(i, med + 0.02, f"{med:.2f}", ha="center", fontsize=9)
    ax.axhline(0.5, ls="--", lw=1, color="k", alpha=0.5)
    ax.set_xticks(range(len(arms)))
    ax.set_xticklabels([a[0] for a in arms], fontsize=9)
    ax.set_title(title)
    ax.set_ylim(0.3, 1.05)
axt.set_ylabel("AUROC  (corner vs length-matched bar)")
axt.text(1.5, 0.52, "chance", fontsize=8, alpha=0.6)
fig.suptitle(
    "N3 — C₈ C-cell REQUIRED to fit compositional corner-vs-bar (C₁ ≈ chance);\n"
    "but local orientation-invariance ≠ global rotation-invariance (held-out ≈ chance, both depths)",
    fontsize=10.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
