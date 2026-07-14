#!/usr/bin/env python3
"""Pose P2: spatial-backbone multi-pose localization (P1 unblock) + skeleton A/B."""
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


fig, (axm, axo) = plt.subplots(1, 2, figsize=(10.5, 4.6))

# left: all-joint MAE per seed, baseline vs skeleton
bmae, hmae = load("pb", "all_joint_mae"), load("pbh", "all_joint_mae")
x = range(len(bmae))
axm.plot(x, bmae, "o-", color="#457b9d", label=f"backbone (med {st.median(bmae):.2f}px)")
axm.plot(x, hmae, "s--", color="#2a9d8f", label=f"+skeleton (med {st.median(hmae):.2f}px)")
axm.axhline(1.0, ls=":", color="k", alpha=0.4)
axm.set_xlabel("seed")
axm.set_ylabel("all-joint MAE (px, 28px image)")
axm.set_title("P1 UNBLOCK: multi-pose localization,\nspatial backbone, NO coord channels")
axm.legend(fontsize=8)
axm.set_ylim(0, max(max(bmae), max(hmae)) * 1.1)

# right: elbow clean vs occluded (over-constrained middle joint)
groups = ["elbow\nCLEAN", "elbow\nOCCLUDED"]
b = [st.median(load("pb", "clean_elbow_err")), st.median(load("pb", "occ_elbow_err"))]
h = [st.median(load("pbh", "clean_elbow_err")), st.median(load("pbh", "occ_elbow_err"))]
w = 0.35
axo.bar([i - w / 2 for i in range(2)], b, w, color="#457b9d", label="backbone")
axo.bar([i + w / 2 for i in range(2)], h, w, color="#2a9d8f", label="+skeleton")
for i, (bv, hv) in enumerate(zip(b, h)):
    axo.text(i - w / 2, bv + 0.03, f"{bv:.2f}", ha="center", fontsize=8)
    axo.text(i + w / 2, hv + 0.03, f"{hv:.2f}", ha="center", fontsize=8)
axo.set_xticks(range(2))
axo.set_xticklabels(groups)
axo.set_ylabel("elbow error (px)")
axo.set_title("Occlusion recoverable by backbone alone\n(middle joint over-constrained → skeleton neutral)")
axo.legend(fontsize=8)

fig.suptitle("SBSH Pose P2 — the P1 unblock: ScBlock spatial backbone + soft_argmax keypoint head", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=140)
print("wrote", out)
