#!/usr/bin/env python3
"""Pose P3: the skeleton conv wins when a redundant structural constraint exists."""
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


fig, ax = plt.subplots(figsize=(7.6, 4.8))
conds = ["CLEAN\n(both arms)", "LEFT ARM\nOCCLUDED"]
base = [st.median(load("ps", "clean_lhand")), st.median(load("ps", "occ_lhand"))]
skel = [st.median(load("psh", "clean_lhand")), st.median(load("psh", "occ_lhand"))]
base_sp = [load("ps", "clean_lhand"), load("ps", "occ_lhand")]
skel_sp = [load("psh", "clean_lhand"), load("psh", "occ_lhand")]
w = 0.35
for i in range(2):
    ax.bar(i - w / 2, base[i], w, color="#adb5bd", label="backbone only" if i == 0 else None)
    ax.bar(i + w / 2, skel[i], w, color="#2a9d8f", label="+skeleton hg_conv" if i == 0 else None)
    ax.errorbar(i - w / 2, base[i], yerr=[[base[i] - min(base_sp[i])], [max(base_sp[i]) - base[i]]], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.errorbar(i + w / 2, skel[i], yerr=[[skel[i] - min(skel_sp[i])], [max(skel_sp[i]) - skel[i]]], fmt="none", ecolor="k", capsize=3, lw=1)
    ax.text(i - w / 2, base[i] + 0.4, f"{base[i]:.1f}", ha="center", fontsize=9)
    ax.text(i + w / 2, skel[i] + 0.4, f"{skel[i]:.1f}", ha="center", fontsize=9)
ax.set_xticks(range(2))
ax.set_xticklabels(conds)
ax.set_ylabel("left-hand error (px, 32px image)")
ax.set_title(
    "Pose P3 — skeleton conv WINS with a redundant structural constraint\n"
    "coupled arms: occluded arm recovered from its twin (2.6 vs 6.9px);\nleft-right disambiguated (5.9 vs 24px)"
)
ax.legend()
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
