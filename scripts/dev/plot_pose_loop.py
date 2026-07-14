#!/usr/bin/env python3
"""Pose P4: the closed loop breaks the shared-elin hg_conv (skeleton hurts → per-edge transform needed)."""
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


base = load("pl", "clean_coupler")
skel = load("plh", "clean_coupler")
mid = st.median(load("pl", "midpoint_oracle"))

fig, ax = plt.subplots(figsize=(7.4, 4.6))
x = range(len(base))
ax.plot(x, base, "o-", color="#457b9d", label=f"backbone only (med {st.median(base):.1f}px)")
ax.plot(x, skel, "s--", color="#e76f51", label=f"+skeleton 4-cycle (med {st.median(skel):.1f}px — HURTS)")
ax.axhline(mid, ls=":", color="k", alpha=0.6, label=f"midpoint-oracle {mid:.1f}px (shared-elin ceiling)")
ax.set_xlabel("seed")
ax.set_ylabel("clean coupler error (px, 32px image)")
ax.set_title(
    "Pose P4 — the closed loop BREAKS the shared-elin hg_conv\n"
    "loop closure C=B+D−A is asymmetric; shared elin → neighbour-midpoint (wrong)\n"
    "→ skeleton hurts → motivates a per-edge transform"
)
ax.legend(fontsize=8)
ax.set_ylim(0, max(max(base), max(skel)) * 1.15)
fig.tight_layout()
fig.savefig(out, dpi=140)
print("wrote", out)
