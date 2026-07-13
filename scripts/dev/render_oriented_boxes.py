#!/usr/bin/env python3
"""Render the oriented-head demo: target vs learned oriented boxes + loss curve.

Reads the JSON emitted by `examples/oriented_head_demo.rs` and draws each node's
anchor (grey), ground-truth target box (green) and the box learned under the
closed-form Gaussian-KLD loss (red, dashed), plus the training loss curve.

Usage:
    python scripts/dev/render_oriented_boxes.py \
        reports/figures/oriented-head-boxes.json \
        reports/figures/oriented-head-boxes.png
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def corners(box):
    """4 corners of an oriented box [cx, cy, w, h, theta]."""
    cx, cy, w, h, th = box
    c, s = np.cos(th), np.sin(th)
    r = np.array([[c, -s], [s, c]])
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (local @ r.T) + np.array([cx, cy])


def anchor_box(a):
    """Anchor [cx, cy, w, h] as an axis-aligned oriented box."""
    return [a[0], a[1], a[2], a[3], 0.0]


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/figures/oriented-head-boxes.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/oriented-head-boxes.png"
    with open(src) as f:
        d = json.load(f)

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={"width_ratios": [1.6, 1]})

    for a in d["anchors"]:
        ax.add_patch(Polygon(corners(anchor_box(a)), closed=True, fill=False,
                             edgecolor="0.75", lw=1.0, ls=":"))
    for t in d["targets"]:
        ax.add_patch(Polygon(corners(t), closed=True, fill=False,
                             edgecolor="#1a9850", lw=2.4, label="target"))
    for b in d["learned"]:
        ax.add_patch(Polygon(corners(b), closed=True, fill=False,
                             edgecolor="#d73027", lw=2.0, ls="--", label="learned (KLD)"))
    # de-dup legend
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=10)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 16)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # image coords (row down)
    ax.set_title("Oriented head — target vs learned boxes\n(closed-form Gaussian-KLD, no autograd)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")

    curve = np.array(d["loss_curve"])
    axl.plot(curve[:, 0], curve[:, 1], color="#4575b4", lw=2.0)
    axl.axhline(0.0, color="0.8", lw=0.8)
    axl.set_yscale("log")
    axl.set_title(f"KLD loss: {d['l0']:.3f} -> {d['l1']:.4f}")
    axl.set_xlabel("Adam step")
    axl.set_ylabel("mean KLD loss (log)")
    axl.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
