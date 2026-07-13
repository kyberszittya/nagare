#!/usr/bin/env python3
"""Render the SBSH detector demo: the adaptive quadtree grid, GT oriented boxes,
and the detector's predictions, plus the training-loss curve.

Reads the JSON from `examples/sbsh_detector_demo.rs`.

Usage:
    python scripts/dev/render_detections.py \
        reports/figures/sbsh-detections.json reports/figures/sbsh-detections.png
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle


def corners(box):
    cx, cy, w, h, th = box
    c, s = np.cos(th), np.sin(th)
    r = np.array([[c, -s], [s, c]])
    local = np.array([[-w / 2, -h / 2], [w / 2, -h / 2], [w / 2, h / 2], [-w / 2, h / 2]])
    return (local @ r.T) + np.array([cx, cy])


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/figures/sbsh-detections.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/sbsh-detections.png"
    with open(src) as f:
        d = json.load(f)
    g = d["g"]
    img = np.array(d["img"], dtype=float).reshape(g, g)

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(13, 6.2), gridspec_kw={"width_ratios": [1.5, 1]})

    ax.imshow(img, cmap="gray", origin="upper", extent=[0, g, g, 0], alpha=0.85)
    # adaptive quadtree grid (leaf cells) — the "hypergraph grid" that replaces YOLO's fixed grid
    for c in d["cells"]:
        y0, x0, y1, x1 = c
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="#3690c0", lw=0.5, alpha=0.55))
    # GT (green) and detections (red dashed)
    for t in d["gt"]:
        ax.add_patch(Polygon(corners(t), closed=True, fill=False, edgecolor="#1a9850", lw=2.4, label="GT"))
    for det in d["dets"]:
        ax.add_patch(Polygon(corners(det["box"]), closed=True, fill=False,
                             edgecolor="#d73027", lw=1.8, ls="--", label="detection"))
    h, l = ax.get_legend_handles_labels()
    seen = dict(zip(l, h))
    ax.legend(seen.values(), seen.keys(), loc="upper left", fontsize=9, framealpha=0.9)
    ax.set_xlim(0, g)
    ax.set_ylim(g, 0)
    ax.set_aspect("equal")
    ax.set_title(f"SBSH detector — adaptive quadtree grid ({len(d['cells'])} leaves)\n"
                 f"held-out P={d['precision']:.2f} R={d['recall']:.2f} F1={d['f1']:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    curve = np.array(d["loss_curve"])  # (it, obj_loss, box_loss)
    axl.plot(curve[:, 0], curve[:, 1], color="#4575b4", lw=1.8, label="objectness (BCE)")
    axl.plot(curve[:, 0], curve[:, 2], color="#f46d43", lw=1.8, label="box (KLD)")
    axl.set_title("training loss")
    axl.set_xlabel("scene (train step)")
    axl.set_ylabel("loss")
    axl.legend(fontsize=9)
    axl.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
