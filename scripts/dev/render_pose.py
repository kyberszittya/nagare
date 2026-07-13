#!/usr/bin/env python3
"""Render the SBSH pose P0 smoke: GT stick figure vs joints predicted through the
soft-argmax head, plus the MSE loss curve.

Usage:
    python scripts/dev/render_pose.py reports/figures/pose-smoke.json reports/figures/pose-smoke.png
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/figures/pose-smoke.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/pose-smoke.png"
    with open(src) as f:
        d = json.load(f)
    g = d["g"]
    gt = np.array(d["gt"])
    pred = np.array(d["pred"])
    edges = d["edges"]

    fig, (ax, axl) = plt.subplots(1, 2, figsize=(11, 5.4), gridspec_kw={"width_ratios": [1, 1]})

    # skeleton edges (GT green, pred red dashed)
    for a, b in edges:
        ax.plot([gt[a, 0], gt[b, 0]], [gt[a, 1], gt[b, 1]], "-", color="#1a9850", lw=2.5, zorder=1)
        ax.plot([pred[a, 0], pred[b, 0]], [pred[a, 1], pred[b, 1]], "--", color="#d73027", lw=1.8, zorder=2)
    ax.scatter(gt[:, 0], gt[:, 1], s=90, color="#1a9850", label="GT joints", zorder=3)
    ax.scatter(pred[:, 0], pred[:, 1], s=40, color="#d73027", label="soft-argmax pred", zorder=4)
    ax.set_xlim(0, g)
    ax.set_ylim(g, 0)  # image coords (row down)
    ax.set_aspect("equal")
    ax.set_title(f"SBSH pose P0 — joints via soft-argmax head\nmax joint error {d['max_err']:.3f} px (no autograd)")
    ax.set_xlabel("x (col)")
    ax.set_ylabel("y (row)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.25)

    curve = np.array(d["loss_curve"])
    axl.plot(curve[:, 0], curve[:, 1], color="#4575b4", lw=2)
    axl.set_yscale("log")
    axl.set_title("coordinate MSE (log)")
    axl.set_xlabel("Adam step")
    axl.set_ylabel("MSE (px^2)")
    axl.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
