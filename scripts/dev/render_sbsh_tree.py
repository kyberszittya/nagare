#!/usr/bin/env python3
"""Render the SBSH dynamic-quadtree smoke: the synthetic oriented-shape scene, the adaptive tree
leaves (green — note how they shrink onto the shapes), and the ground-truth oriented boxes (orange).
The visual proof of Hinge 1 (the tree concentrates cells on content). Data from `examples/sbsh_tree_smoke`.

    cargo run --release --example sbsh_tree_smoke -- --out /tmp/sbsh
    uv run --with matplotlib scripts/dev/render_sbsh_tree.py /tmp/sbsh
"""

from __future__ import annotations

import math
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np

SRC = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/sbsh")
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures" / "sbsh-tree-smoke.png"


def main() -> None:
    lines = (SRC / "boxes.txt").read_text().splitlines()
    g = int(lines[0])
    scene = np.frombuffer((SRC / "scene.bin").read_bytes(), dtype=np.uint8).reshape(g, g)
    it = iter(lines[1:])
    gts, leaves = [], []
    for ln in it:
        p = ln.split()
        if p[0] == "GT":
            gts = [next(it).split() for _ in range(int(p[1]))]
        elif p[0] == "LEAF":
            leaves = [next(it).split() for _ in range(int(p[1]))]

    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=140)
    ax.imshow(scene, cmap="gray", origin="upper", vmin=0, vmax=255)
    for x0, y0, x1, y1 in ((int(a), int(b), int(c), int(d)) for a, b, c, d in leaves):
        ax.add_patch(mp.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="#2ecc71", lw=0.5, alpha=0.8))
    for cx, cy, w, h, th in ((float(v) for v in gt) for gt in gts):
        r = mp.Rectangle((-w / 2, -h / 2), w, h, fill=False, ec="#e67e22", lw=2)
        t = matplotlib.transforms.Affine2D().rotate(th).translate(cx, cy) + ax.transData
        r.set_transform(t)
        ax.add_patch(r)
    ax.set_xlim(0, g)
    ax.set_ylim(g, 0)
    ax.set_title(f"SBSH dynamic quadtree ({len(leaves)} leaves): cells shrink onto objects\n"
                 "green = tree leaves · orange = ground-truth oriented boxes", fontsize=9.5)
    ax.axis("off")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")
    _ = math  # (kept for potential angle annotations)


if __name__ == "__main__":
    main()
