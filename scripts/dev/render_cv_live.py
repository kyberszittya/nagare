#!/usr/bin/env python3
"""Render the Nagare CV live demo → an animated GIF: for one sample per class, the image rotates
360° and each model's prediction is shown (green = correct, red = wrong). Reads the `frames.bin`
+ `meta.txt` dumped by `examples/cv_live`.

    python render_cv_live.py --in /tmp/cv_live --out reports/gifs/cv_live_mnist.gif
"""

from __future__ import annotations

import argparse
import math
import pathlib

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ind", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Nagare CV — model predictions as the image rotates")
    a = ap.parse_args()
    ind = pathlib.Path(a.ind)

    lines = (ind / "meta.txt").read_text().splitlines()
    m, frames, g, nmod = (int(v) for v in lines[0].split())
    names = lines[1].split(",")
    true = [int(v) for v in lines[2].split(",")]
    preds = {}
    for ln in lines[3:]:
        p = ln.split()
        preds[(int(p[0]), int(p[1]))] = [int(v) for v in p[2:]]

    raw = np.fromfile(ind / "frames.bin", dtype=np.uint8).reshape(m, frames, g, g)

    cols = min(5, m)
    rows = math.ceil(m / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.7, rows * 2.1), dpi=130)
    axes = np.array(axes).reshape(-1)
    ims, texts = [], []
    for s in range(m):
        ax = axes[s]
        im = ax.imshow(raw[s, 0], cmap="gray", vmin=0, vmax=255, animated=True)
        ax.set_title(f"true: {true[s]}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        tt = [ax.text(0.5, -0.13 - 0.16 * j, "", transform=ax.transAxes, ha="center", fontsize=7) for j in range(nmod)]
        ims.append(im)
        texts.append(tt)
    for s in range(m, len(axes)):
        axes[s].axis("off")

    def update(f):
        for s in range(m):
            ims[s].set_array(raw[s, f])
            pr = preds[(s, f)]
            for j in range(nmod):
                ok = pr[j] == true[s]
                texts[s][j].set_text(f"{names[j]}: {pr[j]}")
                texts[s][j].set_color("#1a8a3a" if ok else "#c0392b")
        deg = int(f / frames * 360)
        fig.suptitle(f"{a.title}   (rotation {deg}°)", fontsize=11)
        return [*ims, *(t for tt in texts for t in tt)]

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    anim = FuncAnimation(fig, update, frames=frames, interval=140, blit=False)
    out = pathlib.Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(out), writer=PillowWriter(fps=7))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
