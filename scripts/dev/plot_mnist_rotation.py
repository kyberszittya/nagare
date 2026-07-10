#!/usr/bin/env python3
"""Plot the real-data (MNIST) regime split: spatial arms win upright but collapse under rotation;
the phase-pool holds. Data from `examples/mnist_cv` on kato15 (8k train / 2k test).

    uv run --with matplotlib scripts/dev/plot_mnist_rotation.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["raw-pixel\nlinear", "patch-embed\n(spatial)", "phase-pool\n|DFT|"]
UPRIGHT = [0.8805, 0.8515, 0.4155]
ROTATED = [0.2485, 0.2595, 0.2870]

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=140)
    x = range(len(ARMS))
    w = 0.38
    ax.bar([i - w / 2 for i in x], UPRIGHT, w, label="upright test", color="#3b6fb0")
    ax.bar([i + w / 2 for i in x], ROTATED, w, label="rotated test", color="#c9772e")
    ax.axhline(0.1, color="#c0392b", ls=":", lw=1.0)
    ax.text(len(ARMS) - 0.5, 0.115, "chance", color="#c0392b", fontsize=8, ha="right")
    for i, (u, r) in enumerate(zip(UPRIGHT, ROTATED)):
        ax.text(i, max(u, r) + 0.02, f"drop {r - u:+.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(list(x))
    ax.set_xticklabels(ARMS)
    ax.set_ylabel("MNIST test accuracy")
    ax.set_ylim(0, 1.0)
    ax.set_title(
        "MNIST — spatial arms win upright but COLLAPSE under rotation; phase-pool HOLDS\n"
        "(trained upright; rotated = randomly-rotated test digits)",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT / "mnist-rotation-robustness.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
