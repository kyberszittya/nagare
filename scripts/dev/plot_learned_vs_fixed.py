#!/usr/bin/env python3
"""Learned vs fixed orientation field under the SAME |DFT| rotation invariant. Three arms —
fixed (frozen central-difference kernel), learned-scratch (random init), learned-warmstart (kernel
init AT central-diff, then trained) — on two datasets, upright + rotated, median/IQR over 5 seeds.

The warmstart arm is decisive: if it ≈ fixed, the central difference is a local optimum under the
pool (learning can't improve the hand-designed field). Data from `examples/cv_learned_field`.

    uv run --with matplotlib scripts/dev/plot_learned_vs_fixed.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Filled from the kato15 5-seed run (median over seeds). {dataset: {arm: (up, ro)}}
DATA = {
    "KTH-TIPS2-b (11 materials)": {
        "chance": 1.0 / 11.0,
        "fixed": (0.5051, 0.4116),
        "scratch": (0.3990, 0.3779),
        "warmstart": (0.4032, 0.3300),
    },
    "MNIST (10 digits)": {
        "chance": 1.0 / 10.0,
        "fixed": (0.4205, 0.2950),
        "scratch": (0.4490, 0.2160),
        "warmstart": (0.4230, 0.2700),
    },
}
ARMS = ["fixed", "scratch", "warmstart"]
COLORS = {"fixed": "#3b6fb0", "scratch": "#9aa7b4", "warmstart": "#c9772e"}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=140, sharey=True)
    for ax, (title, d) in zip(axes, DATA.items()):
        x = range(2)  # upright, rotated
        w = 0.26
        for i, arm in enumerate(ARMS):
            up, ro = d[arm]
            ax.bar([p + (i - 1) * w for p in x], [up, ro], w, label=arm, color=COLORS[arm])
        ax.axhline(d["chance"], color="#c0392b", ls=":", lw=0.9)
        ax.text(1.4, d["chance"] + 0.008, "chance", fontsize=7.5, color="#c0392b", ha="right")
        ax.set_xticks(list(x))
        ax.set_xticklabels(["upright", "rotated"])
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.7)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("test accuracy (median / 5 seeds)")
    fig.suptitle(
        "Learned vs fixed orientation field under the same |DFT| invariant — "
        "does training the 3×3 kernel beat the central difference?",
        fontsize=10.5,
    )
    fig.tight_layout()
    out = OUT / "cv-learned-vs-fixed.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
