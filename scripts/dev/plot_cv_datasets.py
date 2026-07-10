#!/usr/bin/env python3
"""Nagare CV on two real datasets — the phase-pool's rank flip across domains: worst on MNIST
digits (spatial, upright), best on KTH-TIPS textures (rotation-nuisance, orientation-driven).
Both evaluated upright + randomly-rotated (edge-clamp rotation). Data from `examples/cv_bench`.

    uv run --with matplotlib scripts/dev/plot_cv_datasets.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = ["raw-pixel", "patch-embed", "phase-pool"]
DATA = {
    "MNIST (digits) — spatial task": {
        "up": [0.8805, 0.8545, 0.4155],
        "ro": [0.2485, 0.2655, 0.2870],
    },
    "KTH-TIPS (textures) — rotation-nuisance": {
        "up": [0.5123, 0.4286, 0.6059],
        "ro": [0.3103, 0.3251, 0.4581],
    },
}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), dpi=140, sharey=True)
    for ax, (title, d) in zip(axes, DATA.items()):
        x = range(len(ARMS))
        w = 0.38
        ax.bar([i - w / 2 for i in x], d["up"], w, label="upright", color="#3b6fb0")
        ax.bar([i + w / 2 for i in x], d["ro"], w, label="rotated", color="#c9772e")
        # highlight the phase-pool group
        ax.axvspan(2 - 0.5, 2 + 0.5, color="#c9772e", alpha=0.06)
        ax.axhline(0.1, color="#c0392b", ls=":", lw=0.9)
        ax.set_xticks(list(x))
        ax.set_xticklabels(ARMS, fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=9, loc="upper right")
    axes[0].set_ylabel("test accuracy")
    fig.suptitle(
        "Phase-pool rank flip: WORST on digits (spatial), BEST on textures (rotation-nuisance)\n"
        "and the best-and-most-robust arm under rotation on textures (0.61→0.46)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT / "cv-datasets-phase-pool.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
