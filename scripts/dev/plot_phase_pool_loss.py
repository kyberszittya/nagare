#!/usr/bin/env python3
"""Loss curve of a learned front-end trained THROUGH the differentiable invariant phase-pool —
the visual proof that `phase_pool_backward` propagates gradient and reduces the loss it defines.
Data from `examples/phase_pool_curve` (CSV: step,loss).

    cargo run --release --example phase_pool_curve
    uv run --with matplotlib scripts/dev/plot_phase_pool_loss.py
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV = ROOT / "reports" / "figures" / "phase_pool_loss.csv"
OUT = ROOT / "reports" / "figures" / "phase-pool-loss-curve.png"


def main() -> None:
    steps, loss = [], []
    with CSV.open() as f:
        for row in csv.DictReader(f):
            steps.append(int(row["step"]))
            loss.append(float(row["loss"]))

    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=140)
    ax.plot(steps, loss, "-", color="#3b6fb0", lw=2)
    ax.scatter([steps[0], steps[-1]], [loss[0], loss[-1]], color="#c9772e", zorder=3)
    ax.annotate(f"{loss[0]:.2f}", (steps[0], loss[0]), textcoords="offset points", xytext=(6, 4), fontsize=9)
    ax.annotate(f"{loss[-1]:.3f}", (steps[-1], loss[-1]), textcoords="offset points", xytext=(-32, 6), fontsize=9)
    ax.set_xlabel("gradient step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title(
        "Global-pooling backpropagation: a learned front-end trained\n"
        "THROUGH the invariant phase-pool — loss falls, gradients flow x←W←pool←head",
        fontsize=10,
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
