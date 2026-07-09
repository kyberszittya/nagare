#!/usr/bin/env python3
"""Plot the Nagare CV approach ladder on rotation-invariant shape ID: the quaternion-PHASE pool
(pool the rotor's orientation phase, take |DFT| = rotation-invariant) dominates every earlier
approach. Data measured across the session's vision tests.

    uv run --with matplotlib scripts/dev/plot_phase_pool.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (approach, median acc, note)
LADDER = [
    ("vector\nrotor-pool", 0.52, "failed"),
    ("raw\nhistogram", 0.606, "covariant"),
    ("single-θ\ncanonical", 0.600, ""),
    ("C_8\ngroup-conv", 0.650, ""),
    ("phase-pool\n|DFT|", 0.969, "PHASE"),
    ("phase-pool\n+ entropy", 0.969, "PHASE"),
]
# entropy ablation, per seed
SEEDS = [0, 1, 2, 3]
PHASE = [0.919, 0.969, 0.962, 0.969]
PHENT = [0.938, 0.969, 0.962, 0.969]

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.0, 4.7), dpi=140, gridspec_kw={"width_ratios": [2, 1]})

    x = range(len(LADDER))
    colors = ["#8a8f98", "#8a8f98", "#3b6fb0", "#3b6fb0", "#c9772e", "#c9772e"]
    bars = axL.bar(list(x), [a[1] for a in LADDER], 0.62, color=colors)
    axL.axhline(0.25, color="#c0392b", ls=":", lw=1.0)
    axL.text(len(LADDER) - 0.5, 0.265, "chance", color="#c0392b", fontsize=8, ha="right")
    for b, a in zip(bars, LADDER):
        axL.text(b.get_x() + b.get_width() / 2, a[1] + 0.012, f"{a[1]:.2f}", ha="center", fontsize=8.5)
    axL.set_xticks(list(x))
    axL.set_xticklabels([a[0] for a in LADDER], fontsize=8)
    axL.set_ylabel("median test accuracy")
    axL.set_ylim(0.2, 1.02)
    axL.set_title("Rotation-invariant shape ID — pooling the PHASE wins\n(pool the rotor's orientation, not the rotated vector)", fontsize=10)
    axL.grid(axis="y", alpha=0.25)

    xb = range(len(SEEDS))
    w = 0.38
    axR.bar([i - w / 2 for i in xb], PHASE, w, label="phase-pool |DFT|", color="#c9772e")
    axR.bar([i + w / 2 for i in xb], PHENT, w, label="+ entropy", color="#3b6fb0")
    axR.set_xticks(list(xb))
    axR.set_xticklabels([f"s{s}" for s in SEEDS])
    axR.set_ylim(0.88, 1.0)
    axR.set_ylabel("test accuracy")
    axR.set_title("Entropy feedback\n(helps where headroom exists; task saturates ~0.97)", fontsize=9.5)
    axR.legend(fontsize=8, loc="lower right")
    axR.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out = OUT / "phase-pool-cv-ladder.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
