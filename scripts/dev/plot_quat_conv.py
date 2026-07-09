#!/usr/bin/env python3
"""Plot the quaternion-for-CV arc: rotor in the POOL fails (3 ways), rotor in the CONVOLUTION
(on the equivariant gradient field) wins. Data measured by tests + probes over this session.

    uv run --with matplotlib scripts/dev/plot_quat_conv.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Panel A — where the rotor lives: (design, rotor-acc, baseline-acc, baseline-name)
ARC = [
    ("pool: free\n(v0)", 0.52, 0.71, "mean-pool"),
    ("pool: geo-angle\non tokens (v1)", 0.56, 0.71, "mean-pool"),
    ("pool: on\ngradients (v2)", 0.45, 0.44, "raw-pool"),
    ("CONV: on\ngradients", 0.60, 0.29, "raw-conv"),
]
# Panel B — the conv win, per seed
SEEDS = [0, 1, 2, 3]
CANON = [0.663, 0.556, 0.600, 0.594]
RAW = [0.294, 0.287, 0.275, 0.312]

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.5, 4.7), dpi=140)

    x = range(len(ARC))
    w = 0.38
    axA.bar([i - w / 2 for i in x], [a[1] for a in ARC], w, label="rotor", color="#c9772e")
    axA.bar([i + w / 2 for i in x], [a[2] for a in ARC], w, label="baseline", color="#8a8f98")
    axA.axhline(0.25, color="#c0392b", ls=":", lw=1.0)
    axA.text(3.5, 0.265, "chance", color="#c0392b", fontsize=8, ha="right")
    axA.set_xticks(list(x))
    axA.set_xticklabels([a[0] for a in ARC], fontsize=8)
    axA.set_ylabel("test accuracy")
    axA.set_ylim(0.2, 0.8)
    axA.set_title("Where does the quaternion belong?\nPOOL fails (3 ways); CONV on gradients wins", fontsize=9.5)
    axA.legend(fontsize=9, loc="upper right")
    axA.grid(axis="y", alpha=0.25)

    xb = range(len(SEEDS))
    axB.bar([i - w / 2 for i in xb], CANON, w, label="rotor-canonical", color="#3b6fb0")
    axB.bar([i + w / 2 for i in xb], RAW, w, label="raw", color="#8a8f98")
    axB.axhline(0.25, color="#c0392b", ls=":", lw=1.0)
    axB.set_xticks(list(xb))
    axB.set_xticklabels([f"seed {s}" for s in SEEDS])
    axB.set_ylabel("test accuracy")
    axB.set_ylim(0.2, 0.75)
    axB.set_title("Quaternion patch conv: canonical vs raw\n4/4 seeds, Δ +0.31 median (raw ≈ chance)", fontsize=9.5)
    axB.legend(fontsize=9, loc="upper right")
    axB.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Nagare CV — quaternion in the convolution, not the pool (rotation-invariant shape ID)",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT / "quat-conv-cv.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
