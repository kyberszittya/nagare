#!/usr/bin/env python3
"""Invariant holonomy + multi-head, 3-way ablation on the (data×init) robustness grid (25 cells/graph).
Median ΔAUROC (inner+holonomy − inner) for RAW M=1 (covariant quaternion), INV M=1 (gauge-invariant
scalar Re(H)), INV M=4 (invariant + 4 heads). Finding: multi-head helps ONLY when invariant — INV M=4
is the strongest config (OTC +0.0067/23-of-25 robust; Alpha +0.0028, rescued from raw's wash).
Data from `examples/cpml_signed_link --grid --holo-invariant`.

    uv run --with matplotlib scripts/dev/plot_holonomy_invariant.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (median ΔAUROC, helps/25) per arm per graph.
DATA = {
    "Bitcoin-Alpha": {"RAW M=1": (-0.00041, 10), "INV M=1": (0.00260, 13), "INV M=4": (0.00279, 14)},
    "Bitcoin-OTC": {"RAW M=1": (0.00351, 19), "INV M=1": (0.00228, 15), "INV M=4": (0.00666, 23)},
}
ARMS = ["RAW M=1", "INV M=1", "INV M=4"]
COLORS = {"RAW M=1": "#9aa7b4", "INV M=1": "#e0a33e", "INV M=4": "#2e7d32"}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.2, 4.7), dpi=140)
    graphs = list(DATA)
    x = range(len(graphs))
    w = 0.26
    for i, arm in enumerate(ARMS):
        meds = [DATA[g][arm][0] for g in graphs]
        ax.bar([p + (i - 1) * w for p in x], meds, w, label=arm, color=COLORS[arm])
        for p, g in zip(x, graphs):
            md, hp = DATA[g][arm]
            ax.annotate(f"{hp}/25", (p + (i - 1) * w, md + (0.0002 if md >= 0 else -0.0006)),
                        ha="center", va="bottom" if md >= 0 else "top", fontsize=8)
    ax.axhline(0.0, color="#555", ls="--", lw=1)
    ax.set_xticks(list(x))
    ax.set_xticklabels(graphs)
    ax.set_ylabel("median ΔAUROC over 25 (data×init) cells")
    ax.set_ylim(-0.002, 0.009)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title(
        "Holonomy: invariant scalar + multi-head. Multi-head helps ONLY when invariant.\n"
        "INV M=4 = strongest (OTC +0.0067/23-of-25 robust; Alpha +0.0028, rescued from raw's wash)",
        fontsize=9.0,
    )
    fig.tight_layout()
    out = OUT / "holonomy-invariant-ablation.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
