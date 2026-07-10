#!/usr/bin/env python3
"""Holonomy-channel robustness grid: ΔAUROC (inner+holonomy − inner) over a 5×5 (data-seed × init-seed)
grid per graph. The proper grid REVERSES the earlier single-init picture: robustly POSITIVE on
Bitcoin-OTC (median +0.0035, 19/25 help, IQR>0), a WASH on Bitcoin-Alpha (median ~0, 10/25 help).
The effect is small relative to init variance (inner core itself swings ±0.015), so few-seed reads
mislead. Data from `examples/cpml_signed_link --grid`.

    uv run --with matplotlib scripts/dev/plot_holonomy_robustness.py
"""

from __future__ import annotations

import pathlib
import statistics as st

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ALPHA = [-0.00492, 0.00506, 0.01075, -0.00163, -0.00374, -0.00017, -0.00092, 0.00731, -0.00493,
         0.00255, 0.01294, -0.00866, 0.01045, 0.00003, 0.00311, -0.00041, -0.00820, -0.00226,
         -0.00454, 0.00972, -0.01467, 0.00301, -0.00188, 0.00371, -0.01148]
OTC = [-0.00271, 0.00412, 0.00393, 0.00118, 0.00425, 0.00185, 0.00480, 0.01578, 0.00196, 0.00357,
       0.00366, -0.00475, 0.00611, 0.00061, -0.00241, 0.00571, -0.00261, 0.00189, 0.00337, 0.00950,
       -0.00136, 0.00633, 0.00351, 0.00871, -0.00522]
DATA = {"Bitcoin-Alpha\n(wash)": ALPHA, "Bitcoin-OTC\n(robust +)": OTC}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=140)
    labels = list(DATA)
    for i, (lab, vs) in enumerate(DATA.items()):
        xs = [i + (j % 5 - 2) * 0.03 for j in range(len(vs))]
        colors = ["#2e7d32" if v > 0 else "#c0392b" for v in vs]
        ax.scatter(xs, vs, c=colors, s=26, alpha=0.8, zorder=3)
        med = st.median(vs)
        ax.plot([i - 0.28, i + 0.28], [med, med], color="#222", lw=2.4, zorder=4)
        ax.annotate(f"median {med:+.4f}\n{sum(1 for v in vs if v > 0.0005)}/25 help",
                    (i, max(vs) + 0.0012), ha="center", fontsize=9)
    ax.axhline(0.0, color="#555", ls="--", lw=1)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("ΔAUROC  (inner + holonomy − inner)")
    ax.set_ylim(-0.02, 0.02)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title(
        "Holonomy robustness grid (5 data-seeds × 5 init-seeds = 25/graph)\n"
        "green=helps, red=hurts — robustly POSITIVE on OTC, a WASH on Alpha (reverses single-init reads)",
        fontsize=9.2,
    )
    fig.tight_layout()
    out = OUT / "holonomy-robustness-grid.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
