#!/usr/bin/env python3
"""Holonomy gain vs inner-core base AUROC across three signed graphs (INV M=4, 25-cell data×init grid
each). The gain is NOT monotonic in graph density — it tracks HEADROOM: OTC (base 0.90) is the sweet
spot (+0.0067, 23/25 robust); Epinions (densest, but base 0.933 near-ceiling) gains little (+0.0007,
14/25, not robust); Alpha (base 0.88 but sparse) is modest (+0.0028, 14/25). The density prediction
failed; headroom-bounded is the corrected mechanism. Data from `examples/cpml_signed_link --grid`.

    uv run --with matplotlib scripts/dev/plot_holonomy_headroom.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# graph: (inner base AUROC (median), holonomy median Δ, helps/25, robust?)
G = {
    "Bitcoin-Alpha\n(sparse)": (0.882, 0.00279, 14, False),
    "Bitcoin-OTC\n(sweet spot)": (0.904, 0.00666, 23, True),
    "Epinions\n(densest, near-ceiling)": (0.933, 0.00068, 14, False),
}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.8), dpi=140)
    for lab, (base, dm, hp, rob) in G.items():
        color = "#2e7d32" if rob else "#c9772e"
        ax.scatter([base], [dm], s=160, color=color, zorder=3, edgecolor="#222")
        ax.annotate(f"{lab}\nΔ {dm:+.4f}, {hp}/25", (base, dm), textcoords="offset points",
                    xytext=(8, 8 if lab.startswith("Bitcoin-OTC") else -28), fontsize=8.5)
    xs = [G[k][0] for k in G]
    ys = [G[k][1] for k in G]
    ax.plot(xs, ys, ls=":", color="#888", lw=1, zorder=1)
    ax.axhline(0.0, color="#555", ls="--", lw=1)
    ax.set_xlabel("inner-core base AUROC (headroom = 1 − base)")
    ax.set_ylabel("holonomy median ΔAUROC (INV M=4, 25 cells)")
    ax.set_xlim(0.87, 0.945)
    ax.set_ylim(-0.001, 0.008)
    ax.grid(alpha=0.25)
    ax.set_title(
        "Holonomy gain tracks HEADROOM, not density: OTC (moderate base) is the sweet spot;\n"
        "Epinions is densest but near-ceiling (0.933) → little to add. Density prediction FAILED.",
        fontsize=9.0,
    )
    fig.tight_layout()
    out = OUT / "holonomy-headroom.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
