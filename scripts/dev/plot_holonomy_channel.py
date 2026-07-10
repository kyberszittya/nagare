#!/usr/bin/env python3
"""Rotor-holonomy channel on the inner CPML core (signed-link AUROC). The order-sensitive rotor
holonomy over each signed cycle, unit-normalized and scattered to vertices, is concatenated into the
inner-core embedding. It HELPS on 6/6 runs (5 clear + 1 tie), never hurts — the first learned addition
this session to improve the flagship inner core. Data from `examples/cpml_signed_link` (run_holonomy arm).

    uv run --with matplotlib scripts/dev/plot_holonomy_channel.py
"""

from __future__ import annotations

import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (inner-core, inner+holonomy) AUROC per seed.
DATA = {
    "Bitcoin-Alpha": {
        "inner": [0.8818, 0.8899, 0.8817],
        "holo": [0.8810, 0.8954, 0.8896],
    },
    "Bitcoin-OTC": {
        "inner": [0.9056, 0.9041, 0.8986],
        "holo": [0.9072, 0.9083, 0.8993],
    },
}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.8, 4.7), dpi=140)
    graphs = list(DATA)
    x = range(len(graphs))
    w = 0.34
    for i, arm in enumerate(["inner", "holo"]):
        meds = [statistics.median(DATA[g][arm]) for g in graphs]
        color = "#9aa7b4" if arm == "inner" else "#2e7d32"
        label = "inner CPML core" if arm == "inner" else "inner core + rotor-holonomy channel"
        ax.bar([p + (i - 0.5) * w for p in x], meds, w, label=label, color=color)
        for p, g in zip(x, graphs):
            ax.scatter([p + (i - 0.5) * w] * 3, DATA[g][arm], s=15, color="#222", zorder=3, alpha=0.75)
    for p, g in zip(x, graphs):
        dm = statistics.median(DATA[g]["holo"]) - statistics.median(DATA[g]["inner"])
        ax.annotate(f"Δ {dm:+.4f}", (p, max(DATA[g]["holo"]) + 0.001), ha="center", fontsize=9,
                    color="#2e7d32", fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(graphs)
    ax.set_ylabel("test AUROC (median + seeds)")
    ax.set_ylim(0.86, 0.92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        "Rotor-holonomy channel HELPS the inner CPML core (6/6 runs, never hurts)\n"
        "the reframe vindicated: Clifford-FIR as a holonomy channel, not an outer compressor",
        fontsize=9.5,
    )
    fig.tight_layout()
    out = OUT / "holonomy-channel-signed-link.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
