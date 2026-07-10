#!/usr/bin/env python3
"""Gömb-Soma Step-1 gate: does the FULL three-shell cascade (outer Clifford-FIR → HSiKAN → inner
CPML tiers) beat the INNER CPML core alone on real signed-link AUROC? Answer: no — the cascade loses
on every seed and both graphs, and costs 5–6s vs the inner core's near-instant. So Gömb-Soma's
compression/routing premise (hold AUROC at lower compute) is moot: the cheap model is already better.
Data from `examples/cpml_signed_link` (added `run_cascade` arm).

    uv run --with matplotlib scripts/dev/plot_gomb_soma_gate.py
"""

from __future__ import annotations

import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (inner-core, full-cascade) AUROC per seed, from the 3-seed sweep.
DATA = {
    "Bitcoin-Alpha": {
        "inner": [0.8818, 0.8899, 0.8817],
        "cascade": [0.8486, 0.8488, 0.8546],
    },
    "Bitcoin-OTC": {
        "inner": [0.9056, 0.9041, 0.8986],
        "cascade": [0.8941, 0.8916, 0.8846],
    },
}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=140)
    graphs = list(DATA)
    x = range(len(graphs))
    w = 0.34
    for i, arm in enumerate(["inner", "cascade"]):
        meds = [statistics.median(DATA[g][arm]) for g in graphs]
        color = "#3b6fb0" if arm == "inner" else "#c9772e"
        label = "inner CPML core" if arm == "inner" else "FULL cascade (outer+HSiKAN+inner)"
        ax.bar([p + (i - 0.5) * w for p in x], meds, w, label=label, color=color)
        for p, g in zip(x, graphs):  # per-seed points
            ax.scatter([p + (i - 0.5) * w] * 3, DATA[g][arm], s=14, color="#222", zorder=3, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(graphs)
    ax.set_ylabel("test AUROC (median + seeds)")
    ax.set_ylim(0.82, 0.92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        "Gömb-Soma Step-1 GATE (NEGATIVE): the full cascade LOSES to the inner core\n"
        "on 6/6 runs (Alpha −0.033, OTC −0.012) and is 5–6s slower — nothing to compress",
        fontsize=9.5,
    )
    fig.tight_layout()
    out = OUT / "gomb-soma-cascade-gate.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
