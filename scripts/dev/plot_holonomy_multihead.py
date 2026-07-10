#!/usr/bin/env python3
"""Multi-head rotor-holonomy on the inner CPML core, and an honesty check on the single-head positive.
M=4 gives a consistent +0.003 on both graphs but does NOT beat M=1's Alpha peak (+0.008); on OTC, M=1
is a wash. Changing only the edge-head init seed (22→90) leaves Alpha robust but flips OTC M=1 from
+0.003 to −0.001 — the holonomy gain is robust on Alpha, init-sensitive on OTC. Data from
`examples/cpml_signed_link` (--holo-heads).

    uv run --with matplotlib scripts/dev/plot_holonomy_multihead.py
"""

from __future__ import annotations

import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-seed AUROC (head-init seed +90).
DATA = {
    "Bitcoin-Alpha": {
        "inner": [0.8818, 0.8899, 0.8817],
        "M=1": [0.8769, 0.8898, 0.8946],
        "M=4": [0.8752, 0.8847, 0.8889],
    },
    "Bitcoin-OTC": {
        "inner": [0.9056, 0.9041, 0.8986],
        "M=1": [0.9029, 0.9060, 0.9023],
        "M=4": [0.9082, 0.9070, 0.9019],
    },
}
ARMS = ["inner", "M=1", "M=4"]
COLORS = {"inner": "#9aa7b4", "M=1": "#2e7d32", "M=4": "#3b6fb0"}
OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8.0, 4.7), dpi=140)
    graphs = list(DATA)
    x = range(len(graphs))
    w = 0.26
    for i, arm in enumerate(ARMS):
        meds = [statistics.median(DATA[g][arm]) for g in graphs]
        label = {"inner": "inner CPML core", "M=1": "+ holonomy M=1", "M=4": "+ holonomy M=4"}[arm]
        ax.bar([p + (i - 1) * w for p in x], meds, w, label=label, color=COLORS[arm])
        for p, g in zip(x, graphs):
            ax.scatter([p + (i - 1) * w] * 3, DATA[g][arm], s=13, color="#222", zorder=3, alpha=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(graphs)
    ax.set_ylabel("test AUROC (median + seeds)")
    ax.set_ylim(0.86, 0.92)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=9, loc="lower right")
    ax.set_title(
        "Multi-head holonomy: M=4 is consistent (+0.003 both graphs) but doesn't beat M=1's Alpha peak;\n"
        "M=1 OTC gain is init-sensitive (flips +0.003→−0.001 with head seed) — robust on Alpha only",
        fontsize=8.8,
    )
    fig.tight_layout()
    out = OUT / "holonomy-multihead-signed-link.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
