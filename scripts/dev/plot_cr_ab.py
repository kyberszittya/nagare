#!/usr/bin/env python3
"""Plot the learnable-Chebyshev-CR edge-encoder A/B: binary vs tanh vs cr test
AUROC per graph (median + min/max whiskers over seeds).

Reads the cr_edge_encoder stdout logs (one per graph, lines like
'... binary: test AUROC init X -> final Y'). Usage:
    python scripts/dev/plot_cr_ab.py out.png log1.txt=bitcoin-otc log2.txt=bitcoin-alpha
"""
import re
import statistics
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODES = ["binary", "tanh", "cr"]
COLORS = {"binary": "#7f7f7f", "tanh": "#4575b4", "cr": "#d73027"}


def parse(path):
    out = {m: [] for m in MODES}
    for line in open(path):
        for m in MODES:
            if f" {m}:" in line:
                out[m].append(float(re.search(r"final ([0-9.]+)", line).group(1)))
    return out


def main():
    png = sys.argv[1]
    specs = [a.split("=") for a in sys.argv[2:]]
    graphs = [(name, parse(path)) for path, name in specs]

    fig, ax = plt.subplots(figsize=(9, 5.4))
    x = np.arange(len(graphs))
    w = 0.26
    for i, m in enumerate(MODES):
        med = [statistics.median(g[1][m]) for g in graphs]
        lo = [med[j] - min(g[1][m]) for j, g in enumerate(graphs)]
        hi = [max(g[1][m]) - med[j] for j, g in enumerate(graphs)]
        ax.bar(x + (i - 1) * w, med, w, yerr=[lo, hi], capsize=4, color=COLORS[m], alpha=0.9,
               label={"binary": "binary (sign ±1)", "tanh": "tanh (fixed)",
                      "cr": "Chebyshev-CR (learnable)"}[m])
    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in graphs])
    n_seeds = len(graphs[0][1]["binary"])
    ax.set_ylabel(f"test AUROC (median, min–max over {n_seeds} seeds)")
    ax.set_ylim(0.85, 0.93)
    ax.set_title("Learnable Chebyshev-CR edge-weight encoder vs fixed encodings\n"
                 "(standalone end-to-end sign predictor; warm-started spline)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
