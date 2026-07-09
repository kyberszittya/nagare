#!/usr/bin/env python3
"""Plot the 3-way inner-mechanism comparison on real signed graphs: flat vs fixed
degree-tier routing vs learned signed hypergraph convolution (signed-link AUROC).

Data measured by `examples/cpml_signed_link.rs` (Bitcoin Alpha/OTC 5 seeds, Slashdot 3).

    uv run --with matplotlib scripts/dev/plot_hgconv.py
"""

from __future__ import annotations

import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = {
    "Bitcoin\nAlpha": {
        "flat": [0.8686, 0.8700, 0.8808, 0.8712, 0.8593],
        "tier": [0.8818, 0.8899, 0.8817, 0.8723, 0.8852],
        "hgconv": [0.8662, 0.8524, 0.8852, 0.8677, 0.8736],
    },
    "Bitcoin\nOTC": {
        "flat": [0.8999, 0.8956, 0.8971, 0.9019, 0.8936],
        "tier": [0.9056, 0.9041, 0.8986, 0.9023, 0.9016],
        "hgconv": [0.9027, 0.8945, 0.8994, 0.9025, 0.8993],
    },
    "Slashdot": {
        "flat": [0.8906, 0.8898, 0.8901],
        "tier": [0.8943, 0.8923, 0.8905],
        "hgconv": [0.8909, 0.8896, 0.8941],
    },
}

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    graphs = list(DATA)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), dpi=140)
    x = range(len(graphs))
    w = 0.26
    med = lambda g, k: statistics.median(DATA[g][k])  # noqa: E731
    ax.bar([i - w for i in x], [med(g, "flat") for g in graphs], w, label="L=1 flat (fixed)", color="#8a8f98")
    ax.bar(list(x), [med(g, "tier") for g in graphs], w, label="L=3 tier routing (fixed)", color="#3b6fb0")
    ax.bar([i + w for i in x], [med(g, "hgconv") for g in graphs], w, label="hypergraph conv (learned)", color="#c9772e")
    ax.set_xticks(list(x))
    ax.set_xticklabels(graphs)
    ax.set_ylabel("median test AUROC")
    ax.set_ylim(0.85, 0.915)
    ax.set_title(
        "Signed-link inner mechanism: fixed tier routing wins\n"
        "learned hypergraph conv (1-round, linear) sits at the flat baseline, below tier — all 3 graphs",
        fontsize=9.5,
    )
    ax.legend(fontsize=9, loc="lower right", ncol=3)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT / "hgconv-vs-tier-signed-link.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
