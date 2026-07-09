#!/usr/bin/env python3
"""Plot the HSiKAN-on-Iris-graph CR-Chebyshev vs Kochanek-Bartels comparison.

Data are the measured per-seed held-out accuracies from
`tests/hsikan_graph_spline.rs` (kNN=6, 754 signed triangles, 5 seeds). Re-run that
test to regenerate the numbers; this script only renders them.

Usage (no repo dependency — matplotlib is pulled ephemerally):
    uv run --with matplotlib scripts/dev/plot_hsikan_spline.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2, 3, 4]
CHEB = [0.868, 0.974, 0.868, 0.947, 0.974]
KB = [0.868, 0.974, 0.895, 0.947, 0.947]
CHEB_MED, KB_MED = 0.947, 0.947
NP_CHEB, NP_KB = 96, 352

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.8), dpi=140)
    x = range(len(SEEDS))
    w = 0.38
    ax.bar([i - w / 2 for i in x], CHEB, w, label=f"Chebyshev-CR ({NP_CHEB} params)", color="#3b6fb0")
    ax.bar([i + w / 2 for i in x], KB, w, label=f"Kochanek-Bartels ({NP_KB} params)", color="#c9772e")
    ax.axhline(CHEB_MED, color="#3b6fb0", ls="--", lw=1.2, alpha=0.8)
    ax.axhline(KB_MED, color="#c9772e", ls=":", lw=1.4, alpha=0.9)
    ax.text(len(SEEDS) - 0.5, CHEB_MED + 0.002, f"median {CHEB_MED:.3f} (tie)", ha="right", fontsize=9)

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("held-out accuracy")
    ax.set_ylim(0.80, 1.0)
    ax.set_title(
        "HSiKAN on the Iris signed graph — spline basis A/B\n"
        "(hsikan → scatter_mean → linear → softmax; basis is the only varying factor)",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT / "hsikan-graph-spline-cheb-vs-kb.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
