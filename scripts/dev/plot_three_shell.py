#!/usr/bin/env python3
"""Plot the Gömb three-shell inner-core ablation: L=3 (tiered) vs L=1 (flat).

Data are the measured per-seed final BCEs from `tests/gomb_three_shell.rs` (5 seeds, L=3
three-shell teacher target). Re-run that test to regenerate; this only renders.

    uv run --with matplotlib scripts/dev/plot_three_shell.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2, 3, 4]
L3 = [0.0092, 0.0699, 0.1606, 0.0117, 0.0121]  # tiered inner (L=3)
L1 = [0.0133, 0.0100, 0.0413, 0.0453, 0.0102]  # flat inner (L=1)
L3_MED, L1_MED = 0.0121, 0.0133

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.7), dpi=140)
    x = range(len(SEEDS))
    w = 0.38
    ax.bar([i - w / 2 for i in x], L3, w, label="L=3 tiered inner (CPML)", color="#3b6fb0")
    ax.bar([i + w / 2 for i in x], L1, w, label="L=1 flat inner", color="#c9772e")
    ax.axhline(L3_MED, color="#3b6fb0", ls="--", lw=1.1, alpha=0.8)
    ax.axhline(L1_MED, color="#c9772e", ls=":", lw=1.3, alpha=0.9)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"seed {s}" for s in SEEDS])
    ax.set_ylabel("final BCE (lower = better)")
    ax.set_title(
        "Gömb three-shell: does the CPML inner tier-stratification earn its weight?\n"
        "median tie (0.0121 vs 0.0133); L=3 lower on only 2/5 seeds — NOT a robust win",
        fontsize=9.5,
    )
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = OUT / "gomb-three-shell-inner-ablation.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
