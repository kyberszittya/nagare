#!/usr/bin/env python3
"""Plot the CPML inner-core justification: L=3 tiered vs L=1 flat signed-link AUROC on
real heavy-tailed graphs (Bitcoin Alpha/OTC 5 seeds, Slashdot 3 seeds).

Data measured by `examples/cpml_signed_link.rs`. Re-run to regenerate; this only renders.

    uv run --with matplotlib scripts/dev/plot_cpml_justify.py
"""

from __future__ import annotations

import pathlib
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per-seed test AUROC
DATA = {
    "Bitcoin\nAlpha": {
        "L3": [0.8818, 0.8899, 0.8817, 0.8723, 0.8852],
        "L1": [0.8686, 0.8700, 0.8808, 0.8712, 0.8593],
    },
    "Bitcoin\nOTC": {
        "L3": [0.9056, 0.9041, 0.8986, 0.9023, 0.9016],
        "L1": [0.8999, 0.8956, 0.8971, 0.9019, 0.8936],
    },
    "Slashdot": {
        "L3": [0.8943, 0.8923, 0.8906],
        "L1": [0.8906, 0.8898, 0.8901],
    },
}

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    graphs = list(DATA)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.6), dpi=140)
    x = range(len(graphs))
    w = 0.38

    l3med = [statistics.median(DATA[g]["L3"]) for g in graphs]
    l1med = [statistics.median(DATA[g]["L1"]) for g in graphs]
    axL.bar([i - w / 2 for i in x], l3med, w, label="L=3 tiered inner (CPML)", color="#3b6fb0")
    axL.bar([i + w / 2 for i in x], l1med, w, label="L=1 flat inner", color="#c9772e")
    axL.set_xticks(list(x))
    axL.set_xticklabels(graphs)
    axL.set_ylabel("median test AUROC")
    axL.set_ylim(0.85, 0.92)
    axL.set_title("Median signed-link AUROC — tiered vs flat inner", fontsize=10)
    axL.legend(fontsize=9, loc="lower right")
    axL.grid(axis="y", alpha=0.25)

    # per-seed ΔAUROC — the consistency story (every point > 0).
    for i, g in enumerate(graphs):
        deltas = [a - b for a, b in zip(DATA[g]["L3"], DATA[g]["L1"])]
        axR.scatter([i] * len(deltas), deltas, color="#3b6fb0", s=42, zorder=3)
        axR.scatter([i], [statistics.median(deltas)], color="#c0392b", marker="_", s=800, zorder=2)
    axR.axhline(0.0, color="#444", lw=1.0)
    axR.set_xticks(list(x))
    axR.set_xticklabels(graphs)
    axR.set_ylabel("ΔAUROC (L=3 − L=1)")
    axR.set_title("Per-seed Δ (red = median) — 13/13 positive, never hurts", fontsize=10)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle(
        "CPML inner core justified: degree-tier stratification helps on real heavy-tailed "
        "signed graphs\n(contrast the toy uniform-degree 2c ablation, which tied)",
        fontsize=10.5,
    )
    fig.tight_layout()
    out = OUT / "cpml-justify-signed-link.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
