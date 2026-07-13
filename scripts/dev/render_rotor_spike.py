#!/usr/bin/env python3
"""Render the rotor-spike tuning demo: orientation tuning curves narrowing with
kappa (V1-like), and the population "spike" sharpening.

Usage:
    python scripts/dev/render_rotor_spike.py \
        reports/figures/rotor-spike-tuning.json reports/figures/rotor-spike-tuning.png
"""
import json
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "reports/figures/rotor-spike-tuning.json"
    dst = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/rotor-spike-tuning.png"
    with open(src) as f:
        d = json.load(f)
    kappas = d["kappas"]
    thetas = np.array(d["thetas"]) * 180.0 / np.pi  # degrees
    k = d["k"]
    stim_deg = d["stim"] * 180.0 / np.pi
    colors = ["#4575b4", "#f46d43", "#7b3294"]

    fig, (ax, axp) = plt.subplots(1, 2, figsize=(13, 5.4))

    for i, kap in enumerate(kappas):
        ax.plot(thetas, d["curves"][i], color=colors[i % 3], lw=2.0, label=f"κ = {kap:g}")
    ax.set_title("Orientation tuning of the μ=0 unit\n(narrows → 'spike' as κ grows — V1-like)")
    ax.set_xlabel("stimulus orientation (deg)")
    ax.set_ylabel("normalised response y₀")
    ax.set_xlim(0, 180)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    bins = np.arange(k) * 180.0 / k
    for i, kap in enumerate(kappas):
        axp.plot(bins, d["pops"][i], color=colors[i % 3], lw=2.0, marker="o", ms=4, label=f"κ = {kap:g}")
    axp.axvline(stim_deg, color="0.5", ls="--", lw=1.2, label=f"stimulus {stim_deg:.0f}°")
    axp.set_title("Population spike to one stimulus\n(divisive-normalised bank)")
    axp.set_xlabel("preferred orientation μ_k (deg)")
    axp.set_ylabel("normalised response y_k")
    axp.set_xlim(0, 180)
    axp.legend(fontsize=9)
    axp.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(dst, dpi=150)
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
