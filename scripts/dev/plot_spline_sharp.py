#!/usr/bin/env python3
"""Plot the KB-vs-Chebyshev-CR spline-fit A/B on sharp vs smooth targets.

Data are the measured median MSEs (4 seeds) from `tests/spline_sharp_fit.rs`. Re-run that
test to regenerate; this only renders. Two panels: matched-grid (KB ⊇ CR, superset) and
matched-params (~32 each — the real budget question).

    uv run --with matplotlib scripts/dev/plot_spline_sharp.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

TARGETS = ["sine\n(smooth)", "step\n(sharp)", "kink\n(sharp)"]
# (title, cheb-label, cheb-mse, kb-label, kb-mse)
PANELS = [
    (
        "[1] matched GRID (g8)\nKB ⊇ Catmull-Rom → KB wins everywhere",
        "Cheb 8p",
        [1.047e-3, 1.520e-2, 4.705e-4],
        "KB 32p",
        [6.889e-7, 2.259e-3, 3.371e-5],
    ),
    (
        "[2] matched PARAMS (~32)\nfiner-grid Cheb wins — crushingly on the step",
        "Cheb g32 32p",
        [6.513e-7, 6.646e-6, 2.636e-5],
        "KB g8 32p",
        [6.889e-7, 2.259e-3, 3.371e-5],
    ),
]

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.6), dpi=140, sharey=True)
    x = range(len(TARGETS))
    w = 0.38
    for ax, (title, cl, cm, kl, km) in zip(axes, PANELS):
        ax.bar([i - w / 2 for i in x], cm, w, label=cl, color="#3b6fb0")
        ax.bar([i + w / 2 for i in x], km, w, label=kl, color="#c9772e")
        ax.set_yscale("log")
        ax.set_xticks(list(x))
        ax.set_xticklabels(TARGETS)
        ax.set_title(title, fontsize=9.5)
        ax.legend(fontsize=9)
        ax.grid(axis="y", which="both", alpha=0.22)
    axes[0].set_ylabel("median fit MSE (log)")
    # annotate the decisive step gap on panel 2
    axes[1].annotate(
        "≈340×",
        xy=(1, 2.259e-3),
        xytext=(1.25, 3e-4),
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="#444"),
    )
    fig.suptitle(
        "Kochanek-Bartels vs Chebyshev-CR spline fit — tangent DOF vs grid refinement",
        fontsize=11,
    )
    fig.tight_layout()
    out = OUT / "spline-kb-vs-cheb-sharp.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
