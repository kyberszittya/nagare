#!/usr/bin/env python3
"""KTH-TIPS2-b (11 materials, 3564 train / 1188 test, 64x64) — the hard texture bench.

Two reads of the phase-pool result:
  (A) R-sweep, upright-trained: an INTERIOR optimum at R=4 — between MNIST (monotone-up, needs
      max locality) and KTH-TIPS v1 (monotone-down, needs global). 11-class materials carry
      meso-structure, so moderate locality is the sweet spot.
  (B) rotation-augmented training makes the phase arms near-perfectly rotation-invariant
      (drop -> ~0, even positive); the best rotated arm is spatial-phase R=4 at 0.564, while
      pixel/patch/mix stay stuck near chance (1/11 = 0.091). Data from `examples/cv_bench`.

    uv run --with matplotlib scripts/dev/plot_kth_tips2.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHANCE = 1.0 / 11.0

# Panel A — spatial phase-map R-sweep, upright-trained (upright / rotated)
R = [1, 2, 4, 8]
UP = [0.4916, 0.5598, 0.5993, 0.5084]
RO = [0.4040, 0.4133, 0.3535, 0.2795]

# Panel B — rotated-test accuracy per arm, upright-train vs rotation-augmented-train
ARMS = ["pixel", "patch", "phase\nR=1", "sphase\nR=2", "sphase\nR=4", "sphase\nR=8", "mix"]
RO_UPTRAIN = [0.1759, 0.1902, 0.4040, 0.4133, 0.3535, 0.2795, 0.1911]
RO_AUGTRAIN = [0.1532, 0.2214, 0.4571, 0.5253, 0.5640, 0.4756, 0.1987]

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.8, 4.7), dpi=140)

    axL.plot(R, UP, "o-", color="#3b6fb0", lw=2, label="upright test")
    axL.plot(R, RO, "s--", color="#c9772e", lw=2, label="rotated test")
    axL.axvline(4, color="#2e7d32", ls=":", lw=1.2)
    axL.annotate("interior optimum\n(meso-structure)", (4, 0.5993),
                 textcoords="offset points", xytext=(8, -6), fontsize=8, color="#2e7d32")
    axL.axhline(CHANCE, color="#c0392b", ls=":", lw=0.9)
    axL.text(7.2, CHANCE + 0.01, "chance 1/11", fontsize=7.5, color="#c0392b", ha="right")
    axL.set_xlabel("spatial phase-map grid R  (1 = global invariant -> local)")
    axL.set_ylabel("test accuracy")
    axL.set_title("KTH-TIPS2-b R-sweep (train-upright):\nR has an INTERIOR optimum on 11-class materials", fontsize=9.5)
    axL.set_xticks(R)
    axL.set_ylim(0, 0.7)
    axL.legend(fontsize=9)
    axL.grid(alpha=0.25)

    x = range(len(ARMS))
    w = 0.38
    axR.bar([i - w / 2 for i in x], RO_UPTRAIN, w, label="upright-trained", color="#9aa7b4")
    axR.bar([i + w / 2 for i in x], RO_AUGTRAIN, w, label="rotation-augmented", color="#c9772e")
    axR.axhline(CHANCE, color="#c0392b", ls=":", lw=0.9)
    axR.text(len(ARMS) - 0.6, CHANCE + 0.008, "chance", fontsize=7.5, color="#c0392b", ha="right")
    axR.set_xticks(list(x))
    axR.set_xticklabels(ARMS, fontsize=7.5)
    axR.set_ylabel("rotated-test accuracy")
    axR.set_ylim(0, 0.7)
    axR.set_title("Augmentation makes phase arms rotation-invariant;\npixel/patch/mix stay near chance", fontsize=9.5)
    axR.legend(fontsize=9)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle("KTH-TIPS2-b (11 materials) — phase-pool is 2-3x pixels; best robust arm = spatial-phase R=4 (0.564)", fontsize=10.5)
    fig.tight_layout()
    out = OUT / "cv-kth-tips2-11class.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
