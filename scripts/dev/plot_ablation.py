#!/usr/bin/env python3
"""CV ablation: the spatial phase-map R-sweep (locality↔invariance trade) is domain-dependent —
locality lifts digits, global is best for textures — and rotation-augmentation flattens the
rotation drop. Data from `examples/cv_bench` on kato15.

    uv run --with matplotlib scripts/dev/plot_ablation.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# R-sweep upright accuracy (train-upright)
MNIST_R = [1, 2, 4, 7]
MNIST_UP = [0.4155, 0.7630, 0.8555, 0.8835]
KTH_R = [1, 2, 4, 8]
KTH_UP = [0.6059, 0.6010, 0.5665, 0.5123]

# Best rotation-robust accuracy (train-augmented), rotated-test, per arm
AUG_ARMS = ["pixel", "patch", "phase\nR=1", "sphase\nR=2", "sphase\nR=4", "sphase\nR=7/8", "mix"]
MNIST_AUG_RO = [0.4815, 0.4680, 0.3840, 0.4240, 0.5450, 0.5965, 0.5560]
KTH_AUG_RO = [0.3153, 0.3448, 0.5567, 0.5813, 0.5123, 0.4089, 0.3793]

OUT = pathlib.Path(__file__).resolve().parents[2] / "reports" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 4.7), dpi=140)

    axL.plot(MNIST_R, MNIST_UP, "o-", color="#3b6fb0", lw=2, label="MNIST (digits)")
    axL.plot(KTH_R, KTH_UP, "s-", color="#c9772e", lw=2, label="KTH-TIPS (textures)")
    axL.set_xlabel("spatial phase-map grid R  (1 = global invariant → local)")
    axL.set_ylabel("upright test accuracy")
    axL.set_title("Spatial phase map (Dir 2): locality lifts digits,\nglobal is best for textures — a domain knob", fontsize=9.5)
    axL.set_xticks([1, 2, 4, 7, 8])
    axL.legend(fontsize=9)
    axL.grid(alpha=0.25)

    x = range(len(AUG_ARMS))
    w = 0.38
    axR.bar([i - w / 2 for i in x], MNIST_AUG_RO, w, label="MNIST", color="#3b6fb0")
    axR.bar([i + w / 2 for i in x], KTH_AUG_RO, w, label="KTH-TIPS", color="#c9772e")
    axR.set_xticks(list(x))
    axR.set_xticklabels(AUG_ARMS, fontsize=7.5)
    axR.set_ylabel("rotated-test accuracy")
    axR.set_title("Rotation-augmented training (Dir 3): best robust arm\n= spatial-phase (MNIST R=7, KTH R=2), not the crude mix", fontsize=9.5)
    axR.legend(fontsize=9)
    axR.grid(axis="y", alpha=0.25)

    fig.suptitle("Nagare CV ablation — the spatial phase map is the lift (locality × phase), domain-tuned by R", fontsize=11)
    fig.tight_layout()
    out = OUT / "cv-ablation-sweep.png"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
