import json, pathlib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np
root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root/"reports/figures/pytorch_bench.json"))["rows"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, r in zip(axes, d):
    arms = ["Nagare KAN", "Nagare MLP\n(1 thread)", "PyTorch MLP\n(1 thread)", "PyTorch MLP\n(6 threads)"]
    ms = [r["ms_kan"], r["ms_mlp_1thr"], r["ms_pytorch_1thr"], r["ms_pytorch_6thr"]]
    colors = ["#2a9d5a", "#2a7de1", "#e08a2a", "#c99a5a"]
    ax.bar(range(4), ms, color=colors)
    for i, v in enumerate(ms): ax.text(i, v+max(ms)*0.01, f"{v:.0f}", ha="center", fontsize=9)
    ax.set_xticks(range(4)); ax.set_xticklabels(arms, fontsize=8)
    met = r["metric"]; acc = f"{met}≈{r['nagare_kan'] if met=='R2' else r['nagare_kan']:.3f}"
    ax.set_title(f"{r['dataset']}  (all arms {met} {r['pytorch_mlp']:.3f}, KAN {r['nagare_kan']:.3f})", fontsize=10)
    ax.set_ylabel("train wall-clock (ms, median/3)"); ax.grid(axis="y", alpha=0.3)
fig.suptitle("Nagare vs PyTorch on CPU (same data, same accuracy): the closed-form KAN is 2–21× faster; the dense MLP is a wash",
             fontsize=11, y=1.03)
fig.tight_layout(); out = root/"reports/figures/pytorch-bench.png"
fig.savefig(out, dpi=140, bbox_inches="tight"); print("wrote", out)
