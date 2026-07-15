#!/usr/bin/env python3
"""So-far dataset snapshot: accuracy/R2, wall-clock, and entropy-pool AUROC."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/dataset_snapshot_results.json"))
sup = d["supervised"]; au = d["auroc_entropy_pool"]
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.4))

# Panel 1: accuracy / R2, grouped bars
names = [r["dataset"] for r in sup]
xs = range(len(names)); w = 0.26
ax1.bar([x - w for x in xs], [r["evolvent"] for r in sup], w, label="evolvent (1-pass RLS)", color="#2a7de1")
ax1.bar([x for x in xs], [r["sgd"] for r in sup], w, label="1-pass SGD", color="#9aa0a6")
ax1.bar([x + w for x in xs], [r["mlp"] for r in sup], w, label="200-epoch MLP", color="#e08a2a")
ax1.set_xticks(list(xs)); ax1.set_xticklabels([f"{r['dataset']}\n({r['metric']})" for r in sup], fontsize=7, rotation=15)
ax1.set_ylabel("R² (reg) / accuracy (cls) — median of 3 seeds"); ax1.set_ylim(0, 1.05)
ax1.set_title("Accuracy / R² per dataset"); ax1.legend(fontsize=8, loc="lower left"); ax1.grid(axis="y", alpha=0.3)
ax1.annotate("MLP wins (learned\nfeatures matter)", (1, 0.44), fontsize=7, color="#b5651d", ha="center")

# Panel 2: wall-clock (evolvent 1-pass vs MLP 200-epoch), log
ax2.bar([x - 0.2 for x in xs], [r["evolvent_ms"] for r in sup], 0.4, label="evolvent (1 pass)", color="#2a7de1")
ax2.bar([x + 0.2 for x in xs], [r["mlp_ms"] for r in sup], 0.4, label="MLP (200 epochs)", color="#e08a2a")
ax2.set_yscale("log"); ax2.set_xticks(list(xs)); ax2.set_xticklabels(names, fontsize=7, rotation=15)
ax2.set_ylabel("train wall-clock (ms, log) — indicative, single-shot")
ax2.set_title("Compute time: one-pass vs 200-epoch"); ax2.legend(fontsize=8); ax2.grid(axis="y", alpha=0.3)

# Panel 3: entropy-pool AUROC clean vs hard
tnames = [r["task"] for r in au]; txs = range(len(tnames))
ax3.bar([x - 0.2 for x in txs], [r["clean"] for r in au], 0.4, label="clean", color="#2a9d5a")
ax3.bar([x + 0.2 for x in txs], [r["hard"] for r in au], 0.4, label="hard (noisy+missing+few-shot)", color="#c0392b")
ax3.set_xticks(list(txs)); ax3.set_xticklabels(tnames); ax3.set_ylim(0.9, 1.005)
ax3.set_ylabel("test AUROC"); ax3.set_title("Entropy-pool learner AUROC (on-mission)")
ax3.legend(fontsize=8, loc="lower left"); ax3.grid(axis="y", alpha=0.3)

fig.suptitle("Nagare dataset snapshot (2026-07-16) — accuracy/R², compute time, AUROC · so far", fontsize=12, y=1.02)
fig.tight_layout()
out = root / "reports/figures/dataset-snapshot.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
