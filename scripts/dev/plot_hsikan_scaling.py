#!/usr/bin/env python3
"""HSiKAN structural-leverage H1 scaling figure (article Fig 9c).

Reads docs/article/data/hsikan_ladder.json (5-seed richness ladder, `structural`
target = B^2 x). Left: per-model test error (median +/- IQR) vs chain length.
Right: the scramble-isolated structure-benefit ratio (HSiKAN.scrambled /
HSiKAN.true) vs chain length -- the H1 scaling curve (3.7x -> 61x).
"""
import json
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = json.load(open(sys.argv[1]))
out = sys.argv[2]
rows = data["rows"]
n = [r["n_nodes"] for r in rows]


def med(model):
    return [r[model]["median"] for r in rows]


def iqr(model):
    return [r[model]["iqr"] for r in rows]


fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4.6))

# left: per-model error vs chain length (log y)
models = [
    ("hsikan_true", "HSiKAN · true structure", "#2a9d8f", "o-"),
    ("hsikan_scrambled", "HSiKAN · scrambled", "#e76f51", "s--"),
    ("mlp", "MLP (flat net)", "#457b9d", "^-"),
    ("deepsets", "DeepSets (no msg-pass)", "#adb5bd", "d:"),
]
for key, label, col, style in models:
    ax.errorbar(n, med(key), yerr=iqr(key), fmt=style, color=col, capsize=3, lw=1.6, ms=5, label=label)
ax.set_yscale("log")
ax.set_xlabel("chain length n (richness)")
ax.set_ylabel("test error (median ± IQR, log)")
ax.set_title("HSiKAN·true stays low & flat; scramble / MLP degrade with depth\n(structural target = B²x, params-matched ~3700, 5 seeds)")
ax.legend(fontsize=8)
ax.set_xticks(n)

# right: structure benefit ratio (scr/true) = H1 scaling
ben = [r["hsikan_scrambled"]["median"] / r["hsikan_true"]["median"] for r in rows]
mlp_hk = [r["mlp"]["median"] / r["hsikan_true"]["median"] for r in rows]
axb.plot(n, ben, "o-", color="#e76f51", lw=2, ms=6, label="structure benefit (scramble/true)")
axb.plot(n, mlp_hk, "^--", color="#457b9d", lw=1.4, ms=5, label="MLP / HSiKAN gap")
for x, y in zip(n, ben):
    axb.annotate(f"{y:.0f}×", (x, y), textcoords="offset points", xytext=(0, 7), fontsize=8, ha="center", color="#c1440e")
axb.set_xlabel("chain length n (richness)")
axb.set_ylabel("benefit ratio (×)")
axb.set_title("H1 scaling: destroying incidence hurts HSiKAN\nmore as the chain lengthens — 3.7× → 61×")
axb.legend(fontsize=8, loc="upper left")
axb.set_xticks(n)

fig.suptitle("HSiKAN structural-leverage — H1 (scaling) SUPPORTED; the causal double-dissociation gives H2", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(out, dpi=140)
print("wrote", out)
