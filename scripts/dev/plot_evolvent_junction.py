#!/usr/bin/env python3
"""E4 figure: information-form (junction-tree) evolvent — exact = dense at O(n*w) storage.
Left: R2 (info == dense exactly, block trails). Right: info-matrix storage % of dense vs n."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e4_results.json"))
rows = d["rows"]
ns = [r["n"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Left: test R2 per arm (zoomed near 1.0 to show the block gap)
w = 0.25
xs = range(len(ns))
ax1.bar([x - w for x in xs], [r["dense_r2"] for r in rows], w, label="DENSE (O(n²), exact)", color="#555")
ax1.bar([x for x in xs], [r["info_r2"] for r in rows], w, label="INFO (O(n·w), EXACT)", color="#2a7de1")
ax1.bar([x + w for x in xs], [r["block_r2"] for r in rows], w, label="BLOCK (O(n·w), approx)", color="#e08a2a")
ax1.set_ylim(0.9990, 1.0)
ax1.set_xticks(list(xs)); ax1.set_xticklabels([f"n={n}" for n in ns])
ax1.set_ylabel("test R²  (median of 5 seeds)")
ax1.set_title("INFO matches DENSE exactly; BLOCK drops separator coupling")
ax1.legend(fontsize=8, loc="lower left"); ax1.grid(axis="y", alpha=0.3)

# Right: storage — info nnz % of dense vs n (the junction-tree win scales)
ax2.plot(ns, [r["info_pct"] for r in rows], "o-", color="#2a7de1", lw=2, label="INFO nnz  (banded J)")
ax2.plot(ns, [100.0]*len(ns), "--", color="#555", label="DENSE  (100%, O(n²))")
for r in rows:
    ax2.annotate(f"{r['info_pct']}%", (r["n"], r["info_pct"]),
                 textcoords="offset points", xytext=(4, 6), fontsize=8, color="#2a7de1")
ax2.set_xscale("log", base=2); ax2.set_xticks(ns); ax2.set_xticklabels(ns)
ax2.set_xlabel("features  n"); ax2.set_ylabel("precision / info-matrix storage (% of dense)")
ax2.set_title("Info-matrix storage O(n·w): shrinks as n grows (21.6% → 2.8%)")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle("Evolvent E4 — junction-tree (information-form) precision: exact as dense, sparse as the hypergraph",
             fontsize=11, y=1.02)
fig.tight_layout()
out = root / "reports/figures/evolvent-junction.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
