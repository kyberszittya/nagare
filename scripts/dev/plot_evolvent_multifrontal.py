#!/usr/bin/env python3
"""E5 figure: multifrontal Cholesky over a branching hypergraph — exact = dense,
solve-time win grows as (d/w)^2. Left: R2 (MF==DENSE, BLOCK trails). Right: flop
speedup (grows) + storage % (drops) vs d."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e5_results.json"))
rows = d["rows"]
ds = [r["d"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

w = 0.25
xs = range(len(ds))
ax1.bar([x - w for x in xs], [r["dense_r2"] for r in rows], w, label="DENSE (O(d³), exact)", color="#555")
ax1.bar([x for x in xs], [r["mf_r2"] for r in rows], w, label="MULTIFRONTAL (O(d·w³), EXACT)", color="#2a7de1")
ax1.bar([x + w for x in xs], [r["block_r2"] for r in rows], w, label="BLOCK (drops separators)", color="#e08a2a")
ax1.set_ylim(0.9985, 1.0)
ax1.set_xticks(list(xs)); ax1.set_xticklabels([f"d={r['d']}\n{r['cliques']} cliques" for r in rows], fontsize=8)
ax1.set_ylabel("test R²  (median of 5 seeds)")
ax1.set_title("MULTIFRONTAL == DENSE exactly; BLOCK drops the coupling")
ax1.legend(fontsize=8, loc="lower left"); ax1.grid(axis="y", alpha=0.3)

# Right: flop speedup (grows) + storage % (drops)
axb = ax2.twinx()
l1 = ax2.plot(ds, [r["flops_speedup"] for r in rows], "o-", color="#2a7de1", lw=2, label="factorization speedup ×")
ax2.set_yscale("log")
for r in rows:
    ax2.annotate(f"{r['flops_speedup']}×", (r["d"], r["flops_speedup"]),
                 textcoords="offset points", xytext=(4, 6), fontsize=8, color="#2a7de1")
l2 = axb.plot(ds, [r["store_pct"] for r in rows], "s--", color="#e0662a", lw=2, label="storage (% of dense d²)")
ax2.set_xlabel("features  d"); ax2.set_ylabel("factorization flop speedup vs dense d³/6 (log)")
axb.set_ylabel("frontal storage (% of dense)")
ax2.set_title("Solve-time win grows as (d/w)²: 18× → 1270×")
ax2.legend(l1 + l2, [x.get_label() for x in l1 + l2], fontsize=8, loc="center right")
ax2.grid(alpha=0.3)

fig.suptitle("Evolvent E5 — multifrontal (clique-tree) Cholesky: exact as dense, sparse+fast as the hypergraph; online update touches only log₂(N) cliques",
             fontsize=10.5, y=1.02)
fig.tight_layout()
out = root / "reports/figures/evolvent-multifrontal.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
