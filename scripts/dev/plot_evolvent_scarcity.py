#!/usr/bin/env python3
"""E6 figure: the separator coupling is worth up to ~0.16 R2 in the data-scarce
regime. Left: R2 vs per (MF==DENSE hold; BLOCK starves at low per). Right: the
gap MF-BLOCK vs per (peaks mid-scarcity, vanishes when rich)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e6_results.json"))
rows = d["rows"]
per = [r["per"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

ax1.plot(per, [r["mf_r2"] for r in rows], "o-", color="#2a7de1", lw=2, label="MULTIFRONTAL (exact)")
ax1.plot(per, [r["dense_r2"] for r in rows], "x", color="#111", ms=8, label="DENSE (== MF)")
ax1.plot(per, [r["block_r2"] for r in rows], "s--", color="#e08a2a", lw=2, label="BLOCK (drops separators)")
ax1.fill_between(per, [r["block_r2"] for r in rows], [r["mf_r2"] for r in rows], color="#e08a2a", alpha=0.15)
ax1.axvline(d["clique_arity"], color="#888", ls=":", lw=1)
ax1.annotate("clique arity ≈ 8\n(scarce ← | → rich)", (d["clique_arity"], 0.45), fontsize=8, ha="center", color="#666")
ax1.set_xscale("log", base=2); ax1.set_xticks(per); ax1.set_xticklabels(per)
ax1.set_xlabel("measurements per clique  (per)"); ax1.set_ylabel("test R²  (median of 5 seeds)")
ax1.set_title("BLOCK starves when data < clique arity; MF/DENSE pool through separators")
ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3)

ax2.plot(per, [r["gap"] for r in rows], "o-", color="#c0392b", lw=2)
ax2.fill_between(per, 0, [r["gap"] for r in rows], color="#c0392b", alpha=0.15)
for r in rows:
    ax2.annotate(f"{r['gap']:.3f}", (r["per"], r["gap"]),
                 textcoords="offset points", xytext=(3, 6), fontsize=8, color="#c0392b")
ax2.set_xscale("log", base=2); ax2.set_xticks(per); ax2.set_xticklabels(per)
ax2.set_xlabel("measurements per clique  (per)"); ax2.set_ylabel("R² gap: MF − BLOCK  (value of the coupling)")
ax2.set_title("Separator coupling worth ~0.16 R² when scarce, ~0 when rich")
ax2.grid(alpha=0.3)

fig.suptitle("Evolvent E6 — the discriminating test: what the cross-clique (separator) coupling is worth (depth 6, d=441, 63 cliques)",
             fontsize=10.5, y=1.02)
fig.tight_layout()
out = root / "reports/figures/evolvent-scarcity.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
