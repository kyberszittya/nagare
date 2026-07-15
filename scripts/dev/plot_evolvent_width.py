#!/usr/bin/env python3
"""E8 figure: the width/representability boundary. A non-additive (product)
cross-clique term is representable only by a clique wide enough to host it.
MF-WIDE (width-4 multifrontal) == DENSE-WIDE, flat; NARROW (linear-only) collapses
as beta grows."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e8_results.json"))
rows = d["rows"]
beta = [r["beta"] for r in rows]
fig, ax = plt.subplots(figsize=(7.6, 4.6))

ax.plot(beta, [r["mf_wide_r2"] for r in rows], "o-", color="#2a7de1", lw=2.2,
        label="MF-WIDE — multifrontal, width-4 clique tree (hosts the product)")
ax.plot(beta, [r["dense_wide_r2"] for r in rows], "x", color="#111", ms=9, mew=2,
        label="DENSE-WIDE (== MF-WIDE, exact ceiling)")
nar = [r["narrow_r2"] for r in rows]
lo = [max(0.0, r["narrow_r2"] - r["narrow_q1"]) for r in rows]
hi = [max(0.0, r["narrow_q3"] - r["narrow_r2"]) for r in rows]
ax.errorbar(beta, nar, yerr=[lo, hi], fmt="s--", color="#c0392b", lw=2.2, capsize=4,
            label="NARROW — linear-only (product OMITTED)")
ax.fill_between(beta, nar, [r["mf_wide_r2"] for r in rows], color="#c0392b", alpha=0.10)
ax.axhline(0, color="#999", lw=0.8)
ax.set_xlabel("β  —  weight of the non-additive product term  rk0·rk1")
ax.set_ylabel("test R²  (median of 5 seeds)")
ax.set_title("E8 — width/representability boundary: the product is fittable only by a\nclique wide enough to host it; the narrow model omits it regardless of data")
ax.legend(fontsize=8.5, loc="lower left")
ax.grid(alpha=0.3)
ax.annotate("MF storage = 14% of dense (width 4)", (2.0, 0.95), fontsize=8, color="#2a7de1")
fig.tight_layout()
out = root / "reports/figures/evolvent-width.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
