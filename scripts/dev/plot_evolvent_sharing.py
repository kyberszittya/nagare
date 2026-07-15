#!/usr/bin/env python3
"""E7 figure: separator-sharing (star topology). The block deficit jumps when a
separator is shared, then PLATEAUS with fan-out (partial refutation of 'more
sharing -> monotonically bigger gap'). MF == DENSE throughout."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e7_results.json"))
rows = d["rows"]
fo = [r["fanout"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

ax1.plot(fo, [r["mf_r2"] for r in rows], "o-", color="#2a7de1", lw=2, label="MULTIFRONTAL (exact)")
ax1.plot(fo, [r["dense_r2"] for r in rows], "x", color="#111", ms=8, label="DENSE (== MF)")
ax1.plot(fo, [r["block_r2"] for r in rows], "s--", color="#e08a2a", lw=2, label="BLOCK (drops shared separator)")
ax1.fill_between(fo, [r["block_r2"] for r in rows], [r["mf_r2"] for r in rows], color="#e08a2a", alpha=0.15)
ax1.set_xscale("log", base=2); ax1.set_xticks(fo); ax1.set_xticklabels(fo)
ax1.set_xlabel("fan-out  (cliques sharing one separator = fanout+1)"); ax1.set_ylabel("test R²  (median of 5 seeds)")
ax1.set_title("BLOCK deficit is large whenever a separator is shared")
ax1.legend(fontsize=8, loc="lower right"); ax1.grid(alpha=0.3)

gap = [r["gap_med"] for r in rows]
lo = [r["gap_med"] - r["gap_q1"] for r in rows]
hi = [r["gap_q3"] - r["gap_med"] for r in rows]
ax2.errorbar(fo, gap, yerr=[lo, hi], fmt="o-", color="#c0392b", lw=2, capsize=4, label="gap MF−BLOCK (median [IQR])")
ax2.axhline(0.16, color="#888", ls=":", lw=1)
ax2.annotate("E6 within-tree gap ≈ 0.16 (per=4)", (1, 0.165), fontsize=8, color="#666")
ax2.set_xscale("log", base=2); ax2.set_xticks(fo); ax2.set_xticklabels(fo)
ax2.set_ylim(0, 0.42)
ax2.set_xlabel("fan-out"); ax2.set_ylabel("R² gap: MF − BLOCK")
ax2.set_title("Gap JUMPS with sharing, then PLATEAUS — not monotone in fan-out")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle("Evolvent E7 — separator-sharing axis (star topology, per=4 scarce): sharing matters, but the gap saturates with fan-out (multi-seed refutes monotone growth)",
             fontsize=10, y=1.02)
fig.tight_layout()
out = root / "reports/figures/evolvent-sharing.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
