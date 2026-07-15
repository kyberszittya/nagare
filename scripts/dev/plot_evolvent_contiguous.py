#!/usr/bin/env python3
"""E10 figure: the contiguous-storage rewrite. Left: MF solve time before/after +
dense (log-log). Right: flop-to-measured gap closing (5x -> 1.8x)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e10_results.json"))
rows = d["rows"]
ds = [r["d"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

ax1.plot(ds, [r["dense_us"] for r in rows], "s--", color="#555", lw=2, label="DENSE Gauss O(d³)")
ax1.plot(ds, [r["mf_before_us"] for r in rows], "o--", color="#e08a2a", lw=2, label="MF before (Vec-per-clique)")
ax1.plot(ds, [r["mf_after_us"] for r in rows], "o-", color="#2a7de1", lw=2.4, label="MF after (contiguous)")
ax1.set_xscale("log", base=2); ax1.set_yscale("log"); ax1.set_xticks(ds); ax1.set_xticklabels(ds)
ax1.annotate("32 µs", (889, 31.7), textcoords="offset points", xytext=(-36, -12), fontsize=8, color="#2a7de1")
ax1.annotate("85 µs", (889, 85.1), textcoords="offset points", xytext=(-36, 4), fontsize=8, color="#e08a2a")
ax1.set_xlabel("features  d"); ax1.set_ylabel("solve wall-clock (µs, log) — criterion median")
ax1.set_title("MF solve 2.6–2.8× faster after the rewrite")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")

ax2.plot(ds, [r["flop_over_measured"] for r in rows], "o-", color="#2a7de1", lw=2.4, label="after (contiguous)")
ax2.plot(ds, [r["analytic_flops"] / r["speedup_before"] for r in rows], "o--", color="#e08a2a", lw=2, label="before (Vec-per-clique)")
ax2.axhline(1.0, color="#999", ls=":", lw=1)
ax2.annotate("flop ceiling (1.0×)", (105, 1.03), fontsize=8, color="#666")
for r in rows:
    ax2.annotate(f"{r['flop_over_measured']:.2f}×", (r["d"], r["flop_over_measured"]), textcoords="offset points", xytext=(3, -12), fontsize=8, color="#2a7de1")
ax2.set_xscale("log", base=2); ax2.set_xticks(ds); ax2.set_xticklabels(ds)
ax2.set_xlabel("features  d"); ax2.set_ylabel("flop count / measured speedup  (→1 = at ceiling)")
ax2.set_title("Flop-to-measured gap closed: ~5× → ~1.8×")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

fig.suptitle("Evolvent E10 — contiguous-storage rewrite: same exact answer, 2.7× faster solve, measured speedup now 16.7×–682× (flop gap ~5× → ~1.8×)",
             fontsize=10, y=1.02)
fig.tight_layout()
out = root / "reports/figures/evolvent-contiguous.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
