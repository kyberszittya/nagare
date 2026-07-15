#!/usr/bin/env python3
"""E9 figure: wall-clock validation. Left: solve time vs d (log-log) — multifrontal
vs dense diverge. Right: measured speedup vs analytic flop count — same growth,
measured ~5x below (implementation constants)."""
import json, pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = pathlib.Path(__file__).resolve().parents[2]
d = json.load(open(root / "reports/figures/evolvent_e9_results.json"))
rows = d["rows"]
ds = [r["d"] for r in rows]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.3))

ax1.plot(ds, [r["mf_us"] for r in rows], "o-", color="#2a7de1", lw=2, label="MULTIFRONTAL O(d·w³)")
ax1.plot(ds, [r["dense_us"] for r in rows], "s--", color="#555", lw=2, label="DENSE Gauss O(d³)")
ax1.set_xscale("log", base=2); ax1.set_yscale("log")
ax1.set_xticks(ds); ax1.set_xticklabels(ds)
for r in rows:
    ax1.annotate(f"{r['mf_us']:.0f}µs", (r["d"], r["mf_us"]), textcoords="offset points", xytext=(3, -12), fontsize=7, color="#2a7de1")
ax1.annotate("21.8 ms", (889, 21753), textcoords="offset points", xytext=(-40, 4), fontsize=8, color="#555")
ax1.annotate("85 µs", (889, 85), textcoords="offset points", xytext=(-38, 4), fontsize=8, color="#2a7de1")
ax1.set_xlabel("features  d"); ax1.set_ylabel("solve wall-clock (µs, log) — criterion median")
ax1.set_title("Solve time: multifrontal vs dense diverge with d")
ax1.legend(fontsize=8); ax1.grid(alpha=0.3, which="both")

ax2.plot(ds, [r["analytic_flops"] for r in rows], "^--", color="#888", lw=2, label="analytic flop ratio (E5 claim)")
ax2.plot(ds, [r["measured_speedup"] for r in rows], "o-", color="#c0392b", lw=2.2, label="MEASURED wall-clock speedup")
for r in rows:
    ax2.annotate(f"{r['measured_speedup']:.0f}×", (r["d"], r["measured_speedup"]), textcoords="offset points", xytext=(4, 6), fontsize=8, color="#c0392b")
ax2.set_xscale("log", base=2); ax2.set_yscale("log")
ax2.set_xticks(ds); ax2.set_xticklabels(ds)
ax2.set_xlabel("features  d"); ax2.set_ylabel("speedup vs dense (×, log)")
ax2.set_title("Measured speedup grows as predicted, ~5× below flop count")
ax2.legend(fontsize=8); ax2.grid(alpha=0.3, which="both")

fig.suptitle("Evolvent E9 — wall-clock validation (criterion): the O(d·w³) win is REAL and growing (5.9× → 255×), ~5× below the analytic flop count",
             fontsize=10.5, y=1.02)
fig.tight_layout()
out = root / "reports/figures/evolvent-wallclock.png"
fig.savefig(out, dpi=140, bbox_inches="tight")
print("wrote", out)
