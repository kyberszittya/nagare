#!/usr/bin/env python3
"""Parse the signed-link benchmark logs (5 seeds x {epinions, slashdot}) into a
median/IQR table + a grouped-bar figure.

Reads /tmp/sl_bench/<ds>_s<seed>.log (the cpml_signed_link harness output),
writes a markdown table to stdout, a JSON summary, and a PNG figure.

Usage:
    python scripts/dev/analyze_signed_link_bench.py \
        /tmp/sl_bench reports/figures/signed-link-bench.png reports/figures/signed-link-bench.json
"""
import glob
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARMS = [
    ("flat", r"L=1 flat  inner \(fixed\):\s+test AUROC ([0-9.]+)"),
    ("inner_l3", r"L=3 tiered inner \(fixed\):\s+test AUROC ([0-9.]+)"),
    ("hgconv", r"signed hypergraph conv \(learned\): test AUROC ([0-9.]+)"),
    ("cascade", r"FULL cascade L=3.*: test AUROC ([0-9.]+)"),
    ("holo_m1", r"holonomy \(M=1\):\s+test AUROC ([0-9.]+)"),
    ("holo_m4", r"holonomy \(M=4\):\s+test AUROC ([0-9.]+)"),
]
LABELS = {
    "flat": "L=1 flat",
    "inner_l3": "L=3 inner",
    "hgconv": "hg-conv",
    "cascade": "cascade",
    "holo_m1": "holo M=1",
    "holo_m4": "holo M=4",
}


def parse_log(path):
    txt = open(path).read()
    row = {}
    for key, pat in ARMS:
        m = re.search(pat, txt)
        row[key] = float(m.group(1)) if m else None
    mv = re.search(r"V=(\d+) edges=(\d+) tri=(\d+)", txt)
    if mv:
        row["V"], row["edges"], row["tri"] = int(mv.group(1)), int(mv.group(2)), int(mv.group(3))
    mt = re.search(r"real ([0-9.]+)", txt)
    row["wall"] = float(mt.group(1)) if mt else None
    return row


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "/tmp/sl_bench"
    png = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/signed-link-bench.png"
    js = sys.argv[3] if len(sys.argv) > 3 else "reports/figures/signed-link-bench.json"

    datasets = ["epinions", "slashdot"]
    summary = {}
    for ds in datasets:
        rows = [parse_log(p) for p in sorted(glob.glob(f"{root}/{ds}_s*.log"))]
        rows = [r for r in rows if r.get("flat") is not None]
        if not rows:
            continue
        stats = {"n_seeds": len(rows)}
        for key, _ in ARMS:
            vals = np.array([r[key] for r in rows if r[key] is not None])
            stats[key] = {
                "median": float(np.median(vals)),
                "q25": float(np.percentile(vals, 25)),
                "q75": float(np.percentile(vals, 75)),
                "vals": vals.tolist(),
            }
        stats["V"] = rows[0].get("V")
        stats["edges"] = rows[0].get("edges")
        stats["wall_median"] = float(np.median([r["wall"] for r in rows if r.get("wall")]))
        summary[ds] = stats

    # Markdown table
    print("\n## Signed-link prediction — test AUROC (median over 5 seeds, IQR)\n")
    hdr = "| dataset | V | edges | " + " | ".join(LABELS[k] for k, _ in ARMS) + " |"
    print(hdr)
    print("|" + "---|" * (len(ARMS) + 3))
    for ds in datasets:
        if ds not in summary:
            continue
        s = summary[ds]
        cells = []
        for key, _ in ARMS:
            m, lo, hi = s[key]["median"], s[key]["q25"], s[key]["q75"]
            cells.append(f"{m:.4f} [{lo:.4f},{hi:.4f}]")
        print(f"| {ds} | {s['V']:,} | {s['edges']:,} | " + " | ".join(cells) + " |")
    print()

    # Grouped-bar figure
    keys = [k for k, _ in ARMS]
    x = np.arange(len(keys))
    w = 0.38
    colors = {"epinions": "#4575b4", "slashdot": "#d73027"}
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, ds in enumerate(datasets):
        if ds not in summary:
            continue
        s = summary[ds]
        med = [s[k]["median"] for k in keys]
        lo = [s[k]["median"] - s[k]["q25"] for k in keys]
        hi = [s[k]["q75"] - s[k]["median"] for k in keys]
        ax.bar(x + (i - 0.5) * w, med, w, yerr=[lo, hi], capsize=3,
               color=colors[ds], alpha=0.85, label=f"{ds} (V={s['V']:,})")
    # flat-baseline reference lines
    for ds in datasets:
        if ds in summary:
            ax.axhline(summary[ds]["flat"]["median"], color=colors[ds], ls=":", lw=1, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[k] for k in keys])
    ax.set_ylabel("test AUROC (median ± IQR, 5 seeds)")
    ax.set_ylim(0.6, 0.96)
    ax.set_title("CPML signed-link prediction — Epinions & Slashdot\n(dotted = each graph's L=1 flat baseline)")
    ax.legend(loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")

    os.makedirs(os.path.dirname(js), exist_ok=True)
    json.dump(summary, open(js, "w"), indent=2)
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
