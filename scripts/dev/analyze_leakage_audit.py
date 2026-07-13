#!/usr/bin/env python3
"""Full 2x2 leakage audit: {strict, transductive} x {real, shuffled-train}
inner-core AUROC per graph. The label-shuffle audit's headline:

  - STRICT: real high, shuffle -> chance  (no leakage; the honest protocol)
  - TRANSDUCTIVE: real high, shuffle RETAINS high (test-edge signs leak into the
    features) -> retention = leakage fraction.

Reads the strict TSV (cond in {real, shuffle}) and the transductive TSV (cond in
{transd-real, transd-shuffle}); writes a markdown table with the leakage
fraction, a JSON summary, and a 4-bar-per-graph figure.

Usage:
    python scripts/dev/analyze_leakage_audit.py \
        /tmp/audit/results.tsv /tmp/audit/results_transd.tsv \
        reports/figures/leakage-audit.png reports/figures/leakage-audit.json
"""
import json
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["bitcoin-alpha", "bitcoin-otc", "slashdot", "epinions", "reddit-body"]
# canonical condition key -> (label, source-cond-in-tsv)
REMAP = {"real": "strict-real", "shuffle": "strict-shuffle",
         "transd-real": "transd-real", "transd-shuffle": "transd-shuffle"}
CONDS = ["strict-real", "strict-shuffle", "transd-real", "transd-shuffle"]


def load(tsv, vals):
    for line in open(tsv):
        p = line.strip().split("\t")
        if len(p) < 4 or not p[3]:
            continue
        vals[(p[0], REMAP.get(p[1], p[1]))].append(float(p[3]))


def med_iqr(v):
    a = np.array(v)
    return {"median": float(np.median(a)), "q25": float(np.percentile(a, 25)),
            "q75": float(np.percentile(a, 75)), "n": int(a.size)}


def main():
    strict_tsv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit/results.tsv"
    transd_tsv = sys.argv[2] if len(sys.argv) > 2 else "/tmp/audit/results_transd.tsv"
    png = sys.argv[3] if len(sys.argv) > 3 else "reports/figures/leakage-audit.png"
    js = sys.argv[4] if len(sys.argv) > 4 else "reports/figures/leakage-audit.json"

    vals = defaultdict(list)
    load(strict_tsv, vals)
    load(transd_tsv, vals)

    summary = {}
    print("\n## Full leakage audit — inner-core AUROC, {strict,transductive} x {real,shuffle} (median, 5 seeds)\n")
    print("| graph | strict real | strict shuffle | transd real | transd shuffle | leakage frac |")
    print("|---|---|---|---|---|---|")
    for ds in ORDER:
        if any((ds, c) not in vals for c in CONDS):
            continue
        s = {c: med_iqr(vals[(ds, c)]) for c in CONDS}
        # leakage fraction: how much of the transductive score survives shuffle,
        # above chance: (transd_shuffle - 0.5) / (transd_real - 0.5).
        tr_real, tr_shuf = s["transd-real"]["median"], s["transd-shuffle"]["median"]
        leak = (tr_shuf - 0.5) / max(tr_real - 0.5, 1e-6)
        summary[ds] = {**{c: s[c] for c in CONDS}, "leakage_frac": leak}
        print(f"| {ds} | {s['strict-real']['median']:.4f} | {s['strict-shuffle']['median']:.4f} | "
              f"{tr_real:.4f} | {tr_shuf:.4f} | **{leak*100:.0f}%** |")
    print()

    ds_present = [d for d in ORDER if d in summary]
    x = np.arange(len(ds_present))
    w = 0.2
    colors = {"strict-real": "#1a9850", "strict-shuffle": "#91cf60",
              "transd-real": "#d73027", "transd-shuffle": "#fc8d59"}
    labels = {"strict-real": "strict / real", "strict-shuffle": "strict / shuffle",
              "transd-real": "transductive / real", "transd-shuffle": "transductive / shuffle"}
    fig, ax = plt.subplots(figsize=(12, 6))
    for i, c in enumerate(CONDS):
        med = [summary[d][c]["median"] for d in ds_present]
        lo = [summary[d][c]["median"] - summary[d][c]["q25"] for d in ds_present]
        hi = [summary[d][c]["q75"] - summary[d][c]["median"] for d in ds_present]
        ax.bar(x + (i - 1.5) * w, med, w, yerr=[lo, hi], capsize=2,
               color=colors[c], alpha=0.9, label=labels[c])
    ax.axhline(0.5, color="0.35", ls="--", lw=1.2, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_present, rotation=12)
    ax.set_ylabel("inner-core AUROC (median ± IQR, 5 seeds)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Leakage audit: strict vs transductive under label shuffle\n"
                 "strict/shuffle -> chance (honest); transductive/shuffle RETAINS (test-sign leakage)")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")
    json.dump(summary, open(js, "w"), indent=2)
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
