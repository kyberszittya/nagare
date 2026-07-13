#!/usr/bin/env python3
"""Label-shuffle audit analysis: strict-protocol inner-core AUROC, real vs
shuffled training labels, per graph. A genuine structural learner under the
strict protocol must collapse toward chance (0.5) under shuffle; retention would
indicate leakage.

Reads /tmp/audit/results.tsv (ds<TAB>cond<TAB>seed<TAB>inner_auroc), writes a
markdown table, a JSON summary, and a grouped-bar figure.

Usage:
    python scripts/dev/analyze_shuffle_audit.py \
        /tmp/audit/results.tsv reports/figures/shuffle-audit.png reports/figures/shuffle-audit.json
"""
import json
import sys
from collections import defaultdict

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ORDER = ["bitcoin-alpha", "bitcoin-otc", "slashdot", "epinions"]


def main():
    tsv = sys.argv[1] if len(sys.argv) > 1 else "/tmp/audit/results.tsv"
    png = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/shuffle-audit.png"
    js = sys.argv[3] if len(sys.argv) > 3 else "reports/figures/shuffle-audit.json"

    vals = defaultdict(list)  # (ds, cond) -> [auroc]
    for line in open(tsv):
        parts = line.strip().split("\t")
        if len(parts) < 4 or not parts[3]:
            continue
        ds, cond, _seed, a = parts[0], parts[1], parts[2], float(parts[3])
        vals[(ds, cond)].append(a)

    def stat(ds, cond):
        v = np.array(vals.get((ds, cond), []))
        if v.size == 0:
            return None
        return {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                "q75": float(np.percentile(v, 75)), "n": int(v.size), "vals": v.tolist()}

    summary = {}
    print("\n## Label-shuffle audit — strict-protocol inner-core AUROC (median/IQR, 5 seeds)\n")
    print("| graph | real | shuffled train | drop | verdict |")
    print("|---|---|---|---|---|")
    for ds in ORDER:
        r, s = stat(ds, "real"), stat(ds, "shuffle")
        if r is None or s is None:
            continue
        drop = r["median"] - s["median"]
        # structural if shuffle collapses near chance AND real is high
        verdict = "STRUCTURAL (no leakage)" if s["median"] < 0.6 and r["median"] > 0.8 else \
                  ("LEAKAGE SUSPECTED" if s["median"] > 0.7 else "partial")
        summary[ds] = {"real": r, "shuffle": s, "drop": drop, "verdict": verdict}
        print(f"| {ds} | {r['median']:.4f} [{r['q25']:.4f},{r['q75']:.4f}] | "
              f"{s['median']:.4f} [{s['q25']:.4f},{s['q75']:.4f}] | {drop:+.4f} | {verdict} |")
    print()

    # figure
    ds_present = [d for d in ORDER if d in summary]
    x = np.arange(len(ds_present))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    rmed = [summary[d]["real"]["median"] for d in ds_present]
    rerr = [[summary[d]["real"]["median"] - summary[d]["real"]["q25"] for d in ds_present],
            [summary[d]["real"]["q75"] - summary[d]["real"]["median"] for d in ds_present]]
    smed = [summary[d]["shuffle"]["median"] for d in ds_present]
    serr = [[summary[d]["shuffle"]["median"] - summary[d]["shuffle"]["q25"] for d in ds_present],
            [summary[d]["shuffle"]["q75"] - summary[d]["shuffle"]["median"] for d in ds_present]]
    ax.bar(x - w / 2, rmed, w, yerr=rerr, capsize=3, color="#1a9850", alpha=0.9, label="real labels")
    ax.bar(x + w / 2, smed, w, yerr=serr, capsize=3, color="#d73027", alpha=0.9, label="shuffled train labels")
    ax.axhline(0.5, color="0.4", ls="--", lw=1.2, label="chance (0.5)")
    ax.set_xticks(x)
    ax.set_xticklabels(ds_present, rotation=15)
    ax.set_ylabel("strict-protocol inner-core AUROC (median ± IQR)")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Label-shuffle audit under the STRICT protocol\n"
                 "real → high, shuffled train → chance = structural learning, no leakage\n"
                 "(a transductive/leaky model would keep its score under shuffle)")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=150)
    print(f"wrote {png}")

    json.dump(summary, open(js, "w"), indent=2)
    print(f"wrote {js}")


if __name__ == "__main__":
    main()
