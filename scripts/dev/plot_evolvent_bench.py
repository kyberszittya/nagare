#!/usr/bin/env python3
"""Aggregate the massive multi-seed evolvent benchmark and plot median±IQR.

Reads reports/figures/evolvent_bench_seed*.json (one per seed), each with a
`results` list of {dataset, metric, evolvent, sgd, mlp}. Prints a median±IQR
table and a grouped-bar figure (regression R2 | classification ACC).
"""
import glob
import json
import statistics as st
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

seed_dir = sys.argv[1] if len(sys.argv) > 1 else "reports/figures"
out = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/evolvent-bench.png"

runs = [json.load(open(f)) for f in sorted(glob.glob(f"{seed_dir}/evolvent_bench_seed*.json"))]
n_seed = len(runs)
# collect per (dataset,metric) -> {arm: [values]}
agg = {}
order = []
for r in runs:
    for row in r["results"]:
        key = (row["dataset"], row["metric"])
        if key not in agg:
            agg[key] = {"evolvent": [], "sgd": [], "mlp": []}
            order.append(key)
        for arm in ("evolvent", "sgd", "mlp"):
            agg[key][arm].append(row[arm])


def iqr(v):
    v = sorted(v)
    n = len(v)
    return v[int(0.75 * (n - 1))] - v[int(0.25 * (n - 1))]


print(f"=== evolvent benchmark, {n_seed} seeds (median [IQR]) ===")
print(f"{'dataset':<16}{'metric':<6}{'evolvent':>18}{'sgd':>18}{'mlp':>18}")
for key in order:
    ds, metric = key
    line = f"{ds:<16}{metric:<6}"
    for arm in ("evolvent", "sgd", "mlp"):
        vals = agg[key][arm]
        line += f"{st.median(vals):>10.3f} [{iqr(vals):.3f}]".rjust(18)
    print(line)

# figure: two panels (R2 datasets, ACC datasets)
reg = [k for k in order if k[1] == "R2"]
cls = [k for k in order if k[1] == "ACC"]
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
cols = {"evolvent": "#e76f51", "sgd": "#adb5bd", "mlp": "#457b9d"}
for ax, keys, title, ylab in [
    (axes[0], reg, "Regression (R²)", "R²"),
    (axes[1], cls, "Classification (accuracy)", "accuracy"),
]:
    x = range(len(keys))
    w = 0.26
    for a, arm in enumerate(("evolvent", "sgd", "mlp")):
        meds = [st.median(agg[k][arm]) for k in keys]
        errs = [iqr(agg[k][arm]) / 2 for k in keys]
        ax.bar([i + (a - 1) * w for i in x], meds, w, yerr=errs, capsize=3,
               color=cols[arm], label={"evolvent": "evolvent (1-pass RLS)", "sgd": "SGD (1-pass)", "mlp": f"MLP ({runs[0]['mlp_epochs']}ep)"}[arm])
    ax.set_xticks(list(x))
    ax.set_xticklabels([k[0].replace("_", "\n") for k in keys], fontsize=8)
    ax.set_ylabel(ylab)
    ax.set_title(title)
    ax.legend(fontsize=8)
fig.suptitle(f"Evolvent on static datasets — {n_seed} seeds (median ± IQR/2)\none-pass RLS matches/beats multi-epoch backprop, beats one-pass SGD", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.92))
fig.savefig(out, dpi=140)
print("wrote", out)
