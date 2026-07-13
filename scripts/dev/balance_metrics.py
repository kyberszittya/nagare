#!/usr/bin/env python3
"""Balance-as-Z2-holonomy metric across the 4 signed graphs (Alpha, OTC,
Slashdot, Epinions) — the faithful, committed reconstruction of the (never-
committed) `holonomy_theorems.py` empirical panel.

The "new balance metric" is signed balance = Z2 cycle holonomy: a cycle is
BALANCED iff its edge-sign product is +1 (Cartwright-Harary). We measure, per
graph:
  - the negative-edge fraction q,
  - the BALANCED-TRIAD FRACTION over uniformly sampled triangles (the core
    scalar; the report's 200k-sample estimator), and
  - the balanced fraction at longer cycle lengths k=4 (holonomy at length),
and, as the theoretical panel, P(all cycles balanced) vs q on K_n (Monte Carlo)
— balance is rare for random signs yet real networks sit deep in it.

Self-check: the report's known triad fractions are Alpha 0.870, Slashdot 0.917,
Epinions 0.930; a faithful reconstruction should reproduce these (OTC is the new
fourth point).

Usage:
    python scripts/dev/balance_metrics.py <signed_data_dir> <out.png> <out.json> [n_tri]
"""
import json
import random
import sys
from collections import defaultdict


def load_signed(path):
    """Undirected signed graph: adjacency sets + sign lookup (first sign wins)."""
    adj = defaultdict(set)
    sign = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                continue
            p = [t for t in line.replace(",", " ").replace("\t", " ").split() if t]
            if len(p) < 3:
                continue
            u, v = int(p[0]), int(p[1])
            s = 1 if float(p[2]) > 0 else -1
            if u == v:
                continue
            key = (u, v) if u < v else (v, u)
            if key not in sign:
                sign[key] = s
                adj[u].add(v)
                adj[v].add(u)
    return adj, sign


def skey(a, b):
    return (a, b) if a < b else (b, a)


def balanced_triad_fraction(adj, sign, n_samples, rng, adjl=None):
    """Triangle-UNIFORM balance via wedge sampling (unbiased), both definitions.

    Sample an apex u with probability proportional to C(deg_u, 2), then two
    distinct random neighbours a,b; if a-b is an edge the wedge is a triangle.
    Each triangle is hit by its 3 wedges uniformly => uniform over triangles
    (unlike an edge-then-common-neighbour sample, which under-weights the dense
    internally-balanced clubs that dominate the true triangle count).

    Returns a dict: strong (CH, product +1: +++,+--), weak (Davis, also ---),
    the 4 triad-type fractions, and the closed-wedge count `tot`.
    """
    if adjl is None:
        adjl = {u: list(nb) for u, nb in adj.items()}
    verts = [u for u in adjl if len(adjl[u]) >= 2]
    if not verts:
        return None
    weights = [len(adjl[u]) * (len(adjl[u]) - 1) for u in verts]  # ∝ C(deg,2)
    c = {"ppp": 0, "ppm": 0, "pmm": 0, "mmm": 0}
    tot = 0
    batch = max(n_samples * 20, 100_000)
    while tot < n_samples:
        apexes = rng.choices(verts, weights=weights, k=batch)
        for u in apexes:
            nb = adjl[u]
            a = nb[rng.randrange(len(nb))]
            b = nb[rng.randrange(len(nb))]
            if a == b or b not in adj[a]:
                continue  # open wedge (not a triangle)
            signs = [sign[skey(u, a)], sign[skey(u, b)], sign[skey(a, b)]]
            nneg = sum(1 for s in signs if s < 0)
            c[["ppp", "ppm", "pmm", "mmm"][nneg]] += 1
            tot += 1
            if tot >= n_samples:
                break
    strong = (c["ppp"] + c["pmm"]) / tot       # product +1
    weak = (c["ppp"] + c["pmm"] + c["mmm"]) / tot  # Davis: also allow ---
    return {"strong": strong, "weak": weak, "tot": tot, **{k: v / tot for k, v in c.items()}}


def balanced_quad_fraction(adj, sign, n_samples, rng):
    """Fraction of sampled 4-cycles (u-v-w-x-u, chordless not required) balanced."""
    edges = list(sign.keys())
    bal, tot = 0, 0
    attempts = 0
    max_attempts = n_samples * 80
    while tot < n_samples and attempts < max_attempts:
        attempts += 1
        u, v = rng.choice(edges)
        w = rng.choice(tuple(adj[v])) if adj[v] else None
        if w is None or w == u:
            continue
        # x adjacent to both w and u, closing the 4-cycle
        closers = (adj[w] & adj[u]) - {v}
        if not closers:
            continue
        x = rng.choice(tuple(closers))
        if x in (u, v, w):
            continue
        prod = (sign[skey(u, v)] * sign[skey(v, w)] * sign[skey(w, x)] * sign[skey(x, u)])
        bal += 1 if prod > 0 else 0
        tot += 1
    return (bal / tot if tot else None), tot


def p_balanced_vs_q(ns, qs, n_signings, rng):
    """P(all triangles balanced) on K_n vs negative-edge fraction q (MC).

    On a complete graph, balance <=> all triangles balanced (triangles generate
    the cycle space), so this is the Z2-holonomy balance probability.
    """
    import itertools

    out = {}
    for n in ns:
        verts = list(range(n))
        tris = list(itertools.combinations(verts, 3))
        pe = list(itertools.combinations(verts, 2))
        row = []
        for q in qs:
            ok = 0
            for _ in range(n_signings):
                s = {e: (-1 if rng.random() < q else 1) for e in pe}
                if all(s[skey(a, b)] * s[skey(b, c)] * s[skey(a, c)] > 0 for a, b, c in tris):
                    ok += 1
            row.append(ok / n_signings)
        out[n] = row
    return out


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else \
        "/Users/kyberszittya/hakiko_ai_ws/03_implementation/nagare_data/signed"
    out_png = sys.argv[2] if len(sys.argv) > 2 else "reports/figures/balance-metrics.png"
    out_json = sys.argv[3] if len(sys.argv) > 3 else "reports/figures/balance-metrics.json"
    n_tri = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000

    graphs = [
        ("bitcoin-alpha", "soc-sign-bitcoinalpha.csv"),
        ("bitcoin-otc", "soc-sign-bitcoinotc.csv"),
        ("slashdot", "soc-sign-Slashdot090221.txt"),
        ("epinions", "soc-sign-epinions.txt"),
    ]
    rng = random.Random(0)
    summary = {}
    print(f"\n## Balance metric (signed balance = Z2 holonomy), {n_tri:,} sampled triads\n")
    print("| graph | V | E | neg-frac q | STRONG (CH) | WEAK (Davis) | +++ | +-- | ++- | --- |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name, fn in graphs:
        adj, sign = load_signed(f"{data_dir}/{fn}")
        V, E = len(adj), len(sign)
        neg = sum(1 for s in sign.values() if s < 0) / max(E, 1)
        bt = balanced_triad_fraction(adj, sign, n_tri, rng)
        bq, nq = balanced_quad_fraction(adj, sign, min(n_tri, 100_000), rng)
        summary[name] = {
            "V": V, "E": E, "neg_frac": neg,
            "triad": bt, "balanced_quad": bq, "n_quads": nq,
        }
        print(f"| {name} | {V:,} | {E:,} | {neg:.3f} | {bt['strong']:.4f} | {bt['weak']:.4f} | "
              f"{bt['ppp']:.3f} | {bt['pmm']:.3f} | {bt['ppm']:.3f} | {bt['mmm']:.3f} |")

    # Theoretical fragility panel.
    qs = [i / 20 for i in range(21)]
    ns = [3, 4, 5, 6]
    pbal = p_balanced_vs_q(ns, qs, 4000, rng)

    # ---- figure ----
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 5.4))
    colors = ["#4575b4", "#74add1", "#f46d43", "#d73027"]
    for n, c in zip(ns, ["#762a83", "#5aae61", "#f46d43", "#2166ac"]):
        axl.plot(qs, pbal[n], lw=2, marker="o", ms=3, label=f"K{n}", color=c)
    axl.set_title("Balance is fragile for random signs\nP(all cycles balanced) vs negative-edge fraction q")
    axl.set_xlabel("negative-edge fraction q")
    axl.set_ylabel("P(balanced)")
    axl.legend(title="complete graph")
    axl.grid(True, alpha=0.3)

    names = [g[0] for g in graphs]
    bt = [summary[n]["triad"]["strong"] for n in names]
    negs = [summary[n]["neg_frac"] for n in names]
    x = np.arange(len(names))
    axr.bar(x, bt, 0.6, color=colors, alpha=0.9)
    axr.axhline(0.5, color="0.6", ls="--", lw=1, label="chance (balanced=unbalanced)")
    for i, (b, q) in enumerate(zip(bt, negs)):
        axr.text(i, b + 0.01, f"{b:.3f}\nq={q:.2f}", ha="center", fontsize=9)
    axr.set_title("Real networks sit deep in the balanced regime\n(empirical balanced-triad fraction)")
    axr.set_xticks(x)
    axr.set_xticklabels(names, rotation=15)
    axr.set_ylabel("balanced-triad fraction")
    axr.set_ylim(0, 1.0)
    axr.legend(loc="lower right", fontsize=9)
    axr.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    summary["p_balanced"] = {"qs": qs, "ns": ns, "curves": pbal}
    json.dump(summary, open(out_json, "w"), indent=2)
    print(f"\nwrote {out_png}\nwrote {out_json}")


if __name__ == "__main__":
    main()
