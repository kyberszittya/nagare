#!/usr/bin/env python3
"""Preprocess the SNAP Reddit Hyperlinks body network into a signed integer
edgelist the cpml_signed_link / balance_metrics tooling can read.

Source (download first):
    curl -sSL -o reddit-body.tsv \
      https://snap.stanford.edu/data/soc-redditHyperlinks-body.tsv

The TSV is SOURCE_SUBREDDIT, TARGET_SUBREDDIT, POST_ID, TIMESTAMP,
LINK_SENTIMENT (+1/-1), PROPERTIES. Nodes are subreddit *names*; there are
multiple hyperlinks per pair. We map names -> ids and aggregate multi-edges to
the NET-sentiment sign per (src,tgt) pair (ties dropped).

Usage:
    python scripts/dev/prep_reddit_signed.py reddit-body.tsv soc-sign-reddit-body.csv
"""
import csv
import sys
from collections import defaultdict


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "reddit-body.tsv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "soc-sign-reddit-body.csv"
    net = defaultdict(int)
    with open(src) as f:
        r = csv.reader(f, delimiter="\t")
        next(r)  # header
        for row in r:
            if len(row) < 5:
                continue
            try:
                s = int(row[4])
            except ValueError:
                continue
            net[(row[0], row[1])] += s
    ids = {}

    def nid(name):
        if name not in ids:
            ids[name] = len(ids)
        return ids[name]

    n_pos = n_neg = n_tie = 0
    with open(dst, "w") as out:
        for (u, v), s in net.items():
            if s == 0:
                n_tie += 1
                continue
            sign = 1 if s > 0 else -1
            n_pos += sign > 0
            n_neg += sign < 0
            out.write(f"{nid(u)},{nid(v)},{sign}\n")
    tot = n_pos + n_neg
    print(f"reddit-body: V={len(ids)} pairs={tot} pos={n_pos} neg={n_neg} "
          f"neg-frac={n_neg / max(tot, 1):.3f} ties-dropped={n_tie} -> {dst}")


if __name__ == "__main__":
    main()
