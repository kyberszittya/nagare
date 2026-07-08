#!/usr/bin/env python
"""Plot the HSiKAN mixed-arity entropy-vs-constant multi-seed comparison (Phase 1c).

Reads the per-seed final-BCE lines emitted by `tests/hsikan_multiseed.rs`
(`SEED <s> entropy <e> constant <c>`) and renders a paired scatter + medians —
honest about the overlap. Run (no permanent install):
  cargo test --test hsikan_multiseed -- --nocapture | grep '^SEED' > /tmp/multiseed_data.txt
  uv run --with matplotlib --with numpy python scripts/dev/plot_multiseed.py /tmp/multiseed_data.txt
"""
from __future__ import annotations

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

data_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/multiseed_data.txt"
ent, con = [], []
for line in open(data_path):
    parts = line.split()
    if len(parts) >= 6 and parts[0] == "SEED":
        ent.append(float(parts[3]))
        con.append(float(parts[5]))
ent, con = np.array(ent), np.array(con)
wins = int((ent < con).sum())

fig, ax = plt.subplots(figsize=(6.0, 4.5))
for e, c in zip(ent, con):
    ax.plot([1, 2], [e, c], color="0.75", lw=0.8, zorder=1)
ax.scatter([1] * len(ent), ent, color="#2aa76a", zorder=3, s=28,
           label=f"entropy (median {np.median(ent):.3f})")
ax.scatter([2] * len(con), con, color="#c85a33", zorder=3, s=28,
           label=f"constant (median {np.median(con):.3f})")
ax.plot([0.83, 1.17], [np.median(ent)] * 2, color="#177a4a", lw=2.5)
ax.plot([1.83, 2.17], [np.median(con)] * 2, color="#9c3f1f", lw=2.5)
ax.set_xlim(0.6, 2.4)
ax.set_xticks([1, 2])
ax.set_xticklabels(["entropy gate", "constant gate"])
ax.set_ylabel("final BCE  (lower is better)")
ax.set_title(
    f"HSiKAN mixed-arity: entropy vs constant gate ({len(ent)} seeds)\n"
    f"entropy < constant in {wins}/{len(ent)} seeds (paired); IQRs overlap"
)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
out = "reports/figures/hsikan-multiseed-entropy-vs-constant.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}  (entropy<constant {wins}/{len(ent)})")
