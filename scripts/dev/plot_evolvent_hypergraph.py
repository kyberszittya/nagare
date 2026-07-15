#!/usr/bin/env python3
"""E3 — the pairwise item vs the hypergraph tensor (5-seed medians)."""
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

out = sys.argv[1] if len(sys.argv) > 1 else "reports/figures/evolvent-hypergraph.png"

# 5-seed medians (examples/evolvent_hypergraph.rs, fixed RNG). Stable to <0.01.
layouts = ["DISJOINT", "OVERLAP chain"]
pairwise = [0.161, 0.172]
dense = [0.997, 0.997]
block = [0.997, 0.997]
nnz_dense, nnz_block = 19600, 980

fig, (ax, axc) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [2, 1]})
x = range(len(layouts))
w = 0.26
ax.bar([i - w for i in x], pairwise, w, color="#adb5bd", label="dense-PAIRWISE (no 3-way)")
ax.bar([i for i in x], dense, w, color="#457b9d", label="dense-HYPEREDGE (O(d²))")
ax.bar([i + w for i in x], block, w, color="#e76f51", label="block-HYPEREDGE (O(d·w))")
for i in x:
    ax.text(i - w, pairwise[i] + 0.02, f"{pairwise[i]:.2f}", ha="center", fontsize=9)
    ax.text(i + w, block[i] + 0.02, f"{block[i]:.3f}", ha="center", fontsize=8, color="#c1440e")
ax.set_xticks(list(x))
ax.set_xticklabels(layouts)
ax.set_ylabel("test R²  (3-way hypergraph target)")
ax.set_ylim(0, 1.08)
ax.set_title("The pairwise item can't hold a genuine 3-way interaction\n(R² ≈ 0.15); the hyperedge tensor recovers it (0.997)")
ax.legend(fontsize=8, loc="center left")

# cost panel
axc.bar([0, 1], [nnz_dense, nnz_block], color=["#457b9d", "#e76f51"], width=0.6)
axc.set_xticks([0, 1])
axc.set_xticklabels(["dense\nO(d²)", "block\nO(d·w)"])
axc.set_ylabel("precision storage (nnz)")
axc.set_title(f"…at {nnz_dense // nnz_block}× less precision\n(matching dense accuracy)")
for i, v in enumerate([nnz_dense, nnz_block]):
    axc.text(i, v + 400, f"{v}", ha="center", fontsize=9)
axc.set_ylim(0, nnz_dense * 1.15)

fig.suptitle("E3 — pairwise precision vs the hypergraph-clique tensor (block-structured RLS)", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(out, dpi=140)
print("wrote", out)
