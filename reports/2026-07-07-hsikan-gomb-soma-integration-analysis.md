# HSiKAN / Gömb / Gömb-Soma onto Nagare — integration analysis & target selection

Created-at: 2026-07-07 18:33 JST
Scope: a discovery-first analysis before porting three architectures onto Nagare.
Companion plan: `nagare_github/docs/HANDOFF-hsikan-gomb-soma.md` (commit `bdb79de`).

## Summary

Before building, we consulted the existing HSiKAN/Gömb work in `hymeko_neuro`
(§6.5 #15). The consultation changed the target: **signed-KAN's advantage over
graph GNNs is on MIXED-ARITY hypergraphs, not on plain arity-2 signed-link.** This
both (a) recontextualises the 2026-07-07 pure-Nagare signed-link result as living
in a prevalence-confounded regime, and (b) points the whole HSiKAN/Gömb/Gömb-Soma
integration at the one regime where it is measured to win — which is also where
Nagare's higher-arity cycle pool is the natural substrate.

## Prior work consulted

`hymeko_neuro/assets/docs/archive/HSIKAN_GAP_CLOSING_PLAN.md` (2026-05) and the
`hymeko_neuro/hyperedge/{highway_signedkan,signedkan}.py` implementations.

### Finding 1 — arity-2 signed-link is prevalence-confounded
On Bitcoin (≈90% positive edges) a node-popularity heuristic (an MLP on node
identity, no graph) explains ~91% of test variance. On this regime KAN-family
architectures are only SGCN-competitive; the absolute AUROC is inflated by edge
prevalence. **Implication:** the 2026-07-07 pure-Nagare result (BTC-Alpha 0.904,
Slashdot 0.910, Epinions 0.951) is competitive but lives in this confounded
regime — it should not be read as a graph-structure win, and it should not be
re-chased.

### Finding 2 — mixed-arity is where signed-KAN beats SGCN
Measured (prior, PyTorch):

| model | Bitcoin-Alpha AUC | F1m |
|---|---:|---:|
| SGCN (no aux) | 0.870 | 0.678 |
| SGCN + balance | 0.886 | 0.715 |
| **HSiKAN mixed-arity + weak balance** | **0.934** | **0.773** |

HSiKAN Pareto-dominates SGCN+balance (**+0.048 AUC, +0.058 F1m**) — the one
direction "HSiKAN can claim that SGCN cannot": higher-arity signed n-tuples, which
graph-only GNNs cannot represent. Ablations on record: dropping the highway gate
→ AUC 0.57 (chance); dropping sign-conditioned branches → catastrophic. Both are
load-bearing and must be preserved in the port.

## Why Nagare is the right substrate

Nagare's datum is the **signed cycle / hyperedge pool**, which represents
higher-arity signed structure natively (`hymeko_graph` mixed-arity cycles +
`clifford_fir` holonomy). A closed-form HSiKAN over that pool targets exactly the
mixed-arity regime, and adds Nagare's standing win (competitive quality at
closed-form speed / 3–160× memory) on top of the accuracy differentiation.

## Integration plan (phased; each a closed-form Rust op, each gated by a test)

1. **HSiKAN core** → `src/ops/hsikan.rs`: signed-KAN basis (reuse the shipped
   Chebyshev in `ops/catmull_rom`) + highway gate + sign-conditioned branches,
   forward+backward pair (or local update). Test: match PyTorch HSiKAN on a
   mixed-arity toy.
2. **Gömb** shells: rotor/Clifford holonomy (`ops/cayley_rotor`, `clifford_fir`)
   aggregated into spherical shells. Test: shell aggregation vs flat pooling on
   mixed-arity.
3. **Gömb-Soma**: Soma quadtree spatial compression + entropy routing over the
   shells. Test: hold AUROC at lower compute.

**Bar to beat:** SGCN+balance ~0.886 (Bitcoin-Alpha mixed-arity); reproduce the
PyTorch HSiKAN 0.934 in closed-form Nagare, then claim the speed/memory win.

## Risks / gotchas

- Nagare is closed-form (no autograd): the KAN basis + highway need hand-derived
  backward, like the existing `ops/*`.
- Do **not** re-implement spline basis — reuse `ops/catmull_rom.rs` Chebyshev.
- Build the **mixed-arity toy / loader first**; there is no arity-2 shortcut.
- Do **not** re-benchmark the confounded arity-2 signed-link (known KAN ≈ SGCN).

## Status / next

Discovery + target selection done; the arity-2 signed-link chain is closed and
prevalence-caveated (incl. in the Kato/Katalin daily report). Next session starts
at **Phase 1** (HSiKAN core as a Nagare op). Repo at `main`, 45 tests green.
