# HANDOFF (continue tomorrow): HSiKAN → Gömb → Gömb-Soma onto Nagare

Written 2026-07-07 (end of session). Continue here next session. This is a
**multi-session architectural port**: reimplement three existing PyTorch
architectures as Nagare *closed-form Rust kernels* (no backprop, MapReduce).

## The target is MIXED-ARITY, not plain signed-link (critical, from prior work)

Prior in-house benchmarks (`hymeko_neuro/assets/docs/archive/HSIKAN_GAP_CLOSING_PLAN.md`)
already established:
- On **arity-2 signed-link** (Bitcoin/Slashdot), signed-KAN is only SGCN-competitive
  and the AUROC is **prevalence-confounded** (a node-popularity MLP explains ~91% on
  Bitcoin's 90%-positive edges). *This is the regime the 2026-07-07 pure-Nagare
  0.90–0.95 result lives in.* Do NOT re-chase it.
- On **mixed-arity hypergraphs + weak balance**, HSiKAN **beats SGCN by ~+0.048 AUC**
  (Bitcoin-Alpha 0.934 vs SGCN+bal 0.886). That is the one direction "HSiKAN can
  claim that SGCN cannot" — higher-arity signed n-tuples, which graph-only GNNs can't
  represent, and which Nagare's cycle/hyperedge pool represents natively.

**So: build for the mixed-arity regime.** That is where the differentiation is real
and where Nagare is the natural substrate.

## Code to port (do NOT rebuild — §6.1)

- **HSiKAN core:** `hymeko_neuro/hyperedge/highway_signedkan.py` (132 LOC) +
  `hyperedge/signedkan.py` (`MultiLayerSignedKAN`, the basis + signed branches).
  Highway gate on the inner spline; B-spline/Chebyshev basis; sign-conditioned
  branches. (Ablations on record: dropping highway → AUC 0.57 (chance); dropping
  sign-conditioned branches → catastrophic. Keep both.)
- **Gömb:** `hymeko_neuro/eval/interpret/gomb_signature.py` + the gomb models/
  benchmarks (`eval/benchmarks/*gomb*`); spherical/Clifford shells.
- **Soma:** the quadtree spatial compression (VOC quadtree experiments) + the
  entropy pool/gate (already in Nagare as `pooling`/entropy gate).

## Phased plan (each a closed-form Rust op; each gated by a discriminating test)

| phase | build | Nagare substrate | discriminating test |
|---|---|---|---|
| 1 | **HSiKAN core** as a closed-form Rust op: signed-KAN basis (Chebyshev, reuse `ops/catmull_rom` Chebyshev) + highway gate + sign-conditioned branches; closed-form/local update | over the cycle/hyperedge pool (`hymeko_graph` mixed-arity cycles) | does it match the PyTorch HSiKAN on a mixed-arity toy? |
| 2 | **Gömb** shells: rotor/Clifford holonomy (`ops/cayley_rotor`, `clifford_fir`) aggregated into spherical shells | rotor holonomy → shell readout | does shell aggregation beat flat pooling on mixed-arity? |
| 3 | **Gömb-Soma**: Soma quadtree spatial compression + entropy routing over the shells | the cognitive readout stack | does compression+routing hold AUROC at lower compute? |

Target metric: mixed-arity signed-hypergraph link/tuple-sign AUROC vs SGCN(+balance).
The bar to beat is SGCN+balance ~0.886 (Bitcoin-Alpha mixed-arity); HSiKAN reached
0.934 in PyTorch — reproduce that in closed-form Nagare, then claim the speed/memory
win on top.

## Rigor / gotchas

- Nagare = closed-form, no autograd. The KAN basis + highway must have hand-derived
  backward (like the existing `ops/*`), or use the local update rule.
- Reuse the Chebyshev basis already in `ops/catmull_rom.rs` (train-CR / deploy-Chebyshev
  is a shipped Nagare pattern) — do not re-implement spline basis.
- Keep highway + sign-conditioned branches (ablations show both are load-bearing).
- Build the mixed-arity toy / dataset loader first (there is no arity-2 shortcut here).
- kato15 available for baselines (`ssh kato15`, torch env `~/envs/hymeko`).

## Where we are

Repo at `main`; 45 tests green; the signed-link (arity-2) evidence chain is closed
(competitive, prevalence-caveated). Start Phase 1: port `signedkan.py`'s basis +
highway into `nagare_github/src/ops/hsikan.rs` with a forward+backward pair and an
in-module test, then a mixed-arity toy.
