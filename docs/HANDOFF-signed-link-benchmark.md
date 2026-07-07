# HANDOFF → Codex: the decisive Nagare benchmark (signed-link prediction AUROC)

Written 2026-07-07 by the previous agent. Self-contained: you do not need that
session's context. Repo: `nagare_github` = `github.com/kyberszittya/nagare`,
branch `main`, base commit `162a5c7`. 45 tests green; standalone (vendored
`hymeko_clifford` + `hymeko_graph`). Read `CLAUDE.md`/`CODEX.md` in the framework
repo for the operating contract (multi-seed, proper baselines, report, no
over-claim).

## The one question this answers

Does **closed-form local learning + holonomy** hold a **competitive AUROC on a
real signed-graph task** against proper GNN/backprop baselines? This is the
direct test of the framework's *central* thesis — signed balance = Z₂ holonomy —
which the moons/spiral/xor toys do **not** test (they have no signed-cycle
structure). Everything else measured so far (7.6 µs forward, 160× memory, AUROC
1.0 on toys) is systems/toy evidence; **this experiment decides whether Nagare is
a genuinely different *learning* mechanism or just a fast lean forward.**

**Decision rule (be honest either way):** if Nagare's AUROC is within noise of —
or above — SGCN/SiGAT/backprop-MLP on Slashdot **and** Epinions (multi-seed
median/IQR), that is the big result. If it is clearly below, report it plainly;
the framework's value then stays "fast/lean CPU forward + clean systems," not
"competitive novel learning." Do **not** over-claim before the numbers exist
(measure the baseline ceiling first).

## What already exists (use it, don't rebuild — §6.1)

- **The model:** `NagareRuntime` (`src/runtime.rs`): `new(k, d_in, d_out, lr,
  seed)`, `predict(batch, features, n_vertices) -> logits`, `step(...)`. Pipeline
  = `clifford_fir_forward → scatter_mean_forward → linear → bce_with_logits`,
  with closed-form backward + Adam. This is the cycle-pool signed model; the
  `runtime_training` tests pass.
- **Signed-graph + cycles** (vendored `hymeko_graph`): `SignedGraph`, `Sign`,
  `TopKCyclesBatch`, `enumerate_top_k_cycles*` (+ per-vertex/parallel/adaptive
  variants), balance pruners `CartwrightHararyPruner` / `DavisWeakBalancePruner`
  / `BalanceMode`. Cycles carry the holonomy/balance signal.
- **Kernels/metrics:** `ops/` (all with FD-tested backward), `metrics::{cross_entropy,
  softmax2, clifford_probability_error}`. AUROC helper exists in
  `examples/auroc_eval.rs` (Mann-Whitney, tie-aware) — **promote it to
  `metrics::auroc`**.

## Gaps to fill (the actual task)

1. **Dataset loader → `SignedGraph`.** Slashdot (`soc-Slashdot090221` /
   Epinions `soc-Epinions1` from SNAP, signed variants) + wiki-Elec if you want a
   third. Use the **standard sign-prediction protocol** from the signed-link
   literature (Leskovec-Huttenlocher-Kleinberg 2010; SGCN Derr et al. 2018; SiGAT
   Huang et al. 2019): a train/test **edge** split (typical 80/20, remove test
   edge signs), predict the sign of held-out edges. Match a published split so
   the comparison is fair.
2. **Edge-sign head.** `NagareRuntime` currently emits per-vertex logits; signed
   link prediction needs `sign(u,v)`. Form an edge feature from the two
   endpoints' cycle-pool embeddings (concat / Hadamard / signed-difference), then
   a closed-form readout → BCE on edge sign. Keep it closed-form/local to test
   the thesis (a backprop head would defeat the purpose).
3. **AUROC** on held-out test edges (+ F1/accuracy), **5 seeds**, median/IQR.
4. **Proper baselines (not strawmen), cite or reproduce:**
   - **SGCN** and/or **SiGAT** (reference impls exist) — the signed-GNN state of
     practice.
   - **Backprop MLP** and **logistic regression** on the standard 23 signed
     features (FExtra, Leskovec 2010) — same split, same features.
   - Cite published Slashdot/Epinions AUROC where you use it as an anchor.

## Rigor (from the contract)

- Measure the **baseline ceiling first** (don't optimize under an un-anchored
  metric). Multi-seed median/IQR — single seed is a point estimate. Same split +
  same features for feature-based baselines. Distinguish measured / inferred /
  hypothesis in the report. Emit the result **numerically + plotted** (AUROC bars
  with IQR, Nagare vs baselines × {Slashdot, Epinions}). Write
  `reports/<date>-nagare-signed-link-auroc.md`.
- Holonomy caveat to keep in mind: the toy "spiral-only" result is a toy artifact
  — signed-link prediction is where holonomy (balance) is the actual structure,
  so this is the fair test of the leading idea, not a rerun.

## Provenance / state

- Base commit `162a5c7`. As of writing there were ~7 uncommitted session
  artifacts (the `scaling_bench` / `auroc_eval` examples + 2026-07-06/07 reports)
  — these should be committed before you branch (the previous agent was to do so;
  verify `git status` is clean and start from `main`).
- Prior context: a "Slashdot 5-seed AUC parity" was a frozen NAGARE-plan
  prerequisite never executed. This is finally it.
- Compute: kato15 (`ssh kato15`, tcsh — drive via `bash -ls` + heredoc; cargo at
  `~/.cargo`, torch env `~/envs/hymeko`) is available for baselines if needed.

## Baseline ceiling — MEASURED 2026-07-07 (harness validated)

Data downloaded (kato15 `/tmp/hajdu/signed/`) and the harness validated with a
leakage-free signed-degree logistic (`reports/2026-07-07-nagare-signed-link-progress.md`):

| dataset | pos frac | baseline AUROC (3-seed) |
|---|---:|---:|
| Bitcoin-Alpha | 0.936 | ~0.91 |
| Slashdot | 0.774 | ~0.912 |
| Epinions | 0.853 | ~0.951 |

These match the literature → loader/split/no-leakage-features/AUROC are correct.
**Nagare's target band:** floor ~0.91–0.95 (above), ceiling ~0.93–0.97 (published
SGCN/SiGAT). Land in/above on multi-seed median to be "competitive." No Nagare
number exists yet.

## Deliverable

A report with the AUROC table (Nagare vs SGCN/SiGAT/backprop-MLP/logistic,
5-seed median/IQR, on Slashdot + Epinions), the plot, and the honest verdict
against the decision rule above.
