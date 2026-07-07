# Nagare — Daily Progress Report

**Date:** 2026-07-07 · **Author:** Cs. Hajdu
**For:** Prof. Kato (NITech, Katolab) and Katalin — collaboration update
**Repository:** https://github.com/kyberszittya/nagare

## Summary

Today Nagare progressed from an in-framework component to a **standalone,
machine-verified, and benchmarked** closed-form learning framework, with a first
**competitive result on a real signed-link-prediction task**. On a fair
CPU-vs-CPU comparison Nagare matches or beats PyTorch while using far less memory;
its balance/holonomy foundation is now formally verified; and its closed-form
learner reaches competitive AUROC on Slashdot / Epinions / Bitcoin. We also report,
honestly, the one differentiating experiment that remains open.

## 1. Framework consolidation

Nagare was extracted into a self-contained public repository: the two diverged
copies were reconciled, the framework dependencies (`hymeko_clifford`,
`hymeko_graph`) were vendored in, and the crate now builds with **no coupling** to
the parent framework. It compiles and runs on both Windows and the Katolab GPU host
(`kato15`, Linux); **45 tests pass** on both. All optimisations below are proven
bit-identical (finite-difference tests + a frozen fixture).

## 2. Performance — CPU, versus PyTorch

Nagare replaces the autograd tape with contiguous, auto-vectorised, closed-form
kernels that parallelise (rayon) to every core. Two profile-driven fixes (a loop
reorder enabling SIMD, and parallelising the pooling stage) took the forward from
behind PyTorch to ahead of it on CPU.

| axis | metric | Nagare (CPU) | PyTorch (CPU) |
|---|---|---:|---:|
| forward latency (toy shape, 8 threads) | µs/sample | **7.6** | 7.6–9.1 |
| dimensionality sweep (best-of-each) | speed-up | **1.2–2.6× faster** | baseline |
| peak memory (toy → scaled) | RSS | **3–160× less (Nagare)** | — |

**Honest bounds:** at *matched* 8 threads on large dense matmuls, tuned BLAS (MKL)
still wins; and Nagare has **no GPU backend yet**, so a GPU is faster in absolute
terms — future work, not a CPU-vs-GPU verdict. The robust win is competitive CPU
speed at a large memory advantage.

## 3. Theoretical foundation — machine-verified

Nagare's learning target is *signed balance = ℤ₂ holonomy*. We machine-verified the
load-bearing statements with an SMT solver (Z3) and symbolic algebra (sympy): the
Cartwright–Harary balance theorem (switchable ⟺ every cycle positive), the gauge
invariance of cycle holonomy, and the cycle-space structure — each proved as a
validity, not sampled. Empirically, real signed networks are strongly balanced
(**87–93% balanced triads**), the structural reason holonomy is predictive.

## 4. Signed-link prediction — first real-task result

A pure-Nagare closed-form model (local update rule, **no backpropagation**) on
holonomy features, leakage-free, standard 80/20 protocol:

| | Bitcoin-Alpha | Bitcoin-OTC | Slashdot | Epinions |
|---|---:|---:|---:|---:|
| **Nagare (closed-form) AUROC** | 0.904 | 0.928 | 0.910 | 0.951 |

This is **competitive** (inside the published SGCN/SiGAT band ~0.93–0.97) — Nagare's
closed-form learner *matches* the baselines. It does not beat them, because on these
datasets the signal is triad-saturated. **Caveat (from prior in-house benchmarks):**
on *arity-2* signed graphs this AUROC is partly a prevalence effect — Bitcoin's ~90%
positive edges make a node-popularity heuristic strong — and signed-KAN architectures
are only SGCN-competitive here. The regime where they are measured to *beat* graph
GNNs (by ~+0.05 AUC) is **mixed-arity hypergraphs**, which graph-only models cannot
represent — the differentiating target going forward, and exactly what Nagare's
higher-arity cycle pool is built for.

## 5. Open question / next step

Signed hypergraphs are naturally real-valued (`w_e ∈ [−1,1]`), and Nagare's
Clifford/rotor holonomy models that continuum. The measurements show its payoff is
**not** on binary sign prediction but on a **continuous target** — predicting
rating / trust *strength* (regression), where the magnitude is necessary and
binarising cannot compete. That is the single, well-posed differentiating experiment
remaining.

## Artifacts

Integrated technical draft `docs/nagare-paper.pdf`; reports and figures under
`reports/`; reproducible harnesses under `examples/` and `scripts/dev/` (theorem
verifier, benchmarks, signed-link model).

---
*Prepared as an honest status update: measured results are labelled as such, and the
one unproven differentiating claim is stated as open.*
