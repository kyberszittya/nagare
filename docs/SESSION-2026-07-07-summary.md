# Nagare — Session Summary (2026-07-07)

Self-contained catch-up for a new session. Repo: `github.com/kyberszittya/nagare`
(local `d:\hakiko_ai_ws\03_implementation\nagare_github\`), branch `main`, all work
pushed. Framework parent: `d:\hakiko_ai_ws\03_implementation\hymeko_framework_rust\`
(the crate `hymeko_nagare/` there is **FROZEN** — do not develop it). Compute host:
`kato15` (Katolab, `ssh kato15`, Linux 32c + RTX 6000, torch env `~/envs/hymeko`,
cargo at `~/.cargo`; tcsh shell → drive via `bash -ls` + heredoc).

## What Nagare is
Closed-form ML framework in Rust for signed hypergraphs: `(forward,backward)`
kernel pairs (no autograd tape), struct-of-arrays cycle pool, multivector/Clifford
gradients, rayon MapReduce. Learning target: **signed balance = ℤ₂ holonomy**.

## This session, in order (each committed on `main`)

1. **Reconciliation + detachment.** Two diverged trees existed; made `nagare_github`
   the canonical superset — ported the framework's `project_alpha_mix` kernel + 5
   ops + frozen seed-53 fixture + `gather_batch` (bit-identical), kept GitHub's
   `run_stress_ablation` harness. Then **vendored** `hymeko_clifford` + `hymeko_graph`
   into `vendor/` → self-contained; builds on kato15 from `vendor/` alone.
   Platform-aware fixture guard (Windows/Linux libm ULP differences). **45 tests
   green** on both OSes.
2. **Order-shuffle science.** New `StressKind::{Shuffled,ShuffledFewShot}` +
   `tests/order_shuffle.rs`. Verdict (3-seed): the **fitted projection gate does NOT
   robustly beat the constant gate** (the 0.2506 result was seed-53 only); holonomy
   signal is spiral-only and gate-independent.
3. **Forward optimization (profile-driven).** Profiler `examples/forward_profile.rs`.
   The **ikj/SAXPY reorder** (strided scalar reduction → contiguous broadcast-FMA,
   bit-identical) in `fused_entropy_update` + `linear_forward`: fused 2.9×, whole
   forward 2.1×. Then **parallelised the pool** (Amdahl). Result on kato15: forward
   parity **22.5 → 7.6 µs/sample**, Nagare crosses under PyTorch.
4. **Nagare vs PyTorch (fair).** Best-of-each CPU: **Nagare beats PyTorch-CPU 1.2–2.6×
   across a dimensionality sweep** (MKL oversubscribes at 32t; rayon scales). GPU
   (PyTorch-CUDA) dominates 20–80× — **Nagare has no GPU backend yet** (future, not a
   loss). Memory **160× (toy) → 3.2× (scale)**, always Nagare's favour. NOTE: pool
   parallelisation helps only hidden ≥ ~16 — the shipped 7-channel learner pool is
   kept SERIAL (parallelising it regressed 16%).
5. **Theorems machine-verified** (`scripts/dev/holonomy_theorems.py`, Z3 + sympy):
   Cartwright–Harary (switchable ⟺ all cycles positive) as SMT validity; gauge
   invariance; cycle-space dim = m−n+1 over GF(2). Balance structure figure: real
   nets are 87–93% balanced.
6. **Signed-link prediction** (data on kato15 `/tmp/hajdu/signed/`; Slashdot 549k,
   Epinions 841k, Bitcoin-Alpha/OTC): validated leakage-free harness; baseline
   ceiling Slashdot 0.912 / Epinions 0.951 / BTC ~0.91. Signed holonomy **lifts**
   AUROC (+0.01–0.02). Nonlinearity **negative** (MLP < linear). Deeper holonomy
   (A²+A³+A⁴) **triad-dominated** (marginal). Weighted [−1,1] **worse** than ±1 for
   the binary sign target. **Pure-Nagare Rust model** `examples/signed_link.rs`
   (closed-form local update, no backprop): AUROC **BTC-Alpha 0.904, BTC-OTC 0.928,
   Slashdot 0.910, Epinions 0.951** — competitive, matches baselines.
7. **Docs:** integrated working paper `docs/nagare-paper.{tex,pdf}` (architecture +
   speed + verified theory + evidence chain); comprehensive review `docs/nagare-review.tex`;
   **daily report for Kato + Katalin** `reports/2026-07-07-nagare-daily-report.{pdf,md}`.

## Current honest state
- **Proven:** standalone framework, verified balance/holonomy theory, competitive
  CPU speed + big memory win, competitive signed-link AUROC via closed-form learning.
- **Caveat:** the arity-2 signed-link AUROC is **prevalence-confounded** (node-popularity
  MLP ~0.91 on Bitcoin's 90%-positive edges) — competitive, but not a graph-structure
  win. Don't re-chase arity-2.
- **Open (differentiating):** (a) rotor holonomy on a *continuous strength-regression*
  target (where [−1,1] magnitude is necessary); (b) the mixed-arity hypergraph regime.

## NEXT SESSION (Phase 1): HSiKAN → Gömb → Gömb-Soma onto Nagare
Plan: `docs/HANDOFF-hsikan-gomb-soma.md`. Analysis: `reports/2026-07-07-hsikan-gomb-soma-integration-analysis.md`.
- **Critical prior finding** (consulted `hymeko_neuro/.../HSIKAN_GAP_CLOSING_PLAN.md`):
  signed-KAN beats SGCN only on **MIXED-ARITY** hypergraphs (+0.048 AUC, 0.886→0.934
  on Bitcoin-Alpha), not arity-2 signed-link. Ablations: highway gate + sign-conditioned
  branches are load-bearing (keep both).
- **Start:** port `hymeko_neuro/hyperedge/{highway_signedkan,signedkan}.py` (Chebyshev
  basis + highway + sign branches) → closed-form Nagare op `src/ops/hsikan.rs`
  (forward+backward pair; reuse `ops/catmull_rom.rs` Chebyshev). Build a **mixed-arity
  toy first**. Bar to beat: SGCN+bal ~0.886 → reproduce 0.934 in closed-form, then
  claim the speed/memory win. Then Gömb shells (`cayley_rotor`/`clifford_fir`), then
  Gömb-Soma (quadtree compression + entropy routing).

## Repo landmarks
- Kernels: `src/ops/*` (all FD-tested). Runtime: `src/runtime.rs` (`NagareRuntime`
  cycle-pool). Learner: `src/{features,pooling,projection,learner}.rs`. Vendored:
  `vendor/{hymeko_clifford,hymeko_graph}`.
- Examples: `forward_profile`, `scaling_bench`, `auroc_eval`, `signed_link`, `toy_compare`.
- Python harnesses (framework `scripts/dev/`): `signed_link_{baseline,holonomy,nonlinear,deep,weighted}.py`,
  `holonomy_theorems.py`, `scaling_bench_torch.py`.
- Key commits: `162a5c7` consolidation → … → `6a4ef47` integration analysis (latest).
- Rules: closed-form/no-backprop; bit-identical optimisations proven; multi-seed;
  §6.1 consult-before-build; don't re-chase confounded arity-2.
