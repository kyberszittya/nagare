---
title: Nagare handoff — HSiKAN parallel/benchmark FROZEN → auto-holonomy NEXT
date: 2026-07-16
kind: handoff
read_first: true
---

# Handoff: freeze HSiKAN result, start auto-holonomy

Purpose: bank the HSiKAN parallelization + PyTorch benchmark (done, committed,
assimilated) and orient the next session onto the real frontier — **auto-holonomy** —
so it starts on the correct task instead of re-deriving the framing at a full context
budget.

---

## Part 1 — FROZEN (do NOT redo)

**HSiKAN edge-chunk parallelization + vs-PyTorch CPU benchmark.** Complete.

- Commits: `04901aa` (parallelization), `0978563` (further scenarios). Pushed to
  `origin/main`.
- Finding: **F-HSIKAN-PAR** (`reports/framework/canonical_findings.json`).
- Component: HSiKAN → **v1.1**, default parallel (`canonical_components.json`).
- Report: `reports/2026-07-16-hsikan-parallel-vs-pytorch.md`; figures
  `hsikan-bench.png`, `hsikan-scenarios.png`; data `reports/figures/hsikan_pytorch_bench.json`.

**Settled results (do not re-measure):**
- `hsikan_forward`/`hsikan_backward` partition T hyperedges into `min(threads,T)` rayon
  chunks; public API + opaque cache unchanged; `chunk_parallel_matches_serial` gate.
- Single thread: 2.3× faster + 4–5× less RSS than CPU PyTorch (42× on tiny models).
- Matched threads: Nagare scales (7.3× @16 on Mac), PyTorch saturates ~6–8 and **thrashes
  33× at 32 threads** (over-subscription on d=16). Nagare degrades gracefully.
- Scaling flat in d (7.3–7.9× at ≤16 threads) → **not** d-bandwidth-bound below 16; the
  kato15 16→32 flattening cause is **unproven** (don't assert bandwidth).
- Deploy (forward-only, `hsikan_forward_chunked`): **16 MB** vs PyTorch 469 MB = 29× less;
  closed-form backward +29% vs PyTorch tape-replay +150%.

**Open (optional, low priority):** parallelize `clifford_fir`/`gomb_shell` with the same
edge/bank-chunk pattern so the full Gömb-Soma cascade inherits the win; isolate the 16→32
flattening with a 32-thread dim-sweep on kato15.

---

## Part 2 — NEXT: auto-holonomy (the frontier)

### What it is (from the thesis, memory `feedback-nagare-closed-form-thesis-not-gaps`)

Differentiation as **connection transport**, not a stored tape. Ambrose–Singer: holonomy
= the integral of curvature around a loop. The weight update is a **holonomy** computed by
parallel-transporting signal along the network path, driven by **global + instantaneous
entropy feedback** broadcast to all layers (DFA/neuromodulation style), not a sequential
one-path chain rule. Geometric-algebra realization: the connection lives in Clifford
algebra; rotors are the transport operators; hypergraph structure reduces to a simplicial
complex whose boundary operator is the discrete connection.

### What already EXISTS — reuse, do not reinvent (§6.1). Verified 2026-07-16:

| piece | symbol(s) | role in auto-holonomy |
|---|---|---|
| Cayley rotor | `cayley_rotor_forward/backward` | the transport operator (bivector→Spin(3), FD-verified) |
| Rotor holonomy | `rotor_holonomy_forward/backward` | ordered non-commutative loop product = the holonomy feature |
| Global entropy pool | `global_entropy_pool_forward/backward` | rotation-invariant covariance eigen-entropy = the feedback signal |
| Fused entropy update | `fused_entropy_update_forward/backward` | **already an entropy-driven weight update** — the seed of the broadcast rule |
| EvolventHead | `src/online.rs` (`Evolvent`) | closed-form one-pass online learner — proof non-tape learning works at a layer |
| Spine | `gomb_shell`, `clifford_fir`, `hsikan_*` | the deep net whose composition must become holonomy-native |

### The OPEN problem — compositional auto-holonomy

Each spine layer already has a hand-derived, tape-free backward — but they are composed by
**sequential chain rule** (tape-free ≠ holonomy-native; it's still backprop-shaped). The
frontier: replace the chain-rule composition with a **connection-transport composition +
global entropy broadcast**, so the deep net's update is a single holonomy driven by the
entropy readout, applied to all layers at once.

### De-risked plan (two steps — DISCRIMINATING TASK FIRST)

**Step 1 — the task where holonomy is NECESSARY (before any op).** F-HOLO-2 is the binding
lesson: on the last synthetic task, **trivial raw-field entropy (0.944) BEAT the deep
holonomy net (0.894)** — the task was solvable without holonomy, so it could not measure
holonomy's value. Do not repeat that. First build a controlled task where **trivial entropy
FAILS but holonomy SUCCEEDS**: per-node-varying twist that must be un-wound (path-dependent,
so a single global covariance can't see it), or compositional/multi-scale structure. Measure
the ceiling: the trivial baseline must be at chance and a privileged/oracle holonomy readout
must clear it. Only then is there a metric that can rank auto-holonomy. (Evaluation-metric
integrity + F-HOLO-2.)

**Step 2 — the compositional auto-holonomy op.** Given a task with a real gap, build the
holonomy-native update: entropy (`global_entropy_pool`) drives a broadcast transport update
composed of `cayley_rotor` transports and `rotor_holonomy` loop products across layers,
FD-gate every new backward, and A/B it against (a) the same net trained by the existing
sequential closed-form chain rule and (b) trivial entropy. Success = beats BOTH on the
Step-1 task, at a layer *and* composed.

### Success criterion

Auto-holonomy is real only if, on a task where trivial entropy is at chance, the
global-entropy-broadcast connection-transport update matches or beats the sequential
closed-form chain rule **and** does so with the global/instantaneous property (one
broadcast, not a path sweep). Anything less is a mechanism demo (cf. RotorMeshNet =
MECHANISM_DEMO), not a capability.

### Before coding

Write the plan bundle (`docs/plans/<date>-auto-holonomy/` — plan.{tex,pdf,tikz,mmd}) per
CLAUDE.md §2; this handoff is the scoping input, not the plan. Read
`feedback-nagare-closed-form-thesis-not-gaps`, F-HOLO-1/F-HOLO-2, and the system map
(artifact `cca528aa`) first.
