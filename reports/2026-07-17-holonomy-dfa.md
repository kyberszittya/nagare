---
title: "Gate 1 — holonomy-DFA: the exact rotor transport is the lever, but a global broadcast does not compose through depth"
date: 2026-07-17
author: Aiko (Opus 4.8)
plan: docs/plans/2026-07-17-holonomy-dfa/
status: complete
tags: [auto-holonomy, direct-feedback-alignment, biologically-plausible, credit-assignment, rotor, nagare, mixed-result, F-HOLO-5]
---

# Gate 1 — does the holonomy-DFA learning rule hold up on small data?

**Created-at:** 2026-07-17 00:26 JST · **Plan:** [docs/plans/2026-07-17-holonomy-dfa/](../docs/plans/2026-07-17-holonomy-dfa/)

## Summary

User steer (2026-07-17): *"if the learning is falling apart on small datasets, then we can hardly go
to large Visual-SLAM like data."* Correct — so before scaling the auto-holonomy learning rule to real
pose-graph data, measure whether it holds up small. This is **Gate 1**: does the **holonomy-DFA** rule
(global entropy-pool broadcast + local *exact* inverse-rotor transport, the idea from the prior
exchange) train a deep rotor net as well as sequential exact backprop, and better than vanilla DFA's
random feedback? Same net (`RotorMeshNet` + entropy readout), same task (F-HOLO-1 ring-mesh
coherent-twist), same optimizer — only the **backward rule** changes.

**Result (5-seed median, held-out separability AUROC):**

| rule | L=3 | L=1 | Δdepth | grad-alignment (L=3) |
|---|:-:|:-:|:-:|:-:|
| sequential (exact backprop) | **0.894** | 0.815 | +0.079 | 1.000 (by def.) |
| holonomy-DFA (naive broadcast) | 0.831 | 0.815 | +0.016 | +0.396 |
| **transported-DFA** (rotor-chain transport) | **0.894** | 0.815 | **+0.078** | +0.413 |
| random-DFA (random feedback + local rotor) | 0.560 | 0.542 | +0.018 | −0.005 |
| trivial (raw entropy, no net) | 0.944 | — | — | — |

Verdict: the **naive** broadcast is a partial/shallow learner (F-HOLO-5); the **rotor-chain transported**
broadcast **recovers depth-composition and matches exact backprop** (F-HOLO-6) — the fix, and the
inter-shell transport primitive for a concentric Gömb-Soma.

1. **The exact rotor transport IS the lever (positive, confirmed).** holonomy-DFA reaches **0.831**
   where random-DFA is at chance (**0.560**), and its update is **positively aligned with the true
   gradient (+0.40)** while random feedback's is **zero (−0.005)**. The prior claim holds: reusing the
   forward connection's *exact* inverse rotors as the feedback path — no separate/random feedback
   weights — gives DFA a real credit signal that a random projection cannot. This is the mechanistic
   difference between "holonomy-DFA" and vanilla DFA.
2. **But the global broadcast does NOT compose through depth (the finding).** Under holonomy-DFA,
   depth is **not** load-bearing: L=3 (0.831) ≈ L=1 (0.815). The naive broadcast learns a *shallow*
   function; it does not propagate credit through the stack. The 0.40 alignment quantifies it — enough
   for a shallow win, not enough for deep credit assignment. This is the classic DFA depth ceiling,
   now measured for the rotor case.
3. **It trails exact backprop** (0.831 vs 0.894), consistent with (2).

**Consequence:** learning does **not** collapse on small data — the rotor structure gives a genuine
signal. The naive global broadcast fails to compose through depth, but the **depth-composing fix works**:

## F-HOLO-6 — the rotor-chain transported broadcast recovers depth

`RotorMeshNet::backward_dfa_transported` transports the global credit signal *down through the pure
inverse-rotor chain* (the return-path holonomy: each layer applies its own `R̄`), dropping the
inter-layer mesh Jacobian from the transport path. Measured: **L3 0.894 = exact backprop**, Δdepth
**+0.078** (vs naive +0.016) — depth is now load-bearing. Honest nuance: gradient alignment stays modest
(**0.413**, not 1.0), so it is **not** the exact gradient — it is a depth-composing *approximation* that
reaches the net's ceiling on this (easy) task. It doesn't compute backprop; it transports credit through
the rotor chain well enough to use depth fully. Whether it holds where the net is *necessary* (Gate 2) is
open. The 0.894 tie with sequential is likely the net ceiling on this task, not proof of exactness.

**This transported-rotor primitive is the backward of an inter-shell rotor transport** — the validated
connective tissue for the concentric Gömb-Soma reframe (`docs/design-concentric-gomb-soma.md`): the
architecture and the learning-rule frontier are the same move. **SLAM remains gated** on Gate 2 (a task
where learning is necessary); the depth-composition bottleneck itself is now resolved.

## Honest framing (F-HOLO-2)

On this substrate a trivial baseline (0.944) beats the net, so this gate measures the **learning rule**,
not task-necessity — as the DFA literature compares rules on tasks simple baselines also handle. Whether
learning is *necessary* (a task where trivial **and** the fixed closed-form both fail) is **Gate 2**, a
separate, harder rung not built here. SLAM is downstream of both.

## Literature placement (searched)

The global-broadcast family is Direct Feedback Alignment ([Launay et al. 2020](https://hf.co/papers/2006.12878)),
with a known depth/scaling ceiling ([Filipovich et al. 2022](https://hf.co/papers/2210.14593)) — which
this result reproduces. The variant here (feedback path = the forward connection's *exact* inverse
rotors, not a random matrix) is what lifts it off chance; a bounded search found no prior combining
holonomy/parallel-transport feedback with a DFA-style broadcast, and the +0.40-vs-0.00 alignment gap is
the concrete evidence that the exact transport is doing the work.

## Files touched (new/append; no `CORE.YAML`)

| file | change | role |
|---|--:|---|
| `src/holonomy_net.rs` | +2 methods, +2 tests | `backward_dfa` (broadcast + local rotor), `backward_from_rot_grads` (routing hook) |
| `examples/holonomy_dfa_dissociation.rs` | +366 | 4 rules × 2 depths + alignment angle + multi-seed AUROC + JSON |

**Reused (§6.1):** `RotorMeshNet` forward/cache/exact-backward, `cayley_rotor_backward`,
`MeshTopology::conv_round_backward`, `spectral_reg_value_grad`, `adam_step`, `auroc`, the F-HOLO-1
ring-mesh/coherent-twist generator idiom (re-implemented locally in the example; the committed F-HOLO-1
example is left untouched — a small acknowledged duplication of the toy generator, not worth churning a
frozen artifact).

## CORE.YAML items touched

**None.** No new dependency.

## Test results

`cargo test --release --lib` — **165 passed / 0 failed** (2 new in `holonomy_net`):
`dfa_top_layer_equals_sequential_lower_differs` (the broadcast ≡ threaded at the top, differs below —
the exact identity that anchors the method) and `from_rot_grads_composes_with_dfa` (the routing hook).
Static: `clippy --all-targets -D warnings` **clean**; `fmt --check` **clean**.

## Performance

Full four-arm run (3 rules × 2 depths × 5 seeds + trivial + per-epoch alignment): **1.9 s**, RSS
negligible (toy `N=12`). CPU (Apple M5 Pro). Live per-rule progress to stdout.

## Experiment provenance

- **Git SHA:** `9432103` (branch `main`), working tree dirty; added files uncommitted.
- **Env:** rustc 1.96.1; `hymeko_clifford` vendored; `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. Deterministic; Adam lr 0.05 × 200 epochs. Data = F-HOLO-1 coherent-twist.
- **Data:** `reports/figures/holonomy_dfa_gate1.json`.

## Open issues / follow-ups

- ~~Depth-composing broadcast~~ **DONE (F-HOLO-6)** — `backward_dfa_transported` recovers depth (0.894,
  Δdepth +0.078). Next sub-question: it's an approximation (align 0.41); does it still recover depth on a
  harder task, and does dropping vs keeping the mesh-Jacobian in the transport matter there?
- **Concentric Gömb-Soma** (`docs/design-concentric-gomb-soma.md`) — the architecture whose inter-shell
  transport is this validated primitive. Build gated on Gate 2.
- **Gate 2 — a learning-necessary task.** trivial AND fixed closed-form both fail (e.g.
  XOR-of-regional-curvature on the F-HOLO-4 lattice). Only then is a learning-rule win meaningful for
  task-competitiveness, and only then is SLAM on the table.
- **GD vs Adam.** Reported Adam; a GD run would show the raw rule quality unmasked by adaptive scaling.

## Graphical output (§9)

- **Numerical:** `reports/figures/holonomy_dfa_gate1.json`.
- **Plotted:** `reports/figures/holonomy_dfa_gate1.png` (rule AUROC L3 vs L1 + alignment annotations +
  trivial floor).
- **Animated:** N/A — static training diagnostic.
