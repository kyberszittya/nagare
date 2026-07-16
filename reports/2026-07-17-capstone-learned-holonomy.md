---
title: "Capstone — learning holonomy-native on the non-abelian task: the designed op must stay FIXED"
date: 2026-07-17
author: Aiko (Opus 4.8)
plan: (extends docs/plans/2026-07-17-noncommutativity)
status: complete
tags: [auto-holonomy, non-abelian, end-to-end, no-autograd, thesis, nagare, mixed-result, F-HOLO-10]
---

# Capstone — end-to-end learning through the holonomy op, and what it reveals

**Created-at:** 2026-07-17 03:52 JST · **Follows:** F-HOLO-9 (non-commutativity) · **Learning discipline:** F-HOLO-6 (composed closed-form backwards, no tape)

## Summary

The capstone question: does the **learned** holonomy machinery discover the commutator on the F-HOLO-9
non-abelian task, where a generic MLP over the raw edges is at chance? Built a fully **end-to-end**
network with `rotor_holonomy` as a differentiable **core layer** —
`raw edges → learned per-edge linear W1 → rotor_holonomy (2 cycles) → [H_A', H_B'] → learned head` —
trained by **composing the crate's closed-form backwards** (`linear_backward ∘ rotor_holonomy_backward ∘
linear_backward`), **no autograd tape** (the F-HOLO-6 discipline). A hard FD gate on the end-to-end
gradient ran first.

**Result (5-seed median, held-out AUROC; FD gate: max |analytic − fd| = 6.3e-5, PASS):**

| arm | AUROC |
|---|:-:|
| generic-MLP (raw edges) | 0.541 |
| learned-holo-net, **trainable** W1 | **0.513** |
| learned-holo-net, **frozen** W1 (fixed `rotor_holonomy` + learned head) | **0.968** |
| fixed-commutator (context) | 1.000 |

Two findings, and a clean diagnosis:

1. **Making the composition trainable CORRUPTS it.** With the pre-composition layer `W1` trainable, the
   net is at **chance (0.513)** — *worse* than the generic MLP. The FD gate passing (6.3e-5) rules out a
   gradient bug: the gradient is exactly right; the *optimization* fails. The trainable `W1` is pushed
   off identity by gradients backpropagated through the initially-random head, and — because the signal
   lives in the delicate ordered quaternion product — a perturbed `W1` destroys the holonomy features
   before the head can learn.
2. **Freezing the composition (the fixed designed op) + learning the readout WORKS.** With `W1` frozen at
   identity, the net reaches **0.968** — learning a readout *on* the fixed `rotor_holonomy` op solves the
   non-abelian task a generic learner (0.541) cannot.

**This is the Nagare thesis, measured.** The holonomy op is a **fixed, exact, designed representation**;
its value is realized by computing it and learning *on top*, **not** by training *through* it. End-to-end
"make everything learnable" actively *hurts* here — the designed op must stay fixed. `designed > learned`
is not a slogan in this experiment; it is the difference between 0.968 and 0.513, with the gradient
provably correct in both.

## The arc, closed

Six results now give a complete, honest characterization of when and how the framework's specific
machinery is justified:

- **Scalar / 2nd-order signals** → generic baselines match or beat the framework's specific ops
  (F-HOLO-2, real-spine, F-HOLO-8). Use the simple thing.
- **Non-abelian signals** → the fixed `rotor_holonomy` op is **necessary** (F-HOLO-9); learning enters as
  a **readout on the fixed op** (F-HOLO-10, 0.968), **not** as end-to-end training through it (0.513).
- **The learning rule** (F-HOLO-5/6): a global-broadcast credit rule is shallow; a rotor-chain
  transported broadcast recovers depth — but that whole thread is about learning *node-field* rotor
  representations, a different regime from the *fixed input-composition* that non-abelian tasks need.

The synthesis: **Nagare's designed holonomy ops are the load-bearing primitive for order-dependent,
non-abelian structure; learning is best applied as a readout on those fixed ops, and end-to-end
differentiability through them is a liability, not an asset, on this class of tasks** — exactly the
exact/designed-over-learned thesis.

## Integrity note

The FD gate (6.3e-5) is the crux: it proves the trainable-W1 failure is *not* a gradient bug — the
composed closed-form backward is correct — so the 0.513 is a genuine optimization/representation result,
not a phantom. The frozen-W1 arm (0.968) is the discriminating test that localizes the failure to the
trainable pre-composition, as hypothesized, rather than the head or the gradient.

## Files touched (new/append; no `CORE.YAML`)

| file | change |
|---|---|
| `examples/noncommute_learned.rs` | the end-to-end `HoloNet` (linear→rotor_holonomy→head), composed closed-form backward, FD gate, trainable/frozen W1 arms |

**Reused (§6.1):** `linear_forward/backward`, `rotor_holonomy_forward/backward`, `adam_step`,
`sample_noncommute`, `commutator_angle`, `auroc`. No new lib code; the model composes existing ops.

## CORE.YAML items touched

**None.** No new dependency.

## Test results

`cargo test --release --lib` — **173 passed / 0 failed** (no new lib tests; the FD gate is a runtime
integrity assertion in the example, F-HOLO-1 precedent). Static: `clippy --all-targets -D warnings`
**clean**; `fmt --check` **clean**. FD gate: 6.3e-5 PASS.

## Performance

Full run (5 seeds × {trainable-W1, frozen-W1, generic-MLP}): **272 s**, RSS negligible. CPU (Apple M5 Pro).

## Experiment provenance

- **Git SHA:** `9432103` (branch `main`), dirty; added file uncommitted.
- **Env:** rustc 1.96.1; `hymeko_clifford` vendored; `rayon`. macOS 26.5.2, Apple M5 Pro.
- **Seeds:** 0–4. `reports/figures/noncommute_learned.json`.

## Open issues / follow-ups

- **Real non-abelian data (pose-graph SLAM).** F-HOLO-9/10 justify it: the deploy pattern is the *fixed*
  `rotor_holonomy` (loop-closure consistency) with a learned readout — not an end-to-end trained rotor
  stack. The g2o benchmarks are the real test of the now-justified pattern.
- **Why the trainable-through-holonomy landscape is bad** (curiosity, not blocking): the ordered
  quaternion product's sensitivity + the non-unit drift of a linear `W1` — a projected/rotor-constrained
  `W1` (staying on the manifold) might behave better, but the finding (fixed op wins) already stands.

## Graphical output (§9)

- **Numerical:** `reports/figures/noncommute_learned.json`.
- **Plotted:** `reports/figures/noncommute_learned.png` (4-arm bars).
- **Animated:** N/A.
