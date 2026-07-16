---
title: "Deep holonomy net — the mechanism half of 'one step to deep-representation learning' (compositional auto-holonomy, closed-form, FD-verified through depth)"
date: 2026-07-16
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, clifford, quaternion, deep, mechanism, fd-verified]
---

# Deep holonomy net — mechanism half proven

Date: 2026-07-16 · Mac (arm64, CPU) · on-mission (holonomy embedding, no autograd)

## Summary

The agreed path from readout-learning to deep-representation learning had two halves: a **mechanism** half (does the
holonomy feedback compose *through depth* as a closed-form operation, not a tape?) and an **empirical** half (does
it learn *useful* deep features?). This is the mechanism half, **proven in code**.

`RotorMeshNet` (`src/holonomy_net.rs`) is a deep stack of *rotate then mix*:
- **rotate** — a learned per-node rotor (Cayley bivector → unit quaternion, `cayley_rotor`) transports the node
  3-vector field. The bivectors are the **learned representation**.
- **mix** — one mesh contraction round (`MeshTopology::conv_round`, node→edge→node) spreads the transported field
  across the simplicial neighborhood.

Stacking is genuinely deep: pure `SO(3)` rotations would collapse to one rotation, but the mesh mix *between* rotors
re-combines the field, so depth adds capacity (regression test `depth_collapses_without_the_mesh_mix_but_not_with_it`).

**The crux, verified:** the deep stack's end-to-end backward — composing the two FD-verified atoms — matches finite
differences w.r.t. **every layer's bivectors and the input field** (`deep_backward_matches_fd`, 3 layers, tol 2e-2).
The gradient handed down between layers is `grad_v = R̄·grad`, the upstream gradient transported by the **inverse
rotor** (quaternion conjugate): the **adjoint holonomy transport**, closed-form, replacing backprop's tape.

So: the compositional auto-holonomy composes through depth with a closed-form, FD-verified backward, no autograd —
exactly the half I said I was confident in, now demonstrated rather than asserted.

## What is / isn't done (honest)

- **Done:** the deep, learnable, closed-form, FD-verified holonomy transform (the mechanism). Reuses the existing
  FD-verified atoms `cayley_rotor` + `MeshTopology::conv_round` — no reimplementation, pure composition + backward
  threading.
- **Not done — the empirical half:** the **learning rule** is not wired. Descending this closed-form gradient by
  iterative steps is ordinary GD (a stepping stone, *not* the thesis); the on-thesis **instantaneous entropy/holonomy
  feedback** is the next step. And the **discriminating experiment** — entropy-top beats mean-top **through depth**
  (the double dissociation from the single-layer entropy-pool result, run over the deep stack) — is not yet run.
  That experiment is the empirical crux that decides whether "one step" holds.

## Clifford / simplicial grounding

- The rotors are quaternions = the even subalgebra of Cl(3,0), Spin(3)=SU(2) — exact geometric-algebra transport,
  not an analogy. `rotor_holonomy` (ordered non-commutative rotor product) is the loop-holonomy sibling available
  for the next layer.
- The mesh mix is the simplicial coboundary/boundary contraction (`hg_message` via `MeshTopology`): node/edge fields
  = 0-/1-cochains. Hypergraph→simplices gives the calculus; this net rides on it.

## Tests / gates

| item | result |
|---|---|
| `holonomy_net::deep_backward_matches_fd` | pass (3-layer composed backward == FD, bivecs + input) |
| `holonomy_net::depth_collapses_without_the_mesh_mix_but_not_with_it` | pass (depth is load-bearing) |
| full suite | **185 / 0** · fmt + clippy clean |

## Files touched

| file | change |
|---|---|
| `src/holonomy_net.rs` | new — `RotorMeshNet` (deep rotate+mix holonomy stack) + `RotorMeshCache`, closed-form backward, 2 tests |
| `src/lib.rs` | module + re-exports |
| `reports/framework/canonical_components.json` | `RotorMeshNet` registered (MECHANISM_DEMO) |

## Provenance

Nagare `ca79477`+ on Hajdus-MacBook-Pro (arm64, CPU-only), `cargo 1.96.1`. No new dependency (reuses `cayley_rotor`
+ `mesh_tensor`). CPU-only — no kato15 needed.

## Next

- **The learning rule:** wire the entropy readout (`global_entropy_pool`) as the objective on the deep field and
  the **entropy/holonomy feedback** as an instantaneous update to the bivectors (not iterative GD).
- **The discriminating experiment:** entropy-top vs mean-top, deep vs single-layer, on the rotation-varied task —
  the double dissociation *through depth*. That decides "one step vs short arc."
