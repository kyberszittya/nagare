---
title: "Nagare — rotor_holonomy op: order-sensitive rotor product over signed cycles"
date: 2026-07-10
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, holonomy, rotor, clifford, signed-graph, closed-form-op]
---

# `rotor_holonomy` — Clifford-FIR reframed as a transmission channel

Date: 2026-07-10 · Nagare (Mac author box), branched from `88e3cca` · CPU

## Why (from the Gömb-Soma gate)

The gate showed the shells hurt **as an outer compressor**. Reading the source pinned the reason:
`clifford_fir_forward` is a rotor-weighted **sum** `out[j] = Σᵢ coefᵢ · x_{vᵢ}[j]` (a linear filter),
and pooling it into an 8-dim per-vertex bottleneck **discards the multiplicative, order-sensitive
holonomy** a rotor traversing a loop should accumulate. Hajdu's reframe: use the rotor as a **running
holonomy** — a *transmission channel* around the cycle — not a sum-filter. "Running holonomy" is a
first-class Nagare concept (the toy lift's order-sensitive channels 5–6); no cycle-holonomy-product op
existed (Clifford-FIR is the sum; `cayley_rotor` is a single rotor). This op is that missing piece.

## What the op is

`src/ops/rotor_holonomy.rs` — per signed cycle, the **ordered quaternion product**
`H_c = q_{k-1} ⋯ q_1 q_0` of per-edge rotors (unit quaternions, from `cayley_rotor` upstream). Because
SO(3) rotors do **not commute**, `H_c` is *order-sensitive* — a genuine holonomy, not the trivial
abelian sum-of-angles — and it generalizes signed-graph **balance** (the sign product `e^{iπ·#neg}`)
into a learned geometric quantity.

- `rotor_holonomy_forward(edge_quats, n_cycles, k) -> (holo (n·4), prefixes (n·k·4))`
- `rotor_holonomy_backward(edge_quats, prefixes, grad_holo, n_cycles, k) -> grad_edge_quats`

**Backward (closed-form, hand-derived):** with `H = Sᵢ · qᵢ · Pᵢ₋₁` and the quaternion adjoint
identities `⟨qx,y⟩ = ⟨x,q̄y⟩`, `⟨xq,y⟩ = ⟨x,yq̄⟩` (Euclidean inner product on ℝ⁴, hold for non-unit `q`):
`∂L/∂qᵢ = conj(Sᵢ) · grad_H · conj(Pᵢ₋₁)`, with suffixes accumulated in the backward pass. Hamilton
product inline; **no new deps**.

## Test results

| test | result |
|---|---|
| `backward_matches_fd` | ok — directional-derivative FD check (5 dirs), non-unit non-coplanar quats |
| `order_sensitive_for_noncommuting_rotors` | ok — swapped-order edges give a different holonomy (proves non-abelian) |
| `k1_is_identity_map`, `all_identity_edges_give_identity_holonomy`, `matches_hand_product_k3` | ok — algebraic sanity |
| full suite | **121 passed / 0 failed** (+5) |
| gate | `cargo fmt --check` + `cargo clippy --all-targets -D warnings` clean |

The FD check validates the conjugate-adjoint backward derivation first try; the order-sensitivity test
confirms the *running* (non-abelian) property that makes the holonomy non-trivial.

## Files touched

| file | change |
|---|---|
| `src/ops/rotor_holonomy.rs` | new op (fwd/bwd + 5 tests), ~200 LOC |
| `src/ops/mod.rs`, `src/lib.rs` | register + re-export |

No new ops-crate deps, no CORE.YAML (repo has none). Plan bundle:
`docs/plans/2026-07-10-rotor-holonomy/` (tex/pdf/tikz/mmd, gitignored).

## Next milestone (the discriminating test — not this turn)

Wire the holonomy into the **inner CPML core** (the flagship winner): per-edge bivector → `cayley_rotor`
→ per-edge quaternion → `rotor_holonomy` → per-cycle holonomy → **(a)** direct per-cycle feature
scattered to vertices, and/or **(b)** holonomy *phase* pooled per vertex via `phase_pool` into a
`|DFT|` invariant — the graph analogue of the CV orientation invariant, unifying both sides of the
session on one op. Discriminating test: inner core **+ holonomy channel** vs inner core alone, signed-link
AUROC, multi-seed. That is where the reframe is confirmed or refuted; this milestone only lands the
FD-verified primitive.

## Provenance

- Nagare (Apple Silicon), branched from `88e3cca`; CPU. No data, no GPU.
- Reproduce: `cargo test --release rotor_holonomy`.
- Quaternion convention `(w, x, y, z)`, Hamilton product. Deterministic test fixtures (seeded by index).
