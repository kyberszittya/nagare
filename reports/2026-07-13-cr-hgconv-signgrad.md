---
title: "Nagare — hg_message sign-gradient ops (FD-verified) + the hgconv CR is an honest negative"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, hg-message, chebyshev-cr, sign-gradient, negative-result]
---

# `hg_message` sign gradients + hgconv learnable-CR (honest negative)

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `11c1b47`+ · CPU

## Summary

Extended the signed-hypergraph message-passing kernels with the missing gradient — **w.r.t. the corner
signs** — so the signs can be made learnable (the op previously treated them as a constant structural
quantity). Two new FD-verified ops, then wired a learnable Chebyshev-CR onto the hgconv arm (`--cr-hg`) and
A/B'd it. **The ops are a clean reusable primitive; the hgconv CR itself is a negative** — unlike the holonomy
CR, it does not help.

## The ops (FD-verified, reusable)

`src/ops/hg_message.rs`:
- `hg_node_to_edge_sign_grad`: `∂L/∂σ[c,i] = Σ_d grad_he[c,d]·s[v]·x[v,d]/k`, `v = cycles[c,i]`.
- `hg_edge_to_node_sign_grad`: `∂L/∂σ[c,i] = Σ_d grad_out[v,d]·s[v]·h_e[c,d]`.

Both are exact (elementary dot-products) and **FD-verified** against the forwards (2 new tests; full suite
**147/0**). These enable *any* learnable-sign use of the signed-HGNN kernels — not just the CR — which is the
lasting value here regardless of the CR outcome.

## The hgconv CR — negative

`--cr-hg` re-encodes the hypergraph signs via the learnable CR each step; the signs enter **both** kernels, so
the coef gradient sums both sign-gradients → `chebyshev_cr_backward` (warm-started, same as `--cr-holo`).

**Paired 5-seed A/B (`--cr-hg` vs base, `--real-weights`, hgconv arm):**

| graph | hgconv base | hgconv + CR | paired Δ | seeds Δ>0 |
|---|---|---|---|---|
| bitcoin-otc | 0.8992 | 0.8970 | **−0.0005** | 2/5 |
| bitcoin-alpha | 0.8627 | 0.8657 | **−0.0057** | 2/5 |

High variance, paired median **negative**, only 2/5 seeds positive on each — **not a win.** This contrasts
sharply with the holonomy CR (robust +0.001–0.0015, 9/10 seeds on Bitcoin).

## Why holonomy-CR helps but hgconv-CR doesn't

- **Holonomy:** the sign feeds a *geometric* rotor construction (per-edge quaternion → ordered holonomy
  product). Reshaping the sign magnitude smoothly reshapes the rotor angle/coherence — a well-conditioned,
  monotone effect the CR can exploit.
- **Hgconv:** the sign multiplies node/edge features inside *learned linear* message-passing (`vproj`/`elin`).
  The CR and those linears co-adapt with high variance, and hgconv already *ties* the flat baseline (little
  structure to gain). So a learnable sign reshaping mostly adds optimisation noise.

An honest, useful negative: the learnable-magnitude idea pays on the geometric (holonomy) path, not the linear
(hgconv) path. `--cr-hg` stays opt-in and **not recommended** (documented negative); the sign-gradient ops
stay because they are a correct, FD-verified, reusable primitive.

## Files touched

| file | change |
|---|---|
| `src/ops/hg_message.rs` | `hg_node_to_edge_sign_grad`, `hg_edge_to_node_sign_grad` + 2 FD tests |
| `src/lib.rs` | re-export the two sign-grad ops |
| `examples/cpml_signed_link.rs` | `--cr-hg` wiring (learnable CR on hgconv signs); shared CR consts |

Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` clean; full suite **147/0** (+2). No new
deps, no CORE.YAML (`hg_message` is not CORE-listed).

## Standing verdict on the CR arc

- **Holonomy path:** learnable Chebyshev-CR is a small, robust win on magnitude data (opt-in `--cr-holo`).
- **Hgconv path:** learnable Chebyshev-CR is a negative (`--cr-hg`, not recommended); the sign-gradient ops it
  required are kept as a reusable FD-verified primitive.
- **Overall:** magnitude via a learnable HSiKAN CR basis carries a little genuine signal — but only on the
  geometric path, only where magnitude exists, and only once warm-started. Sharpened, honest, bounded.

## Provenance

- Mac (Apple Silicon), Nagare `11c1b47`+; CPU. Data: `nagare_data/signed/soc-sign-bitcoin{otc,alpha}.csv`.
  5 seeds; default invocation (`--real-weights [--cr-hg]`), hgconv arm; CR k=6, grid=8, warm 1/3.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <csv> --real-weights --cr-hg`
  (read the `signed hypergraph conv` line); `cargo test --release hg_message` for the op FD tests.
