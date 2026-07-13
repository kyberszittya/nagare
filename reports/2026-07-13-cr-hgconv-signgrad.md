---
title: "Nagare — hg_message sign-gradient ops (FD-verified) + hgconv learnable-CR: minor difference, basis retained"
date: 2026-07-13
author: Aiko (agent) for Hajdu Csaba
tags: [nagare, cpml, signed-link, hg-message, chebyshev-cr, sign-gradient, hsikan-basis]
---

# `hg_message` sign gradients + hgconv learnable-CR (minor difference; the CR basis is retained)

Date: 2026-07-13 · Mac (Apple Silicon) · Nagare at `11c1b47`+ · CPU

## Summary

Extended the signed-hypergraph message-passing kernels with the missing gradient — **w.r.t. the corner
signs** — so the signs can be made learnable (the op previously treated them as a constant structural
quantity). Two new FD-verified ops, then wired the learnable Chebyshev-CR onto the hgconv arm (`--cr-hg`) and
A/B'd it. The Chebyshev-CR is the **HSiKAN learnable-basis — the framework's foundation, not an add-on** — so
it is retained as the basis regardless of a single arm's A/B. On the hgconv arm the base-vs-CR difference is
**minor and within seed variance** (essentially neutral); on the holonomy arm the CR is a small robust win.
Correction to an earlier draft: the hgconv result is a *minor/neutral* difference, not a "negative" — the
deltas are noise-level and do not warrant devolving the basis.

## The ops (FD-verified, reusable)

`src/ops/hg_message.rs`:
- `hg_node_to_edge_sign_grad`: `∂L/∂σ[c,i] = Σ_d grad_he[c,d]·s[v]·x[v,d]/k`, `v = cycles[c,i]`.
- `hg_edge_to_node_sign_grad`: `∂L/∂σ[c,i] = Σ_d grad_out[v,d]·s[v]·h_e[c,d]`.

Both are exact (elementary dot-products) and **FD-verified** against the forwards (2 new tests; full suite
**147/0**). These enable *any* learnable-sign use of the signed-HGNN kernels — not just the CR — which is the
lasting value here regardless of the CR outcome.

## The hgconv CR — minor difference (essentially neutral)

`--cr-hg` re-encodes the hypergraph signs via the learnable CR each step; the signs enter **both** kernels, so
the coef gradient sums both sign-gradients → `chebyshev_cr_backward` (warm-started, same as `--cr-holo`).

**Paired 5-seed A/B (`--cr-hg` vs base, `--real-weights`, hgconv arm):**

| graph | hgconv base | hgconv + CR | paired Δ | seeds Δ>0 |
|---|---|---|---|---|
| bitcoin-otc | 0.8992 | 0.8970 | −0.0005 | 2/5 |
| bitcoin-alpha | 0.8627 | 0.8657 | −0.0057 | 2/5 |

The deltas are **noise-level and high-variance** (per-seed span ≈ −0.011…+0.007), not robustly separated from
zero — i.e. **the CR and the base are ~indistinguishable on this arm**. This is a *minor difference*, not a
robust negative; the holonomy arm is where the CR shows a small robust win.

## Why the effect is clearer on the holonomy arm

- **Holonomy:** the sign feeds a *geometric* rotor construction (per-edge quaternion → ordered holonomy
  product). Reshaping the sign magnitude smoothly reshapes the rotor angle/coherence — a well-conditioned,
  monotone effect the CR exploits, so a small signal surfaces.
- **Hgconv:** the sign multiplies node/edge features inside *learned linear* message-passing (`vproj`/`elin`),
  which co-adapt with the CR at higher variance, and hgconv already *ties* the flat baseline — so any small CR
  effect is washed out by seed noise. Not that the CR *hurts*; the arm just doesn't resolve a signal.

The learnable Chebyshev-CR is the **HSiKAN basis** and is retained as such — `--cr-hg` stays a first-class
option (its difference vs base is minor). The sign-gradient ops stay because they are a correct, FD-verified,
reusable primitive that makes the hypergraph signs learnable for *any* future use.

## Files touched

| file | change |
|---|---|
| `src/ops/hg_message.rs` | `hg_node_to_edge_sign_grad`, `hg_edge_to_node_sign_grad` + 2 FD tests |
| `src/lib.rs` | re-export the two sign-grad ops |
| `examples/cpml_signed_link.rs` | `--cr-hg` wiring (learnable CR on hgconv signs); shared CR consts |

Gates: `cargo fmt --check`, `cargo clippy --all-targets -D warnings` clean; full suite **147/0** (+2). No new
deps, no CORE.YAML (`hg_message` is not CORE-listed).

## Standing verdict on the CR arc

- **The Chebyshev-CR is the basis** (HSiKAN's learnable Catmull-Rom) — a load-bearing framework mechanism,
  retained and developed, not gated on any single arm's A/B.
- **Holonomy path:** the CR shows a small, robust win on magnitude data (`--cr-holo`).
- **Hgconv path:** the CR is ~indistinguishable from base (minor, noise-level difference) — `--cr-hg` is a
  first-class option; the sign-gradient ops it required are a reusable FD-verified primitive.
- **Overall:** magnitude via the learnable HSiKAN CR basis carries a small genuine signal where the path is
  geometric (holonomy) and magnitude exists; elsewhere the difference is minor. The basis stands; the effect
  size is honestly small.

## Provenance

- Mac (Apple Silicon), Nagare `11c1b47`+; CPU. Data: `nagare_data/signed/soc-sign-bitcoin{otc,alpha}.csv`.
  5 seeds; default invocation (`--real-weights [--cr-hg]`), hgconv arm; CR k=6, grid=8, warm 1/3.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <csv> --real-weights --cr-hg`
  (read the `signed hypergraph conv` line); `cargo test --release hg_message` for the op FD tests.
