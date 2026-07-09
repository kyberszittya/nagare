# Nagare — signed hypergraph convolution kernels + a 3-way inner-mechanism A/B

Date: 2026-07-09 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Built the **signed hypergraph message-passing kernels** (the propagation core of a signed HGNN
convolution on cycles-as-hyperedges) as FD-verified Nagare ops, then composed them into a
one-round signed hypergraph conv and A/B'd it against the fixed degree-tier routing on real
signed graphs.

**Finding (honest negative for HGConv-as-embedding):** the learned one-round signed hypergraph
conv sits at the **flat baseline** and **loses to the fixed degree-tier routing on all three
graphs** (13 runs). The cheap degree prior beats a learned signed-propagation round in this
role. The primitive is correct; its *simplest application* isn't the win.

## What was built

**New ops `src/ops/hg_message.rs`** — two dual signed message-passing kernels (port of the
`CapsuleHypergraphRouter` propagation in `cpml.py`), each with a hand-derived, FD-verified
backward:
- `hg_node_to_edge` — `h_e[c] = (1/k) Σ_i σ[c,i]·s[v]·x[v]` (signed, `D^{-1/2}`-scaled node→edge mean).
- `hg_edge_to_node` — `out[v] = s[v]·Σ σ[c,i]·h_e[c]` (signed edge→node scatter, per-node scaled).

The per-node scale `s_v` (e.g. `D_v^{-1/2}`) is a fixed structural quantity (like `signs`), so no
gradient flows through it. The learnable `vertex_proj`/`edge_lin`/head are plain `linear` (reused,
not re-implemented). 4 FD/unit tests.

**Composed conv (in `examples/cpml_signed_link.rs`, 3rd arm):**
`x0 → vertex_proj → node→edge(σ, D^{-1/2}) → edge_lin → edge→node(σ, D^{-1/2}/D) → concat(x0,·)
→ edge head`. Same features/triangles/edges/AUROC as the tier arms — a clean 3-way comparison.

## Results (median test AUROC)

| graph | L=1 flat | L=3 tier (fixed) | HGConv (learned) | HGConv vs flat | HGConv vs tier |
|---|---|---|---|---|---|
| Bitcoin Alpha | 0.8700 | **0.8818** | 0.8677 | −0.002 (below) | −0.014 |
| Bitcoin OTC | 0.8971 | **0.9023** | 0.8994 | +0.002 (above) | −0.003 |
| Slashdot | 0.8901 | **0.8923** | 0.8909 | +0.001 (above) | −0.001 |

HGConv ≈ flat (within ±0.002, higher variance) and **below tier on every graph**. Plot:
`reports/figures/hgconv-vs-tier-signed-link.png`.

## Reading (measured / inferred / hypothesis)

- **Measured:** the kernels are correct (FD-verified). As a one-round *linear* standalone node
  embedding, the signed hypergraph conv does not beat flat aggregation and loses to the fixed
  degree-tier routing on 3/3 real graphs.
- **Important scope caveat (why this isn't "HGConv doesn't work"):** the reference
  `CapsuleHypergraphRouter` uses the hypergraph conv as a **soft router** — it emits per-cycle
  *tier logits* that reweight the tier aggregation (`capsule_soft`), i.e. a learned replacement
  for the *fixed* `TierSpec.assign` routing, feeding **into** the tier core. Here it was used as
  a standalone embedding — a different, simpler role. The negative is about *that* role.
- **Untested hypotheses for the gap:** (1) reduced form — linear `edge_lin` (reference has a GELU
  MLP), single round, `DH=8`; (2) signed averaging over triangle neighbourhoods may dilute the
  strong pairwise triad signal the degree features preserve; (3) on these graphs signed-link is
  degree-dominated, which the tier prior captures cheaply. The faithful **HGConv-as-router →
  tier core** is the natural next test, and the one the reference actually intends.

This continues the session's regime discipline: a mechanism's value depends on how it's used.
The tier prior earns its weight (justified last step); the hypergraph conv, in its simplest
standalone form, does not — measured, not assumed.

## Files touched

| file | change |
|---|---|
| `src/ops/hg_message.rs` | **new** — 2 signed message-passing kernels (fwd+bwd) + 4 FD/unit tests |
| `src/ops/mod.rs`, `src/lib.rs` | +mod / +re-export |
| `examples/cpml_signed_link.rs` | +HGConv arm (`run_hgconv`) + per-corner triangle signs + `D^{-1/2}` scales; 3-way output |
| `scripts/dev/plot_hgconv.py`, `reports/figures/hgconv-vs-tier-signed-link.png` | **new** — plot |

## CORE / deps

**None.** Kernels reuse nothing new; `hg_message` is standalone; no dependency change.

## Test results / provenance

- `hg_message`: 4/4 (both backwards FD-matched, signed-scale check, round-trip). Full suite **96/0**;
  clippy `-D warnings` + fmt clean.
- Data (repo-external, SNAP): `nagare_data/signed/`. Bitcoin 5 seeds, Slashdot 3 seeds, Mac-measured.
- Reproduce: `cargo run --release --example cpml_signed_link -- --data <file> --seed <s>`.
- Repo `github.com/kyberszittya/nagare`. Rust 1.96.1. Leakage-free (train edges only).
