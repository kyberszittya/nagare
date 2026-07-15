---
title: "Nagare — canonical results collection (baseline before the next, non-CV direction)"
date: 2026-07-15
author: Aiko (agent) for Hajdu Csaba
scope: the whole Nagare crate (holonomy_learn) — op library + every result-line, with a domain-transfer lens
---

# Nagare — canonical results collection

Date: 2026-07-15 · Nagare (`holonomy_learn`, github.com/kyberszittya/nagare) at `dc111b7` · Mac + kato15.

## 0. What Nagare is

A **closed-form, no-autograd** ML framework in Rust. Every op ships a **hand-derived, finite-difference-verified**
forward/backward pair — there is no autograd anywhere in the crate. That discipline is the product: a growing
library of verified backprops that compose into models, plus the assimilation framework (§5) that turns each
experiment into shared, registered, regression-protected capability.

Scale: **31 FD-verified backward ops · 168 passing tests · 81 result reports · 0 autograd.** fmt + clippy clean.

## 1. The op library (the foundational, domain-general asset)

Each op below has a hand-derived backward verified against finite differences. Grouped by kind; **the first four
groups are domain-general** (nothing CV about them), the fifth is CV-specific.

| group | ops | transfers to |
|---|---|---|
| **Function approximation** | `kan`, `hsikan` (signed KAN), `catmull_rom`/`chebyshev_cr`, `kochanek_bartels` | any regression/representation task |
| **Geometry / rotors** | `dihedral`, `cayley_rotor`, `rotor_holonomy`, `rotor_spike`, `clifford_fir` | any signal with rotational/phase structure (not just images) |
| **Graph / hypergraph** | `hg_message` (signed node↔edge + sign-grads), `signed_scatter`, `scatter`, `gomb_shell`, `cpml_tier`, `project_alpha_mix` | relational / signed-network data |
| **Entropy / pooling** | `spectral_entropy`, `fused_entropy_update`, `global_entropy_pool` (covariance eigen-entropy), `phase_pool`, `softmax_k` | any set/distribution readout with an invariance |
| **Core** | `linear`, `mse`, `loss` (BCE), `adam` | everything |
| **CV-specific** | `conv2d`, `group_pool`, `sc_block`, `soft_argmax`, `oriented_head`, `oriented_descriptor`, `gaussian_kld`, `quadtree`, `patch_projection`, `fsr_mixer` | images / spatial grids |

## 2. Result-lines

### 2A. Relational / signed-graph (NON-CV — already a second domain)

The Nature-style **leakage audit** line on signed-link prediction (Bitcoin-OTC/Alpha, Slashdot, Epinions, Reddit).

- **Signed-balance metric** = Z₂ cycle holonomy (strong Cartwright–Harary), measured across graphs with **unbiased
  wedge (triangle-uniform) sampling** after catching an edge-then-neighbour estimator bias (Slashdot 0.797→~0.87).
- **Label-shuffle audit** — the strict-protocol CPML core learns **signed structure, not leakage** (4-graph); full
  **2×2 strict-vs-transductive Table 2** complete (5 graphs).
- **Rotor-holonomy channel** on the CPML core — a **positive-but-marginal** win whose **gain tracks headroom**
  (density prediction on Epinions failed; single-head positive kept with a multi-head honesty correction).
- **Learnable Chebyshev-CR edge encoder** — real `[−1,1]` weights **beat the ±1 indicator** (OTC 0.9076 vs 0.9041,
  8-seed), warm-started to avoid co-divergence; wired onto the holonomy path (small robust paired win); hg-conv CR
  a minor/neutral difference (basis retained — see the "don't devolve a foundational basis on noise" rule).

### 2B. CV — the Neocognitron arc (assimilated; see `assimilations/neocognitron-arc.md`)

Closed-form S/C hierarchy, rotation-native. Headlines (5-seed): **entropy-top** rotation-invariant recognition
**held-out AUROC 1.000** + equivariant **pose 0.9° MAE** + **~3300 updates/s** (user hypothesis confirmed);
the **skeleton envelope** P2 neutral / **P3 decisive** (2.6 vs 6.9 px) / **P4 hurts** (loop closure inexpressible by
a shared transform). Governing law **F-ARC-1**: a prior pays iff the base lacks the signal **and** the op can
express the constraint.

### 2C. CV — the SBSH "Gömb-Soma" detector (a closed-form YOLO alternative)

Dynamic **quadtree** grid (replaces YOLO's fixed S×S) + oriented **Gaussian-KLD** boxes + biological **rotor-spike**
tuning. Held-out **P=0.656, R=0.966, F1=0.781**. Phases 1–5 each landed as an FD-clean op.

### 2D. CV — texture / rotation benchmarks

KTH-TIPS2-b materials (11-class), `D_n` group-conv vs single-frame canonicalisation, `phase_pool` `|DFT|`
rotation invariant, learned-vs-fixed orientation field.

### 2E. HSiKAN structural-leverage arc (representation, domain-general)

H1 (scaling) + H2 (causal) **supported** via a scramble + DeepSets double-dissociation; structural benefit grows
**3.7→61×** with chain length. Torch-parity + multiseed harnesses in `tests/`.

### 2F. Systems / performance

CPU **scatter-locality**: sparse-mm vs `index_add` **2.9× @1-thread** once the accumulator exceeds L3 (tail win;
GPU atomics are where Nagare pays). KAN baselines (iris, california, graph-vs-KAN).

## 3. Transfer lens — what carries beyond CV

For a non-CV direction, the reusable substrate is **everything in §1 groups 1–4 plus §2A/§2E**:

- **Signed hypergraph message passing** (`hg_message`, `signed_scatter`, sign-gradients) + **signed-balance
  holonomy** — a relational engine already validated on real signed networks, *independent of vision*.
- **HSiKAN / KAN** — a verified learnable function basis for any tabular/relational representation.
- **Rotors / Clifford / holonomy** — geometry on any signal with rotational or cyclic structure.
- **Entropy pooling** — `global_entropy_pool`'s idea (an invariant + equivariant readout from a weighted covariance)
  and `spectral_entropy` generalise to any point-set/distribution, not just pixels.
- **The closed-form + FD-verified discipline** and **the assimilation lifecycle** (§5) — the method itself.

**CV-specific and not directly transferable:** `conv2d`, `group_pool`, `sc_block`, the SBSH detector, the pose
nets, `oriented_*`/`gaussian_kld`/`quadtree`/`patch_projection`. These stay as the CV application of the substrate.

## 4. Gate / provenance state

- Suite **168 / 0**; `cargo fmt --check` + `cargo clippy --all-targets -D warnings` clean; no autograd; no
  `CORE.YAML` edits across the arc.
- Mac = origin at `dc111b7`; kato15 (RTX 6000, `~/nagare`) syncs via base64-piped bash (csh login shell).

## 5. Framework source-of-truth (machine-readable)

- `reports/framework/canonical_components.json` — the component registry (query before creating anything new).
- `reports/framework/canonical_findings.json` — the findings ledger (F-ARC-1 + F-N2b/N3/ENT/P2/P3/P4).
- `reports/framework/assimilations/<id>.{md,json}` — per-experiment assimilation records.
- `reports/framework/nagare_results_collection.json` — this collection, machine-readable.
- Binding operating rule: every completed experiment is assimilated into shared framework code (extract + import +
  guard + register + regression) **before** the next run.

## 6. Ready for the next direction

The substrate is consolidated and gated. The relational (§2A) and representation (§2E) lines already demonstrate
Nagare beyond vision, so a new non-CV direction has a validated, verified, domain-general base to build on rather
than a CV-entangled one.
