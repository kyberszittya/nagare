---
title: "HANDOFF — SBSH: Gömb-Soma object detection (a closed-form, rotation-native, dynamic-spatial-tree YOLO competitor)"
date: 2026-07-11
author: Aiko (agent) for Hajdu Csaba
status: design-handoff (no code yet)
tags: [nagare, gomb-soma, object-detection, yolo, dynamic-spatial-tree, quadtree, phase-pool, holonomy, closed-form]
---

# HANDOFF — SBSH: Gömb-Soma object detection

**SBSH** = *Soma-Based Spatial Hierarchy* (working codename — rename at will). The pitch: an object
detector built entirely on the **Nagare closed-form (no-autograd) stack**, that is **rotation-native**
(the CV phase-pool / rotor-holonomy invariants are its features), spends compute **adaptively** via a
**dynamic spatial tree** (the Soma quadtree) with **entropy routing**, and predicts **oriented boxes**
per tree node. It reuses everything built this session; the graph-side and vision-side primitives fuse
into one hierarchical signed-patch-graph.

This is a **design handoff**, not an implementation. Read it, then start at §8.

---

## 1. Honest positioning — what is and is NOT novel (read first)

Per the operating contract (search before claiming novelty), a 2-query search was run 2026-07-11. **Every
individual mechanism in this proposal already has prior art.** Do not oversell any single piece; the
contribution is the *specific closed-form integration*, and even that must be searched properly (Phase 0)
before any paper claim.

| mechanism | nearest prior art (found) | implication |
|---|---|---|
| rotation-equivariant / oriented detection | ReDet, ReDet2, R2Det, Oriented-YOLOX, AdaR-YOLOv8, rotation-equivariant transformers, ReDiffDet (CVPR 2025) | **mature field.** Rotation-equivariance is *not* our novelty. |
| adaptive quadtree / conditional compute in vision | QDM (quadtree region-adaptive sparse diffusion, 2503.12015), QuadTree-Attention (ICLR 2022), quadtree sparse-event compression | adaptive spatial trees for compute-savings **exist**. |
| closed-form / feedback-free learning | Forward Projection (2501.16476, Nature Comms 2026), CSNNs | no-backprop learning **exists** as a research thread. |

**The specific claim that *may* be defensible** (to be verified in Phase 0, not asserted now): *a
detector whose features are hand-derived closed-form rotation invariants (phase-pool `|DFT|` + rotor
holonomy), whose spatial support is a per-image dynamic quadtree with entropy-gated conditional compute,
and whose head is the Nagare inner-CPML/KAN — trained with hand-derived FD-verified backward, no
autograd.* Frame the value as **rotation-native + adaptive-compute + interpretable/closed-form**, not
"beats YOLO mAP." The honest competitive arena is **oriented/aerial detection** (arbitrary object
orientation — where the ReDet family lives and where our invariances actually pay off), not COCO.

**Required Phase-0 search** (before writing any novelty claim): name the exact nearest works for the
*combination* (closed-form + quadtree + rotation-invariant detection) or state "none found, bounded
search." Do not conflate "quadtrees exist in vision" with "this specific detector exists."

---

## 2. What exists to build on (the reuse map — §6.1)

**Do not re-implement any of these.** The detector is ~90% assembly of shipped, FD-verified ops.

Vision primitives (this session):
- `src/vision.rs` — `orientation_histogram`, `phase_features`, **`spatial_phase_features(r)`** (the R×R
  spatial phase map — *the seed of the dynamic tree: R is a locality knob, make it spatially adaptive*),
  `rotate_image`, `PhaseFeature`.
- `src/ops/phase_pool.rs` — **differentiable** global orientation `|DFT|` invariant (fwd/bwd, FD-verified).
- `src/ops/rotor_holonomy.rs` — order-sensitive rotor product over cycles (fwd/bwd, FD-verified); the
  gauge-invariant magnitude `Re(H)=cos(θ/2)` is the deployable feature.
- `src/ops/dihedral.rs` — `DihedralGroup{n,reflect}` + `dihedral_steer` (C_n/D_n equivariant conv).
- `src/ops/cayley_rotor.rs` — bivector→unit-quaternion rotor (fwd/bwd).
- `src/ops/patch_projection.rs` — N-D patch-embed (fwd/bwd) — the tokeniser.
- `src/ops/hg_message.rs` — signed hypergraph message passing (node↔edge).

Gömb / graph stack:
- `src/ops/gomb_shell.rs` (outer Clifford-FIR), `src/ops/hsikan.rs` (spline-KAN), `src/ops/cpml_tier.rs`
  (**degree-tier structural routing** — *the template for entropy routing: structural decision, no
  backward through the routing, gradients flow through node features*).
- `src/ops/scatter.rs`, `signed_scatter.rs` (aggregation, fwd/bwd), `src/ops/kan.rs` / `softmax_k.rs` /
  `linear.rs` / `mse.rs` / `loss.rs` (heads + losses), `src/ops/spectral_entropy.rs` (the **entropy
  signal** for split/route decisions), `src/ops/adam.rs`.

Data / harness patterns: `src/cv_data.rs` (image loaders, standardise, rot-augment), `examples/cv_bench.rs`
(dataset-general CV harness), `examples/cpml_signed_link.rs` (grid/robustness harness — the `--grid`
`(data×init)` methodology is mandatory for any "helps" claim here too).

**New ops actually needed** (everything else is assembly): (a) a **dynamic quadtree build** (structural,
no backward — like `cpml_tier`); (b) **im2col-at-a-node** / node feature pooling with backward; (c) an
**oriented-bbox head + loss** (regress `cx,cy,w,h,θ`; the angle `θ` couples naturally to the phase/holonomy
features); (d) **oriented NMS** (post-proc, no backward). Small, focused, each FD-verified where it has a
backward.

---

## 3. SBSH architecture (data flow)

```
image
  │  patch_projection (tokenise) + equivariant stem
  ▼
EQUIVARIANT FEATURE FIELD  ── phase_pool |DFT| (rotation-invariant) ⊕ rotor-holonomy magnitude
  │                            ⊕ dihedral_steer (D_n-equivariant conv)   [all closed-form]
  ▼
DYNAMIC SPATIAL TREE (Soma quadtree)  ── split a node when its orientation-entropy / gradient-energy
  │   (structural, per image, no backward)   exceeds a threshold → detail regions get fine cells,
  │                                           flat regions stay coarse  (R becomes spatially adaptive)
  ▼
ENTROPY ROUTING (Gömb-Soma)  ── coarse/low-entropy nodes → cheap path; detail nodes → full cascade
  │   (conditional compute; the "hold accuracy at lower compute" thesis, restated for detection)
  ▼
PER-NODE HEAD (inner-CPML / Chebyshev-KAN, closed-form)  ── per tree node/leaf predict:
  │      objectness · class · ORIENTED bbox (cx,cy,w,h,θ)   (multi-scale anchors for free:
  ▼                                                          coarse nodes = big objects, leaves = small)
ORIENTED NMS → detections
```

**"Mixing with this" — the fusion (graph × vision):** the tree's patch adjacency *is* a signed patch
graph. The **inner CPML tier core** routes tree nodes by "detail-degree" (the tree analogue of vertex
degree); **rotor holonomy** around image-patch loops encodes local orientation flow; **phase-pool `|DFT|`**
are the node descriptors. So the entire signed-graph stack (the flagship) and the entire CV stack collapse
onto one object: a **hierarchical signed-patch-graph** whose leaves are detection anchors. That collapse is
the intellectually interesting part of the proposal.

---

## 4. Dynamic spatial trees — the core mechanism

The single most differentiating piece vs YOLO's fixed `S×S` grid.

- **Build (structural, per image):** start from the root (whole image); recursively split a node into 4
  quadrants iff a **split score** exceeds a threshold, up to a max depth / cell budget. Split score =
  orientation-entropy of the node's phase histogram (reuse `orientation_histogram` + an entropy read) or
  gradient energy. High-detail / high-disorder regions subdivide; flat regions stay coarse.
- **Adaptive R:** in `spatial_phase_features`, `R` is a *global* locality knob (R=1 global-invariant →
  high-R local). The tree makes **R spatially adaptive** — depth = local R. That is the precise
  generalisation of the CV ablation's "R is a domain knob" into "R is a per-region knob."
- **No backward through the split** (like `cpml_tier`): the tree structure is a fixed structural decision
  for the forward; gradients flow through the *node features*, not the split boolean. This keeps the
  closed-form contract clean (no discrete-structure gradient needed). If a learned/soft split is wanted
  later, that is a separate research sub-task (straight-through / soft-quadtree), explicitly out of scope
  for v0.
- **Payoff to measure:** at equal cell budget, does the adaptive tree put more cells on objects than a
  uniform grid (a coverage/recall gain)? And does entropy routing hold mAP at lower FLOPs than dense
  processing? Those are the two discriminating tests that justify the mechanism.

---

## 5. Phased plan (each a closed-form step, each gated by a discriminating test)

| phase | build | Nagare substrate | discriminating test |
|---|---|---|---|
| **0** | novelty search + dataset choice + a **synthetic oriented-shapes** generator (rotated rectangles/triangles at known `θ`) | reuse `cv_data`, `rotate_image`, `tests/common/vision.rs` | is the exact combined detector already published? name it or "none found, bounded". |
| **1** | **dynamic quadtree op** (split-by-orientation-entropy; structural, no bwd) + node feature pool (fwd/bwd, FD) | `orientation_histogram`, `spectral_entropy`, `cpml_tier` pattern | at equal cell budget, adaptive tree covers objects **> uniform grid** (coverage/recall). |
| **2** | **equivariant stem** → per-node descriptor: `phase_pool` `|DFT|` ⊕ `rotor_holonomy` magnitude ⊕ `dihedral_steer` | `phase_pool`, `rotor_holonomy`, `dihedral`, `cayley_rotor`, `patch_projection` | node descriptor is **rotation-robust** (rotate image → descriptor stable), matched to the CV-arc numbers. |
| **3** | **oriented-bbox head** on nodes: closed-form KAN/inner-core → `objectness, class, (cx,cy,w,h,θ)` + **oriented IoU loss** (new, FD-verified where differentiable) | `kan`, `cpml_tier`, `softmax_k`, `linear`, `loss` | detects oriented boxes on the synthetic set (mAP@0.5 well above chance; angle error small). |
| **4** | **entropy routing** (cheap path for flat nodes; full cascade for detail nodes) | `spectral_entropy`, Gömb-Soma routing | **hold mAP at measured lower FLOPs** vs full-compute (the Gömb-Soma efficiency thesis). |
| **5** | **benchmark vs a YOLO baseline** on a real oriented set (HRSC2016 / DOTA subset) + **rotation-robustness** sweep | full stack | honest table: rotation-robustness / FLOP-adaptivity / interpretability wins; mAP likely *below* a tuned YOLO — report it straight. |

Multi-seed + `(data-seed × init-seed)` grids for every "helps" claim (the discipline this project has paid
for three times: single_hsikan, option-msdm, holonomy). Live per-step logging, never blind.

---

## 6. Rigor / gotchas (hard-won, do not relearn)

- **Closed-form, no autograd.** Every op with parameters needs a **hand-derived, FD-verified backward**
  (the `phase_pool` / `rotor_holonomy` template). Structural decisions (tree split, routing, NMS) are
  **fixed per forward, no backward** (the `cpml_tier` template).
- **Do not over-compress** (the Gömb-Soma gate lesson): the tree + routing must **carry** information, not
  bottleneck it. The gate showed Clifford-FIR-as-outer-compressor *hurt*; a node pool that crushes the
  descriptor into too-few dims will do the same. Keep node descriptors rich; route, don't squeeze.
- **Invariance is a lever, but conditionally** (the holonomy arc): rotation-invariance helps most where
  there is **headroom** and the nuisance is real (oriented/aerial). On upright, canonical-pose data
  (COCO-style), a spatial detector may simply win — pick the arena where the invariance pays (§1).
- **Condition the geometry** (the unit-rotor lesson): rotors are unit quaternions; angle features must be
  well-scaled (normalise) or training diverges and you'll misread a conditioning artifact as a negative.
- **Metric integrity** (the RL/CV lessons): measure mAP with a divergence/artifact guard; a physics-style
  "explosion counted as success" has a detection analogue (degenerate boxes inflating recall). Anchor
  every score to a demonstrator/oracle ceiling before optimising under it.
- **No `unwrap` in non-test, `Result`/`thiserror`, complexity gates, `cargo fmt`/`clippy -D warnings`** —
  the standing contract.
- **Prototype, not a YOLO-killer.** State the bar honestly: this is a *closed-form, rotation-native,
  adaptive-compute research detector*. Winning axes are robustness-under-rotation, compute-adaptivity, and
  interpretability — not raw COCO mAP.

---

## 7. Datasets & competitive arena

- **v0 synthetic** — rotated shapes at known `θ` (generated, like the CV arc's synthetic textures). Cheap,
  controllable, isolates rotation-robustness + oriented-box regression.
- **Real oriented** — **HRSC2016** (ships, oriented) or a **DOTA** subset (aerial, arbitrary orientation).
  This is the ReDet-family arena and where phase-pool/holonomy invariance is *supposed* to pay. Data is
  external (fetch script pattern like `fetch_signed_datasets.sh`), kept repo-external.
- **Upright control** — one canonical-pose set (e.g., a small COCO/VOC subset) to *confirm the scope
  boundary*: expect the spatial baseline to win there (the CV-arc rank-flip, restated for detection).

---

## 8. Where to start (first concrete step)

1. **Phase 0 novelty search** — 4-query search naming the nearest combined works (closed-form detector,
   quadtree detection, rotation-invariant detection); write the positioning paragraph honestly.
2. **Discovery pass** (`find`/`grep`/`ls`) confirming none of the §2 ops are duplicated — then build only
   the four genuinely-new ops (§2).
3. **Synthetic oriented-shapes generator** + a 1-image smoke of the tree build (does it subdivide onto the
   shapes?). This is the cheapest test that the central mechanism is even sound.
4. Only then write the §2-plan-bundle for Phase 1 (the dynamic quadtree op) and implement.

Do **not** start by training a full detector. Start by proving the dynamic tree concentrates cells on
content and that a node descriptor is rotation-robust — the two hinges the whole thing turns on.

---

## 9. Open questions / risks

- **Oriented-IoU backward.** Rotated-box IoU is piecewise and its gradient is fiddly (this is why the
  field uses surrogates: KLD/GWD losses in Oriented-YOLOX etc.). Plan a **surrogate oriented loss**
  (Gaussian/KLD) with a clean closed-form backward rather than exact rotated-IoU gradient.
- **Discrete tree ↔ closed-form training.** v0 uses a fixed (non-learned) split → clean. A *learned* split
  is a real research problem (soft-quadtree / straight-through) — keep it out of v0; note it as the main
  extension.
- **Assignment.** YOLO's label assignment (which cell predicts which object) becomes *tree-node
  assignment* — define it (object → smallest node fully containing it, or IoU-max leaf). Get this right
  early; it dominates detector quality.
- **Is the arena right?** If rotation-robustness does not pay even on HRSC/DOTA vs a tuned oriented-YOLO,
  the honest outcome is "closed-form + adaptive-compute detector, competitive on FLOPs/interpretability,
  behind on mAP" — a legitimate, publishable *systems* result, not a failure. Decide up front that this is
  an acceptable landing.

---

## 10. One-paragraph summary for the next agent

Build a closed-form (no-autograd) object detector on the Nagare stack. Tokenise with `patch_projection`;
make features from the **rotation-invariant** CV primitives (`phase_pool` `|DFT|`, `rotor_holonomy`
magnitude, `dihedral_steer`). Lay a **dynamic quadtree** over the image that subdivides where
orientation-entropy is high (structural, no backward — `cpml_tier` template), so `R` (the CV locality
knob) becomes spatially adaptive; **entropy-route** flat vs detail nodes for conditional compute. Predict
**oriented boxes** per tree node with a closed-form inner-CPML/KAN head + a surrogate (KLD) oriented loss;
oriented-NMS. The image thereby becomes a **hierarchical signed-patch-graph**, fusing the flagship graph
stack with the CV stack. Everything individual is prior art (ReDet family; quadtree-attention;
forward-projection) — the bet is the *closed-form, rotation-native, adaptive-compute integration*, tested
in the **oriented/aerial arena** with the project's mandatory `(data×init)` robustness discipline. Start
by proving the tree concentrates cells on content and the node descriptor is rotation-robust; do not train
a full detector first.
