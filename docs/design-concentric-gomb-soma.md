# Design — Concentric Gömb-Soma (vision, gated on Gate 2)

**Author:** Aiko for Hajdu Csaba · **Date:** 2026-07-17 · **Status:** design captured, build gated · **Origin:** user reframe 2026-07-17

> This is a *design/vision* doc, not a build plan. It captures the concentric Gömb-Soma reframe and
> grounds it in existing ops + tonight's validated transport primitive. **The full heterogeneous net is
> NOT to be built until a Gate-2 task makes its richness necessary** (see §5) — building an elaborate
> architecture before a task discriminates it is the "architecture-astronomy" trap (CLAUDE.md §3/§6.5).

## 1. The reframe

**Current Gömb** (`gomb_shell::gomb_outer_forward`): `M` parallel Clifford-FIR banks over one signed
cycle pool, concatenated — a *flat, wide, homogeneous* layer (the V1-bank "volume"). Despite the name
球 (ball/sphere), nothing is concentric; it is one wide linear layer in a linear stack
(ChebyCR → hg → Clifford-FIR → Gömb → HSiKAN → CPML → rotor-holonomy → entropy).

**Proposed Gömb-Soma** = **concentric shells**, geometrically honest to 球:
- **Shells** are nested (radial), not a flat stack. Concrete grounding: the **CPML degree-tiers**
  (`cpml_tier`) already stratify cycles by vertex-degree — the tiers *are* the concentric shells,
  nested by connectivity depth (innermost = high-degree hubs, outer = periphery).
- **Each shell is heterogeneous**: a *mix* of `clifford_fir` (signed-cycle aggregation), `cpml_tier`
  (degree-restricted pooling), and `hsikan` (highway signed-KAN) — not one op type repeated.
- **Highway wiring varies per shell** (`HsikanConfig::use_highway` per shell) — inner shells more
  gated/residual, outer shells more transformative, or vice-versa (a hyperparameter of the design).
- **Inter-shell transport = a rotor operation** (`rotor_spike`-family forward; the validated
  transported-rotor primitive backward, §3) — signal moves *radially* between shells by rotor
  transport, not a plain linear projection.

## 2. Why it is more than aesthetics — the F-HOLO-5/6 convergence

Tonight's Gate-1 finding (F-HOLO-5/6, `reports/2026-07-17-holonomy-dfa.md`):
- A **naive** global broadcast (`backward_dfa`) does **not** compose credit through depth (L3≈L1,
  0.831).
- The **rotor-chain transported** broadcast (`backward_dfa_transported`) **recovers depth-composition**
  and matches exact backprop (0.894, Δdepth +0.078 vs +0.016), by transporting the credit signal down
  through the **pure inverse-rotor chain** (the return-path holonomy) — *not* the exact gradient
  (alignment stays ~0.41), a depth-composing approximation that reaches the net ceiling.

**That transported-rotor primitive IS the backward of an inter-shell rotor transport.** So the concentric
architecture and the auto-holonomy learning-rule frontier are the *same* move: making the inter-shell
rotors explicit and geometric gives credit a depth-composing path for free. The reframe is the
architectural home of the fix, not a parallel idea.

## 3. The connective primitive (validated)

Forward inter-shell transport `T_{s→s+1}`: a per-node/per-cycle rotor applied to the shell's field
(a `rotor_spike`- or `cayley_rotor`-parameterized rotation), optionally sparsified by divisive
normalization (the `rotor_spike` gain control) so only orientation-selective channels fire radially.

Backward (credit): `RotorMeshNet::backward_dfa_transported` — the credit is transported inward/outward by
the **pure inverse rotors** of the shells it passes, composing through the shell stack. Measured to
recover depth-composition (F-HOLO-6). No stored tape; reuses the forward rotors; global/instantaneous in
the DFA sense.

## 4. Open questions (must be resolved before build)

1. **`rotor_spike` is a per-pixel 2-vector (V1 orientation) op** — input `(gx,gy)` per pixel, von-Mises
   tuning + Carandini–Heeger divisive normalization. "Rotor-spike transportation" of a per-*cycle*
   Clifford field between shells needs a **real definition** (what is the orientation of a cycle-pool
   feature? a bivector angle? the principal axis of its covariance?). This is a genuine op-design task,
   not a plug-in.
2. **The transport is an approximation, not the gradient** (alignment ~0.41). On the easy F-HOLO-1 task
   it reaches the ceiling; whether it holds on a harder task (Gate 2) where the net is *necessary* is
   unknown. The concentric depth must be *earned* on such a task.
3. **Heterogeneous-shell schedule** (which ops in which shell, highway per shell) is unconstrained —
   it needs an ablation, which needs a discriminating task (Gate 2).

## 5. Build gate

**Do not build the full concentric net until Gate 2 exists:** a task where trivial *and* the fixed
closed-form readout *both* fail, so a deep learned representation is *necessary* (candidate:
XOR-of-regional-curvature on the F-HOLO-4 lattice). Only then can the concentric architecture's richness
(heterogeneous shells, varying highway, radial transport) be measured against the linear stack and shown
to earn its complexity. Until then: the transport primitive (`backward_dfa_transported`) is built and
validated; the shell assembly is captured here and deferred.

## 6. Reuse ledger (no new scaffolding until build)

Everything the design needs exists: `gomb_shell`, `clifford_fir`, `cpml_tier`, `hsikan` (+`use_highway`),
`rotor_spike`, `cayley_rotor`, `rotor_holonomy`, and the validated `RotorMeshNet::backward_dfa_transported`.
The build, when authorized, is a *composition* of these under a `ConcentricShellStack` config — not new ops.
