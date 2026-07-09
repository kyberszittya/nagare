# Nagare — quaternion convolution generalised to dihedral-equivariant hypergraph convolution

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

Generalised the vision quaternion convolution along the two axes the user asked for —
**hypergraph convolution** and **dihedral rotation groups** — and *proved* the result correct:
the signed hypergraph message-passing (`hg_message`) is **`D_n`-equivariant** under a finite
dihedral rotor group, so a dihedral **group-convolution** over a hypergraph is well-defined and
its group-pool is **`D_n`-invariant**. One primitive (`dihedral_steer`) unifies the vision case
(patch grid = a hypergraph) and the graph case (signed cycle hypergraph).

## The unification

- The vision quat-conv canonicalised an equivariant field (image gradients) by **one continuous
  rotor** (`cayley_rotor`, `θ_p`). That is the `n→∞` / single-frame case.
- The generalisation steers the geometric messages to **all `|G|` frames of a finite dihedral
  group** `D_n` (n rotations × reflection), message-passes each frame through `hg_message`, and
  group-pools → a `D_n`-invariant hypergraph convolution.
- Both are the **same rotor action** on the `xy`-plane; the patch grid and the signed cycle graph
  are both hypergraphs, so the primitive serves vision and graphs identically.

## New op — `src/ops/dihedral.rs`

`DihedralGroup { n, reflect }` (`C_n` or full `D_n`, `|G| = n` or `2n`) + `dihedral_steer_forward
/ _backward`. Each group element is a rotor (z-rotation, i.e. unit quaternion `cos(α/2)+sin(α/2)k`)
with an optional `y→−y` reflection; `z` passes through. The **fixed** group elements use the exact
planar rotor action — `cayley_rotor`'s Rodrigues parameterisation is singular at `α=π` (180°), so
it stays the tool for the *learned/continuous* rotor, not the discrete group. Backward is the
orthogonal-transpose accumulation over the group.

Tests: order counts (`C_4`=4, `D_4`=8, `D_6`=12); identity + isometry (every image preserves
norm); the 90°/180° actions exact (180° being the Cayley singularity, handled cleanly here); and
the backward matches finite difference.

## The generalisation, proven (`tests/dihedral_hypergraph.rs`)

Three properties — the *definition* of a group-equivariant conv, verified not asserted:

1. **`hg_node_to_edge` is `D_n`-equivariant** — `steer_g ∘ hg = hg ∘ steer_g` for all `g ∈ D_4`
   (|Δ| < 1e-5). `hg_message` is a per-component signed weighted sum, and the group acts as a
   fixed linear map on each 3-vector, so they commute.
2. **`hg_edge_to_node` is `D_n`-equivariant** — same, over `C_6`.
3. **The group-pooled hypergraph conv is `D_n`-invariant** — `steer → node→edge → edge→node →
   pool over the group` is unchanged when the input is transformed by any `g ∈ D_4` (the transform
   permutes the group index; an orderless pool is unmoved). |Δ| < 1e-4.

Together: a **dihedral group convolution on the signed hypergraph** is a correct group-equivariant
operator. Applied to the patch-grid-as-hypergraph, it is the dihedral generalisation of the vision
quat-conv; applied to the signed cycle graph, it is a rotation-equivariant signed hypergraph conv.

## Files touched

| file | change |
|---|---|
| `src/ops/dihedral.rs` | **new** — `DihedralGroup` + `dihedral_steer` fwd/bwd + 4 tests |
| `src/ops/mod.rs`, `src/lib.rs` | +mod / +re-export |
| `tests/dihedral_hypergraph.rs` | **new** — 3 equivariance/invariance proofs over `hg_message` |

## CORE / deps

**None.** `dihedral` is standalone (planar rotor action); composes with the existing `hg_message`
kernels. No dependency change.

## Test results / provenance

- `dihedral` 4/4, `dihedral_hypergraph` 3/3; full suite green on Mac (arm64); clippy `-D warnings`
  + fmt clean. Mac-only (kato detached 2026-07-10).
- Property tests are exact/analytic (no seeds, no data). Repo `github.com/kyberszittya/nagare`.

## Open / next

- **Wire the dihedral group-conv into the vision model** (replace the single-θ canonicalisation
  with a `D_n` group-conv on the gradient field) and measure whether the group-conv's steerable
  capacity beats single canonicalisation on rotated shapes / MNIST.
- A learnable per-element filter (group conv proper) with the closed-form backward through
  `dihedral_steer_backward` + `hg_message` backward.
