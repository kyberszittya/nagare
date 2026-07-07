# Nagare — a comprehensive review: what makes it fast, and how good it is

Created-at: 2026-07-07 16:20 JST
Repo: `nagare_github` = `github.com/kyberszittya/nagare` (standalone). Evidence host: kato15 (i9-14900KF + RTX 6000). All claims below are backed by the 2026-07-06/07 reports in `reports/` and the 45-test suite.

---

## 1. What Nagare is

Nagare (流れ, "flow") is a **paradigm-native, closed-form ML framework in Rust**
for signed-hypergraph / point-set computation. It is deliberately *not* a tensor
autograd engine. Its four design commitments:

1. **Closed-form (forward, backward) pairs, no autograd tape.** Every operator is
   two plain Rust functions over flat `&[f32]` buffers, with hand-derived
   analytic gradients — there is no computation graph to build, traverse, or free.
2. **Struct-of-arrays cycle pool as the universal datum**, not a dense tensor —
   contiguous, cache-friendly, no pointer chasing.
3. **Multivector-valued gradients** (Clifford algebra), so the sign structure of
   signed graphs is first-class (via `hymeko_clifford`).
4. **MapReduce parallelism over rows/cycles** (commutative, embarrassingly
   parallel) rather than tensor-batch dispatch.

Layers: `ops/` (forward+backward kernels: `linear`, `fused_entropy_update`,
`project_alpha_mix`, `clifford_fir`, `cayley_rotor`, `fsr_mixer`,
`catmull_rom`/Chebyshev, `scatter`, `signed_scatter`, `adam`) → `runtime`
(cycle-pool composition FIR→scatter→linear→BCE + Adam) → the holonomy local
learner (`features` quaternion lift → `pooling` → `projection` gate → linear
readout).

---

## 2. What makes it fast — the exact mechanisms

Each mechanism below is a design decision with a measured consequence.

### 2.1 No autograd tape (structural)
A PyTorch forward builds a dynamic graph and a gradient tape; the runtime carries
a Python interpreter + the Torch/MKL/allocator stack. Nagare's forward is a
straight-line sequence of function calls over `Vec<f32>`. Consequence, measured:
**peak RSS 4.9 MiB vs PyTorch 780 MiB (~160×) on the toy shape** — a static binary
vs a runtime. This is the single most robust Nagare advantage (see §4.3).

### 2.2 Struct-of-arrays, flat buffers
All state is `Vec<f32>` in row-major SoA. The hot loops stream contiguous memory;
there is no `Box<dyn Trait>` pointer chasing. This is what makes the kernels
*vectorizable in the first place* (§2.3).

### 2.3 The `ikj` / SAXPY kernel reorder (the big single-kernel win)
The dominant matmul (`fused_entropy_update`, 4H+1→H over B·P rows) and
`linear_forward` were originally written as a scalar reduction with a **strided**
weight load:
```rust
for j { acc = b[j]; for i { acc += x[i] * w[i*out_dim + j]; } out[j] = acc; }
```
A strided-gather scalar reduction **cannot be autovectorized** by LLVM. Reordered
to `ikj` (SAXPY) form — seed the output row with the bias, then for each input `i`
broadcast `x[i]` and add a *contiguous* W-row into the *contiguous* output row —
the inner loop is element-wise over contiguous memory and **autovectorizes to
AVX FMA**. It is **bit-identical** (same summation order; Rust does not contract
to FMA by default). Measured: **`fused_update` 2.9× faster, whole forward 2.1×**,
zero accuracy change (`2026-07-06-nagare-forward-opt-ikj.md`). *Discipline note:*
this was found by profiling, after two wrong guesses (`linear_forward`, "MKL
magic") — see the profiler `examples/forward_profile.rs`.

### 2.4 Rayon MapReduce that actually scales to all cores
Every row/cycle is independent, so the kernels are `par_chunks_mut` over the
output. This is the decisive advantage on a many-core CPU: **PyTorch/MKL
*oversubscribes* at 32 threads on these op sizes (4–14× *slower* than 8t), while
Nagare's rayon scales cleanly**. Result: at each engine's best thread count,
Nagare-CPU (32t) **beats PyTorch-CPU (8t) across the entire dimensionality sweep,
1.2–2.6×** (`2026-07-07-nagare-pytorch-scaling.md`).

### 2.5 Parallelizing the serial pool (Amdahl)
After §2.3, the only serial stages were the `global_pool` calls. On a fast 32-core
box a serial section dominates the wall (Amdahl). Parallelizing the pool over the
independent batch dimension (bit-identical) collapsed the parity forward
**15.7 → 7.6 µs/sample**, the step that took Nagare from *behind* to
*matching/beating* PyTorch on CPU (`2026-07-07-nagare-pytorch-parity-reversed.md`).
Caveat learned: the *small* 7-channel learner pool is too tiny to parallelize
(16% regression) — parallelize pools only for hidden ≥ ~16.

### 2.6 Operator fusion, no materialized broadcast
`fused_entropy_update` computes the update linear for the implicit row
`[h | pooled | entropy]` **without ever materializing** that wide broadcast
tensor. Measured: per-forward allocation halved (4.81 → 2.43 MB); the fixture
still matches PyTorch to 9.7e-8.

### 2.7 Chebyshev-deploy fast path
Train with the flexible Catmull-Rom / CR-Chebyshev basis, **deploy** with the
cheap Chebyshev evaluation: measured **1.7–1.9× faster than PyTorch and 5 MiB vs
553 MiB** for the deploy classifier (`2026-07-01-nagare-*cheby*`). Train-time
flexibility, deploy-time thrift.

---

## 3. How good it is — accuracy and AUROC

Quality is measured, not asserted. The closed-form **local** update rule is just
`W += lr · gate · φ · (y − p)` — no reverse-mode propagation into the feature
generator.

### 3.1 AUROC (Mann-Whitney), `examples/auroc_eval.rs`, seed 53

| task | clean AUROC | hard AUROC (few-shot+22%noise+45%missing) |
|---|---:|---:|
| moons | 1.000 | 1.000 |
| spiral | 1.000 | 0.952–0.959 |
| xor | 1.000 | 1.000 |

**Clean AUROC = 1.000 everywhere** (perfect ranking) with **60 parameters** and
closed-form local updates. Under heavy corruption AUROC stays **0.95–1.0** even
where thresholded accuracy falls to ~0.86 (spiral) — the ranking is robust. The
three gates (entropy / constant / projection) are AUROC-identical (§3.3).

### 3.2 Closed-form matches backprop, at a fraction of the parameters
The entropy-pool local learner matches a backprop-like baseline's accuracy on
moons/spiral/xor with **60 vs 2,836 parameters (~47× fewer)** and ~23–25× faster
forward (`2026-07-01/02` reports). So "good" here means: *closed-form local
learning is competitive with backpropagation on these tasks*, not merely "it runs."

### 3.3 Honest science: the gate does not do the work (order-shuffle ablation)
A discriminating test (point-order shuffle, `tests/order_shuffle.rs`) showed the
**fitted projection gate does not robustly beat a plain constant gate** (3-seed
median), and the holonomy signal is **spiral-specific and gate-independent**
(`2026-07-06-nagare-order-shuffle-ablation.md`). This is reported as a *result*,
not hidden: the quality comes from the pooled features + closed-form update, not
from the projection-gate mechanism. The AUROC table above corroborates it (gates
tie).

### 3.4 What is NOT yet measured
Real signed-graph / signed-link AUROC (Slashdot / Epinions — the framework's
actual target metric) is the **planned benchmark, not yet wired into the
standalone**. The toy AUROC demonstrates the closed-form learner separates
cleanly; the real-task AUROC is the next evidence to collect.

---

## 4. Benchmark summary (kato15)

### 4.1 Forward parity (hidden=32 toy shape, 8 threads, same fixture)
Nagare **7.6 µs/sample** vs PyTorch 7.6–9.1 → **Nagare matches/beats** on speed,
verified correct to 9.7e-8. Fig `2026-07-07-nagare-pytorch-parity-final-kato15.png`.

### 4.2 Fair CPU-vs-CPU dimensionality scaling (best-of-each thread count)
| (D,H) | Nagare CPU | PyTorch CPU | Nagare adv. |
|---|---:|---:|---:|
| (2,32) | 4.5 | 11.9 | 2.6× |
| (16,64) | 14.5 | 26.7 | 1.8× |
| (64,128) | 41.5 | 51.1 | 1.2× |
| (128,256) | 122 | 157 | 1.3× |

A young hand-written closed-form framework **beats a mature MKL-GEMM ecosystem on
its own substrate across the whole sweep**. Fig
`2026-07-07-nagare-pytorch-scaling-kato15.png`.

### 4.3 Memory
Nagare wins at every scale: ~160× (toy, 4.9 vs 780 MiB) → ~3.2× (D128/H256, 304 vs
982 MiB). The advantage shrinks as activations dominate but never inverts.

---

## 5. Honest bounds & roadmap

- **Matched 8 threads, large dense matmul (D≥64):** PyTorch-CPU edges ahead
  ~1.2–1.3× — decades of BLAS blocking/packing vs hand-vectorized loops. Nagare
  recovers by scaling to all cores (where MKL oversubscribes).
- **GPU:** PyTorch-CUDA is faster in absolute terms, but Nagare has **no GPU
  backend yet** — this is a *reference ceiling and a defined next step*
  (the kernels are embarrassingly parallel and GPU-friendly), not a CPU-vs-GPU
  verdict.
- **Real-task AUROC** (signed-link prediction) is the next quality evidence.
- **Further CPU speed:** a tuned micro-GEMM (matrixmultiply/gemm crate) could
  close the matched-thread gap, at the cost of a dependency.

---

## 6. Reproducibility

- **45 tests green** on Windows and Linux/kato15 (`cargo test`); clippy + fmt
  clean; error handling per contract; bit-identical optimizations proven by the
  frozen seed-53 fixture + FD tests + the order-shuffle mechanism test.
- **Self-contained:** vendored `hymeko_clifford` + `hymeko_graph`
  (`2026-07-06-nagare-vendor-detach-kato15.md`); builds on a fresh Linux host
  from `vendor/` alone.
- Harnesses in the repo: `examples/{forward_profile, scaling_bench, auroc_eval,
  toy_compare}.rs`; PyTorch side `scripts/dev/*` (framework).

## One-line summary

**Nagare is fast because it replaces the autograd tape + tensor-batch dispatch
with contiguous, autovectorized, closed-form kernels that MapReduce cleanly to
every core (where BLAS oversubscribes) — and it is good because that closed-form
local learner reaches AUROC ≈ 1.0 with ~47× fewer parameters than backprop, on a
static binary that uses 3–160× less memory.**
