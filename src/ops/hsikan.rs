//! Highway-SignedKAN (HSiKAN) core as a closed-form Nagare op.
//!
//! Port of the *single* `SignedKANLayer.forward` from
//! `hymeko_neuro/hyperedge/signedkan.py` — the ablation-critical core:
//! a **sign-conditioned** aggregation over each signed hyperedge with an
//! **inner Chebyshev spline**, a **Schmidhuber highway gate** on that
//! spline, a per-sign mean over the edge's vertices, and a **diagonal
//! outer Chebyshev spline** summed across sign branches.
//!
//! Both spline stages reuse the Chebyshev–Catmull-Rom basis already in
//! [`crate::ops::catmull_rom`] (`chebyshev_cr_forward` / `_backward`); no
//! spline basis is re-implemented here. Gradients are hand-derived and
//! closed-form, matching the operator idiom in this crate (no autograd).
//!
//! Forward, per hyperedge `e = (v_1..v_k)` with signs `σ_i ∈ {+1,-1}`:
//! ```text
//!   h_e = Σ_s  outer_s( (1/|I_s|) Σ_{i∈I_s} [ T_i ⊙ inner_s(h_{v_i})
//!                                            + (1-T_i) ⊙ h_{v_i} ] )
//!   T_i = σ(W_T h_{v_i} + b_T),   I_s = { i : σ_i = s }
//! ```
//!
//! # Preconditions (whole op)
//! - `grid >= 4` (Catmull-Rom needs ≥ 4 knots), `cheb_k >= 1`.
//! - `n_branches ∈ {1, 2}` (branch 0 → +1, branch 1 → −1).
//! - Inputs are clamped to `[-1, 1]` inside the spline (inherited); the
//!   trusted conditioning range for the gradient is `h_v, agg ∈ (-1, 1)`.

use crate::ops::catmull_rom::{chebyshev_cr_backward, chebyshev_cr_forward, CatmullRomCache};
use crate::ops::kochanek_bartels::{kb_backward, kb_forward, KbCache};

/// Shape + architecture of one HSiKAN layer evaluation.
#[derive(Debug, Clone, Copy)]
pub struct HsikanConfig {
    /// Number of hyperedges `T`.
    pub n_edges: usize,
    /// Vertices per hyperedge `k` (uniform arity per call).
    pub arity: usize,
    /// Hidden dimension `d`.
    pub hidden: usize,
    /// Sign branches `S ∈ {1, 2}`.
    pub n_branches: usize,
    /// Catmull-Rom control points `G` (≥ 4).
    pub grid: usize,
    /// Chebyshev order `k`.
    pub cheb_k: usize,
    /// Enable the highway skip gate on the inner spline.
    pub use_highway: bool,
    /// Univariate basis for the inner + outer spline activations.
    pub spline_kind: SplineKind,
}

/// Which univariate basis the inner/outer HSiKAN splines use.
///
/// The two bases carry **different learnable parametrisations**, packed into the same
/// `inner_coef`/`outer_coef` buffers per branch (length [`HsikanConfig::branch_len`]):
/// - `ChebyshevCr` — `(d, cheb_k)` Chebyshev coefficients (the default; PyTorch-parity path).
/// - `KochanekBartels` — `(d, grid)` control points followed by `(d, grid, 3)` raw TCB
///   tangents, i.e. `d·grid·4` values per branch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SplineKind {
    /// Chebyshev-parametrised Catmull-Rom.
    ChebyshevCr,
    /// Kochanek-Bartels (tension/continuity/bias) spline.
    KochanekBartels,
}

impl HsikanConfig {
    /// Construct and validate a config.
    ///
    /// # Panics
    /// Panics if `grid < 4`, `cheb_k == 0`, `hidden == 0`, `arity == 0`,
    /// or `n_branches` is not 1 or 2.
    #[allow(clippy::too_many_arguments)] // flat layout surface, mirrors kwargs boundary
    pub fn new(
        n_edges: usize,
        arity: usize,
        hidden: usize,
        n_branches: usize,
        grid: usize,
        cheb_k: usize,
        use_highway: bool,
    ) -> Self {
        assert!(grid >= 4, "Catmull-Rom needs grid >= 4");
        assert!(cheb_k >= 1);
        assert!(hidden >= 1);
        assert!(arity >= 1);
        assert!(n_branches == 1 || n_branches == 2);
        Self {
            n_edges,
            arity,
            hidden,
            n_branches,
            grid,
            cheb_k,
            use_highway,
            spline_kind: SplineKind::ChebyshevCr,
        }
    }

    /// Select the spline basis (builder-style; leaves every other field intact).
    ///
    /// Switching to `KochanekBartels` changes [`Self::branch_len`], so the caller must
    /// size `inner_coef`/`outer_coef` to the KB layout (`d·grid·4` per branch).
    pub fn with_spline_kind(mut self, spline_kind: SplineKind) -> Self {
        self.spline_kind = spline_kind;
        self
    }

    fn n_rows(&self) -> usize {
        self.n_edges * self.arity
    }

    /// Packed learnable length of one sign branch's spline params.
    fn branch_len(&self) -> usize {
        match self.spline_kind {
            SplineKind::ChebyshevCr => self.hidden * self.cheb_k,
            // KB: (d·grid) control points ++ (d·grid·3) raw TCB tangents.
            SplineKind::KochanekBartels => self.hidden * self.grid * 4,
        }
    }
}

/// Learnable parameters (borrowed flat buffers).
#[derive(Debug, Clone, Copy)]
pub struct HsikanParams<'a> {
    /// Inner-spline Chebyshev coefficients, flat `(S, d, cheb_k)`.
    pub inner_coef: &'a [f32],
    /// Outer-spline Chebyshev coefficients, flat `(S, d, cheb_k)`.
    pub outer_coef: &'a [f32],
    /// Highway gate weight, flat `(d, d)` row-major (out, in).
    pub gate_w: &'a [f32],
    /// Highway gate bias, flat `(d)`.
    pub gate_b: &'a [f32],
}

/// Hyperedge incidence (borrowed flat buffers).
#[derive(Debug, Clone, Copy)]
pub struct HsikanEdges<'a> {
    /// Vertex ids per edge, flat `(T, k)`.
    pub vertices: &'a [u32],
    /// Signs per edge, flat `(T, k)`, each `+1` or `-1`.
    pub signs: &'a [i8],
}

/// Cache saved by the forward for the closed-form backward.
#[derive(Debug, Clone)]
pub struct HsikanCache {
    n_nodes: usize,
    h_v: Vec<f32>,
    inner_pre: Vec<Vec<f32>>,
    t_gate: Vec<f32>,
    inner_spline: Vec<BranchSpline>,
    outer_spline: Vec<BranchSpline>,
    basis: Vec<f32>,
    counts: Vec<f32>,
}

/// Per-branch spline forward state saved for the closed-form backward, tagged by basis.
/// `Cheb` also stores its derived Catmull-Rom control points; `Kb` stores the raw control
/// points + TCB tangents its backward re-reads.
#[derive(Debug, Clone)]
enum BranchSpline {
    Cheb {
        cache: CatmullRomCache,
        control: Vec<f32>,
    },
    Kb {
        cache: KbCache,
        control: Vec<f32>,
        tcb: Vec<f32>,
    },
}

/// One branch's spline forward (basis-dispatched). Returns `(y, saved state, basis)`;
/// `basis` is empty for bases that don't use a shared knot basis (KB).
fn branch_spline_forward(
    block: &[f32],
    x: &[f32],
    n: usize,
    d: usize,
    cfg: HsikanConfig,
) -> (Vec<f32>, BranchSpline, Vec<f32>) {
    match cfg.spline_kind {
        SplineKind::ChebyshevCr => {
            let (y, cache, control, basis) =
                chebyshev_cr_forward(block, x, n, d, cfg.grid, cfg.cheb_k);
            (y, BranchSpline::Cheb { cache, control }, basis)
        }
        SplineKind::KochanekBartels => {
            let (control, tcb) = block.split_at(d * cfg.grid);
            let (y, cache) = kb_forward(control, tcb, x, n, d, cfg.grid);
            let spline = BranchSpline::Kb {
                cache,
                control: control.to_vec(),
                tcb: tcb.to_vec(),
            };
            (y, spline, Vec::new())
        }
    }
}

/// One branch's spline backward → `(grad over the packed param block, grad w.r.t. input)`.
fn branch_spline_backward(
    spline: &BranchSpline,
    basis: &[f32],
    grad_y: &[f32],
    cfg: HsikanConfig,
) -> (Vec<f32>, Vec<f32>) {
    match spline {
        BranchSpline::Cheb { cache, control } => {
            let bw = chebyshev_cr_backward(control, basis, cache, grad_y, cfg.cheb_k);
            (bw.grad_coef, bw.grad_x)
        }
        BranchSpline::Kb {
            cache,
            control,
            tcb,
        } => {
            let bw = kb_backward(control, tcb, cache, grad_y);
            let mut block = bw.grad_coef; // (d·grid)
            block.extend_from_slice(&bw.grad_tcb); // ++ (d·grid·3)
            (block, bw.grad_x)
        }
    }
}

/// Gradients returned by the backward pass.
#[derive(Debug, Clone)]
pub struct HsikanBackward {
    /// Gradient w.r.t. node embeddings, flat `(n_nodes, d)`.
    pub grad_x: Vec<f32>,
    /// Gradient w.r.t. inner params, same packed per-branch layout as `inner_coef`
    /// (`(S, d, cheb_k)` for Chebyshev; `(S, d, grid)` control ++ `(S, d, grid, 3)` TCB for KB).
    pub grad_inner_coef: Vec<f32>,
    /// Gradient w.r.t. outer params, same packed per-branch layout as `outer_coef`.
    pub grad_outer_coef: Vec<f32>,
    /// Gradient w.r.t. gate weight, flat `(d, d)`.
    pub grad_gate_w: Vec<f32>,
    /// Gradient w.r.t. gate bias, flat `(d)`.
    pub grad_gate_b: Vec<f32>,
}

/// Branch `s` → its sign value. Valid for `S ≤ 2`.
fn sign_value(branch: usize) -> i8 {
    if branch == 0 {
        1
    } else {
        -1
    }
}

/// Gather `h_v[r, :] = x[vertices[r], :]`, flat `(n_rows, d)`.
fn gather(x: &[f32], vertices: &[u32], n_rows: usize, d: usize) -> Vec<f32> {
    let mut h_v = vec![0.0f32; n_rows * d];
    for (r, &v) in vertices.iter().enumerate() {
        let src = v as usize * d;
        h_v[r * d..r * d + d].copy_from_slice(&x[src..src + d]);
    }
    h_v
}

/// Highway transform gate `T = σ(W h_v + b)`, flat `(n_rows, d)`.
fn compute_gate(gate_w: &[f32], gate_b: &[f32], h_v: &[f32], n_rows: usize, d: usize) -> Vec<f32> {
    let mut t_gate = vec![0.0f32; n_rows * d];
    for r in 0..n_rows {
        let row = &h_v[r * d..r * d + d];
        for c in 0..d {
            let w_row = &gate_w[c * d..c * d + d];
            let z: f32 = gate_b[c] + w_row.iter().zip(row).map(|(w, h)| w * h).sum::<f32>();
            t_gate[r * d + c] = 1.0 / (1.0 + (-z).exp());
        }
    }
    t_gate
}

type InnerForward = (Vec<Vec<f32>>, Vec<BranchSpline>, Vec<f32>);

/// Inner spline for every branch (same input `h_v`, per-branch packed params).
fn inner_forward(inner_coef: &[f32], h_v: &[f32], cfg: HsikanConfig) -> InnerForward {
    let (n_rows, d, bl) = (cfg.n_rows(), cfg.hidden, cfg.branch_len());
    let mut pre = Vec::with_capacity(cfg.n_branches);
    let mut splines = Vec::with_capacity(cfg.n_branches);
    let mut basis = Vec::new();
    for s in 0..cfg.n_branches {
        let block = &inner_coef[s * bl..(s + 1) * bl];
        let (y, spline, b) = branch_spline_forward(block, h_v, n_rows, d, cfg);
        pre.push(y);
        splines.push(spline);
        if !b.is_empty() {
            basis = b;
        }
    }
    (pre, splines, basis)
}

/// Sign-masked per-sign mean over each edge's vertices → `(counts, agg)`.
///
/// `agg` is flat `(T, S, d)`; `counts` is flat `(T, S)` (clamped ≥ 1).
fn aggregate(
    signs: &[i8],
    inner_pre: &[Vec<f32>],
    t_gate: &[f32],
    h_v: &[f32],
    cfg: HsikanConfig,
) -> (Vec<f32>, Vec<f32>) {
    let (t, k, d, s_br) = (cfg.n_edges, cfg.arity, cfg.hidden, cfg.n_branches);
    let mut counts = vec![0.0f32; t * s_br];
    let mut agg = vec![0.0f32; t * s_br * d];
    for ti in 0..t {
        for s in 0..s_br {
            let sv = sign_value(s);
            let mut cnt = 0.0f32;
            for i in 0..k {
                let m = if signs[ti * k + i] == sv { 1.0 } else { 0.0 };
                cnt += m;
                let r = ti * k + i;
                let base = (ti * s_br + s) * d;
                let out = &mut agg[base..base + d];
                let inner_row = &inner_pre[s][r * d..r * d + d];
                let hv_row = &h_v[r * d..r * d + d];
                let gate_row = if cfg.use_highway {
                    Some(&t_gate[r * d..r * d + d])
                } else {
                    None
                };
                accumulate_gated(out, inner_row, gate_row, hv_row, m);
            }
            let cc = cnt.max(1.0);
            counts[ti * s_br + s] = cc;
            for v in agg[(ti * s_br + s) * d..(ti * s_br + s) * d + d].iter_mut() {
                *v /= cc;
            }
        }
    }
    (agg, counts)
}

/// Add `m · [T⊙inner + (1-T)⊙h_v]` (highway) or `m · inner` (plain) into `out`.
///
/// Rows are pre-sliced by the caller; `gate_row` is `None` when highway is off.
fn accumulate_gated(
    out: &mut [f32],
    inner_row: &[f32],
    gate_row: Option<&[f32]>,
    hv_row: &[f32],
    m: f32,
) {
    match gate_row {
        Some(tg) => {
            for (((o, &inn), &t), &hv) in out.iter_mut().zip(inner_row).zip(tg).zip(hv_row) {
                *o += m * (t * inn + (1.0 - t) * hv);
            }
        }
        None => {
            for (o, &inn) in out.iter_mut().zip(inner_row) {
                *o += m * inn;
            }
        }
    }
}

type OuterForward = (Vec<f32>, Vec<BranchSpline>);

/// Diagonal outer spline per branch, summed over branches → `h_e (T, d)`.
fn outer_forward(outer_coef: &[f32], agg: &[f32], cfg: HsikanConfig) -> OuterForward {
    let (t, d, s_br, bl) = (cfg.n_edges, cfg.hidden, cfg.n_branches, cfg.branch_len());
    let mut h_e = vec![0.0f32; t * d];
    let mut splines = Vec::with_capacity(s_br);
    for s in 0..s_br {
        let mut agg_s = vec![0.0f32; t * d];
        for ti in 0..t {
            agg_s[ti * d..ti * d + d]
                .copy_from_slice(&agg[(ti * s_br + s) * d..(ti * s_br + s) * d + d]);
        }
        let block = &outer_coef[s * bl..(s + 1) * bl];
        let (out_s, spline, _basis) = branch_spline_forward(block, &agg_s, t, d, cfg);
        for (acc, v) in h_e.iter_mut().zip(&out_s) {
            *acc += v;
        }
        splines.push(spline);
    }
    (h_e, splines)
}

/// Forward HSiKAN layer. Returns `h_e (T, d)` and a backward cache.
///
/// # Preconditions
/// Buffer lengths match `cfg`: `vertices`/`signs` are `(T·k)`, coef buffers
/// `(S·d·cheb_k)`, `gate_w` `(d·d)`, `gate_b` `(d)`, `x` a multiple of `d`.
///
/// # Postconditions
/// `h_e.len() == T·d`; `cache` carries every intermediate the backward needs.
///
/// # Panics
/// Panics if any precondition on buffer lengths is violated.
pub fn hsikan_forward(
    params: HsikanParams<'_>,
    x: &[f32],
    edges: HsikanEdges<'_>,
    cfg: HsikanConfig,
) -> (Vec<f32>, HsikanCache) {
    let (n_rows, d) = (cfg.n_rows(), cfg.hidden);
    assert_eq!(edges.vertices.len(), n_rows);
    assert_eq!(edges.signs.len(), n_rows);
    assert_eq!(params.inner_coef.len(), cfg.n_branches * cfg.branch_len());
    assert_eq!(params.outer_coef.len(), cfg.n_branches * cfg.branch_len());
    assert_eq!(params.gate_w.len(), d * d);
    assert_eq!(params.gate_b.len(), d);
    assert_eq!(x.len() % d, 0);
    let n_nodes = x.len() / d;

    let h_v = gather(x, edges.vertices, n_rows, d);
    let (inner_pre, inner_spline, basis) = inner_forward(params.inner_coef, &h_v, cfg);
    let t_gate = if cfg.use_highway {
        compute_gate(params.gate_w, params.gate_b, &h_v, n_rows, d)
    } else {
        Vec::new()
    };
    let (agg, counts) = aggregate(edges.signs, &inner_pre, &t_gate, &h_v, cfg);
    let (h_e, outer_spline) = outer_forward(params.outer_coef, &agg, cfg);

    let cache = HsikanCache {
        n_nodes,
        h_v,
        inner_pre,
        t_gate,
        inner_spline,
        outer_spline,
        basis,
        counts,
    };
    (h_e, cache)
}

/// Forward-only HSiKAN over hyperedges in `chunk_t`-sized batches — the deploy /
/// feature-extraction path. Returns `h_e (T, d)` **without a backward cache**, so the
/// peak intermediate stays `O(chunk_t · k · S · d)` instead of the naive
/// `O(T · k · S · d)`: each chunk's cache is dropped before the next. Inherits the
/// PyTorch `HSIKAN_CHUNK_T` streaming cap; the output is bit-identical to
/// [`hsikan_forward`]'s `h_e`.
///
/// # Preconditions
/// Same buffer-length contract as [`hsikan_forward`]; `chunk_t` is clamped to `≥ 1`.
///
/// # Postconditions
/// `h_e.len() == T·d`; peak heap use during the call is bounded by one chunk.
pub fn hsikan_forward_chunked(
    params: HsikanParams<'_>,
    x: &[f32],
    edges: HsikanEdges<'_>,
    cfg: HsikanConfig,
    chunk_t: usize,
) -> Vec<f32> {
    let (t, k, d) = (cfg.n_edges, cfg.arity, cfg.hidden);
    assert_eq!(edges.vertices.len(), t * k);
    assert_eq!(edges.signs.len(), t * k);
    let step = chunk_t.max(1);
    let mut h_e = Vec::with_capacity(t * d);
    let mut start = 0;
    while start < t {
        let end = (start + step).min(t);
        let sub_edges = HsikanEdges {
            vertices: &edges.vertices[start * k..end * k],
            signs: &edges.signs[start * k..end * k],
        };
        let sub_cfg = HsikanConfig {
            n_edges: end - start,
            ..cfg
        };
        // The chunk's cache is dropped here — only h_e is retained.
        let (chunk_he, _cache) = hsikan_forward(params, x, sub_edges, sub_cfg);
        h_e.extend_from_slice(&chunk_he);
        start = end;
    }
    h_e
}

/// Outer-spline backward → `(grad_outer_coef, grad_agg)`.
fn outer_backward(cache: &HsikanCache, grad_he: &[f32], cfg: HsikanConfig) -> (Vec<f32>, Vec<f32>) {
    let (t, d, s_br, bl) = (cfg.n_edges, cfg.hidden, cfg.n_branches, cfg.branch_len());
    let mut grad_outer_coef = vec![0.0f32; s_br * bl];
    let mut grad_agg = vec![0.0f32; t * s_br * d];
    for s in 0..s_br {
        let (grad_block, grad_x) =
            branch_spline_backward(&cache.outer_spline[s], &cache.basis, grad_he, cfg);
        grad_outer_coef[s * bl..(s + 1) * bl].copy_from_slice(&grad_block);
        for ti in 0..t {
            grad_agg[(ti * s_br + s) * d..(ti * s_br + s) * d + d]
                .copy_from_slice(&grad_x[ti * d..ti * d + d]);
        }
    }
    (grad_outer_coef, grad_agg)
}

/// Distribute `grad_agg` for branch `s` back through the mean + highway skip.
///
/// Fills `grad_inner_s` (grad w.r.t. the inner spline output) and accumulates
/// the skip-path vertex grad into `grad_hv` and the gate grad into `grad_t_gate`.
fn distribute_branch(
    s: usize,
    edges: HsikanEdges<'_>,
    cache: &HsikanCache,
    grad_agg: &[f32],
    cfg: HsikanConfig,
    grad_hv: &mut [f32],
    grad_t_gate: &mut [f32],
) -> Vec<f32> {
    let (t, k, d, s_br) = (cfg.n_edges, cfg.arity, cfg.hidden, cfg.n_branches);
    let sv = sign_value(s);
    let mut grad_inner_s = vec![0.0f32; cfg.n_rows() * d];
    for ti in 0..t {
        let cc = cache.counts[ti * s_br + s];
        for i in 0..k {
            if edges.signs[ti * k + i] != sv {
                continue;
            }
            let r = ti * k + i;
            for c in 0..d {
                let g_ig = grad_agg[(ti * s_br + s) * d + c] / cc;
                if cfg.use_highway {
                    let tg = cache.t_gate[r * d + c];
                    grad_inner_s[r * d + c] = g_ig * tg;
                    grad_hv[r * d + c] += g_ig * (1.0 - tg);
                    grad_t_gate[r * d + c] +=
                        g_ig * (cache.inner_pre[s][r * d + c] - cache.h_v[r * d + c]);
                } else {
                    grad_inner_s[r * d + c] = g_ig;
                }
            }
        }
    }
    grad_inner_s
}

/// Inner-spline backward across branches → `(grad_inner_coef, grad_hv, grad_t_gate)`.
fn inner_backward(
    edges: HsikanEdges<'_>,
    cache: &HsikanCache,
    grad_agg: &[f32],
    cfg: HsikanConfig,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let (n_rows, d, s_br, bl) = (cfg.n_rows(), cfg.hidden, cfg.n_branches, cfg.branch_len());
    let mut grad_inner_coef = vec![0.0f32; s_br * bl];
    let mut grad_hv = vec![0.0f32; n_rows * d];
    let mut grad_t_gate = vec![0.0f32; if cfg.use_highway { n_rows * d } else { 0 }];
    for s in 0..s_br {
        let grad_inner_s = distribute_branch(
            s,
            edges,
            cache,
            grad_agg,
            cfg,
            &mut grad_hv,
            &mut grad_t_gate,
        );
        let (grad_block, grad_x) =
            branch_spline_backward(&cache.inner_spline[s], &cache.basis, &grad_inner_s, cfg);
        grad_inner_coef[s * bl..(s + 1) * bl].copy_from_slice(&grad_block);
        for (g, b) in grad_hv.iter_mut().zip(&grad_x) {
            *g += b;
        }
    }
    (grad_inner_coef, grad_hv, grad_t_gate)
}

/// Highway-gate backward: gate param grads + gate-path vertex grad into `grad_hv`.
fn gate_backward(
    params: HsikanParams<'_>,
    cache: &HsikanCache,
    grad_t_gate: &[f32],
    cfg: HsikanConfig,
    grad_hv: &mut [f32],
) -> (Vec<f32>, Vec<f32>) {
    let (n_rows, d) = (cfg.n_rows(), cfg.hidden);
    let mut grad_gate_w = vec![0.0f32; d * d];
    let mut grad_gate_b = vec![0.0f32; d];
    if !cfg.use_highway {
        return (grad_gate_w, grad_gate_b);
    }
    for r in 0..n_rows {
        for c in 0..d {
            let tg = cache.t_gate[r * d + c];
            let gz = grad_t_gate[r * d + c] * tg * (1.0 - tg);
            grad_gate_b[c] += gz;
            let w_row = &params.gate_w[c * d..c * d + d];
            for j in 0..d {
                grad_gate_w[c * d + j] += gz * cache.h_v[r * d + j];
                grad_hv[r * d + j] += gz * w_row[j];
            }
        }
    }
    (grad_gate_w, grad_gate_b)
}

/// Scatter per-row vertex grad back to node embeddings, flat `(n_nodes, d)`.
fn scatter_grad(
    edges: HsikanEdges<'_>,
    grad_hv: &[f32],
    cfg: HsikanConfig,
    n_nodes: usize,
) -> Vec<f32> {
    let d = cfg.hidden;
    let mut grad_x = vec![0.0f32; n_nodes * d];
    for (r, &v) in edges.vertices.iter().enumerate() {
        let dst = v as usize * d;
        for c in 0..d {
            grad_x[dst + c] += grad_hv[r * d + c];
        }
    }
    grad_x
}

/// Backward HSiKAN layer.
///
/// # Preconditions
/// `grad_he.len() == T·d`; `cache` is the value returned by the matching
/// [`hsikan_forward`]; `params`/`edges`/`cfg` match that forward call.
///
/// # Postconditions
/// Returns gradients for the node embeddings and all four parameter groups.
/// When `cfg.use_highway` is false, the two gate gradients are all-zero.
///
/// # Panics
/// Panics if `grad_he.len() != T·d`.
pub fn hsikan_backward(
    params: HsikanParams<'_>,
    edges: HsikanEdges<'_>,
    cache: &HsikanCache,
    grad_he: &[f32],
    cfg: HsikanConfig,
) -> HsikanBackward {
    assert_eq!(grad_he.len(), cfg.n_edges * cfg.hidden);
    let (grad_outer_coef, grad_agg) = outer_backward(cache, grad_he, cfg);
    let (grad_inner_coef, mut grad_hv, grad_t_gate) = inner_backward(edges, cache, &grad_agg, cfg);
    let (grad_gate_w, grad_gate_b) = gate_backward(params, cache, &grad_t_gate, cfg, &mut grad_hv);
    let grad_x = scatter_grad(edges, &grad_hv, cfg, cache.n_nodes);
    HsikanBackward {
        grad_x,
        grad_inner_coef,
        grad_outer_coef,
        grad_gate_w,
        grad_gate_b,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small, fixed, interior-valued fixture: T=2, k=3, d=3, S=2, G=5, cheb_k=4.
    /// Each edge has both signs present so both branches aggregate ≥ 1 vertex.
    struct Fixture {
        cfg: HsikanConfig,
        x: Vec<f32>,
        vertices: Vec<u32>,
        signs: Vec<i8>,
        inner_coef: Vec<f32>,
        outer_coef: Vec<f32>,
        gate_w: Vec<f32>,
        gate_b: Vec<f32>,
    }

    impl Fixture {
        fn new(use_highway: bool) -> Self {
            Self::with_kind(use_highway, SplineKind::ChebyshevCr)
        }

        /// Fixture at a chosen spline basis; params are sized to that basis's packed
        /// per-branch length, so the same generator fills Chebyshev (24) or KB (120).
        fn with_kind(use_highway: bool, spline_kind: SplineKind) -> Self {
            let cfg =
                HsikanConfig::new(2, 3, 3, 2, 5, 4, use_highway).with_spline_kind(spline_kind);
            let n_nodes = 5;
            // Small interior values so both splines stay inside [-1, 1].
            let x: Vec<f32> = (0..n_nodes * 3)
                .map(|i| 0.15 * ((i as f32 * 1.7).sin()))
                .collect();
            let vertices = vec![0u32, 1, 2, 2, 3, 4];
            let signs = vec![1i8, -1, 1, -1, 1, -1];
            let total = cfg.n_branches * cfg.branch_len();
            let inner_coef: Vec<f32> = (0..total).map(|i| 0.1 * ((i as f32 * 0.9).cos())).collect();
            let outer_coef: Vec<f32> = (0..total).map(|i| 0.1 * ((i as f32 * 1.3).sin())).collect();
            let gate_w: Vec<f32> = (0..9).map(|i| 0.05 * ((i as f32 * 0.7).sin())).collect();
            let gate_b = vec![-2.0f32, -1.5, -2.5];
            Self {
                cfg,
                x,
                vertices,
                signs,
                inner_coef,
                outer_coef,
                gate_w,
                gate_b,
            }
        }

        fn params(&self) -> HsikanParams<'_> {
            HsikanParams {
                inner_coef: &self.inner_coef,
                outer_coef: &self.outer_coef,
                gate_w: &self.gate_w,
                gate_b: &self.gate_b,
            }
        }

        fn edges(&self) -> HsikanEdges<'_> {
            HsikanEdges {
                vertices: &self.vertices,
                signs: &self.signs,
            }
        }

        /// Scalar loss = Σ h_e, used as the finite-difference objective.
        fn loss(&self) -> f32 {
            hsikan_forward(self.params(), &self.x, self.edges(), self.cfg)
                .0
                .iter()
                .sum()
        }
    }

    #[test]
    fn forward_shape_and_finite() {
        let f = Fixture::new(true);
        let (h_e, _) = hsikan_forward(f.params(), &f.x, f.edges(), f.cfg);
        assert_eq!(h_e.len(), f.cfg.n_edges * f.cfg.hidden);
        assert!(h_e.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn chunked_matches_naive() {
        // Streaming over T (incl. a mid-batch chunk boundary) is bit-equal to the
        // single-batch forward's h_e.
        let f = Fixture::new(true);
        let (naive, _) = hsikan_forward(f.params(), &f.x, f.edges(), f.cfg);
        for chunk in [1usize, 2, 100] {
            let chunked = hsikan_forward_chunked(f.params(), &f.x, f.edges(), f.cfg, chunk);
            assert_eq!(chunked.len(), naive.len());
            for (a, b) in chunked.iter().zip(&naive) {
                assert!((a - b).abs() < 1e-6, "chunk={chunk}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn highway_off_ignores_gate_params() {
        // With the gate disabled, perturbing gate params must not move the output.
        let mut f = Fixture::new(false);
        let base = f.loss();
        f.gate_w[0] += 0.5;
        f.gate_b[1] -= 0.5;
        assert!(
            (f.loss() - base).abs() < 1e-9,
            "gate params leaked with highway off"
        );
    }

    /// Central-difference check of one analytic gradient buffer against `loss`.
    fn assert_fd(name: &str, analytic: &[f32], mut perturb: impl FnMut(usize, f32) -> f32) {
        let eps = 1e-3;
        for (idx, &a) in analytic.iter().enumerate() {
            let numeric = (perturb(idx, eps) - perturb(idx, -eps)) / (2.0 * eps);
            assert!(
                (a - numeric).abs() < 1e-2,
                "{name}[{idx}]: analytic={a} numeric={numeric}"
            );
        }
    }

    /// Full central-difference sweep of every analytic gradient buffer, parameterised by a
    /// fixture builder so it serves both spline bases.
    fn fd_sweep(build: impl Fn() -> Fixture) {
        let f = build();
        let (h_e, cache) = hsikan_forward(f.params(), &f.x, f.edges(), f.cfg);
        let grad_he = vec![1.0f32; h_e.len()];
        let g = hsikan_backward(f.params(), f.edges(), &cache, &grad_he, f.cfg);

        assert_fd("grad_x", &g.grad_x, |idx, e| {
            let mut ff = build();
            ff.x = f.x.clone();
            ff.x[idx] += e;
            ff.loss()
        });
        assert_fd("grad_inner_coef", &g.grad_inner_coef, |idx, e| {
            let mut ff = build();
            ff.inner_coef = f.inner_coef.clone();
            ff.inner_coef[idx] += e;
            ff.loss()
        });
        assert_fd("grad_outer_coef", &g.grad_outer_coef, |idx, e| {
            let mut ff = build();
            ff.outer_coef = f.outer_coef.clone();
            ff.outer_coef[idx] += e;
            ff.loss()
        });
        assert_fd("grad_gate_w", &g.grad_gate_w, |idx, e| {
            let mut ff = build();
            ff.gate_w = f.gate_w.clone();
            ff.gate_w[idx] += e;
            ff.loss()
        });
        assert_fd("grad_gate_b", &g.grad_gate_b, |idx, e| {
            let mut ff = build();
            ff.gate_b = f.gate_b.clone();
            ff.gate_b[idx] += e;
            ff.loss()
        });
    }

    #[test]
    fn backward_matches_finite_difference() {
        fd_sweep(|| Fixture::new(true));
    }

    #[test]
    fn kb_backward_matches_finite_difference() {
        // Kochanek-Bartels basis: every packed grad (control points + TCB tangents),
        // plus grad_x and gate grads, must match finite difference.
        fd_sweep(|| Fixture::with_kind(true, SplineKind::KochanekBartels));
    }
}
