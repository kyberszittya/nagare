//! SBSH dynamic quadtree (Phase 1) — a per-image adaptive spatial partition + a differentiable node
//! mean-pool. The tree spends resolution where a per-pixel **energy** map is high (object edges), the
//! CV `R` locality knob made spatially adaptive. Two pieces:
//!
//! - [`quadtree_build`] — **structural, no backward** (the `cpml_tier` discipline): energy → leaf cells
//!   + a per-pixel leaf assignment. The split is a fixed structural decision; no gradient flows through it.
//! - [`node_pool_forward`] / [`node_pool_backward`] — **differentiable, FD-verified**: pool a per-pixel
//!   feature field into per-cell means, so gradients flow from a per-node head/loss back to the field.
//!
//! Single-image (`n = 1`): the tree is per-image. See `reports/2026-07-11-sbsh-tree-smoke.md` (both hinges
//! validated) and `docs/plans/2026-07-11-sbsh-quadtree/`.
//!
//! # node_pool math
//! With `c = assign[p]`, `N_c = |{p : assign[p] = c}|`:
//! ```text
//!   node[c][j] = (1/N_c) Σ_{p: assign[p]=c} field[p][j]
//!   ∂L/∂field[p][j] = (1/N_{assign[p]}) · ∂L/∂node[assign[p]][j]
//! ```
//! (a commutative group-mean; the adjoint is the broadcast-by-1/N_c — the `scatter_mean` pattern).

/// Dynamic-quadtree build parameters.
#[derive(Clone, Copy, Debug)]
pub struct QuadtreeConfig {
    /// Image side (`energy.len() == g*g`).
    pub g: usize,
    /// Maximum subdivision depth.
    pub max_depth: usize,
    /// A cell smaller than `2*min_side` on its short side is never split.
    pub min_side: usize,
    /// Split a cell only if its mean energy exceeds this threshold.
    pub thresh: f32,
}

/// A built quadtree: leaf cells `[y0, x0, y1, x1)` and the per-pixel leaf index (`g*g`).
pub struct Quadtree {
    /// Leaf boxes, half-open `[y0, x0, y1, x1)`.
    pub cells: Vec<[usize; 4]>,
    /// Per-pixel (row-major) leaf-cell index.
    pub assign: Vec<u32>,
}

/// Mean of `energy` over the half-open cell `[y0,x0,y1,x1)`.
fn mean_energy(energy: &[f32], g: usize, cell: [usize; 4]) -> f32 {
    let [y0, x0, y1, x1] = cell;
    let mut e = 0.0f32;
    for i in y0..y1 {
        for j in x0..x1 {
            e += energy[i * g + j];
        }
    }
    e / (((y1 - y0) * (x1 - x0)).max(1)) as f32
}

#[allow(clippy::too_many_arguments)]
fn recurse(
    energy: &[f32],
    g: usize,
    cell: [usize; 4],
    depth: usize,
    cfg: &QuadtreeConfig,
    cells: &mut Vec<[usize; 4]>,
    assign: &mut [u32],
) {
    let [y0, x0, y1, x1] = cell;
    let side = (y1 - y0).min(x1 - x0);
    if depth < cfg.max_depth
        && side >= 2 * cfg.min_side
        && mean_energy(energy, g, cell) > cfg.thresh
    {
        let my = (y0 + y1) / 2;
        let mx = (x0 + x1) / 2;
        for q in [
            [y0, x0, my, mx],
            [y0, mx, my, x1],
            [my, x0, y1, mx],
            [my, mx, y1, x1],
        ] {
            recurse(energy, g, q, depth + 1, cfg, cells, assign);
        }
    } else {
        let idx = cells.len() as u32;
        cells.push(cell);
        for i in y0..y1 {
            for j in x0..x1 {
                assign[i * g + j] = idx;
            }
        }
    }
}

/// Build a dynamic quadtree from a per-pixel energy map (structural — no backward).
///
/// # Preconditions
/// `energy.len() == cfg.g * cfg.g`, `cfg.g >= 1`, `cfg.min_side >= 1`.
///
/// # Postconditions
/// `assign.len() == g*g`; every pixel is assigned exactly one leaf; `assign[p] < cells.len()` for all `p`;
/// `Σ_c N_c == g*g`.
///
/// # Panics
/// If `energy.len() != cfg.g * cfg.g`.
pub fn quadtree_build(energy: &[f32], cfg: &QuadtreeConfig) -> Quadtree {
    assert_eq!(energy.len(), cfg.g * cfg.g);
    assert!(cfg.g >= 1 && cfg.min_side >= 1);
    let mut cells = Vec::new();
    let mut assign = vec![0u32; cfg.g * cfg.g];
    recurse(
        energy,
        cfg.g,
        [0, 0, cfg.g, cfg.g],
        0,
        cfg,
        &mut cells,
        &mut assign,
    );
    Quadtree { cells, assign }
}

/// Per-cell mean-pool of a per-pixel feature field. Returns `(node_features (n_cells*d), counts (n_cells))`.
///
/// # Preconditions
/// `field.len() == assign.len() * d`, `assign[p] < n_cells` for all `p`.
///
/// # Panics
/// If `field.len() != assign.len() * d`.
pub fn node_pool_forward(
    field: &[f32],
    assign: &[u32],
    n_cells: usize,
    d: usize,
) -> (Vec<f32>, Vec<u32>) {
    assert_eq!(field.len(), assign.len() * d);
    let mut node = vec![0.0f32; n_cells * d];
    let mut counts = vec![0u32; n_cells];
    for (p, &c) in assign.iter().enumerate() {
        let c = c as usize;
        counts[c] += 1;
        for j in 0..d {
            node[c * d + j] += field[p * d + j];
        }
    }
    for c in 0..n_cells {
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                node[c * d + j] *= inv;
            }
        }
    }
    (node, counts)
}

/// Backward of [`node_pool_forward`]. Given `grad_node`, returns `grad_field` (broadcast by `1/N_c`).
///
/// # Preconditions
/// `grad_node.len() == n_cells*d`, `counts.len() == n_cells`, `assign[p] < n_cells`.
///
/// # Panics
/// If the length preconditions do not hold.
pub fn node_pool_backward(
    assign: &[u32],
    grad_node: &[f32],
    counts: &[u32],
    n_cells: usize,
    d: usize,
) -> Vec<f32> {
    assert_eq!(grad_node.len(), n_cells * d);
    assert_eq!(counts.len(), n_cells);
    let mut grad = vec![0.0f32; assign.len() * d];
    for (p, &c) in assign.iter().enumerate() {
        let c = c as usize;
        if counts[c] > 0 {
            let inv = 1.0 / counts[c] as f32;
            for j in 0..d {
                grad[p * d + j] = grad_node[c * d + j] * inv;
            }
        }
    }
    grad
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The tree makes cells FINER where energy is high than where it is flat (the concentration property),
    /// and `assign` covers every pixel exactly once.
    #[test]
    fn concentrates_on_high_energy_region() {
        let g = 32;
        // High energy in the top-left quadrant, ~0 elsewhere.
        let mut energy = vec![0.0f32; g * g];
        for i in 0..g / 2 {
            for j in 0..g / 2 {
                energy[i * g + j] = 1.0;
            }
        }
        // Root mean energy = 0.25 (a quarter is high); thresh 0.1 lets the root split, then only the
        // high-energy quadrant keeps subdividing.
        let cfg = QuadtreeConfig {
            g,
            max_depth: 5,
            min_side: 1,
            thresh: 0.1,
        };
        let qt = quadtree_build(&energy, &cfg);

        // Coverage: every pixel assigned to a valid leaf; counts sum to g*g.
        assert_eq!(qt.assign.len(), g * g);
        assert!(qt.assign.iter().all(|&c| (c as usize) < qt.cells.len()));
        let (_, counts) = node_pool_forward(&vec![0.0f32; g * g], &qt.assign, qt.cells.len(), 1);
        assert_eq!(counts.iter().map(|&c| c as usize).sum::<usize>(), g * g);

        // Mean leaf side: high-energy quadrant vs flat region.
        let side = |c: &[usize; 4]| (c[2] - c[0]) as f32;
        let inq = |c: &[usize; 4]| c[0] < g / 2 && c[1] < g / 2; // top-left
        let hi: Vec<f32> = qt.cells.iter().filter(|c| inq(c)).map(side).collect();
        let lo: Vec<f32> = qt.cells.iter().filter(|c| !inq(c)).map(side).collect();
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len().max(1) as f32;
        assert!(
            mean(&hi) < mean(&lo),
            "high-energy cells should be finer: {} vs {}",
            mean(&hi),
            mean(&lo)
        );
    }

    /// `node_pool` backward matches finite differences (directional-derivative check).
    #[test]
    fn node_pool_backward_matches_fd() {
        let (npx, d, n_cells) = (40usize, 3usize, 5usize);
        // Deterministic assignment covering all cells.
        let assign: Vec<u32> = (0..npx).map(|p| (p % n_cells) as u32).collect();
        let field: Vec<f32> = (0..npx * d).map(|i| (i as f32 * 0.7).sin()).collect();
        let w: Vec<f32> = (0..n_cells * d).map(|k| (k as f32 * 1.3).cos()).collect();
        let (_node, counts) = node_pool_forward(&field, &assign, n_cells, d);
        let grad_node = w.clone(); // L = Σ node·w
        let ana = node_pool_backward(&assign, &grad_node, &counts, n_cells, d);
        let loss = |f: &[f32]| -> f32 {
            let (node, _) = node_pool_forward(f, &assign, n_cells, d);
            node.iter().zip(&w).map(|(a, b)| a * b).sum()
        };
        let eps = 1e-3;
        for dir in 0..4 {
            let u: Vec<f32> = (0..field.len())
                .map(|i| ((i as f32 + dir as f32 * 5.0) * 0.6).sin())
                .collect();
            let a: f32 = ana.iter().zip(&u).map(|(g, ui)| g * ui).sum();
            let fp: Vec<f32> = field.iter().zip(&u).map(|(f, ui)| f + eps * ui).collect();
            let fm: Vec<f32> = field.iter().zip(&u).map(|(f, ui)| f - eps * ui).collect();
            let num = (loss(&fp) - loss(&fm)) / (2.0 * eps);
            assert!(
                (a - num).abs() < 2e-3 + 2e-3 * num.abs(),
                "dir {dir}: {a} vs fd {num}"
            );
        }
    }

    /// Uniform field → every node equals that constant; backward distributes evenly.
    #[test]
    fn uniform_field_pools_to_constant() {
        let assign = vec![0u32, 0, 1, 1, 1];
        let d = 2;
        let field: Vec<f32> = (0..5 * d).map(|_| 3.0).collect();
        let (node, counts) = node_pool_forward(&field, &assign, 2, d);
        assert!(node.iter().all(|&v| (v - 3.0).abs() < 1e-6));
        assert_eq!(counts, vec![2, 3]);
        let grad = node_pool_backward(&assign, &vec![1.0; 2 * d], &counts, 2, d);
        // pixel in cell 0 gets 1/2, cell 1 gets 1/3.
        assert!((grad[0] - 0.5).abs() < 1e-6 && (grad[4] - 1.0 / 3.0).abs() < 1e-6);
    }
}
