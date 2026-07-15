//! `ScBlock` — one **S/C block** of the Nagare Neocognitron, the stackable
//! rotation-tolerant backbone. It composes the two cells the crate already has,
//! both closed-form and FD-verified, into a single unit with a hand-derived
//! FD-verified backward (no autograd):
//!
//! ```text
//!  x (C_in,H,W)
//!    └─ S-cell: conv2d (C_in → 2K)          [learned oriented feature bank]
//!         └─ split into K oriented units (gx,gy) per location
//!              └─ C-cell: group_pool per unit (dihedral orbit attention)
//!                   └─ resp map (K,H,W)     [rotation-invariant magnitudes]
//! ```
//!
//! The block's output is the **K-channel invariant response map** (`resp`), which
//! is exactly the shape a *following* `ScBlock`'s conv takes as input — so blocks
//! stack: `x → ScBlock → ScBlock → … → head`. This is the joint-discriminative,
//! rotation-tolerant backbone the pose P1 report asked for. The A/B knob is the
//! C-cell group (`DihedralGroup`): `C_n` (rotation-invariant orbit) vs `C_1`
//! (no orbit / orientation-specific baseline) — one field, nothing else changes.
//!
//! # Backward (FD-verified, composed)
//! `grad_resp_map → [group_pool_backward per unit] → grad conv-output → conv2d_backward`.
//! The per-unit oriented pool uses the invariant `resp` path of `group_pool`
//! (its `orient` output is not carried between blocks).
//!
//! **No novelty claimed** — Neocognitron = Fukushima 1980; the Nagare-specific
//! part is the closed-form no-autograd composition and the rotor C-cell.

use crate::ops::conv2d::{conv2d_backward, conv2d_forward, ConvLayer, ConvShape};
use crate::ops::dihedral::DihedralGroup;
use crate::ops::group_pool::{group_pool_backward, group_pool_forward, GroupPoolOut};
use std::f32::consts::PI;

/// Oriented Sobel gradient bank for a `c_in=1`, `3×3`, `2k`-channel S-cell: unit
/// `u` gets a rotated Sobel gradient pair at angle `u·π/k` (channel `2u` along φ,
/// `2u+1` along φ+90°). Flat `(2k, 1, 3, 3)` in `ConvLayer` layout.
///
/// **Why it exists (assimilated from the entropy-top finding, 2026-07-14).**
/// Warm-starting the S-cell oriented makes the response map structured from step
/// 0, so a downstream [`crate::global_entropy_pool_forward`]'s covariance is
/// anisotropic and its `Hs` gradient is informative. A *random* conv gives a
/// near-isotropic covariance → uninformative `Hs` → the conv never receives an
/// edge-forming gradient and training stalls (measured: 2/5 seeds stuck at
/// chance without this; 5/5 reach the target with it). This is the S-cell
/// cold-start guard — see `reports/framework/canonical_findings.json` `F-ENT-2`.
///
/// # Preconditions
/// `k >= 1`. # Postconditions: `out.len() == 2*k*9`.
pub fn oriented_sobel_bank(k: usize) -> Vec<f32> {
    debug_assert!(k >= 1);
    let gx = [-1.0f32, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    let gy = [-1.0f32, -2.0, -1.0, 0.0, 0.0, 0.0, 1.0, 2.0, 1.0];
    let mut w = vec![0.0f32; 2 * k * 9];
    for u in 0..k {
        let phi = u as f32 * PI / k as f32;
        let (cp, sp) = (phi.cos(), phi.sin());
        for t in 0..9 {
            w[(2 * u) * 9 + t] = cp * gx[t] + sp * gy[t];
            w[(2 * u + 1) * 9 + t] = -sp * gx[t] + cp * gy[t];
        }
    }
    w
}

/// One S/C block: a conv S-cell (`C_in → 2K`) feeding `K` oriented-unit C-cell
/// pools over a shared dihedral group.
#[derive(Clone, Debug)]
pub struct ScBlock {
    /// S-cell convolution, `c_out = 2K` (each unit is a `(gx, gy)` channel pair).
    pub conv: ConvLayer,
    /// Per-unit oriented-pool filters, `K × 3` flat.
    pub filt: Vec<f32>,
    /// Number of oriented units `K`.
    pub k: usize,
    /// C-cell symmetry group (`C_n` invariant orbit, or `C_1` baseline).
    pub group: DihedralGroup,
    /// Group-attention temperature.
    pub tau: f32,
}

impl ScBlock {
    /// New block with `k` oriented units over `c_in` input channels, `kh×kw`
    /// conv, zero-padded to preserve spatial size. `filt` starts mildly oriented.
    ///
    /// # Preconditions
    /// `k >= 1`, `kh`/`kw` odd (so `pad = k/2` keeps `H,W`).
    pub fn new(
        c_in: usize,
        k: usize,
        kh: usize,
        kw: usize,
        group: DihedralGroup,
        tau: f32,
        seed: u64,
    ) -> Self {
        debug_assert!(k >= 1, "ScBlock needs at least one oriented unit");
        let conv = ConvLayer::new(c_in, 2 * k, kh, kw, seed);
        // Distinct small oriented seeds per unit so the K units don't start identical.
        let filt = (0..k)
            .flat_map(|u| {
                let s = 0.4 + 0.1 * u as f32;
                [s, -0.2, 0.1]
            })
            .collect();
        ScBlock {
            conv,
            filt,
            k,
            group,
            tau,
        }
    }

    /// Like [`ScBlock::new`] but with the S-cell warm-started to an oriented
    /// Sobel bank ([`oriented_sobel_bank`]) — the canonical cold-start fix for
    /// blocks feeding an entropy pool. Requires `c_in == 1`, `kh == kw == 3`; the
    /// pool filters and everything else are unchanged and the conv stays fully
    /// learnable.
    ///
    /// # Panics
    /// If `c_in != 1` or `kh != 3` or `kw != 3`.
    pub fn new_oriented(
        c_in: usize,
        k: usize,
        kh: usize,
        kw: usize,
        group: DihedralGroup,
        tau: f32,
        seed: u64,
    ) -> Self {
        assert_eq!(c_in, 1, "oriented warm-start requires c_in==1");
        assert_eq!(
            (kh, kw),
            (3, 3),
            "oriented warm-start requires a 3×3 S-cell"
        );
        let mut b = Self::new(c_in, k, kh, kw, group, tau, seed);
        b.conv.w = oriented_sobel_bank(k);
        b
    }

    /// Zero-shaped gradient buffer for this block.
    pub fn zero_grad(&self) -> ScBlockGrad {
        ScBlockGrad {
            conv: self.conv.zero_grad(),
            filt: vec![0.0; self.filt.len()],
        }
    }
}

/// Forward intermediates needed by the backward pass. (`conv2d_backward`
/// recomputes from the block input `x`, so the conv output is not retained.)
pub struct ScBlockCache {
    pools: Vec<GroupPoolOut>,
    oh: usize,
    ow: usize,
}

impl ScBlockCache {
    /// Output spatial dims `(H', W')`.
    pub fn out_hw(&self) -> (usize, usize) {
        (self.oh, self.ow)
    }
}

/// Gradient buffer mirroring [`ScBlock`]'s learnable parameters.
#[derive(Clone, Debug)]
pub struct ScBlockGrad {
    pub conv: ConvLayer,
    pub filt: Vec<f32>,
}

/// Assemble the `(np × 3)` oriented-unit field for unit `u` from the conv output.
fn unit_field(conv_y: &[f32], u: usize, np: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; np * 3];
    let (gx0, gy0) = ((2 * u) * np, (2 * u + 1) * np);
    for p in 0..np {
        v[p * 3] = conv_y[gx0 + p];
        v[p * 3 + 1] = conv_y[gy0 + p];
    }
    v
}

/// `ScBlock` forward. `x` is `(c_in, H, W)` flat; returns the `(K, H', W')`
/// rotation-invariant response map plus the cache for backward.
///
/// # Panics
/// If `x` disagrees with `s`, or `blk.conv.c_out != 2*blk.k`.
///
/// # Postconditions
/// Output length is `k * H' * W'`; each channel is a per-location invariant
/// group-pool response (independent of the input's dihedral frame up to the
/// group's angular resolution).
pub fn sc_block_forward(blk: &ScBlock, x: &[f32], s: ConvShape) -> (Vec<f32>, ScBlockCache) {
    assert_eq!(blk.conv.c_out, 2 * blk.k, "conv c_out must equal 2K");
    assert_eq!(blk.filt.len(), 3 * blk.k, "filt must be K×3");
    let (y, oh, ow) = conv2d_forward(&blk.conv, x, s);
    let np = oh * ow;
    let mut resp_map = vec![0.0f32; blk.k * np];
    let mut pools = Vec::with_capacity(blk.k);
    for u in 0..blk.k {
        let v = unit_field(&y, u, np);
        let gp = group_pool_forward(&v, blk.group, &blk.filt[u * 3..u * 3 + 3], blk.tau);
        resp_map[u * np..u * np + np].copy_from_slice(&gp.resp);
        pools.push(gp);
    }
    (resp_map, ScBlockCache { pools, oh, ow })
}

/// `ScBlock` backward. Given `grad_resp_map` `(K, H', W')`, returns
/// `(grad_x (c_in,H,W), grad_block)`.
///
/// # Panics
/// If `grad_resp_map.len() != k * H' * W'`.
pub fn sc_block_backward(
    blk: &ScBlock,
    x: &[f32],
    s: ConvShape,
    cache: &ScBlockCache,
    grad_resp_map: &[f32],
) -> (Vec<f32>, ScBlockGrad) {
    let np = cache.oh * cache.ow;
    assert_eq!(
        grad_resp_map.len(),
        blk.k * np,
        "grad_resp_map must be K×H'×W'"
    );
    let mut grad_y = vec![0.0f32; 2 * blk.k * np];
    let mut grad_filt = vec![0.0f32; 3 * blk.k];
    for u in 0..blk.k {
        let (gv, gfilt) = group_pool_backward(
            &cache.pools[u],
            blk.group,
            &blk.filt[u * 3..u * 3 + 3],
            blk.tau,
            &grad_resp_map[u * np..u * np + np],
        );
        let (gx0, gy0) = ((2 * u) * np, (2 * u + 1) * np);
        for p in 0..np {
            grad_y[gx0 + p] = gv[p * 3];
            grad_y[gy0 + p] = gv[p * 3 + 1];
        }
        grad_filt[u * 3..u * 3 + 3].copy_from_slice(&gfilt);
    }
    let (grad_x, grad_conv) = conv2d_backward(&blk.conv, x, s, &grad_y);
    (
        grad_x,
        ScBlockGrad {
            conv: grad_conv,
            filt: grad_filt,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Scalar loss `Σ w·resp` and its analytic grads via `sc_block_backward`.
    fn loss_and_grads(
        blk: &ScBlock,
        x: &[f32],
        s: ConvShape,
        w: &[f32],
    ) -> (f32, Vec<f32>, ScBlockGrad) {
        let (resp, cache) = sc_block_forward(blk, x, s);
        let loss = resp.iter().zip(w).map(|(r, wi)| r * wi).sum();
        let (gx, gblk) = sc_block_backward(blk, x, s, &cache, w);
        (loss, gx, gblk)
    }

    fn only_loss(blk: &ScBlock, x: &[f32], s: ConvShape, w: &[f32]) -> f32 {
        let (resp, _) = sc_block_forward(blk, x, s);
        resp.iter().zip(w).map(|(r, wi)| r * wi).sum()
    }

    #[test]
    fn backward_matches_fd() {
        let group = DihedralGroup::new(4, false);
        let mut blk = ScBlock::new(2, 3, 3, 3, group, 0.3, 7);
        let s = ConvShape {
            c_in: 2,
            h: 5,
            w: 5,
            pad: 1,
        };
        let mut xs: u64 = 99;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let x: Vec<f32> = (0..s.c_in * s.h * s.w).map(|_| nx()).collect();
        let (_, oh, ow) = conv2d_forward(&blk.conv, &x, s);
        let w: Vec<f32> = (0..blk.k * oh * ow).map(|_| nx()).collect();
        let (_, gx, gblk) = loss_and_grads(&blk, &x, s, &w);

        let eps = 1e-3f32;
        let rel = |a: f32, n: f32| (a - n).abs() < 2e-2 + 3e-2 * n.abs();

        // grad w.r.t. input
        for i in [0usize, 7, 23, 40] {
            let mut xp = x.clone();
            xp[i] += eps;
            let lp = only_loss(&blk, &xp, s, &w);
            let mut xm = x.clone();
            xm[i] -= eps;
            let lm = only_loss(&blk, &xm, s, &w);
            let num = (lp - lm) / (2.0 * eps);
            assert!(rel(gx[i], num), "grad_x[{i}] ana {} vs fd {num}", gx[i]);
        }
        // grad w.r.t. conv weights
        for i in [0usize, 5, 17, 30] {
            let orig = blk.conv.w[i];
            blk.conv.w[i] = orig + eps;
            let lp = only_loss(&blk, &x, s, &w);
            blk.conv.w[i] = orig - eps;
            let lm = only_loss(&blk, &x, s, &w);
            blk.conv.w[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                rel(gblk.conv.w[i], num),
                "grad_conv.w[{i}] ana {} vs fd {num}",
                gblk.conv.w[i]
            );
        }
        // grad w.r.t. conv bias
        for i in 0..blk.conv.b.len() {
            let orig = blk.conv.b[i];
            blk.conv.b[i] = orig + eps;
            let lp = only_loss(&blk, &x, s, &w);
            blk.conv.b[i] = orig - eps;
            let lm = only_loss(&blk, &x, s, &w);
            blk.conv.b[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                rel(gblk.conv.b[i], num),
                "grad_conv.b[{i}] ana {} vs fd {num}",
                gblk.conv.b[i]
            );
        }
        // grad w.r.t. pool filters
        for i in 0..blk.filt.len() {
            let orig = blk.filt[i];
            blk.filt[i] = orig + eps;
            let lp = only_loss(&blk, &x, s, &w);
            blk.filt[i] = orig - eps;
            let lm = only_loss(&blk, &x, s, &w);
            blk.filt[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                rel(gblk.filt[i], num),
                "grad_filt[{i}] ana {} vs fd {num}",
                gblk.filt[i]
            );
        }
    }

    #[test]
    fn two_block_stack_backward_matches_fd() {
        // x → block1 (1→K1 resp map) → block2 (K1→K2 resp map) → Σ w·resp2.
        let g = DihedralGroup::new(4, false);
        let mut b1 = ScBlock::new(1, 2, 3, 3, g, 0.3, 1);
        let b2 = ScBlock::new(2, 2, 3, 3, g, 0.3, 2);
        let s1 = ConvShape {
            c_in: 1,
            h: 6,
            w: 6,
            pad: 1,
        };
        let mut xs: u64 = 5;
        let mut nx = || {
            xs = xs.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((xs >> 33) as f32) / (u32::MAX as f32) - 0.5
        };
        let x: Vec<f32> = (0..36).map(|_| nx()).collect();

        let fwd = |b1: &ScBlock, x: &[f32]| -> (Vec<f32>, usize, usize) {
            let (r1, c1) = sc_block_forward(b1, x, s1);
            let (oh1, ow1) = c1.out_hw();
            let s2 = ConvShape {
                c_in: b1.k,
                h: oh1,
                w: ow1,
                pad: 1,
            };
            let (r2, c2) = sc_block_forward(&b2, &r1, s2);
            let (oh2, ow2) = c2.out_hw();
            (r2, oh2, ow2)
        };
        let (_r2, oh2, ow2) = fwd(&b1, &x);
        let w: Vec<f32> = (0..b2.k * oh2 * ow2).map(|_| nx()).collect();
        let loss = |b1: &ScBlock, x: &[f32]| -> f32 {
            fwd(b1, x).0.iter().zip(&w).map(|(r, wi)| r * wi).sum()
        };

        // Analytic grad w.r.t. block-1 conv weights, through both blocks.
        let (r1, c1) = sc_block_forward(&b1, &x, s1);
        let (oh1, ow1) = c1.out_hw();
        let s2 = ConvShape {
            c_in: b1.k,
            h: oh1,
            w: ow1,
            pad: 1,
        };
        let (_, c2) = sc_block_forward(&b2, &r1, s2);
        let (grad_r1, _g2) = sc_block_backward(&b2, &r1, s2, &c2, &w);
        let (_gx, g1) = sc_block_backward(&b1, &x, s1, &c1, &grad_r1);

        let eps = 1e-3f32;
        for i in [0usize, 4, 11, 16] {
            let orig = b1.conv.w[i];
            b1.conv.w[i] = orig + eps;
            let lp = loss(&b1, &x);
            b1.conv.w[i] = orig - eps;
            let lm = loss(&b1, &x);
            b1.conv.w[i] = orig;
            let num = (lp - lm) / (2.0 * eps);
            assert!(
                (g1.conv.w[i] - num).abs() < 2e-2 + 3e-2 * num.abs(),
                "stacked grad b1.conv.w[{i}] ana {} vs fd {num}",
                g1.conv.w[i]
            );
        }
    }

    #[test]
    fn oriented_bank_is_structured_not_isotropic() {
        // Guard (F-ENT-2): the oriented warm-start must produce a genuinely
        // oriented S-cell so a downstream entropy pool sees an anisotropic
        // response. Unit 0 is exactly Sobel gx/gy; every unit's two filters are a
        // rotated gradient pair, so each has non-trivial gradient energy.
        let w = oriented_sobel_bank(4);
        assert_eq!(w.len(), 2 * 4 * 9);
        // unit 0 channel 0 == Sobel gx
        assert_eq!(&w[0..9], &[-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0]);
        for u in 0..4 {
            let energy: f32 = w[(2 * u) * 9..(2 * u + 1) * 9].iter().map(|v| v * v).sum();
            assert!(
                energy > 1.0,
                "unit {u} filter is degenerate (energy {energy})"
            );
        }
    }

    #[test]
    fn new_oriented_installs_the_bank() {
        let g = DihedralGroup::new(8, false);
        let b = ScBlock::new_oriented(1, 3, 3, 3, g, 0.3, 5);
        assert_eq!(b.conv.w, oriented_sobel_bank(3));
        assert_eq!(b.conv.c_out, 6);
    }

    #[test]
    fn c1_group_is_orientation_specific() {
        // With C_1 (no orbit) the block still runs and produces a K-channel map.
        let g = DihedralGroup::new(1, false);
        let blk = ScBlock::new(1, 2, 3, 3, g, 0.3, 3);
        let s = ConvShape {
            c_in: 1,
            h: 5,
            w: 5,
            pad: 1,
        };
        let x: Vec<f32> = (0..25).map(|i| (i as f32 * 0.1).sin()).collect();
        let (resp, cache) = sc_block_forward(&blk, &x, s);
        assert_eq!(resp.len(), blk.k * 25);
        assert_eq!(cache.out_hw(), (5, 5));
    }
}
