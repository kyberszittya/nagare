//! Dihedral group-orbit attention pool — the rotation-equivariant **C-cell** of
//! the Nagare Neocognitron. Steers each geometric feature 3-vector to all `|G|`
//! frames of a dihedral (dyadic) group (`dihedral_steer`), scores each frame
//! with a shared oriented filter, and **soft-max-pools** over the group orbit (a
//! differentiable attention; temperature `τ→0` recovers the exact `D_n`-invariant
//! group-max). Output: a rotation-**invariant** pooled response, plus the
//! equivariant dominant orientation (circular soft-argmax over the group angles).
//!
//! This is the C-cell that pools the S-cell (`conv2d`) responses to build
//! **rotation** tolerance (Fukushima's C-cell built shift tolerance; this adds
//! rotation). The softmax-over-group is a discrete **rotor attention** (the
//! group elements are rotors); the continuous `cayley_rotor` variant is a
//! follow-on. `rotor_spike` supplies the sharp orientation tuning upstream.
//!
//! # Forward
//! `steered = dihedral_steer(v)`; `score[g,i] = ⟨filt, steered[g,i]⟩`;
//! `p[g,i] = softmax_g(score/τ)`; `resp[i] = Σ_g p[g,i]·score[g,i]` (soft group-max,
//! invariant); `orient[i] = atan2(Σ_g p sin α_g, Σ_g p cos α_g)` (equivariant).
//!
//! # Backward (FD-verified, for `resp`)
//! `∂resp/∂score[k,i] = p[k,i]·(1 + (score[k,i]−resp[i])/τ)`; then through the
//! filter (`score = ⟨filt, steered⟩`) and `dihedral_steer_backward`.
//!
//! **No novelty claimed** (group-equivariant CNNs, Cohen & Welling 2016; steerable
//! CNNs, Weiler et al.). The op is the closed-form, FD-verified C-cell.

use crate::ops::dihedral::{dihedral_steer_backward, dihedral_steer_forward, DihedralGroup};
use std::f32::consts::PI;

const MIN_TAU: f32 = 1e-4;

/// Forward output: the invariant pooled `resp`, the equivariant `orient`, plus
/// the state the backward reproduces from.
pub struct GroupPoolOut {
    /// Rotation-invariant pooled response, per input vector (`n_vec`).
    pub resp: Vec<f32>,
    /// Equivariant dominant orientation (radians), per input vector (`n_vec`).
    pub orient: Vec<f32>,
    p: Vec<f32>,       // |G| * n_vec attention weights
    steered: Vec<f32>, // |G| * n_vec * 3
    score: Vec<f32>,   // |G| * n_vec
}

/// Group-orbit attention pool forward. `v` is `n_vec` geometric 3-vectors flat;
/// `filt` is the shared oriented filter (3-vector). See the module docs.
///
/// # Panics
/// If `v.len() != n_vec*3` or `filt.len() != 3`.
pub fn group_pool_forward(v: &[f32], group: DihedralGroup, filt: &[f32], tau: f32) -> GroupPoolOut {
    assert_eq!(filt.len(), 3);
    let n_vec = v.len() / 3;
    assert_eq!(v.len(), n_vec * 3);
    let tau = tau.max(MIN_TAU);
    let go = group.order();
    let steered = dihedral_steer_forward(v, group, n_vec);
    let mut score = vec![0.0f32; go * n_vec];
    for gi in 0..go {
        for i in 0..n_vec {
            let b = (gi * n_vec + i) * 3;
            score[gi * n_vec + i] =
                filt[0] * steered[b] + filt[1] * steered[b + 1] + filt[2] * steered[b + 2];
        }
    }
    let mut p = vec![0.0f32; go * n_vec];
    let mut resp = vec![0.0f32; n_vec];
    let mut orient = vec![0.0f32; n_vec];
    for i in 0..n_vec {
        let mx = (0..go)
            .map(|gi| score[gi * n_vec + i])
            .fold(f32::MIN, f32::max);
        let mut sum = 0.0f32;
        for gi in 0..go {
            let e = ((score[gi * n_vec + i] - mx) / tau).exp();
            p[gi * n_vec + i] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        let (mut r, mut sc, mut ss) = (0.0f32, 0.0f32, 0.0f32);
        for gi in 0..go {
            let pv = p[gi * n_vec + i] * inv;
            p[gi * n_vec + i] = pv;
            r += pv * score[gi * n_vec + i];
            let ang = 2.0 * PI * (gi % group.n) as f32 / group.n as f32;
            sc += pv * ang.cos();
            ss += pv * ang.sin();
        }
        resp[i] = r;
        orient[i] = ss.atan2(sc);
    }
    GroupPoolOut {
        resp,
        orient,
        p,
        steered,
        score,
    }
}

/// Group-orbit attention pool backward, for the invariant `resp`. Given
/// `grad_resp` (`n_vec`), returns `(grad_v (n_vec*3), grad_filt (3))`.
///
/// # Panics
/// If `grad_resp.len() != n_vec`.
pub fn group_pool_backward(
    out: &GroupPoolOut,
    group: DihedralGroup,
    filt: &[f32],
    tau: f32,
    grad_resp: &[f32],
) -> (Vec<f32>, Vec<f32>) {
    let n_vec = out.resp.len();
    assert_eq!(grad_resp.len(), n_vec);
    let tau = tau.max(MIN_TAU);
    let go = group.order();
    let mut grad_steered = vec![0.0f32; go * n_vec * 3];
    let mut grad_filt = [0.0f32; 3];
    for gi in 0..go {
        // `i` is a genuine multi-array index (out.resp/grad_resp + the `gi*n_vec+i`
        // offset into p/score/steered), not a single-slice iteration.
        #[allow(clippy::needless_range_loop)]
        for i in 0..n_vec {
            let idx = gi * n_vec + i;
            // ∂resp/∂score[k,i] = p·(1 + (score−resp)/τ).
            let d = out.p[idx] * (1.0 + (out.score[idx] - out.resp[i]) / tau);
            let gs = grad_resp[i] * d;
            let b = idx * 3;
            for c in 0..3 {
                grad_steered[b + c] = gs * filt[c];
                grad_filt[c] += gs * out.steered[b + c];
            }
        }
    }
    let grad_v = dihedral_steer_backward(&grad_steered, group, n_vec);
    (grad_v, grad_filt.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backward_matches_fd() {
        let group = DihedralGroup::new(4, true); // D_4 → 8 frames
        let v = vec![0.6f32, 0.3, 0.1, -0.4, 0.7, -0.2, 0.2, -0.5, 0.3];
        let filt = [0.8f32, -0.5, 0.3];
        let tau = 0.5f32;
        let out = group_pool_forward(&v, group, &filt, tau);
        let gr = vec![0.7f32, -0.4, 0.5];
        let (gv, gf) = group_pool_backward(&out, group, &filt, tau, &gr);
        let loss = |vv: &[f32], ff: &[f32]| -> f32 {
            group_pool_forward(vv, group, ff, tau)
                .resp
                .iter()
                .zip(&gr)
                .map(|(&r, &g)| r * g)
                .sum()
        };
        let eps = 1e-3;
        let chk = |a: f32, n: f32, w: &str, i: usize| {
            assert!((a - n).abs() < 1e-3 + 2e-2 * n.abs(), "{w}[{i}] {a} vs {n}");
        };
        for i in 0..v.len() {
            let (mut a, mut b) = (v.clone(), v.clone());
            a[i] += eps;
            b[i] -= eps;
            chk(
                gv[i],
                (loss(&a, &filt) - loss(&b, &filt)) / (2.0 * eps),
                "gv",
                i,
            );
        }
        for i in 0..3 {
            let (mut a, mut b) = (filt.to_vec(), filt.to_vec());
            a[i] += eps;
            b[i] -= eps;
            chk(gf[i], (loss(&v, &a) - loss(&v, &b)) / (2.0 * eps), "gf", i);
        }
    }

    #[test]
    fn response_is_group_invariant() {
        // Rotating v by a group element leaves the orbit (group closure) → same resp.
        let group = DihedralGroup::new(6, true);
        let v = vec![0.6f32, 0.8, 0.2];
        let filt = [1.0f32, 0.3, -0.2];
        let base = group_pool_forward(&v, group, &filt, 0.4).resp[0];
        // element 1 = 60° rotation of v.
        let steered = dihedral_steer_forward(&v, group, 1);
        let vr = steered[3..6].to_vec();
        let rot = group_pool_forward(&vr, group, &filt, 0.4).resp[0];
        assert!(
            (base - rot).abs() < 1e-4,
            "not group-invariant: {base} vs {rot}"
        );
    }

    #[test]
    fn small_tau_approaches_group_max() {
        let group = DihedralGroup::new(8, false);
        let v = vec![0.9f32, 0.2, 0.1];
        let filt = [1.0f32, -0.4, 0.2];
        let out = group_pool_forward(&v, group, &filt, 0.02);
        let max_score = (0..group.order())
            .map(|gi| out.score[gi])
            .fold(f32::MIN, f32::max);
        assert!(
            (out.resp[0] - max_score).abs() < 0.02,
            "resp {} vs max {max_score}",
            out.resp[0]
        );
    }
}
