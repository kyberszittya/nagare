//! Dihedral rotation-group steering — generalises the single continuous rotor canonicalisation
//! (`vision_quat_conv`) to a finite **dihedral group** `D_n`: `|G| = n` rotations (angles
//! `2πk/n`) optionally × a reflection. This is the building block for
//! - a dihedral **group-convolution** on vision: steer a geometric descriptor to all `|G|`
//!   group frames, apply a shared filter, then group-pool (max/mean) → `D_n`-invariant response;
//! - a dihedral-equivariant **hypergraph convolution**: steer the geometric 3-vector messages of
//!   [`crate::ops::hg_message`] across the group.
//!
//! Each group element is a rotor acting on the `xy`-plane (a z-rotation, i.e. a unit quaternion
//! `cos(α/2)+sin(α/2)k`) with an optional `y→−y` reflection; the `z` component passes through.
//! The exact planar action is used for the **fixed** group elements — the Cayley/Rodrigues
//! parameterisation of [`crate::ops::cayley_rotor`] is singular at `α=π`, so it stays the tool
//! for the *learned/continuous* rotor, not the discrete group.

use std::f32::consts::PI;

/// A dihedral group `D_n`: `n` rotations, plus `n` reflections when `reflect`.
#[derive(Debug, Clone, Copy)]
pub struct DihedralGroup {
    /// Rotation order `n ≥ 1`.
    pub n: usize,
    /// Include the reflection coset (full `D_n`) when true; pure cyclic `C_n` when false.
    pub reflect: bool,
}

impl DihedralGroup {
    /// Construct `D_n` (or `C_n` if `!reflect`).
    ///
    /// # Panics
    /// Panics if `n == 0`.
    pub fn new(n: usize, reflect: bool) -> Self {
        assert!(n >= 1, "dihedral order must be >= 1");
        Self { n, reflect }
    }

    /// Group order `|G|` = `n` (C_n) or `2n` (D_n).
    pub fn order(&self) -> usize {
        self.n * if self.reflect { 2 } else { 1 }
    }

    /// Group elements as `(cos α, sin α, reflect)`, reflection coset last.
    fn elements(&self) -> Vec<(f32, f32, bool)> {
        let mut e = Vec::with_capacity(self.order());
        for r in 0..(if self.reflect { 2 } else { 1 }) {
            for k in 0..self.n {
                let a = 2.0 * PI * k as f32 / self.n as f32;
                e.push((a.cos(), a.sin(), r == 1));
            }
        }
        e
    }
}

/// Steer each 3-vector to all `|G|` group frames → flat `(|G|, n_vec, 3)`.
///
/// Element `g = (cos, sin, refl)` maps `(x, y, z)` by an optional `y→−y` reflection, then the
/// planar rotation by `α`; `z` is unchanged.
///
/// # Preconditions
/// `v.len() == n_vec * 3`.
///
/// # Postconditions
/// Returns `steered` with `steered[(g·n_vec + i)*3 .. +3]` the image of vector `i` under `g`;
/// each image has the same norm as its source (rotations + reflections are isometries).
pub fn dihedral_steer_forward(v: &[f32], group: DihedralGroup, n_vec: usize) -> Vec<f32> {
    assert_eq!(v.len(), n_vec * 3);
    let els = group.elements();
    let mut out = vec![0.0f32; els.len() * n_vec * 3];
    for (gi, &(c, s, refl)) in els.iter().enumerate() {
        let sy = if refl { -1.0 } else { 1.0 };
        for i in 0..n_vec {
            let (x, y, z) = (v[i * 3], sy * v[i * 3 + 1], v[i * 3 + 2]);
            let base = (gi * n_vec + i) * 3;
            out[base] = x * c - y * s;
            out[base + 1] = x * s + y * c;
            out[base + 2] = z;
        }
    }
    out
}

/// Backward of [`dihedral_steer_forward`] → `grad_v` flat `(n_vec, 3)`.
///
/// Each group action is orthogonal, so `grad_v = Σ_g Tᵀ_g · grad_steered_g` — rotate the group
/// slice's grad by `−α`, then reflect `y`, and accumulate.
pub fn dihedral_steer_backward(
    grad_steered: &[f32],
    group: DihedralGroup,
    n_vec: usize,
) -> Vec<f32> {
    let els = group.elements();
    assert_eq!(grad_steered.len(), els.len() * n_vec * 3);
    let mut grad_v = vec![0.0f32; n_vec * 3];
    for (gi, &(c, s, refl)) in els.iter().enumerate() {
        let sy = if refl { -1.0 } else { 1.0 };
        for i in 0..n_vec {
            let base = (gi * n_vec + i) * 3;
            let (gx, gy, gz) = (
                grad_steered[base],
                grad_steered[base + 1],
                grad_steered[base + 2],
            );
            // Rᵀ = R(−α): (x,y) → (x c + y s, −x s + y c); then reflect y.
            let rx = gx * c + gy * s;
            let ry = -gx * s + gy * c;
            grad_v[i * 3] += rx;
            grad_v[i * 3 + 1] += sy * ry;
            grad_v[i * 3 + 2] += gz;
        }
    }
    grad_v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn order_counts() {
        assert_eq!(DihedralGroup::new(4, false).order(), 4); // C_4
        assert_eq!(DihedralGroup::new(4, true).order(), 8); // D_4
        assert_eq!(DihedralGroup::new(6, true).order(), 12);
    }

    #[test]
    fn identity_preserves_and_rotations_are_isometries() {
        let g = DihedralGroup::new(4, true); // D_4 → 90° steps + reflections
        let v = vec![0.6f32, 0.8, 0.0, -0.3, 0.4, 1.0];
        let out = dihedral_steer_forward(&v, g, 2);
        // Element 0 = identity (α=0, no reflection).
        assert!((out[0] - 0.6).abs() < 1e-6 && (out[1] - 0.8).abs() < 1e-6);
        // Every image preserves the source norm.
        let nrm = |a: &[f32]| (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        for gi in 0..g.order() {
            for i in 0..2 {
                let src = nrm(&v[i * 3..i * 3 + 3]);
                let img = nrm(&out[(gi * 2 + i) * 3..(gi * 2 + i) * 3 + 3]);
                assert!((src - img).abs() < 1e-5, "g{gi} vec{i}: {src} vs {img}");
            }
        }
    }

    #[test]
    fn quarter_turn_rotates_the_plane() {
        let g = DihedralGroup::new(4, false); // C_4
        let out = dihedral_steer_forward(&[1.0, 0.0, 0.0], g, 1);
        // element 1 = 90°: (1,0) → (0,1).
        assert!((out[3] - 0.0).abs() < 1e-6 && (out[4] - 1.0).abs() < 1e-6);
        // element 2 = 180°: (1,0) → (-1,0) (singular for Cayley; exact here).
        assert!((out[6] + 1.0).abs() < 1e-6 && (out[7]).abs() < 1e-6);
    }

    #[test]
    fn backward_matches_finite_difference() {
        let g = DihedralGroup::new(3, true); // D_3 → 6 elements
        let v = vec![0.3f32, -0.7, 0.2, 0.5, 0.1, -0.4];
        let steered = dihedral_steer_forward(&v, g, 2);
        let grad_steered = vec![1.0f32; steered.len()];
        let gv = dihedral_steer_backward(&grad_steered, g, 2);
        let eps = 1e-3;
        for (idx, &a) in gv.iter().enumerate() {
            let (mut vp, mut vm) = (v.clone(), v.clone());
            vp[idx] += eps;
            vm[idx] -= eps;
            let sp: f32 = dihedral_steer_forward(&vp, g, 2).iter().sum();
            let sm: f32 = dihedral_steer_forward(&vm, g, 2).iter().sum();
            let num = (sp - sm) / (2.0 * eps);
            assert!((a - num).abs() < 1e-3, "grad_v[{idx}] {a} vs {num}");
        }
    }
}
