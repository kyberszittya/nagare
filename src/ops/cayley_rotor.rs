//! Cayley-parameterized quaternion rotor op.
//!
//! The forward maps each 3-component bivector to a unit quaternion and rotates
//! the corresponding 3-vector. The backward is closed form: the vector gradient
//! is the inverse rotation, and the bivector gradient is the expanded
//! quaternion-rotation Jacobian followed by the Cayley normalization Jacobian.

use hymeko_clifford::{cayley_to_unit_quat, quat_conjugate, quat_rotate};

fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}

fn grad_quat_from_rotate(q: [f32; 4], v: [f32; 3], grad_r: [f32; 3]) -> [f32; 4] {
    let [w, x, y, z] = q;
    let [vx, vy, vz] = v;
    let [gx, gy, gz] = grad_r;

    let grad_w = gx * (2.0 * (y * vz - z * vy))
        + gy * (2.0 * (z * vx - x * vz))
        + gz * (2.0 * (x * vy - y * vx));

    let grad_x = gx * (2.0 * y * vy + 2.0 * z * vz)
        + gy * (2.0 * y * vx - 4.0 * x * vy - 2.0 * w * vz)
        + gz * (2.0 * z * vx + 2.0 * w * vy - 4.0 * x * vz);

    let grad_y = gx * (-4.0 * y * vx + 2.0 * x * vy + 2.0 * w * vz)
        + gy * (2.0 * x * vx + 2.0 * z * vz)
        + gz * (-2.0 * w * vx + 2.0 * z * vy - 4.0 * y * vz);

    let grad_z = gx * (-4.0 * z * vx - 2.0 * w * vy + 2.0 * x * vz)
        + gy * (2.0 * w * vx - 4.0 * z * vy + 2.0 * y * vz)
        + gz * (2.0 * x * vx + 2.0 * y * vy);

    [grad_w, grad_x, grad_y, grad_z]
}

/// Forward Cayley-rotor op.
///
/// # Args
/// - `bivec`: Flat `(n, 3)` Rodrigues parameters.
/// - `v`: Flat `(n, 3)` vectors to rotate.
/// - `n`: Number of triples.
///
/// # Preconditions
/// `bivec.len() == n * 3`, `v.len() == n * 3`, and all inputs are finite.
///
/// # Postconditions
/// Returns `(rotated, quats)` with shapes `(n, 3)` and `(n, 4)`. Each saved
/// quaternion is unit length to floating-point tolerance.
///
/// # Panics
/// Panics if input lengths do not match `n`.
pub fn cayley_rotor_forward(bivec: &[f32], v: &[f32], n: usize) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(bivec.len(), n * 3);
    assert_eq!(v.len(), n * 3);
    let mut rotated = vec![0.0; n * 3];
    let mut quats = vec![0.0; n * 4];

    for i in 0..n {
        let b = [bivec[i * 3], bivec[i * 3 + 1], bivec[i * 3 + 2]];
        let vi = [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]];
        let q = cayley_to_unit_quat(b);
        let r = quat_rotate(q, vi);
        rotated[i * 3..i * 3 + 3].copy_from_slice(&r);
        quats[i * 4..i * 4 + 4].copy_from_slice(&q);
    }
    (rotated, quats)
}

/// Backward Cayley-rotor op.
///
/// # Args
/// - `bivec`: Flat `(n, 3)` Rodrigues parameters from forward.
/// - `v`: Flat `(n, 3)` vectors from forward.
/// - `quats`: Flat `(n, 4)` saved unit quaternions from forward.
/// - `grad_r`: Flat `(n, 3)` incoming gradient on rotated vectors.
/// - `n`: Number of triples.
///
/// # Preconditions
/// Buffer lengths must match the documented shapes. `quats` must be the saved
/// forward quaternions for `bivec`.
///
/// # Postconditions
/// Returns `(grad_bivec, grad_v)`, both flat `(n, 3)`.
///
/// # Panics
/// Panics if input lengths do not match `n`.
pub fn cayley_rotor_backward(
    bivec: &[f32],
    v: &[f32],
    quats: &[f32],
    grad_r: &[f32],
    n: usize,
) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(bivec.len(), n * 3);
    assert_eq!(v.len(), n * 3);
    assert_eq!(quats.len(), n * 4);
    assert_eq!(grad_r.len(), n * 3);

    let mut grad_bivec = vec![0.0; n * 3];
    let mut grad_v = vec![0.0; n * 3];

    for i in 0..n {
        let b = [bivec[i * 3], bivec[i * 3 + 1], bivec[i * 3 + 2]];
        let vi = [v[i * 3], v[i * 3 + 1], v[i * 3 + 2]];
        let q = [
            quats[i * 4],
            quats[i * 4 + 1],
            quats[i * 4 + 2],
            quats[i * 4 + 3],
        ];
        let gr = [grad_r[i * 3], grad_r[i * 3 + 1], grad_r[i * 3 + 2]];

        let gv = quat_rotate(quat_conjugate(q), gr);
        grad_v[i * 3..i * 3 + 3].copy_from_slice(&gv);

        let grad_q = grad_quat_from_rotate(q, vi, gr);
        let l = (1.0 + b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        let q_dot_grad = dot4(q, grad_q);
        grad_bivec[i * 3] = (grad_q[1] - q[1] * q_dot_grad) / l;
        grad_bivec[i * 3 + 1] = (grad_q[2] - q[2] * q_dot_grad) / l;
        grad_bivec[i * 3 + 2] = (grad_q[3] - q[3] * q_dot_grad) / l;
    }

    (grad_bivec, grad_v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, tol: f32) {
        assert!((a - b).abs() < tol, "a={a} b={b}");
    }

    #[test]
    fn forward_identity_returns_input_vectors() {
        let bivec = vec![0.0; 6];
        let v = vec![1.0, 2.0, 3.0, -0.5, 0.25, 1.5];
        let (rotated, quats) = cayley_rotor_forward(&bivec, &v, 2);
        assert_eq!(rotated, v);
        assert_eq!(quats, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn forward_known_z_rotation() {
        let bivec = vec![0.0, 0.0, 1.0];
        let v = vec![1.0, 0.0, 0.0];
        let (rotated, _) = cayley_rotor_forward(&bivec, &v, 1);
        assert_close(rotated[0], 0.0, 1e-6);
        assert_close(rotated[1], 1.0, 1e-6);
        assert_close(rotated[2], 0.0, 1e-6);
    }

    #[test]
    fn backward_matches_numerical_for_bivec_and_v() {
        let bivec = vec![0.2, -0.3, 0.4, -0.1, 0.25, -0.35];
        let v = vec![0.7, -1.1, 0.6, -0.2, 0.8, 1.3];
        let n = 2;
        let (rotated, quats) = cayley_rotor_forward(&bivec, &v, n);
        let grad_r = vec![1.0; rotated.len()];
        let (grad_bivec, grad_v) = cayley_rotor_backward(&bivec, &v, &quats, &grad_r, n);
        let eps = 1e-3;

        for idx in 0..bivec.len() {
            let mut plus = bivec.clone();
            plus[idx] += eps;
            let mut minus = bivec.clone();
            minus[idx] -= eps;
            let loss_plus: f32 = cayley_rotor_forward(&plus, &v, n).0.iter().sum();
            let loss_minus: f32 = cayley_rotor_forward(&minus, &v, n).0.iter().sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (grad_bivec[idx] - numeric).abs() < 1e-2,
                "bivec[{idx}]: analytic={} numeric={}",
                grad_bivec[idx],
                numeric
            );
        }

        for idx in 0..v.len() {
            let mut plus = v.clone();
            plus[idx] += eps;
            let mut minus = v.clone();
            minus[idx] -= eps;
            let loss_plus: f32 = cayley_rotor_forward(&bivec, &plus, n).0.iter().sum();
            let loss_minus: f32 = cayley_rotor_forward(&bivec, &minus, n).0.iter().sum();
            let numeric = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (grad_v[idx] - numeric).abs() < 1e-2,
                "v[{idx}]: analytic={} numeric={}",
                grad_v[idx],
                numeric
            );
        }
    }
}
