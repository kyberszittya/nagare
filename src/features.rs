use hymeko_clifford::{cayley_to_unit_quat, quat_mul, quat_rotate};

use crate::VERTEX_FEATURES;

/// Channel indices of the raw-geometry group `(x, y, r)`.
pub const GEOMETRY_CHANNELS: [usize; 3] = [0, 1, 2];
/// Channel indices of the rotor-rotated group `(rx, ry)`.
pub const ROTOR_CHANNELS: [usize; 2] = [3, 4];
/// Channel indices of the running-holonomy group `(w, z)` — order-sensitive.
pub const HOLONOMY_CHANNELS: [usize; 2] = [5, 6];

/// Quaternion periodic feature lift for `(x, y)` point sets.
///
/// The lift emits `[x, y, r, rotated_x, rotated_y, holonomy_w, holonomy_z]`.
pub fn quaternion_periodic_features(x: &[f32], batch: usize, points: usize) -> Vec<f32> {
    assert_eq!(x.len(), batch * points * 2);
    let mut out = vec![0.0; batch * points * VERTEX_FEATURES];
    for b in 0..batch {
        let mut hol = [1.0, 0.0, 0.0, 0.0];
        for p in 0..points {
            let src = (b * points + p) * 2;
            let px = x[src];
            let py = x[src + 1];
            let r = (px * px + py * py).sqrt();
            let angle = py.atan2(px);
            let q = cayley_to_unit_quat([0.0, 0.0, 0.5 * angle.sin()]);
            hol = quat_mul(hol, q);
            let rotated = quat_rotate(q, [px, py, r]);
            let dst = (b * points + p) * VERTEX_FEATURES;
            out[dst] = px;
            out[dst + 1] = py;
            out[dst + 2] = r;
            out[dst + 3] = rotated[0];
            out[dst + 4] = rotated[1];
            out[dst + 5] = hol[0];
            out[dst + 6] = hol[3];
        }
    }
    out
}
