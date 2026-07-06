//! Mechanism regression for the point-order-shuffle holonomy ablation.
//!
//! The scientific claim of the ablation rests on one fact: a within-sample
//! point-order shuffle perturbs *only* the order-sensitive holonomy channels.
//! This test proves it directly on the pooled descriptor, so a future change
//! that leaked the shuffle into the geometry/rotor channels (or that made the
//! holonomy channels accidentally order-invariant) would fail loudly.

use holonomy_learn::{
    make_dataset, shuffle_point_order, structural_pool_features, Task, STRUCTURAL_FEATURES,
    VERTEX_FEATURES,
};

/// Pooled statistics per channel: mean, std, max, sign-entropy.
const POOL_STATS: usize = 4;
/// Per-point (permutation-invariant once pooled) channel indices.
const GEOMETRY_ROTOR: [usize; 5] = [0, 1, 2, 3, 4];
/// Order-sensitive running-holonomy channel indices.
const HOLONOMY: [usize; 2] = [5, 6];

#[test]
fn shuffle_leaves_geometry_invariant_but_perturbs_holonomy() {
    let data = make_dataset(Task::Spiral, 12, 32, 5);
    let shuffled = shuffle_point_order(&data, 71);
    let clean = structural_pool_features(&data);
    let shuf = structural_pool_features(&shuffled);
    assert_eq!(clean.len(), shuf.len());

    let mut max_geo_dev = 0.0f32;
    let mut max_holonomy_dev = 0.0f32;
    for sample in 0..data.samples {
        let base = sample * STRUCTURAL_FEATURES;
        for stat in 0..POOL_STATS {
            let off = base + stat * VERTEX_FEATURES;
            for &ch in &GEOMETRY_ROTOR {
                max_geo_dev = max_geo_dev.max((clean[off + ch] - shuf[off + ch]).abs());
            }
            for &ch in &HOLONOMY {
                max_holonomy_dev = max_holonomy_dev.max((clean[off + ch] - shuf[off + ch]).abs());
            }
        }
    }

    // Geometry/rotor pooled statistics are permutation-invariant up to f32
    // summation order (max and sign-entropy are bit-exact; mean/std differ only
    // by rounding).
    assert!(
        max_geo_dev < 1.0e-3,
        "geometry/rotor pooled stats must be shuffle-invariant, got dev={max_geo_dev}"
    );
    // Holonomy pooled statistics track a non-commutative quaternion trajectory
    // whose partial products depend on order: they must move far above the
    // geometry rounding floor.
    assert!(
        max_holonomy_dev > 1.0e-2,
        "holonomy pooled stats must change under shuffle, got dev={max_holonomy_dev}"
    );
    assert!(
        max_holonomy_dev > 10.0 * max_geo_dev.max(1.0e-9),
        "holonomy must move much more than geometry: hol={max_holonomy_dev} geo={max_geo_dev}"
    );
}
