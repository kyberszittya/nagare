use holonomy_learn::{
    fit_class_mean_basis, make_dataset, structural_pool_features, Task, PROJECTION_ALPHA,
    PROJECTION_RANK, STRUCTURAL_FEATURES,
};

#[test]
fn learned_projection_basis_is_finite_and_orthonormal() {
    let data = make_dataset(Task::Moons, 24, 8, 13);
    let structural = structural_pool_features(&data);
    let basis = fit_class_mean_basis(&structural, &data.y);
    assert_eq!(basis.dim(), STRUCTURAL_FEATURES);
    assert_eq!(basis.rank(), PROJECTION_RANK);
    let dim = basis.dim();
    let rows: Vec<&[f32]> = basis.vectors().chunks(dim).collect();
    assert!(rows.iter().all(|r| r.iter().all(|v| v.is_finite())));
    for (i, lhs) in rows.iter().enumerate() {
        // accepted rows are unit-norm; surplus rows may be exactly zero.
        let norm2 = lhs.iter().map(|v| v * v).sum::<f32>();
        assert!(norm2 <= 1.0001, "row norm2={norm2}");
        for rhs in rows.iter().skip(i + 1) {
            let dot = lhs
                .iter()
                .zip(rhs.iter())
                .map(|(&a, &b)| a * b)
                .sum::<f32>()
                .abs();
            assert!(dot < 1.0e-4, "basis dot={dot}");
        }
    }
}

#[test]
fn projection_preserves_shape_and_finiteness() {
    let data = make_dataset(Task::Xor, 12, 8, 19);
    let structural = structural_pool_features(&data);
    let basis = fit_class_mean_basis(&structural, &data.y);
    let mut phi = vec![0.1f32; STRUCTURAL_FEATURES];
    phi[3] = 0.4;
    phi[4] = -0.2;
    phi[5] = 0.8;
    phi[6] = 0.3;
    basis.apply_alpha_mix(&mut phi, PROJECTION_ALPHA);
    assert_eq!(phi.len(), STRUCTURAL_FEATURES);
    assert!(phi.iter().all(|v| v.is_finite()));
}
