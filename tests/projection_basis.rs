use holonomy_learn::{
    learn_holonomy_projection_basis, make_dataset, project_onto_holonomy_subspace,
    structural_pool_features, Task, LOCAL_FEATURES,
};

#[test]
fn learned_projection_basis_is_finite_and_orthogonal() {
    let data = make_dataset(Task::Moons, 24, 8, 13);
    let structural = structural_pool_features(&data);
    let basis = learn_holonomy_projection_basis(&structural, &data.y);
    assert!(basis.iter().flatten().all(|v| v.is_finite()));
    for (i, lhs) in basis.iter().enumerate() {
        let lhs_norm = lhs.iter().map(|v| v * v).sum::<f32>();
        assert!(lhs_norm <= 1.0001);
        for rhs in basis.iter().skip(i + 1) {
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
    let basis = learn_holonomy_projection_basis(&structural, &data.y);
    let mut phi = vec![0.1; LOCAL_FEATURES];
    phi[3] = 0.4;
    phi[4] = -0.2;
    phi[5] = 0.8;
    phi[6] = 0.3;
    project_onto_holonomy_subspace(&mut phi, &basis);
    assert_eq!(phi.len(), LOCAL_FEATURES);
    assert!(phi.iter().all(|v| v.is_finite()));
}
