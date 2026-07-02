use holonomy_learn::{
    evaluate_local, make_dataset, run_stress_ablation, Config, EntropyPoolLocalLearner, GateMode,
    Task,
};

#[test]
fn projection_gate_update_reduces_loss_on_tiny_batch() {
    let data = make_dataset(Task::Xor, 24, 8, 12);
    let mut model = EntropyPoolLocalLearner::new_with_gate(7, GateMode::Projection);
    let before = evaluate_local(&model, &data).loss;
    model.train(&data, 8, 8, 0.05, 7);
    let after = evaluate_local(&model, &data).loss;
    assert!(after < before, "before={before} after={after}");
}

#[test]
fn stress_ablation_smoke_runs() {
    let cfg = Config {
        tasks: vec![Task::Moons],
        n_train: 24,
        n_test: 12,
        n_points: 8,
        epochs: 2,
        batch_size: 8,
        lr: 0.05,
        seed: 3,
    };
    let rows = run_stress_ablation(&cfg);
    assert_eq!(rows.len(), 4);
    for row in rows {
        assert!((0.0..=1.0).contains(&row.entropy_metrics.acc));
        assert!((0.0..=1.0).contains(&row.constant_metrics.acc));
        assert!((0.0..=1.0).contains(&row.projection_metrics.acc));
        assert!(row.projection_params < 2836);
    }
}
