//! AUROC (Mann-Whitney) of the closed-form local learner on the toy tasks.
//!
//! Reports test AUROC per gate on clean and hard (noisy+missing+few-shot) data,
//! so the "good" side of the framework is a measured quality number, not just
//! accuracy. Score per sample = logit[1] - logit[0].
//!
//! Run: `cargo run --release --example auroc_eval`

use holonomy_learn::{
    corrupt_dataset, evaluate_local, make_dataset, Dataset, EntropyPoolLocalLearner, GateMode, Task,
};

/// Rank-based AUROC = P(score(pos) > score(neg)); ties get average rank.
fn auroc(scores: &[f32], labels: &[usize]) -> f64 {
    let n = scores.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| scores[a].total_cmp(&scores[b]));
    // average ranks (1-based), handling ties.
    let mut rank = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && scores[idx[j]] == scores[idx[i]] {
            j += 1;
        }
        let avg = (i + 1 + j) as f64 / 2.0; // average of ranks [i+1 .. j]
        for &k in &idx[i..j] {
            rank[k] = avg;
        }
        i = j;
    }
    let n_pos = labels.iter().filter(|&&y| y == 1).count();
    let n_neg = n - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return f64::NAN;
    }
    let rank_pos: f64 = (0..n).filter(|&k| labels[k] == 1).map(|k| rank[k]).sum();
    (rank_pos - (n_pos * (n_pos + 1)) as f64 / 2.0) / (n_pos * n_neg) as f64
}

fn scores(model: &EntropyPoolLocalLearner, data: &Dataset) -> Vec<f32> {
    let logits = model.predict_dataset(data);
    (0..data.samples)
        .map(|s| logits[2 * s + 1] - logits[2 * s])
        .collect()
}

fn train_eval(task: Task, gate: GateMode, hard: bool) -> (f64, f64) {
    let (n_train, noise, miss) = if hard {
        (32, 0.22, 0.45)
    } else {
        (192, 0.0, 0.0)
    };
    let train = corrupt_dataset(&make_dataset(task, n_train, 32, 53), noise, miss, 20_007);
    let test = corrupt_dataset(&make_dataset(task, 96, 32, 54), noise, miss, 30_007);
    let mut model = EntropyPoolLocalLearner::new_with_gate(71, gate);
    model.train(&train, 50, 32, 0.05, 7);
    let m = evaluate_local(&model, &test);
    (auroc(&scores(&model, &test), &test.y), f64::from(m.acc))
}

fn main() {
    let tasks = [Task::Moons, Task::Spiral, Task::Xor];
    let gates = [
        ("entropy", GateMode::Entropy),
        ("constant", GateMode::Constant),
        ("projection", GateMode::Projection),
    ];
    for &hard in &[false, true] {
        println!(
            "=== {} ===",
            if hard {
                "HARD (noisy+missing+few-shot)"
            } else {
                "CLEAN"
            }
        );
        println!(
            "{:<12} {:>16} {:>16} {:>16}",
            "task", "entropy AUROC", "constant AUROC", "projection AUROC"
        );
        for &task in &tasks {
            let cells: Vec<String> = gates
                .iter()
                .map(|&(_, g)| {
                    let (au, acc) = train_eval(task, g, hard);
                    format!("{au:.3} (acc {acc:.2})")
                })
                .collect();
            println!(
                "{:<12} {:>16} {:>16} {:>16}",
                task.as_str(),
                cells[0],
                cells[1],
                cells[2]
            );
        }
    }
}
