use std::{env, fs::File, io::Write, path::PathBuf, str::FromStr};

use holonomy_learn::{
    evaluate_local, forward_timing, make_dataset, run_stress_ablation, Config,
    EntropyPoolLocalLearner, GateMode, Task,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (cfg, out) = parse_args()?;
    let mut local_rows = Vec::new();
    for (idx, &task) in cfg.tasks.iter().enumerate() {
        let train = make_dataset(task, cfg.n_train, cfg.n_points, cfg.seed + idx as u64 * 100);
        let test = make_dataset(
            task,
            cfg.n_test,
            cfg.n_points,
            cfg.seed + idx as u64 * 100 + 1,
        );
        let mut model =
            EntropyPoolLocalLearner::new_with_gate(cfg.seed + idx as u64 * 17, GateMode::Entropy);
        model.train(
            &train,
            cfg.epochs,
            cfg.batch_size,
            cfg.lr,
            cfg.seed + idx as u64,
        );
        let metrics = evaluate_local(&model, &test);
        let timing = forward_timing(&model, &test, 80);
        println!(
            "{} local acc={:.3} loss={:.6} median_us={:.3} params={}",
            task.as_str(),
            metrics.acc,
            metrics.loss,
            timing.median_us_per_sample,
            model.n_params()
        );
        local_rows.push((task, metrics, timing, model.n_params()));
    }

    let stress_rows = run_stress_ablation(&cfg);
    for row in &stress_rows {
        println!(
            "{} {} entropy={:.3}/{:.6} constant={:.3}/{:.6} projection={:.3}/{:.6}",
            row.task.as_str(),
            row.stress.as_str(),
            row.entropy_metrics.acc,
            row.entropy_metrics.loss,
            row.constant_metrics.acc,
            row.constant_metrics.loss,
            row.projection_metrics.acc,
            row.projection_metrics.loss
        );
    }

    if let Some(path) = out {
        write_json(&path, &cfg, &local_rows, &stress_rows)?;
    }
    Ok(())
}

fn parse_args() -> Result<(Config, Option<PathBuf>), String> {
    let mut cfg = Config::default();
    let mut out = None;
    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let key = &args[i];
        let value = args
            .get(i + 1)
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--tasks" => {
                cfg.tasks = value
                    .split(',')
                    .map(Task::from_str)
                    .collect::<Result<Vec<_>, _>>()?
            }
            "--n-train" => cfg.n_train = value.parse().map_err(|_| "bad --n-train".to_string())?,
            "--n-test" => cfg.n_test = value.parse().map_err(|_| "bad --n-test".to_string())?,
            "--n-points" => {
                cfg.n_points = value.parse().map_err(|_| "bad --n-points".to_string())?
            }
            "--epochs" => cfg.epochs = value.parse().map_err(|_| "bad --epochs".to_string())?,
            "--batch-size" => {
                cfg.batch_size = value.parse().map_err(|_| "bad --batch-size".to_string())?
            }
            "--lr" => cfg.lr = value.parse().map_err(|_| "bad --lr".to_string())?,
            "--seed" => cfg.seed = value.parse().map_err(|_| "bad --seed".to_string())?,
            "--out" => out = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown flag {key}")),
        }
        i += 2;
    }
    Ok((cfg, out))
}

fn write_json(
    path: &PathBuf,
    cfg: &Config,
    local_rows: &[(Task, holonomy_learn::Metrics, holonomy_learn::Timing, usize)],
    stress_rows: &[holonomy_learn::StressRow],
) -> std::io::Result<()> {
    let mut file = File::create(path)?;
    writeln!(file, "{{")?;
    writeln!(file, "  \"engine\": \"nagare-holonomy-learn\",")?;
    writeln!(file, "  \"n_train\": {},", cfg.n_train)?;
    writeln!(file, "  \"n_test\": {},", cfg.n_test)?;
    writeln!(file, "  \"n_points\": {},", cfg.n_points)?;
    writeln!(file, "  \"epochs\": {},", cfg.epochs)?;
    writeln!(file, "  \"local_rows\": [")?;
    for (idx, (task, metrics, timing, params)) in local_rows.iter().enumerate() {
        let comma = if idx + 1 == local_rows.len() { "" } else { "," };
        writeln!(file, "    {{")?;
        writeln!(file, "      \"task\": \"{}\",", task.as_str())?;
        writeln!(file, "      \"acc\": {:.6},", metrics.acc)?;
        writeln!(file, "      \"loss\": {:.6},", metrics.loss)?;
        writeln!(
            file,
            "      \"clifford_error\": {:.6},",
            metrics.clifford_error
        )?;
        writeln!(
            file,
            "      \"median_us_per_sample\": {:.6},",
            timing.median_us_per_sample
        )?;
        writeln!(file, "      \"params\": {}", params)?;
        writeln!(file, "    }}{comma}")?;
    }
    writeln!(file, "  ],")?;
    writeln!(file, "  \"stress_rows\": [")?;
    for (idx, row) in stress_rows.iter().enumerate() {
        let comma = if idx + 1 == stress_rows.len() {
            ""
        } else {
            ","
        };
        writeln!(file, "    {{")?;
        writeln!(file, "      \"task\": \"{}\",", row.task.as_str())?;
        writeln!(file, "      \"stress\": \"{}\",", row.stress.as_str())?;
        writeln!(
            file,
            "      \"entropy_loss\": {:.6},",
            row.entropy_metrics.loss
        )?;
        writeln!(
            file,
            "      \"constant_loss\": {:.6},",
            row.constant_metrics.loss
        )?;
        writeln!(
            file,
            "      \"projection_loss\": {:.6},",
            row.projection_metrics.loss
        )?;
        writeln!(
            file,
            "      \"projection_params\": {}",
            row.projection_params
        )?;
        writeln!(file, "    }}{comma}")?;
    }
    writeln!(file, "  ]")?;
    writeln!(file, "}}")?;
    Ok(())
}
