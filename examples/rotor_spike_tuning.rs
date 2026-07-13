//! Demo — the rotor-spike op's orientation tuning curves, showing that the
//! concentration `kappa` controls tuning width (narrow = spike), like a V1
//! orientation-selective neuron. Dumps JSON for `render_rotor_spike.py`.
//!
//! Run: `cargo run --release --example rotor_spike_tuning -- <out.json>`

use holonomy_learn::rotor_spike_forward;
use std::io::Write;

/// A field of `np` pixels all at orientation `theta0` (unit magnitude).
fn oriented_field(np: usize, theta0: f32) -> Vec<f32> {
    let (s, c) = theta0.sin_cos();
    let mut f = vec![0.0f32; np * 2];
    for p in 0..np {
        f[p * 2] = c;
        f[p * 2 + 1] = s;
    }
    f
}

fn main() {
    let out_path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "reports/figures/rotor-spike-tuning.json".to_string());
    let pi = std::f32::consts::PI;
    let (np, k) = (24usize, 16usize);
    let kappas = [1.0f32, 4.0, 16.0];
    let n_theta = 90usize;

    // Tuning curve of the mu=0 unit (bin 0) as the stimulus orientation sweeps.
    let mut curves: Vec<Vec<f32>> = Vec::new();
    let mut thetas = Vec::new();
    for ti in 0..n_theta {
        thetas.push(pi * ti as f32 / n_theta as f32);
    }
    for &kappa in &kappas {
        let mut row = Vec::with_capacity(n_theta);
        for &t0 in &thetas {
            let y = rotor_spike_forward(&oriented_field(np, t0), 1, np, k, kappa).spike;
            row.push(y[0]); // response of the unit tuned to mu=0
        }
        curves.push(row);
    }

    // Population response (all K bins) to a single oblique stimulus, per kappa —
    // the "spike" sharpening.
    let stim = 3.0 * pi / 8.0;
    let mut pops: Vec<Vec<f32>> = Vec::new();
    for &kappa in &kappas {
        pops.push(rotor_spike_forward(&oriented_field(np, stim), 1, np, k, kappa).spike);
    }

    let f2 = |v: &[f32]| {
        v.iter()
            .map(|x| format!("{x:.5}"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let rows = |m: &[Vec<f32>]| {
        m.iter()
            .map(|r| format!("[{}]", f2(r)))
            .collect::<Vec<_>>()
            .join(",")
    };
    let json = format!(
        "{{\n  \"kappas\": [{}],\n  \"thetas\": [{}],\n  \"curves\": [{}],\n  \"k\": {k}, \"stim\": {stim:.5},\n  \"pops\": [{}]\n}}\n",
        f2(&kappas),
        f2(&thetas),
        rows(&curves),
        rows(&pops),
    );
    if let Some(parent) = std::path::Path::new(&out_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut fh = std::fs::File::create(&out_path).expect("create json");
    fh.write_all(json.as_bytes()).expect("write json");
    println!("rotor-spike tuning: kappas {kappas:?}; wrote {out_path}");
}
