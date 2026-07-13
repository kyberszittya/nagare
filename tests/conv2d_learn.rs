//! Integration — a conv2d layer learns a KNOWN filter end to end under MSE:
//! `x → conv2d(learned) → MSE(conv2d(x, known_filter))`. Proves the composed
//! backward (`x-grad unused; kernel/bias ← conv2d_backward ← MSE`) recovers the
//! target filter (loss → ~0), i.e. the S-cell trains.

use holonomy_learn::{adam_step, conv2d_backward, conv2d_forward, AdamState, ConvLayer, ConvShape};
use rand::{Rng, SeedableRng};

#[test]
fn learns_a_known_filter() {
    let s = ConvShape {
        c_in: 1,
        h: 8,
        w: 8,
        pad: 1,
    };
    let (c_in, c_out, kh, kw) = (1, 1, 3, 3);
    // Known target filter: a Sobel-like horizontal edge detector.
    let mut target = ConvLayer::new(c_in, c_out, kh, kw, 0);
    target.w = vec![-1.0, 0.0, 1.0, -2.0, 0.0, 2.0, -1.0, 0.0, 1.0];
    target.b = vec![0.0];

    // A few random input images; targets are the known filter's outputs.
    let mut rng = rand::rngs::StdRng::seed_from_u64(4);
    let imgs: Vec<Vec<f32>> = (0..6)
        .map(|_| {
            (0..s.c_in * s.h * s.w)
                .map(|_| rng.random::<f32>() * 2.0 - 1.0)
                .collect()
        })
        .collect();
    let tgts: Vec<Vec<f32>> = imgs
        .iter()
        .map(|x| conv2d_forward(&target, x, s).0)
        .collect();

    let mut l = ConvLayer::new(c_in, c_out, kh, kw, 7);
    let mut sw = AdamState::new(l.w.len());
    let mut sb = AdamState::new(l.b.len());
    let mse = |l: &ConvLayer| -> f32 {
        let mut e = 0.0f32;
        let mut n = 0usize;
        for (x, t) in imgs.iter().zip(&tgts) {
            let y = conv2d_forward(l, x, s).0;
            for (a, b) in y.iter().zip(t) {
                e += (a - b).powi(2);
                n += 1;
            }
        }
        e / n as f32
    };
    let l0 = mse(&l);
    for _ in 0..1200 {
        let mut gw = vec![0.0f32; l.w.len()];
        let mut gb = vec![0.0f32; l.b.len()];
        let mut n = 0usize;
        for (x, t) in imgs.iter().zip(&tgts) {
            let (y, oh, ow) = conv2d_forward(&l, x, s);
            let go: Vec<f32> = y.iter().zip(t).map(|(&a, &b)| 2.0 * (a - b)).collect();
            let (_gx, gl) = conv2d_backward(&l, x, s, &go);
            for (a, b) in gw.iter_mut().zip(&gl.w) {
                *a += b;
            }
            for (a, b) in gb.iter_mut().zip(&gl.b) {
                *a += b;
            }
            n += oh * ow;
        }
        let inv = 1.0 / n as f32;
        for v in gw.iter_mut() {
            *v *= inv;
        }
        for v in gb.iter_mut() {
            *v *= inv;
        }
        adam_step(&mut l.w, &gw, &mut sw, 0.05);
        adam_step(&mut l.b, &gb, &mut sb, 0.05);
    }
    let l1 = mse(&l);
    assert!(l1 < 1e-4, "conv did not learn the filter: {l0} -> {l1}");
    // The recovered kernel matches the target (up to training tolerance).
    for (a, b) in l.w.iter().zip(&target.w) {
        assert!((a - b).abs() < 0.05, "kernel {a} vs target {b}");
    }
}
