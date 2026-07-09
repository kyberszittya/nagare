//! Discriminating test: does Kochanek-Bartels beat Chebyshev-CR where its TCB tangents
//! *should* matter — sharp local structure — and is the gain real at a matched param budget?
//!
//! The HSiKAN-on-Iris comparison tied (saturated task). This isolates the representational
//! question directly by fitting univariate targets `y=f(x)` on `[-1,1]` under two lenses:
//!
//!   1. **Matched grid** (both `grid=8`): Chebyshev-CR (8 params, fixed CR tangents) vs KB
//!      (8 control + 24 TCB = 32 params). KB ⊇ Catmull-Rom, so it must fit ≥ as well
//!      *everywhere* — this just confirms KB is a strict superset, not a sharp-specific win.
//!   2. **Matched params (~32)**: a *finer-grid* Chebyshev (`grid=cheb_k=32`, 32 params) vs
//!      the same KB (`grid=8`, 32 params). Same budget, spent differently — more control
//!      points (Cheb) vs tangent control (KB). This is the real question: on sharp targets,
//!      is KB's tangent parametrisation a *better use of the same params*?
//!
//! A **smooth** sine is the control target throughout. Verdicts are reported (§3), not gated;
//! median MSE over 4 init seeds.

use holonomy_learn::{
    adam_step, chebyshev_cr_backward, chebyshev_cr_forward, kb_backward, kb_forward, AdamState,
    CatmullRomCache, KbCache,
};
use rand::{rngs::StdRng, Rng, SeedableRng};
use std::f32::consts::PI;

const N: usize = 201;
const EPOCHS: usize = 4000;
const LR: f32 = 0.02;

/// Backward cache for one basis (kept only within an epoch).
enum FitCache {
    Cheb {
        cache: CatmullRomCache,
        control: Vec<f32>,
        basis: Vec<f32>,
    },
    Kb(KbCache),
}

/// Forward output: fitted values + the (basis-specific) backward cache.
type SplineOut = (Vec<f32>, FitCache);

/// A single-channel univariate spline with a packed param buffer (mirrors the HSiKAN packing).
trait Spline {
    fn n_params(&self) -> usize;
    fn forward(&self, params: &[f32], x: &[f32]) -> SplineOut;
    fn backward(&self, params: &[f32], cache: &FitCache, grad_y: &[f32]) -> Vec<f32>;
    fn label(&self) -> String;
}

struct ChebSpline {
    grid: usize,
    cheb_k: usize,
}
impl Spline for ChebSpline {
    fn n_params(&self) -> usize {
        self.cheb_k
    }
    fn forward(&self, params: &[f32], x: &[f32]) -> SplineOut {
        let (y, cache, control, basis) =
            chebyshev_cr_forward(params, x, N, 1, self.grid, self.cheb_k);
        (
            y,
            FitCache::Cheb {
                cache,
                control,
                basis,
            },
        )
    }
    fn backward(&self, _params: &[f32], cache: &FitCache, grad_y: &[f32]) -> Vec<f32> {
        let FitCache::Cheb {
            cache,
            control,
            basis,
        } = cache
        else {
            unreachable!("cheb backward on non-cheb cache")
        };
        chebyshev_cr_backward(control, basis, cache, grad_y, self.cheb_k).grad_coef
    }
    fn label(&self) -> String {
        format!("Cheb g{}k{}", self.grid, self.cheb_k)
    }
}

struct KbSpline {
    grid: usize,
}
impl Spline for KbSpline {
    fn n_params(&self) -> usize {
        self.grid * 4 // (grid) control ++ (grid·3) TCB tangents
    }
    fn forward(&self, params: &[f32], x: &[f32]) -> SplineOut {
        let (control, tcb) = params.split_at(self.grid);
        let (y, cache) = kb_forward(control, tcb, x, N, 1, self.grid);
        (y, FitCache::Kb(cache))
    }
    fn backward(&self, params: &[f32], cache: &FitCache, grad_y: &[f32]) -> Vec<f32> {
        let FitCache::Kb(cache) = cache else {
            unreachable!("kb backward on non-kb cache")
        };
        let (control, tcb) = params.split_at(self.grid);
        let bw = kb_backward(control, tcb, cache, grad_y);
        let mut g = bw.grad_coef;
        g.extend_from_slice(&bw.grad_tcb);
        g
    }
    fn label(&self) -> String {
        format!("KB g{}", self.grid)
    }
}

/// Adam-fit a spline to `target`; returns final MSE.
fn fit(spline: &dyn Spline, x: &[f32], target: &[f32], seed: u64) -> f32 {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut params: Vec<f32> = (0..spline.n_params())
        .map(|_| (rng.random::<f32>() * 2.0 - 1.0) * 0.1)
        .collect();
    let mut state = AdamState::new(params.len());
    let mut mse = 0.0;
    for _ in 0..EPOCHS {
        let (y, cache) = spline.forward(&params, x);
        let mut grad_y = vec![0.0f32; N];
        let mut s = 0.0f32;
        for i in 0..N {
            let e = y[i] - target[i];
            s += e * e;
            grad_y[i] = 2.0 * e / N as f32;
        }
        mse = s / N as f32;
        let gp = spline.backward(&params, &cache, &grad_y);
        adam_step(&mut params, &gp, &mut state, LR);
    }
    mse
}

fn median(mut v: Vec<f32>) -> f32 {
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}
fn med_mse(spline: &dyn Spline, x: &[f32], target: &[f32]) -> f32 {
    median((0..4).map(|s| fit(spline, x, target, s)).collect())
}

/// A named target function and whether it has sharp local structure.
type Target = (&'static str, bool, fn(f32) -> f32);

const TARGETS: [Target; 3] = [
    ("sine  (smooth control)", false, |x| (1.5 * PI * x).sin()),
    ("step  (steep tanh)", true, |x| (10.0 * (x - 0.15)).tanh()),
    ("kink  (V-corner)", true, |x| 2.0 * (x - 0.1).abs() - 1.0),
];

/// Fit both splines to every target; print the MSE table; return KB-sharp-wins count.
fn compare(cheb: &dyn Spline, kb: &dyn Spline, x: &[f32]) -> u32 {
    eprintln!(
        "  {} ({}p)  vs  {} ({}p):",
        cheb.label(),
        cheb.n_params(),
        kb.label(),
        kb.n_params()
    );
    let mut kb_sharp_wins = 0;
    for (name, sharp, f) in TARGETS {
        let target: Vec<f32> = x.iter().map(|&xi| f(xi)).collect();
        let mc = med_mse(cheb, x, &target);
        let mk = med_mse(kb, x, &target);
        eprintln!(
            "    {name:24}  Cheb {mc:.3e}   KB {mk:.3e}   → {} lower (KB/Cheb {:.2}×)",
            if mk < mc { "KB" } else { "Cheb" },
            mk / mc.max(1e-12)
        );
        if sharp && mk < mc {
            kb_sharp_wins += 1;
        }
    }
    kb_sharp_wins
}

#[test]
fn kb_vs_cheb_on_sharp_targets() {
    let x: Vec<f32> = (0..N)
        .map(|i| -1.0 + 2.0 * i as f32 / (N - 1) as f32)
        .collect();
    let kb = KbSpline { grid: 8 }; // 32 params

    eprintln!("Univariate spline fit — median MSE over 4 seeds:");
    eprintln!("[1] matched GRID (KB ⊇ Catmull-Rom — expect KB ≥ everywhere, not sharp-specific):");
    let w_grid = compare(&ChebSpline { grid: 8, cheb_k: 8 }, &kb, &x); // Cheb 8p vs KB 32p

    eprintln!("[2] matched PARAMS (~32 — the real question: better use of the same budget?):");
    let w_param = compare(
        &ChebSpline {
            grid: 32,
            cheb_k: 32,
        },
        &kb,
        &x,
    ); // both 32p

    eprintln!("  verdict: KB lower on sharp targets — matched-grid {w_grid}/2, matched-params {w_param}/2");

    // Sanity gates only (verdicts above are the measurement): every arm fits the smooth
    // control, and the richer Chebyshev (32p) is a real, converged competitor.
    let sine: Vec<f32> = x.iter().map(|&xi| (1.5 * PI * xi).sin()).collect();
    assert!(
        med_mse(
            &ChebSpline {
                grid: 32,
                cheb_k: 32
            },
            &x,
            &sine
        ) < 0.01
    );
    assert!(med_mse(&kb, &x, &sine) < 0.01);
}
