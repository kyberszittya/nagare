//! **Gate 1 — holonomy-DFA learning-rule diagnostic.** Does the biologically-plausible credit
//! assignment (global entropy broadcast + local inverse-rotor transport) train a deep rotor net
//! as well as sequential exact backprop, and better than a random feedback matrix? A small-data
//! verdict *before* any thought of scaling to Visual-SLAM.
//!
//! Same net (`RotorMeshNet` + entropy readout + logistic), same task (F-HOLO-1 ring-mesh
//! coherent-twist), same optimizer — only the BACKWARD rule differs:
//!   sequential   — exact reverse-mode chain (`RotorMeshNet::backward`). Ground truth.
//!   holonomy-DFA — broadcast the SAME global error to every layer + local exact rotor adjoint
//!                  (`backward_dfa`). The idea. No weight transport, no tape, no feedback weights.
//!   random-DFA   — fixed random projection of the output error per layer + local rotor adjoint
//!                  (`backward_from_rot_grads`). Vanilla DFA control.
//!   trivial      — raw entropy readout, no net. Floor context.
//!
//! Reported (5 seeds, held-out AUROC, depth 3 and 1): does holonomy-DFA match sequential? does
//! depth help under it? does it beat random-DFA? and the **gradient-alignment angle**
//! cos∠(rule update, true gradient) — the mechanistic tell.
//!
//! HONEST (F-HOLO-2): on this substrate a trivial baseline beats the net, so this measures the
//! LEARNING RULE, not task-necessity (as the DFA literature does). Gate 2 (a task where learning
//! is necessary) is a separate rung; SLAM is downstream of both.
//!
//! Run: `cargo run --release --example holonomy_dfa_dissociation [-- --json <path>]`

use holonomy_learn::{
    adam_step, auroc, spectral_reg_value_grad, AdamState, MeshTopology, RotorMeshNet,
    SpectralEntropyConfig,
};

const N: usize = 12; // ring nodes
const D: usize = 3;
const SEEDS: u64 = 5;

#[derive(Clone, Copy, PartialEq)]
enum Rule {
    Sequential,
    HolonomyDfa,
    TransportedDfa,
    RandomDfa,
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 32) as u32 as f32) / 4294967296.0
    }
    fn g(&mut self) -> f32 {
        (-2.0 * self.f().max(1e-7).ln()).sqrt() * (std::f32::consts::TAU * self.f()).cos()
    }
}

/// Ring mesh: N triangular hyperedges {i,i+1,i+2} mod N (F-HOLO-1 substrate).
fn ring_mesh() -> MeshTopology {
    let (mut cyc, mut sig) = (Vec::new(), Vec::new());
    for i in 0..N {
        for j in 0..3 {
            cyc.push(((i + j) % N) as u32);
            sig.push(1.0);
        }
    }
    MeshTopology::new(cyc, sig, vec![1.0 / 3.0f32.sqrt(); N], N, 3)
}

fn rodrigues(axis: [f32; 3], ang: f32, v: [f32; 3]) -> [f32; 3] {
    let (c, s) = (ang.cos(), ang.sin());
    let dot = axis[0] * v[0] + axis[1] * v[1] + axis[2] * v[2];
    let cross = [
        axis[1] * v[2] - axis[2] * v[1],
        axis[2] * v[0] - axis[0] * v[2],
        axis[0] * v[1] - axis[1] * v[0],
    ];
    [
        v[0] * c + cross[0] * s + axis[0] * dot * (1.0 - c),
        v[1] * c + cross[1] * s + axis[1] * dot * (1.0 - c),
        v[2] * c + cross[2] * s + axis[2] * dot * (1.0 - c),
    ]
}

/// One zero-mean sample. label 0 = coherent twist, 1 = isotropic (F-HOLO-1 gen).
fn gen_sample(rng: &mut Rng, label: u8) -> Vec<f32> {
    let mut v = vec![0.0f32; N * D];
    if label == 0 {
        let a = [rng.g(), rng.g(), rng.g()];
        let n = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt().max(1e-6);
        let axis = [a[0] / n, a[1] / n, a[2] / n];
        let u = [rng.g(), rng.g(), rng.g()];
        let theta = 0.4 + 0.3 * rng.f();
        for i in 0..N {
            let r = rodrigues(axis, theta * i as f32, u);
            for c in 0..D {
                v[i * D + c] = r[c] + 0.15 * rng.g();
            }
        }
    } else {
        for x in v.iter_mut() {
            *x = rng.g();
        }
    }
    for c in 0..D {
        let m: f32 = (0..N).map(|i| v[i * D + c]).sum::<f32>() / N as f32;
        for i in 0..N {
            v[i * D + c] -= m;
        }
    }
    v
}

fn entropy_cfg() -> SpectralEntropyConfig {
    SpectralEntropyConfig {
        lam_0: 1.0,
        lam_a: 0.0,
        lam_b: 1.0,
        lam_kl: 0.0,
        ..SpectralEntropyConfig::default()
    }
}

/// Entropy readout: scalar H_norm of the field.
fn readout(field: &[f32], cfg: &SpectralEntropyConfig) -> f32 {
    spectral_reg_value_grad(field, N, D, cfg, 1.0).0
}
/// Gradient of the entropy readout w.r.t. the field, scaled by the upstream scalar grad.
fn readout_backward(field: &[f32], cfg: &SpectralEntropyConfig, up: f32) -> Vec<f32> {
    let (_h, g, _) = spectral_reg_value_grad(field, N, D, cfg, 1.0);
    g.iter().map(|x| x * up).collect()
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn sep(scores: &[f32], labels: &[u8]) -> f64 {
    let a = auroc(scores, labels);
    a.max(1.0 - a)
}

fn gen_set(rng: &mut Rng, n: usize) -> (Vec<Vec<f32>>, Vec<u8>) {
    let mut xs = Vec::new();
    let mut ys = Vec::new();
    for k in 0..n {
        let lab = (k % 2) as u8;
        xs.push(gen_sample(rng, lab));
        ys.push(lab);
    }
    (xs, ys)
}

fn median(v: &[f64]) -> f64 {
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    s[s.len() / 2]
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let (mut dot, mut na, mut nb) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..a.len() {
        dot += a[i] as f64 * b[i] as f64;
        na += (a[i] as f64).powi(2);
        nb += (b[i] as f64).powi(2);
    }
    if na < 1e-12 || nb < 1e-12 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Train one (rule, depth); return (held-out AUROC, mean grad-alignment cos vs the true gradient).
fn train(rule: Rule, depth: usize, seed: u64, epochs: usize, lr: f32) -> (f64, f64) {
    let topo = ring_mesh();
    let cfg = entropy_cfg();
    let mut rng = Rng(101 + seed);
    let (xtr, ytr) = gen_set(&mut rng, 120);
    let (xte, yte) = gen_set(&mut rng, 120);

    let mut bivecs: Vec<Vec<f32>> = (0..depth)
        .map(|_| (0..N * D).map(|_| 0.1 * rng.g()).collect())
        .collect();
    let mut w = 0.0f32;
    let mut b = 0.0f32;

    // fixed random feedback vectors for random-DFA (one (N*D) per layer)
    let fb: Vec<Vec<f32>> = (0..depth)
        .map(|_| {
            (0..N * D)
                .map(|_| rng.g() / (N as f32 * D as f32).sqrt())
                .collect()
        })
        .collect();

    let mut adam_b: Vec<AdamState> = (0..depth).map(|_| AdamState::new(N * D)).collect();
    let mut adam_w = AdamState::new(1);
    let mut adam_bias = AdamState::new(1);

    let mut align_sum = 0.0f64;
    let mut align_cnt = 0usize;

    for _ep in 0..epochs {
        let mut gb: Vec<Vec<f32>> = (0..depth).map(|_| vec![0.0f32; N * D]).collect();
        let mut gb_seq: Vec<Vec<f32>> = (0..depth).map(|_| vec![0.0f32; N * D]).collect();
        let (mut gw, mut gbias) = (0.0f32, 0.0f32);

        for (x, &y) in xtr.iter().zip(&ytr) {
            let net = RotorMeshNet::new(&topo, bivecs.clone());
            let (out, cache) = net.forward(x);
            let feat = readout(&out, &cfg);
            let logit = w * feat + b;
            let dlogit = sigmoid(logit) - y as f32;
            gw += dlogit * feat;
            gbias += dlogit;
            let grad_field = readout_backward(&out, &cfg, dlogit * w);

            // the true gradient (for alignment), always
            let (seq_g, _gv0) = net.backward(&cache, &grad_field);
            // the rule's gradient
            let rule_g = match rule {
                Rule::Sequential => seq_g.clone(),
                Rule::HolonomyDfa => net.backward_dfa(&cache, &grad_field),
                Rule::TransportedDfa => net.backward_dfa_transported(&cache, &grad_field),
                Rule::RandomDfa => {
                    let rot_grads: Vec<Vec<f32>> = (0..depth)
                        .map(|l| fb[l].iter().map(|bi| bi * dlogit).collect())
                        .collect();
                    net.backward_from_rot_grads(&cache, &rot_grads)
                }
            };
            for l in 0..depth {
                for i in 0..N * D {
                    gb[l][i] += rule_g[l][i];
                    gb_seq[l][i] += seq_g[l][i];
                }
            }
        }
        // alignment of this epoch's batch gradient with the true gradient
        if rule != Rule::Sequential {
            let flat = |g: &[Vec<f32>]| -> Vec<f32> { g.concat() };
            align_sum += cosine(&flat(&gb), &flat(&gb_seq));
            align_cnt += 1;
        }
        // Adam update (rule gradient)
        let m = xtr.len() as f32;
        for l in 0..depth {
            for gi in gb[l].iter_mut() {
                *gi /= m;
            }
            adam_step(&mut bivecs[l], &gb[l], &mut adam_b[l], lr);
        }
        let mut ww = [w];
        adam_step(&mut ww, &[gw / m], &mut adam_w, lr);
        w = ww[0];
        let mut bb = [b];
        adam_step(&mut bb, &[gbias / m], &mut adam_bias, lr);
        b = bb[0];
    }

    // eval
    let scores: Vec<f32> = xte
        .iter()
        .map(|x| {
            let net = RotorMeshNet::new(&topo, bivecs.clone());
            let (out, _) = net.forward(x);
            w * readout(&out, &cfg) + b
        })
        .collect();
    let a = sep(&scores, &yte);
    let align = if align_cnt > 0 {
        align_sum / align_cnt as f64
    } else {
        1.0
    };
    (a, align)
}

/// Trivial floor: raw entropy of the input field + logistic, no net.
fn train_trivial(seed: u64) -> f64 {
    let cfg = entropy_cfg();
    let mut rng = Rng(101 + seed);
    let (xtr, ytr) = gen_set(&mut rng, 120);
    let (xte, yte) = gen_set(&mut rng, 120);
    let (mut w, mut b) = (0.0f32, 0.0f32);
    let (mut sw, mut sb) = (AdamState::new(1), AdamState::new(1));
    for _ in 0..300 {
        let (mut gw, mut gbias) = (0.0f32, 0.0f32);
        for (x, &y) in xtr.iter().zip(&ytr) {
            let feat = readout(x, &cfg);
            let dlogit = sigmoid(w * feat + b) - y as f32;
            gw += dlogit * feat;
            gbias += dlogit;
        }
        let m = xtr.len() as f32;
        let mut ww = [w];
        adam_step(&mut ww, &[gw / m], &mut sw, 0.05);
        w = ww[0];
        let mut bb = [b];
        adam_step(&mut bb, &[gbias / m], &mut sb, 0.05);
        b = bb[0];
    }
    let scores: Vec<f32> = xte.iter().map(|x| w * readout(x, &cfg) + b).collect();
    sep(&scores, &yte)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let json_path = args
        .iter()
        .position(|a| a == "--json")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let t0 = std::time::Instant::now();
    let epochs = 200;
    let lr = 0.05;

    println!(
        "== Gate 1: holonomy-DFA learning-rule diagnostic (N={N}, {SEEDS} seeds, Adam lr={lr}) =="
    );
    println!("   substrate = F-HOLO-1 ring-mesh coherent-twist (measures the RULE, not task-necessity)\n");

    let rules = [
        ("sequential (exact)", Rule::Sequential),
        ("holonomy-DFA (naive)", Rule::HolonomyDfa),
        ("transported-DFA", Rule::TransportedDfa),
        ("random-DFA", Rule::RandomDfa),
    ];
    // main table: AUROC at depth 3 and 1, plus alignment (depth 3)
    println!(
        "  {:<20} {:>9} {:>9} {:>12}",
        "rule", "L=3 AUROC", "L=1 AUROC", "align(L=3)"
    );
    let mut rows: Vec<(String, f64, f64, f64)> = vec![];
    for (name, rule) in rules {
        let a3: Vec<(f64, f64)> = (0..SEEDS).map(|s| train(rule, 3, s, epochs, lr)).collect();
        let a1: Vec<f64> = (0..SEEDS)
            .map(|s| train(rule, 1, s, epochs, lr).0)
            .collect();
        let auc3 = median(&a3.iter().map(|x| x.0).collect::<Vec<_>>());
        let auc1 = median(&a1);
        let align = median(&a3.iter().map(|x| x.1).collect::<Vec<_>>());
        rows.push((name.to_string(), auc3, auc1, align));
        println!(
            "  {name:<20} {auc3:>9.3} {auc1:>9.3} {align:>12.3}  [{:.1}s]",
            t0.elapsed().as_secs_f32()
        );
    }
    let triv = median(&(0..SEEDS).map(train_trivial).collect::<Vec<_>>());
    println!(
        "  {:<20} {:>9} {:>9} {:>12}",
        "trivial (no net)", "—", "—", "—"
    );
    println!("  (trivial raw-entropy floor/context: {triv:.3})");

    // verdict — the question is whether the TRANSPORTED broadcast recovers depth-composition
    // that the naive broadcast (F-HOLO-5) lacked.
    let get = |name: &str| -> (f64, f64, f64) {
        let r = rows.iter().find(|r| r.0.starts_with(name)).unwrap();
        (r.1, r.2, r.3) // (L3, L1, align)
    };
    let (seq, _, _) = get("sequential");
    let (naive, naive1, naive_align) = get("holonomy-DFA");
    let (trans, trans1, trans_align) = get("transported-DFA");
    let (rand, _, _) = get("random-DFA");
    println!("\n== VERDICT — does transported (rotor-chain) transport recover depth? ==");
    let depth_naive = naive - naive1;
    let depth_trans = trans - trans1;
    println!(
        "  naive holonomy-DFA:  L3 {naive:.3}  L1 {naive1:.3}  Δdepth {depth_naive:+.3}  align {naive_align:.3}"
    );
    println!(
        "  transported-DFA:     L3 {trans:.3}  L1 {trans1:.3}  Δdepth {depth_trans:+.3}  align {trans_align:.3}"
    );
    println!("  bounds: sequential {seq:.3} (align 1.0)  ·  random {rand:.3} (align ~0)");
    let recovers_depth = depth_trans >= depth_naive + 0.03;
    let better_align = trans_align >= naive_align + 0.05;
    let closes_gap = trans >= naive + 0.02;
    println!(
        "\n  (1) transported uses depth more than naive: Δ {depth_trans:+.3} vs {depth_naive:+.3} => {}",
        if recovers_depth { "YES" } else { "no" }
    );
    println!(
        "  (2) better gradient alignment: {trans_align:.3} vs {naive_align:.3} => {}",
        if better_align { "YES" } else { "no" }
    );
    println!(
        "  (3) closes the gap to sequential: {trans:.3} vs naive {naive:.3} (seq {seq:.3}) => {}",
        if closes_gap { "YES" } else { "no" }
    );
    let pass = recovers_depth || better_align;
    println!(
        "\n  RESULT => {}",
        if pass {
            "rotor-chain transport RECOVERS depth-composition — the inter-shell transport primitive works (proceed to concentric design + Gate 2)"
        } else {
            "rotor-chain transport does NOT recover depth — the mesh-Jacobian coupling matters; naive transport is insufficient"
        }
    );
    println!("\n== done in {:.1}s ==", t0.elapsed().as_secs_f32());

    if let Some(path) = json_path {
        let rows_json: Vec<String> = rows
            .iter()
            .map(|(n, a3, a1, al)| {
                format!("{{\"rule\":\"{n}\",\"auroc_l3\":{a3:.4},\"auroc_l1\":{a1:.4},\"align_l3\":{al:.4}}}")
            })
            .collect();
        let out = format!(
            "{{\n  \"task\":\"holonomy_dfa_gate1_ring_mesh\",\"N\":{N},\"seeds\":{SEEDS},\
             \"epochs\":{epochs},\"lr\":{lr},\n  \"trivial_floor\":{triv:.4},\
             \"gate_pass\":{pass},\n  \"rules\":[{}]\n}}\n",
            rows_json.join(",")
        );
        std::fs::write(&path, out).expect("write json");
        println!("wrote {path}");
    }
}
