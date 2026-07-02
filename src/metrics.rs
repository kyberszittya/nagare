use hymeko_clifford::{Multivector, Signature};

#[derive(Clone, Debug)]
pub struct Metrics {
    pub acc: f32,
    pub loss: f32,
    pub entropy: f32,
    pub clifford_error: f32,
}

#[derive(Clone, Debug)]
pub struct CeOut {
    pub loss: f32,
    pub acc: f32,
    pub entropy: f32,
}

pub fn cross_entropy(logits: &[f32], labels: &[usize]) -> CeOut {
    let mut loss = 0.0;
    let mut correct = 0usize;
    let mut entropy = 0.0;
    for i in 0..labels.len() {
        let (p0, p1) = softmax2(logits[2 * i], logits[2 * i + 1]);
        loss += if labels[i] == 0 { -p0.ln() } else { -p1.ln() };
        correct += usize::from((p1 > p0) == (labels[i] == 1));
        entropy += entropy2(p0, p1);
    }
    CeOut {
        loss: loss / labels.len() as f32,
        acc: correct as f32 / labels.len() as f32,
        entropy: entropy / labels.len() as f32,
    }
}

/// Clifford `Cl(2,0)` probability-vector squared error.
pub fn clifford_probability_error(logits: &[f32], labels: &[usize]) -> f32 {
    assert_eq!(logits.len(), labels.len() * 2);
    let sig = Signature::euclidean(2);
    let mut sum = 0.0;
    for b in 0..labels.len() {
        let (p0, p1) = softmax2(logits[2 * b], logits[2 * b + 1]);
        let mut err = Multivector::zero(2);
        err.components[1] = (p0 - f32::from(labels[b] == 0)) as f64;
        err.components[2] = (p1 - f32::from(labels[b] == 1)) as f64;
        sum += err.geo(&err, &sig).components[0] as f32;
    }
    sum / labels.len() as f32
}

pub fn binary_entropy(p: f32) -> f32 {
    let q = 1.0 - p;
    -(p * p.max(1.0e-12).ln() + q * q.max(1.0e-12).ln()) / std::f32::consts::LN_2
}

pub fn entropy2(p0: f32, p1: f32) -> f32 {
    -(p0 * p0.max(1.0e-12).ln() + p1 * p1.max(1.0e-12).ln()) / std::f32::consts::LN_2
}

pub fn softmax2(a: f32, b: f32) -> (f32, f32) {
    let m = a.max(b);
    let ea = (a - m).exp();
    let eb = (b - m).exp();
    (ea / (ea + eb), eb / (ea + eb))
}
