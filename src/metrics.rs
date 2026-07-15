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

/// Area under the ROC curve (Mann–Whitney U form) for binary `labels` ranked by
/// `scores`. Returns `0.5` for a degenerate (single-class) set. Canonical eval
/// utility — consolidated from ~7 duplicated example copies (2026-07-15).
///
/// # Preconditions
/// `scores.len() == labels.len()`; `labels` in `{0,1}`.
pub fn auroc(scores: &[f32], labels: &[u8]) -> f64 {
    debug_assert_eq!(scores.len(), labels.len());
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[a]
            .partial_cmp(&scores[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let (mut rank_sum, mut n_pos) = (0.0f64, 0u64);
    for (r, &i) in idx.iter().enumerate() {
        if labels[i] == 1 {
            rank_sum += (r + 1) as f64;
            n_pos += 1;
        }
    }
    let n_neg = scores.len() as u64 - n_pos;
    if n_pos == 0 || n_neg == 0 {
        return 0.5;
    }
    (rank_sum - (n_pos * (n_pos + 1) / 2) as f64) / (n_pos * n_neg) as f64
}

#[cfg(test)]
mod auroc_tests {
    use super::auroc;

    #[test]
    fn perfect_and_chance_and_degenerate() {
        // perfectly separable → 1.0
        assert!((auroc(&[0.1, 0.2, 0.8, 0.9], &[0, 0, 1, 1]) - 1.0).abs() < 1e-9);
        // inverted → 0.0
        assert!((auroc(&[0.9, 0.8, 0.2, 0.1], &[0, 0, 1, 1]) - 0.0).abs() < 1e-9);
        // single-class → 0.5 guard
        assert!((auroc(&[0.1, 0.4, 0.7], &[1, 1, 1]) - 0.5).abs() < 1e-9);
    }
}
