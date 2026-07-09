//! Minimal tabular dataset loading + standardisation for the KAN classifier/regressor.
//!
//! Parses a CSV whose **last column is the label** (string or number) and whose leading
//! columns are numeric features. Features are **min-max standardised per column into
//! `[-1, 1]`** — the Chebyshev spline's trusted input range (`ops::kan`). std-only; no
//! CSV crate.

/// A loaded, standardised tabular dataset.
#[derive(Debug, Clone)]
pub struct Tabular {
    /// Features, flat `(n, d)`, each column in `[-1, 1]`.
    pub x: Vec<f32>,
    /// Integer class labels `(n,)` (dense, `0..n_classes`).
    pub y: Vec<usize>,
    /// Rows.
    pub n: usize,
    /// Feature columns.
    pub d: usize,
    /// Distinct classes.
    pub n_classes: usize,
    /// Label strings in class-index order (for reporting).
    pub class_names: Vec<String>,
}

/// Load and standardise a label-last CSV.
///
/// # Preconditions
/// Every non-empty, non-`#` line has the same number of comma-separated fields (≥ 2);
/// all but the last parse as `f32`.
///
/// # Panics
/// Panics on a ragged row or a non-numeric feature.
pub fn load_csv(text: &str) -> Tabular {
    let rows: Vec<Vec<&str>> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(|l| l.split(',').map(str::trim).collect())
        .collect();
    assert!(!rows.is_empty(), "no data rows");
    let d = rows[0].len() - 1;
    assert!(d >= 1, "need ≥ 1 feature column + 1 label column");

    // Features + label strings.
    let mut x = Vec::with_capacity(rows.len() * d);
    let mut label_strs = Vec::with_capacity(rows.len());
    for r in &rows {
        assert_eq!(r.len(), d + 1, "ragged row: {r:?}");
        for f in &r[..d] {
            x.push(
                f.parse::<f32>()
                    .unwrap_or_else(|_| panic!("non-numeric feature {f:?}")),
            );
        }
        label_strs.push(r[d].to_string());
    }
    let n = rows.len();

    // Dense class indices in sorted-unique order (deterministic).
    let mut class_names: Vec<String> = label_strs.clone();
    class_names.sort();
    class_names.dedup();
    let y: Vec<usize> = label_strs
        .iter()
        .map(|s| class_names.iter().position(|c| c == s).unwrap())
        .collect();

    standardise_minmax(&mut x, n, d);
    Tabular {
        x,
        y,
        n,
        d,
        n_classes: class_names.len(),
        class_names,
    }
}

/// Per-column min-max standardisation into `[-1, 1]` (constant columns → 0).
fn standardise_minmax(x: &mut [f32], n: usize, d: usize) {
    for col in 0..d {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for row in 0..n {
            let v = x[row * d + col];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let span = hi - lo;
        for row in 0..n {
            let slot = &mut x[row * d + col];
            *slot = if span > 0.0 {
                2.0 * (*slot - lo) / span - 1.0
            } else {
                0.0
            };
        }
    }
}

impl Tabular {
    /// Deterministic LCG-shuffled train/test split; returns `(train_idx, test_idx)`.
    pub fn split(&self, test_frac: f32, seed: u64) -> (Vec<usize>, Vec<usize>) {
        let mut order: Vec<usize> = (0..self.n).collect();
        let mut st = seed
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        for i in (1..order.len()).rev() {
            st = st
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            order.swap(i, (st >> 33) as usize % (i + 1));
        }
        let cut = ((self.n as f32) * (1.0 - test_frac)) as usize;
        (order[..cut].to_vec(), order[cut..].to_vec())
    }

    /// Gather the rows in `idx` into `(x_sub (m, d), y_sub (m,))`.
    pub fn gather(&self, idx: &[usize]) -> (Vec<f32>, Vec<usize>) {
        let mut x = Vec::with_capacity(idx.len() * self.d);
        let mut y = Vec::with_capacity(idx.len());
        for &i in idx {
            x.extend_from_slice(&self.x[i * self.d..(i + 1) * self.d]);
            y.push(self.y[i]);
        }
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_and_standardises() {
        let csv = "1.0,2.0,a\n3.0,4.0,b\n5.0,6.0,a\n";
        let t = load_csv(csv);
        assert_eq!((t.n, t.d, t.n_classes), (3, 2, 2));
        assert_eq!(t.y, vec![0, 1, 0]); // a<b sorted
                                        // Column 0: {1,3,5} → min-max [-1,1] → {-1, 0, 1}.
        assert!((t.x[0] + 1.0).abs() < 1e-6 && t.x[2].abs() < 1e-6 && (t.x[4] - 1.0).abs() < 1e-6);
        assert!(t.x.iter().all(|&v| (-1.0..=1.0).contains(&v)));
    }

    #[test]
    fn split_is_disjoint_and_covers() {
        let csv = (0..20)
            .map(|i| format!("{i},x"))
            .collect::<Vec<_>>()
            .join("\n");
        let t = load_csv(&csv);
        let (tr, te) = t.split(0.25, 7);
        assert_eq!(tr.len() + te.len(), 20);
        let mut all: Vec<usize> = tr.iter().chain(&te).copied().collect();
        all.sort();
        all.dedup();
        assert_eq!(all.len(), 20, "split not a partition");
    }
}
