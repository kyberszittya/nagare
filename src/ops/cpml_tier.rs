//! CPML tier stratification — the Gömb **inner core**'s defining primitive.
//!
//! Port of `hymeko_neuro/hyperedge/cpml.py` `{TierSpec.assign, restrict_cycles_to_tier}` (the
//! Concentric-Pyramid Multi-Layer). Vertices are stratified by **degree percentile** into
//! concentric tiers (periphery → centre; the V1→V2→V4→IT cortical metaphor of Felleman & Van
//! Essen 1991), and each tier processes only the cycles that **touch** one of its vertices.
//! The per-tier aggregation + `concat(X₀, H₀…H_{L-1})` readout is composed from existing ops
//! (`linear`, `scatter_mean`) — this module supplies only the routing, which is the piece
//! Nagare did not already have.
//!
//! The stratification is a **fixed structural routing** derived from graph degrees (like the
//! cycle enumeration), not a learnable/differentiable map — so there is no backward here;
//! `tier_of` is a fixed input to the differentiable part, exactly as `cycles`/`signs` are.
//!
//! Note on the acronym: the `architecture/cognitive_stack` README frames the inner core as
//! "grade-preserving polynomial layers, grade-0 readout ⟨·⟩₀". The **implemented**
//! `InnerCPMLCore` (`models/hymeko_gomb/shells.py`) instead wraps this tier-stratified CPML;
//! the code is authoritative and is what this port mirrors.

/// Tier boundaries on degree-percentile space: `cuts[0]=0 … cuts[L]=1`, strictly increasing.
/// Tier 0 is the outermost (lowest degree), tier `L-1` the innermost (highest degree).
#[derive(Debug, Clone)]
pub struct TierSpec {
    cuts: Vec<f32>,
}

impl TierSpec {
    /// Construct from explicit percentile cuts.
    ///
    /// # Preconditions
    /// `cuts.len() >= 2`, `cuts[0] == 0.0`, `cuts[last] == 1.0`, strictly increasing.
    ///
    /// # Panics
    /// If the cuts are not a valid strictly-increasing `[0, 1]` partition.
    pub fn new(cuts: Vec<f32>) -> Self {
        assert!(cuts.len() >= 2, "need at least one tier");
        assert!(
            (cuts[0] - 0.0).abs() < 1e-6 && (cuts[cuts.len() - 1] - 1.0).abs() < 1e-6,
            "cuts must span [0, 1]"
        );
        assert!(
            cuts.windows(2).all(|w| w[1] > w[0]),
            "cuts must be strictly increasing"
        );
        Self { cuts }
    }

    /// Uniform `n_tiers` split of `[0, 1]` (the `InnerCPMLCore` default = `linspace`).
    pub fn uniform(n_tiers: usize) -> Self {
        assert!(n_tiers >= 1);
        let cuts = (0..=n_tiers).map(|i| i as f32 / n_tiers as f32).collect();
        Self { cuts }
    }

    /// Number of tiers `L`.
    pub fn n_tiers(&self) -> usize {
        self.cuts.len() - 1
    }

    /// Map each vertex to its tier index in `{0, …, L-1}` by degree percentile.
    ///
    /// Percentile is `rank / (N-1)` (ascending, ties broken by stable sort). Tier 0 uses a
    /// closed-left interval `[cuts[0], cuts[1]]`; every later tier is half-open `(cuts[ℓ],
    /// cuts[ℓ+1]]` — matching the reference `TierSpec.assign`.
    ///
    /// # Postconditions
    /// `out.len() == degrees.len()`; every entry is in `0..L`; the map is monotone
    /// non-decreasing in degree.
    pub fn assign(&self, degrees: &[f32]) -> Vec<usize> {
        let n = degrees.len();
        if n == 0 {
            return Vec::new();
        }
        // Stable ascending argsort → percentile rank per vertex.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| degrees[a].total_cmp(&degrees[b]));
        let mut ranks = vec![0.0f32; n];
        for (pos, &i) in order.iter().enumerate() {
            ranks[i] = if n > 1 {
                pos as f32 / (n - 1) as f32
            } else {
                0.0
            };
        }
        let l = self.n_tiers();
        let mut tiers = vec![0usize; n];
        for (i, &r) in ranks.iter().enumerate() {
            for ell in 0..l {
                let (lo, hi) = (self.cuts[ell], self.cuts[ell + 1]);
                let inside = if ell == 0 {
                    r >= lo && r <= hi
                } else {
                    r > lo && r <= hi
                };
                if inside {
                    tiers[i] = ell;
                }
            }
        }
        tiers
    }
}

/// Per-vertex incidence degree = number of cycle-corners landing on each vertex.
///
/// The unsigned corner count over the cycle pool (mirrors the reference's degree pass); a
/// natural degree proxy for tier stratification when only the cycle pool is on hand.
///
/// # Preconditions
/// `cycles` is flat `(M·k)` with every entry `< n_nodes`.
pub fn cycle_incidence_degrees(cycles: &[u32], n_nodes: usize) -> Vec<f32> {
    let mut deg = vec![0.0f32; n_nodes];
    for &v in cycles {
        deg[v as usize] += 1.0;
    }
    deg
}

/// Indices of cycles that **touch** at least one tier-`ell` vertex (hard incidence routing).
///
/// # Preconditions
/// `cycles` is flat `(M·k)`; `tier_of[v]` is defined for every vertex id in `cycles`.
pub fn tier_cycle_indices(cycles: &[u32], k: usize, tier_of: &[usize], ell: usize) -> Vec<usize> {
    assert!(k >= 1 && cycles.len().is_multiple_of(k));
    (0..cycles.len() / k)
        .filter(|&c| (0..k).any(|i| tier_of[cycles[c * k + i] as usize] == ell))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_tiers_partition_and_size() {
        let ts = TierSpec::uniform(3);
        assert_eq!(ts.n_tiers(), 3);
        // 10 distinct degrees → percentiles 0, 1/9, …, 1. Cuts at 1/3, 2/3.
        let degrees: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let tiers = ts.assign(&degrees);
        // Every tier index valid; monotone non-decreasing in degree (already sorted input).
        assert!(tiers.iter().all(|&t| t < 3));
        assert!(
            tiers.windows(2).all(|w| w[1] >= w[0]),
            "tiers not monotone in degree"
        );
        // Lowest degree → tier 0, highest → tier 2.
        assert_eq!(tiers[0], 0);
        assert_eq!(tiers[9], 2);
    }

    #[test]
    fn single_tier_is_flat() {
        let ts = TierSpec::uniform(1);
        let tiers = ts.assign(&[3.0, 1.0, 9.0, 2.0]);
        assert!(
            tiers.iter().all(|&t| t == 0),
            "L=1 must place everyone in tier 0"
        );
    }

    #[test]
    fn assign_is_monotone_under_permutation() {
        let ts = TierSpec::new(vec![0.0, 0.2, 0.8, 1.0]);
        let degrees = [5.0f32, 1.0, 8.0, 2.0, 9.0, 3.0, 7.0, 0.5, 6.0, 4.0];
        let tiers = ts.assign(&degrees);
        // Higher degree ⇒ tier ≥. Check every ordered pair.
        for i in 0..degrees.len() {
            for j in 0..degrees.len() {
                if degrees[i] < degrees[j] {
                    assert!(tiers[i] <= tiers[j], "degree order violated at ({i},{j})");
                }
            }
        }
    }

    #[test]
    fn tier_routing_covers_touching_cycles() {
        // 3 triangles over 5 vertices; tiers assigned by hand.
        let cycles = [
            0u32, 1, 2, /* c0 */ 2, 3, 4, /* c1 */ 0, 3, 4, /* c2 */
        ];
        let tier_of = [0usize, 0, 1, 2, 2]; // v0,v1→t0  v2→t1  v3,v4→t2
                                            // tier 0: cycles touching v0 or v1 → c0, c2.
        assert_eq!(tier_cycle_indices(&cycles, 3, &tier_of, 0), vec![0, 2]);
        // tier 1: cycles touching v2 → c0, c1.
        assert_eq!(tier_cycle_indices(&cycles, 3, &tier_of, 1), vec![0, 1]);
        // tier 2: cycles touching v3 or v4 → c1, c2.
        assert_eq!(tier_cycle_indices(&cycles, 3, &tier_of, 2), vec![1, 2]);
        // A cycle spanning multiple tiers appears in each — overlapping, not a partition.
    }

    #[test]
    fn incidence_degrees_count_corners() {
        let cycles = [0u32, 1, 2, 2, 3, 4, 0, 3, 4];
        let deg = cycle_incidence_degrees(&cycles, 5);
        assert_eq!(deg, vec![2.0, 1.0, 2.0, 2.0, 2.0]); // v0:c0,c2 v1:c0 v2:c0,c1 v3:c1,c2 v4:c1,c2
    }
}
