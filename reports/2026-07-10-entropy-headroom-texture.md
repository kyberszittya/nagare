# Nagare CV — entropy feedback on a headroom task (honest: redundant with |DFT| as a feature)

Date: 2026-07-10 · Author: Aiko (agent) for Hajdu Csaba

## Summary

The shape phase-pool saturated at 0.97, hiding whether phase **entropy** adds over `|DFT|`. Built a
**headroom task** — orientation-disorder textures where classes differ by *how many* orientations
are present (at random angles). It has genuine headroom (`|DFT|` = 0.795, not saturated), but
**entropy as a feature is neutral (Δ −0.005) — it does not add.** The mechanism is clear and it's an
honest negative for entropy-*as-a-classification-feature*, with a redirect to where "entropy
feedback" actually belongs.

## Task + result (4 seeds, orientation-disorder textures)

Class `c` = a texture from `N_ORI[c] ∈ {1,2,3,6}` gratings at **random** orientations (+noise).
Random angles ⇒ inherently rotation-invariant (raw histogram can't; shift-invariant `|DFT|` can).

| feature | median acc |
|---|---|
| raw histogram (covariant) | 0.670 |
| phase-pool `\|DFT\|` (invariant) | **0.795** |
| phase-pool `\|DFT\|` + entropy | 0.790 |

`|DFT|` leaves headroom (0.795 < ceiling), yet **entropy is neutral** (Δ −0.005; the slight dip is
added-dimension noise).

## Why entropy doesn't add (analyzed, not asserted)

Phase entropy `H(h) = −Σ p log p` and `|DFT(h)|` are **both shift-invariant summaries of the same
orientation histogram**, and entropy is essentially a *function of the `|DFT|` magnitudes*: a flat
histogram (high entropy) has small `|c_k|` for `k≥1`; a peaked one (low entropy) has large `|c_k|`
(Parseval ties `Σh²` to `Σ|c_k|²`). So a linear classifier already reads the orientation disorder
straight off `|DFT|`, and the explicit entropy scalar carries no new linearly-useful information.
I *expected* the nonlinearity of `−Σp log p` to help a linear model; the measurement says the
`|DFT|` magnitudes already span what's needed. Honest negative, mechanism understood.

## Where "entropy feedback" actually belongs

The redundancy is specific to **entropy-as-a-static-feature on a fixed descriptor**. The framework's
"entropy feedback" is **entropy-as-a-regulariser on a *learned* representation** — the entropy-gated
HSiKAN update and the `spectral_entropy` op are training-time feedback, not features. The phase-pool
has *no learned representation to regularise* (it's a linear classifier on fixed invariants), so it
is the wrong place to look for entropy's value. The genuine test is **spectral-entropy regularisation
of the quaternion *conv*'s learned feature map** (`vision_quat_conv`) — untested, and the honest next
step for the entropy thread.

## Files touched

| file | change |
|---|---|
| `tests/common/vision.rs` | `+B/NK`, `phase_histogram`, `PhaseFeature`, `phase_features`, `train_linear` (§6.1 — shared by both phase tests) |
| `tests/vision_phase_pool.rs` | refactored onto `common::vision` (removed local copies; same result) |
| `tests/vision_texture_entropy.rs` | **new** — orientation-disorder headroom task, entropy-vs-`\|DFT\|` |

## CORE / deps

**None.** No dependency change.

## Test results

- Full suite **106 / 0** (2 ignored: the heavy group-conv measurement); textures + phase-pool ~1.8 s
  each; clippy `-D warnings` + fmt clean. Mac-only.

## Open / next

- **The real entropy-feedback test:** add `spectral_entropy` regularisation to the quat-conv's
  learned feature map (a training-time feedback, not a feature) — where the framework's entropy
  mechanism can genuinely contribute.
- The phase-pool `|DFT|` remains the CV winner (0.97 on shapes, 0.80 on textures); entropy is a
  neutral add-on as a feature.
