# HSiKAN → Nagare, Phase 1c — entropy-gated local learning on mixed-arity features

Date: 2026-07-08 · Author: Aiko (agent) for Hajdu Csaba
Plan: `docs/plans/2026-07-08-hsikan-nagare-phase1/` · Follows 1a/1b

## Summary

The learning-rule half of the entropy integration (user directive: HSiKAN/Gömb
should use entropy feedback). HSiKAN (`src/ops/hsikan.rs`) is used as a **fixed
mixed-arity feature extractor** — both arity-3 and arity-4 hyperedges share ONE
parameter set (the mixed-arity claim) — and a linear readout is trained on the
per-edge embeddings by **Nagare's entropy-gated local delta rule** (the `learner.rs`
substrate):

```
Δw = lr · gate · φ · (y − p),   gate = 0.25 + H(softmax(logits))  [Entropy]  or 1.0  [Constant]
```

This is the *Nagare-native* learner (rich fixed features + cheap entropy-gated
linear readout), not generic SGD. Entropy-vs-constant is **measured**, not assumed.

## Measurement (single seed — suggestive, not a verdict)

Mixed-arity toy: 10 nodes, 6 arity-3 + 6 arity-4 edges; balanced linearly-separable
labels from a random linear teacher in the fixed feature space; 300 epochs, lr 0.1.

| gate | BCE (init → final) | train acc |
|---|---|---|
| **entropy** | 0.6933 → **0.2857** | 1.000 |
| constant | 0.6933 → 0.3104 | 1.000 |

Both reach 100% train accuracy (separable), but the **entropy gate reaches lower
loss** — the *opposite* of the standing arity-2 result (where constant won all 12
stress rows). **This is one seed on one teacher-separable toy: suggestive, not a
verdict.** Per §3, any ranking claim needs multi-seed median/IQR — flagged as the
real experiment for the benchmark stage. What 1c *asserts* (the pass condition) is
only that the local update **drives learning** in both modes (loss halves, acc ≥ 0.9);
which gate wins is reported, not asserted.

## Files touched

| file | change | lines |
|---|---|---|
| `tests/hsikan_layer.rs` | **new** — mixed-arity feature extractor + entropy-gated readout (Entropy vs Constant) | 232 |

Reuses `entropy2` / `softmax2` / `cross_entropy` from the crate (§6.1) — the entropy
math is not re-implemented; only the test orchestration is local.

## CORE.YAML / deps

**None.** No dependency added.

## Test results (both machines)

| gate | Mac arm64 | kato15 x86_64 |
|---|---|---|
| `entropy_gated_local_learning_on_mixed_arity` | ✅ | ✅ |
| `shared_params_features_finite_both_arities` | ✅ | ✅ |
| clippy `-D warnings` / fmt | ✅ / ✅ | ✅ / ✅ |
| full suite | **51 / 0** | **51 / 0** |

The entropy-vs-constant values were **bit-identical on both boxes** (0.2857 / 0.3104),
so the single-seed signal is at least platform-stable — but it is still one seed
(see "Open" for the multi-seed follow-up).

## The two-mechanism split (user chose "both")

1. **Entropy-gated local update** — *this report (1c)*. Done.
2. **HSiKAN spectral-entropy regulariser** (`hyperedge/entropy_reg.py`) — **Phase 1c′,
   next.** Faithful closed-form port needs a symmetric **Jacobi eigensolver** +
   hand-derived **spectral-entropy backward** (grad through `eigvals(AᵀA)` →
   `dλ_i/dA = 2·A·uᵢuᵢᵀ`), FD-tested — its own §2 plan + §3 test. Not bolted onto 1c
   because it is a genuine new op, not a loss term.

## Open / follow-up

1. **1c′** — spectral-entropy regulariser as a closed-form Nagare op (eigensolver +
   spectral gradient), then wired into HSiKAN training as the loss term.
2. **Multi-seed entropy-vs-constant** on mixed-arity HSiKAN features — turn the
   suggestive single-seed signal into a median/IQR claim (the actual science; mirrors
   `run_stress_ablation`).
3. **1d** — forward/train latency + peak RSS + the `chunk_t` streaming cap.

## Provenance

- Repo `github.com/kyberszittya/nagare`, base `a8ca716` (working tree dirty).
- Deterministic: fixed seeds (extractor 7, teacher 11, readout 99). Rust 1.96.1 both boxes.
- Not committed yet — awaiting user's go + the kato15→GitHub deploy key.
