# Projection Gate Holonomy Ablation

Date: 2026-07-02

## Scope

This run checks the claim that the entropy/holonomy gate should not be simple scalar multiplication. A new `projection` gate was added beside the previous `entropy` and `constant` gates in the Nagare toy learner.

The projection gate keeps the simultaneous global-pooling update path, but replaces the scalar entropy feedback channel with a holonomy-aligned projection of the pooled feature vector. The first implementation uses a fixed hand-built projector over the structural pooled channels derived from rotated and quaternion/holonomy components.

## Implementation

Changed files:

- `hymeko_nagare/examples/entropy_pool_learning_compare.rs`
- `hymeko_nagare/tests/entropy_pool_learning_compare.rs`
- `docs/plans/2026-07-02-projection-gate-holonomy-ablation/plan.tex`
- `docs/plans/2026-07-02-projection-gate-holonomy-ablation/plan.tikz`
- `docs/plans/2026-07-02-projection-gate-holonomy-ablation/plan.mmd`
- `docs/plans/2026-07-02-projection-gate-holonomy-ablation/plan.pdf`

Artifacts:

- `reports/2026-07-02-projection-gate-holonomy-ablation.json`
- `reports/2026-07-02-projection-gate-holonomy-ablation-rss.json`

## Main Nagare Result

Generated toy tasks: moons, spiral, xor.

Configuration:

- train samples: 192
- test samples: 96
- points per sample: 32
- hidden width for backprop-like baseline: 24
- epochs: 50
- local learner parameters: 60
- backprop-like parameters: 2836

| Task | Local acc | Local loss | Local median us/sample | Backprop-like acc | Backprop-like loss | Backprop median us/sample | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| moons | 1.000 | 0.001513 | 5.599 | 1.000 | 0.000129 | 135.699 | 24.2x |
| spiral | 1.000 | 0.004486 | 5.929 | 1.000 | 0.000001 | 130.492 | 22.0x |
| xor | 1.000 | 0.006714 | 6.139 | 1.000 | 0.000094 | 133.003 | 21.7x |

Peak RSS from the sampled run was 8,122,368 bytes, about 7.75 MiB.

## Gate Ablation

The stress ablation compares three gates:

- `entropy`: scalar entropy feedback, `0.25 + entropy`.
- `constant`: no entropy gate, constant feedback.
- `projection`: fixed holonomy-axis projection over pooled structural channels.

| Task | Stress | Entropy acc/loss | Constant acc/loss | Projection acc/loss | Projection readout |
|---|---|---:|---:|---:|---|
| moons | clean | 1.000 / 0.001513 | 1.000 / 0.000465 | 1.000 / 0.001327 | better than entropy, worse than constant |
| moons | noisy | 1.000 / 0.001631 | 1.000 / 0.000500 | 1.000 / 0.001253 | better than entropy, worse than constant |
| moons | missing | 1.000 / 0.001656 | 1.000 / 0.000516 | 1.000 / 0.001421 | better than entropy, worse than constant |
| moons | fewshot noisy missing | 1.000 / 0.011536 | 1.000 / 0.005352 | 1.000 / 0.020177 | worse |
| spiral | clean | 1.000 / 0.004487 | 1.000 / 0.001581 | 1.000 / 0.001676 | close to constant, better than entropy |
| spiral | noisy | 0.990 / 0.043077 | 0.990 / 0.040756 | 0.990 / 0.051638 | worse |
| spiral | missing | 1.000 / 0.015772 | 1.000 / 0.011051 | 1.000 / 0.012821 | better than entropy, worse than constant |
| spiral | fewshot noisy missing | 0.927 / 0.286730 | 0.938 / 0.269849 | 0.906 / 0.324541 | worse |
| xor | clean | 1.000 / 0.006731 | 1.000 / 0.002713 | 1.000 / 0.003012 | close to constant, better than entropy |
| xor | noisy | 1.000 / 0.006969 | 1.000 / 0.002783 | 1.000 / 0.003129 | close to constant, better than entropy |
| xor | missing | 1.000 / 0.008426 | 1.000 / 0.003485 | 1.000 / 0.003558 | close to constant, better than entropy |
| xor | fewshot noisy missing | 1.000 / 0.033405 | 1.000 / 0.022253 | 1.000 / 0.023323 | close to constant, better than entropy |

## Interpretation

The conceptual correction is right: in a holonomy/quaternion setting, the gate should be a projection-like operation, not just scalar multiplication.

The current fixed projector is not yet the right projector. It improves over scalar entropy in many easy rows and often tracks the constant gate closely, but it does not beat the constant gate overall. On the hardest spiral row it is clearly worse: 0.906 accuracy versus 0.927 for entropy and 0.938 for constant.

So the readout is:

1. Local holonomy/global-pooling learning is still fast: about 5.6 to 6.1 us/sample on the main run.
2. The local learner remains much smaller than the backprop-like baseline: 60 parameters versus 2836.
3. The 21.7x to 24.2x speedup against the toy backprop-like baseline is reproducible for these settings.
4. Scalar entropy feedback is not validated as the best gate.
5. Projection is the right algebraic direction, but the current hand-coded projection basis is too narrow.

## Next Step

Replace the fixed projection mask with a richer native projector:

1. Orthogonal low-rank projector: learn or estimate a small projector basis from pooled Clifford/quaternion features, then apply `P phi` instead of `g * phi`.
2. Rotor-bundle projector: build the projector from quaternion sandwich features so periodic components affect the subspace, not just the final scalar gate.
3. Class-conditional projector: keep a tiny projector per output class and select or mix through entropy/alpha feedback.
4. Sparse projector: store only active structural channels so the projection remains microsecond-scale.
5. Backward/parity gate: add finite-difference checks before treating the richer projector as a real Nagare op.

## Verification

Commands run:

```powershell
rustfmt --check hymeko_nagare/examples/entropy_pool_learning_compare.rs hymeko_nagare/tests/entropy_pool_learning_compare.rs
cargo test -p hymeko_nagare --test entropy_pool_learning_compare -- --nocapture
cargo clippy -p hymeko_nagare --example entropy_pool_learning_compare --test entropy_pool_learning_compare --no-deps -- -D warnings
cargo run -p hymeko_nagare --release --example entropy_pool_learning_compare -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --hidden 24 --epochs 50 --batch-size 32 --lr 0.05 --seed 53 --out reports\2026-07-02-projection-gate-holonomy-ablation.json
```

All verification commands passed.
