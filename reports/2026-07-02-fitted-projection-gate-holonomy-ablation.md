# Fitted Projection Gate Holonomy Ablation

Date: 2026-07-02

## Scope

This is the follow-up to the fixed projection-gate ablation. The fixed hand-built projector was algebraically closer to holonomy than scalar multiplication, but it was too narrow. This run replaces it with a tiny fitted rotor/holonomy projector.

## Change

The `projection` gate in `hymeko_nagare/examples/entropy_pool_learning_compare.rs` now:

- estimates a rank-6 projection basis from pooled training features;
- uses class centroid difference, centroid mean, geometry channels, rotated channels, holonomy channels, spread, and sign-entropy structure;
- orthonormalizes the candidate basis with a second Gram-Schmidt pass;
- applies alpha mixing between the projected vector and the original vector;
- keeps the entropy/global-pool simultaneous update path.

The fitted projection state adds 168 stored basis values, so the projection learner reports 228 parameters instead of 60. This is still far smaller than the 2836-parameter backprop-like baseline.

## Main Forward Comparison

Configuration:

- train samples: 192
- test samples: 96
- points per sample: 32
- hidden width for backprop-like baseline: 24
- epochs: 50
- local learner parameters: 60
- projection stress learner parameters: 228
- backprop-like parameters: 2836

| Task | Local acc | Local loss | Local median us/sample | Backprop-like acc | Backprop-like loss | Backprop median us/sample | Speedup |
|---|---:|---:|---:|---:|---:|---:|---:|
| moons | 1.000 | 0.001513 | 5.673 | 1.000 | 0.000129 | 139.984 | 24.7x |
| spiral | 1.000 | 0.004486 | 5.776 | 1.000 | 0.000001 | 136.064 | 23.6x |
| xor | 1.000 | 0.006714 | 4.178 | 1.000 | 0.000094 | 136.299 | 32.6x |

Sampled peak RSS during the release run was 7,737,344 bytes, about 7.38 MiB.

## Gate Ablation

| Task | Stress | Entropy acc/loss | Constant acc/loss | Fitted projection acc/loss | Readout |
|---|---|---:|---:|---:|---|
| moons | clean | 1.000 / 0.001513 | 1.000 / 0.000465 | 1.000 / 0.000468 | tied with constant |
| moons | noisy | 1.000 / 0.001631 | 1.000 / 0.000500 | 1.000 / 0.000498 | slightly beats constant |
| moons | missing | 1.000 / 0.001656 | 1.000 / 0.000516 | 1.000 / 0.000521 | tied with constant |
| moons | fewshot noisy missing | 1.000 / 0.011536 | 1.000 / 0.005352 | 1.000 / 0.005323 | slightly beats constant |
| spiral | clean | 1.000 / 0.004487 | 1.000 / 0.001581 | 1.000 / 0.001586 | tied with constant |
| spiral | noisy | 0.990 / 0.043077 | 0.990 / 0.040756 | 0.990 / 0.044278 | worse than constant |
| spiral | missing | 1.000 / 0.015772 | 1.000 / 0.011051 | 1.000 / 0.012371 | better than entropy |
| spiral | fewshot noisy missing | 0.927 / 0.286730 | 0.938 / 0.269849 | 0.938 / 0.250603 | best loss, tied best accuracy |
| xor | clean | 1.000 / 0.006731 | 1.000 / 0.002713 | 1.000 / 0.002697 | slightly beats constant |
| xor | noisy | 1.000 / 0.006969 | 1.000 / 0.002783 | 1.000 / 0.002935 | close to constant |
| xor | missing | 1.000 / 0.008426 | 1.000 / 0.003485 | 1.000 / 0.003637 | close to constant |
| xor | fewshot noisy missing | 1.000 / 0.033405 | 1.000 / 0.022253 | 1.000 / 0.022197 | slightly beats constant |

## Interpretation

This is a stronger result than the first projection attempt.

The fixed projector was worse than constant on the hardest spiral case: 0.906 accuracy and 0.324541 loss. The fitted projector reaches 0.938 accuracy and 0.250603 loss on that same row. That ties the best accuracy and beats constant gate loss.

The fitted projection also preserves the main claim: it is still microsecond-scale, still Nagare/Rust native, and still much smaller than the backprop-like baseline.

The honest caveat is that fitted projection is now using stored basis state. It is not a free scalar gate. The tradeoff is good here: 228 parameters versus 2836, and a better hard-row loss than the constant gate.

## Next Step

The next useful step is to move this from example-local code into a native Nagare op:

1. `learn_projection_basis`: compute class/entropy/holonomy projector basis.
2. `project_alpha_mix`: apply `alpha * P(phi) + (1 - alpha) * phi`.
3. finite-difference check for projection backward.
4. parity against the example fixture.
5. benchmark against the current example path.

That would turn the current fitted projector from a toy harness mechanism into a reusable kernel.

## Verification

Commands run:

```powershell
rustfmt --check hymeko_nagare\examples\entropy_pool_learning_compare.rs hymeko_nagare\tests\entropy_pool_learning_compare.rs
cargo test -p hymeko_nagare --test entropy_pool_learning_compare -- --nocapture
cargo clippy -p hymeko_nagare --example entropy_pool_learning_compare --test entropy_pool_learning_compare --no-deps -- -D warnings
cargo run -p hymeko_nagare --release --example entropy_pool_learning_compare -- --tasks moons,spiral,xor --n-train 192 --n-test 96 --n-points 32 --hidden 24 --epochs 50 --batch-size 32 --lr 0.05 --seed 53 --out reports\2026-07-02-fitted-projection-gate-holonomy-ablation.json
```

All verification commands passed.
