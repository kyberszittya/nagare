# Architecture

`nagare-holonomy-learn` is an experimental local-update learner for generated
point-set tasks.

Pipeline:

1. Generate a point set such as moons, spiral, or xor.
2. Lift each point through quaternion periodic features with `hymeko_clifford`
   rotor helpers:
   `[x, y, r, rotated_x, rotated_y, holonomy_w, holonomy_z]`.
3. Globally pool mean, spread, max, and sign entropy per lifted channel.
4. Build a local feature vector with one feedback channel.
5. Apply one of three gates:
   - scalar entropy feedback;
   - constant feedback;
   - fitted holonomy projection.
6. Train a two-logit local learner with simultaneous sample updates.
7. Report cross entropy, accuracy, Clifford probability error, timing, and
   parameter count.

The fitted projection gate estimates a rank-6 basis from class centroids and
rotor/holonomy channel groups, then applies:

```text
phi' = alpha * P(phi) + (1 - alpha) * phi
```

The current alpha is `0.72`.
