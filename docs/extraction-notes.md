# Extraction Notes

This repository was initialized from the HyMeKo workspace experiments on
2026-07-02. It intentionally avoids editing or depending on protected HyMeKo
core crates.

Included:

- quaternion feature lift using `hymeko_clifford` rotor helpers;
- Clifford probability error using `hymeko_clifford`;
- fitted projection gate;
- toy datasets and corruption modes;
- local learner and stress ablation;
- copied reports and raw JSON artifacts from the original run.

Not included yet:

- native Nagare op registration;
- projection backward kernel;
- finite-difference backward check;
- PyTorch CPU/GPU harness;
- graph or pgraph feature enrichment.

Next implementation step:

1. Add `project_alpha_mix_forward` and `project_alpha_mix_backward` as explicit
   kernel functions.
2. Add finite-difference checks for projection gradients.
3. Add a text fixture parser for parity against `fixtures/moons_spiral_xor_seed53.txt`.
