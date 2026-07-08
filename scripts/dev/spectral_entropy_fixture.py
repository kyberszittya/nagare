#!/usr/bin/env python
"""Generate the spectral-entropy regulariser parity fixture (Phase 1c′).

Drives the **real** `hymeko_neuro.hyperedge.entropy_reg.EntropyRegulariser` (float64)
on a fixed matrix A, takes ONE call (KL=0 ⇒ lam_eff = lam_0), and dumps A, reg, and
the autograd gradient ∇_A reg. `tests/spectral_entropy_parity.rs` checks the closed-form
Rust op (`spectral_reg_value_grad`, lam_eff = lam_0) against it. Both eigensolvers
(torch eigvalsh vs Rust Jacobi) yield the same eigenvalues, and ∇_A = 2·A·U diag(w)Uᵀ
is eigenvector-basis-invariant, so parity is meaningful across the two solvers.

Run on kato15:
  PYTHONPATH=~/hymeko_framework_rust ~/envs/hymeko/bin/python \
    scripts/dev/spectral_entropy_fixture.py --out tests/fixtures/spectral_entropy.txt
"""
from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import torch


def _stub_unused_dataset_deps() -> None:
    """Stub absent dataset-synthesis-only deps so the hyperedge package imports
    on a torch env without sklearn/pandas/networkx (the regulariser never uses
    them). See scripts/dev/hsikan_parity_fixture.py for the rationale."""
    import importlib.abc
    import importlib.machinery

    absent = []
    for pkg in ("sklearn", "pandas", "networkx"):
        try:
            __import__(pkg)
        except ModuleNotFoundError:
            absent.append(pkg)
    if not absent:
        return
    absent = tuple(absent)

    class _StubLoader(importlib.abc.Loader):
        def create_module(self, spec):
            module = types.ModuleType(spec.name)
            module.__path__ = []
            module.__getattr__ = lambda _name: (lambda *a, **k: None)
            return module

        def exec_module(self, module):
            pass

    class _StubFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".")[0] in absent:
                return importlib.machinery.ModuleSpec(fullname, _StubLoader())
            return None

    sys.meta_path.insert(0, _StubFinder())


def fmt(a) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(a).ravel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()

    _stub_unused_dataset_deps()
    from hymeko_neuro.hyperedge.entropy_reg import EntropyRegConfig, EntropyRegulariser

    rng = np.random.default_rng(args.seed)
    n, d = 5, 3  # n >= d → AᵀA path (matches _spectral_distribution)
    a_np = (0.6 * rng.standard_normal((n, d))).astype(np.float32)

    cfg = EntropyRegConfig(
        lam_0=1.0, lam_a=1.0, lam_b=1.0, lam_KL=0.5,
        eta=5.0, target_entropy=0.5, kl_normalized=False, momentum=0.0,
    )
    reg_obj = EntropyRegulariser(cfg)
    a = torch.tensor(a_np.astype(np.float64), requires_grad=True)
    reg = reg_obj(a)          # first call: KL=0 → lam_eff = lam_0
    reg.backward()
    grad = a.grad.detach().numpy()

    with open(args.out, "w") as f:
        f.write("# spectral-entropy regulariser parity fixture (Phase 1c′)\n")
        f.write(f"# generator scripts/dev/spectral_entropy_fixture.py seed={args.seed}\n")
        f.write(f"# torch={torch.__version__}; float64 reference; first call lam_eff=lam_0\n")
        f.write(f"n {n}\nd {d}\n")
        f.write(f"lam_0 {cfg.lam_0}\nlam_a {cfg.lam_a}\nlam_b {cfg.lam_b}\nlam_kl {cfg.lam_KL}\n")
        f.write(f"eta {cfg.eta}\ntarget {cfg.target_entropy}\n")
        f.write(f"reg {repr(float(reg.item()))}\n")
        f.write(f"a {fmt(a_np)}\n")
        f.write(f"grad {fmt(grad)}\n")
    print(f"wrote {args.out}: n={n} d={d} reg={float(reg.item()):.6f} "
          f"H_norm={reg_obj.last_h_norm:.5f} lam_eff={reg_obj.last_lam_eff:.5f}")


if __name__ == "__main__":
    main()
