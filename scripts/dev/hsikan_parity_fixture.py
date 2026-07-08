#!/usr/bin/env python
"""Generate the HSiKAN Rust<->PyTorch parity fixture (Phase 1b).

Runs the **real** ``hymeko_neuro`` ``SignedKANLayer`` (spline_kind="catmull_rom",
inner_skip="highway", outer_skip="none", S=2) on CPU in float64, and dumps a
std-parseable text fixture that ``tests/hsikan_torch_parity.rs`` checks the
closed-form Rust op (``src/ops/hsikan.rs``) against.

The Rust op is parametrised in **Chebyshev coefficients**; PyTorch's catmull_rom
activation in **CR control points**. Since
``chebyshev_cr(coef) == catmull_rom(chebyshev_control_points(coef))`` and both
CR evaluators are bit-identical (``core/splines.catmull_rom`` ==
``hyperedge/splines._catmull_rom_eval`` == ``src/ops/catmull_rom.rs``), we set the
layer's control points = ``cheb_coef @ basis.T`` where ``basis`` is the same
Chebyshev-at-uniform-knots matrix ``catmull_rom.rs::chebyshev_knot_basis`` builds.
The fixture stores the *Chebyshev* coefs (what the Rust op consumes) and the
float64 PyTorch output (the reference). Rust op is f32 → compare at tol 1e-3.

Run on kato15 (torch env + framework present):
  PYTHONPATH=~/hymeko_framework_rust ~/envs/hymeko/bin/python \
    scripts/dev/hsikan_parity_fixture.py --out tests/fixtures/hsikan_torch_parity.txt
"""
from __future__ import annotations

import argparse
import sys
import types

import numpy as np
import torch


def _stub_unused_dataset_deps() -> None:
    """Install an import hook stubbing absent dataset-synthesis-only packages.

    Importing `hymeko_neuro.hyperedge.signedkan` transitively runs
    `data.datasets.synth`, which imports sklearn/pandas/networkx for dataset
    *synthesis* — a code path the SignedKANLayer forward never touches. On a
    torch env lacking those, we return a package-shaped stub for any of their
    submodules (attributes resolve to no-ops), so the real layer loads. Only
    genuinely-absent packages are stubbed; a real install always wins.
    """
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
            module.__path__ = []  # mark as package so submodules resolve
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


def chebyshev_knot_basis(grid: int, k: int) -> np.ndarray:
    """(grid, k) Chebyshev T_0..T_{k-1} at uniform knots on [-1, 1].

    Mirrors ``src/ops/catmull_rom.rs::chebyshev_knot_basis`` exactly.
    """
    basis = np.zeros((grid, k), dtype=np.float64)
    for g in range(grid):
        x = -1.0 + 2.0 * g / (grid - 1)
        terms = np.zeros(k)
        terms[0] = 1.0
        if k > 1:
            terms[1] = x
        for j in range(2, k):
            terms[j] = 2.0 * x * terms[j - 1] - terms[j - 2]
        basis[g] = terms
    return basis


def control_points(cheb: np.ndarray, grid: int, k: int) -> np.ndarray:
    """(S, d, k) Chebyshev coefs -> (S, d, grid) CR control points."""
    return cheb @ chebyshev_knot_basis(grid, k).T


def fmt(a) -> str:
    return " ".join(repr(float(v)) for v in np.asarray(a).ravel())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=53)
    args = ap.parse_args()

    _stub_unused_dataset_deps()
    from hymeko_neuro.hyperedge.signedkan import SignedKANConfig, SignedKANLayer

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    d, grid, k, s_br, n_nodes = 3, 6, 4, 2, 6
    x = (0.3 * rng.standard_normal((n_nodes, d))).astype(np.float32)
    inner = (0.2 * rng.standard_normal((s_br, d, k))).astype(np.float32)
    outer = (0.2 * rng.standard_normal((s_br, d, k))).astype(np.float32)
    gate_w = (0.1 * rng.standard_normal((d, d))).astype(np.float32)
    gate_b = np.array([-2.0, -1.5, -2.5], dtype=np.float32)

    cfg = SignedKANConfig(
        n_nodes=n_nodes, hidden_dim=d, grid=grid, k=3,
        use_minus_branch=True, spline_kind="catmull_rom",
        inner_skip="highway", outer_skip="none",
    )
    layer = SignedKANLayer(cfg).double().eval()
    with torch.no_grad():
        layer.inner.coef.copy_(torch.tensor(control_points(inner, grid, k)))
        layer.outer.coef.copy_(torch.tensor(control_points(outer, grid, k)))
        layer.gate_inner.weight.copy_(torch.tensor(gate_w.astype(np.float64)))
        layer.gate_inner.bias.copy_(torch.tensor(gate_b.astype(np.float64)))
    x_t = torch.tensor(x.astype(np.float64))

    # Mixed-arity: same shared layer, two arities (k=3 and k=4).
    edge_sets = [
        (3, [[0, 1, 2], [2, 3, 4]], [[1, -1, 1], [-1, 1, -1]]),
        (4, [[0, 1, 2, 3], [1, 2, 4, 5]], [[1, -1, 1, -1], [1, 1, -1, -1]]),
    ]
    cases = []
    for arity, verts, signs in edge_sets:
        tv = torch.tensor(verts, dtype=torch.long)
        ts = torch.tensor(signs, dtype=torch.long)
        with torch.no_grad():
            h_e = layer(x_t, tv, ts).numpy()  # (T, d)
        cases.append((arity, np.array(verts), np.array(signs), h_e))

    with open(args.out, "w") as f:
        f.write("# HSiKAN Rust<->PyTorch parity fixture (Phase 1b)\n")
        f.write(f"# generator scripts/dev/hsikan_parity_fixture.py seed={args.seed}\n")
        f.write(f"# torch={torch.__version__}; float64 reference; Rust op is f32 (compare tol 1e-3)\n")
        f.write(f"d {d}\ngrid {grid}\ncheb_k {k}\nn_branches {s_br}\nn_nodes {n_nodes}\nn_cases {len(cases)}\n")
        f.write(f"x {fmt(x)}\n")
        f.write(f"inner_coef {fmt(inner)}\n")
        f.write(f"outer_coef {fmt(outer)}\n")
        f.write(f"gate_w {fmt(gate_w)}\n")
        f.write(f"gate_b {fmt(gate_b)}\n")
        for arity, verts, signs, h_e in cases:
            f.write(f"case_arity {arity}\n")
            f.write(f"case_n_edges {verts.shape[0]}\n")
            f.write(f"case_vertices {' '.join(str(int(v)) for v in verts.ravel())}\n")
            f.write(f"case_signs {' '.join(str(int(s)) for s in signs.ravel())}\n")
            f.write(f"case_h_e {fmt(h_e)}\n")
    print(f"wrote {args.out}: d={d} grid={grid} cheb_k={k} S={s_br} cases={len(cases)}")
    for arity, _, _, h_e in cases:
        print(f"  arity={arity}: h_e range [{h_e.min():.5f}, {h_e.max():.5f}]")


if __name__ == "__main__":
    main()
