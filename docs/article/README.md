# Nagare — article figures (algorithms, flowcharts, system architecture)

Diagram assets for the Nagare framework paper. Every figure has an editable source; the two architecture
figures also have LaTeX/TikZ.

## Contents

| Fig | subject | Mermaid source | rendered SVG | TikZ (LaTeX) |
|---|---|---|---|---|
| 0 | **The whole Nagare pipeline (end-to-end)** | `nagare-diagrams.md` §0 | `svg/fig0-pipeline.svg` | — |
| 1 | System architecture (layered) | `nagare-diagrams.md` §1 | `svg/fig1-architecture.svg` | `tikz/fig1-architecture.tex` |
| 2 | Closed-form op contract (no autograd) | §2 | `svg/fig2-op-contract.svg` | — |
| 3 | Neocognitron pipeline (S/C + entropy top) | §3 | `svg/fig3-neocognitron.svg` | `tikz/fig3-neocognitron.tex` |
| 4 | Entropy global pool (invariant + equivariant) | §4 | `svg/fig4-entropy-pool.svg` | — |
| 5 | Evolvent update (forgetting-RLS loop) | §5 | `svg/fig5-evolvent.svg` | — |
| 6 | Signed-link balance & leakage audit | §6 | `svg/fig6-signed-link.svg` | — |
| 7 | SBSH detector | §7 | (renders from source) | — |
| 8 | Assimilation lifecycle | §8 | `svg/fig8-assimilation.svg` | — |
| 9 | HSiKAN structural-leverage (summary) | §9 | `svg/fig9-hsikan.svg` | — |
| 9a | HSiKAN experimental design (falsification protocol) | §9a | `svg/fig9a-hsikan-design.svg` | — |
| 9b | HSiKAN double-dissociation (per-node arch vs structure) | §9b | `svg/fig9b-hsikan-dissociation.svg` | — |
| 9c | HSiKAN H1 scaling — **measured plot** (5 seeds, data `data/hsikan_ladder.json`) | §9c | `svg/fig9c-hsikan-scaling.png` | — |
| 10 | CV rotation-invariant texture descriptor (phase_pool / dihedral) | §10 | `svg/fig10-cv-rotinvariant.svg` | — |
| 11 | Rotor & holonomy geometry primitives | §11 | `svg/fig11-rotor-holonomy.svg` | — |
| 12 | KAN / HSiKAN learnable spline op (Chebyshev-CR) | §12 | `svg/fig12-kan-spline.svg` | — |
| 13 | Scatter-locality (systems / performance) | §13 | `svg/fig13-scatter-locality.svg` | — |

## How to use

- **Mermaid** (`nagare-diagrams.md`): renders on GitHub/GitLab and in Mermaid-aware editors; all 8 validated.
- **SVG** (`svg/`): vector, drop straight into a paper or slide.
- **TikZ** (`tikz/`): standalone-compilable (`pdflatex fig1-architecture.tex`); native LaTeX for the article.
  (Not compiled in this repo — no local `pdflatex`; compile in the paper's toolchain.)

## Provenance

The diagrams reflect the canonical framework state: `reports/framework/nagare_results_collection.md` (result
lines), `canonical_components.json` (components), `canonical_findings.json` (findings, incl. F-ARC-1 and F-EVO-1/2/3).
Regenerate SVGs by re-rendering the Mermaid sources.
