# Nagare — algorithms, flowcharts, and system architecture (article figures)

Source diagrams for the Nagare framework (`holonomy_learn`). Each is provided as Mermaid (renders in most
venues); the two architecture figures also have TikZ under `docs/article/tikz/`. Nagare is closed-form and
FD-verified with **no autograd**; that discipline is the substrate every figure sits on.

---

## Fig. 0 — The whole Nagare pipeline (end-to-end)

The unified flow: any input domain enters the closed-form op library, ops compose into a model, the model is
learned by one of **two regimes** (backprop *or* evolvent), evaluation feeds outputs, and the assimilation loop
integrates each result back into the framework.

```mermaid
flowchart TB
  subgraph IN["Inputs (domain-general substrate)"]
    direction LR
    D1["images"]
    D2["signed graphs"]
    D3["data streams"]
    D4["point sets / scenes"]
  end
  IN --> OPS["Closed-form op library<br/>forward + FD-verified backward · NO autograd<br/>KAN/HSiKAN · rotors · hg_message · conv2d · entropy pool · ..."]
  OPS --> MODEL["Model = composed ops<br/>Neocognitron stack · CPML core · SBSH detector · RFF + head"]
  MODEL --> L{"learning regime"}
  L -->|backprop| BP["forward -> closed-form backward SWEEP -> Adam<br/>iterative · O(d)/step · sample-slow"]
  L -->|evolvent| EV["forward -> one-shot RLS update (NO backward sweep)<br/>O(d^2)/step · sample-fast (F-EVO-2)"]
  BP --> EVAL["evaluation<br/>prequential / held-out · multi-seed median"]
  EV --> EVAL
  EVAL --> APP["outputs: recognition · pose · detection · link prediction · online tracking"]
  EVAL --> ASSIM["assimilation lifecycle<br/>findings ledger + component registry"]
  ASSIM -. "integrate + guard + register" .-> OPS
  ASSIM -. "next experiment" .-> MODEL
```

---

## Fig. 1 — System architecture (layered)

```mermaid
flowchart TB
  subgraph APP["Applications"]
    direction LR
    NC["Neocognitron<br/>recognition + pose"]
    DET["SBSH detector<br/>oriented-box, dynamic quadtree grid"]
    SL["Signed-link<br/>balance + leakage audit (relational)"]
    EV["Evolvent<br/>online incremental learning"]
  end
  subgraph LRN["Learners"]
    direction LR
    BP["Backprop<br/>closed-form gradient + Adam<br/>O(d)/step, iterative (sample-slow)"]
    RLS["EvolventHead<br/>forgetting-RLS<br/>O(d^2)/step, one-pass (sample-fast)"]
  end
  subgraph OPS["Closed-form op library — 31 FD-verified backwards"]
    direction LR
    CORE["Core<br/>linear · mse · loss · adam"]
    FA["Function approx<br/>kan · hsikan · chebyshev-CR · kochanek-bartels"]
    GEO["Geometry / rotors<br/>dihedral · cayley_rotor · rotor_holonomy · rotor_spike · clifford_fir"]
    GR["Graph / hypergraph<br/>hg_message · signed_scatter · gomb_shell · cpml_tier"]
    ENT["Entropy / pooling<br/>global_entropy_pool · spectral_entropy · phase_pool"]
    CV["CV<br/>conv2d · group_pool · sc_block · soft_argmax · gaussian_kld · quadtree"]
  end
  DISC["Discipline: closed-form · hand-derived backward · FD-verified · NO autograd<br/>+ Assimilation lifecycle (component registry + findings ledger)"]
  APP --> LRN
  LRN --> OPS
  OPS --> DISC
```

---

## Fig. 2 — The closed-form op contract (per-op, no autograd)

```mermaid
flowchart LR
  X["input x"] --> F["forward(x) = y"]
  F --> Y["output y"]
  GY["grad_y = dL/dy"] --> B["backward(grad_y)<br/>hand-derived adjoint"]
  B --> GX["grad_x, grad_params"]
  F -. "verify" .-> FD{"FD check<br/>|analytic - central-diff| &lt; tol ?"}
  B -. "verify" .-> FD
  FD -->|pass| OK["op is CANONICAL<br/>(registered + tested)"]
  FD -->|fail| FIX["fix the adjoint"]
```

---

## Fig. 3 — Neocognitron pipeline (S/C hierarchy + entropy top)

```mermaid
flowchart TB
  IMG["image"] --> S["S-cell: conv2d<br/>learned oriented filters"]
  S --> V["oriented units (gx, gy)"]
  V --> C["C-cell: group_pool<br/>dihedral orbit attention<br/>(LOCAL orientation-invariance)"]
  C --> R["response map (K, H, W)"]
  R --> STACK["stack: ScBlock x N<br/>(compositional hierarchy)"]
  STACK --> EP["global_entropy_pool<br/>(GLOBAL rotation-invariance)"]
  EP --> INV["invariant Hs, trace"]
  EP --> EQV["equivariant principal angle"]
  INV --> REC["linear head -> recognition"]
  EQV --> POSE["pose (orientation)"]
  STACK --> SA["soft_argmax + skeleton hg_conv"]
  SA --> KP["keypoints (pose P2/P3/P4)"]
  PRIN["F-ARC-1: an explicit prior pays iff the base lacks the signal AND the op can express it"]:::note
  classDef note fill:#fff3cd,stroke:#d0a800,color:#5a4a00;
```

---

## Fig. 4 — Entropy global pool (one covariance → invariant recognition + equivariant pose)

```mermaid
flowchart TB
  R["response channel (H, W)"] --> W["weights w_i = resp_i^2"]
  W --> COV["response-weighted spatial covariance<br/>[[a, b], [b, d]]"]
  COV --> T["trace T = a + d"]
  COV --> DET["det Dt = a*d - b^2"]
  T --> Q["q = Dt / T^2"]
  DET --> Q
  Q --> HS["eigenvalue entropy Hs = -sum e*ln e<br/>ROTATION-INVARIANT"]
  COV --> ANG["principal angle = 0.5*atan2(2b, a-d)<br/>ROTATION-EQUIVARIANT"]
  HS --> RECOG["recognition (arrangement, invariant)"]
  ANG --> POSE["pose (orientation, equivariant)"]
  NB["O(H*W) + one 2x2 eigen · no |G| steering · closed-form backward FD-verified"]:::note
  classDef note fill:#fff3cd,stroke:#d0a800,color:#5a4a00;
```

---

## Fig. 5 — Evolvent update (forgetting-RLS online learning loop)

```mermaid
flowchart TB
  START(["stream sample (phi, y)"]) --> PRED["predict yhat = phi . w"]
  PRED --> ERR["residual e = y - yhat<br/>(prequential error)"]
  ERR --> PPHI["Pphi = P . phi"]
  PPHI --> GAIN["Kalman gain g = Pphi / (lambda + phi^T Pphi)"]
  GAIN --> WUP["w += g * e"]
  WUP --> PUP["P = (P - g (Pphi)^T) / lambda"]
  PUP --> GUARD{"trace(P) &gt; cap ?<br/>(windup guard, F-EVO-1)"}
  GUARD -->|yes| SCALE["scale P down"]
  GUARD -->|no| NEXT
  SCALE --> NEXT(["next sample"])
  NOTE["closed-form, O(d^2), NO backward sweep · verified vs batch ridge<br/>one-pass sample-efficient; per-update slower than backprop O(d)"]:::note
  classDef note fill:#fff3cd,stroke:#d0a800,color:#5a4a00;
```

---

## Fig. 6 — Signed-link balance & leakage audit (relational)

```mermaid
flowchart TB
  G["signed graph<br/>(OTC, Alpha, Slashdot, Epinions, Reddit)"] --> WEDGE["wedge (triangle-uniform) sampling<br/>unbiased balance estimator"]
  WEDGE --> BAL["balance = Z2 cycle holonomy<br/>(strong Cartwright-Harary)"]
  G --> CORE["CPML core<br/>+ rotor-holonomy channel<br/>+ Chebyshev-CR edge encoder [-1,1]"]
  CORE --> AUDIT["2x2 leakage audit"]
  AUDIT --> A1["strict x real"]
  AUDIT --> A2["strict x shuffle"]
  AUDIT --> A3["transductive x real"]
  AUDIT --> A4["transductive x shuffle"]
  A3 --> LK["leakage fraction =<br/>(transd_shuffle - 0.5) / (transd_real - 0.5)"]
  A4 --> LK
```

---

## Fig. 7 — SBSH detector (closed-form YOLO alternative)

```mermaid
flowchart LR
  IMG["image"] --> QT["dynamic quadtree<br/>(adaptive grid, replaces fixed SxS)"]
  QT --> NF["node features<br/>(+ mean-intensity DC)"]
  NF --> HEAD["oriented-bbox head<br/>Gaussian-KLD loss + anchor prior"]
  HEAD --> CS["center-sampling"]
  CS --> NMS["NMS"]
  NMS --> OUT["oriented boxes<br/>P 0.656 · R 0.966 · F1 0.781"]
  RS["rotor_spike: V1-like narrow orientation tuning"]:::note
  classDef note fill:#fff3cd,stroke:#d0a800,color:#5a4a00;
```

---

## Fig. 8 — Assimilation lifecycle (framework governance)

```mermaid
flowchart LR
  E["experiment"] --> EV["evidence review"]
  EV --> NC["novelty classification"]
  NC --> CD["canonical decision"]
  CD --> FI["framework integration<br/>(extract + import)"]
  FI --> RP["regression protection<br/>(guard + test)"]
  RP --> ST["source-of-truth update<br/>(component registry + findings ledger + assimilation report)"]
  ST --> AUTH{"next experiment<br/>authorized?"}
  AUTH -->|yes| E
```

---

## Fig. 9 — HSiKAN structural-leverage (signed KAN + causal double-dissociation)

```mermaid
flowchart TB
  IN["structured input<br/>(signed hypergraph / support chain)"] --> HSK["HSiKAN<br/>signed KAN over the structure<br/>(per-edge Chebyshev-CR basis)"]
  HSK --> OUT["prediction"]
  IN --> T1["arm: TRUE structure"]
  IN --> T2["arm: SCRAMBLED structure (ablate topology)"]
  IN --> T3["arm: DeepSets (ablate relations)"]
  T1 --> DD["double-dissociation<br/>true &gt; scramble AND true &gt; DeepSets<br/>=&gt; structure is CAUSAL (H2)"]
  T2 --> DD
  T3 --> DD
  DD --> SC["structural benefit grows 3.7x -&gt; 61x with chain length (H1 scaling)"]
```

### Fig. 9a — HSiKAN experimental design (falsification protocol)

```mermaid
flowchart TB
  H["Hypothesis: HSiKAN leverages STRUCTURE better than flat nets<br/>H1 scaling + H2 causal (NOT the naive HSiKAN &gt; MLP)"] --> TGT{"target"}
  TGT -->|bag: structure-FREE, per-node| BAG["bag target"]
  TGT -->|structural: B^2 x, needs msg-pass| STR["structural target"]
  BAG --> ARMS["arms (params-matched ~3700, 5 seeds):<br/>HSiKAN · MLP · DeepSets"]
  STR --> ARMS
  ARMS --> ABL["H2 ablation (Stage 0): degree/sign-preserving SCRAMBLE<br/>data from TRUE graph, model built on SCRAMBLED graph<br/>(preserves node/edge/degree; destroys higher-order incidence)"]
  ABL --> LAD["H1 ladder (Stage 2): chain length n = 4..16"]
  LAD --> V["verdicts corrected to the right construct<br/>(one-sided; structure-benefit GROWTH) — no threshold p-hacking"]
```

### Fig. 9b — the double-dissociation (what each ablation removes)

```mermaid
flowchart TB
  subgraph BAGT["bag target (structure-free)"]
    B1["DeepSets matches HSiKAN, beats MLP 76M×"] --> B2["=&gt; the flat win is PER-NODE architecture,<br/>NOT structure (Stage-1 18× confound explained)"]
  end
  subgraph STRT["structural target (B^2 x)"]
    S1["DeepSets is the WORST model (can't compute B^2 x)"] --> S2["=&gt; MESSAGE-PASSING is the structural part"]
  end
  B2 --> DD["DOUBLE DISSOCIATION:<br/>per-node architecture and structure are SEPARABLE"]
  S2 --> DD
  DD --> VER["H2 causal SUPPORTED: scramble drives HSiKAN below MLP (robust)"]
```

### Fig. 9c — H1 scaling (measured, 5 seeds)

Real result plot: `svg/fig9c-hsikan-scaling.png` (from `data/hsikan_ladder.json`). Left: per-model test error
(median ± IQR) vs chain length — HSiKAN·true stays low and flat (~0.001–0.005) while scramble/MLP degrade with
depth; DeepSets stuck (~0.14–0.19). Right: the scramble-isolated structure-benefit ratio grows monotonically
**3.7× → 11× → 15× → 62× → 61×**.

| n | HSiKAN·true | HSiKAN·scr | DeepSets | MLP | benefit (scr/true) | MLP/HK |
|---|---|---|---|---|---|---|
| 4 | 0.0008 | 0.0030 | 0.098 | 0.0033 | **3.7×** | 4.1× |
| 6 | 0.0044 | 0.0487 | 0.189 | 0.0178 | **11.0×** | 4.0× |
| 8 | 0.0054 | 0.0838 | 0.141 | 0.0415 | **15.4×** | 7.6× |
| 12 | 0.0044 | 0.271 | 0.170 | 0.0975 | **61.7×** | 22.2× |
| 16 | 0.0035 | 0.213 | 0.140 | 0.111 | **60.9×** | 31.6× |

---

## Fig. 10 — CV rotation-invariant texture descriptor (phase_pool / dihedral)

```mermaid
flowchart TB
  IMG["image"] --> FLD["learned orientation field (gx, gy)<br/>(quat-conv / dihedral-steer)"]
  FLD --> HIST["soft orientation histogram h<br/>(magnitude-weighted, circular)"]
  HIST --> DFT["|DFT(h)| magnitudes, low bins<br/>rotation shifts h -&gt; |DFT| invariant"]
  DFT --> FEAT["rotation-INVARIANT descriptor (phase_pool)"]
  FLD --> GC["D_n group-conv arm<br/>steer to |G| frames -&gt; group-max"]
  GC --> FEAT
  FEAT --> CLS["classify (KTH-TIPS2-b materials, rotated shapes)"]
```

---

## Fig. 11 — Rotor & holonomy geometry primitives

```mermaid
flowchart TB
  CYC["signed cycle: v0 -&gt; v1 -&gt; ... -&gt; v0"] --> RP["rotor product along the cycle<br/>(cayley_rotor per edge)"]
  RP --> HOL["rotor_holonomy<br/>ORDER-SENSITIVE loop invariant"]
  HOL --> USE["holonomy channel -&gt; CPML core (signed-link)"]
  ANG["orientation theta"] --> RS["rotor_spike<br/>von-Mises narrow tuning<br/>+ divisive normalization (V1-like)"]
  RS --> BANK["oriented tuning bank -&gt; detector / C-cell"]
```

---

## Fig. 12 — KAN / HSiKAN learnable spline op (Chebyshev-CR)

```mermaid
flowchart LR
  X["input value x"] --> KNOT["Chebyshev knots + control points"]
  KNOT --> CR["Catmull-Rom / Chebyshev-CR spline eval"]
  CR --> Y["y = learnable per-edge nonlinearity"]
  GY["grad_y"] --> BW["closed-form backward<br/>(coefs + input) · FD-verified"]
  BW --> GX["grad_x, grad_coefs"]
  NOTE["real [-1,1] edge weights beat the +/-1 indicator (OTC 0.9076 vs 0.9041)"]:::note
  classDef note fill:#fff3cd,stroke:#d0a800,color:#5a4a00;
```

---

## Fig. 13 — Scatter-locality (systems / performance)

```mermaid
flowchart LR
  G["sparse graph gather/scatter"] --> SM["sparse-mm (locality-preserving)"]
  G --> IA["index_add (scatter)"]
  SM --> CMP["compare, per-thread"]
  IA --> CMP
  CMP --> R1["2.9x @ 1-thread once accumulator &gt; L3"]
  CMP --> R2["end-to-end 1.3-1.4x only above cache AND static graph<br/>GPU atomics (82%) = where Nagare pays"]
```
