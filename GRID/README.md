# GRID — Graph Link Disguise (reimplementation + benchmark)

A from-scratch reimplementation of **GRID**, the post-hoc link-stealing defence
of Lou et al., *"Graph Link Disguise: ..."* (IEEE S&P 2025, arXiv
[2501.10985](https://arxiv.org/abs/2501.10985)), together with the experiments
we ran to test how well it actually defends the black boxes in this benchmark.

The authors released no code, so everything here is our own implementation of
the algorithms described in the paper. The paper PDF and our draft
author-contact email are intentionally **not** included in this folder.

---

## What GRID does

GRID defends against *similarity-based* link-stealing attacks (LSA), where an
attacker infers whether an edge `(i, j)` exists from how similar the served
posteriors `v_i` and `v_j` are. GRID adds a small noise vector `s` to the
posteriors of a **subset** of nodes so that adjacent-node similarity is dragged
down to the level of *n-hop indirect* neighbours — **without changing any
predicted label**.

Per the paper, the implementation (`grid_defense.py`) covers:

- **Similarity** `sim(a,b) = pearson(a,b) + cosine(a,b)` (paper Eqn. 3).
- **Core-node selection** (Algorithm 1): a greedy vertex cover over the edges
  whose endpoint similarity exceeds a floor `δ`, keeping the higher-degree
  endpoint. Only these "core" nodes are perturbed.
- **`δ` estimation**: mean similarity over sampled exactly-`n`-hop node pairs —
  the "indirect neighbour" level GRID disguises adjacent nodes down to.
- **OPT-GRID objective** (Algorithm 2): projected gradient descent that pushes
  each core node's similarity to its neighbours down to the `δ` floor (hinge
  loss), subject to the constraints.
- **ConstraintCheck projection** (Algorithm 3) + a hard finaliser enforcing, per
  node: argmax preserved (zero label loss), `v+s` a valid distribution
  (`≥0`, sums to 1), and distortion budget `‖s‖₁ ≤ θ`.

Only the **target** (served) posteriors are perturbed; the attacker's
reference/shadow model is left untouched, matching the paper's threat model.

### Hyperparameters

| Name | Flag | Default | Meaning |
|------|------|---------|---------|
| Distortion budget | `--theta` | `0.4` | L1 cap on the noise, `‖s‖₁ ≤ θ`. |
| Indirect-neighbour hop | `--nhop` | `3` | distance defining the `δ` similarity floor. |
| Solver iterations | `--iters` | `60` | projected-GD steps per core node. |
| Solver step size | `--lr` | `0.25` | gradient step for the OPT-GRID objective. |

---

## Key findings

**Headline:** in this benchmark GRID's strength tracks how *diffuse* the target
posteriors already are. It meaningfully suppresses the attack only on
low-confidence **GCond**-condensed boxes; on confident coarsening / whole boxes
(posteriors near one-hot) the effect is small. Across every variation we tried —
a soft 2-layer GCN, an (near-)optimal solver, and `θ` pushed all the way to
`1.0`, all at **0 % utility loss** — we could not reproduce the large
uncoarsened AUC drops reported in the paper.

All numbers below are attack AUC (lower = better defence); `Δ` is GRID − no-defence.
`link_pred` and box/node accuracy are **unchanged** in every run (zero label changes).

### 1. Whole (uncoarsened) KRON box, θ = 0.4 — `results/grid_vs_baseline_theta0.4.csv`

| dataset | attack | no-defence | GRID | Δ |
|---------|--------|-----------:|-----:|----:|
| cora | Attack-0/2 | 0.9202 | 0.8936 | −0.027 |
| cora | Attack-1 | 0.9185 | 0.8849 | −0.034 |
| citeseer | Attack-0/2 | 0.9535 | 0.9121 | −0.041 |

Modest drops; several attacks move up slightly (within run-to-run noise).

### 2. Full method × ratio sweep, θ = 0.4, n = 3 — `results/grid_all_theta0.4.csv` (414 rows)

Mean Δ over all 8 attacks, by coarsening/condensation method:

| method | mean Δ | note |
|--------|-------:|------|
| **gcond** | **−0.079** | by far the most defendable (diffuse posteriors) |
| fgc  | −0.015 | |
| kron | −0.011 | |
| ugc  | −0.010 | |
| goren | −0.009 | |

Largest single drop: `citeseer / gcond / r=0.25 / Attack-1`,
0.917 → 0.623 (**−0.294**); `cora / gcond / r=0.25 / Attack-1`, 0.899 → 0.639
(−0.259). Box accuracy is unchanged throughout.

### 3. `n`-hop sweep — `results/nhop_sweep.csv`

Raising the indirect-neighbour floor from `n=3` to `5`/`6` barely moves the
needle (cora Attack-1: −0.034 → −0.042); the `θ` budget binds first.

### 4. θ sweep (whole KRON box) — `results/theta06.csv`, `results/theta10.csv`

| attack | θ=0.4 | θ=0.6 | θ=1.0 |
|--------|------:|------:|------:|
| cora Attack-1 | −0.034 | −0.040 | −0.071 |
| citeseer Attack-0/2 | −0.041 | −0.056 | −0.112 |
| citeseer Attack-1 | +0.001 | −0.026 | −0.089 |

Even at θ=1.0 (the maximum possible L1 budget) box accuracy is unchanged (0
label changes, max `‖s‖₁ = 1.0`) — but the drops still fall short of the paper.

### 5. Paper-faithful soft 2-layer GCN — `paper_faithful_gcn/results/soft_gcn_grid.csv`

Retraining the target as a standard 2-layer Kipf GCN (softer posteriors, mean
peak-class cosine ≈ 0.93–0.97, vs near-one-hot before) roughly **doubles** GRID's
effect (cora Attack-1: 0.949 → 0.870, −0.080; citeseer Attack-1: 0.972 → 0.886,
−0.086) — still well below the paper's reported uncoarsened drops.

### 6. Paper-faithful D1 (40 % training subsample) — `paper_faithful_d1/results/d1_grid.csv`

Training on a 40 % node subsample (to mimic the paper's `D1` setup) did **not**
help (cora Attack-1: −0.028; citeseer Attack-1: −0.069), refuting the idea that
training-set overlap explained the gap. The gap is on the defence/`θ` side.

---

## Directory layout

```
GRID/
├── grid_defense.py            # core: GRID algorithms 1–3 + solver + validation
├── run_grid.py                # self-contained: GRID on the whole KRON box (cora/citeseer)
├── run_grid_all.py            # full sweep over all methods × ratios (needs external attack dirs)
├── nhop_sweep.py              # n-hop sweep driver
├── run_theta06.py             # θ=0.6 driver (+ box-accuracy check)
├── run_theta10.py             # θ=1.0 driver (+ box-accuracy check)
├── aggregate_summary.py, add_boxacc.py, format_full.py,
│   top_drops.py, max_drop.py, diag_optimizer.py   # analysis / tabulation helpers
├── plot_degree_dist.py, plot_degree_all.py         # degree-distribution plots
├── artifacts/posteriors/      # served posteriors (.npy) used as GRID input
├── attack/                    # LSA attack CODE only (He et al.); see "Data" below
│   ├── lsa/                   #   attack models, baselines, data loaders
│   ├── partial_graph_generation.py
│   └── run_attack.py
├── paper_faithful_gcn/        # soft 2-layer GCN variant (train_soft_gcn.py, run_soft.py) + results/
├── paper_faithful_d1/         # 40% subsample variant (d1_experiment.py) + results/
└── results/                   # all CSVs + per-attack JSON (attacks/, attacks_all/, nhop_sweep/, theta06/, theta10/)
```

---

## Data dependency (why `attack/data/` is not here)

`attack/` ships the link-stealing attack **code** only. The datasets, black-box
prediction pickles, and generated attack datasets
(`attack/data/{dataset,pred,attack_dataset}`) are **excluded** — they are large
(>1 GB when materialised) and fully regenerable.

To re-run:

- **`run_grid.py`** (self-contained) needs `attack/data/dataset/original/` and
  the KRON black-box posterior pickles in `attack/data/pred/`. Populate these
  from the sibling **`LSA Benchmarking/KRON Benchmarking/attack/`** in this repo;
  the attack datasets under `attack/data/attack_dataset/` are then generated
  on the fly.
- **`run_grid_all.py`** (full multi-method sweep) expects an external
  `../lsa/<Method> Benchmarking/attack/` layout (one per coarsening method) and
  will not run standalone from this folder. The CSVs it produced are included.

## How to run (once data is in place)

```bash
# whole-box cora/citeseer, θ=0.4 (writes results/grid_vs_baseline_theta0.4.csv)
python run_grid.py --theta 0.4 --nhop 3 --runs 5

# θ sweeps
python run_theta06.py
python run_theta10.py

# n-hop sweep
python nhop_sweep.py

# apply GRID to one box directly
python grid_defense.py --dataset cora --setting whole --theta 0.4 --nhop 3
```

Dependencies: `numpy`, `torch`, `scipy`, `scikit-learn` (same environment as the
`LSA Benchmarking` attacks).

---

## Provenance & citation

- **Defence:** Lou et al., *Graph Link Disguise* (GRID), IEEE S&P 2025 —
  arXiv [2501.10985](https://arxiv.org/abs/2501.10985). Reimplemented here from
  the paper; no official code was available.
- **Attack:** the He et al. link-stealing-attack suite (the `attack/lsa/` code),
  as vendored in this repository's `LSA Benchmarking`.

This folder is a reproduction study; the negative/nuanced results above are ours
and reflect this benchmark's black boxes, not a claim about the original paper's
setup.
