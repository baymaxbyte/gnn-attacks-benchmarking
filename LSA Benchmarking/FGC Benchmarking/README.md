# FGC leakage benchmark

Runs the 8 link-stealing attacks of He et al. (USENIX Security 2021) against a GNN
whose training graph has been coarsened by FGC (Kumar et al., ICML 2023), at each of
the paper's three coarsening ratios plus a no-coarsening control.

Results of the shipped run are in [results/FGC_RESULTS.md](results/FGC_RESULTS.md),
with per-run numbers in `results/*.csv` and the provenance of every step in
`manifest.json`.

---

## What the pipeline does

```
            ┌─────────────────── black box ───────────────────┐
  original  │                                                 │
  graph ───►│  FGC coarsens  ──►  train GCN on coarse graph   │
  (A,X,Y)   │                                                 │
            └──────────────────────┬──────────────────────────┘
                                   │
     adversary sends an ORIGINAL node id ──►  receives a posterior
```

Coarsening is internal to the black box. The adversary never sees the coarse graph,
its features, or the supernode assignment. She queries original node ids, holds the
**original pre-coarsening attributes**, knows a random 10% of labels, and is scored
on recovering **original** edges.

At inference the trained GCN runs over the original adjacency, which is FGC's own
evaluation protocol: the coarse graph shapes the weights but never the serving path.

## Quick start

```bash
python config.py                      # check dependencies
python run_benchmark.py               # whole pipeline, both datasets
```

Or stage by stage:

```bash
python step01_coarsen.py              # coarsen at r = 0.3, 0.1, 0.05
python step02_blackbox.py             # train + validate the 8 black boxes
python step03_attacker.py             # reference model g + attack datasets
python step04_attacks.py              # 8 attacks per box
python step05_aggregate.py            # CSVs, figures, writeup
```

`r` is the fraction of nodes **kept**, so the coarse graph has `k = int(n*r)`
supernodes.

| r | Cora | Citeseer |
|---|---|---|
| 0.3 | 812 supernodes | 998 |
| 0.1 | 270 | 332 |
| 0.05 | 135 | 166 |

## Stages

| Stage | Script | Does | Writes |
|---|---|---|---|
| 1 | `step01_coarsen.py` | runs the FGC solver, selects `beta` per setting | `artifacts/coarsened/*.npz` |
| 2 | `step02_blackbox.py` | trains a GCN per setting, checks it against the paper | `artifacts/posteriors/*.npy`, `results/step02_validation.json` |
| 3 | `step03_attacker.py` | trains the adversary's reference model `g`, builds attack datasets on original edges | `attack/data/pred/`, `attack/data/attack_dataset/` |
| 4 | `step04_attacks.py` | runs the 8 attacks and a link-prediction baseline | `results/attacks/*.json` |
| 5 | `step05_aggregate.py` | compiles CSVs, figures and the writeup | `results/*.csv`, `results/figures/`, `results/FGC_RESULTS.md` |

Stage 2 compares every box against the published node-classification accuracy and
reports the deviation per box. Measured accuracies, targets and per-run values land
in `results/step02_validation.json`.

Every stage appends its inputs, outputs and headline numbers to `manifest.json`.

## Configuration

| Element | Value | Source |
|---|---|---|
| classifier | 2x `GCNConv`, hidden 64, dropout 0.5, log_softmax | authors' notebook |
| optimiser | Adam, lr 0.01, weight decay 1e-4, 60 epochs | paper section 4.2 |
| coarse features | `X_c = pinv(C_hard) @ X` (supernode mean) | paper Algorithm 2 |
| coarse labels | `argmax(pinv(C_hard) @ onehot(Y))` (majority) | paper Algorithm 2 |
| coarse adjacency | `-offdiag(C_hard' L C_hard)`, thresholded at 0.1, binarised | authors' notebook |
| FGC solver | 50 iterations, lambda 0.001, alpha 1, gamma 0.001 | authors' notebook |
| beta | selected per setting from {0.001, 0.01, 0.1, 1, 10} | unpublished; see reproduction notes |
| serving | `model.eval()`, posteriors exponentiated and renormalised | ours |
| attack model | MLP 3 hidden layers x 32, ReLU, dropout, softmax | He et al. section 5.1 |
| attack optimiser | Adam, lr 0.001, cross-entropy, batch 128, 50 epochs, 5 runs | He et al. section 5.1 |
| reference model `g` | graph-free mirror of the target, raw binary features | He et al. footnote |
| shadow target | the other dataset at the same coarsening ratio | He et al. default |
| adversary's labels | random 10%, seed 123 | He et al. section 5.1 |
| pairs | all original edges + equal non-edges, random 50/50 split, seed 1 | He et al. section 5.1 |

Attacks 0 and 2 are unsupervised and run once; the other six run 5 times.

## Outputs

```
results/FGC_RESULTS.md                generated writeup
results/<ds>_attacks_detailed.csv     one row per (setting, attack, run)
results/<ds>_attacks_summary.csv      one row per (setting, attack), averaged
results/attacks/                      raw per-configuration JSON
results/figures/                      4 plots per dataset
results/step01_beta_search.json       the beta selection
results/step02_validation.json        per-box accuracies vs the paper
artifacts/coarsened/                  coarsenings produced by stage 1
artifacts/posteriors/                 served posteriors
manifest.json                         every step, its inputs, outputs, numbers
logs/                                 stdout of each attack run
```

`detailed` is 148 rows = 4 settings x (6 supervised attacks x 5 runs + 2
unsupervised x 1 + baseline x 5). `summary` is 36 rows with `mean / std / min / max`
per metric plus `auc_delta_vs_whole`. Attacks 0 and 2 leave `accuracy` blank rather
than zero, since there is no trained classifier to score.

Column names match the companion GCond benchmark's summary CSV, so the two
concatenate directly.

## Diagnostics

Three optional scripts, all of which write to `results/`:

| Script | Question |
|---|---|
| `diagnose_coarsening.py` | how much label information the partition destroys; sensitivity to init seed and iteration count |
| `diagnose_citeseer.py` | why Citeseer's partition degenerates above k ~ 700 |
| `compare_variants.py` | the notebook's block-update order against a textbook alternating minimisation |

## Dependencies

Python packages in `requirements.txt`: torch, torch-geometric, numpy, scipy,
networkx, scikit-learn, matplotlib. No external repository is needed — the FGC
solver is reimplemented in `fgc_solver.py` and the attack implementation is
vendored under `attack/`.

Everything runs on CPU. The workloads are small enough that extra threads and MPS
both make them slower, so parallelism comes from running configurations in separate
single-threaded processes (`--workers`, default 8).

`attack/data/attack_dataset/` is generated by stage 3 and reaches several GB,
because each record inlines both endpoints' full attribute vectors. It is gitignored
and regenerable.

## Layout

```
run_benchmark.py        master script
config.py               paths + self-check
common.py               artifact conventions, manifest, parallel helper
data.py                 Planetoid loading
fgc_solver.py           the FGC solver and coarse-graph construction
blackbox_fgc.py         the black box
step01..step05          the stages
diagnose_*.py           optional diagnostics
attack/                 attack implementation (our own reimplementation)
  lsa/                    the 8 attacks, feature construction, models
  partial_graph_generation.py, run_attack.py
  data/dataset/original/  Planetoid pickles
  data/pred/              served posteriors of the shipped run
artifacts/  results/  logs/
```

## Attribution

- **Attacks**: He, Jia, Backes, Gong, Zhang. *Stealing Links from Graph Neural
  Networks.* USENIX Security 2021. The implementation in `attack/` is our own
  reimplementation from the paper.
- **Coarsening**: Kumar, Sharma, Saxena, Kumar. *Featured Graph Coarsening with
  Similarity Guarantees.* ICML 2023. `fgc_solver.py` is our own reimplementation of
  the authors' released solver.

