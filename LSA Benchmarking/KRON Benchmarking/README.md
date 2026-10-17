# KRON leakage benchmark

Runs the 8 link-stealing attacks of He et al. (USENIX Security 2021) against a GNN
trained on a Kron-reduced graph (Dörfler & Bullo, *Kron Reduction of Graphs*,
arXiv 1102.2950).

Results of the shipped run are in [results/KRON_RESULTS.md](results/KRON_RESULTS.md),
with per-run numbers in `results/*.csv` and the provenance of every step in
`manifest.json`.

---

## What the pipeline does

Kron reduction eliminates a set of **interior** nodes by taking the Schur complement
of the graph Laplacian with respect to them:

```
L_red = L[a,a] − L[a,b] · inv(L[b,b]) · L[b,a]
```

The surviving **boundary** nodes `a` are a subset of the *original* nodes and keep
their own features and labels, so unlike a coarsening nothing is merged into a
synthetic supernode. The reduced edge weights are the Schur-complement conductances,
`A_red[i,j] = −L_red[i,j]`. Because the input graph is loop-less, `L_red · 1 = 0`, so
the reduced graph is itself a valid weighted graph with no self-loops — step 01 checks
this directly (`max|L_red · 1| ≈ 1e-14`).

The interior is the lowest-degree `n·r` nodes, chosen so that every connected
component keeps at least one boundary node, which is exactly the condition under which
each interior Laplacian block is invertible and the Schur complement is well defined.
So the reduction keeps the high-degree backbone and eliminates the periphery.

```
            ┌─────────────────── black box ───────────────────┐
  original  │  Kron reduce (Schur complement of L)  ──►        │
  graph ───►│  keep the high-degree backbone        ──►  train │
  (A,X,Y)   │  a GCN on the reduced graph                      │
            └──────────────────────┬──────────────────────────┘
                                   │
     adversary sends an ORIGINAL node id ──►  receives a posterior
```

The adversary never sees the reduction, which nodes were eliminated, or the
Schur-complement edge weights. She holds the original node attributes, knows a random
10% of labels, and is scored on recovering **original** edges. At inference the trained
GCN runs over the original graph with unit weights, so each reduced box differs from
its own un-reduced control only in the graph it was trained on.

## Quick start

```bash
python config.py                      # check dependencies
```

Then stage by stage:

```bash
python step01_reduce.py               # Kron reduction at r = 0.3, 0.5, 0.7
python step02_blackbox.py             # train + evaluate the 8 black boxes
python step03_attacker.py             # reference model g + attack datasets
python step04_attacks.py              # 8 attacks per box
python step05_aggregate.py            # CSVs, figures, writeup
```

`r` is the fraction of nodes **removed**.

| r | Cora nodes kept | Citeseer nodes kept |
|---|---|---|
| 0.3 | 1896 | 2329 |
| 0.5 | 1354 | 1664 |
| 0.7 | 812 | 998 |

## Stages

| Stage | Script | Does | Writes |
|---|---|---|---|
| 1 | `step01_reduce.py` | Kron / Schur-complement reduction, lowest-degree interior | `artifacts/reduced/*.npz`, `results/step01_reduction.json` |
| 2 | `step02_blackbox.py` | trains a GCN per box, evaluates on the original graph | `artifacts/posteriors/*.npy`, `results/step02_validation.json` |
| 3 | `step03_attacker.py` | reference model `g`, attack datasets on original edges | `attack/data/pred/`, `attack/data/attack_dataset/` |
| 4 | `step04_attacks.py` | the 8 attacks and a link-prediction baseline | `results/attacks/*.json` |
| 5 | `step05_aggregate.py` | CSVs, figures, writeup | `results/*.csv`, `results/figures/`, `results/KRON_RESULTS.md` |

Every stage appends its inputs, outputs and headline numbers to `manifest.json`.

## Configuration

| Element | Value | Source |
|---|---|---|
| reduction | Kron / Schur complement of the Laplacian | arXiv 1102.2950 |
| interior nodes | lowest-degree `n·r`, one boundary node per component minimum | ours |
| ratio | fraction of nodes removed | matches the UGC and GOREN benchmarks |
| reduced weights | Schur-complement conductances, threaded through the GCN | Schur complement |
| classifier | GCN, 3 `GCNConv`: d → 64 → 64 → classes, edge weights threaded through every layer | ours |
| classifier optimiser | Adam, lr 0.003, decay 0.0005, 500 epochs, dropout 0.5 | ours |
| split | random 60/20/20, best-validation weights | ours |
| inference | original graph, unit weights | ours |
| attack model | MLP 3 hidden layers × 32, ReLU, dropout, softmax | He et al. §5.1 |
| attack optimiser | Adam, lr 0.001, cross-entropy, batch 128, 50 epochs, 5 runs | He et al. §5.1 |
| reference model `g` | graph-free mirror of the target, original features | He et al. footnote |
| shadow target | the other dataset at the same setting | He et al. default |
| adversary's labels | random 10%, seed 123 | He et al. §5.1 |
| pairs | all original edges + equal non-edges, random 50/50 split | He et al. §5.1 |

Attacks 0 and 2 are unsupervised and run once; the other six run 5 times.

The Kron reduction paper is a spectral / resistive analysis of electrical networks and
never runs node classification, so the classifier, split and optimiser above are ours.
It uses neither Cora nor Citeseer. The rows marked "ours" have no published
counterpart; the one paper-grounded property, that the reduced matrix is again a valid
Laplacian, is asserted in stage 1.

## Outputs

```
results/KRON_RESULTS.md               generated writeup
results/<ds>_attacks_detailed.csv     one row per (setting, attack, run)
results/<ds>_attacks_summary.csv      one row per (setting, attack), averaged
results/attacks/                      raw per-configuration JSON
results/figures/                      2 plots per dataset
results/step01_reduction.json         reduction geometry per setting
results/step02_validation.json        per-box accuracies
artifacts/reduced/                    reduced graphs (kept ids, edges, weights, features)
artifacts/posteriors/                 served posteriors
manifest.json                         every step, its inputs, outputs, numbers
logs/                                 stdout of each attack run
```

`detailed` is 148 rows = 4 settings × (6 supervised attacks × 5 runs + 2 unsupervised
× 1 + baseline × 5). `summary` is 36 rows with `mean / std / min / max` per metric,
plus `auc_delta_vs_whole`. Attacks 0 and 2 leave `accuracy` blank rather than zero,
since there is no trained classifier to score.

Column names match the companion GCond, FGC, UGC and GOREN benchmarks' summary CSVs, so
all five concatenate — they attack identical node pairs under a shared pair seed.

## Dependencies

Python packages: torch, torch-geometric, numpy, scipy, networkx, scikit-learn,
matplotlib. The attack implementation is vendored under `attack/`.

Kron reduction is a dense linear solve, so the whole pipeline runs on those packages
alone — there is no external coarsening library or separate interpreter to set up.

Everything runs on CPU. The workloads are small enough that extra threads and MPS both
make them slower, so parallelism comes from running configurations in separate
single-threaded processes (`--workers`).

`attack/data/attack_dataset/` is generated by stage 3 and reaches several GB. It is
gitignored and regenerable.

## Layout

```
config.py               paths, ratios, selection rule, self-check
common.py               artifact conventions, manifest, parallel helper
data.py                 Planetoid loading and the 60/20/20 split
kron_reduce.py          Schur-complement reduction and boundary selection
blackbox_kron.py        the black box: train on the reduction, serve on the original
step01..step05          the stages
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
- **Reduction**: Dörfler, Bullo. *Kron Reduction of Graphs with Applications to
  Electrical Networks.* IEEE Transactions on Circuits and Systems I, 2013
  (arXiv 1102.2950). `kron_reduce.py` is our own implementation of the Schur-complement
  reduction described there.

Please cite those papers, not this benchmark.
