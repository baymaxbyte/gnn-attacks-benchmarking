# GOREN leakage benchmark

Runs the 8 link-stealing attacks of He et al. (USENIX Security 2021) against a GNN
trained on a coarse graph whose edge weights GOREN (Cai, Wang & Wang, ICLR 2021) has
relearned.

Results of the shipped run are in [results/GOREN_RESULTS.md](results/GOREN_RESULTS.md),
with per-run numbers in `results/*.csv` and the provenance of every step in
`manifest.json`.

---

## What the pipeline does

GOREN does not build a partition. It takes a coarse graph produced by one of Loukas's
schemes and relearns its **edge weights** against a spectral objective. So this
benchmark is a paired comparison:

| variant | coarse edge weights |
|---|---|
| `bl` | `SᵗAS`, the count of original edges between each pair of supernodes |
| `goren` | the same topology, weights predicted by GOREN's GIN |

Same k, same partition, same coarse features, same coarse labels, same split, same
classifier. Only the weight vector differs.

```
            ┌─────────────────── black box ───────────────────┐
  original  │  Loukas coarsen  ──►  GOREN relearns weights    │
  graph ───►│                  ──►  train GCN on coarse graph │
  (A,X,Y)   │                                                 │
            └──────────────────────┬──────────────────────────┘
                                   │
     adversary sends an ORIGINAL node id ──►  receives a posterior
```

The adversary never sees the coarsening, the partition, the edge weights, or which
variant she is querying. She holds the original node attributes, knows a random 10% of
labels, and is scored on recovering **original** edges. At inference the trained GCN
runs over the original graph with unit weights.

## Quick start

```bash
python config.py                      # check dependencies
python run_benchmark.py               # whole pipeline, both datasets
```

Or stage by stage:

```bash
python step01_coarsen.py              # Loukas coarsening at r = 0.3, 0.5, 0.7
python step02a_goren_weights.py       # train GOREN's GIN, produce learned weights
python step02_blackbox.py             # train + evaluate the 14 black boxes
python step03_attacker.py             # reference model g + attack datasets
python step04_attacks.py              # 8 attacks per box
python step05_aggregate.py            # CSVs, figures, writeup
```

`r` is the fraction of nodes **removed**.

| r | Cora supernodes | Citeseer supernodes |
|---|---|---|
| 0.3 | 1896 | 2330 |
| 0.5 | 1354 | 1664 |
| 0.7 | 814 | 1000 |

## Stages

| Stage | Script | Does | Writes |
|---|---|---|---|
| 1 | `step01_coarsen.py` | Loukas `variation_neighborhoods`, baseline weights | `artifacts/coarsened/*.npz`, `results/step01_coarsening.json` |
| 2a | `step02a_goren_weights.py` | trains GOREN's GIN, reports Eigenerror reduction | `artifacts/coarsened/goren_w_*.npy`, `results/step02a_goren_weights.json` |
| 2 | `step02_blackbox.py` | trains a GCN per box, both variants | `artifacts/posteriors/*.npy`, `results/step02_validation.json` |
| 3 | `step03_attacker.py` | reference model `g`, attack datasets on original edges | `attack/data/pred/`, `attack/data/attack_dataset/` |
| 4 | `step04_attacks.py` | the 8 attacks and a link-prediction baseline | `results/attacks/*.json` |
| 5 | `step05_aggregate.py` | CSVs, figures, writeup | `results/*.csv`, `results/figures/`, `results/GOREN_RESULTS.md` |

Every stage appends its inputs, outputs and headline numbers to `manifest.json`.

Stage 2a reports the paper's own success metric, the relative Eigenerror reduction, so
the weight learner can be checked independently of any classifier.

## Configuration

| Element | Value | Source |
|---|---|---|
| coarsening | Loukas `variation_neighborhoods`, K=10 | `graph_coarsening` |
| ratio | fraction of nodes removed | Loukas convention |
| baseline weights | `SᵗAS` | ours |
| GOREN model | GIN, 2 layers, emb 32, LDP node features, edge attrs init 1, final ReLU | paper §3 |
| GOREN optimiser | Adam, lr 0.001, 400 epochs | paper §3 |
| GOREN loss | diagonal of `x'Lx − (Px)'L̂(Px)` over the bottom-40 Laplacian eigenvectors, Rayleigh-normalised by `CᵗC` | authors' `loss_util` |
| projections | `P = 1/\|s\|` for signals, `C = 1/√\|s\|` for the Laplacian | authors' `loss_util` |
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

GOREN's paper evaluates coarsenings spectrally and never runs node classification, so
the classifier, split and optimiser above are ours. The rows marked "ours" have no
published counterpart.

## Outputs

```
results/GOREN_RESULTS.md              generated writeup
results/<ds>_attacks_detailed.csv     one row per (setting, attack, run)
results/<ds>_attacks_summary.csv      one row per (setting, attack), averaged
results/attacks/                      raw per-configuration JSON
results/figures/                      3 plots per dataset
results/step01_coarsening.json        coarsening geometry per setting
results/step02a_goren_weights.json    Eigenerror before and after GOREN
results/step02_validation.json        per-box accuracies
artifacts/coarsened/                  coarsening matrices, coarse graphs, learned weights
artifacts/posteriors/                 served posteriors
manifest.json                         every step, its inputs, outputs, numbers
logs/                                 stdout of each attack run
```

`detailed` is 259 rows = 7 settings × (6 supervised attacks × 5 runs + 2 unsupervised
× 1 + baseline × 5). `summary` is 63 rows with `mean / std / min / max` per metric,
plus `auc_delta_vs_whole` and `auc_goren_minus_bl`. Attacks 0 and 2 leave `accuracy`
blank rather than zero, since there is no trained classifier to score.

Column names match the companion GCond, FGC and UGC benchmarks' summary CSVs, so all
four concatenate — they attack identical node pairs under a shared pair seed.

## Dependencies

Python packages in `requirements.txt`: torch, torch-geometric, numpy, scipy, networkx,
scikit-learn, matplotlib. The attack implementation is vendored under `attack/`.

**Stage 1 has one external dependency.** Loukas's coarsening needs `pygsp` and the
GOREN repository's own `sparsenet` package, so it runs in that checkout's interpreter:

```bash
git clone https://github.com/Chen-Cai-OSU/GraphCoarsening
export GOREN_HOME=/path/to/GraphCoarsening
export GOREN_COARSEN_PYTHON=/path/to/that/venv/bin/python3
```

Stages 2a onward need none of this — the coarsening artifacts are shipped, so the rest
of the pipeline runs on `requirements.txt` alone.

Everything runs on CPU. The workloads are small enough that extra threads and MPS both
make them slower, so parallelism comes from running configurations in separate
single-threaded processes (`--workers`).

`attack/data/attack_dataset/` is generated by stage 3 and reaches several GB. It is
gitignored and regenerable.

## Layout

```
run_benchmark.py        master script
config.py               paths, ratios, variants, self-check
common.py               artifact conventions, manifest, parallel helper
data.py                 Planetoid loading and the 60/20/20 split
goren_coarsen.py        Loukas coarsening wrapper, coarse-graph construction
_coarsen_worker.py      stage 1, runs under the GOREN checkout's interpreter
goren_weights.py        GOREN's GIN weight learner and the spectral metrics
blackbox_goren.py       the black box, bl and goren variants
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
- **Weight refinement**: Cai, Wang, Wang. *Graph Coarsening with Neural Networks.*
  ICLR 2021. `goren_weights.py` is our own reimplementation of the authors' released
  model and loss.
- **Coarsening**: Loukas. *Graph Reduction with Spectral and Cut Guarantees.* JMLR
  2019, via the `graph_coarsening` package.

Please cite those papers, not this benchmark.
