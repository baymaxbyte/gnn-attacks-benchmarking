# UGC leakage benchmark

Runs the 8 link-stealing attacks of He et al. (USENIX Security 2021) against a GNN
whose training graph has been coarsened by UGC (Universal Graph Coarsening, NeurIPS
2024), at each of the paper's three coarsening ratios plus a no-coarsening control.

Results of the shipped run are in [results/UGC_RESULTS.md](results/UGC_RESULTS.md),
with per-run numbers in `results/*.csv` and the provenance of every step in
`manifest.json`.

---

## What the pipeline does

```
            ┌─────────────────── black box ───────────────────┐
  original  │  X_aug = [(1-α)X | α·A]                         │
  graph ───►│  LSH coarsen  ──►  train GCN on coarse graph    │
  (A,X,Y)   │                                                 │
            └──────────────────────┬──────────────────────────┘
                                   │
     adversary sends an ORIGINAL node id ──►  receives a posterior
```

Coarsening is internal to the black box. The adversary never sees the coarse graph,
the supernode assignment, α, or the augmented feature matrix. She queries original
node ids, holds only the **original unaugmented attributes**, knows a random 10% of
labels, and is scored on recovering **original** edges.

The feature augmentation is what distinguishes UGC. Every node's input vector
carries its own adjacency row, and that augmented matrix is used at training *and*
at inference — while inference also runs over the original adjacency. The graph
therefore reaches the model twice.

## Quick start

```bash
python config.py                      # check dependencies
python run_benchmark.py               # whole pipeline, both datasets
```

Or stage by stage:

```bash
python step01_coarsen.py              # calibrate bin widths, coarsen
python step02_blackbox.py             # train + validate the 8 black boxes
python step03_attacker.py             # reference model g + attack datasets
python step04_attacks.py              # 8 attacks per box
python step05_aggregate.py            # CSVs, figures, writeup
```

`--ratio` is the percentage of nodes **removed**, matching the paper.

| ratio | Cora supernodes | Citeseer supernodes |
|---|---|---|
| 30 | 1896 | 2329 |
| 50 | 1354 | 1664 |
| 70 | 812 | 998 |

UGC is an LSH method, so the supernode count is not set directly — it emerges from a
bin width. Stage 1 calibrates one per (dataset, ratio).

## Stages

| Stage | Script | Does | Writes |
|---|---|---|---|
| 1 | `step01_coarsen.py` | calibrates bin widths, hashes and merges | `artifacts/coarsened/*.npz`, `results/step01_calibration.json` |
| 2 | `step02_blackbox.py` | trains a GCN per setting, checks it against the paper | `artifacts/posteriors/*.npy`, `results/step02_validation.json` |
| 3 | `step03_attacker.py` | trains the adversary's reference model `g`, builds attack datasets on original edges | `attack/data/pred/`, `attack/data/attack_dataset/` |
| 4 | `step04_attacks.py` | runs the 8 attacks and a link-prediction baseline | `results/attacks/*.json` |
| 5 | `step05_aggregate.py` | compiles CSVs, figures and the writeup | `results/*.csv`, `results/figures/`, `results/UGC_RESULTS.md` |

Every stage appends its inputs, outputs and headline numbers to `manifest.json`.

## Configuration

| Element | Value | Source |
|---|---|---|
| α (heterophily factor) | Cora **0.19**, Citeseer **0.26**, fixed across ratios | paper Table 6, `run.sh` |
| features | `[(1-α)X | α·A]`, dim d+n (Cora 4141, Citeseer 7030) | `UGC.py` |
| coarsening | LSH: project onto 1000 uniform random vectors, floor into bins, merge on modal bin | `UGC.py` |
| bin width | calibrated per (dataset, ratio) | `BinWidthFinder.py` |
| model | GCN, 3 `GCNConv` layers: (d+n) → 64 → 64 → classes | `GCN.py` |
| optimiser | Adam, lr 0.003, decay 0.0005, 500 epochs, `NLLLoss` | paper Table 8 |
| dropout | 0.5 after each of the first two layers | `GCN.py` |
| edge weights | passed through every layer | `GCN.py` |
| split | random 60/20/20, best-validation weights | paper |
| coarse labels | `argmax(PᵗY)` with non-training rows of Y zeroed | `UGC.py` |
| training | supernodes holding at least one training node | `UGC.py` |
| serving | `model.eval()`, posteriors exponentiated and renormalised | ours |
| attack model | MLP 3 hidden layers × 32, ReLU, dropout, softmax | He et al. §5.1 |
| attack optimiser | Adam, lr 0.001, cross-entropy, batch 128, 50 epochs, 5 runs | He et al. §5.1 |
| reference model `g` | graph-free mirror of the target, on original features | He et al. footnote |
| shadow target | the other dataset at the same coarsening ratio | He et al. default |
| adversary's labels | random 10%, seed 123 | He et al. §5.1 |
| pairs | all original edges + equal non-edges, random 50/50 split | He et al. §5.1 |

Attacks 0 and 2 are unsupervised and run once; the other six run 5 times.

Accuracy is a genuine held-out estimate: the split is a proper 60/20/20 and the
coarse labels aggregate only training labels.

## Outputs

```
results/UGC_RESULTS.md                generated writeup
results/<ds>_attacks_detailed.csv     one row per (setting, attack, run)
results/<ds>_attacks_summary.csv      one row per (setting, attack), averaged
results/attacks/                      raw per-configuration JSON
results/figures/                      3 plots per dataset
results/step01_calibration.json       bin-width search, per setting
results/step02_validation.json        per-box accuracies vs the paper
artifacts/coarsened/                  coarsenings produced by stage 1
artifacts/posteriors/                 served posteriors
manifest.json                         every step, its inputs, outputs, numbers
logs/                                 stdout of each attack run
```

`detailed` is 148 rows = 4 settings × (6 supervised attacks × 5 runs + 2
unsupervised × 1 + baseline × 5). `summary` is 36 rows with `mean / std / min / max`
per metric plus `auc_delta_vs_whole`. Attacks 0 and 2 leave `accuracy` blank rather
than zero, since there is no trained classifier to score.

Column names match the companion GCond and FGC benchmarks' summary CSVs, so all
three concatenate. Set `GCOND_RESULTS` and `FGC_RESULTS` to their `results/`
directories to have stage 5 emit the cross-benchmark table.

## Dependencies

Python packages in `requirements.txt`: torch, torch-geometric, numpy, scipy,
networkx, scikit-learn, matplotlib. No external repository is needed — the UGC
coarsening is reimplemented in `ugc_solver.py` and the attack implementation is
vendored under `attack/`.

Everything runs on CPU. The workloads are small enough that extra threads and MPS
both make them slower, so parallelism comes from running configurations in separate
single-threaded processes (`--workers`, default 8).

`attack/data/attack_dataset/` is generated by stage 3 and reaches several GB. It is
gitignored and regenerable.

## Layout

```
run_benchmark.py        master script
config.py               paths, α, ratios, self-check
common.py               artifact conventions, manifest, parallel helper
data.py                 Planetoid loading and the 60/20/20 split
ugc_solver.py           augmentation, LSH hashing, bin-width search, coarse graph
blackbox_ugc.py         the black box
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
- **Coarsening**: Kumar, Sharma, Vashishtha, Kumar et al. *Universal Graph
  Coarsening.* NeurIPS 2024. `ugc_solver.py` is our own reimplementation of the
  authors' released code.

Please cite those papers, not this benchmark.
