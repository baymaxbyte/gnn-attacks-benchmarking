# GCond leakage benchmark

Runs the 8 link-stealing attacks of He et al. (USENIX Security 2021) against a
black box whose training graph has been condensed by GCond (Jin et al., ICLR
2022), at each of GCond's three reduction rates plus a no-condensation control.

Results of the shipped run are in
[results/GCOND_RESULTS.md](results/GCOND_RESULTS.md), with the per-run numbers in
`results/*.csv` and the full provenance of every step in `manifest.json`.

---

## What the pipeline does

```
            ┌─────────────────── black box ───────────────────┐
  original  │                                                 │
  graph ───►│  GCond condenses  ──►  train GCN on condensed   │
  (A,X,Y)   │                        graph                    │
            └──────────────────────┬──────────────────────────┘
                                   │
     adversary sends an ORIGINAL node id ──►  receives a posterior
```

Condensation is internal to the black box: the adversary never sees the condensed
graph, its features or its node count. She queries original node ids, holds the
original pre-condensation attributes, knows a random 10% of labels, and is scored
on her ability to recover **original** edges. A reference model `g` is trained
only for the attacks that assume attribute knowledge.

Each run produces one black box per reduction setting, one adversary per dataset,
and 8 attack scores plus a link-prediction baseline per setting.

## Datasets and attack directions

Runs on **four** datasets — the two homophilous citation graphs plus two
heterophilous Wikipedia graphs added later:

| dataset | family | nodes | undirected edges | classes |
|---|---|---|---|---|
| cora | citation (Planetoid) | 2708 | 5278 | 7 |
| citeseer | citation (Planetoid) | 3327 | 4552 | 6 |
| chameleon | Wikipedia (heterophilous) | 2277 | 31371 | 5 |
| squirrel | Wikipedia (heterophilous) | 5201 | 198353 | 5 |

Attacks run in four target→shadow directions (shadow = the other dataset in the
same family, at the same setting):

| direction | target → shadow |
|---|---|
| A | cora → citeseer |
| B | citeseer → cora |
| C | chameleon → squirrel |
| D | squirrel → chameleon |

The dense graphs cap positive pairs at 10 000 (`MAX_POS`) so the attack datasets
stay tractable; cora/citeseer (< 10 000 edges) are unaffected and reproduce exactly.

**Condensation budget (standard).** A single `CONDENSE_EPOCHS = 600` is used for
every dataset — the paper's value and the one the cora/citeseer Table 2 gate uses.
GCond is a CUDA method; on CPU the dense chameleon/squirrel condensations are slow
(squirrel r=1.0 ≈ hours), so run stage 1 on a GPU host where possible. chameleon and
squirrel have no published target, so their boxes are reported, not gated.

## Quick start

```bash
python config.py                      # check dependencies
python run_benchmark.py               # all four datasets, directions A–D
```

`run_benchmark.py` condenses/boxes/attacks `--datasets cora,citeseer,chameleon,squirrel`
across `--ratios 0.25,0.5,1.0` and `--directions A,B,C,D` by default (e.g.
`--datasets chameleon,squirrel --directions C,D` for the heterophilous pair only).
It points `GCOND_HOME`/`GCOND_PYTHON` at `../../../../GCond/GCond-main` unless you
export them.

`--r`/`--ratios` is GCond's CLI reduction rate: a fraction of the dataset's
**labeling rate**, so `0.25 / 0.5 / 1.0` are the paper's three settings.

| `--r` | Cora | Citeseer | chameleon | squirrel |
|---|---|---|---|---|
| 0.25 | 35 nodes | 30 | 56 | 129 |
| 0.5 | 70 nodes | 60 | 113 | 259 |
| 1.0 | 140 nodes | 120 | 227 | 519 |

| Flag | Effect |
|---|---|
| `--datasets` | which datasets to condense/box/attack (default all four) |
| `--ratios` | condensation rates (default `0.25,0.5,1.0`) |
| `--directions` | attack directions (default `A,B,C,D`) |
| `--no_whole` | skip the no-condensation control (not recommended) |
| `--skip_condense` | reuse condensed graphs already in `artifacts/condensed/` |
| `--skip_attacks` | stop after the validation gate |
| `--repeats N` | black-box evaluation repeats (default 10; squirrel is slow — try 3) |
| `--runs N` | attack-model repeats (default 5) |
| `--no_clustering` | skip the K-means figures, which re-read the attack JSONL |
| `--force` | recompute instead of reusing artifacts |
| `--device` | `cpu` or `cuda` |
| `--gate_tolerance` | accuracy points of deviation from the paper allowed (paperless datasets are not gated) |

## Stages

`run_benchmark.py` runs these in order; each is also runnable standalone.

| Stage | Script | Does | Writes |
|---|---|---|---|
| 1 | `step01_condense.py` | invokes GCond's `train_gcond_transduct.py` for target and shadow | `artifacts/condensed/{adj,feat,label}_*.pt` |
| 2 | `step02_blackbox.py` | trains a GCN per setting, checks it against GCond Table 2 | `results/step02_validation.json` |
| 3 | `step03_attacker.py` | trains the adversary's reference model `g`, builds attack datasets on original edges | `attack/data/pred/*_pred.pkl`, `attack/data/attack_dataset/` |
| 4 | `step04_attacks.py` | runs the 8 attacks and the link-prediction baseline | `results/attacks/*.json` |
| 5 | `step05_aggregate.py` | compiles CSVs, figures and the writeup | `results/*.csv`, `results/figures/`, `results/GCOND_RESULTS.md` |

Stage 2 is a **gate**. If any black box deviates from the published accuracy by
more than `--gate_tolerance` points, the run aborts before a single attack is
executed. The per-box targets, measured means and deviations are written to
`results/step02_validation.json`.

Every stage appends its inputs, outputs and headline numbers to `manifest.json`,
so a finished run is self-describing.

## Outputs

```
results/<ds>_attacks_detailed.csv     one row per (setting, attack, run)
results/<ds>_attacks_summary.csv      one row per (setting, attack), averaged
results/figures/                      6 plots per dataset
results/attacks/                      raw per-configuration JSON
results/GCOND_RESULTS.md              generated writeup
results/step02_validation.json        per-run black-box accuracies vs the paper
results/unsupervised_clustering.json  K-means cluster geometry
manifest.json                         every step, its inputs, outputs, numbers
logs/                                 stdout of each condensation and attack run
```

`detailed` is 148 rows = 4 settings × (6 supervised attacks × 5 runs + 2
unsupervised × 1 + baseline × 5). `summary` is 36 rows with `mean / std / min /
max` per metric.

Attacks 0 and 2 are unsupervised, so they run once and their `accuracy` column is
blank rather than zero. They still report AUC and K-means precision/recall/F1,
the latter in `results/unsupervised_clustering.json`.

## Dependencies

GCond's code is **not redistributed** here. Clone it and point `config.py` at it:

```bash
git clone https://github.com/ChandlerBang/GCond
export GCOND_HOME=/path/to/GCond/GCond-main
export GCOND_PYTHON=/path/to/that/venv/bin/python3   # optional
```

`python config.py` prints the resolved paths and fails loudly if anything is
missing. On a machine without CUDA that checkout needs two small patches: a CPU
fallback in `train_gcond_transduct.py`, and `--save` redirected away from the
authors' bundled `saved_ours/` directory so it is not overwritten. Python
packages for this repo are in `requirements.txt`; GCond additionally needs
`deeprobust`.

`attack/data/attack_dataset/` is generated by stage 3 and reaches several GB,
because each record inlines both endpoints' full attribute vectors. It is
gitignored and regenerable.

## Layout

```
run_benchmark.py        master script
config.py               external dependency paths + self-check
common.py               artifact conventions, manifest writer
blackbox_gcond.py       the black box
step01..step05          the stages
manifest.json           machine-readable record of the shipped run
attack/                 attack implementation (our own reimplementation)
  lsa/                    the 8 attacks, feature construction, models
  partial_graph_generation.py, run_attack.py
  data/dataset/original/  Planetoid pickles
  data/pred/              served posteriors of the shipped run
artifacts/condensed/    condensed graphs produced by stage 1
results/                CSVs, figures, JSON, writeup
logs/                   per-run stdout
```

## Attribution

- **Attacks**: He, Jia, Backes, Gong, Zhang. *Stealing Links from Graph Neural
  Networks.* USENIX Security 2021. The implementation in `attack/` is our own
  reimplementation from the paper.
- **Condensation**: Jin, Zhao, Zhang, Liu, Tang, Shah. *Graph Condensation for
  Graph Neural Networks.* ICLR 2022. Code used from the authors' repository.

Please cite those papers, not this benchmark.
