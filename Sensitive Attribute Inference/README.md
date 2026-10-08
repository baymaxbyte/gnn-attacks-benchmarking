# Sensitive-attribute inference benchmark

Does reducing a GNN's training graph change how much a **hidden sensitive attribute**
of a node leaks? Given a black box trained on a *reduced* graph, and holding every
feature of a node except one designated-sensitive bit, can an adversary recover that
bit?

The box is a GNN trained on a reduction of the whole graph. The adversary probes it
with a two-world counterfactual: force the sensitive feature column to 0 for every
node and serve, then force it to 1 and serve. The pair of posteriors per node is the
attack's input; an MLP maps it to the true bit and is scored by AUC. Results of the
shipped run are in [results/SENSITIVE_RESULTS.md](results/SENSITIVE_RESULTS.md), with
per-cell numbers in `results/*.csv` and the provenance of every step in
`manifest.json`.

---

## What the pipeline does

Unlike the membership benchmark there is **no member / non-member split** — membership
is not what is being measured. The box simply trains on a reduction of the full graph,
and the attack runs afterwards against the trained box:

```
  G_orig ──reduce──►  reduced graph ──train──►  black-box GCN
   (A,X,Y)                                          │
                                                    │  two-world probe on the ORIGINAL graph:
                                                    ▼
   X with column s := 0 ──serve──► p0 ┐
   X with column s := 1 ──serve──► p1 ┘──►  Z_u = [p0_u ‖ p1_u]  ──► attack MLP ──► bit s of u?
```

The sensitive column is overwritten **before** any method-specific serving transform
(row-normalisation for GCond, adjacency augmentation for UGC), so the probe reaches the
model the same way the training features did. `Z_u` has dimension `2·C` (two posteriors
of `C` classes).

**The attack** (per node, from its two-world `Z_u`):
- an **MLP(32)** attack classifier is trained on a stratified split of the *same*
  dataset's nodes (where the adversary knows the true bit) and evaluated on the held-out
  half. No shadow dataset — the probe output `Z` is already a transferable signal.
- scored by **AUC** (0.5 = no leakage). Accuracy and the majority-class baseline are
  reported alongside, because the sensitive bits are imbalanced (20-40% ones).

### The sensitive attribute

Cora and Citeseer have no native sensitive attribute, so the most class-balanced binary
features stand in — three per dataset, so the finding does not hinge on one arbitrary
choice:

| dataset | sensitive features (fraction of 1s) |
|---|---|
| cora | 1177 (40%), 1263 (36%), 507 (25%) |
| citeseer | 2568 (21%), 65 (20%), 729 (20%) |

### The German variant (real sensitive attribute) — [`german/`](german/)

Cora/Citeseer stand-ins leak roughly the same regardless of the reduction. To test the
attack against a *genuine* protected attribute, [`german/`](german/) runs the identical
pipeline on the **German credit graph** (1000 nodes, 27 features, 2 classes), whose
sensitive attribute **Gender** is a real feature column (index 0, Female=1). See
[`german/README.md`](german/README.md) and `german/results/`.

Here leakage is *not* flat — it varies strongly with the reduction, and condensation
is the worst offender:

| cell | Gender AUC | |
|---|---|---|
| control (no reduction) | 0.72 | gender already recoverable |
| GCond (all ratios) | 0.83–0.86 | **highest**, above control, despite the worst task accuracy |
| KRON 0.3 / 0.5 / 0.7 | 0.62 / 0.73 / 0.82 | climbs as more nodes are removed |
| GOREN | 0.69–0.77 | around / above control |
| FGC | 0.58–0.65 | below control |
| UGC r0.7 | 0.54 | only cell near chance |

The box barely beats German's 70% majority baseline on the credit label, yet Gender
leaks strongly — and GCond's extreme condensation *concentrates* the sensitive signal
rather than removing it, reinforcing the headline finding. (`german/` ships its own
`build_german.py`, artifacts and results; it is self-contained.)

## Quick start

```bash
python config.py                 # check dependencies and that the graphs are present
python step01_reduce.py          # reduce the WHOLE graph with every method x ratio
python step02_probe.py           # train 3 boxes per cell, save two-world Z=[p0||p1]
python step03_attack.py          # MLP attack: recover the sensitive bit from Z, AUC
python step04_aggregate.py       # CSVs, figures, writeup
```

The shipped `artifacts/` already contain the reduced graphs **and** the served two-world
`Z`, so steps 03-04 reproduce the results without re-running the (slow, external)
reductions or retraining the boxes.

## Methods and ratios

| method | what it does to the graph | ratios |
|---|---|---|
| control | nothing (plain GNN on the whole graph) — the reference | — |
| KRON | Schur-complement node elimination | 0.3 / 0.5 / 0.7 removed |
| UGC | LSH coarsening on augmented features | 0.3 / 0.5 / 0.7 removed |
| FGC | featured graph coarsening (optimisation) | 0.3 / 0.1 / 0.05 (k=⌊m·r⌋) |
| GOREN | Loukas coarsening, counts (`bl`) or learned (`goren`) weights | 0.3 / 0.5 / 0.7 × {bl, goren} |
| GCond | gradient-matching condensation to synthetic nodes | 0.013 / 0.026 / 0.052 kept |

The full grid is 19 cells per dataset. The shipped run covers **37 cells** (19 cora,
18 citeseer — citeseer's most aggressive GCond ratio `r0.052` is omitted), each probed
for **3 sensitive features** with **3 target seeds**, i.e. 111 (cell, feature) attack
results.

## Stages

| Stage | Script | Writes |
|---|---|---|
| 1 | `step01_reduce.py` | `artifacts/reduced/{cell}.npz` (one reduced graph per cell) |
| 2 | `step02_probe.py` | `artifacts/Z/{cell}_feat{idx}_seed{s}.npy`, `results/step02_boxes.json` |
| 3 | `step03_attack.py` | `results/attacks/{cell}_feat{idx}.json` |
| 4 | `step04_aggregate.py` | `results/*.csv`, `results/figures/`, `results/SENSITIVE_RESULTS.md` |

Every stage appends its inputs, outputs and headline numbers to `manifest.json`.
`step02_probe.py` and `step03_attack.py` take `--only {cora,citeseer}` and
`--methods m1,m2`; `step02` runs configurations in parallel (`--workers`); `step04`
takes `--no_figures`.

## Configuration

| Element | Value |
|---|---|
| graph | whole graph, no split (box trains on its reduction) |
| classifier | one uniform 3-layer GCN (d → 64 → 64 → classes), edge weights threaded |
| optimiser | Adam, lr 0.003, decay 5e-4, 500 epochs, dropout 0.5 |
| target seeds | 3 per cell |
| probe | sensitive column forced to 0 then 1, served on the original graph; `Z=[p0‖p1]` (dim 2·C) |
| serving transform | per method (`plain` / `plain_norm` / `ugc_aug`), applied after the column is set |
| attack classifier | MLP(32), stratified same-dataset node split (50/50), standardised inputs |
| scoring | AUC recovering the sensitive bit; accuracy + majority baseline for context |

The uniform classifier is held fixed across methods on purpose, so any leakage
difference is attributable to the reduction rather than the architecture. `box_acc` in
`step02_boxes.json` is a node-classification sanity check, not the attack metric.

## Outputs

```
results/SENSITIVE_RESULTS.md         generated writeup (AUC per method/ratio/feature)
results/<ds>_sia_summary.csv         one row per cell: box acc + per-feature AUC
results/<ds>_sia_detailed.csv        one row per (cell, feature): AUC, acc, baseline, vs control
results/attacks/                     raw per-(cell,feature) JSON (per-seed AUCs)
results/figures/                     sensitive-attribute AUC per cell and feature, per dataset
results/step02_boxes.json            per-box node-classification accuracy
artifacts/reduced/                   reduced training graph per cell
artifacts/Z/                         two-world Z=[p0||p1] per cell, feature and seed
manifest.json                        every step, its inputs, outputs, numbers
```

## Dependencies

`requirements.txt`: torch, torch-geometric, numpy, scipy, scikit-learn, matplotlib.

**Stage 1 has two external dependencies, needed only to re-run the reductions:**
GCond's condensation and GOREN's Loukas coarsening run under their own checkouts'
interpreters (`GCOND_HOME` + its venv, `GRAPHCOARSENING_HOME` + pygsp). KRON, UGC, FGC
and the control need none of this. Because the reductions and the served `Z` are shipped
under `artifacts/`, stages 3-4 reproduce every result on `requirements.txt` alone.

Everything runs on CPU; parallelism comes from running configurations in separate
single-threaded processes (`--workers`).

## Layout

```
config.py  common.py  data.py  blackbox.py
step01_reduce.py .. step04_aggregate.py      the four stages
reductions/
  base.py                 the reduced-graph contract + helpers
  control.py kron.py ugc.py fgc.py goren.py gcond.py   per-method adapters
  _*_solver.py _goren_* _gcond_driver.py _goren_worker.py   vendored method code
attacks/
  sia.py                  the sensitive-attribute MLP attack classifier
cora.npz  citeseer.npz    whole graphs (all nodes, full adjacency + features + labels)
artifacts/  results/  logs/  manifest.json
```

## Attribution

- **Reductions**: GCond (Jin et al., ICLR 2022); FGC (Kumar et al., ICML 2023); UGC
  (Kataria et al., NeurIPS 2024); GOREN (Cai, Wang & Wang, ICLR 2021); Kron reduction
  (Dörfler & Bullo, arXiv 1102.2950). Each method's solver under `reductions/` is our
  own reimplementation / adaptation to reduce an arbitrary graph.
- **Sensitive-attribute inference** follows the attribute-inference setting against
  graph models: Duddu et al., *Quantifying Privacy Leakage in Graph Embedding*
  (arXiv 2010.00906), and the black-box attribute-inference study of Olatunji et al.,
  *Does Black-box Attribute Inference Attacks on Graph Neural Networks Constitute
  Privacy Risk?* (arXiv 2306.00578).

Please cite those papers, not this benchmark.
