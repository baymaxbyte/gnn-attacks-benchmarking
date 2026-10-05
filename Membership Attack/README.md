# Node membership-inference benchmark

Does reducing a GNN's training graph lower its **node membership leakage**? Given a
black box trained on a *reduced* graph, can an adversary query it with an original
node's features and tell whether that node was part of the training graph?

Each dataset is split into members and non-members, every reduction method is
applied to the members only, a uniform GCN is trained on the reduction, and a set of
membership attacks is scored by AUC. Results of the shipped run are in
[results/MEMBERSHIP_RESULTS.md](results/MEMBERSHIP_RESULTS.md), with per-cell numbers
in `results/*.csv` and the provenance of every step in `manifest.json`.

---

## What the pipeline does

The split is **inductive**, which is what makes membership well-defined:

| set | | role |
|---|---|---|
| `V_in` | 80% | members — the only nodes any training graph is built from |
| `V_out` | 20% | non-members — their nodes **and edges** are removed before any reduction |

```
  G_orig ──split──►  V_in (members)        ──reduce──►  reduced graph ──► train GCN ─┐
   (A,X,Y)           V_out (non-members)   removed                                   │
                                                                                     ▼
   query an ORIGINAL node u ───────────────────────────────►  posterior ──► attack ──► u ∈ V_in?
```

Every reduction method is applied to `V_in` **only**, so `V_out` never touches the
model. Membership is therefore defined identically across methods (member = in
`V_in`); the methods differ only in *how* `V_in` reaches the model — which is exactly
what the benchmark measures. At inference the trained box is run over the **original**
graph, giving a posterior for every node; members are the `V_in` rows, non-members
the `V_out` rows.

**The attacks** (per node, from its posterior):
- **threshold** (unsupervised): rank nodes by confidence, entropy, loss, or
  correctness — members tend to score higher.
- **shadow-MLP** (supervised): an MLP trained on the *other* dataset at the same
  cell (where membership is known) and transferred to the target. This is the
  "attack classifier" — it consumes a single node's posterior features and predicts
  member / non-member.

AUC 0.5 = no leakage.

## Quick start

```bash
python config.py                 # check dependencies
python step01_split.py           # member / non-member split (seed 42)
python step02_reduce.py          # reduce V_in with every method x ratio
python step03_blackbox.py        # train 3 boxes per cell, serve posteriors
python step04_attack.py          # threshold + shadow-MLP attacks
python step05_aggregate.py       # CSVs, figures, writeup
```

The shipped `artifacts/` already contain the reductions and served posteriors, so
steps 03-05 reproduce the results without re-running the (slow, external) reductions.

## Methods and ratios

| method | what it does to V_in | ratios |
|---|---|---|
| control | nothing (plain GNN on V_in) — the reference | — |
| KRON | Schur-complement node elimination | 0.3 / 0.5 / 0.7 removed |
| UGC | LSH coarsening on augmented features | 0.3 / 0.5 / 0.7 removed |
| FGC | featured graph coarsening (optimisation) | 0.3 / 0.1 / 0.05 (k=⌊m·r⌋) |
| GOREN | Loukas coarsening, counts (`bl`) or learned (`goren`) weights | 0.3 / 0.5 / 0.7 × {bl, goren} |
| GCond | gradient-matching condensation to synthetic nodes | 0.013 / 0.026 / 0.052 kept |

38 cells per run (19 per dataset × 2), each trained with 3 seeds.

## Stages

| Stage | Script | Writes |
|---|---|---|
| 1 | `step01_split.py` | `artifacts/splits/{ds}_split.npz` |
| 2 | `step02_reduce.py` | `artifacts/reduced/{cell}.npz` |
| 3 | `step03_blackbox.py` | `artifacts/posteriors/{cell}_seed{s}.npy`, `results/step03_boxes.json` |
| 4 | `step04_attack.py` | `results/attacks/{cell}.json` |
| 5 | `step05_aggregate.py` | `results/*.csv`, `results/figures/`, `results/MEMBERSHIP_RESULTS.md` |

Every stage appends its inputs, outputs and headline numbers to `manifest.json`.

## Configuration

| Element | Value |
|---|---|
| split | uniform-random 20% non-members, seed 42, shared by all methods |
| classifier | one uniform 3-layer GCN (d → 64 → 64 → classes), edge weights threaded |
| optimiser | Adam, lr 0.003, decay 5e-4, 500 epochs, dropout 0.5 |
| target seeds | 3 per cell |
| inference | over the original graph; features per method (`plain` / `plain_norm` / `ugc_aug`) |
| threshold attacks | confidence, entropy, loss, correctness (AUC each) |
| shadow attack | MLP (32), trained on the other dataset at the same cell, balanced |
| scoring | AUC separating members (V_in) from non-members (V_out) |

The uniform classifier is held fixed across methods on purpose, so any leakage
difference is attributable to the reduction rather than the architecture.

## Outputs

```
results/MEMBERSHIP_RESULTS.md          generated writeup
results/<ds>_membership_summary.csv    one row per cell: AUCs + acc gap
results/<ds>_membership_detailed.csv   one row per (cell, attack)
results/attacks/                       raw per-cell JSON (per-seed AUCs)
results/figures/                       membership AUC vs reduction, per dataset
results/step03_boxes.json              per-box member/non-member accuracies
artifacts/splits/                      member / non-member split per dataset
artifacts/reduced/                     reduced training graph per cell
artifacts/posteriors/                  served posteriors per cell and seed
manifest.json                          every step, its inputs, outputs, numbers
```

## Dependencies

`requirements.txt`: torch, torch-geometric, numpy, scipy, scikit-learn, matplotlib.

**Stage 2 has two external dependencies, needed only to re-run the reductions:**
GCond's condensation and GOREN's Loukas coarsening run under their own checkouts'
interpreters (`GCOND_HOME` + its venv, `GRAPHCOARSENING_HOME` + pygsp). KRON, UGC,
FGC and the control need none of this. Because the reductions and posteriors are
shipped under `artifacts/`, stages 3-5 reproduce every result on `requirements.txt`
alone.

Everything runs on CPU; parallelism comes from running configurations in separate
single-threaded processes (`--workers`).

## Layout

```
config.py  common.py  data.py  splits.py  blackbox.py
step01_split.py .. step05_aggregate.py       the five stages
reductions/
  base.py                 the reduced-graph contract + helpers
  control.py kron.py ugc.py fgc.py goren.py gcond.py   per-method adapters
  _*_solver.py _goren_* _gcond_driver.py _goren_worker.py   vendored method code
attacks/
  features.py             dimension-agnostic per-node features
  threshold.py            unsupervised threshold attacks
  shadow_clf.py           supervised shadow MLP (the attack classifier)
cora.npz  citeseer.npz    whole graphs (all nodes, full adjacency + features + labels)
artifacts/  results/  logs/  manifest.json
```

## Attribution

- **Reductions**: GCond (Jin et al., ICLR 2022); FGC (Kumar et al., ICML 2023); UGC
  (Kataria et al., NeurIPS 2024); GOREN (Cai, Wang & Wang, ICLR 2021); Kron reduction
  (Dörfler & Bullo, arXiv 1102.2950). Each method's solver under `reductions/` is our
  own reimplementation / adaptation to reduce an arbitrary subgraph.
- **Membership inference** follows the node-level MIA setting of He et al. (arXiv
  2102.05429) and Olatunji et al. (arXiv 2101.06570).

Please cite those papers, not this benchmark.
