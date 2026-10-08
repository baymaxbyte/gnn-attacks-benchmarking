# Sensitive-attribute inference on German (real sensitive attribute)

A self-contained copy of the sensitive-attribute inference benchmark pointed at the
**German credit graph**, run across all reduction variants. The parent benchmark
(Cora/Citeseer) has no native sensitive attribute and uses the most class-balanced
feature bits as stand-ins. German is different: it carries a genuine binary sensitive
attribute — **Gender** — so the two-world probe targets a real protected attribute.

Results of the shipped run: [results/SENSITIVE_RESULTS.md](results/SENSITIVE_RESULTS.md),
per-cell numbers in `results/german_sia_summary.csv` / `_detailed.csv`.

## The question

Given a black box trained on a reduced German graph, and holding every feature of a
node except Gender, can an adversary recover Gender? The box is served twice — Gender
forced to 0 (Male-world) and to 1 (Female-world) — and the per-node pair of posteriors
`Z=[p0||p1]` is fed to an MLP that predicts the true Gender bit, scored by AUC
(0.5 = no leakage).

## The dataset

Built by `build_german.py` from the NIFTY-format German credit files vendored under
`../../LSA_pytorch/.../data/dataset/german/` (`german.csv`, `german_edges.txt`):

| | |
|---|---|
| nodes | 1000 |
| edges | 21,742 (symmetrised, self-loops removed, binarised) |
| features | 27 (= csv columns minus `GoodCustomer`, `OtherLoansAtStore`, `PurposeOfLoan`) |
| label | `GoodCustomer` (credit risk), 2 classes, 70% positive |
| sensitive attribute | **`Gender`** at feature index 0 (Female=1, 31% positive) |

Feature columns are **min-max normalised to [0,1]**. Binary columns — including Gender —
are left untouched by min-max, so injecting 0/1 in the two-world probe stays exactly
Male-world vs Female-world; the continuous columns (`Age`, `LoanDuration`, `LoanAmount`,
`LoanRateAsPercentOfIncome`, `YearsAtCurrentHome`, `NumberOfOtherLoansAtBank`) are
rescaled so the GCN trains stably and GCond's row-normalisation is well-behaved.

## Running it

Everything runs under the GCond repo's venv (it has torch + PyG + sklearn). The
external reductions need their home dirs because this folder is one level deeper than
the parent, so the adapters' default relative paths don't resolve:

```bash
PY="../../../GCond/GCond-main/.venv/bin/python3"

"$PY" build_german.py                              # -> german.npz
"$PY" config.py                                    # dependency + data check
"$PY" step01_reduce.py --methods control,kron,ugc,fgc
GRAPHCOARSENING_HOME="../../../GraphCoarsening/GraphCoarsening-main" \
GOREN_COARSEN_PYTHON="../../../GraphCoarsening/GraphCoarsening-main/.venv/bin/python3" \
  "$PY" step01_reduce.py --methods goren
GCOND_HOME="../../../GCond/GCond-main" \
GCOND_PYTHON="../../../GCond/GCond-main/.venv/bin/python3" \
  "$PY" step01_reduce.py --methods gcond
"$PY" step02_probe.py --workers 6                  # train boxes + two-world probe
"$PY" step03_attack.py                             # MLP attack, AUC
"$PY" step04_aggregate.py                          # CSVs, figure, writeup
```

(Use absolute paths for the env vars in practice; relative ones only work from this
folder.) 19 cells — control, KRON/UGC/FGC × 3 ratios, GOREN × 3 ratios × {bl, goren},
GCond × 3 ratios — each with 3 target seeds, one sensitive feature.

## Headline result

Gender leaks, and — unlike the flat Cora/Citeseer stand-ins — the amount depends
strongly on the reduction:

| | AUC | note |
|---|---|---|
| control (no reduction) | 0.72 | gender already recoverable from the full-graph box |
| GCond (all ratios) | 0.83–0.86 | **highest**, above control, despite the worst task accuracy (50–62%) |
| KRON | 0.62 → 0.73 → 0.82 | leakage **climbs** as more nodes are removed |
| GOREN | 0.69–0.77 | around / slightly above control |
| FGC | 0.58–0.65 | **below** control |
| UGC r0.7 | 0.54 | the only cell near chance — strongest protection |

The box barely beats German's 70% majority baseline on the credit task, yet Gender is
far more recoverable — the posterior encodes the protected attribute even when it is a
poor credit classifier. GCond's extreme condensation **concentrates** that signal
rather than removing it, echoing the Cora/Citeseer finding that condensation does not
blunt sensitive-attribute leakage.

![german_sia.png](results/figures/german_sia.png)

## Caveats specific to this run

- **UGC `alpha`**: the augmentation mixing weight has no German-tuned value in the UGC
  paper; a mid-range `0.2` is used (`reductions/ugc.py`). UGC results are sensitive to
  it, so treat the UGC column as indicative.
- **GOREN learned weights**: for `r0.5_goren` and `r0.7_goren` the GIN weight-learning
  collapsed to all-zero edge weights on this graph (the `bl` count-weight variants are
  unaffected); those two boxes effectively pass only self-loops.
- **Low task accuracy** across the board is a property of German (a small, heavily
  class-imbalanced credit graph), not a pipeline fault. `box_acc` is only a sanity
  check; the attack metric is AUC.

## Attribution

German credit graph as distributed with the fairness/link-stealing literature (NIFTY:
Agarwal et al., UAI 2021; FairGNN: Dai & Wang, WSDM 2021). Reduction and attack
attributions are as in the parent benchmark's README.
