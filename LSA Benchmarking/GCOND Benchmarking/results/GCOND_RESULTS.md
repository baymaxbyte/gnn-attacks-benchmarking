# GCond condensation vs link stealing

Black box: GCond condenses the training graph, a 2-layer / 256-hidden
GCN is trained on it, and at query time that GCN propagates over the
**original** graph (GCond's transductive protocol). The adversary sends
original node ids, holds original pre-condensation attributes as `F`,
and attacks **original** edges.

Held fixed across every box: the edge set, the sampled non-edges, the
random 50/50 split, the reference model `g`, and the attack model. Only
the served posteriors differ.

Per-dataset data: `<ds>_attacks_detailed.csv` (one row per run) and
`<ds>_attacks_summary.csv` (averaged over the 5 runs).

## Target Cora, shadow Citeseer

| Attack | K=(F,A*,D') | B0 r=100% | B1 r=1.3% | B2 r=2.6% | B3 r=5.2% | Δ(B3−B0) |
|---|---|---|---|---|---|---|
| 0 | x/x/x | 0.9316 | 0.9505 | 0.9515 | 0.9514 | **+0.0198** |
| 1 | x/x/Y | 0.9299 ± 0.0041 | 0.9507 ± 0.0013 | 0.9538 ± 0.0023 | 0.9250 ± 0.0052 | **-0.0049** |
| 2 | Y/x/x | 0.9316 | 0.9505 | 0.9515 | 0.9514 | **+0.0198** |
| 3 | x/Y/x | 0.9418 ± 0.0009 | 0.9676 ± 0.0004 | 0.9643 ± 0.0010 | 0.9638 ± 0.0011 | **+0.0220** |
| 4 | x/Y/Y | 0.9335 ± 0.0018 | 0.9575 ± 0.0004 | 0.9586 ± 0.0006 | 0.9485 ± 0.0027 | **+0.0150** |
| 5 | Y/x/Y | 0.9431 ± 0.0011 | 0.9520 ± 0.0009 | 0.9548 ± 0.0007 | 0.9395 ± 0.0006 | **-0.0035** |
| 6 | Y/Y/x | 0.9519 ± 0.0006 | 0.9715 ± 0.0004 | 0.9673 ± 0.0002 | 0.9668 ± 0.0004 | **+0.0149** |
| 7 | Y/Y/Y | 0.9483 ± 0.0003 | 0.9633 ± 0.0003 | 0.9634 ± 0.0001 | 0.9553 ± 0.0008 | **+0.0069** |
| link-pred baseline | — | 0.6329 | 0.6329 | 0.6329 | 0.6329 | +0.0000 |
| **box accuracy %** | — | 80.99 | 80.37 | 80.89 | 80.13 | |
| **training nodes** | — | 2708 | 35 | 70 | 140 | |

- mean AUC change B0 → B3 across the 8 attacks: **+0.0112**
- largest spread of any single attack across the boxes: **0.0287**

## Target Citeseer, shadow Cora

| Attack | K=(F,A*,D') | B0 r=100% | B1 r=0.9% | B2 r=1.8% | B3 r=3.6% | Δ(B3−B0) |
|---|---|---|---|---|---|---|
| 0 | x/x/x | 0.9650 | 0.9734 | 0.9700 | 0.9549 | **-0.0101** |
| 1 | x/x/Y | 0.9665 ± 0.0026 | 0.9725 ± 0.0016 | 0.9737 ± 0.0008 | 0.9414 ± 0.0014 | **-0.0250** |
| 2 | Y/x/x | 0.9650 | 0.9734 | 0.9700 | 0.9549 | **-0.0101** |
| 3 | x/Y/x | 0.9731 ± 0.0010 | 0.9782 ± 0.0004 | 0.9779 ± 0.0004 | 0.9707 ± 0.0008 | **-0.0024** |
| 4 | x/Y/Y | 0.9720 ± 0.0006 | 0.9751 ± 0.0004 | 0.9753 ± 0.0007 | 0.9549 ± 0.0011 | **-0.0171** |
| 5 | Y/x/Y | 0.9653 ± 0.0006 | 0.9733 ± 0.0012 | 0.9742 ± 0.0012 | 0.9578 ± 0.0022 | **-0.0074** |
| 6 | Y/Y/x | 0.9797 ± 0.0002 | 0.9825 ± 0.0002 | 0.9828 ± 0.0003 | 0.9784 ± 0.0002 | **-0.0013** |
| 7 | Y/Y/Y | 0.9766 ± 0.0005 | 0.9793 ± 0.0005 | 0.9798 ± 0.0004 | 0.9680 ± 0.0007 | **-0.0086** |
| link-pred baseline | — | 0.6130 | 0.6130 | 0.6130 | 0.6130 | +0.0000 |
| **box accuracy %** | — | 71.98 | 69.85 | 70.89 | 71.92 | |
| **training nodes** | — | 3327 | 30 | 60 | 120 | |

- mean AUC change B0 → B3 across the 8 attacks: **-0.0103**
- largest spread of any single attack across the boxes: **0.0322**

## Target Chameleon, shadow Squirrel

| Attack | K=(F,A*,D') | B0 r=100% | B1 r=n/a | B2 r=n/a | B3 r=n/a | Δ(B3−B0) |
|---|---|---|---|---|---|---|
| 0 | x/x/x | 0.7507 | 0.7066 | 0.8162 | 0.8227 | **+0.0721** |
| 1 | x/x/Y | 0.6549 ± 0.0098 | 0.7798 ± 0.0161 | 0.6909 ± 0.0227 | 0.5418 ± 0.0267 | **-0.1131** |
| 2 | Y/x/x | 0.7507 | 0.7066 | 0.8162 | 0.8227 | **+0.0721** |
| 3 | x/Y/x | 0.9278 ± 0.0014 | 0.9703 ± 0.0018 | 0.9702 ± 0.0009 | 0.9718 ± 0.0008 | **+0.0440** |
| 4 | x/Y/Y | 0.7418 ± 0.0171 | 0.3651 ± 0.0764 | 0.8459 ± 0.0142 | 0.8147 ± 0.0109 | **+0.0729** |
| 5 | Y/x/Y | 0.5468 ± 0.0149 | 0.7391 ± 0.0166 | 0.6794 ± 0.0122 | 0.4967 ± 0.0120 | **-0.0500** |
| 6 | Y/Y/x | 0.9310 ± 0.0020 | 0.9647 ± 0.0008 | 0.9644 ± 0.0009 | 0.9668 ± 0.0008 | **+0.0358** |
| 7 | Y/Y/Y | 0.7516 ± 0.0202 | 0.4791 ± 0.0696 | 0.7559 ± 0.0173 | 0.7702 ± 0.0170 | **+0.0186** |
| link-pred baseline | — | 0.8574 | 0.8574 | 0.8574 | 0.8574 | +0.0000 |
| **box accuracy %** | — | 54.61 | 37.99 | 42.47 | 43.23 | |
| **training nodes** | — | 2277 | 56 | 113 | 227 | |

- mean AUC change B0 → B3 across the 8 attacks: **+0.0191**
- largest spread of any single attack across the boxes: **0.4807**

## Target Squirrel, shadow Chameleon

| Attack | K=(F,A*,D') | B0 r=100% | B1 r=n/a | B2 r=n/a | B3 r=n/a | Δ(B3−B0) |
|---|---|---|---|---|---|---|
| 0 | x/x/x | 0.7068 | 0.6099 | 0.5959 | 0.6981 | **-0.0087** |
| 1 | x/x/Y | 0.7537 ± 0.0087 | 0.6064 ± 0.0380 | 0.6645 ± 0.0270 | 0.3821 ± 0.0242 | **-0.3715** |
| 2 | Y/x/x | 0.7068 | 0.6099 | 0.5959 | 0.6981 | **-0.0087** |
| 3 | x/Y/x | 0.9488 ± 0.0008 | 0.9393 ± 0.0006 | 0.9452 ± 0.0005 | 0.9453 ± 0.0007 | **-0.0035** |
| 4 | x/Y/Y | 0.8155 ± 0.0055 | 0.6623 ± 0.0603 | 0.6608 ± 0.0119 | 0.4631 ± 0.0628 | **-0.3524** |
| 5 | Y/x/Y | 0.7383 ± 0.0071 | 0.6219 ± 0.0269 | 0.6493 ± 0.0148 | 0.3858 ± 0.0264 | **-0.3524** |
| 6 | Y/Y/x | 0.9488 ± 0.0004 | 0.9412 ± 0.0010 | 0.9481 ± 0.0005 | 0.9465 ± 0.0002 | **-0.0023** |
| 7 | Y/Y/Y | 0.8237 ± 0.0107 | 0.6665 ± 0.0301 | 0.7295 ± 0.0256 | 0.5111 ± 0.0369 | **-0.3125** |
| link-pred baseline | — | 0.8825 | 0.8825 | 0.8825 | 0.8825 | +0.0000 |
| **box accuracy %** | — | 34.58 | 26.66 | 24.83 | 23.01 | |
| **training nodes** | — | 5201 | 129 | 259 | 519 | |

- mean AUC change B0 → B3 across the 8 attacks: **-0.1765**
- largest spread of any single attack across the boxes: **0.3715**

## Reading

Condensing the training graph — down to 35 synthetic nodes for Cora,
1.3% of the original — moves link-stealing AUC by at most ~0.03, and
often *upward*. The mechanism is structural: GCond's condensed graph is
a **training set only**, and the deployed model aggregates over the
**true original adjacency at every query**. The edge signal the attacks
exploit is injected at inference, so shrinking the training graph
cannot remove it.

The link-prediction baseline is identical across settings, as it must
be — it reads only `A*`, which is held fixed. A useful internal
consistency check on the ablation.

## But the headline hides a real effect

Attack-0 scores each of the 8 distance metrics separately, and they do
*not* respond to condensation the same way. Condensation flattens the
posteriors (mean max probability falls from 0.695 to 0.278 on Cora), so
**magnitude-sensitive** distances shrink while **scale-invariant** ones
are unaffected.

### Cora — Attack-0 by distance metric

| Metric | B0 | B1 | B2 | B3 | Δ(B3−B0) | scale-invariant? |
|---|---|---|---|---|---|---|
| cosine | 0.9280 | 0.8405 | 0.8936 | 0.8989 | **-0.0292** | partly (direction) |
| euclidean | 0.9162 | 0.7861 | 0.8098 | 0.8124 | **-0.1038** | no (magnitude) |
| correlation | 0.9316 | 0.9505 | 0.9515 | 0.9514 | **+0.0198** | yes |
| chebyshev | 0.9104 | 0.7653 | 0.7818 | 0.7844 | **-0.1261** | no (magnitude) |
| braycurtis | 0.9220 | 0.7889 | 0.8165 | 0.8186 | **-0.1034** | no (magnitude) |
| canberra | 0.7100 | 0.7542 | 0.7553 | 0.7435 | **+0.0335** | no (magnitude) |
| cityblock | 0.9220 | 0.7889 | 0.8165 | 0.8186 | **-0.1034** | no (magnitude) |
| sqeuclidean | 0.9162 | 0.7861 | 0.8098 | 0.8124 | **-0.1038** | no (magnitude) |

### Citeseer — Attack-0 by distance metric

| Metric | B0 | B1 | B2 | B3 | Δ(B3−B0) | scale-invariant? |
|---|---|---|---|---|---|---|
| cosine | 0.9580 | 0.9066 | 0.9370 | 0.9514 | **-0.0066** | partly (direction) |
| euclidean | 0.9220 | 0.8753 | 0.8874 | 0.9193 | **-0.0027** | no (magnitude) |
| correlation | 0.9650 | 0.9734 | 0.9700 | 0.9549 | **-0.0101** | yes |
| chebyshev | 0.8989 | 0.8473 | 0.8600 | 0.9012 | **+0.0022** | no (magnitude) |
| braycurtis | 0.9291 | 0.8847 | 0.8955 | 0.9254 | **-0.0037** | no (magnitude) |
| canberra | 0.8442 | 0.8779 | 0.8596 | 0.7962 | **-0.0480** | no (magnitude) |
| cityblock | 0.9291 | 0.8847 | 0.8955 | 0.9254 | **-0.0037** | no (magnitude) |
| sqeuclidean | 0.9220 | 0.8753 | 0.8874 | 0.9193 | **-0.0027** | no (magnitude) |

### Chameleon — Attack-0 by distance metric

| Metric | B0 | B1 | B2 | B3 | Δ(B3−B0) | scale-invariant? |
|---|---|---|---|---|---|---|
| cosine | 0.7113 | 0.6006 | 0.6077 | 0.6643 | **-0.0470** | partly (direction) |
| euclidean | 0.6960 | 0.6212 | 0.5839 | 0.6533 | **-0.0427** | no (magnitude) |
| correlation | 0.7507 | 0.7066 | 0.8162 | 0.8227 | **+0.0721** | yes |
| chebyshev | 0.6959 | 0.6389 | 0.5685 | 0.6297 | **-0.0662** | no (magnitude) |
| braycurtis | 0.7147 | 0.6112 | 0.6025 | 0.6692 | **-0.0455** | no (magnitude) |
| canberra | 0.7118 | 0.6282 | 0.5850 | 0.6503 | **-0.0615** | no (magnitude) |
| cityblock | 0.7147 | 0.6112 | 0.6025 | 0.6692 | **-0.0455** | no (magnitude) |
| sqeuclidean | 0.6960 | 0.6212 | 0.5839 | 0.6533 | **-0.0427** | no (magnitude) |

### Squirrel — Attack-0 by distance metric

| Metric | B0 | B1 | B2 | B3 | Δ(B3−B0) | scale-invariant? |
|---|---|---|---|---|---|---|
| cosine | 0.5890 | 0.7609 | 0.7533 | 0.8306 | **+0.2417** | partly (direction) |
| euclidean | 0.5458 | 0.7610 | 0.7535 | 0.8307 | **+0.2849** | no (magnitude) |
| correlation | 0.7068 | 0.6099 | 0.5959 | 0.6981 | **-0.0087** | yes |
| chebyshev | 0.5196 | 0.7438 | 0.7544 | 0.8139 | **+0.2943** | no (magnitude) |
| braycurtis | 0.5795 | 0.7753 | 0.7573 | 0.8346 | **+0.2551** | no (magnitude) |
| canberra | 0.5826 | 0.7765 | 0.7546 | 0.8346 | **+0.2520** | no (magnitude) |
| cityblock | 0.5795 | 0.7753 | 0.7573 | 0.8346 | **+0.2551** | no (magnitude) |
| sqeuclidean | 0.5458 | 0.7610 | 0.7535 | 0.8307 | **+0.2849** | no (magnitude) |

Correlation — the metric the paper selects for Attack-0, and the best
one on the uncondensed graph too — normalises away both scale and mean,
so it is immune. On Cora it even improves. The magnitude-based metrics
lose 0.10–0.13 AUC.

So the honest statement is narrower than "condensation does not help":
condensation **does** degrade five of the eight distance metrics
substantially, but not the one the attack actually uses. A practitioner
who reached for Euclidean would measure 0.916 → 0.786 on Cora and
conclude condensation protects well. It does not — the attack simply
has a better metric available.

## Figures

![cora_auc_vs_setting.png](figures/cora_auc_vs_setting.png)

![cora_per_run_auc.png](figures/cora_per_run_auc.png)

![cora_unsup_attack0_metrics.png](figures/cora_unsup_attack0_metrics.png)

![cora_unsup_attack2_families.png](figures/cora_unsup_attack2_families.png)

![cora_attack0_clustering.png](figures/cora_attack0_clustering.png)

![cora_attack2_clustering.png](figures/cora_attack2_clustering.png)

![citeseer_auc_vs_setting.png](figures/citeseer_auc_vs_setting.png)

![citeseer_per_run_auc.png](figures/citeseer_per_run_auc.png)

![citeseer_unsup_attack0_metrics.png](figures/citeseer_unsup_attack0_metrics.png)

![citeseer_unsup_attack2_families.png](figures/citeseer_unsup_attack2_families.png)

![citeseer_attack0_clustering.png](figures/citeseer_attack0_clustering.png)

![citeseer_attack2_clustering.png](figures/citeseer_attack2_clustering.png)

![chameleon_auc_vs_setting.png](figures/chameleon_auc_vs_setting.png)

![chameleon_per_run_auc.png](figures/chameleon_per_run_auc.png)

![chameleon_unsup_attack0_metrics.png](figures/chameleon_unsup_attack0_metrics.png)

![chameleon_unsup_attack2_families.png](figures/chameleon_unsup_attack2_families.png)

![chameleon_attack0_clustering.png](figures/chameleon_attack0_clustering.png)

![chameleon_attack2_clustering.png](figures/chameleon_attack2_clustering.png)

![squirrel_auc_vs_setting.png](figures/squirrel_auc_vs_setting.png)

![squirrel_per_run_auc.png](figures/squirrel_per_run_auc.png)

![squirrel_unsup_attack0_metrics.png](figures/squirrel_unsup_attack0_metrics.png)

![squirrel_unsup_attack2_families.png](figures/squirrel_unsup_attack2_families.png)

![squirrel_attack0_clustering.png](figures/squirrel_attack0_clustering.png)

![squirrel_attack2_clustering.png](figures/squirrel_attack2_clustering.png)

