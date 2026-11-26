# todo #9 (rate-distortion quantization) + #12 (OT/GW class barycenter)

Both collapse each node's served posterior to a per-class representative, via
per-predicted-class k-means (M codewords/class): **M=1 = #12 barycenter** (full
collapse), **M=4 = #9 quantization** (higher rate = more fidelity = more leak).
Quantised the SAME soft-GCN whole-graph posteriors GRID/Fisher-Rao used; ran the
identical LSA harness (Attack-0/1/6, random + near3). `quant_defense.py` →
`quant_auc.csv`. **argmax preserved exactly (label_changes = 0) → node accuracy
unchanged (cora 80.9%, citeseer 71.7%).**

## Attack-1 AUC (no-defense → defended)

| eval | dataset | no-def | #12 barycenter (M=1) | #9 quant (M=4) |
|---|---|---|---|---|
| random | cora | 0.958 | 0.849 (−0.109) | 0.909 (−0.049) |
| random | citeseer | 0.972 | 0.846 (−0.126) | 0.926 (−0.046) |
| near3 | cora | 0.800 | 0.635 (−0.165) | 0.706 (−0.094) |
| near3 | citeseer | 0.803 | 0.613 (−0.190) | 0.680 (−0.123) |

(Attack-0 tracks Attack-1; Attack-6 moves least, −0.02 to −0.06.)

## Cross-method, Attack-1 DEFENDED AUC (lower = more private; all ~zero accuracy cost except funnel)

| | random cora/citeseer | near3 cora/citeseer |
|---|---|---|
| Fisher-Rao | **0.770 / 0.819** | **0.526 / 0.524** |
| GRID (θ=0.4) | 0.870 / 0.886 | 0.605 / 0.575 |
| **#12 barycenter** | 0.849 / 0.846 | 0.635 / 0.613 |
| funnel (λ=3) | 0.876 / 0.856 | 0.668 / 0.585 |
| #9 quant (M=4) | 0.909 / 0.926 | 0.706 / 0.680 |

## Verdict (corrects the earlier prediction)

I predicted the class-collapse would drive the within-class/near3 leak to ~0.5
(the ceiling). **Wrong:** near3 Attack-1 only falls to ~0.61–0.64 (M=1), not 0.5.
Reason: near3 negatives are distance-3 pairs that **cross class boundaries**, so
collapsing to class centroids leaves the irreducible same-class-vs-cross-class
(= class-detection) signal intact. And it does NOT beat Fisher-Rao/GRID — the
aggressive per-node perturbations still reach lower AUC.

Still, two useful points:
- **#12 barycenter is a dead-simple, truly zero-cost defense that is 2nd best on
  random** (beats GRID and the funnel; only FR is better) and competitive on
  near3 — i.e. "serve the class-mean posterior" is a strong baseline.
- **#9's rate knob behaves as expected:** more codewords (M=4) = more fidelity =
  more leak (weaker defense) than M=1.

Consistent with the whole study: Fisher-Rao (aggressive zero-utility posterior
perturbation) remains the best; none of the methods break the irreducible
class-detection floor.
