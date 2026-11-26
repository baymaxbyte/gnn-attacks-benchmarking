# Privacy-funnel (adversarial representation) — todo #1, v1 prototype

**Goal:** learn a representation that keeps label info `I(out;Y)` while an
adversarial edge-discriminator fails to recover edges `min I(out;A)` — targeting
the WITHIN-CLASS edge-vs-non-edge distinction (the "extra" info beyond class,
since against random/cross-class negatives link-stealing = class detection = the
task). Gated GCN encoder (learnable graph gates g1,g2 so it can trade toward a
graph-free MLP) + classifier + adversary; sweep privacy weight λ. Metric: node
test accuracy vs within-class link-leakage AUC from a held-out trained attacker
(0.5 = private, 1 = leaks). `funnel.py` → `results/funnel_tradeoff.png`.

## Results (v1)

| dataset | λ | node acc | within-class leak AUC | rand-neg AUC | graph gates |
|---|---|---|---|---|---|
| cora | 0 | 86.5% | 0.801 | 0.941 | (0.81, 0.89) |
| cora | 1 | 87.3% | 0.803 | 0.940 | (0.82, 0.90) |
| cora | 5 | 73.0% | 0.866 | 0.882 | (0.84, 0.93) |
| cora | 20 | 73.0% | 0.912 | 0.838 | (0.96, 0.96) |
| citeseer | 0 | 76.9% | 0.833 | 0.946 | (0.23, 0.55) |
| citeseer | 1 | 78.8% | 0.843 | 0.944 | (0.23, 0.58) |
| citeseer | 5 | 47.9% | 0.917 | 0.795 | (0.50, 0.85) |
| citeseer | 20 | 24.5% | 0.705 | 0.752 | (0.74, 0.90) |

## Verdict: v1 FAILED (training-stability, not a clean impossibility)

The naive adversarial funnel did **not** produce a usable tradeoff:
- As λ grows, **node accuracy collapses** (cora 86→73; citeseer 77→48→24) while
  the **within-class leak AUC does not fall — it rises** (cora 0.80→0.91). The
  opposite of the goal on both axes.
- The **graph-gate lever moved the wrong way** (rose toward more graph, e.g.
  cora 0.81→0.96), instead of down toward a private graph-free MLP.
- `rand-neg AUC` fell with λ — meaning the pressure degraded *class* structure
  (hence the accuracy loss), not the edge signal.

This is an **unstable minimax**, not evidence the funnel is impossible: the
encoder overfits to fooling the *in-loop* adversary (resampled each epoch) in a
way a *fresh* held-out attacker exploits more; the `(σ(D)−0.5)²` confusion term
plus alternating updates don't converge; and a C-dim posterior bottleneck leaves
little room. Unlike #2/#6 (clean refutations), #1 is "prototype needs a better
design."

## v2 plan (if continued)

1. **Gradient-reversal layer** (DANN-style) instead of the hand-rolled confusion
   term — standard and far more stable.
2. **Operate on the hidden embedding** `h` (HID-dim), not the C-dim posteriors —
   more capacity to drop edge info while a separate linear head keeps the label.
3. **Stronger, periodically-reset in-loop adversary** + evaluate against a
   held-out attacker each epoch to stop overfitting the adversary.
4. **Anneal λ** from 0, and balance D/encoder step counts.
5. Re-measure accuracy vs within-class leakage **and** run the real LSA harness
   (Attack-0/1/6 under random + near3) on the funnel posteriors for comparability
   with GRID / Fisher-Rao.

## Where #1 sits vs the others

- #6 Fisher-Rao: capped by argmax vs random negatives.
- #2 edge-scramble: capped by utility↔privacy coupling (small AUC drop, big acc cost).
- #1 funnel: v1 unstable; a stabilized v2 is the one remaining idea that could, in
  principle, remove the within-class edge signal while keeping class info — but
  v1 gives no evidence yet that it beats the coupling.

---

## v2 (DANN gradient-reversal) — WORKS: first favorable tradeoff

`funnel_v2.py`: single-optimizer GRL (stable), λ annealed 0→λ_max, gated encoder,
held-out attacker for the leak metric. → `results/funnel_v2_tradeoff.png`.

| dataset | λ_max | node acc | within-class leak AUC | rand-neg AUC | graph gates |
|---|---|---|---|---|---|
| cora | 0 | 87.5% | 0.791 | 0.941 | (0.79, 0.88) |
| cora | 1 | 86.9% | 0.747 | 0.934 | (0.45, 0.63) |
| cora | 3 | 82.1% | 0.643 | 0.905 | (0.15, 0.27) |
| cora | 10 | 78.7% | 0.590 | 0.880 | (0.14, 0.16) |
| citeseer | 0 | 76.4% | 0.811 | 0.945 | (0.15, 0.48) |
| citeseer | 1 | 75.9% | 0.702 | 0.923 | (0.10, 0.17) |
| citeseer | 3 | 75.9% | 0.667 | 0.896 | (0.10, 0.18) |
| citeseer | 10 | 74.8% | 0.684 | 0.895 | (0.13, 0.18) |

**Result: a real, monotonic privacy–utility tradeoff — the first of the three
directions to achieve one.**
- citeseer: within-class leak 0.811 → **0.667 at ~zero accuracy cost** (76.4→75.9).
- cora: leak 0.791 → 0.643 at −5 pts, → 0.590 at −9 pts.
- Mechanism: the GRL drives the graph gates DOWN (cora 0.79→0.14), so neighbours
  are no longer over-smoothed (kills the within-class edge signal), while node
  features still carry the class label (accuracy largely held). This is exactly
  the lever #2's edge-scramble lacked.

**Caveats / honest scope:**
- Leak falls toward but does not reach 0.5 — residual within-class leakage remains.
- `rand-neg AUC` stays high (0.88–0.94): against random (cross-class) negatives
  the attack is just class detection = the task, which is irreducible. The funnel
  only removes the *extra* within-class edge signal.
- The leak metric here is a strong held-out within-class MLP attacker, NOT the
  LSA harness — so it is NOT yet on the same scale as GRID/Fisher-Rao's near3
  numbers. Next step: run the real LSA harness (Attack-0/1/6, random + near3) on
  the funnel posteriors to compare apples-to-apples.

**Bottom line:** #1 (v2) is the only direction that produced a favorable
tradeoff on the reducible (within-class) leakage, by learning to down-weight the
graph rather than perturbing a fixed one. It does not beat the irreducible
class-detection leak (random negatives), consistent with the project's core
finding, but it meaningfully reduces the within-class signal at low utility cost
— worth developing further (and benchmarking on the LSA harness).

---

## v2 on the real LSA harness — reconciliation (the funnel does NOT beat GRID/FR)

`funnel_lsa_eval.py`: ran the He et al. attacks (Attack-0/1/6, 5 runs) under random
AND near3 negatives on the funnel's served posteriors, baseline (λ=0) vs funnel
(λ=3). Same axis as GRID / Fisher-Rao. Data: `results/lsa_eval/funnel_lsa.csv`.

Funnel baseline→funnel AUC (node acc cost: cora −5.4 pts, citeseer ~0):

| eval | dataset | Attack-0 | Attack-1 | Attack-6 |
|---|---|---|---|---|
| random | cora | 0.925→0.873 (−0.053) | 0.932→0.876 (−0.056) | 0.954→0.922 (−0.032) |
| random | citeseer | 0.914→0.854 (−0.061) | 0.927→0.856 (−0.071) | 0.965→0.937 (−0.027) |
| near3 | cora | 0.720→0.664 (−0.056) | 0.745→0.668 (−0.077) | 0.811→0.769 (−0.042) |
| near3 | citeseer | 0.660→0.577 (−0.083) | 0.701→0.585 (−0.116) | 0.824→0.777 (−0.047) |

**Cross-method, Attack-1 DEFENDED AUC (lower = more private; same LSA harness):**

| eval | method | cora | citeseer | accuracy cost |
|---|---|---|---|---|
| random | GRID (θ=0.4) | 0.870 | 0.886 | 0 |
| random | Fisher-Rao | **0.770** | **0.819** | 0 |
| random | funnel (λ=3) | 0.876 | 0.856 | −5pts / ~0 |
| near3 | GRID | 0.605 | 0.575 | 0 |
| near3 | Fisher-Rao | **0.526** | **0.524** | 0 |
| near3 | funnel | 0.668 | 0.585 | −5pts / ~0 |

**Verdict:** the funnel's apparent win was metric-specific (a strong within-class
MLP). On the actual He et al. attacks it is the WEAKEST of the three — smaller
AUC reductions than GRID and far smaller than Fisher-Rao — AND it costs accuracy,
whereas GRID/Fisher-Rao are exactly zero-utility. (Caveat: funnel is on the LCC
with a gated GCN, GRID/FR on the whole soft GCN, so baselines differ slightly;
but the DEFENDED AUC — the actual privacy achieved — still favours FR clearly.)

## Final takeaway (all four experiments)

On homophilous citation graphs, the best practical link-stealing defence we found
is the simple **zero-utility posterior perturbation (Fisher-Rao, then GRID)** — it
reduces the hard (near-hop) leakage the most at no accuracy cost. The structural
(edge-scramble) and learned (privacy-funnel) approaches did NOT beat it:
edge-scramble trades big accuracy for tiny AUC gains; the funnel underperforms on
the real attacks and costs accuracy. And NONE escape the irreducible floor:
against random (cross-class) negatives, link-stealing = class detection = the
task, so Attack-1 can't be pushed below ~0.77 without destroying utility. Link
privacy and GNN utility are fundamentally coupled on these graphs.
