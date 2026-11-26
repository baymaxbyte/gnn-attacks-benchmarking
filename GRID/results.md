# GRID reproduction + Fisher–Rao defense — Results

Scope: our reproduction of GRID (Lou et al., S&P 2025, arXiv 2501.10985) and a
Fisher–Rao geodesic variant we built, evaluated with the He et al. link-stealing
attacks on the paper-faithful soft 2-layer GCN (cora/citeseer). Companion docs:
`authors_review.md` (correspondence + gap analysis), `fisher_rao_diag/README.md`
(the room diagnostic).

---

## 1. Headline — AUC change on the 3 paper attacks

Soft GCN target, random-unlinked negatives (He et al. protocol), 5 runs,
whole-graph shadow. AUC (lower = better defense); Δ vs no-defense.

| dataset | attack | no-defense | GRID (θ=0.4) | **Fisher–Rao** | paper (GRID) |
|---|---|---|---|---|---|
| cora | Attack-0 | 0.941 | 0.905 (−0.036) | **0.793 (−0.148)** | 0.930 → 0.685 |
| cora | Attack-1 | 0.949 | 0.870 (−0.080) | **0.770 (−0.180)** | 0.942 → 0.728 |
| cora | Attack-6 | 0.956 | 0.952 (−0.004) | **0.921 (−0.035)** | 0.964 → 0.704 |
| citeseer | Attack-0 | 0.966 | 0.906 (−0.060) | **0.841 (−0.124)** | 0.942 → 0.691 |
| citeseer | Attack-1 | 0.972 | 0.886 (−0.086) | **0.819 (−0.154)** | 0.965 → 0.732 |
| citeseer | Attack-6 | 0.979 | 0.973 (−0.006) | **0.963 (−0.016)** | 0.981 → 0.711 |

`link_pred` baseline unchanged (0.633 cora / 0.613 citeseer); **label_changes = 0**
for both defenses → node/box accuracy preserved exactly.

**Read:** the Fisher–Rao defense roughly **doubles** GRID's drop on the
posterior-similarity attacks (Attack-0/1) and moves Attack-6 a little more too,
all at zero label change — closing much of the gap to the paper on Attack-0/1,
though still short, and still far short on Attack-6.

Data: `paper_faithful_gcn/results/fr_defense/fr_defense_auc.csv`.

### 1b. Same defenses under a HARD-negative (near3) evaluation

The table above uses **random** unlinked negatives (the attacker's realistic,
AUC-maximising choice). For completeness, the same soft target and defenses under
**near3** negatives (unlinked pairs at graph distance 3 — a defense-favourable
eval that pre-weakens the attack). Same pairs for all three targets.

| dataset | attack | no-defense | GRID | **Fisher–Rao** |
|---|---|---|---|---|
| cora | Attack-0 | 0.766 | 0.681 (−0.086) | **0.578 (−0.188)** |
| cora | Attack-1 | 0.800 | 0.605 (−0.195) | **0.526 (−0.274)** |
| cora | Attack-6 | 0.826 | 0.828 (+0.001) | **0.801 (−0.025)** |
| citeseer | Attack-0 | 0.778 | 0.581 (−0.197) | **0.553 (−0.225)** |
| citeseer | Attack-1 | 0.803 | 0.575 (−0.228) | **0.524 (−0.279)** |
| citeseer | Attack-6 | 0.847 | 0.844 (−0.002) | **0.814 (−0.032)** |

**Fisher–Rao beats GRID under BOTH evaluations** (not just random): the earlier
near3-GRID drop (~−0.20) looked large only because near3 is a harder eval with a
low baseline — Fisher–Rao under that same eval does better still, pushing
Attack-0/1 to ~0.52–0.58 (near chance). Under the realistic **random** eval,
Fisher–Rao caps Attack-1 at ~0.77–0.82 (the argmax limit), and Attack-6 stays
~undefended in both evals. Data: `results/fr_vs_grid_near3/fr_vs_grid_near3.csv`.

---

## 2. What the Fisher–Rao defense is

Same objective and threat model as GRID — perturb the served posteriors of the
vertex-cover **core nodes** to drag adjacent-node similarity (corr+cos) down to
the n-hop floor δ, preserving each argmax (zero label change). It differs only in
the **move geometry**: instead of GRID's additive, L1-budgeted step
(`‖s‖₁ ≤ θ`), it descends the hinge objective along a **geodesic on the √p
sphere** (the simplex's Fisher–Rao metric). Two consequences:
- The Riemannian gradient does not vanish at a vertex, so it does **not** stall
  on near-one-hot posteriors (plain mirror/natural-gradient descent freezes
  there — a bug we hit and fixed).
- There is **no θ hyperparameter**; the move runs along the geodesic until it
  reaches the floor δ or the argmax boundary.

---

## 3. Merits

- **Zero utility loss, same as GRID.** Every argmax preserved (label_changes = 0);
  `link_pred` and box accuracy unchanged. The privacy is bought without touching
  the task.
- **~2× stronger than GRID at its published setting** on Attack-0/1 (e.g. cora
  Attack-1 −0.18 vs −0.08; citeseer Attack-0 −0.12 vs −0.06), approaching the
  paper's magnitude on these two attacks.
- **No budget to tune.** GRID's defense strength hinges on θ (θ=0.4 badly
  under-shoots); the geodesic reaches the floor automatically.
- **Robust on confident posteriors.** It works on near-one-hot boxes where an
  additive/mirror step cannot move, because it travels on the sphere.
- **Distortion-efficient near the vertex** (from the room diagnostic): reaches a
  given similarity at a smaller move than L1 in the confident regime.

## 4. Demerits / limits

- **Not a fundamental improvement over a large-budget GRID.** The room diagnostic
  (`fisher_rao_diag/`) shows that, swept to full budget, an L1 move reaches
  *equal or lower* similarity than the geodesic (FR/L1 drop ratio 0.68–0.80×).
  So Fisher–Rao's win over GRID is really "reach the n-hop floor without a θ
  cap," which a large-θ GRID also achieves — it is a better operating point, not
  a new capability.
- **The argmax cap is unbroken.** Adjacent nodes are same-class, so a
  label-preserving move cannot make their posteriors as dissimilar as a random
  different-class pair. The lowest reachable adjacent similarity (~1.0 corr+cos)
  stays far above the random-unlinked level (~0.25–0.38). Hence Attack-1 only
  reaches ~0.77–0.82, never near chance (0.5).
- **Attack-6 barely moves** (−0.016 to −0.035). It is gray-box and uses node
  attributes + the partial graph (paper Table 1); a posterior-only defense
  cannot touch those features. The paper's largest reported drop is Attack-6
  (→~0.70), which we cannot reproduce with any posterior perturbation.
- **Larger confidence distortion than GRID θ=0.4.** Reaching the floor means a
  bigger move (higher KL), so the "minor perturbation" framing is weaker; the
  served confidences are noticeably changed even though labels are not.
- **Extra machinery.** Needs the √p-geodesic solver (naive mirror descent is
  wrong — it freezes at vertices).

---

## 5. How this reconciles with the "room" diagnostic

No contradiction. Two different questions:
- *Room diagnostic* (max similarity reduction at zero label change): Fisher–Rao
  gives **no more ultimate room** than a full-budget L1 move, and neither can
  reach the random-negative level — the limit is argmax preservation.
- *Defense experiment* (actual attack AUC at the methods' operating points):
  Fisher–Rao **beats GRID(θ=0.4)** because it pushes all the way to the floor
  budget-free, whereas GRID(θ=0.4) stops far short.

Together: Fisher–Rao is a **better-engineered member of the same capped family**
(GRID), not an escape from the cap.

---

## 6. Bottom line

The Fisher–Rao defense is the strongest zero-label posterior perturbation we have
— about double GRID's effect on the similarity attacks, no θ to tune, stable on
confident posteriors. But it inherits the family's two hard ceilings: it cannot
drive the similarity attacks to chance (argmax cap vs random negatives), and it
cannot defend the feature-based Attack-6 at all. To go further, the defense must
change **what is released** rather than perturb posteriors post-hoc — i.e.
privacy-funnel / adversarial **condensation** (#1, #7) or **spectrum-keep edge
scramble** (#2) from `todo.md` — or accept a bounded label change.

---

## 7. Evidence / files

- `paper_faithful_gcn/fr_defense.py` → `results/fr_defense/fr_defense_auc.csv` — the AUC table (§1).
- `fisher_rao_diag/fr_diag.py`, `fisher_rao_diag/README.md`, `results/fr_room_frontier.png` — the room diagnostic (§5).
- `paper_faithful_gcn/results/soft_gcn_grid.csv` — GRID on the same soft target (reference).
- `authors_review.md` — authors' reply and the full gap analysis (random vs d1 vs near3; the Attack-6 discrepancy).
