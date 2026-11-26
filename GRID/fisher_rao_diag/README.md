# Fisher–Rao "room" diagnostic (todo.md direction A / #6)

**Question.** GRID budgets its noise in the Euclidean L1 ball and stalls on
confident posteriors. Does moving along the simplex's intrinsic **Fisher–Rao
geometry** (a geodesic on the `√p` sphere) give materially more room to cut
adjacent-node similarity, at zero label change (argmax preserved)?

**Run.** `python fr_diag.py` → `results/fr_room_frontier.png`, `results/fr_room_summary.json`.
Target = the paper-faithful **soft GCN** posteriors (non-degenerate, matching the
paper's confidence), with the saturated **KRON** box as a contrast. Both methods
minimise the same objective (mean corr+cos to neighbours) and preserve argmax;
they differ only in the move:
- **L1 / GRID** — additive projected gradient in `‖q−p‖₁ ≤ θ` (swept θ).
- **Fisher–Rao** — geodesic descent in `u = √q`; its Riemannian gradient does not
  vanish at a vertex (fixing the mirror-descent "freeze" that stalls at one-hot).

## Result — Fisher–Rao is **not** the fix

| soft GCN | sim0 (corr+cos) | n-hop floor δ | GRID θ=0.4 reaches | min reachable L1 | min reachable FR | random-unlinked level |
|---|---|---|---|---|---|---|
| cora | 1.863 | 1.294 | 1.693 (drop 0.17) | **1.029** | 1.197 | ≈ 0.25 |
| citeseer | 1.937 | 1.601 | 1.716 (drop 0.22) | **0.983** | 1.291 | ≈ 0.38 |

Three findings:

1. **The n-hop floor is reachable by both at zero label change** (L1 reaches
   1.03 < δ=1.29 on cora). GRID's weakness at θ=0.4 is just its small *budget*,
   not geometry — it stops at 1.69, well above what's reachable.
2. **Fisher–Rao gives no more ultimate room** — swept to full budget, L1 reaches
   *lower* similarity than the FR geodesic (FR/L1 drop ratio 0.68–0.80×); on
   confident nodes (top ≥ 0.8) they are ~equal. FR's only real win is escaping
   the vertex-freeze of mirror descent; it does not beat a fully-budgeted L1
   move. (The FR geodesic descent may be mildly under-tuned, so read this as
   "no material advantage," not "strictly worse.")
3. **The real wall is argmax preservation.** The *minimum* reachable adjacent
   similarity (~1.0 corr+cos) is still far above the random-unlinked level
   (~0.25–0.38). Adjacent nodes are same-class, so a label-preserving
   perturbation can never make their posteriors as dissimilar as a random
   different-class pair. No zero-label posterior perturbation — L1, Fisher–Rao,
   or otherwise — can disguise a true link as a random non-link.

## Implication

Direction #6 (and the whole post-hoc, zero-label posterior-perturbation family,
GRID included) is capped by the argmax constraint vs the attack's
random-unlinked negatives. The lever is **not** a smarter move geometry. Pivot to
directions that change *what is released* rather than perturb it post-hoc —
privacy-funnel / adversarial **condensation** (#1, #7), **spectrum-keep edge
scramble** (#2), or defenses that accept a bounded label change.
