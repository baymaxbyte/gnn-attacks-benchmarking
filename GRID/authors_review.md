# GRID — Authors' Review & Our Verification Findings

A record of our correspondence with the GRID authors (Jiadong Lou et al., IEEE
S&P 2025, arXiv [2501.10985](https://arxiv.org/abs/2501.10985)) about our
reproduction, each of their clarifications, and what we found when we tested
them. Date: 2026-11-07.

---

## 1. Context

We reimplemented GRID from the paper (Algorithm 1 core-node selection, the
OPT-GRID objective, Algorithm 3 projection; sim = Pearson-correlation + cosine)
and benchmarked it with the He et al. link-stealing attack. We reproduce the
**no-defense baselines** closely, but our **GRID-defended AUCs come out much
higher (weaker defense)** than the paper's Tables 2–3. We emailed the authors
for the code or clarifications.

**Our questions to the authors:**
1. Could you share the GRID implementation or the Attack evaluation script?
2. Is the ℓ1 budget θ a hard cap, and do you perturb both endpoints of a core edge or only the core node?
3. How are the evaluation node pairs sampled (D4), and is AUC computed per-attack over them?
4. Target GCN details (layers / hidden units / epochs / early stopping)?

**What we had already checked before emailing:**
- **Utility:** every argmax preserved → box accuracy unchanged (cora 88.7%, citeseer 75.0%) — consistent with the zero-utility guarantee.
- **Solver:** reaches within ~6% of the analytic optimum; more iterations don't help.
- **Budget sweep:** even at θ=1.0 the drops stay modest (cora Attack-1 → 0.85, citeseer → 0.84).
- **Posterior confidence:** matching the paper's adjacent cosine (~0.94) with a soft 2-layer GCN helps only partially; a paper-faithful D1 target (40% node subsample, 20% labels removed) did not close the gap.

---

## 2. Authors' reply (verbatim) — Jiadong Lou

> Hi Chitraansh,
>
> Thank you for your interest in our GRID paper and for sharing your detailed
> reproduction results. We appreciate the effort you have put into this work.
>
> Unfortunately, we are not able to share our implementation or evaluation
> scripts at this time. However, we would like to highlight a few points that
> may help with your reproduction.
>
> First, the distortion budget θ is an upper bound on the L1 norm of the noise
> vector, rather than a required distortion level. GRID applies perturbations to
> the selected core nodes, with one noise vector per selected node, rather than
> independently perturbing both endpoints of each edge.
>
> Second, we suggest checking the core-node selection and the construction of
> n-hop indirect neighbors. In particular, the similarity threshold δ and the
> similarity-weighted node degrees in Algorithm 1 are important. It may also be
> helpful to compare the similarity distributions of adjacent and n-hop indirect
> neighboring nodes before and after applying GRID, rather than relying only on
> the average cosine similarity.
>
> Third, as described in Section 5.1.3, the evaluation dataset D4 is sampled from
> D1 using linked and unlinked node pairs. For Attack-1 and Attack-6, we follow
> the evaluation settings of He et al. Differences in data sampling and attack
> evaluation protocols may also affect the results.
>
> We hope these suggestions are helpful, and thank you again for your interest in
> our work.
>
> Best regards,
>
> Jiadong Lou

---

## 3. Each hint mapped to our implementation / findings

| # | Author hint | Our status |
|---|---|---|
| 1 | θ is an **upper bound** on ‖noise‖₁; **one noise vector per core node**, not both endpoints of an edge. | **Already matched exactly.** `_finalise`/`constraint_check` hard-cap ‖s‖₁ ≤ θ; we perturb only vertex-cover core nodes, one vector each. → The mechanism is structurally faithful; the gap is not here. |
| 2 | Check δ and **similarity-weighted node degrees** in Alg. 1; compare the **full similarity distributions** of adjacent vs n-hop pairs, not just the average cosine. | **Partially done; now completed.** We implement the δ-threshold + sim-weighted-degree cover. We had only logged *means*; the distribution check (below) confirmed the defense is weak on near-one-hot posteriors and revealed a **floor mismatch**. |
| 3 | D4 is **sampled from D1** (linked + unlinked pairs); Attack-1/6 follow He et al.; data-sampling / eval-protocol differences matter. | **Tested; see below.** The literal D1 restriction does **not** reproduce the big drops. The real lever is the **negative-pair similarity distribution**. Attack-6 is unreproducible in any configuration. |

---

## 4. Findings

### 4.1 The mechanism is faithful
The authors' hint #1 describes exactly our implementation (θ as a hard cap; one
noise vector per core node). Combined with our earlier checks (near-optimal
solver; zero label change; θ-sweep to 1.0), this rules out the noise-crafting
mechanism as the source of the gap.

### 4.2 Similarity distributions before/after GRID (hint #2)
Self-contained diagnostic on the served posteriors (`diag_sim_dist.py`), cosine
similarity:

| | adjacent (before → after) | 3-hop (floor) | random-unlinked |
|---|---|---|---|
| cora | mean 0.918 → 0.898, **median 1.000 → 0.970** | 0.661 | 0.208 |
| citeseer | mean 0.977 → 0.940, **median 0.998 → 0.959** | 0.876 | 0.340 |

- The **mean hid a near-one-hot spike** — most adjacent pairs sit at cosine ≈ 1, where there is no room to move. GRID barely shifts the distribution. Exactly why the authors said not to trust the average.
- **Similarity-only AUC** (ceiling for a similarity attack), before → after GRID:
  - linked vs **random-unlinked** (the attack's negatives): cora 0.918 → 0.896, citeseer 0.948 → 0.916 — barely drops, and matches our measured attack AUCs.
  - linked vs **3-hop**: cora 0.691 → 0.642, citeseer 0.700 → 0.565 — drops a lot.
- **Floor mismatch:** GRID disguises adjacent pairs down to the 3-hop level, but 3-hop similarity (0.66 / 0.88) is far above random-unlinked (0.21 / 0.34). So even a perfect push to the floor leaves linked pairs easily separable from the attack's random negatives.

Figures: `results/diag/sim_dist_cora_whole.png`, `results/diag/sim_dist_citeseer_whole.png`.

### 4.3 D4 / negative-sampling experiment (hint #3)
We ran the real He et al. attacks on the paper-faithful **soft GCN** (adjacent
cosine ≈ 0.93–0.97), before/after GRID (θ=0.4, n=3), under three evaluation
negative-sampling schemes, base and GRID sharing identical pairs
(`paper_faithful_gcn/step3_negmodes.py`):

- `random` — uniform unlinked pairs (literal He et al.; our setup).
- `d1` — pairs restricted to a 40% node subgraph (the paper's literal D4, Sec 5.1.3).
- `near3` — negatives = unlinked pairs at graph distance exactly 3 (GRID's disguise floor).

**Attack-1 AUC, no-defense → GRID:**

| mode | cora | citeseer |
|---|---|---|
| `random` | 0.949 → 0.878 (−0.071) | 0.970 → 0.891 (−0.080) |
| `d1` (paper D4) | 0.945 → 0.915 (**−0.030**) | 0.964 → 0.933 (**−0.031**) |
| `near3` (floor) | 0.787 → 0.613 (−0.174) | 0.818 → 0.593 (−0.225) |
| **paper (Table 3)** | 0.942 → 0.728 | 0.965 → 0.732 |

Figure: `results/d4/negmode_attack1.png`. Data: `results/d4/negmode_results.csv`.

**Key points:**
- The literal **D4 restriction (`d1`) does NOT reproduce** the big drops — if anything GRID helps *less* (Δ ≈ −0.03), because the baseline stays ~0.95 and GRID can't move near-vertex posteriors far.
- `near3` matches the paper's *delta* (−0.17 / −0.23 vs −0.21 / −0.23) but **`near3` lowers the baseline by itself**: cora Attack-1 random-base 0.949 → near3-base 0.787 is **−0.162 from the hard negatives alone (no defense)**, then −0.174 more from GRID. So `near3`'s low AUC is ~half negative-artifact, ~half GRID.
- **Mutual exclusivity:** a high no-defense baseline (~0.95) and a large GRID drop are mutually exclusive in our reproduction. GRID's pure (fixed-negative) delta is large only when the negatives sit at the n-hop floor, which itself depresses the baseline. The paper reports both at once, which our faithful implementation cannot produce at zero utility loss.

### 4.4 All eight attacks under `near3` (no-defense → GRID AUC)
`paper_faithful_gcn/near3_all.py` → `results/d4/near3_all_attacks.csv`.

| Attack | knowledge (paper Table 1) | cora | citeseer |
|---|---|---|---|
| Attack-0 | posteriors only (unsup.) | 0.753 → 0.667 (−0.087) | 0.779 → 0.585 (−0.194) |
| Attack-1 | + shadow | 0.789 → 0.602 (−0.187) | 0.805 → 0.582 (−0.224) |
| Attack-2 | + node attributes (unsup.) | 0.753 → 0.667 (−0.087) | 0.779 → 0.585 (−0.194) |
| Attack-3 | + partial graph | 0.789 → 0.785 (**−0.004**) | 0.811 → 0.783 (**−0.028**) |
| Attack-4 | partial graph + shadow | 0.796 → 0.687 (−0.110) | 0.818 → 0.682 (−0.137) |
| Attack-5 | attributes + shadow | 0.812 → 0.706 (−0.105) | 0.837 → 0.692 (−0.145) |
| Attack-6 | attributes + partial graph | 0.818 → 0.825 (**+0.007**) | 0.848 → 0.843 (**−0.006**) |
| Attack-7 | attributes + partial + shadow | 0.822 → 0.778 (−0.044) | 0.844 → 0.779 (−0.065) |
| link-pred | structure only | 0.582 → 0.582 (0) | 0.522 → 0.522 (0) |

Drops track **what the attack uses**: posterior-similarity attacks (0, 1, 2)
collapse; feature/structure attacks (3, 6, 7) barely move. GRID perturbs only
posteriors, so attacks leaning on node attributes or known edges are out of its
reach. (Attack-0 and Attack-2 are the unsupervised pair; our harness scores them
identically.)

### 4.5 Paper's reported numbers (Cora/Citeseer, GCN, no-defense → GRID AUC)
The paper tabulates per-attack AUC only for the three representative attacks:

| Attack | cora | citeseer |
|---|---|---|
| Attack-0 | 0.930 → 0.685 (−0.245) | 0.942 → 0.691 (−0.251) |
| Attack-1 | 0.942 → 0.728 (−0.214) | 0.965 → 0.732 (−0.233) |
| Attack-6 | 0.964 → 0.704 (−0.260) | 0.981 → 0.711 (−0.270) |

Attacks 2/3/4/5/7 have **no** Cora/Citeseer AUC in the paper (only a Pubmed
accuracy-under-GRID ablation in Table 7, a Figure-5 ablation, and the blanket
claim of "~25% AUC degradation" on the citation datasets). ILS attacks A2/A9 are
also reported dropping (Table 6). Model accuracy is unchanged (GCN: cora 84.8%,
citeseer 73.3%).

---

## 5. Conclusions

1. **The mechanism is faithful.** Confirmed by the authors' hint #1 plus our
   solver/θ/zero-utility checks. The gap is not in how GRID crafts or caps noise.
2. **On near-one-hot posteriors GRID has almost no room.** Median adjacent cosine
   moves only 1.00 → 0.97 (cora) / 0.998 → 0.959 (citeseer); even θ=1.0 is modest.
3. **The gap is the negative-pair distribution, not the D1 restriction.** The
   paper's literal D4 (`d1`, random-unlinked within a 40% subgraph) gives only
   Δ ≈ −0.03. Large drops appear only when evaluation negatives sit at the n-hop
   floor (`near3`) — but that also depresses the baseline. A high baseline and a
   big GRID drop cannot co-occur in our reproduction at zero utility.
4. **By attack type:** posterior-similarity attacks (0/1/2) collapse under
   GRID+floor-negatives; feature/structure attacks (3/6/7) barely move.
5. **Sharpest contradiction — Attack-6.** The paper's *largest* reported drop is
   Attack-6 (−0.26), and a uniform ~25% across all attacks. We find Attack-6 (and
   Attack-3) essentially **undefended** (Δ ≈ 0). By the paper's own Table 1,
   Attack-6 uses node attributes (F) and the partial graph (A*) — neither of which
   GRID modifies — so how a posterior-only perturbation drives it 0.96 → 0.70
   remains unexplained.

---

## 6. Open questions for the authors

1. **D4 negatives:** are the "unlinked" pairs in D4 drawn uniformly at random, or
   restricted to near/indirect (e.g., 2–3-hop) pairs? Our results only match the
   paper's magnitude under near-hop negatives.
2. **Attack-6 (and the other feature/structure attacks):** how does a
   posterior-only defense reduce an attack that, per Table 1, relies on node
   attributes and the partial graph? What features does your Attack-6 classifier
   actually use?
3. **δ / core selection:** the exact definition of the n-hop similarity floor δ
   used as the optimization target (vs. the Alg. 1 selection threshold), and the
   resulting core-set size (% of nodes) on Cora/Citeseer.

---

## 7. Evidence in this folder

- `diag_sim_dist.py` → `results/diag/sim_dist_{cora,citeseer}_whole.png` — similarity distributions (§4.2).
- `paper_faithful_gcn/step3_negmodes.py` → `results/d4/negmode_results.csv`, `results/d4/negmode_attack1.png` — negative-sampling experiment (§4.3).
- `paper_faithful_gcn/near3_all.py` → `results/d4/near3_all_attacks.csv` — all-8-attacks under near3 (§4.4).
- `grid_defense.py` — the GRID reimplementation under review.
- `results/grid_vs_baseline_theta0.4.csv`, `results/theta06.csv`, `results/theta10.csv`, `paper_faithful_gcn/results/soft_gcn_grid.csv`, `paper_faithful_d1/results/d1_grid.csv` — earlier reproduction runs.
