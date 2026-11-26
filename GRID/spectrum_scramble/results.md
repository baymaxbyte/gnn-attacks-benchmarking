# Spectrum-keep edge-scramble (todo #2) — Results log

## Part 1 — Headroom diagnostic (`spectral_headroom.py`)

**Question:** can the edge set change a lot while the GCN-relevant spectrum
(top-k eigenspace of the propagation operator `Ahat = D̃^{-1/2}(A+I)D̃^{-1/2}`)
stays fixed? Run on the largest connected component (full graphs are
disconnected: cora 78 comps, citeseer 438).

**1. Per-edge spectral cost** (first-order: flipping `(i,j)` shifts `λ_m` by
`≈(u_m[i]−u_m[j])²`):

| | real edges (median / mean) | non-edges (median / mean) | real vs random |
|---|---|---|---|
| cora (LCC n=2485, \|E\|=5069) | 0.0003 / 0.0020 | 0.0193 / 0.0530 | **26× cheaper** |
| citeseer (LCC n=2120, \|E\|=3679) | 0.0001 / 0.0011 | 0.0337 / 0.0618 | **55× cheaper** |

Within-community edges (what homophily / link-stealing rely on) are the cheap
ones — the intuition behind a decoy graph.

**2. Utility headroom** — top-64 `Ahat` eigenspace kept (∈[0,1], 1=identical)
after scrambling r·|E| edges (remove cheapest real + add cheapest decoy), spectral
vs random pick:

| edge-set divergence | cora spectral / random | citeseer spectral / random |
|---|---|---|
| 18% | 0.80 / 0.56 | 0.63 / 0.50 |
| 46% | 0.80 / 0.21 | 0.52 / 0.20 |
| 67% | 0.69 / 0.11 | 0.66 / 0.09 |
| 82% | 0.51 / 0.05 | 0.37 / 0.05 |

**Read:** structured headroom exists — spectral-aware scrambling preserves the
GCN subspace far better than random — but it is **not free** (cora keeps ~0.80,
citeseer only ~0.5 at 46% edge change), so expect a small (cora) / larger
(citeseer) accuracy cost. `eigsh` mildly noisy on citeseer; overlap is a proxy.

Figure: `results/spectral_headroom.png`. Data: `results/spectral_headroom.json`.

## Part 2 — Decoy-graph defense: accuracy (a) + link-stealing AUC (b)

`decoy_defense.py`: train a 2-layer GCN on the scrambled/decoy graph; (a) node
test accuracy; (b) link-stealing AUC with positives = the REAL edges (does the
attacker still recover real links from a decoy-trained model?). Shadow = other
dataset's real-trained box. 5 runs. Data: `results/attacks/decoy_defense_auc.csv`.

| dataset | config | node acc | Attack-0 | Attack-1 | Attack-6 |
|---|---|---|---|---|---|
| cora | real | 85.9% | 0.921 | 0.930 | 0.955 |
| cora | spec r=0.3 | 73.0% (−13) | 0.886 (−0.04) | 0.897 (−0.03) | 0.942 (−0.01) |
| cora | spec r=0.5 | 62.8% (−23) | 0.842 (−0.08) | 0.852 (−0.08) | 0.929 (−0.03) |
| cora | rand r=0.5 | 61.0% | 0.829 (−0.09) | 0.836 (−0.09) | 0.926 (−0.03) |
| citeseer | real | 78.1% | 0.945 | 0.951 | 0.974 |
| citeseer | spec r=0.3 | 71.0% (−7) | 0.924 (−0.02) | 0.919 (−0.03) | 0.962 (−0.01) |
| citeseer | spec r=0.5 | 63.4% (−15) | 0.883 (−0.06) | 0.863 (−0.09) | 0.957 (−0.02) |
| citeseer | rand r=0.5 | 62.3% | 0.873 (−0.07) | 0.863 (−0.09) | 0.951 (−0.02) |

**Verdict: a bad tradeoff — #2 refuted as a practical defense.** Small AUC drops
(Attack-1 only −0.03 to −0.09) at large accuracy cost (−7 to −23 points). GRID /
Fisher–Rao achieve the same-or-larger Attack-1 drop at ZERO accuracy cost. And
`spec r=0.5` ≈ `rand r=0.5` on both AUC and accuracy, so the spectral headroom
barely translated into a better tradeoff in the actual pipeline.

**Why (the core insight):** in a homophilous graph, link privacy and node
accuracy ride on the SAME community structure. The attack exploits within-
community posterior similarity (homophily); the GCN's accuracy comes from the
same within-community smoothing. Removing a real edge doesn't make its endpoints
dissimilar (they stay connected via other within-community paths), so the
attacker still catches them — hence the AUC barely moves even at 50% edge change.
Any rewiring aggressive enough to decorrelate posteriors from real edges also
destroys the smoothing the GCN needs. "Keeping the spectrum" keeps the community
structure, which keeps BOTH accuracy AND the leak.

## Overall (both tested directions hit a wall)

- #6 Fisher–Rao (posterior perturbation): capped by argmax preservation vs the
  attack's random negatives (can't push same-class linked pairs to look like
  random unrelated pairs without changing labels).
- #2 spectrum-keep edge-scramble (decoy graph): capped by the utility↔privacy
  coupling through community structure (can't hide links without hurting the
  node classifier that relies on the same structure).

Both point to the same conclusion: on homophilous graphs, link-privacy and GNN
utility are fundamentally entangled. A defense that truly breaks the link signal
at low utility cost likely must change the TASK/representation (e.g. a learned
privacy-funnel/adversarial condensation that is explicitly trained to keep label
info while destroying adjacency info, #1/#7), or accept a bounded accuracy loss.
