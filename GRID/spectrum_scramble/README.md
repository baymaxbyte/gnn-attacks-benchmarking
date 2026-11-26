# Spectrum-keep edge-scramble — headroom diagnostic (todo #2, first step)

**Idea.** A GCN propagates with `Ahat = D̃^{-1/2}(A+I)D̃^{-1/2}`; its outputs live in
the **top-k eigenspace** of `Ahat` (the smooth, low-frequency structure). If we
can change many edges while keeping that top-k eigenspace fixed, we can release a
spectrally-equivalent **decoy graph** whose edges are decorrelated from the true
ones → link privacy at (near) zero accuracy cost. This script measures whether
that headroom exists.

**Run.** `python spectral_headroom.py` → `results/spectral_headroom.png`,
`results/spectral_headroom.json`. On the largest connected component (the full
cora/citeseer graphs are disconnected — 78 / 438 components — which would swamp a
bottom-k Laplacian band). numpy/scipy only.

## What it computes

1. **Per-edge spectral cost** (first-order perturbation on `L=D−A`): flipping edge
   `(i,j)` shifts eigenvalue `λ_m` by `≈ (u_m[i]−u_m[j])²`, summed over the smooth
   band. Low = cheap.
2. **Utility headroom**: scramble = remove the cheapest `r·|E|` real edges + add
   the cheapest `r·|E|` decoy non-edges; measure the **captured energy of the
   GCN's top-64 `Ahat` eigenspace** (∈ [0,1], 1 = identical) vs a **random**
   scramble, and the edge-set divergence (Jaccard distance).

## Results

| | real-edge vs non-edge cost | top-64 eigenspace kept @ 46% edge divergence (spectral / random) |
|---|---|---|
| cora (LCC n=2485) | real edges **26×** cheaper | **0.80 / 0.21** |
| citeseer (LCC n=2120) | real edges **55×** cheaper | **0.52 / 0.20** |

Overlap vs edge divergence (spectral-aware pick):

| edge divergence | cora | citeseer |
|---|---|---|
| 18% | 0.80 | 0.63 |
| 46% | 0.80 | 0.52 |
| 67% | 0.69 | 0.66 |
| 82% | 0.51 | 0.37 |

(random pick: 0.56 → 0.21 → 0.11 → 0.05 on cora — collapses fast.)

## Reading

- **Structured headroom exists.** Within-community edges are 26–55× spectrally
  cheaper to flip than random pairs, and spectral-aware scrambling preserves the
  GCN subspace **far** better than random (cora: 0.80 vs 0.21 at 46% edge change).
  So the edge set and the GCN-relevant spectrum are genuinely separable — the
  premise of a decoy graph holds directionally.
- **But it is not free.** Even spectral-aware, the retained subspace is ~0.80
  (cora) / ~0.5 (citeseer) at moderate divergence, not ~1. Expect a real (small
  on cora, larger on the sparser citeseer) accuracy cost; the privacy–utility
  tradeoff is much better than random rewiring, not absent.

## Caveats / next step

- Eigenspace overlap is a **proxy** for accuracy; `eigsh` was mildly noisy on
  citeseer (non-monotone r), so treat exact numbers as approximate.
- **Decisive next step:** actually retrain the 2-layer GCN on the scrambled
  graphs and measure (a) node-classification accuracy and (b) the link-stealing
  AUC on a model trained on the decoy graph — to confirm accuracy holds while the
  attack recovers the decoy edges, not the real ones.
