#!/usr/bin/env python3
"""todo #2 (spectrum-keep edge-scramble) — FIRST-STEP diagnostic.

Idea: a GCN propagates with Ahat = D~^{-1/2}(A+I)D~^{-1/2}; its output lives in the
top-k eigenspace of Ahat (the smooth, low-frequency structure). If we can change
many edges while keeping that top-k eigenspace ~fixed, we can release a
spectrally-equivalent *decoy* graph whose edge set is decorrelated from the true
one -> link privacy at ~zero accuracy cost.

Two measurements:
  (1) per-edge spectral cost. First-order perturbation theory on the Laplacian
      L=D-A: flipping edge (i,j) shifts eigenvalue lambda_m by ~ (u_m[i]-u_m[j])^2.
      So WITHIN-community edges (endpoints alike on smooth eigenvectors) are the
      cheap ones -- exactly what homophily / link-stealing rely on.
  (2) utility headroom. Scramble = remove the cheapest r|E| real edges + add the
      cheapest r|E| decoy non-edges; measure how much of the GCN's top-k
      propagation eigenspace survives (captured energy in [0,1], 1 = identical),
      vs a RANDOM scramble, and the edge-set divergence achieved.

NOTE: relative change of the bottom-k Laplacian eigenvalues is a BAD utility
proxy here (those eigenvalues are ~0, so tiny changes look like 100%); the top-k
Ahat eigenspace overlap is the GCN-faithful, well-scaled metric. Run on the
largest connected component (the full graphs are disconnected). numpy/scipy only.
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
sys.path.insert(0, GRIDROOT)
import grid_defense as G

OUT = os.path.join(HERE, "results")
os.makedirs(OUT, exist_ok=True)
KBAND = 64
RS = [0.1, 0.3, 0.5, 0.7]
rng = np.random.RandomState(0)


def graph(ds):
    adj, _ = G.build_adjacency(ds)
    A = (adj + adj.T); A.data[:] = 1.0; A = A.tolil(); A.setdiag(0)
    A = sp.csr_matrix(A); A.eliminate_zeros()
    nfull = A.shape[0]
    ncomp, lab = connected_components(A, directed=False)
    if ncomp > 1:
        big = np.bincount(lab).argmax(); idx = np.where(lab == big)[0]
        A = A[idx][:, idx]
    print(f"  [{ds}] full n={nfull} ({ncomp} comps) -> LCC n={A.shape[0]}")
    return sp.csr_matrix(A)


def norm_adj(A):
    AI = A + sp.eye(A.shape[0])
    dinv = 1.0 / np.sqrt(np.asarray(AI.sum(1)).ravel())
    D = sp.diags(dinv)
    return (D @ AI @ D).tocsr()


def top_eigvecs(A, k):
    vals, vecs = eigsh(norm_adj(A), k=k, which="LA")
    return vecs                                   # [n, k]


def overlap(U0, U1):
    """captured energy of U0's span inside U1's span, in [0,1]."""
    M = U1.T @ U0
    return float((M ** 2).sum() / U0.shape[1])


def run(ds):
    A = graph(ds)
    n = A.shape[0]
    # (1) Laplacian low-freq eigenvectors for the per-edge cost ranking
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    w, U = np.linalg.eigh(L.toarray())
    Ub = U[:, 1:KBAND + 1]                        # smooth band (skip lambda0)
    rows, cols = sp.triu(A, k=1).nonzero()
    E = np.stack([rows, cols], 1)

    def impact(pairs):
        d = Ub[pairs[:, 0]] - Ub[pairs[:, 1]]
        return (d ** 2).sum(1)
    imp_E = impact(E)
    Eset = set(map(tuple, E.tolist()))
    NE = []
    while len(NE) < len(E):
        i, j = rng.randint(n), rng.randint(n)
        if i > j: i, j = j, i
        if i != j and (i, j) not in Eset:
            NE.append((i, j))
    NE = np.array(NE); imp_NE = impact(NE)
    ratio = imp_NE.mean() / imp_E.mean() if imp_E.mean() > 1e-12 else float("inf")

    print(f"===== {ds} =====  LCC n={n}  |E|={len(E)}  band/k={KBAND}")
    print(f"  per-edge spectral cost (sum|dlambda|):  real edges  median={np.median(imp_E):.4f} mean={imp_E.mean():.4f}")
    print(f"                                          non-edges   median={np.median(imp_NE):.4f} mean={imp_NE.mean():.4f}")
    print(f"  -> real (within-community) edges are {ratio:.1f}x spectrally cheaper to flip than random non-edges")

    # (2) utility headroom via Ahat top-k eigenspace overlap
    U0 = top_eigvecs(A, KBAND)
    order_E = np.argsort(imp_E); order_NE = np.argsort(imp_NE)
    rowsout = []
    for r in RS:
        m = int(r * len(E))
        for mode in ("spectral", "random"):
            if mode == "spectral":
                rm, add = E[order_E[:m]], NE[order_NE[:m]]
            else:
                rm, add = E[rng.permutation(len(E))[:m]], NE[rng.permutation(len(NE))[:m]]
            A2 = A.tolil()
            for i, j in rm: A2[i, j] = 0; A2[j, i] = 0
            for i, j in add: A2[i, j] = 1; A2[j, i] = 1
            A2 = sp.csr_matrix(A2)
            ov = overlap(U0, top_eigvecs(A2, KBAND))
            dv = 2 * r / (1 + r)
            rowsout.append(dict(dataset=ds, r=r, mode=mode, divergence=dv, overlap=ov))
        sp_ov = [x["overlap"] for x in rowsout if x["r"] == r and x["mode"] == "spectral"][0]
        rn_ov = [x["overlap"] for x in rowsout if x["r"] == r and x["mode"] == "random"][0]
        print(f"  r={r:.0%}  edge-divergence={2*r/(1+r):.0%}  |  GCN top-{KBAND} eigenspace kept:"
              f"  spectral-pick={sp_ov:.3f}   random-pick={rn_ov:.3f}")
    return dict(ds=ds, imp_E=imp_E, imp_NE=imp_NE, rows=rowsout)


def main():
    summ = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for col, ds in enumerate(["cora", "citeseer"]):
        r = run(ds); summ += r["rows"]
        ax = axes[0, col]
        hi = np.percentile(np.concatenate([r["imp_E"], r["imp_NE"]]), 99)
        b = np.linspace(0, hi, 60)
        ax.hist(r["imp_E"], bins=b, density=True, alpha=0.6, color="#d62728", label="real edges (remove)")
        ax.hist(r["imp_NE"], bins=b, density=True, alpha=0.5, color="#1f77b4", label="non-edges (add)")
        ax.set_title(f"{ds}: per-edge spectral cost"); ax.set_xlabel("sum_m (u_m[i]-u_m[j])^2  (lower = cheaper)")
        ax.set_ylabel("density"); ax.legend(fontsize=8)
        ax2 = axes[1, col]
        for mode, c in (("spectral", "#2ca02c"), ("random", "#999999")):
            xs = [x["divergence"] for x in r["rows"] if x["mode"] == mode]
            ys = [x["overlap"] for x in r["rows"] if x["mode"] == mode]
            ax2.plot(xs, ys, "-o", color=c, label=f"{mode} pick")
        ax2.set_title(f"{ds}: GCN eigenspace kept vs edge divergence")
        ax2.set_xlabel("edge-set divergence (Jaccard dist.)"); ax2.set_ylabel(f"top-{KBAND} eigenspace overlap (1=identical)")
        ax2.set_ylim(0, 1.02); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    fig.suptitle("Spectrum-keep edge-scramble headroom: how much can the edge set change while the\n"
                 "GCN's smooth (top-k) propagation subspace survives?", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, "spectral_headroom.png"); fig.savefig(p, dpi=130)
    json.dump(summ, open(os.path.join(OUT, "spectral_headroom.json"), "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
