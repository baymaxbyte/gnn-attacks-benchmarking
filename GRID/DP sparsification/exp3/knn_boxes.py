#!/usr/bin/env python3
"""EXP3: discard the original adjacency; build our OWN graph with feature-KNN (k=10)
and run the standard black-box GCN on THAT graph.

Question: does a surrogate feature-similarity graph still leak the ORIGINAL private
links? The box is trained AND served on the KNN graph (never sees the real edges),
but the attack's ground-truth pairs are the ORIGINAL edges. If the KNN graph mirrors
the original (both homophilous in features), RC1 (class) + partial RC2 persist; if it
diverges, original-link leakage should fall.

Box == the KRON/UGC/MCC-consistent GCN (mcc_boxes): 3 layers d->64->64->C, dropout .5,
LR .003, wd 5e-4, 500 epochs, best-val, 60/20/20 seed-0 split, trained on the 60% train
labels over the KNN graph, served on all nodes over the KNN graph.

Posteriors -> attack/data/pred/{ds}_exp3knn_pred.pkl.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(os.path.dirname(HERE))
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC)
import mcc_boxes as M

K = 10
torch.set_num_threads(4)


def knn_graph(X, k=K):
    """Cosine top-k feature neighbours (manual, verifiable). L2-normalise with a
    zero-row guard; the 15 zero-feature CiteSeer nodes get sim 0 to all and pick
    arbitrary neighbours (negligible). Symmetrised, self-loops removed."""
    n = X.shape[0]
    Xf = np.asarray(X, dtype=np.float64)
    norm = np.linalg.norm(Xf, axis=1, keepdims=True)
    Xn = np.divide(Xf, norm, out=np.zeros_like(Xf), where=norm > 0)
    S = Xn @ Xn.T
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, k, axis=1)[:, :k]     # top-k by cosine similarity
    rows = np.repeat(np.arange(n), k); cols = idx.reshape(-1)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = A.maximum(A.T).tolil(); A.setdiag(0)         # symmetric, no self-loops
    A = sp.csr_matrix(A); A.data[:] = 1.0; A.eliminate_zeros()
    return A


def build(ds, seed=0):
    A, X, labels, C = M.load_any(ds); n, d = X.shape
    mtr, mva, mte = M.split_6020(labels, C)
    Aknn = knn_graph(X)
    # diagnostics: KNN-graph homophily + overlap with the real graph
    kr, kc = Aknn.tocoo().row, Aknn.tocoo().col
    m = kr < kc
    knn_homo = float((labels[kr[m]] == labels[kc[m]]).mean())
    real = set(map(tuple, np.array(A.nonzero()).T[A.nonzero()[0] < A.nonzero()[1]].tolist()))
    knn_set = set(zip(kr[m].tolist(), kc[m].tolist()))
    overlap = len(real & knn_set)
    print(f"[{ds}] KNN(k={K}) edges={int(Aknn.nnz//2)} homophily={knn_homo:.3f}  "
          f"overlap-with-real={overlap}/{len(real)} ({100*overlap/len(real):.1f}% of real edges)", flush=True)

    Xo = torch.tensor(X); eik = M._ei(Aknn); ewk = torch.ones(eik.shape[1]); yt = torch.from_numpy(labels)
    post, bv = M.train_box(Xo, eik, ewk, yt, torch.from_numpy(mtr), Xo, eik, ewk, labels, mva, d, C)
    acc = float((post.argmax(1)[mte] == labels[mte]).mean())
    M.save(f"{ds}_exp3knn", post)
    print(f"[{ds}] KNN-GCN box: best_val={bv*100:.1f}%  held-out test acc={acc*100:.1f}%  mean_max_p={post.max(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] or ["cora", "citeseer"]
    for ds in datasets:
        build(ds)
