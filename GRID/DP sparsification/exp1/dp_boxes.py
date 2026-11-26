#!/usr/bin/env python3
"""EXP1: edge-level DP graph sparsification -> coarsening -> black-box GCN.

DP mechanism: symmetric RANDOMIZED RESPONSE on the adjacency (the canonical
eps-edge-DP graph mechanism, and the core of Alg. 2 in Arora & Upadhyay 2019 --
no official code exists, so we implement the RR core rather than the full
spectral-SDP Alg. 1). Each potential edge bit is flipped w.p. q = 1/(1+e^eps):
real edges are dropped w.p. q, non-edges are added w.p. q. This gives pure
eps-edge-DP. Smaller eps = more noise = stronger privacy (and denser output).
Calibration showed eps<=2 explodes to ~1M edges; the tractable band is {4,6,8}.

Pipeline (served on the RELEASED DP graph, never the original -- that is the DP
defense; matches exp3's "use the constructed graph"):
  * dp    : DP-sparsify -> plain GCN box trained+served on A_dp  (isolates DP effect)
  * dpmcc : DP-sparsify -> MCC coarsen (r=0.3) -> box trained on coarse, served on A_dp
Attack ground-truth pairs remain the ORIGINAL edges (the private links).

Posteriors -> attack/data/pred/{ds}_exp1dp_e{eps}_pred.pkl and {ds}_exp1dpmcc_e{eps}_pred.pkl.
"""
import os, sys
import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(os.path.dirname(HERE))
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC)
import mcc_boxes as M

EPS = [4.0, 6.0, 8.0]
R = 0.3
HP = {"cora": (0.1, 100.0, 100.0), "citeseer": (100.0, 1.0, 0.001)}   # MCC r=0.3 hyperparams
torch.set_num_threads(4)


def dp_sparsify(A, eps, seed=0):
    """Symmetric randomized-response edge flip -> eps-edge-DP sparse graph."""
    n = A.shape[0]; rng = np.random.RandomState(seed)
    q = 1.0 / (1.0 + np.exp(eps))
    co = A.tocoo(); up = co.row < co.col
    real = set(zip(co.row[up].tolist(), co.col[up].tolist()))
    kept = [e for e in real if rng.random() > q]                 # drop real edge w.p. q
    tot = n * (n - 1) // 2
    n_add = int(rng.binomial(tot - len(real), q))                # add non-edges w.p. q
    added = set()
    while len(added) < n_add:
        i = int(rng.randint(n)); j = int(rng.randint(n))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in real or (a, b) in added:
            continue
        added.add((a, b))
    edges = kept + list(added)
    r = np.array([e[0] for e in edges]); c = np.array([e[1] for e in edges])
    Adp = sp.csr_matrix((np.ones(len(edges)), (r, c)), shape=(n, n))
    Adp = Adp.maximum(Adp.T).tolil(); Adp.setdiag(0)
    Adp = sp.csr_matrix(Adp); Adp.data[:] = 1.0; Adp.eliminate_zeros()
    return Adp, len(kept), len(added)


def build(ds):
    A, X, y, C = M.load(ds); n, d = X.shape
    mtr, mva, mte = M.split_6020(y, C)
    Xo = torch.tensor(X); yt = torch.from_numpy(y)
    al, be, ga = HP[ds]
    for eps in EPS:
        Adp, nkept, nadd = dp_sparsify(A, eps)
        m = int(Adp.nnz // 2)
        kr, kc = Adp.tocoo().row, Adp.tocoo().col; um = kr < kc
        homo = float((y[kr[um]] == y[kc[um]]).mean())
        eidp = M._ei(Adp); ewdp = torch.ones(eidp.shape[1])
        print(f"[{ds}] eps={eps}: A_dp edges={m} (kept_real={nkept}, noise={nadd}) homophily={homo:.3f}", flush=True)
        # (1) DP, no coarsening: GCN trained + served on A_dp
        post, bv = M.train_box(Xo, eidp, ewdp, yt, torch.from_numpy(mtr), Xo, eidp, ewdp, y, mva, d, C)
        acc = float((post.argmax(1)[mte] == y[mte]).mean())
        M.save(f"{ds}_exp1dp_e{eps}", post)
        print(f"    dp    box acc={acc*100:.1f}%  mean_max_p={post.max(1).mean():.3f}", flush=True)
        # (2) DP -> MCC coarsen -> box trained on coarse, served on A_dp
        Cmat, Ad = M.mcc_coarsen(Adp, R, al, be, ga)
        Xc, eic, ewc, Yc, keep = M.build_coarse(Cmat, Ad, X, y, mtr, C)
        post2, bv2 = M.train_box(Xc, eic, ewc, Yc, keep, Xo, eidp, ewdp, y, mva, d, C)
        acc2 = float((post2.argmax(1)[mte] == y[mte]).mean())
        M.save(f"{ds}_exp1dpmcc_e{eps}", post2)
        print(f"    dpmcc box acc={acc2*100:.1f}%  (k={Cmat.shape[1]}, train_sn={int(keep.sum())})  "
              f"mean_max_p={post2.max(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    for ds in ["cora", "citeseer"]:
        build(ds)
