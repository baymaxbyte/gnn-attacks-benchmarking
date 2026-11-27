#!/usr/bin/env python3
"""MCC leakage benchmark -- build the black boxes with the SAME classifier and
serving protocol as the KRON/UGC benchmarks (the documented 'consistent' box, and
the one our recent GRID experiments used), swapping in MCC coarsening.

MCC supplies ONLY the coarsening (block-MM C-update from multi_comp.py, pure torch).
Black box == blackbox_kron.py:
  * 3-layer GCNConv (d->64->64->C), dropout 0.5, LR 0.003, wd 5e-4, 500 epochs,
    Adam, edge weights threaded, BEST-VALIDATION weights;
  * trained on the COARSE graph (coarse node = merged supernode; its label = majority
    label of its TRAIN members; a supernode is a training node iff it contains >=1
    original train node), VALIDATED/TESTED on the ORIGINAL graph with the 60/20/20
    seed-0 split -- held-out, realistic accuracy (no in-sample leakage);
  * whole-graph control: same GCN trained on the original graph's train split;
  * reference g = the same GCN run GRAPH-FREE, trained on 10% labels (seed 123).

Posteriors -> attack/data/pred/{ds}_mccbb_{setting}_pred.pkl and {ds}_gmcc_pred.pkl,
shape (n, n_classes) float64, node order = the attack's Planetoid order.
"""
import os, sys, argparse, pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import random as sprand
from scipy.stats import rv_continuous
import torch
import torch.nn.functional as Fn
from torch_geometric.nn import GCNConv

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(GRIDROOT, "attack")
DATAP = os.path.join(ATTACK, "data", "dataset", "original") + os.sep
PRED = os.path.join(ATTACK, "data", "pred")
sys.path.insert(0, ATTACK)
from lsa.data import load_data_original, load_data

# ---- black box == blackbox_kron.py (the consistent classifier) ----
HIDDEN, DROPOUT, LR, WD, EPOCHS = 64, 0.5, 0.003, 5e-4, 500
SPLIT_SEED, LABEL_RATE, LABEL_SEED = 0, 0.10, 123
MCC_ITERS, MCC_SEED, THRESH, EDGE_THRESH = 10, 1, 1e-10, 0.1
torch.set_num_threads(4)


def load(ds):
    adj, feat, *rest = load_data_original(DATAP, ds)
    labels = np.asarray(rest[2]).argmax(1).astype(np.int64)
    A = sp.csr_matrix(adj); A.data[:] = 1.0; A = A.maximum(A.T).tolil(); A.setdiag(0)
    A = sp.csr_matrix(A); A.eliminate_zeros()
    X = np.asarray(feat.todense(), dtype=np.float32)
    return A, X, labels, int(labels.max()) + 1


def load_any(ds):
    """Like load() but via lsa.data.load_data, which also handles the .npz
    datasets (squirrel/chameleon) in addition to Planetoid (cora/citeseer).
    Same post-processing: binary symmetric adjacency, no self-loops, dense X."""
    adj, feat, *rest = load_data(DATAP, ds)
    labels = np.asarray(rest[2]).argmax(1).astype(np.int64)
    A = sp.csr_matrix(adj); A.data[:] = 1.0; A = A.maximum(A.T).tolil(); A.setdiag(0)
    A = sp.csr_matrix(A); A.eliminate_zeros()
    X = np.asarray(feat.todense(), dtype=np.float32)
    return A, X, labels, int(labels.max()) + 1


def split_6020(labels, nclasses, seed=SPLIT_SEED):
    rng = np.random.RandomState(seed); n = len(labels); tr, va, te = [], [], []
    for c in range(nclasses):
        idx = np.where(labels == c)[0]; idx = idx[rng.permutation(len(idx))]
        a, b = int(round(0.6 * len(idx))), int(round(0.2 * len(idx)))
        tr += list(idx[:a]); va += list(idx[a:a + b]); te += list(idx[a + b:])
    def m(ix):
        z = np.zeros(n, bool); z[ix] = True; return z
    return m(tr), m(va), m(te)


# ====================== MCC coarsening (pure torch, from multi_comp) ======================
class _Gauss(rv_continuous):
    def _rvs(self, size=None, random_state=None):
        return random_state.standard_normal(size)


def _init_C(p, k, seed=MCC_SEED):
    rv = _Gauss(seed=seed)()
    return torch.tensor(sprand(p, k, density=0.15, random_state=seed, data_rvs=rv.rvs).toarray(),
                        dtype=torch.float64)


def _lambda_update(lb, ub, beta, U, Lw, k):
    d = torch.diag(U.t().double() @ Lw @ U)
    lam = 0.5 * (d + torch.sqrt(d.pow(2) + 4.0 / beta))
    lam, _ = torch.sort(lam, descending=True)
    lam = lam.clone(); lam[lam > ub] = ub; lam[lam < lb] = lb
    return lam


def _bracket2(C, theta):
    Lw = (C.t() @ theta @ C).double()
    _, U = torch.linalg.eigh(Lw); U = U.flip(dims=(1,)).double()
    lam = _lambda_update(1e-5, 1e4, 0.5, U, Lw, min(7, Lw.size(0) - 1))
    return U @ torch.diag(lam) @ U.t()


def _update_C(C, theta, ones, alpha, beta, gamma):
    br = (C.t() @ theta @ C) - _bracket2(C, theta)
    Cn = C - gamma * (alpha * (C @ ones) + beta * (theta @ C @ br.to(C.dtype)))
    Cn[Cn < THRESH] = THRESH
    return Cn / Cn.sum(1, keepdim=True).clamp_min(THRESH)


def mcc_coarsen(A, r, alpha, beta, gamma):
    p = A.shape[0]; k = int(p * r)
    Ad = torch.tensor(np.asarray(A.todense()), dtype=torch.float64)
    theta = torch.diag(Ad.sum(1)) - Ad
    C = _init_C(p, k); ones = torch.ones(k, k, dtype=torch.float64)
    for _ in range(MCC_ITERS):
        C = _update_C(C, theta, ones, alpha, beta, gamma)
    return C, Ad


def build_coarse(C, Ad, X, labels, train_mask, nclasses):
    """Hardened one-hot assignment -> coarse graph; coarse labels = majority of
    each supernode's TRAIN members; keep_train marks supernodes with >=1 train node."""
    p, k = C.shape
    a = C.argmax(1).numpy()
    C0 = torch.zeros(p, k, dtype=torch.float64); C0[torch.arange(p), torch.from_numpy(a)] = 1.0
    L = torch.diag(Ad.sum(1)) - Ad
    Wc = (-(C0.t() @ L @ C0)) * (1 - torch.eye(k, dtype=torch.float64)); Wc[Wc < EDGE_THRESH] = 0.0
    counts = C0.sum(0)
    Xc = ((C0.t() @ torch.tensor(X, dtype=torch.float64)) / counts.clamp(min=1).unsqueeze(1)).float()
    Yc = torch.zeros(k, dtype=torch.long); keep = torch.zeros(k, dtype=torch.bool)
    for s in range(k):
        mem = np.where(a == s)[0]
        if len(mem) == 0:
            continue
        trm = mem[train_mask[mem]]
        src = trm if len(trm) > 0 else mem
        v, cc = np.unique(labels[src], return_counts=True); Yc[s] = int(v[cc.argmax()]); keep[s] = len(trm) > 0
    ei = (Wc > 0).nonzero().t().long(); ew = Wc[ei[0], ei[1]].float()
    return Xc, ei, ew, Yc, keep


# =============================== box == blackbox_kron GCN ===============================
class GCN(torch.nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.conv1 = GCNConv(d, HIDDEN); self.conv2 = GCNConv(HIDDEN, 64); self.conv3 = GCNConv(64, c)

    def forward(self, x, ei, ew):
        x = Fn.relu(self.conv1(x, ei, ew)); x = Fn.dropout(x, DROPOUT, training=self.training)
        x = Fn.relu(self.conv2(x, ei, ew)); x = Fn.dropout(x, DROPOUT, training=self.training)
        return Fn.log_softmax(self.conv3(x, ei, ew), dim=1)


def _ei(A):
    s, d = A.nonzero(); return torch.from_numpy(np.vstack([s, d])).long()


def train_box(xtr, eitr, ewtr, ytr, mtr, Xo, eio, ewo, labels, val_mask, d, C, seed=0):
    """Train on (xtr, eitr, ewtr, ytr[mtr]); select by val acc on the ORIGINAL graph."""
    torch.manual_seed(seed); model = GCN(d, C)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    yt = torch.from_numpy(labels); vm = torch.from_numpy(val_mask)
    best, best_state = -1.0, None
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        Fn.nll_loss(model(xtr, eitr, ewtr)[mtr], ytr[mtr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xo, eio, ewo).argmax(1)[vm] == yt[vm]).float().mean())
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        post = model(Xo, eio, ewo).exp().numpy()
    return post / post.sum(1, keepdims=True), best


def reference_g(X, labels, n, d, C):
    rng = np.random.RandomState(LABEL_SEED); idx = torch.from_numpy(np.sort(rng.permutation(n)[:int(n * LABEL_RATE)]))
    Xt = torch.tensor(X); yt = torch.from_numpy(labels)
    ee = torch.zeros((2, 0), dtype=torch.long); ew = torch.zeros(0)
    torch.manual_seed(0); g = GCN(d, C); opt = torch.optim.Adam(g.parameters(), lr=LR, weight_decay=WD)
    g.train()
    for _ in range(EPOCHS):
        opt.zero_grad(); Fn.nll_loss(g(Xt, ee, ew)[idx], yt[idx]).backward(); opt.step()
    g.eval()
    with torch.no_grad():
        post = g(Xt, ee, ew).exp().numpy()
    return post / post.sum(1, keepdims=True)


def save(name, post):
    os.makedirs(PRED, exist_ok=True)
    p = np.asarray(post, dtype=np.float64)
    with open(os.path.join(PRED, f"{name}_pred.pkl"), "wb") as f:
        pickle.dump(p, f)
    print(f"  [saved] {name}_pred.pkl shape={p.shape}  mean_max_p={p.max(1).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--setting", required=True)
    ap.add_argument("--alpha", type=float); ap.add_argument("--beta", type=float); ap.add_argument("--gamma", type=float)
    ap.add_argument("--with_g", action="store_true")
    a = ap.parse_args()

    A, X, labels, Cn = load(a.dataset); n, d = A.shape[0], X.shape[1]
    mtr, mva, mte = split_6020(labels, Cn)
    Xo = torch.tensor(X); eio = _ei(A); ewo = torch.ones(eio.shape[1]); yt = torch.from_numpy(labels)
    print(f"[{a.dataset}] n={n} d={d} classes={Cn}  setting={a.setting}  split {mtr.sum()}/{mva.sum()}/{mte.sum()}")

    if a.setting == "whole":
        post, bv = train_box(Xo, eio, ewo, yt, torch.from_numpy(mtr), Xo, eio, ewo, labels, mva, d, Cn)
    else:
        r = float(a.setting)
        assert None not in (a.alpha, a.beta, a.gamma), "need --alpha/--beta/--gamma"
        C, Ad = mcc_coarsen(A, r, a.alpha, a.beta, a.gamma)
        Xc, eic, ewc, Yc, keep = build_coarse(C, Ad, X, labels, mtr, Cn)
        print(f"  coarsened p={n} -> k={C.shape[1]}  train_supernodes={int(keep.sum())}  coarse_edges={eic.shape[1]//2}"
              f"  (alpha={a.alpha} beta={a.beta} gamma={a.gamma})")
        post, bv = train_box(Xc, eic, ewc, Yc, keep, Xo, eio, ewo, labels, mva, d, Cn)

    test_acc = float((post.argmax(1)[mte] == labels[mte]).mean())
    print(f"  box: best_val={bv*100:.1f}%  held-out test_acc={test_acc*100:.1f}%")
    save(f"{a.dataset}_mccbb_{a.setting}", post)
    if a.with_g:
        print("  training reference g (graph-free, 10% labels)...")
        save(f"{a.dataset}_gmcc", reference_g(X, labels, n, d, Cn))


if __name__ == "__main__":
    main()
