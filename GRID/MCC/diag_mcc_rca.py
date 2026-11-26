#!/usr/bin/env python3
"""RCA: why MCC coarsening does NOT defend link-stealing.

The He et al. attack scores a node pair by the SIMILARITY of the two served
posteriors (correlation distance; closer = predicted linked). So "defending"
would require linked pairs' posteriors to be NO more similar than unlinked pairs'.
We decompose the leak in the saved MCC box posteriors:

  RC1  class homophily  -- linked nodes are same-class, and the box is a class
       predictor, so same-class posteriors are similar. Measured by a CLASS-ONLY
       attacker (one-hot true labels as 'posteriors').
  RC2  serving-time adjacency smoothing -- the box is a GCN SERVED on the ORIGINAL
       graph, so message passing pulls neighbours together beyond class. Measured
       by (a) the SAME-CLASS-restricted leak (linked vs unlinked, both same class)
       and (b) a graph-free-serving control: the SAME trained weights served with
       the edges removed.

If coarsening defended, the coarse settings (0.1/0.3) would show a smaller
linked-vs-unlinked similarity gap than 'whole'. We also check co-merging
(do linked nodes land in the same supernode?).
"""
import os, sys, pickle
import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import mcc_boxes as M
import torch, torch.nn.functional as Fn

PRED = M.PRED
HP = {("cora", "0.3"): (0.1, 100.0, 100.0), ("cora", "0.1"): (10.0, 100.0, 0.1),
      ("citeseer", "0.3"): (100.0, 1.0, 0.001), ("citeseer", "0.1"): (100.0, 100.0, 0.01)}


def cnorm(P):
    Pc = P - P.mean(1, keepdims=True)
    n = np.linalg.norm(Pc, axis=1, keepdims=True); n[n == 0] = 1.0
    return Pc / n


def corrdist(U, I, J):
    return 1.0 - (U[I] * U[J]).sum(1)


def auc(U, pI, pJ, nI, nJ):
    s = np.nan_to_num(np.concatenate([-corrdist(U, pI, pJ), -corrdist(U, nI, nJ)]), nan=0.0)
    y = np.concatenate([np.ones(len(pI)), np.zeros(len(nI))])
    return roc_auc_score(y, s)


def edges(A):
    c = A.tocoo(); m = c.row < c.col
    return c.row[m].astype(int), c.col[m].astype(int)


def neg_pairs(A, n, N, y=None, same=False, seed=0):
    rng = np.random.RandomState(seed)
    linked = set(zip(*[x.tolist() for x in edges(A)]))
    I, J = [], []
    while len(I) < n:
        i, j = int(rng.randint(N)), int(rng.randint(N))
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in linked:
            continue
        if same and y is not None and y[i] != y[j]:
            continue
        I.append(i); J.append(j)
    return np.array(I), np.array(J)


def retrain_serve_both(ds, r, al, be, ga):
    """Train the MCC box (coarse graph), serve ON-graph and GRAPH-FREE with the
    SAME selected weights -> isolates serving-time smoothing (RC2)."""
    A, X, y, C = M.load(ds); mtr, mva, mte = M.split_6020(y, C)
    Xo = torch.tensor(X); eio = M._ei(A); ewo = torch.ones(eio.shape[1])
    Cmat, Ad = M.mcc_coarsen(A, r, al, be, ga)
    Xc, eic, ewc, Yc, keep = M.build_coarse(Cmat, Ad, X, y, mtr, C)
    torch.manual_seed(0); model = M.GCN(X.shape[1], C)
    opt = torch.optim.Adam(model.parameters(), lr=M.LR, weight_decay=M.WD)
    yt = torch.from_numpy(y); vm = torch.from_numpy(mva); best, bs = -1.0, None
    for _ in range(M.EPOCHS):
        model.train(); opt.zero_grad()
        Fn.nll_loss(model(Xc, eic, ewc)[keep], Yc[keep]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xo, eio, ewo).argmax(1)[vm] == yt[vm]).float().mean())
        if va > best:
            best, bs = va, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(bs); model.eval()
    empty = torch.zeros((2, 0), dtype=torch.long); ew0 = torch.zeros(0)
    with torch.no_grad():
        Pg = model(Xo, eio, ewo).exp().numpy(); Pf = model(Xo, empty, ew0).exp().numpy()
    Pg /= Pg.sum(1, keepdims=True); Pf /= Pf.sum(1, keepdims=True)
    return Pg, Pf, Cmat.argmax(1).numpy()


def main():
    for ds in ["cora", "citeseer"]:
        A, X, y, C = M.load(ds); N = A.shape[0]
        pI, pJ = edges(A); npos = len(pI)
        h = float((y[pI] == y[pJ]).mean())
        nI, nJ = neg_pairs(A, npos, N, seed=1)
        h_neg = float((y[nI] == y[nJ]).mean())
        # class-only attacker (one-hot true labels)
        U_cls = cnorm(np.eye(C)[y].astype(float))
        auc_cls = auc(U_cls, pI, pJ, nI, nJ)
        # same-class positives + same-class negatives
        sc = y[pI] == y[pJ]
        spI, spJ = pI[sc], pJ[sc]
        snI, snJ = neg_pairs(A, len(spI), N, y=y, same=True, seed=2)
        print(f"\n################ {ds}  (N={N}, C={C}, edges={npos}) ################")
        print(f"  homophily: P(same-class | linked)={h:.3f}   P(same-class | random non-edge)={h_neg:.3f}")
        print(f"  CLASS-ONLY attacker AUC (one-hot labels) = {auc_cls:.3f}   <- RC1 floor from homophily")
        print(f"  {'setting':>8} {'boxacc':>7} {'mean_maxp':>9} {'leakAUC':>8} {'sameClassAUC':>12} "
              f"{'d_link':>7} {'d_nonlk':>7} {'d_link_sc':>9} {'d_nonlk_sc':>10}")
        for s in ["whole", "0.1", "0.3"]:
            P = np.asarray(pickle.load(open(os.path.join(PRED, f"{ds}_mccbb_{s}_pred.pkl"), "rb")), float)
            U = cnorm(P)
            a_all = auc(U, pI, pJ, nI, nJ)
            a_sc = auc(U, spI, spJ, snI, snJ)
            dl, dn = corrdist(U, pI, pJ).mean(), corrdist(U, nI, nJ).mean()
            dls, dns = corrdist(U, spI, spJ).mean(), corrdist(U, snI, snJ).mean()
            _, _, mte = M.split_6020(y, C)
            acc = float((P.argmax(1)[mte] == y[mte]).mean())
            print(f"  {s:>8} {acc*100:6.1f}% {P.max(1).mean():9.3f} {a_all:8.3f} {a_sc:12.3f} "
                  f"{dl:7.3f} {dn:7.3f} {dls:9.3f} {dns:10.3f}")
        # graph-free serving control (RC2) at 0.3
        al, be, ga = HP[(ds, "0.3")]
        Pg, Pf, asg = retrain_serve_both(ds, 0.3, al, be, ga)
        Ug, Uf = cnorm(Pg), cnorm(Pf)
        print(f"  --- RC2 control (MCC 0.3 box, same weights): serving ON-graph vs GRAPH-FREE ---")
        print(f"      same-class leak AUC:  on-graph={auc(Ug, spI, spJ, snI, snJ):.3f}   "
              f"graph-free={auc(Uf, spI, spJ, snI, snJ):.3f}")
        print(f"      all-pairs  leak AUC:  on-graph={auc(Ug, pI, pJ, nI, nJ):.3f}   "
              f"graph-free={auc(Uf, pI, pJ, nI, nJ):.3f}")
        # co-merging: do linked nodes share a supernode?
        cm_link = float((asg[pI] == asg[pJ]).mean())
        cm_neg = float((asg[nI] == asg[nJ]).mean())
        print(f"      co-merge (0.3): P(same supernode | linked)={cm_link:.3f}  vs  non-edge={cm_neg:.3f}")


if __name__ == "__main__":
    main()
