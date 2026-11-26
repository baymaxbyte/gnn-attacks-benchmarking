#!/usr/bin/env python3
"""PrivGNN (Olatunji et al., TMLR'23) adapted to the Cora/CiteSeer link-stealing
benchmark, consistent with the MCC/KRON boxes.

Mapping (single-graph Cora/CiteSeer -> PrivGNN's private/public two-graph setting):
  * 60/20/20 class-balanced split, seed 0 (same as the boxes). The 20% TEST split
    is held out for evaluation only. The other 80% is the PRIVATE labeled pool.
  * |Q| public query nodes are sampled from the non-test pool. Their true labels
    are NEVER used for the student -- only the DP pseudo-labels.
  * Per query v (PrivGNN Algorithm 1): Poisson-subsample the private pool (ratio
    gamma), take the K feature-nearest private nodes (KNN), build their induced
    private subgraph, train a 2-layer GraphSAGE teacher (inductive, as the paper
    specifies), predict v by attaching it to its K neighbours, then add
    Laplace(0, 1/lambda) noise to the posterior and argmax -> pseudo-label.
  * RELEASED student = the SAME KRON/UGC-consistent 3-layer GCN as every other
    benchmark (mcc_boxes.GCN), trained transductively on the full public graph
    using Q + pseudo-labels, served on ALL original nodes. Its posteriors are what
    the He et al. attacks probe -> keeps the leakage comparison apples-to-apples.

Privacy "settings" = the noise level lambda (lambda = 1/beta); smaller lambda =
more noise = stronger privacy. Teacher CLEAN posteriors are cached once per
dataset (phase=teach), so the lambda sweep (phase=student) is cheap.

Posteriors -> attack/data/pred/{ds}_privgnn_l{lambda}_pred.pkl, shape (n, C),
Planetoid node order. Reuses the existing {ds}_gmcc reference g for the attack.
"""
import os, sys, argparse, pickle, time
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as Fn
from torch_geometric.nn import SAGEConv

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC)
import mcc_boxes as M  # load, split_6020, GCN, _ei, save, PRED, EPOCHS/LR/WD/DROPOUT

ART = os.path.join(HERE, "artifacts"); os.makedirs(ART, exist_ok=True)
PRED = M.PRED

# PrivGNN hyperparameters
GAMMA = 0.3            # Poisson subsampling ratio (paper default)
K_KNN = 128            # KNN size (small, graphs are sparse -- paper: pick by avg degree)
Q_SIZE = 500          # number of public query nodes (paper used 500-1000)
TEACH_HID, TEACH_EPOCHS, TEACH_LR, TEACH_WD, TEACH_DROP = 64, 60, 0.01, 5e-4, 0.5
READOUT_M = 16         # attach the query to its M nearest neighbours for inductive readout
STU_VAL_FRAC = 0.2     # fraction of Q held out as pseudo-validation for the student
SEED = 0
torch.set_num_threads(4)


class SAGE(torch.nn.Module):
    """2-layer GraphSAGE teacher (inductive)."""
    def __init__(self, d, c):
        super().__init__()
        self.c1 = SAGEConv(d, TEACH_HID); self.c2 = SAGEConv(TEACH_HID, c)

    def forward(self, x, ei):
        x = Fn.relu(self.c1(x, ei)); x = Fn.dropout(x, TEACH_DROP, training=self.training)
        return Fn.log_softmax(self.c2(x, ei), dim=1)


def prep(ds):
    A, X, labels, C = M.load(ds)
    mtr, mva, mte = M.split_6020(labels, C)
    pool = np.where(mtr | mva)[0]          # 80% private labeled pool / query source
    rng = np.random.RandomState(SEED)
    Q = np.sort(rng.choice(pool, size=min(Q_SIZE, len(pool)), replace=False))
    return A, X, labels, C, mte, pool, Q


def _induced(A, N):
    sub = A[N][:, N].tocoo()
    return torch.tensor(np.vstack([sub.row, sub.col]), dtype=torch.long)


def teacher_clean(ds, K=K_KNN, gamma=GAMMA):
    """Compute + cache the CLEAN (noise-free) teacher posterior for every query."""
    A, X, labels, C, mte, pool, Q = prep(ds)
    d = X.shape[1]
    clean = np.zeros((len(Q), C), dtype=np.float64)
    t0 = time.time()
    for i, v in enumerate(Q):
        rng = np.random.RandomState(SEED * 100003 + int(v))
        keep = pool[(rng.random(len(pool)) < gamma) & (pool != v)]
        if len(keep) == 0:
            clean[i] = np.ones(C) / C; continue
        k = min(K, len(keep))
        dist = ((X[keep] - X[v]) ** 2).sum(1)
        N = keep[np.argsort(dist)[:k]]
        eiH = _induced(A, N)
        XH = torch.tensor(X[N]); yH = torch.from_numpy(labels[N])
        torch.manual_seed(SEED)
        teach = SAGE(d, C); opt = torch.optim.Adam(teach.parameters(), lr=TEACH_LR, weight_decay=TEACH_WD)
        teach.train()
        for _ in range(TEACH_EPOCHS):
            opt.zero_grad(); Fn.nll_loss(teach(XH, eiH), yH).backward(); opt.step()
        # inductive readout: attach v (index k) to its READOUT_M nearest neighbours
        # (N is distance-sorted, so range(m) are the closest). All-K over-smooths.
        m = min(READOUT_M, len(N))
        Xaug = torch.tensor(np.vstack([X[N], X[v][None]]))
        star = np.array([[k] * m + list(range(m)),
                         list(range(m)) + [k] * m])
        ei_aug = torch.cat([eiH, torch.tensor(star, dtype=torch.long)], 1)
        teach.eval()
        with torch.no_grad():
            clean[i] = teach(Xaug, ei_aug)[k].exp().numpy()
        if (i + 1) % 50 == 0:
            el = time.time() - t0
            print(f"  [{ds}] teacher {i+1}/{len(Q)}  elapsed={el:.0f}s  eta={el/(i+1)*(len(Q)-i-1):.0f}s", flush=True)
    clean = clean / clean.sum(1, keepdims=True).clip(1e-12)
    pl_clean = clean.argmax(1)
    acc = float((pl_clean == labels[Q]).mean())
    np.savez(os.path.join(ART, f"{ds}_teacher.npz"), clean=clean, Q=Q, K=K, gamma=gamma)
    print(f"[{ds}] cached {len(Q)} teacher posteriors  clean pseudo-label acc vs truth = {acc*100:.1f}%  "
          f"(K={K}, gamma={gamma})", flush=True)
    return acc


def train_student(ds, lam, seed=0):
    A, X, labels, C, mte, pool, Q = prep(ds)
    n, d = X.shape
    z = np.load(os.path.join(ART, f"{ds}_teacher.npz"))
    clean, Qc = z["clean"], z["Q"]
    assert np.array_equal(Qc, Q), "cached Q mismatch -- re-run teach"
    # noisy labelling: Laplace(0, 1/lambda) per coordinate, then argmax
    rng = np.random.RandomState(1234 + int(round(lam * 1000)))
    noisy = clean + rng.laplace(0.0, 1.0 / lam, size=clean.shape)
    pl = noisy.argmax(1).astype(np.int64)
    pl_acc = float((pl == labels[Q]).mean())

    Xo = torch.tensor(X); eio = M._ei(A); ewo = torch.ones(eio.shape[1])
    perm = np.random.RandomState(seed).permutation(len(Q))
    nval = int(STU_VAL_FRAC * len(Q))
    val_q, tr_q = Q[perm[:nval]], Q[perm[nval:]]
    y_full = np.zeros(n, dtype=np.int64); y_full[Q] = pl
    yt = torch.from_numpy(y_full)
    mtr = torch.zeros(n, dtype=torch.bool); mtr[tr_q] = True
    mval = torch.zeros(n, dtype=torch.bool); mval[val_q] = True

    torch.manual_seed(seed); model = M.GCN(d, C)
    opt = torch.optim.Adam(model.parameters(), lr=M.LR, weight_decay=M.WD)
    best, best_state = -1.0, None
    for ep in range(M.EPOCHS):
        model.train(); opt.zero_grad()
        Fn.nll_loss(model(Xo, eio, ewo)[mtr], yt[mtr]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xo, eio, ewo).argmax(1)[mval] == yt[mval]).float().mean())
        if va > best:
            best, best_state = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state); model.eval()
    with torch.no_grad():
        post = model(Xo, eio, ewo).exp().numpy()
    post = post / post.sum(1, keepdims=True)
    test_acc = float((post.argmax(1)[mte] == labels[mte]).mean())
    tag = f"l{lam}"
    M.save(f"{ds}_privgnn_{tag}", post)
    print(f"[{ds}] lambda={lam}: pseudo-label acc={pl_acc*100:.1f}%  student held-out test acc={test_acc*100:.1f}%  "
          f"mean_max_p={post.max(1).mean():.3f}", flush=True)
    return pl_acc, test_acc


def main():
    global Q_SIZE
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--phase", required=True, choices=["teach", "student"])
    ap.add_argument("--lam", type=float, help="noise level lambda=1/beta (student phase)")
    ap.add_argument("--K", type=int, default=K_KNN)
    ap.add_argument("--gamma", type=float, default=GAMMA)
    ap.add_argument("--Q", type=int, default=Q_SIZE, help="number of query nodes")
    a = ap.parse_args()
    Q_SIZE = a.Q
    if a.phase == "teach":
        teacher_clean(a.dataset, K=a.K, gamma=a.gamma)
    else:
        assert a.lam is not None, "need --lam"
        train_student(a.dataset, a.lam)


if __name__ == "__main__":
    main()
