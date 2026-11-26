#!/usr/bin/env python3
"""EXP1 GCond: edge-DP sparsify -> GCond graph condensation (gradient matching) on
A_dp -> train the eval GCN on the synthetic graph -> serve on A_dp (full original
nodes) -> posteriors for the LSA attack. Runs in the GCond venv (deeprobust etc.).

Injection: build the dpr dataset normally (Planetoid order, GCond's own split +
feature normalization -- inherent to GCond, flagged), replace dpr.adj with A_dp, then
Transd2Ind rebuilds adj_train from A_dp and the final predict(feat_full, adj_full)
serves on A_dp. reduction_rate r is a fraction of GCond's (Planetoid) train set.

    python gcond_dp.py --dataset cora --eps 8.0 --r 0.5
Posteriors -> <repo>/../../Benchmarking/temp/GRID/attack/data/pred/{ds}_exp1gcond_{r}_e{eps}_pred.pkl
"""
import os, sys, argparse, pickle, time, random
import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(os.path.dirname(HERE))
PRED = os.path.join(ROOT, "Benchmarking", "temp", "GRID", "attack", "data", "pred")

from utils import get_dataset, Transd2Ind
from gcond_agent_transduct import GCond
from models.gcn import GCN


def dp_sparsify(A, eps, seed=0):
    n = A.shape[0]; rng = np.random.RandomState(seed); q = 1.0 / (1.0 + np.exp(eps))
    A = sp.csr_matrix(A); A.data[:] = 1.0; A = A.maximum(A.T)
    co = A.tocoo(); up = co.row < co.col
    real = set(zip(co.row[up].tolist(), co.col[up].tolist()))
    kept = [e for e in real if rng.random() > q]
    tot = n * (n - 1) // 2
    n_add = int(rng.binomial(tot - len(real), q)); added = set()
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
    return Adp


class A:  # args namespace (GCond AUTHOR_ARGS, CPU)
    def __init__(self, ds, r):
        self.dataset = ds; self.reduction_rate = r; self.seed = 1
        self.nlayers = 2; self.sgc = 1; self.hidden = 256
        self.lr_feat = 1e-4; self.lr_adj = 1e-4; self.lr_model = 0.01
        self.weight_decay = 0.0; self.dropout = 0.0; self.alpha = 0
        self.dis_metric = "ours"; self.epochs = 600; self.inner = 0; self.outer = 20
        self.debug = 0; self.one_step = 0; self.keep_ratio = 1.0
        self.normalize_features = True; self.save = 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--eps", type=float, required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()
    random.seed(1); np.random.seed(1); torch.manual_seed(1)
    dev = "cpu"

    dpr = get_dataset(a.dataset, normalize_features=True)
    Adp = dp_sparsify(dpr.adj, a.eps)
    print(f"[{a.dataset}] eps={a.eps}: A_dp edges={int(Adp.nnz//2)}", flush=True)
    dpr.adj = Adp
    data = Transd2Ind(dpr, keep_ratio=1.0)

    args = A(a.dataset, a.r); args.epochs = a.epochs
    t0 = time.time()
    agent = GCond(data, args, device=dev)
    agent.train(verbose=False)
    feat_syn = agent.feat_syn.detach()
    adj_syn = agent.pge.inference(feat_syn).detach()
    labels_syn = agent.labels_syn
    model = GCN(nfeat=feat_syn.shape[1], nhid=256, dropout=0.5, weight_decay=5e-4,
                nlayers=2, nclass=data.nclass, device=dev).to(dev)
    model.fit_with_val(feat_syn, adj_syn, labels_syn, data, train_iters=600, normalize=True, verbose=False)
    model.eval()
    output = model.predict(data.feat_full, data.adj_full)         # serve on A_dp, full nodes
    post = torch.exp(output).detach().cpu().numpy()
    post = post / post.sum(1, keepdims=True)
    acc = float((post.argmax(1)[data.idx_test] == data.labels_full[data.idx_test]).mean())
    os.makedirs(PRED, exist_ok=True)
    name = f"{a.dataset}_exp1gcond_{a.r}_e{a.eps}"
    pickle.dump(np.asarray(post, dtype=np.float64), open(os.path.join(PRED, f"{name}_pred.pkl"), "wb"))
    print(f"[saved] {name}_pred.pkl shape={post.shape} n_syn={feat_syn.shape[0]} "
          f"gcond_test_acc={acc*100:.1f}% secs={time.time()-t0:.0f}", flush=True)


if __name__ == "__main__":
    main()
