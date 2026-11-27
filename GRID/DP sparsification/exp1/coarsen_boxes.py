#!/usr/bin/env python3
"""EXP1 (multi-method): edge-DP sparsify -> coarsen with FGC / UGC / KRON / MCC at
each method's native ratios -> train the consistent 3-layer GCN on the coarse graph,
serve on the DP graph A_dp -> posteriors for the LSA attack. NO GRID.

Coarsening is injected on A_dp (the DP-sparsified graph), never the real graph:
  * mcc  : mcc_coarsen(A_dp, r, hp) -> build_coarse (train-majority labels, keep mask)
  * fgc  : fgc_solver.coarsen(laplacian(A_dp), X, k=int(n*r)) -> build_coarse
  * kron : kron_reduce(A_dp, r) -> node subset (no merge); train on kept, serve on A_dp
  * ugc  : augment [X|alpha*A_dp] -> LSH partition -> coarse_graph (box uses augmented feats)
All boxes serve on A_dp; attack ground-truth = ORIGINAL edges. GOREN/GCond handled
separately (they need their own venvs).

Posteriors -> attack/data/pred/{ds}_exp1{method}_{r}_e{eps}_pred.pkl
Ratio conventions differ per method (fgc=kept; kron/ugc=removed; mcc=kept) -- noted.
"""
import os, sys, importlib.util
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(os.path.dirname(HERE))          # .../GRID
BENCH = os.path.dirname(os.path.dirname(GRIDROOT))          # .../Benchmarking
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC); import mcc_boxes as M
sys.path.insert(0, HERE)
from dp_boxes import dp_sparsify


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m

FGC = _load("fgc_solver", os.path.join(BENCH, "FGC benchmarking", "fgc_solver.py"))
UGC = _load("ugc_solver", os.path.join(BENCH, "UGC benchmarking", "ugc_solver.py"))
KRON = _load("kron_reduce", os.path.join(BENCH, "KRON Benchmarking", "kron_reduce.py"))

EPS = [8.0, 4.0]
# UGC heterophily factor alpha, per dataset (authors' values, UGC paper Table 6).
UGC_ALPHA = {"cora": 0.19, "citeseer": 0.26, "squirrel": 0.20, "chameleon": 0.20}
# MCC (alpha,beta,gamma), per (dataset, ratio). cora/citeseer grid-searched on the
# original graph; squirrel/chameleon val-selected (see mcc_hp_select.py).
MCC_HP = {("cora", 0.3): (0.1, 100., 100.), ("cora", 0.1): (10., 100., 0.1),
          ("citeseer", 0.3): (100., 1., 0.001), ("citeseer", 0.1): (100., 100., 0.01),
          ("squirrel", 0.3): (10., 100., 0.1), ("squirrel", 0.1): (100., 100., 0.01),
          ("chameleon", 0.3): (10., 100., 0.1), ("chameleon", 0.1): (100., 1., 0.001)}
MCC_DEFAULT = (0.1, 100., 100.)   # fallback for any untuned (dataset, ratio)
RATIOS = {"mcc": [0.3, 0.1], "fgc": [0.3, 0.1, 0.05], "kron": [0.3, 0.5, 0.7], "ugc": [0.3, 0.5, 0.7]}
torch.set_num_threads(4)


def mcc_hp(ds, r):
    return MCC_HP.get((ds, r), MCC_DEFAULT)


def coarse_box(ds, method, r, Adp, X, y, C, mtr, mva, mte):
    n, d = X.shape
    Xo = torch.tensor(X); eidp = M._ei(Adp); ewdp = torch.ones(eidp.shape[1])
    if method == "mcc":
        al, be, ga = mcc_hp(ds, r)
        Cmat, Ad = M.mcc_coarsen(Adp, r, al, be, ga)
        Xc, eic, ewc, Yc, keep = M.build_coarse(Cmat, Ad, X, y, mtr, C)
        serveX, din = Xo, d
    elif method == "fgc":
        Ad = np.asarray(Adp.todense(), dtype=np.float64)
        L = np.diag(Ad.sum(1)) - Ad
        Cmat, _, _ = FGC.coarsen(L, X, int(n * r), lam=0.001, alpha=1.0, gamma=0.001, beta=0.001)
        Xc, eic, ewc, Yc, keep = M.build_coarse(Cmat, torch.tensor(Ad), X, y, mtr, C)
        serveX, din = Xo, d
    elif method == "kron":
        rd = KRON.kron_reduce(Adp, r)
        kept = rd["kept_idx"]
        Xc = Xo[kept]; Yc = torch.from_numpy(y[kept]).long()
        eic = torch.from_numpy(rd["edge_index"]).long(); ewc = torch.from_numpy(rd["edge_weight"]).float()
        keep = torch.from_numpy(mtr[kept])
        serveX, din = Xo, d
    elif method == "ugc":
        al = UGC_ALPHA.get(ds, 0.20)
        Xaug = UGC.augment(torch.tensor(X), Adp, al)
        bw = UGC.find_bin_width(Xaug, r)["bin_width"]
        assign = UGC.partition(UGC.hashed_values(Xaug), bw)
        cg = UGC.coarse_graph(assign, Xaug, Adp, y, C, mtr)
        Xc = cg["X_c"]; eic = cg["A_c_edge_index"]; ewc = cg["A_c_edge_attr"]
        Yc = cg["Y_c"]; keep = cg["keep"]
        serveX, din = Xaug, int(Xaug.shape[1])
    post, bv = M.train_box(Xc, eic, ewc, Yc, keep, serveX, eidp, ewdp, y, mva, din, C)
    acc = float((post.argmax(1)[mte] == y[mte]).mean())
    return post, acc, int(Xc.shape[0]), int(keep.sum()), int(eic.shape[1] // 2)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    ap.add_argument("--methods", nargs="+", default=["mcc", "fgc", "kron", "ugc"])
    a = ap.parse_args()
    for ds in a.datasets:
        A, X, y, C = M.load_any(ds); n = A.shape[0]
        mtr, mva, mte = M.split_6020(y, C)
        for eps in EPS:
            Adp, _, _ = dp_sparsify(A, eps)
            print(f"\n[{ds}] eps={eps}: A_dp edges={int(Adp.nnz//2)}", flush=True)
            for method in a.methods:
                for r in RATIOS[method]:
                    try:
                        post, acc, k, ksn, ce = coarse_box(ds, method, r, Adp, X, y, C, mtr, mva, mte)
                        M.save(f"{ds}_exp1{method}_{r}_e{eps}", post)
                        print(f"  {method:4s} r={r}: k={k} train_sn={ksn} coarse_edges={ce} acc={acc*100:.1f}%", flush=True)
                    except Exception as e:
                        print(f"  {method:4s} r={r}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
