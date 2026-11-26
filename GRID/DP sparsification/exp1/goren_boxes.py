#!/usr/bin/env python3
"""EXP1 GOREN: edge-DP sparsify -> Loukas variation_neighborhoods coarsening (run in
the GraphCoarsening pygsp venv via goren_dp_worker.py) -> build coarse graph (C A C^T
topology, C^2 X features, train-majority labels) -> train the consistent 3-layer GCN,
serve on A_dp. GOREN's baseline coarse graph (we don't run the neural reweighting --
it only changes edge weights and the paper never does node classification).

Posteriors -> attack/data/pred/{ds}_exp1goren_{r}_e{eps}_pred.pkl.
"""
import os, sys, subprocess, tempfile
import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(os.path.dirname(HERE))
BENCH = os.path.dirname(os.path.dirname(GRIDROOT))
ROOT = os.path.dirname(BENCH)
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC); import mcc_boxes as M
sys.path.insert(0, HERE)
from dp_boxes import dp_sparsify

GOREN_HOME = os.path.join(ROOT, "GraphCoarsening", "GraphCoarsening-main")
COARSEN_PY = os.path.join(GOREN_HOME, ".venv", "bin", "python3")
RATIOS = [0.3, 0.5, 0.7]        # fraction removed (Loukas convention)
EPS = [8.0, 4.0]
torch.set_num_threads(4)


def coarse_from_C(Cmat, A, X, y, mtr, nclasses):
    """Replicate goren_coarsen.coarse_graph (numpy/scipy only; no pygsp)."""
    Cmat = sp.csr_matrix(Cmat); k, n = Cmat.shape
    A = sp.csr_matrix(A, dtype=float)
    assign = np.asarray(Cmat.argmax(axis=0)).ravel()
    S = sp.csr_matrix((np.ones(n), (np.arange(n), assign)), shape=(n, k))
    A_c = (S.T @ A @ S).tocoo()
    off = A_c.row != A_c.col
    ei = np.vstack([A_c.row[off], A_c.col[off]]).astype(np.int64)
    ew = A_c.data[off].astype(np.float32)
    Csq = Cmat.power(2)
    Xc = np.asarray(Csq @ np.asarray(X, dtype=np.float32), dtype=np.float32)
    Yc = np.zeros(k, np.int64); keep = np.zeros(k, bool)
    for s in range(k):
        mem = np.where(assign == s)[0]; tr = mem[mtr[mem]]
        if len(tr):
            Yc[s] = np.bincount(y[tr], minlength=nclasses).argmax(); keep[s] = True
    return Xc, ei, ew, Yc, keep


def main():
    tmp = tempfile.mkdtemp(prefix="goren_dp_")
    for ds in ["cora", "citeseer"]:
        A, X, y, C = M.load(ds); n, d = X.shape
        mtr, mva, mte = M.split_6020(y, C)
        Xo = torch.tensor(X)
        for eps in EPS:
            Adp, _, _ = dp_sparsify(A, eps)
            adjp = os.path.join(tmp, f"{ds}_adp_e{eps}.npz"); sp.save_npz(adjp, Adp)
            eidp = M._ei(Adp); ewdp = torch.ones(eidp.shape[1])
            for r in RATIOS:
                cp = os.path.join(tmp, f"{ds}_C_{r}_e{eps}.npz")
                res = subprocess.run([COARSEN_PY, os.path.join(HERE, "goren_dp_worker.py"),
                                      "--adj", adjp, "--r", str(r), "--out", cp,
                                      "--goren_home", GOREN_HOME],
                                     capture_output=True, text=True)
                if res.returncode or not os.path.exists(cp):
                    print(f"[{ds}] goren r={r} eps={eps} COARSEN FAILED: {res.stderr[-400:]}", flush=True)
                    continue
                Cmat = sp.load_npz(cp)
                Xc, ei, ew, Yc, keep = coarse_from_C(Cmat, Adp, X, y, mtr, C)
                post, bv = M.train_box(torch.tensor(Xc), torch.from_numpy(ei).long(),
                                       torch.from_numpy(ew).float(), torch.from_numpy(Yc).long(),
                                       torch.from_numpy(keep), Xo, eidp, ewdp, y, mva, d, C)
                acc = float((post.argmax(1)[mte] == y[mte]).mean())
                M.save(f"{ds}_exp1goren_{r}_e{eps}", post)
                print(f"[{ds}] goren r={r} eps={eps}: k={Cmat.shape[0]} train_sn={int(keep.sum())} "
                      f"acc={acc*100:.1f}%  [{res.stdout.strip()}]", flush=True)


if __name__ == "__main__":
    main()
