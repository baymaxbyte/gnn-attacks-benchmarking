#!/usr/bin/env python3
"""EXP3 GOREN: feature-KNN(k=10) graph -> Loukas variation_neighbourhoods coarsening
(run in the GraphCoarsening pygsp venv via exp1's goren_dp_worker.py) -> build coarse
graph -> train the GCN, serve on the KNN graph. Posteriors -> {ds}_exp3goren_{r}."""
import os, sys, subprocess, tempfile
import numpy as np
import scipy.sparse as sp
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DPS = os.path.dirname(HERE)
EXP1 = os.path.join(DPS, "exp1")
GRIDROOT = os.path.dirname(DPS)
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC); import mcc_boxes as M
sys.path.insert(0, HERE); from knn_boxes import knn_graph
sys.path.insert(0, EXP1)
import goren_boxes as GB            # reuse coarse_from_C, GOREN_HOME, COARSEN_PY

RATIOS = [0.3, 0.5, 0.7]
torch.set_num_threads(4)


def main():
    tmp = tempfile.mkdtemp(prefix="goren_knn_")
    for ds in ["cora", "citeseer"]:
        A, X, y, C = M.load_any(ds); n, d = X.shape
        mtr, mva, mte = M.split_6020(y, C)
        Aknn = knn_graph(X)
        Xo = torch.tensor(X); eik = M._ei(Aknn); ewk = torch.ones(eik.shape[1])
        adjp = os.path.join(tmp, f"{ds}_knn.npz"); sp.save_npz(adjp, Aknn)
        print(f"\n[{ds}] KNN base edges={int(Aknn.nnz//2)}", flush=True)
        for r in RATIOS:
            cp = os.path.join(tmp, f"{ds}_C_{r}.npz")
            res = subprocess.run([GB.COARSEN_PY, os.path.join(EXP1, "goren_dp_worker.py"),
                                  "--adj", adjp, "--r", str(r), "--out", cp, "--goren_home", GB.GOREN_HOME],
                                 capture_output=True, text=True)
            if res.returncode or not os.path.exists(cp):
                print(f"  goren r={r}: COARSEN FAILED {res.stderr[-300:]}", flush=True); continue
            Cmat = sp.load_npz(cp)
            Xc, ei, ew, Yc, keep = GB.coarse_from_C(Cmat, Aknn, X, y, mtr, C)
            post, bv = M.train_box(torch.tensor(Xc), torch.from_numpy(ei).long(),
                                   torch.from_numpy(ew).float(), torch.from_numpy(Yc).long(),
                                   torch.from_numpy(keep), Xo, eik, ewk, y, mva, d, C)
            acc = float((post.argmax(1)[mte] == y[mte]).mean())
            M.save(f"{ds}_exp3goren_{r}", post)
            print(f"  goren r={r}: k={Cmat.shape[0]} train_sn={int(keep.sum())} acc={acc*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
