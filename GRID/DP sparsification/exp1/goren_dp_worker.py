#!/usr/bin/env python3
"""GOREN coarsening worker -- runs in the GraphCoarsening venv (pygsp + Loukas's
graph_coarsening + sparsenet). Loads a custom adjacency (my DP-sparsified A_dp,
saved as .npz), runs Loukas's variation_neighborhoods coarsening, writes C.npz.

    <venv python> goren_dp_worker.py --adj A_dp.npz --r 0.5 --out C.npz --goren_home <dir>
"""
import argparse, os, sys, time, warnings
import numpy as np
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj", required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--goren_home", required=True)
    ap.add_argument("--method", default="variation_neighborhoods")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max_levels", type=int, default=10)
    a = ap.parse_args()

    sys.path.insert(0, a.goren_home)
    sys.path.insert(0, os.path.join(a.goren_home, "sparsenet", "evaluation", "graph-coarsening"))
    from graph_coarsening import coarsening_utils as cu
    from pygsp import graphs

    A = sp.load_npz(a.adj)
    n = A.shape[0]
    G = graphs.Graph(sp.csr_matrix(A, dtype=float))
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = cu.coarsen(G, K=a.K, r=a.r, max_levels=a.max_levels, method=a.method)
    Cmat = sp.csr_matrix(out[0])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    sp.save_npz(a.out, Cmat)
    print(f"COARSEN_OK n={n} k={Cmat.shape[0]} achieved_ratio={1 - Cmat.shape[0]/n:.4f} secs={time.time()-t0:.1f}")


if __name__ == "__main__":
    main()
