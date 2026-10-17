"""Runs Loukas's coarsening and writes the coarsening matrix C to an .npz.

The only stage needing pygsp and the GOREN repository's `sparsenet` package, so it
runs under COARSEN_PYTHON while every other stage runs under PYTHON.

    python _coarsen_worker.py --dataset cora --r 0.3 --out C.npz
"""
import argparse
import os
import pickle as pkl
import sys
import time
import warnings

import networkx as nx
import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))


def load_adjacency(datapath, dataset):
    """Adjacency only, built the same way data.py does: binary, symmetric, no loops."""
    with open(os.path.join(datapath, f"ind.{dataset}.graph"), "rb") as f:
        graph = pkl.load(f, encoding="latin1")
    A = sp.csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)), dtype=float)
    A.data[:] = 1.0
    A = A.maximum(A.T)
    A = A.tolil()
    A.setdiag(0)
    A = sp.csr_matrix(A)
    A.eliminate_zeros()
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--datapath", required=True)
    ap.add_argument("--goren_home", required=True)
    ap.add_argument("--method", default="variation_neighborhoods")
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--max_levels", type=int, default=10)
    args = ap.parse_args()

    sys.path.insert(0, args.goren_home)
    sys.path.insert(0, os.path.join(args.goren_home, "sparsenet", "evaluation",
                                    "graph-coarsening"))
    from graph_coarsening import coarsening_utils as cu
    from pygsp import graphs

    A = load_adjacency(args.datapath, args.dataset)
    n = A.shape[0]
    G = graphs.Graph(sp.csr_matrix(A))

    # Cora and Citeseer are disconnected, so the variation cost divides by (nc - 1)
    # on singleton subgraphs. The coarsening is unaffected; silence the warnings.
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = cu.coarsen(G, K=args.K, r=args.r, max_levels=args.max_levels,
                         method=args.method)
    Cmat = sp.csr_matrix(out[0])
    secs = time.time() - t0

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    sp.save_npz(args.out, Cmat)
    assign = np.asarray(Cmat.argmax(axis=0)).ravel()
    sizes = np.bincount(assign, minlength=Cmat.shape[0])
    print(f"COARSEN_OK dataset={args.dataset} r={args.r} n={n} k={Cmat.shape[0]} "
          f"achieved_ratio={1 - Cmat.shape[0] / n:.6f} "
          f"used={int((sizes > 0).sum())} max_size={int(sizes.max())} "
          f"orthonormal={bool(np.allclose((Cmat @ Cmat.T).diagonal(), 1.0))} "
          f"seconds={secs:.2f}")


if __name__ == "__main__":
    main()
