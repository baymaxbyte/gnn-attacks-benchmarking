"""Loukas coarsening of an arbitrary adjacency, run under the GraphCoarsening venv.

Same as the GOREN benchmark's worker, but reads the adjacency from a saved sparse
.npz (the V_in subgraph) instead of a Planetoid dataset, so it can coarsen the
members' induced graph. Writes the coarsening matrix C to an .npz. Needs pygsp and
the graph_coarsening package, so it runs under COARSEN_PYTHON.

    python _goren_worker.py --adj_npz vin.npz --r 0.5 --out C.npz --goren_home ...
"""
import argparse
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adj_npz", required=True)
    ap.add_argument("--r", type=float, required=True)
    ap.add_argument("--out", required=True)
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

    A = sp.load_npz(args.adj_npz)
    A = sp.csr_matrix(A, dtype=float)
    n = A.shape[0]
    G = graphs.Graph(A)

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        out = cu.coarsen(G, K=args.K, r=args.r, max_levels=args.max_levels,
                         method=args.method)
    Cmat = sp.csr_matrix(out[0])
    secs = time.time() - t0

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    sp.save_npz(args.out, Cmat)
    print(f"COARSEN_OK n={n} k={Cmat.shape[0]} "
          f"achieved_ratio={1 - Cmat.shape[0] / n:.6f} seconds={secs:.2f}")


if __name__ == "__main__":
    main()
