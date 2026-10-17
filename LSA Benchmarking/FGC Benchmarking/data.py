"""Dataset loading for the FGC leakage benchmark.

Reads the Planetoid pickles with the same procedure the attack code uses, so node
ids here are the same node ids the attack datasets are built on.

Returned quantities:
    adj       binary symmetric csr, SELF-LOOPS REMOVED
    features  raw binary csr (no normalisation -- FGC's notebook passes
              Planetoid's x straight through)
    labels    integer vector, argmax of the one-hot rows
"""
import os
import pickle as pkl
import sys

import networkx as nx
import numpy as np
import scipy.sparse as sp

import config as C

_NAMES = ["x", "y", "tx", "ty", "allx", "ally", "graph"]


def _parse_index_file(path):
    return [int(line.strip()) for line in open(path)]


def load(dataset, datapath=None):
    """Load one dataset. Returns a dict."""
    if dataset not in C.DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    datapath = datapath or C.DATA

    objects = []
    for nm in _NAMES:
        with open(os.path.join(datapath, f"ind.{dataset}.{nm}"), "rb") as f:
            objects.append(pkl.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objects

    test_idx_reorder = _parse_index_file(os.path.join(datapath, f"ind.{dataset}.test.index"))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == "citeseer":
        # citeseer's test block has gaps (isolated nodes); pad with zero rows
        full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1]))
        tx_ext[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_ext
        ty_ext = np.zeros((len(full), y.shape[1]))
        ty_ext[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_ext

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = sp.csr_matrix(features, dtype=np.float32)

    adj = sp.csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)), dtype=np.float32)

    labels_onehot = np.vstack((ally, ty))
    labels_onehot[test_idx_reorder, :] = labels_onehot[test_idx_range, :]
    labels = labels_onehot.argmax(1).astype(np.int64)

    # binarise, symmetrise, drop self-loops
    adj.data[:] = 1.0
    adj = adj.maximum(adj.T)
    n_self_loops = int(adj.diagonal().sum())
    adj = adj.tolil()
    adj.setdiag(0)
    adj = sp.csr_matrix(adj)
    adj.eliminate_zeros()

    n = adj.shape[0]
    degree = np.asarray(adj.sum(1)).ravel()

    return {
        "dataset": dataset,
        "adj": adj,
        "features": features,
        "labels": labels,
        "n": n,
        "d": features.shape[1],
        "n_classes": int(labels_onehot.shape[1]),
        "n_edges": int(adj.nnz // 2),
        "n_self_loops_removed": n_self_loops,
        "n_isolated": int((degree == 0).sum()),
    }


def laplacian(adj):
    """Combinatorial Laplacian theta = diag(A @ 1) - A, dense float32.

    FGC's notebook builds it as get_laplacian(adj) = diag(adj @ ones) - adj on a
    binary adjacency. Self-loops cancel in this expression, but they are removed
    upstream anyway so the two agree.
    """
    A = np.asarray(adj.todense(), dtype=np.float32)
    return np.diag(A.sum(1)) - A


def summary(dataset):
    m = load(dataset)
    return (f"{m['dataset']:9s} n={m['n']:5d}  d={m['d']:5d}  classes={m['n_classes']}  "
            f"edges={m['n_edges']:5d}  self_loops_removed={m['n_self_loops_removed']:4d}  "
            f"isolated={m['n_isolated']:3d}")


if __name__ == "__main__":
    for ds in C.DATASETS:
        print(summary(ds))
