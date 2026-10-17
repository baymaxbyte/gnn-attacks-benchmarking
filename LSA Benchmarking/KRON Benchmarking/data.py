"""Dataset loading for the KRON leakage benchmark.

Reads the Planetoid pickles the same way the attack code does, so node ids match the
ids the attack datasets are built on -- which matters here because the reduced graph
keeps a subset of these original ids unchanged. Returns adj (binary symmetric csr,
self-loops removed), features (raw binary csr), and integer labels.
"""
import os
import pickle as pkl

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


def summary(dataset):
    m = load(dataset)
    return (f"{m['dataset']:9s} n={m['n']:5d}  d={m['d']:5d}  classes={m['n_classes']}  "
            f"edges={m['n_edges']:5d}  self_loops_removed={m['n_self_loops_removed']:4d}  "
            f"isolated={m['n_isolated']:3d}")


# ---------------------------------------------------------------- splits
def make_split(labels, n_classes, seed=0, scheme="paper"):
    """Class-balanced train/val/test masks over ALL nodes.

    scheme="paper"  60/20/20, the same split used by the UGC and GOREN benchmarks,
                    so the un-reduced control here is comparable to theirs.
    """
    labels = np.asarray(labels)
    n = labels.shape[0]
    rng = np.random.RandomState(seed)

    train_idx, val_idx, test_idx = [], [], []
    if scheme == "paper":
        for c in range(n_classes):
            idx = np.where(labels == c)[0]
            idx = idx[rng.permutation(idx.shape[0])]
            m = idx.shape[0]
            n_tr, n_va = int(round(0.6 * m)), int(round(0.2 * m))
            train_idx.append(idx[:n_tr])
            val_idx.append(idx[n_tr:n_tr + n_va])
            test_idx.append(idx[n_tr + n_va:])
    else:
        raise ValueError(f"unknown split scheme: {scheme}")

    def mask(parts):
        m = np.zeros(n, dtype=bool)
        m[np.concatenate(parts)] = True
        return m

    return {"train_mask": mask(train_idx), "val_mask": mask(val_idx),
            "test_mask": mask(test_idx), "scheme": scheme, "seed": seed}


if __name__ == "__main__":
    for ds in C.DATASETS:
        print(summary(ds))
