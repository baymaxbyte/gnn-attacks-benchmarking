"""Load the whole graphs and the membership split artifacts.

The whole graphs ({ds}.npz) are the full assembled Cora/Citeseer: all nodes, the
full symmetric adjacency (self-loops removed), full features and labels. Node ids
match the five link-stealing benchmarks, so members/non-members line up with them.
"""
import os

import numpy as np
import scipy.sparse as sp

import config as C


def _csr(z, prefix):
    return sp.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"],
                          z[f"{prefix}_indptr"]), shape=tuple(z[f"{prefix}_shape"]))


def load_whole(dataset):
    """Full graph: adj (csr, symmetric, no self-loops), features (csr), labels."""
    if dataset not in C.DATASETS:
        raise ValueError(f"unsupported dataset: {dataset}")
    z = np.load(os.path.join(C.HERE, f"{dataset}.npz"))
    adj = _csr(z, "adj")
    feat = _csr(z, "feat")
    labels = z["labels"].astype(np.int64)
    return {
        "dataset": dataset, "adj": adj, "features": feat, "labels": labels,
        "n": int(z["n"][0]), "d": feat.shape[1],
        "n_classes": int(z["n_classes"][0]), "n_edges": int(adj.nnz // 2),
    }


def summary(dataset):
    m = load_whole(dataset)
    return (f"{m['dataset']:9s} n={m['n']:5d}  d={m['d']:5d}  "
            f"classes={m['n_classes']}  edges={m['n_edges']:5d}")


if __name__ == "__main__":
    for ds in C.DATASETS:
        print(summary(ds))
