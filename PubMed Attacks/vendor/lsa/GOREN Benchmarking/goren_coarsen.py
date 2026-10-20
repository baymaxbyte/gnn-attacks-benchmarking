"""Baseline coarsening via Loukas's graph_coarsening.

Supplies the coarse graph GOREN relearns edge weights on: the coarsening matrix C,
the hard partition, the coarse topology, baseline weights S^T A S, coarse features
(supernode averages) and coarse labels (majority over training labels).
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

import config as C

if C.COARSENING_PKG not in sys.path:
    sys.path.insert(0, C.COARSENING_PKG)


def _pygsp_graph(adj):
    """Loukas's code wants a pygsp Graph."""
    from pygsp import graphs
    W = sp.csr_matrix(adj, dtype=float)
    G = graphs.Graph(W)
    return G


def baseline_coarsen(adj, r, method=C.METHOD, K=C.K_EIGS, max_levels=10):
    """Run Loukas's scheme. Returns the coarsening matrix C, (k, n) sparse."""
    from graph_coarsening import coarsening_utils as cu
    G = _pygsp_graph(adj)
    Call, Gc, Call_list, Gall = None, None, None, None
    out = cu.coarsen(G, K=K, r=r, max_levels=max_levels, method=method)
    # coarsen() returns (C, Gc, Call, Gall)
    Cmat, Gc, Call, Gall = out
    return sp.csr_matrix(Cmat), Gc


def assignment_from_C(Cmat):
    """Hard node -> supernode map. Column v of C has support on v's supernode."""
    Cmat = sp.csr_matrix(Cmat)
    k, n = Cmat.shape
    assign = np.asarray(Cmat.argmax(axis=0)).ravel()
    return assign.astype(np.int64)


def coarse_graph(Cmat, adj, features, labels, n_classes, train_mask):
    """Coarse topology, baseline weights, coarse features and labels.

    Weights: A_c = C A C^T, off-diagonal, which is the coarse graph Loukas's scheme
    implies and therefore the `bl` variant GOREN is asked to improve on.
    Features: supernode average, C^2 X, matching coarsen_vector.
    Labels: majority over each supernode, computed from TRAINING labels only.
    """
    Cmat = sp.csr_matrix(Cmat)
    k, n = Cmat.shape
    A = sp.csr_matrix(adj, dtype=float)

    # Baseline coarse weights: S^T A S, the count of original edges between each
    # pair of supernodes.
    assign_tmp = assignment_from_C(Cmat)
    S = sp.csr_matrix((np.ones(n), (np.arange(n), assign_tmp)), shape=(n, k))
    A_c = (S.T @ A @ S).tocoo()
    keep_off = A_c.row != A_c.col                  # drop self-loops
    rows, cols, vals = A_c.row[keep_off], A_c.col[keep_off], A_c.data[keep_off]
    edge_index = np.vstack([rows, cols]).astype(np.int64)
    edge_weight = vals.astype(np.float32)

    Csq = Cmat.power(2)
    X = features.todense() if sp.issparse(features) else features
    X_c = np.asarray(Csq @ np.asarray(X, dtype=np.float32), dtype=np.float32)

    assign = assignment_from_C(Cmat)
    labels = np.asarray(labels)
    train_mask = np.asarray(train_mask, dtype=bool)
    Y_c = np.zeros(k, dtype=np.int64)
    trainable = np.zeros(k, dtype=bool)
    for s in range(k):
        members = np.where(assign == s)[0]
        tr = members[train_mask[members]]
        if len(tr):
            Y_c[s] = np.bincount(labels[tr], minlength=n_classes).argmax()
            trainable[s] = True

    sizes = np.bincount(assign, minlength=k)
    return {
        "C": Cmat, "assignment": assign, "k": int(k),
        "edge_index": edge_index, "edge_weight": edge_weight,
        "X_c": X_c, "Y_c": Y_c, "keep": trainable, "sizes": sizes,
        "coarse_edges": int(edge_index.shape[1] // 2),
        "achieved_ratio": 1.0 - k / n,
        "trainable_supernodes": int(trainable.sum()),
        "max_supernode_size": int(sizes.max()),
        "mean_supernode_size": float(sizes[sizes > 0].mean()),
        "singleton_supernodes": int((sizes == 1).sum()),
        "empty_supernodes": int((sizes == 0).sum()),
    }


def summary(cg):
    return (f"k={cg['k']:5d}  ratio={cg['achieved_ratio']:.4f}  "
            f"trainable={cg['trainable_supernodes']:5d}  "
            f"maxsz={cg['max_supernode_size']:4d}  "
            f"meansz={cg['mean_supernode_size']:5.2f}  "
            f"edges={cg['coarse_edges']:6d}")
