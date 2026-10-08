"""Shared helpers and the standard reduced-graph contract.

Every adapter's reduce(...) returns a dict with these numpy fields, which
step02_reduce.py saves and blackbox.py consumes:

    edge_index   (2, 2E) int64, local indexing over the reduced nodes
    edge_weight  (2E,) float32, or None for unit weights
    X            (k, feat_dim) float32, features the GNN trains on
    Y            (k,) int64, labels
    train_mask   (k,) bool, reduced nodes that carry a training label
    k            int
    feat_dim     int (d for most methods; d + |V_in| for UGC's augmentation)
    serve_mode   "plain"   -> serve on the original graph with original features
                 "ugc_aug" -> serve with UGC augmentation to the V_in width
    serve_alpha  float, UGC only (the augmentation mixing weight)

A single uniform classifier (3-layer GCN threading edge_weight) trains on all of
these, so any membership-leakage difference is attributable to the reduction rather
than to the architecture.
"""
import numpy as np
import scipy.sparse as sp


def edge_index_from_csr(adj):
    A = sp.csr_matrix(adj).tocoo()
    return np.vstack([A.row, A.col]).astype(np.int64)


def dense(x):
    return np.asarray(x.todense() if sp.issparse(x) else x, dtype=np.float32)


def dense_laplacian(adj):
    A = dense(adj)
    return (np.diag(A.sum(1)) - A).astype(np.float32)
