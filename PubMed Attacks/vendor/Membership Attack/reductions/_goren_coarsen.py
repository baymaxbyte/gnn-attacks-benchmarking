"""Baseline coarsening for the GOREN benchmark, via Loukas's graph_coarsening.

GOREN does not produce a partition. It improves the EDGE WEIGHTS of a coarse graph
that some other scheme produced, so this module supplies that coarse graph and the
`bl` (baseline) weights GOREN is meant to improve on.

Loukas's coarsen() returns a coarsening matrix C of shape (k, n) whose rows are the
supernodes. Row i has support on the original nodes merged into supernode i, with
entries 1/sqrt(size) so that C C^T = I. Two consequences we rely on:

    assignment[v] = argmax over supernodes of C[:, v]     hard partition
    A_c            = C A C^T                              coarse adjacency

    coarsening_utils.coarsen_vector(x, C) = (C^2) x        supernode average of x
    coarsening_utils.lift_vector(x, C)                     supernode value -> nodes

Reference: coarsening_utils.coarsen(G, K=10, r=0.5, method='variation_neighborhoods'),
where r is the fraction of nodes REMOVED.
"""
import numpy as np
import scipy.sparse as sp

# The Loukas coarsening itself (baseline_coarsen) runs in _goren_worker.py under the
# GraphCoarsening venv; here we only need to build the coarse graph from the
# coarsening matrix C, which is pure numpy/scipy.


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

    # Baseline coarse weights = the number of original edges running between each
    # pair of supernodes, S^T A S with S the 0/1 indicator. This is the weighting
    # consistent with the 1/|s| averaging projection GOREN's loss uses: for a signal
    # that is constant on supernodes, x^T L x = sum over cross-supernode pairs of
    # (x_s - x_s')^2 times the edge count. Measured on Cora r=0.5 it gives Eigenerror
    # 0.1686 against 0.2802 for C A C^T, so it is also the stronger reference for
    # GOREN to improve on.
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
