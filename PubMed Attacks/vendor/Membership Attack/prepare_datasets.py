"""Assemble the WikipediaNetwork heterophily graphs (chameleon, squirrel) into the
benchmark's whole-graph npz format, matching cora.npz / citeseer.npz.

Uses PyG WikipediaNetwork(geom_gcn_preprocess=True) -- the same source/preprocessing
lsa_unified used for chameleon (2277 nodes, 2325 feats, 5 classes). Writes <ds>.npz
next to this script (adj/feat as CSR, labels, n, n_classes, n_edges), which
data.load_whole reads. cora/citeseer are left untouched.

    python prepare_datasets.py                 # chameleon + squirrel
    python prepare_datasets.py squirrel
"""
import os
import sys

import numpy as np
import scipy.sparse as sp

HERE = os.path.dirname(os.path.abspath(__file__))

PYG_NAME = {"chameleon": "chameleon", "squirrel": "squirrel"}


def assemble(ds):
    """Build the (adj, feat, labels) representation from PyG (same as lsa_unified)."""
    from torch_geometric.datasets import WikipediaNetwork
    from torch_geometric.utils import to_scipy_sparse_matrix
    root = os.path.join(HERE, "_pyg_cache", ds)
    g = WikipediaNetwork(root=root, name=PYG_NAME[ds], geom_gcn_preprocess=True)[0]
    n = int(g.num_nodes)
    adj = to_scipy_sparse_matrix(g.edge_index, num_nodes=n).tocsr()
    adj = adj.maximum(adj.T)          # symmetric
    adj.setdiag(0)                    # no self-loops
    adj.eliminate_zeros()
    adj.data[:] = 1.0                 # binary
    adj = adj.tocsr()
    feat = sp.csr_matrix(np.asarray(g.x.numpy(), dtype=np.float32))
    labels = np.asarray(g.y.numpy(), dtype=np.int64).ravel()
    return adj, feat, labels, n


def build(ds):
    adj, feat, labels, n = assemble(ds)
    path = os.path.join(HERE, f"{ds}.npz")
    np.savez_compressed(
        path,
        adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),
        feat_data=feat.data, feat_indices=feat.indices, feat_indptr=feat.indptr,
        feat_shape=np.array(feat.shape),
        labels=labels, n=np.array([n]),
        n_classes=np.array([int(labels.max() + 1)]),
        n_edges=np.array([int(adj.nnz // 2)]))
    print(f"wrote {ds}.npz  n={n}  d={feat.shape[1]}  "
          f"classes={int(labels.max() + 1)}  edges={int(adj.nnz // 2)}")


def main():
    which = sys.argv[1:] or ["chameleon", "squirrel"]
    for ds in which:
        build(ds)


if __name__ == "__main__":
    main()
