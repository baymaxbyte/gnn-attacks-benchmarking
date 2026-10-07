"""Control: no reduction. The GNN trains on V_in itself.

This is the reference leakage level -- a plain GNN over the members' induced
subgraph, against which every reduction method is compared.
"""
import numpy as np

from . import base

RATIOS = [None]          # single cell, no reduction


def reduce(dataset, vin_adj, vin_feat, vin_labels, n_classes, ratio=None,
           variant=None, seed=0):
    X = base.dense(vin_feat)
    return {
        "edge_index": base.edge_index_from_csr(vin_adj),
        "edge_weight": None,
        "X": X,
        "Y": np.asarray(vin_labels, dtype=np.int64),
        "train_mask": np.ones(vin_adj.shape[0], dtype=bool),
        "k": int(vin_adj.shape[0]),
        "feat_dim": int(X.shape[1]),
        "serve_mode": "plain",
    }
