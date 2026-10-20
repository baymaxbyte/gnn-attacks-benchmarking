"""KRON adapter: Kron-reduce V_in by Schur complement (arXiv 1102.2950).

Keeps a subset of real V_in nodes (the high-degree backbone), rewired by
Schur-complement conductances. Feature dim stays d, so it serves plainly on the
original graph. ratio = fraction of V_in nodes removed.
"""
import numpy as np

from . import _kron_solver as K
from . import base

RATIOS = [0.3, 0.5, 0.7]          # fraction removed, keep 70/50/30% -- as in KRON bench


def reduce(dataset, vin_adj, vin_feat, vin_labels, n_classes, ratio,
           variant=None, seed=0):
    rd = K.kron_reduce(vin_adj, ratio, selection="degree")
    kept = rd["kept_idx"]
    X = base.dense(vin_feat)[kept]
    return {
        "edge_index": rd["edge_index"].astype(np.int64),
        "edge_weight": rd["edge_weight"].astype(np.float32),
        "X": X,
        "Y": np.asarray(vin_labels, dtype=np.int64)[kept],
        "train_mask": np.ones(len(kept), dtype=bool),
        "k": int(len(kept)),
        "feat_dim": int(X.shape[1]),
        "serve_mode": "plain",
    }
