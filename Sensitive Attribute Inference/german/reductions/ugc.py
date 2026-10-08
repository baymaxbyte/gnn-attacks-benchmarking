"""UGC adapter: LSH coarsening of V_in on augmented features (NeurIPS 2024).

UGC augments features with the adjacency, X_aug = [(1-alpha)X | alpha*A], so the
feature dimension is d + |V_in|. That is what the model trains on, and it must be
what the model is served, so serve_mode='ugc_aug': at query time every original
node is augmented with its adjacency TO THE V_in members (same d + |V_in| width),
which keeps V_out out of training. alpha is the paper's per-dataset value.
ratio = fraction of V_in nodes removed (target for the bin-width search).
"""
import numpy as np
import torch

from . import _ugc_solver as U
from . import base

RATIOS = [0.3, 0.5, 0.7]          # fraction removed, keep 70/50/30% -- as in UGC bench
# cora/citeseer are the UGC paper's tuned values; german has no paper value, so we use
# a mid-range default (the augmentation mixing weight for X_aug = [(1-a)X | a*A]).
ALPHA = {"cora": 0.19, "citeseer": 0.26, "german": 0.2}


def reduce(dataset, vin_adj, vin_feat, vin_labels, n_classes, ratio,
           variant=None, seed=0):
    alpha = ALPHA[dataset]
    X_aug = U.augment(vin_feat, vin_adj, alpha)                 # (m, d + m)
    bw = U.find_bin_width(X_aug, ratio, seed=seed)["bin_width"]
    assign = U.partition(U.hashed_values(X_aug, seed=seed), bw, seed=seed)
    train_mask_vin = np.ones(vin_adj.shape[0], dtype=bool)      # all V_in are members
    cg = U.coarse_graph(assign, X_aug, vin_adj, vin_labels, n_classes,
                        train_mask_vin)
    return {
        "edge_index": cg["A_c_edge_index"].numpy().astype(np.int64),
        "edge_weight": cg["A_c_edge_attr"].numpy().astype(np.float32),
        "X": cg["X_c"].numpy().astype(np.float32),
        "Y": cg["Y_c"].numpy().astype(np.int64),
        "train_mask": cg["keep"].numpy().astype(bool),
        "k": int(cg["k"]),
        "feat_dim": int(cg["X_c"].shape[1]),
        "serve_mode": "ugc_aug",
        "serve_alpha": float(alpha),
    }
