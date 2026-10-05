"""FGC adapter: featured graph coarsening with sparsity (Kumar et al., ICML 2023).

Optimises an assignment C, hardens it, and builds supernode features/labels/edges.
Feature dim stays d, so it serves plainly on the original graph. Edges are
unweighted (the authors' notebook thresholds then binarises, never passing weights
to the GCN). ratio is the notebook's r: nominal supernodes k = int(|V_in| * r);
many end up empty, so the effective k is lower and only non-empty supernodes carry a
training label. Hyperparameters are the notebook-driver values that reproduce the
paper's Table 4 (lam=alpha=... below).
"""
import numpy as np

from . import _fgc_solver as FG
from . import base

RATIOS = [0.3, 0.1, 0.05]          # nominal k = int(m*r) -- same labels as FGC bench
HP = dict(lam=0.001, alpha=1.0, gamma=0.001, beta=0.001)
ITERS = 50


def reduce(dataset, vin_adj, vin_feat, vin_labels, n_classes, ratio,
           variant=None, seed=0):
    m = vin_adj.shape[0]
    k = max(2, int(m * ratio))
    theta = base.dense_laplacian(vin_adj)
    Xd = base.dense(vin_feat)
    C, _, _ = FG.coarsen(theta, Xd, k, iters=ITERS, step="notebook",
                         variant="notebook", **HP)
    cg = FG.coarse_graph(FG.harden(C), theta, Xd, np.asarray(vin_labels),
                         n_classes, edge_threshold=0.1)
    sizes = np.bincount(cg["assignment"].numpy(), minlength=cg["k"])
    return {
        "edge_index": cg["edge_index"].numpy().astype(np.int64),
        "edge_weight": None,
        "X": cg["X_c"].numpy().astype(np.float32),
        "Y": cg["Y_c"].numpy().astype(np.int64),
        "train_mask": (sizes > 0),
        "k": int(cg["k"]),
        "feat_dim": int(cg["X_c"].shape[1]),
        "serve_mode": "plain",
    }
