"""GOREN adapter: Loukas coarsening of V_in, edge weights as counts (bl) or
relearned by GOREN's GIN against a spectral objective (goren).

The Loukas coarsening runs under the GraphCoarsening venv via _goren_worker.py
(needs pygsp); everything after that is numpy/torch. Feature dim stays d
(supernode averages), so it serves plainly on the original graph.

variant "bl"    coarse edge weights = S^T A S (original edge counts)
variant "goren" the same topology, weights predicted by GOREN's GIN
ratio = fraction of V_in nodes removed.
"""
import os
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sp

from . import _goren_coarsen as GC
from . import _goren_weights as GW
from . import base

RATIOS = [0.3, 0.5, 0.7]
VARIANTS = ["bl", "goren"]

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPHCOARSENING_HOME = os.environ.get(
    "GRAPHCOARSENING_HOME",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "GraphCoarsening",
                                 "GraphCoarsening-main")))
_venv = os.path.join(GRAPHCOARSENING_HOME, ".venv", "bin", "python3")
COARSEN_PYTHON = os.environ.get("GOREN_COARSEN_PYTHON", _venv)

K_EIGS = 10
WEIGHT_EPOCHS = 400
N_BOTTOMK = 40


def _loukas_C(vin_adj, ratio, seed=0):
    """Run the Loukas coarsening worker on V_in; return the coarsening matrix C."""
    tmp = tempfile.mkdtemp(prefix="goren_mia_")
    adj_npz = os.path.join(tmp, "vin_adj.npz")
    out_npz = os.path.join(tmp, "C.npz")
    sp.save_npz(adj_npz, sp.csr_matrix(vin_adj, dtype=float))
    cmd = [COARSEN_PYTHON, os.path.join(HERE, "_goren_worker.py"),
           "--adj_npz", adj_npz, "--r", str(ratio), "--out", out_npz,
           "--goren_home", GRAPHCOARSENING_HOME, "--K", str(K_EIGS)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"GOREN coarsening worker failed:\n{p.stdout[-1500:]}\n"
                         f"{p.stderr[-1500:]}")
    return sp.load_npz(out_npz)


def reduce(dataset, vin_adj, vin_feat, vin_labels, n_classes, ratio,
           variant="bl", seed=0):
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    vin_adj = sp.csr_matrix(vin_adj, dtype=float)
    Cmat = _loukas_C(vin_adj, ratio, seed)
    train_mask_vin = np.ones(vin_adj.shape[0], dtype=bool)   # all V_in are members
    cg = GC.coarse_graph(Cmat, vin_adj, vin_feat, np.asarray(vin_labels),
                         n_classes, train_mask_vin)

    edge_weight = cg["edge_weight"].astype(np.float32)
    if variant == "goren":
        res = GW.train_weights(vin_adj, cg["edge_index"], cg["edge_weight"],
                               cg["assignment"], cg["k"], epochs=WEIGHT_EPOCHS,
                               n_bottomk=N_BOTTOMK, seed=seed)
        edge_weight = res["weight"].astype(np.float32)

    return {
        "edge_index": cg["edge_index"].astype(np.int64),
        "edge_weight": edge_weight,
        "X": cg["X_c"].astype(np.float32),
        "Y": cg["Y_c"].astype(np.int64),
        "train_mask": cg["keep"].astype(bool),
        "k": int(cg["k"]),
        "feat_dim": int(cg["X_c"].shape[1]),
        "serve_mode": "plain",
    }
