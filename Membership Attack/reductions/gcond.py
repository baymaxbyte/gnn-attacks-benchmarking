"""GCond adapter: condense V_in into a small synthetic graph (Jin et al., ICLR 2022).

The model never trains on a real node -- it trains on synthetic nodes fitted by
gradient matching -- so this is the method expected to leak membership least. The
condensation runs under the GCond repo's own interpreter via _gcond_driver.py.

GCond condenses the (row-normalised) features, so at serve time the original graph's
features must be row-normalised too: serve_mode='plain_norm'. ratio is GCond's
reduction_rate relative to the members; the defaults reproduce GCond's earlier node
keep-fractions (~1.3 / 2.6 / 5.2%), i.e. its native extreme-condensation regime.
"""
import os
import subprocess
import tempfile

import numpy as np
import scipy.sparse as sp

RATIOS = [0.013, 0.026, 0.052]

HERE = os.path.dirname(os.path.abspath(__file__))
GCOND_HOME = os.environ.get(
    "GCOND_HOME",
    os.path.abspath(os.path.join(HERE, "..", "..", "..", "GCond", "GCond-main")))
_venv = os.path.join(GCOND_HOME, ".venv", "bin", "python3")
GCOND_PYTHON = os.environ.get("GCOND_PYTHON", _venv)
EPOCHS = 600
SEED = 1


def _save_vin(vin_adj, vin_feat, vin_labels, path):
    adj = sp.csr_matrix(vin_adj)
    feat = sp.csr_matrix(vin_feat)
    np.savez_compressed(
        path,
        adj_data=adj.data, adj_indices=adj.indices, adj_indptr=adj.indptr,
        adj_shape=np.array(adj.shape),
        feat_data=feat.data, feat_indices=feat.indices, feat_indptr=feat.indptr,
        feat_shape=np.array(feat.shape),
        labels=np.asarray(vin_labels, dtype=np.int64))


def reduce(dataset, vin_adj, vin_feat, vin_labels, n_classes, ratio,
           variant=None, seed=SEED):
    tmp = tempfile.mkdtemp(prefix="gcond_mia_")
    vin_npz = os.path.join(tmp, "vin.npz")
    out_npz = os.path.join(tmp, "syn.npz")
    _save_vin(vin_adj, vin_feat, vin_labels, vin_npz)

    cmd = [GCOND_PYTHON, os.path.join(HERE, "_gcond_driver.py"),
           "--vin_npz", vin_npz, "--out_npz", out_npz,
           "--reduction_rate", str(ratio), "--seed", str(seed),
           "--epochs", str(EPOCHS)]
    env = dict(os.environ, PYTHONPATH=GCOND_HOME,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    p = subprocess.run(cmd, cwd=GCOND_HOME, capture_output=True, text=True, env=env)
    if p.returncode != 0 or not os.path.exists(out_npz):
        raise SystemExit(f"GCond driver failed:\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")

    z = np.load(out_npz)
    feat_syn, adj_syn, labels_syn = z["feat_syn"], z["adj_syn"], z["labels_syn"]
    k = feat_syn.shape[0]

    # dense weighted synthetic adjacency -> edge_index + weight (drop tiny entries)
    A = adj_syn.copy()
    A[A < 1e-4] = 0.0
    src, dst = np.nonzero(A)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_weight = A[src, dst].astype(np.float32)

    return {
        "edge_index": edge_index,
        "edge_weight": edge_weight if edge_weight.size else None,
        "X": feat_syn.astype(np.float32),
        "Y": labels_syn.astype(np.int64),
        "train_mask": np.ones(k, dtype=bool),
        "k": int(k),
        "feat_dim": int(feat_syn.shape[1]),
        "serve_mode": "plain_norm",
    }
