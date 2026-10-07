"""Condense a V_in subgraph with GCond, run under the GCond repo's interpreter.

GCond's train_gcond_transduct.py loads a whole named dataset; this driver instead
builds a deeprobust-style data object straight from a saved V_in subgraph, so GCond
condenses the members' induced graph. It imports GCond's own modules (so it must run
with GCOND_HOME as cwd / on sys.path, under that repo's venv) and writes the
condensed synthetic graph to an .npz.

    python _gcond_driver.py --vin_npz vin.npz --out_npz syn.npz --reduction_rate 0.026
"""
import argparse

import numpy as np
import scipy.sparse as sp
import torch

from utils import Transd2Ind             # from GCOND_HOME
from gcond_agent_transduct import GCond   # from GCOND_HOME


class _VinData:
    """Minimal stand-in for a deeprobust Dataset: only the attributes Transd2Ind
    reads."""

    def __init__(self, adj, features, labels, idx_train, idx_val, idx_test):
        self.adj = adj
        self.features = features
        self.labels = labels
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.idx_test = idx_test


def _row_normalize(X):
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return (X / s).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vin_npz", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--reduction_rate", type=float, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=600)
    a = ap.parse_args()

    z = np.load(a.vin_npz)
    adj = sp.csr_matrix((z["adj_data"], z["adj_indices"], z["adj_indptr"]),
                        shape=tuple(z["adj_shape"]))
    feat = np.asarray(sp.csr_matrix((z["feat_data"], z["feat_indices"],
                                     z["feat_indptr"]),
                                    shape=tuple(z["feat_shape"])).todense(),
                      dtype=np.float32)
    feat = _row_normalize(feat)                       # matches T.NormalizeFeatures
    labels = z["labels"].astype(np.int64)
    m = adj.shape[0]
    idx = np.arange(m)                                # all V_in are members

    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    data_full = _VinData(adj, feat, labels, idx, idx, idx)
    data = Transd2Ind(data_full, keep_ratio=1.0)

    args = argparse.Namespace(
        dataset="vin", reduction_rate=a.reduction_rate, nlayers=2, sgc=1,
        hidden=256, lr_feat=1e-4, lr_adj=1e-4, lr_model=0.01, weight_decay=0.0,
        dropout=0.0, normalize_features=True, keep_ratio=1.0, seed=a.seed,
        alpha=0, debug=0, inner=0, outer=20, save=0, one_step=0,
        dis_metric="ours", epochs=a.epochs)

    agent = GCond(data, args, device="cpu")
    agent.train(verbose=False)

    feat_syn = agent.feat_syn.detach().cpu().numpy().astype(np.float32)
    adj_syn = agent.pge.inference(agent.feat_syn).detach().cpu().numpy().astype(np.float32)
    labels_syn = agent.labels_syn.cpu().numpy().astype(np.int64)
    np.savez_compressed(a.out_npz, feat_syn=feat_syn, adj_syn=adj_syn,
                        labels_syn=labels_syn)
    print(f"GCOND_OK k={feat_syn.shape[0]} d={feat_syn.shape[1]} "
          f"adj={adj_syn.shape} nnz={(adj_syn > 1e-4).sum()}")


if __name__ == "__main__":
    main()
