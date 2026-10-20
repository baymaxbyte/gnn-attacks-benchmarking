"""The uniform black box: one 3-layer GCN trains on any reduced graph and is served
on the ORIGINAL whole graph, so leakage differences come from the reduction, not the
architecture. edge_weight is threaded through every layer. serve_mode picks the query
features per cell: 'plain' (original), 'plain_norm' (row-normalised, GCond),
'ugc_aug' ([(1-a)X | a*A[:,V_in]], UGC). Served posteriors cover all n nodes and are
split into members (V_in) and non-members (V_out).
"""
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn.functional as Fn
from torch_geometric.nn import GCNConv

import data as D

HIDDEN = 64
DROPOUT = 0.5
LR = 0.003
WEIGHT_DECAY = 0.0005
EPOCHS = 500


class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden)
        self.conv2 = GCNConv(hidden, 64)
        self.conv3 = GCNConv(64, num_classes)

    def forward(self, x, edge_index, edge_weight):
        x = Fn.relu(self.conv1(x, edge_index, edge_weight))
        x = Fn.dropout(x, p=DROPOUT, training=self.training)
        x = Fn.relu(self.conv2(x, edge_index, edge_weight))
        x = Fn.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv3(x, edge_index, edge_weight)
        return Fn.log_softmax(x, dim=1)


def _edge_index(adj):
    A = sp.csr_matrix(adj).tocoo()
    return torch.from_numpy(np.vstack([A.row, A.col])).long()


def _row_normalize(X):
    s = X.sum(1, keepdims=True)
    s[s == 0] = 1.0
    return X / s


def load_reduced(path):
    z = np.load(path, allow_pickle=False)
    ew = None if int(z["weighted"][0]) == 0 else torch.from_numpy(
        z["edge_weight"].astype(np.float32))
    return {
        "edge_index": torch.from_numpy(z["edge_index"]).long(),
        "edge_weight": ew,
        "X": torch.from_numpy(z["X"].astype(np.float32)),
        "Y": torch.from_numpy(z["Y"].astype(np.int64)),
        "train_mask": torch.from_numpy(np.asarray(z["train_mask"], bool)),
        "feat_dim": int(z["feat_dim"][0]),
        "serve_mode": str(z["serve_mode"]),
        "serve_alpha": float(z["serve_alpha"][0]),
    }


def _serve_features(dataset, serve_mode, serve_alpha, member_ids, n_classes):
    """Query features for every ORIGINAL node, matching how the cell was trained."""
    m = D.load_whole(dataset)
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    if serve_mode == "plain":
        Xs = X
    elif serve_mode == "plain_norm":
        Xs = _row_normalize(X)
    elif serve_mode == "ugc_aug":
        A = np.asarray(m["adj"].todense(), dtype=np.float32)
        Xs = np.concatenate([(1.0 - serve_alpha) * X,
                             serve_alpha * A[:, member_ids]], axis=1)
    else:
        raise ValueError(f"unknown serve_mode: {serve_mode}")
    return (torch.from_numpy(Xs.astype(np.float32)),
            _edge_index(m["adj"]), m["labels"], m["n"], m["n_classes"])


def train_and_serve(reduced_path, dataset, member_ids, n_classes, seed=0,
                    epochs=EPOCHS):
    """Train one box on the reduced graph, serve posteriors on the original graph."""
    rd = load_reduced(reduced_path)

    torch.manual_seed(seed)
    model = GCN(rd["feat_dim"], HIDDEN, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    x, ei, ew, y, mask = (rd["X"], rd["edge_index"], rd["edge_weight"],
                          rd["Y"], rd["train_mask"])
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        Fn.nll_loss(model(x, ei, ew)[mask], y[mask]).backward()
        opt.step()

    Xs, ei_o, labels, n, ncls = _serve_features(
        dataset, rd["serve_mode"], rd["serve_alpha"], member_ids, n_classes)
    model.eval()
    with torch.no_grad():
        post = model(Xs, ei_o, None).exp()
        post = (post / post.sum(1, keepdim=True)).numpy().astype(np.float32)

    train_acc = float((post[member_ids].argmax(1) == labels[member_ids]).mean())
    return post, {"train_acc": train_acc, "serve_mode": rd["serve_mode"],
                  "feat_dim": rd["feat_dim"], "k_train": int(mask.sum())}
