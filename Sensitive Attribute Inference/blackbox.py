"""Black box + the two-world sensitive-attribute probe.

A uniform 3-layer GCN trains on a reduced graph (edge weights threaded) and is
served on the ORIGINAL graph. The sensitive-attribute probe serves it twice: once
with the sensitive feature column forced to 0 for every node, once forced to 1, and
returns the pair of posteriors per node.

    p0 = serve(X with column s := 0)      p1 = serve(X with column s := 1)
    Z_u = [p0_u || p1_u]                   -> fed to the attack classifier

The feature column is overwritten BEFORE any method-specific serving transform
(row-normalisation for GCond, adjacency augmentation for UGC), so the probe reaches
the model the same way the training features did.
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


def train_box(reduced_path, n_classes, seed=0, epochs=EPOCHS):
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
    model.eval()
    return model, rd


def _serve_features(dataset, X_raw, serve_mode, serve_alpha):
    """Build the model-input features for every original node from a raw (n, d)
    feature matrix, applying the cell's serving transform."""
    if serve_mode == "plain":
        Xs = X_raw
    elif serve_mode == "plain_norm":
        Xs = _row_normalize(X_raw)
    elif serve_mode == "ugc_aug":
        m = D.load_whole(dataset)
        A = np.asarray(m["adj"].todense(), dtype=np.float32)
        Xs = np.concatenate([(1.0 - serve_alpha) * X_raw, serve_alpha * A], axis=1)
    else:
        raise ValueError(f"unknown serve_mode: {serve_mode}")
    return torch.from_numpy(Xs.astype(np.float32))


@torch.no_grad()
def two_world_posteriors(model, dataset, rd, feat_idx):
    """Serve with the sensitive column forced to 0, then to 1. Returns (p0, p1),
    each (n, n_classes)."""
    m = D.load_whole(dataset)
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    ei = _edge_index(m["adj"])
    out = []
    for val in (0.0, 1.0):
        Xv = X.copy()
        Xv[:, feat_idx] = val
        Xs = _serve_features(dataset, Xv, rd["serve_mode"], rd["serve_alpha"])
        post = model(Xs, ei, None).exp()
        post = (post / post.sum(1, keepdim=True)).numpy().astype(np.float32)
        out.append(post)
    return out[0], out[1]


@torch.no_grad()
def serve_true(model, dataset, rd):
    """Posteriors with the node's TRUE features (no toggle), for a box-accuracy
    sanity check. Returns (posteriors (n, C), labels)."""
    m = D.load_whole(dataset)
    X = np.asarray(m["features"].todense(), dtype=np.float32)
    Xs = _serve_features(dataset, X, rd["serve_mode"], rd["serve_alpha"])
    post = model(Xs, _edge_index(m["adj"]), None).exp()
    post = (post / post.sum(1, keepdim=True)).numpy()
    return post, m["labels"]
