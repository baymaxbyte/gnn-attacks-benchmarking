"""Paper-faithful target model: the canonical 2-layer Kipf GCN ("as described
in [16]") trained on the WHOLE cora/citeseer graph with the standard public
Planetoid split and ordinary ~200-epoch training (early-stopped on val loss) --
i.e. points (1) small 2-layer arch and (3) no 500-epoch best-accuracy
over-saturation. Serves softmax posteriors over all nodes.

Contrast with the benchmark's `whole` box: 3 layers, 64 hidden, 500 epochs,
best-validation-accuracy checkpoint -> near one-hot. This one should be softer
(adjacent cosine ~0.9 like the paper), giving GRID room to work.
"""
import os, sys, pickle
import numpy as np
import scipy.sparse as sp
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
sys.path.insert(0, ATTACK)
from lsa.data import preprocess_adj, preprocess_features  # same preprocessing as the attack
DATAPATH = os.path.join(ATTACK, "data", "dataset", "original") + os.sep
PRED = os.path.join(ATTACK, "data", "pred")

HIDDEN, DROPOUT, LR, WD, EPOCHS, PATIENCE = 16, 0.5, 0.01, 5e-4, 200, 20


def load_planetoid_public(ds):
    """Mirror lsa.data.load_data_original's ordering, but return the PUBLIC split."""
    names = ["x", "y", "tx", "ty", "allx", "ally", "graph"]
    objs = []
    for n in names:
        with open(f"{DATAPATH}ind.{ds}.{n}", "rb") as f:
            objs.append(pickle.load(f, encoding="latin1"))
    x, y, tx, ty, allx, ally, graph = objs
    test_idx_reorder = [int(l) for l in open(f"{DATAPATH}ind.{ds}.test.index")]
    test_idx_range = np.sort(test_idx_reorder)
    if ds == "citeseer":
        full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_ext = sp.lil_matrix((len(full), x.shape[1])); tx_ext[test_idx_range - min(test_idx_range), :] = tx; tx = tx_ext
        ty_ext = np.zeros((len(full), y.shape[1])); ty_ext[test_idx_range - min(test_idx_range), :] = ty; ty = ty_ext
    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = sp.csr_matrix(nx.adjacency_matrix(nx.from_dict_of_lists(graph)))
    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = list(range(len(y)))                       # 20 per class (public split)
    idx_val = list(range(len(y), len(y) + 500))
    return adj, features, labels, idx_train, idx_val, idx_test


class GCN(nn.Module):
    def __init__(self, d, h, c):
        super().__init__()
        self.W0 = nn.Linear(d, h, bias=False)
        self.W1 = nn.Linear(h, c, bias=False)

    def forward(self, X, A):
        h = F.relu(torch.sparse.mm(A, self.W0(X)))
        h = F.dropout(h, DROPOUT, training=self.training)
        return torch.sparse.mm(A, self.W1(h))             # logits


def adj_cosine(P, adj):
    A = sp.coo_matrix(((adj + adj.T) > 0).astype(float)); A.setdiag(0); A = A.tocoo()
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    m = [float((Pn[i] * Pn[j]).sum()) for i, j in zip(A.row, A.col) if i < j]
    return float(np.mean(m))


def train(ds, seed=0, verbose=True):
    adj, features, labels, idx_tr, idx_va, idx_te = load_planetoid_public(ds)
    A = preprocess_adj(adj)                                # D^-1/2 (A+I) D^-1/2 sparse
    X = torch.tensor(preprocess_features(features), dtype=torch.float32)
    y = torch.tensor(labels.argmax(1), dtype=torch.long)
    tr, va, te = (torch.tensor(ix, dtype=torch.long) for ix in (idx_tr, idx_va, idx_te))
    torch.manual_seed(seed)
    model = GCN(X.shape[1], HIDDEN, int(y.max()) + 1)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    best_vl, best_state, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        model.train(); opt.zero_grad()
        out = model(X, A)
        loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(model(X, A)[va], y[va]))
        if vl < best_vl - 1e-4:
            best_vl, best_state, bad = vl, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        post = F.softmax(model(X, A), dim=1).numpy()
    acc = float((post.argmax(1)[idx_te] == labels.argmax(1)[idx_te]).mean())
    stats = dict(test_acc=round(acc * 100, 2), mean_peak=round(float(post.max(1).mean()), 3),
                 median_peak=round(float(np.median(post.max(1))), 3),
                 adj_cosine=round(adj_cosine(post, adj), 3), epochs=ep + 1)
    if verbose:
        print(f"[{ds}] test_acc={stats['test_acc']}%  mean_peak={stats['mean_peak']} "
              f"median_peak={stats['median_peak']}  adjacent_cosine={stats['adj_cosine']} "
              f"(epochs={stats['epochs']})")
    with open(os.path.join(PRED, f"{ds}_softgcn_whole_pred.pkl"), "wb") as f:
        pickle.dump(post.astype(np.float64), f)
    return post, stats


if __name__ == "__main__":
    for ds in ["cora", "citeseer"]:
        train(ds)
