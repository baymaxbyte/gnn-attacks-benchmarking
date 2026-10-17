"""KRON black box: takes original node ids, returns posteriors.

Trains on the Kron-reduced graph (boundary nodes, their own original features and
labels, wired by Schur-complement edge weights) and is evaluated and served on the
ORIGINAL graph with the original 60/20/20 split, so a reduced box differs from its
control only in the training graph. The classifier threads edge_weight through every
layer, or the reduction would be invisible. The paper runs no node classification,
so the classifier, split and optimiser are ours; every accuracy is our own measure.
"""
import numpy as np
import torch
import torch.nn.functional as Fn
from torch_geometric.nn import GCNConv

import common as K
import config as C
import data as D

HIDDEN = 64
DROPOUT = 0.5
LR = 0.003
WEIGHT_DECAY = 0.0005
EPOCHS = 500
SPLIT_SCHEME = "paper"          # 60/20/20, as used in step01
SPLIT_SEED = 0


class GCN(torch.nn.Module):
    """3 GCNConv layers, edge weights threaded through all of them."""

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
    src, dst = adj.nonzero()
    return torch.from_numpy(np.vstack([src, dst])).long()


class KRONBlackBox:
    """r=None is the no-reduction control."""

    def __init__(self, dataset, r=None):
        self.dataset, self.r = dataset, r

        m = D.load(dataset)
        self.meta = m
        self.n, self.d, self.n_classes = m["n"], m["d"], m["n_classes"]
        self.labels = torch.from_numpy(m["labels"])
        self.X = torch.as_tensor(np.asarray(m["features"].todense()),
                                 dtype=torch.float32)
        self.edge_index = _edge_index(m["adj"])
        self.edge_weight = torch.ones(self.edge_index.shape[1])

        self.k = None
        self.reduced = None
        self.model = None
        self.best_val = None

        if r is None:
            sp = D.make_split(m["labels"], self.n_classes, seed=SPLIT_SEED,
                              scheme=SPLIT_SCHEME)
        else:
            z = np.load(K.reduced_path(dataset, r))
            self.k = int(z["kept_idx"].shape[0])
            self.reduced = {
                "kept_idx": z["kept_idx"],
                "X_c": torch.from_numpy(z["X_c"].astype(np.float32)),
                "Y_c": torch.from_numpy(z["Y_c"].astype(np.int64)),
                "edge_index": torch.from_numpy(z["edge_index"]).long(),
                "edge_weight": torch.from_numpy(z["edge_weight"].astype(np.float32)),
                "keep_train": torch.from_numpy(np.asarray(z["keep_train"])),
            }
            sp = {"train_mask": z["train_mask"], "val_mask": z["val_mask"],
                  "test_mask": z["test_mask"], "scheme": SPLIT_SCHEME,
                  "seed": SPLIT_SEED}
        self.split = sp
        self.train_mask = torch.from_numpy(np.asarray(sp["train_mask"]))
        self.val_mask = torch.from_numpy(np.asarray(sp["val_mask"]))
        self.test_mask = torch.from_numpy(np.asarray(sp["test_mask"]))

    # ---------------------------------------------------------------- training
    def _train_inputs(self):
        if self.r is None:
            return (self.X, self.edge_index, self.edge_weight,
                    self.labels, self.train_mask)
        c = self.reduced
        return (c["X_c"], c["edge_index"], c["edge_weight"], c["Y_c"],
                c["keep_train"])

    @torch.no_grad()
    def _val_accuracy(self):
        self.model.eval()
        pred = self.model(self.X, self.edge_index, self.edge_weight).argmax(1)
        return float((pred[self.val_mask] == self.labels[self.val_mask])
                     .float().mean())

    def build(self, run_seed=0):
        x, ei, ew, y, mask = self._train_inputs()
        torch.manual_seed(run_seed)
        self.model = GCN(self.d, HIDDEN, self.n_classes)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
        best_state, best_val, best_epoch = None, -1.0, -1
        for ep in range(EPOCHS):
            self.model.train()
            opt.zero_grad()
            Fn.nll_loss(self.model(x, ei, ew)[mask], y[mask]).backward()
            opt.step()
            va = self._val_accuracy()
            if va > best_val:
                best_val, best_epoch = va, ep
                best_state = {kk: vv.detach().clone()
                              for kk, vv in self.model.state_dict().items()}
        if best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val, self.best_epoch = best_val, best_epoch
        return self

    # ----------------------------------------------------------------- serving
    @torch.no_grad()
    def serve(self):
        if self.model is None:
            raise RuntimeError("call build() first")
        self.model.eval()
        post = self.model(self.X, self.edge_index, self.edge_weight).exp()
        return (post / post.sum(1, keepdim=True)).numpy()

    def accuracy(self, subset="test"):
        pred = self.serve().argmax(1)
        mask = {"test": self.test_mask, "val": self.val_mask,
                "train": self.train_mask}[subset].numpy()
        return float((pred[mask] == self.meta["labels"][mask]).mean())

    def validate(self, repeats=5):
        accs, vals = [], []
        for s in range(repeats):
            self.build(run_seed=s)
            accs.append(self.accuracy("test") * 100)
            vals.append(self.best_val * 100)
        a = np.array(accs)
        out = {"dataset": self.dataset, "r": self.r, "k": self.k,
               "split": {"scheme": SPLIT_SCHEME,
                         "train": int(self.train_mask.sum()),
                         "val": int(self.val_mask.sum()),
                         "test": int(self.test_mask.sum())},
               "test_acc_mean": float(a.mean()), "test_acc_std": float(a.std()),
               "test_acc_runs": [round(float(v), 4) for v in a],
               "best_val_mean": float(np.mean(vals))}
        if self.reduced:
            w = self.reduced["edge_weight"].numpy()
            out["reduced"] = {
                "kept_nodes": self.k,
                "train_nodes": int(self.reduced["keep_train"].sum()),
                "reduced_edges": int(self.reduced["edge_index"].shape[1] // 2),
                "edge_weight_mean": float(w.mean()),
                "edge_weight_min": float(w.min()),
                "edge_weight_max": float(w.max()),
            }
        return out
