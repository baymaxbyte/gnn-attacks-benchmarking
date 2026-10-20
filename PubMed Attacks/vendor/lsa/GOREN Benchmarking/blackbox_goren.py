"""GOREN black box: takes original node ids, returns posteriors.

Trains on the coarse graph, then runs over the ORIGINAL graph at inference. Two
variants share everything except the coarse edge weights:

    bl      S^T A S, the count of original edges between each pair of supernodes
    goren   the same topology with weights predicted by GOREN's GIN
"""
import os

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


def goren_weight_path(dataset, r):
    return os.path.join(C.COARSENED, f"goren_w_{dataset}_r{r}.npy")


class GORENBlackBox:
    """r=None is the no-coarsening control; variant is ignored there."""

    def __init__(self, dataset, r=None, variant="bl"):
        if variant not in C.VARIANTS:
            raise ValueError(f"variant must be one of {C.VARIANTS}")
        self.dataset, self.r, self.variant = dataset, r, variant

        m = D.load(dataset)
        self.meta = m
        self.n, self.d, self.n_classes = m["n"], m["d"], m["n_classes"]
        self.labels = torch.from_numpy(m["labels"])
        self.X = torch.as_tensor(np.asarray(m["features"].todense()),
                                 dtype=torch.float32)
        self.edge_index = _edge_index(m["adj"])
        self.edge_weight = torch.ones(self.edge_index.shape[1])

        self.k = None
        self.coarse = None
        self.model = None
        self.best_val = None

        if r is None:
            sp = D.make_split(m["labels"], self.n_classes, seed=SPLIT_SEED,
                              scheme=SPLIT_SCHEME)
        else:
            z = np.load(K.coarse_path(dataset, r))
            w = z["edge_weight"]
            if variant == "goren":
                wp = goren_weight_path(dataset, r)
                if not os.path.exists(wp):
                    raise SystemExit(f"missing {wp}; run the weight learner first")
                w = np.load(wp)
            self.k = int(z["assignment"].max()) + 1
            self.coarse = {
                "assignment": z["assignment"],
                "X_c": torch.from_numpy(z["X_c"]),
                "Y_c": torch.from_numpy(z["Y_c"]),
                "edge_index": torch.from_numpy(z["edge_index"]).long(),
                "edge_weight": torch.from_numpy(w.astype(np.float32)),
                "keep": torch.from_numpy(z["keep"]),
                "sizes": z["sizes"],
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
        c = self.coarse
        return (c["X_c"], c["edge_index"], c["edge_weight"], c["Y_c"], c["keep"])

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
        out = {"dataset": self.dataset, "r": self.r, "variant": self.variant,
               "k": self.k,
               "split": {"scheme": SPLIT_SCHEME,
                         "train": int(self.train_mask.sum()),
                         "val": int(self.val_mask.sum()),
                         "test": int(self.test_mask.sum())},
               "test_acc_mean": float(a.mean()), "test_acc_std": float(a.std()),
               "test_acc_runs": [round(float(v), 4) for v in a],
               "best_val_mean": float(np.mean(vals))}
        if self.coarse:
            sizes = self.coarse["sizes"]
            w = self.coarse["edge_weight"].numpy()
            out["coarse"] = {
                "trainable_supernodes": int(self.coarse["keep"].sum()),
                "coarse_edges": int(self.coarse["edge_index"].shape[1] // 2),
                "max_supernode_size": int(sizes.max()),
                "mean_supernode_size": float(sizes[sizes > 0].mean()),
                "edge_weight_mean": float(w.mean()),
                "edge_weight_max": float(w.max()),
                "edge_weight_zeros": int((w == 0).sum()),
            }
        return out
