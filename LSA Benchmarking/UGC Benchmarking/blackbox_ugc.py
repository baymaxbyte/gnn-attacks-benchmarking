"""UGC black box: takes original node ids, returns posteriors.

The model trains on the coarse graph and is then run over the ORIGINAL graph at
inference, which is what UGC.py does. The twist specific to UGC is that the feature
matrix it consumes -- at training AND at inference -- is the augmented
[(1-alpha)X | alpha*A], so each node's input vector contains its own adjacency row.
The graph therefore reaches the model twice: as input features and through message
passing.

ratio is the percentage of nodes REMOVED; ratio=None is the no-coarsening control.
"""
import numpy as np
import torch
import torch.nn.functional as Fn
from torch_geometric.nn import GCNConv

import config as C
import data as D
import ugc_solver as S

# ---- classifier. Paper Table 8 gives hidden {64, 64}, lr 0.003, decay 0.0005,
# ---- 500 epochs; GCN.py implements that as three GCNConv layers and passes
# ---- edge_weight through every one of them.
HIDDEN = 64
DROPOUT = 0.5
LR = 0.003
WEIGHT_DECAY = 0.0005
EPOCHS = 500
SPLIT_SCHEME = "paper"      # 60/20/20 as stated in the paper
SPLIT_SEED = 0

# kumar-style gate targets: UGC paper Table 9 (Cora at each ratio). Citeseer is
# absent from every accuracy table in the paper, so it has no published target.
PAPER_ACC = {
    ("cora", 30): 86.30,
    ("cora", 50): 86.30,
    ("cora", 70): 84.63,
    ("cora", None): None,
    ("citeseer", 30): None,
    ("citeseer", 50): None,
    ("citeseer", 70): None,
    ("citeseer", None): None,
}
# Table 4 also separates the ablation on Cora at ratio 50:
PAPER_ABLATION = {"cora": {"fea_only": 83.92, "fea_plus_adj": 86.30}}


class GCN(torch.nn.Module):
    """Verbatim structure of UGC's GCN.py:GCN_."""

    def __init__(self, num_features, hidden_channels, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 64)
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


class UGCBlackBox:
    def __init__(self, dataset, ratio=None, alpha=None,
                 split_scheme=SPLIT_SCHEME, split_seed=SPLIT_SEED, seed=0):
        self.dataset = dataset
        self.ratio = ratio
        self.alpha = C.ALPHA[dataset] if alpha is None else alpha
        self.seed = seed

        m = D.load(dataset)
        self.meta = m
        self.n, self.d, self.n_classes = m["n"], m["d"], m["n_classes"]
        self.labels = torch.from_numpy(m["labels"])

        sp = D.make_split(m["labels"], self.n_classes, seed=split_seed,
                          scheme=split_scheme)
        self.split = sp
        self.train_mask = torch.from_numpy(sp["train_mask"])
        self.val_mask = torch.from_numpy(sp["val_mask"])
        self.test_mask = torch.from_numpy(sp["test_mask"])

        # the augmented matrix is what the model sees, always
        self.X_aug = S.augment(m["features"], m["adj"], self.alpha)
        self.edge_index = _edge_index(m["adj"])
        self.edge_weight = torch.ones(self.edge_index.shape[1])

        self.k = None if ratio is None else C.n_supernodes(self.n, ratio)
        self.bin_width = None
        self.coarse = None
        self.model = None
        self.best_val = None

    # -------------------------------------------------------------- coarsening
    def coarsen(self, bin_width=None, assignment=None, calibrate=True,
                verbose=False):
        """Hash-and-merge. Returns the coarse-graph dict, or None for the control."""
        if self.ratio is None:
            return None
        if assignment is None:
            if bin_width is None:
                if not calibrate:
                    raise ValueError("need bin_width or calibrate=True")
                r = S.find_bin_width(self.X_aug, self.ratio / 100.0,
                                     seed=self.seed, verbose=verbose)
                bin_width = r["bin_width"]
                self.calibration = r
            self.bin_width = bin_width
            bv = S.hashed_values(self.X_aug, seed=self.seed)
            assignment = S.partition(bv, bin_width, seed=self.seed)
        self.coarse = S.coarse_graph(assignment, self.X_aug, self.meta["adj"],
                                     self.meta["labels"], self.n_classes,
                                     self.train_mask)
        return self.coarse

    # ---------------------------------------------------------------- training
    def _train_inputs(self):
        if self.ratio is None:
            return (self.X_aug, self.edge_index, self.edge_weight,
                    self.labels, self.train_mask)
        c = self.coarse
        return (c["X_c"], c["A_c_edge_index"], c["A_c_edge_attr"],
                c["Y_c"], c["keep"])

    @torch.no_grad()
    def _val_accuracy(self):
        self.model.eval()
        pred = self.model(self.X_aug, self.edge_index, self.edge_weight).argmax(1)
        return float((pred[self.val_mask] == self.labels[self.val_mask])
                     .float().mean())

    def build(self, run_seed=0, select_on_val=True):
        """Train on the coarse graph, keeping the best-validation weights.

        Validation is measured on the ORIGINAL graph, which is what UGC.py does
        (it calls val(model, data) with the uncoarsened data every epoch).
        """
        if self.ratio is not None and self.coarse is None:
            self.coarsen()
        x, ei, ew, y, mask = self._train_inputs()

        torch.manual_seed(run_seed)
        self.model = GCN(self.X_aug.shape[1], HIDDEN, self.n_classes)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
        best_state, best_val, best_epoch = None, -1.0, -1
        for ep in range(EPOCHS):
            self.model.train()
            opt.zero_grad()
            out = self.model(x, ei, ew)
            Fn.nll_loss(out[mask], y[mask]).backward()
            opt.step()
            if select_on_val:
                va = self._val_accuracy()
                if va > best_val:
                    best_val, best_epoch = va, ep
                    best_state = {k: v.detach().clone()
                                  for k, v in self.model.state_dict().items()}
        if select_on_val and best_state is not None:
            self.model.load_state_dict(best_state)
        self.best_val, self.best_epoch = best_val, best_epoch
        return self

    # ----------------------------------------------------------------- serving
    @torch.no_grad()
    def serve(self):
        """Posteriors for every ORIGINAL node id, shape (n, n_classes)."""
        if self.model is None:
            raise RuntimeError("call build() first")
        self.model.eval()
        post = self.model(self.X_aug, self.edge_index, self.edge_weight).exp()
        return (post / post.sum(1, keepdim=True)).numpy()

    def accuracy(self, subset="test"):
        """Accuracy on the held-out test split -- a real generalisation estimate.

        Unlike FGC, UGC uses a proper split and its coarse labels aggregate only
        TRAINING labels, so nothing about this number is in-sample.
        """
        post = self.serve()
        pred = post.argmax(1)
        mask = {"test": self.test_mask, "val": self.val_mask,
                "train": self.train_mask}[subset].numpy()
        return float((pred[mask] == self.meta["labels"][mask]).mean())

    # -------------------------------------------------------------- validation
    def validate(self, repeats=5):
        if self.ratio is not None and self.coarse is None:
            self.coarsen()
        accs, vals = [], []
        for s in range(repeats):
            self.build(run_seed=s)
            accs.append(self.accuracy("test") * 100)
            vals.append(self.best_val * 100)
        a = np.array(accs)
        paper = PAPER_ACC.get((self.dataset, self.ratio))
        out = {"dataset": self.dataset, "ratio": self.ratio, "alpha": self.alpha,
               "k": self.k, "bin_width": self.bin_width,
               "feature_dim_augmented": int(self.X_aug.shape[1]),
               "split": {"scheme": self.split["scheme"],
                         "train": int(self.train_mask.sum()),
                         "val": int(self.val_mask.sum()),
                         "test": int(self.test_mask.sum())},
               "test_acc_mean": float(a.mean()), "test_acc_std": float(a.std()),
               "test_acc_runs": [round(float(v), 4) for v in a],
               "best_val_mean": float(np.mean(vals)),
               "paper_mean": paper,
               "delta_vs_paper": (float(a.mean() - paper) if paper else None)}
        if self.coarse:
            out["coarse"] = {kk: vv for kk, vv in self.coarse.items()
                             if not torch.is_tensor(vv)}
        return out
