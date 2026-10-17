"""FGC black box: takes original node ids, returns posteriors.

Two serving modes over one set of trained weights:

    mode "orig"   model(X_original, A_original)[u]        -- what the paper does
    mode "lift"   model(X_coarse, A_coarse)[s(u)]         -- ours

"""
import numpy as np
import torch
import torch.nn.functional as Fn
from torch_geometric.nn import GCNConv

import data as D
import fgc_solver as F

# ---- classifier spec. lr and weight decay are the paper's (section 4.2); the
# ---- rest exists only in the authors' notebook, the paper does not state it.
HIDDEN = 64
DROPOUT = 0.5
LR = 0.01
WEIGHT_DECAY = 1e-4
EPOCHS = 60
EDGE_THRESHOLD = 0.1

# ---- coarsening spec. The paper's Appendix B values (gamma=500, alpha=500,
# ---- lambda=d/2) belong to the numpy solver_v2 and are scale-incompatible with
# ---- this solver's step; they drive C to a near-random partition and ~30%
# ---- accuracy. These are the values from the notebook driver that actually calls
# ---- get_accuracy. beta is unpublished, so it comes from that notebook's own beta
# ---- grid; 0.001 is the grid point that reproduces Table 4.
LAM = 0.001
ALPHA = 1.0
GAMMA = 0.001
BETA = 0.001
STEP = "notebook"
# Block-update order. "notebook" recomputes X_tilde from the OLD C and normalises C
# afterwards, which is literally what the authors' `update` does.
VARIANT = "notebook"

PAPER_R = (0.3, 0.1, 0.05)

# kumar23a Table 4 (FGC with sparsity regulariser) and Table 6 (whole graph)
PAPER_ACC = {
    ("cora", None): (89.50, 1.23),
    ("cora", 0.3): (85.79, 0.24),
    ("cora", 0.1): (81.46, 0.79),
    ("cora", 0.05): (80.01, 0.51),
    ("citeseer", None): (78.08, 1.96),
    ("citeseer", 0.3): (74.64, 1.37),
    ("citeseer", 0.1): (73.36, 0.53),
    ("citeseer", 0.05): (71.02, 0.96),
}


class Net(torch.nn.Module):
    def __init__(self, d, n_classes, hidden=HIDDEN):
        super().__init__()
        self.conv1 = GCNConv(d, hidden)
        self.conv2 = GCNConv(hidden, n_classes)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = Fn.relu(x)
        x = Fn.dropout(x, p=DROPOUT, training=self.training)
        x = self.conv2(x, edge_index)
        return Fn.log_softmax(x, dim=1)          # forward returns log_softmax


def _edge_index(adj):
    src, dst = adj.nonzero()
    return torch.from_numpy(np.vstack([src, dst])).long()


class FGCBlackBox:
    """One trained model; two ways to answer a query.

    r=None is the no-coarsening control: train on the original graph with all n
    labels, which is what the paper's Table 6 "GCN (Whole dataset)" column is.
    """

    def __init__(self, dataset, r=None, beta=BETA, alpha=ALPHA, gamma=GAMMA,
                 lam=LAM, step=STEP, variant=VARIANT,
                 edge_threshold=EDGE_THRESHOLD, iters=F.ITERS):
        self.dataset = dataset
        self.r = r
        self.meta = D.load(dataset)
        self.n = self.meta["n"]
        self.d = self.meta["d"]
        self.n_classes = self.meta["n_classes"]
        self.labels = torch.from_numpy(self.meta["labels"])

        self.X = torch.as_tensor(np.asarray(self.meta["features"].todense()),
                                 dtype=torch.float32)
        self.edge_index = _edge_index(self.meta["adj"])

        self.hp = dict(lam=self.d / 2 if lam is None else lam,
                       alpha=alpha, gamma=gamma, beta=beta)
        self.hp_source = ("notebook driver (line 441) + beta from that notebook's "
                          "beta grid; paper App. B values are scale-incompatible")
        self.step = step
        self.variant = variant
        self.edge_threshold = edge_threshold
        self.iters = iters

        self.k = None if r is None else int(self.n * r)
        self.C = None
        self.coarse = None
        self.model = None

    # -------------------------------------------------------------- coarsening
    def coarsen(self, C=None):
        """Produce (or accept) the coarsening. No-op for the whole-graph control."""
        if self.r is None:
            return None
        if C is None:
            theta = D.laplacian(self.meta["adj"])
            C, _, _ = F.coarsen(theta, self.X.numpy(), self.k, iters=self.iters,
                                step=self.step, variant=self.variant, **self.hp)
        self.C = torch.as_tensor(C, dtype=torch.float32)
        theta = D.laplacian(self.meta["adj"])
        self.coarse = F.coarse_graph(F.harden(self.C), theta, self.X,
                                     self.meta["labels"], self.n_classes,
                                     edge_threshold=self.edge_threshold)
        return self.coarse

    # ---------------------------------------------------------------- training
    def _training_inputs(self):
        if self.r is None:
            return self.X, self.edge_index, self.labels
        return self.coarse["X_c"], self.coarse["edge_index"], self.coarse["Y_c"]

    def build(self, run_seed=0):
        """Train the GCN on the coarse graph (or the original, for the control)."""
        if self.r is not None and self.coarse is None:
            self.coarsen()
        x, ei, y = self._training_inputs()

        torch.manual_seed(run_seed)
        self.model = Net(self.d, self.n_classes)
        opt = torch.optim.Adam(self.model.parameters(), lr=LR,
                               weight_decay=WEIGHT_DECAY)
        self.model.train()
        for _ in range(EPOCHS):
            opt.zero_grad()
            Fn.nll_loss(self.model(x, ei), y).backward()
            opt.step()
        return self

    # ----------------------------------------------------------------- serving
    @torch.no_grad()
    def serve(self, mode="orig", eval_mode=True):
        """Posteriors for every ORIGINAL node id, shape (n, n_classes).

        eval_mode=False reproduces the notebook, which never calls model.eval()
        and therefore predicts with dropout active. We serve with eval_mode=True
        so a query is deterministic.
        """
        if self.model is None:
            raise RuntimeError("call build() first")
        self.model.eval() if eval_mode else self.model.train()

        if mode == "orig":
            logp = self.model(self.X, self.edge_index)
        elif mode == "lift":
            if self.r is None:
                raise ValueError("mode 'lift' undefined for the whole-graph control")
            logp = self.model(self.coarse["X_c"], self.coarse["edge_index"])
            logp = logp[self.coarse["assignment"].long()]       # lift to original ids
        else:
            raise ValueError(f"unknown mode: {mode}")

        post = logp.exp()
        return (post / post.sum(1, keepdim=True)).numpy()

    def accuracy(self, mode="orig", eval_mode=True):
        """Fraction of ALL n original nodes labelled correctly.

        The paper's protocol has no held-out split: coarse labels derive from every
        original label and accuracy is measured over every original node. This
        reproduces that, so the number is in-sample.
        """
        post = self.serve(mode=mode, eval_mode=eval_mode)
        return float((post.argmax(1) == self.meta["labels"]).mean())

    # -------------------------------------------------------------- validation
    def validate(self, repeats=10, modes=("orig", "lift")):
        """Rebuild `repeats` times with C fixed; return per-mode accuracy stats."""
        if self.r is not None and self.coarse is None:
            self.coarsen()
        modes = [m for m in modes if not (m == "lift" and self.r is None)]
        runs = {m: [] for m in modes}
        runs_train_mode = {m: [] for m in modes}
        for s in range(repeats):
            self.build(run_seed=s)
            for m in modes:
                runs[m].append(self.accuracy(m, eval_mode=True))
                runs_train_mode[m].append(self.accuracy(m, eval_mode=False))

        paper = PAPER_ACC.get((self.dataset, self.r))
        out = {"dataset": self.dataset, "r": self.r, "k": self.k,
               "paper_mean": paper[0] if paper else None,
               "paper_std": paper[1] if paper else None}
        for m in modes:
            a = np.array(runs[m]) * 100
            t = np.array(runs_train_mode[m]) * 100
            out[m] = {
                "mean": float(a.mean()), "std": float(a.std()),
                "runs": [round(float(v), 4) for v in a],
                "mean_dropout_on": float(t.mean()),
                "delta_vs_paper": (float(a.mean() - paper[0]) if paper else None),
            }
        if self.coarse:
            out["coarse"] = {kk: vv for kk, vv in self.coarse.items()
                             if not torch.is_tensor(vv)}
            out["collisions"] = self.collision_stats()
        return out

    def collision_stats(self):
        """Mode B serves only k distinct vectors for n ids. Quantify that."""
        if self.r is None:
            return None
        a = self.coarse["assignment"].numpy()
        sizes = np.bincount(a, minlength=self.k)
        nz = sizes[sizes > 0]
        return {
            "distinct_posteriors_lift": int(len(nz)),
            "distinct_posteriors_orig": int(len(np.unique(
                np.round(self.serve("orig"), 10), axis=0))) if self.model else None,
            "mean_group_size": float(nz.mean()),
            "max_group_size": int(nz.max()),
            "identical_pairs": int((nz * (nz - 1) // 2).sum()),
        }
