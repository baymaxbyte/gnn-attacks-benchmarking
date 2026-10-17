"""GCond black box: takes node ids, returns posteriors.
"""
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch

import common as C

GCOND = C.GCOND
CONDENSED = os.path.join(C.HERE, "artifacts", "condensed")

EVAL_HIDDEN = 256
EVAL_NLAYERS = 2
EVAL_DROPOUT = 0.0
EVAL_WEIGHT_DECAY = 5e-4
EVAL_EPOCHS = 600
EVAL_LR = 0.01
EVAL_REPEATS = 10

PAPER_TABLE2 = {
    ("cora", "whole"): (81.2, 0.2), ("cora", "0.25"): (79.8, 1.3),
    ("cora", "0.5"): (80.1, 0.6),   ("cora", "1.0"): (79.3, 0.3),
    ("citeseer", "whole"): (71.7, 0.1), ("citeseer", "0.25"): (70.5, 1.2),
    ("citeseer", "0.5"): (70.6, 0.9),   ("citeseer", "1.0"): (69.8, 1.4),
}
PAPER_R = {"cora": {"0.25": "1.3%", "0.5": "2.6%", "1.0": "5.2%", "whole": "100%"},
           "citeseer": {"0.25": "0.9%", "0.5": "1.8%", "1.0": "3.6%",
                        "whole": "100%"}}


def _import_gcond():
    if GCOND not in sys.path:
        sys.path.insert(0, GCOND)
    cwd = os.getcwd()
    os.chdir(GCOND)
    try:
        import utils as gutils
        from models.gcn import GCN
    finally:
        os.chdir(cwd)
    return gutils, GCN


class GCondBlackBox:
    """One deployment. `setting` is 'whole' or a CLI reduction rate."""

    def __init__(self, dataset, setting, seed=1, device="cpu"):
        self.dataset, self.setting, self.seed = dataset, setting, seed
        self.device = device
        self.gutils, self.GCN = _import_gcond()
        cwd = os.getcwd()
        os.chdir(GCOND)
        try:
            full = self.gutils.get_dataset(dataset, normalize_features=True)
            self.data = self.gutils.Transd2Ind(full, keep_ratio=1.0)
        finally:
            os.chdir(cwd)
        self.n_classes = self.data.nclass
        self.n_nodes = sp.csr_matrix(self.data.adj_full).shape[0]

    def training_set(self):
        """The condensed graph (A', X', Y'), or None for the whole-graph box."""
        if self.setting == "whole":
            return None
        tag = f"{self.dataset}_{self.setting}_{self.seed}"
        paths = {k: os.path.join(CONDENSED, f"{k}_{tag}.pt")
                 for k in ("adj", "feat", "label")}
        missing = [p for p in paths.values() if not os.path.exists(p)]
        if missing:
            raise SystemExit(f"condensed artifacts missing: {missing}\n"
                             f"Run step01_condense.py first.")
        load = lambda p: torch.load(p, map_location="cpu", weights_only=False)  # noqa: E731
        adj, feat, lab = (load(paths[k]) for k in ("adj", "feat", "label"))
        return adj.detach(), feat.detach(), lab.detach()

    def build(self, run_seed=0, verbose=False):
        """Train one evaluation GCN."""
        torch.manual_seed(run_seed)
        np.random.seed(run_seed)
        d = self.data
        n_feat = sp.csr_matrix(d.feat_full).shape[1] \
            if sp.issparse(d.feat_full) else np.asarray(d.feat_full).shape[1]

        model = self.GCN(nfeat=n_feat, nhid=EVAL_HIDDEN, nclass=self.n_classes,
                         nlayers=EVAL_NLAYERS, dropout=EVAL_DROPOUT,
                         lr=EVAL_LR, weight_decay=EVAL_WEIGHT_DECAY,
                         device=self.device).to(self.device)

        cwd = os.getcwd()
        os.chdir(GCOND)
        try:
            if self.setting == "whole":
                model.fit(d.feat_full, d.adj_full, d.labels_full,
                          d.idx_train, d.idx_val, train_iters=EVAL_EPOCHS,
                          normalize=True, verbose=verbose)
            else:
                adj_syn, feat_syn, labels_syn = self.training_set()
                model.fit_with_val(feat_syn, adj_syn, labels_syn, d,
                                   train_iters=EVAL_EPOCHS, normalize=True,
                                   verbose=verbose, noval=False)
        finally:
            os.chdir(cwd)
        return model

    def _full_graph_output(self, model):
        cwd = os.getcwd()
        os.chdir(GCOND)
        try:
            return model.predict(self.data.feat_full, self.data.adj_full)
        finally:
            os.chdir(cwd)

    def test_accuracy(self, model):
        out = self._full_graph_output(model)
        pred = out[self.data.idx_test].argmax(1).cpu().numpy()
        return float((pred == np.asarray(self.data.labels_test)).mean())

    def serve(self, model):
        """The query interface: posteriors for all N original nodes."""
        out = self._full_graph_output(model).detach().cpu().numpy()
        post = np.exp(out)                       # forward returns log_softmax
        return post / post.sum(1, keepdims=True)

    def validate(self, repeats=EVAL_REPEATS, verbose=False):
        accs, served = [], None
        for run in range(repeats):
            model = self.build(run_seed=run, verbose=verbose)
            accs.append(self.test_accuracy(model))
            if run == 0:
                served = self.serve(model)       # the deployed model
            print(f"    run {run + 1}/{repeats}  test acc = {accs[-1]:.4f}",
                  flush=True)
        accs = np.array(accs)
        ref = PAPER_TABLE2.get((self.dataset, self.setting))
        return {
            "dataset": self.dataset,
            "setting": self.setting,
            "paper_r": PAPER_R[self.dataset][self.setting],
            "n_train_nodes": (self.n_nodes if self.setting == "whole"
                              else int(self.training_set()[0].shape[0])),
            "acc_mean": round(float(accs.mean() * 100), 2),
            "acc_std": round(float(accs.std() * 100), 2),
            "acc_runs": [round(float(a * 100), 2) for a in accs],
            "paper_mean": ref[0] if ref else None,
            "paper_std": ref[1] if ref else None,
            "delta": round(float(accs.mean() * 100 - ref[0]), 2) if ref else None,
            "served_posteriors": served,
        }
