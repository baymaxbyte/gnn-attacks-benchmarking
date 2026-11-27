#!/usr/bin/env python3
"""EXP2: replace the black-box GNN with an MLP on node features ONLY (no adjacency).

Directly tests the RCA's RC2 (serving-time graph smoothing). An MLP has no message
passing, so adjacent nodes are not smoothed toward each other at inference -> if the
link leak were purely RC2, it would collapse; whatever remains is the RC1 class-
homophily floor. Everything else matches the GNN box: same 60/20/20 seed-0 split,
true-label training on the 60% train split, best-val selection, 3 layers d->64->64->C,
dropout 0.5, LR 0.003, wd 5e-4, 500 epochs. Served on all nodes (features only).

Posteriors -> attack/data/pred/{ds}_exp2mlp_pred.pkl.
"""
import os, sys
import numpy as np
import torch, torch.nn.functional as Fn

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(os.path.dirname(HERE))
MCC = os.path.join(GRIDROOT, "MCC")
sys.path.insert(0, MCC)
import mcc_boxes as M

HID, DROP = 64, 0.5
torch.set_num_threads(4)


class MLP(torch.nn.Module):
    def __init__(self, d, c):
        super().__init__()
        self.l1 = torch.nn.Linear(d, HID); self.l2 = torch.nn.Linear(HID, 64); self.l3 = torch.nn.Linear(64, c)

    def forward(self, x):
        x = Fn.relu(self.l1(x)); x = Fn.dropout(x, DROP, training=self.training)
        x = Fn.relu(self.l2(x)); x = Fn.dropout(x, DROP, training=self.training)
        return Fn.log_softmax(self.l3(x), dim=1)


def build(ds, seed=0):
    A, X, labels, C = M.load_any(ds); n, d = X.shape
    mtr, mva, mte = M.split_6020(labels, C)
    Xo = torch.tensor(X); yt = torch.from_numpy(labels)
    tm, vm = torch.from_numpy(mtr), torch.from_numpy(mva)
    torch.manual_seed(seed); model = MLP(d, C)
    opt = torch.optim.Adam(model.parameters(), lr=M.LR, weight_decay=M.WD)
    best, bs = -1.0, None
    for _ in range(M.EPOCHS):
        model.train(); opt.zero_grad()
        Fn.nll_loss(model(Xo)[tm], yt[tm]).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            va = float((model(Xo).argmax(1)[vm] == yt[vm]).float().mean())
        if va > best:
            best, bs = va, {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(bs); model.eval()
    with torch.no_grad():
        post = model(Xo).exp().numpy()
    post = post / post.sum(1, keepdims=True)
    acc = float((post.argmax(1)[mte] == labels[mte]).mean())
    M.save(f"{ds}_exp2mlp", post)
    print(f"[{ds}] MLP box: best_val={best*100:.1f}%  held-out test acc={acc*100:.1f}%  mean_max_p={post.max(1).mean():.3f}", flush=True)


if __name__ == "__main__":
    import sys
    datasets = sys.argv[1:] if len(sys.argv) > 1 else ["cora", "citeseer"]
    for ds in datasets:
        build(ds)
