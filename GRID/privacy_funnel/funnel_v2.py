#!/usr/bin/env python3
"""todo #1 privacy-funnel — v2 (stabilized, DANN gradient-reversal).

Fixes v1's unstable alternating minimax: a single optimizer trains the gated-GCN
encoder, the classifier, and the edge-adversary together; a Gradient-Reversal
Layer (GRL) makes the encoder remove the within-class edge signal from the served
posteriors while the classifier keeps label info. lambda is annealed 0 -> lambda_max
(DANN schedule). Privacy is scored by a FRESH held-out attacker (not the in-loop
adversary). Reuses helpers from funnel.py.
"""
import os, sys, json, math
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import funnel as Fb                                   # load_lcc, splits, GCN (gated), leak_auc, rand_auc, same_class_pairs
from funnel import preprocess_adj, preprocess_features
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float32)

EPOCHS, LR, WD = 300, 0.01, 5e-4
LAMBDAS = [0.0, 1.0, 3.0, 10.0]


class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lamb * g, None


def grl(x, lamb): return GRL.apply(x, lamb)


def train(ds, lam_max):
    A, X, y = Fb.load_lcc(ds); n = A.shape[0]; C = int(y.max()) + 1
    support = preprocess_adj(A)
    Xt = torch.tensor(preprocess_features(sp.csr_matrix(X))); yt = torch.tensor(y, dtype=torch.long)
    tr, va, te = Fb.splits(n); tr_t = torch.tensor(tr)
    r0, c0 = sp.triu(A, 1).nonzero(); E = np.stack([r0, c0], 1)

    torch.manual_seed(0)
    enc = Fb.GCN(Xt.shape[1], C)
    disc = nn.Sequential(nn.Linear(2 * C, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=LR, weight_decay=WD)

    for ep in range(EPOCHS):
        lam = lam_max * (2.0 / (1.0 + math.exp(-10 * ep / EPOCHS)) - 1.0)   # anneal 0 -> lam_max
        pos, neg = Fb.same_class_pairs(E, y, n, 1, seed=ep)
        pi = torch.tensor(np.r_[pos[:, 0], neg[:, 0]]); pj = torch.tensor(np.r_[pos[:, 1], neg[:, 1]])
        lab = torch.tensor(np.r_[np.ones(len(pos)), np.zeros(len(neg))], dtype=torch.float32)
        enc.train(); disc.train(); opt.zero_grad()
        logits = enc(Xt, support); p = F.softmax(logits, 1)
        feat = torch.cat([(p[pi] - p[pj]).abs(), p[pi] * p[pj]], 1)
        dlogit = disc(grl(feat, lam)).squeeze(1)
        dloss = F.binary_cross_entropy_with_logits(dlogit, lab)            # D minimizes; enc maximizes via GRL
        closs = F.cross_entropy(logits[tr_t], yt[tr_t])
        (closs + dloss).backward(); opt.step()

    enc.eval()
    with torch.no_grad(): P = F.softmax(enc(Xt, support), 1).numpy()
    acc = float((P.argmax(1)[te] == y[te]).mean())
    g1, g2 = enc.gates()
    return dict(ds=ds, lam=lam_max, acc=acc, leak=Fb.leak_auc(P, y, E), rnd=Fb.rand_auc(P, y, E), g1=g1, g2=g2)


def main():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ds in zip(axes, ["cora", "citeseer"]):
        pts = []
        for lam in LAMBDAS:
            r = train(ds, lam); rows.append(r); pts.append(r)
            print(f"[{ds}] lambda_max={lam:5.1f}  node_acc={r['acc']*100:4.1f}%  within-class leak AUC={r['leak']:.3f}"
                  f"  (rand-neg AUC={r['rnd']:.3f})  graph-gates=({r['g1']:.2f},{r['g2']:.2f})", flush=True)
        ax.plot([p["leak"] for p in pts], [p["acc"] for p in pts], "-o")
        for p in pts:
            ax.annotate(f"λ={p['lam']:.0f}", (p["leak"], p["acc"]), fontsize=8, textcoords="offset points", xytext=(4, 4))
        ax.axvline(0.5, color="green", ls=":", lw=1, label="perfect privacy (0.5)")
        ax.set_xlabel("within-class link-leakage AUC (lower=private)"); ax.set_ylabel("node test accuracy")
        ax.set_title(f"{ds}: funnel v2 (GRL) tradeoff"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Privacy-funnel v2 (gradient-reversal): node accuracy vs within-class link leakage", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "funnel_v2_tradeoff.png"); fig.savefig(p, dpi=130)
    json.dump(rows, open(os.path.join(OUT, "funnel_v2_results.json"), "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
