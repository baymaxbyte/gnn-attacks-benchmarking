#!/usr/bin/env python3
"""todo #1 — privacy-funnel (adversarial) representation. FIRST prototype.

Minimise  I(output ; A)  subject to keeping  I(output ; Y)  -- learned, not
post-hoc. A 2-layer GCN encoder produces served posteriors p; a classifier head
keeps label info; an adversarial edge-discriminator D tries to recover edges
from pair features of p, and the encoder is trained to make D uninformative.

Key design: the adversary targets the WITHIN-CLASS distinction (same-class edge
vs same-class non-edge). Against random (cross-class) negatives link-stealing is
just class detection = the task, so that part is irreducible; the funnel can only
remove the EXTRA within-class edge signal. We sweep the privacy weight lambda and
trace node accuracy vs within-class link-leakage AUC (0.5 = private, 1 = leaks).

Self-contained on the LCC of cora/citeseer. numpy/scipy/torch.
"""
import os, sys, json
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch, torch.nn as nn, torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(GRIDROOT, "attack")
DSDIR = os.path.join(ATTACK, "data", "dataset", "original") + os.sep
sys.path.insert(0, GRIDROOT); sys.path.insert(0, ATTACK)
from lsa.data import load_data_original, preprocess_adj, preprocess_features
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
torch.set_default_dtype(torch.float32)

HID, DROP, LR, WD, EPOCHS = 16, 0.5, 0.01, 5e-4, 250
LAMBDAS = [0.0, 1.0, 5.0, 20.0]
rng = np.random.RandomState(0)


def load_lcc(ds):
    adj, feat, *rest = load_data_original(DSDIR, ds)
    y = np.asarray(rest[2]).argmax(1)
    X = feat.toarray() if sp.issparse(feat) else np.asarray(feat)
    A = sp.csr_matrix((adj + adj.T)); A.data[:] = 1.0; A = A.tolil(); A.setdiag(0)
    A = sp.csr_matrix(A); A.eliminate_zeros()
    _, lab = connected_components(A, directed=False)
    idx = np.where(lab == np.bincount(lab).argmax())[0]
    return sp.csr_matrix(A[idx][:, idx]), X[idx].astype(np.float32), y[idx].astype(int)


def splits(n, seed=0):
    p = np.random.RandomState(seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return p[:a], p[a:b], p[b:]


class GCN(nn.Module):
    """Gated GCN: learnable per-layer mix g in [0,1] between graph-free (X W)
    and graph-propagated (A X W). The funnel can push g->0 (become an MLP, no
    edge leak) at the cost of the graph's accuracy benefit -- giving the
    privacy-utility tradeoff a real lever instead of a fixed propagation."""
    def __init__(s, d, c):
        super().__init__(); s.W0 = nn.Linear(d, HID, bias=False); s.W1 = nn.Linear(HID, c, bias=False)
        s.a1 = nn.Parameter(torch.tensor(2.0)); s.a2 = nn.Parameter(torch.tensor(2.0))   # sigmoid(2)~0.88
    def forward(s, X, A):
        g1, g2 = torch.sigmoid(s.a1), torch.sigmoid(s.a2)
        z0 = s.W0(X); h = F.relu((1 - g1) * z0 + g1 * torch.sparse.mm(A, z0))
        h = F.dropout(h, DROP, training=s.training)
        z1 = s.W1(h); return (1 - g2) * z1 + g2 * torch.sparse.mm(A, z1)
    def gates(s):
        return float(torch.sigmoid(s.a1)), float(torch.sigmoid(s.a2))


class Disc(nn.Module):                                      # edge discriminator on pair features of p
    def __init__(s, c):
        super().__init__(); s.net = nn.Sequential(nn.Linear(2 * c, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(s, pi, pj):
        return s.net(torch.cat([(pi - pj).abs(), pi * pj], 1)).squeeze(1)   # logit


def same_class_pairs(E, y, n, n_neg_per, seed=0):
    r = np.random.RandomState(seed)
    pos = E[y[E[:, 0]] == y[E[:, 1]]]                       # same-class real edges
    eset = set(map(tuple, E.tolist()))
    by_cls = {c: np.where(y == c)[0] for c in np.unique(y)}
    neg = []
    want = len(pos) * n_neg_per
    while len(neg) < want:
        c = y[r.randint(n)]; nodes = by_cls[c]
        if len(nodes) < 2: continue
        i, j = nodes[r.randint(len(nodes))], nodes[r.randint(len(nodes))]
        if i > j: i, j = j, i
        if i != j and (i, j) not in eset: neg.append((i, j))
    return pos, np.array(neg)


def leak_auc(P, y, E, seed=1):
    """Strong attacker: train a held-out MLP to tell same-class EDGE from
    same-class NON-EDGE using pair features of the frozen posteriors P."""
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.preprocessing import StandardScaler
    pos, neg = same_class_pairs(E, y, len(P), 1, seed=seed)
    m = min(len(pos), len(neg)); pos, neg = pos[:m], neg[:m]
    def feat(pairs): return np.concatenate([np.abs(P[pairs[:, 0]] - P[pairs[:, 1]]), P[pairs[:, 0]] * P[pairs[:, 1]]], 1)
    Xp = np.vstack([feat(pos), feat(neg)]).astype(np.float64); yp = np.r_[np.ones(m), np.zeros(m)]
    Xp = StandardScaler().fit_transform(Xp)
    perm = np.random.RandomState(seed).permutation(2 * m); cut = m
    tr, te = perm[:cut], perm[cut:]
    clf = MLPClassifier((64,), max_iter=300, random_state=0).fit(Xp[tr], yp[tr])
    return float(roc_auc_score(yp[te], clf.predict_proba(Xp[te])[:, 1]))


def rand_auc(P, y, E, seed=2):
    """Context: same-class edges vs RANDOM (mostly cross-class) non-edges, cosine."""
    from sklearn.metrics import roc_auc_score
    r = np.random.RandomState(seed); n = len(P)
    pos = E[y[E[:, 0]] == y[E[:, 1]]]
    eset = set(map(tuple, E.tolist())); neg = []
    while len(neg) < len(pos):
        i, j = r.randint(n), r.randint(n)
        if i > j: i, j = j, i
        if i != j and (i, j) not in eset: neg.append((i, j))
    neg = np.array(neg)
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-9)
    def cos(pr): return (Pn[pr[:, 0]] * Pn[pr[:, 1]]).sum(1)
    s = np.r_[cos(pos), cos(neg)]; lab = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    return float(roc_auc_score(lab, s))


def train(ds, lam):
    A, X, y = load_lcc(ds); n = A.shape[0]; C = int(y.max()) + 1
    support = preprocess_adj(A)
    Xt = torch.tensor(preprocess_features(sp.csr_matrix(X))); yt = torch.tensor(y, dtype=torch.long)
    tr, va, te = splits(n)
    tr_t = torch.tensor(tr)
    r0, c0 = sp.triu(A, 1).nonzero(); E = np.stack([r0, c0], 1)
    torch.manual_seed(0)
    enc = GCN(Xt.shape[1], C); disc = Disc(C)
    oE = torch.optim.Adam(enc.parameters(), lr=LR, weight_decay=WD)
    oD = torch.optim.Adam(disc.parameters(), lr=0.01)
    for ep in range(EPOCHS):
        pos, neg = same_class_pairs(E, y, n, 1, seed=ep)      # fresh hard pairs each epoch
        pi = torch.tensor(np.r_[pos[:, 0], neg[:, 0]]); pj = torch.tensor(np.r_[pos[:, 1], neg[:, 1]])
        lab = torch.tensor(np.r_[np.ones(len(pos)), np.zeros(len(neg))], dtype=torch.float32)
        # --- D step (detached p) ---
        enc.eval()
        with torch.no_grad(): p = F.softmax(enc(Xt, support), 1)
        disc.train(); oD.zero_grad()
        ld = F.binary_cross_entropy_with_logits(disc(p[pi], p[pj]), lab); ld.backward(); oD.step()
        # --- Encoder step (classify + confuse D) ---
        enc.train(); oE.zero_grad()
        p = F.softmax(enc(Xt, support), 1)
        lcls = F.cross_entropy(enc(Xt, support)[tr_t], yt[tr_t])
        conf = ((torch.sigmoid(disc(p[pi], p[pj])) - 0.5) ** 2).mean()   # 0 when D says 0.5 everywhere
        (lcls + lam * conf).backward(); oE.step()
    enc.eval()
    with torch.no_grad(): P = F.softmax(enc(Xt, support), 1).numpy()
    acc = float((P.argmax(1)[te] == y[te]).mean())
    g1, g2 = enc.gates()
    return dict(ds=ds, lam=lam, acc=acc, leak=leak_auc(P, y, E), rnd=rand_auc(P, y, E), g1=g1, g2=g2)


def main():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, ds in zip(axes, ["cora", "citeseer"]):
        pts = []
        for lam in LAMBDAS:
            r = train(ds, lam); rows.append(r); pts.append(r)
            print(f"[{ds}] lambda={lam:5.1f}  node_acc={r['acc']*100:4.1f}%  within-class leak AUC={r['leak']:.3f}"
                  f"  (rand-neg AUC={r['rnd']:.3f})  graph-gates=({r['g1']:.2f},{r['g2']:.2f})", flush=True)
        ax.plot([p["leak"] for p in pts], [p["acc"] for p in pts], "-o")
        for p in pts:
            ax.annotate(f"λ={p['lam']:.0f}", (p["leak"], p["acc"]), fontsize=8,
                        textcoords="offset points", xytext=(4, 4))
        ax.axvline(0.5, color="green", ls=":", lw=1, label="perfect privacy (0.5)")
        ax.set_xlabel("within-class link-leakage AUC (lower=private)"); ax.set_ylabel("node test accuracy")
        ax.set_title(f"{ds}: privacy–utility tradeoff"); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.suptitle("Privacy-funnel (adversarial) representation: node accuracy vs within-class link leakage", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(OUT, "funnel_tradeoff.png"); fig.savefig(p, dpi=130)
    json.dump(rows, open(os.path.join(OUT, "funnel_results.json"), "w"), indent=2)
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()
