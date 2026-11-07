"""Full paper-faithful target protocol (all 3 points):
  (1) 2-layer Kipf GCN, 16 hidden        (2) trained on D1 = 40% node subsample
  with 20% of labels removed              (3) standard ~200-epoch training.
The attack then runs on D1 (induced subgraph), baseline vs GRID (theta=0.4,n=3),
shadow = the other dataset's D1. Measures whether the weaker/noisier training
set softens posteriors enough to reproduce the paper's large GRID drops.
"""
import os, sys, json, pickle, subprocess
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(HERE, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
DSDIR = os.path.join(ATTACK, "data", "dataset", "original")
RES = os.path.join(HERE, "results", "attacks")
PY = sys.executable
sys.path.insert(0, GRIDROOT); sys.path.insert(0, ATTACK)
import grid_defense as G
from lsa.data import load_data_original, preprocess_adj, preprocess_features

DIRS = {"cora": "citeseer", "citeseer": "cora"}
ATTACKS = [str(i) for i in range(8)]
FRAC, REMOVE_LABELS = 0.40, 0.20
HIDDEN, DROPOUT, LR, WD, EPOCHS, PATIENCE = 16, 0.5, 0.01, 5e-4, 200, 20
DP = os.path.join(DSDIR, "")


def load_full(ds):
    adj, feat, *_rest = load_data_original(DP, ds)
    y_test = _rest[2]
    labels = np.asarray(y_test).argmax(1)
    X = np.asarray(feat.todense() if sp.issparse(feat) else feat, dtype=np.float32)
    return sp.csr_matrix(adj), X, labels


def build_d1(ds, seed=0):
    adj, X, y = load_full(ds)
    n = adj.shape[0]
    rng = np.random.RandomState(seed)
    sel = np.sort(rng.choice(n, int(FRAC * n), replace=False))
    sub = adj[sel][:, sel].tocoo()
    ei = np.vstack([sub.row, sub.col]).astype(np.int64)
    Xs, ys = X[sel], y[sel]
    np.savez(os.path.join(DSDIR, f"{ds}_d1.npz"), n_nodes=len(sel),
             edge_index=ei, features=Xs, labels=ys)
    return len(sel), ei, Xs, ys


def _support_nbrs(n, ei):
    A = sp.coo_matrix((np.ones(ei.shape[1]), (ei[0], ei[1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(float).tolil(); A.setdiag(0); A = sp.csr_matrix(A); A.eliminate_zeros()
    support = preprocess_adj(A)
    nbrs = [A.indices[A.indptr[i]:A.indptr[i + 1]] for i in range(n)]
    return support, nbrs


class GCN(nn.Module):
    def __init__(s, d, h, c):
        super().__init__(); s.W0 = nn.Linear(d, h, bias=False); s.W1 = nn.Linear(h, c, bias=False)
    def forward(s, X, A):
        h = F.relu(torch.sparse.mm(A, s.W0(X))); h = F.dropout(h, DROPOUT, training=s.training)
        return torch.sparse.mm(A, s.W1(h))


class MLP(nn.Module):               # graph-free reference g
    def __init__(s, d, h, c):
        super().__init__(); s.W0 = nn.Linear(d, h); s.W1 = nn.Linear(h, c)
    def forward(s, X):
        return s.W1(F.dropout(F.relu(s.W0(X)), DROPOUT, training=s.training))


def adj_cosine(P, nbrs):
    Pn = P / (np.linalg.norm(P, axis=1, keepdims=True) + 1e-12)
    m = [float((Pn[i] * Pn[int(j)]).sum()) for i in range(len(nbrs)) for j in nbrs[i] if int(j) > i]
    return float(np.mean(m)) if m else 0.0


def train_target(ds, ei, X, y, seed=0):
    n = len(y); support, nbrs = _support_nbrs(n, ei)
    Xt = torch.tensor(preprocess_features(sp.csr_matrix(X)), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    rng = np.random.RandomState(seed); perm = rng.permutation(n)
    n_lab = int((1 - REMOVE_LABELS) * n)                      # 80% labelled, 20% removed
    tr = torch.tensor(perm[:n_lab]); va = torch.tensor(perm[n_lab:])
    torch.manual_seed(seed); m = GCN(Xt.shape[1], HIDDEN, int(y.max()) + 1)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    best, bst, bad = 1e9, None, 0
    for ep in range(EPOCHS):
        m.train(); opt.zero_grad(); F.cross_entropy(m(Xt, support)[tr], yt[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl = float(F.cross_entropy(m(Xt, support)[va], yt[va]))
        if vl < best - 1e-4: best, bst, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PATIENCE: break
    m.load_state_dict(bst); m.eval()
    with torch.no_grad():
        post = F.softmax(m(Xt, support), 1).numpy()
    acc = float((post.argmax(1)[va] == y[va]).mean())
    print(f"[{ds} D1] n={n} edges={ei.shape[1]//2} held-out_acc={acc*100:.1f}% "
          f"mean_peak={post.max(1).mean():.3f} median_peak={np.median(post.max(1)):.3f} "
          f"adj_cos={adj_cosine(post, nbrs):.3f}")
    pickle.dump(post.astype(np.float64), open(os.path.join(PRED, f"{ds}_d1_tgt_pred.pkl"), "wb"))
    return post, nbrs


def train_ref(ds, X, y, seed=123):
    n = len(y); Xt = torch.tensor(preprocess_features(sp.csr_matrix(X)), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    rng = np.random.RandomState(seed); idx = torch.tensor(np.sort(rng.permutation(n)[:int(0.10 * n)]))
    torch.manual_seed(0); m = MLP(Xt.shape[1], HIDDEN, int(y.max()) + 1)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad(); F.cross_entropy(m(Xt)[idx], yt[idx]).backward(); opt.step()
    m.eval()
    with torch.no_grad():
        post = F.softmax(m(Xt), 1).numpy()
    pickle.dump(post.astype(np.float64), open(os.path.join(PRED, f"{ds}_d1_ref_pred.pkl"), "wb"))


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:]); raise SystemExit("fail " + " ".join(cmd))


def build_attack_ds(ds, tgt, name):
    sh([PY, "partial_graph_generation.py", "--dataset", f"{ds}_d1",
        "--datapath", "data/dataset/original/", "--prediction_path", "data/pred/",
        "--saving_path", "data/attack_dataset/", "--target_pred", tgt,
        "--reference_pred", f"{ds}_d1_ref_pred.pkl", "--name", name, "--seed", "1"])


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "all", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES,
        "--runs", "5", "--tag", tag, "--link_prediction", "--quiet"])
    return out


def parse(p):
    d = json.load(open(p)); out = {a: d["results"][a].get("auc_mean", d["results"][a].get("auc"))
                                   for a in ATTACKS if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict): out["link_pred"] = lp.get("auc_mean", lp.get("auc"))
    return out


def main():
    os.makedirs(RES, exist_ok=True)
    for ds in DIRS:
        n, ei, X, y = build_d1(ds)
        post, nbrs = train_target(ds, ei, X, y)
        train_ref(ds, X, y)
        R, st = G.grid_posteriors(post, nbrs, dsname=f"d1:{ds}", theta=0.4, n_hop=3, verbose=True)
        pickle.dump(R, open(os.path.join(PRED, f"{ds}_d1_grid_pred.pkl"), "wb"))
        build_attack_ds(ds, f"{ds}_d1_tgt_pred.pkl", f"{ds}_d1")
        build_attack_ds(ds, f"{ds}_d1_grid_pred.pkl", f"{ds}_d1_grid")
    res = {}
    for ds, shadow in DIRS.items():
        b = run_attack(f"{ds}_d1", f"{shadow}_d1", f"{ds}_base")
        g = run_attack(f"{ds}_d1_grid", f"{shadow}_d1", f"{ds}_grid")
        res[ds] = (parse(b), parse(g))
    names = {**{str(i): f"Attack-{i}" for i in range(8)}, "link_pred": "link_pred"}
    csv = os.path.join(HERE, "results", "d1_grid.csv")
    with open(csv, "w") as f:
        f.write("dataset,attack,auc_no_defense,auc_grid,delta\n")
        for ds in DIRS:
            base, grid = res[ds]
            print(f"\n=== {ds} D1 (no-defense -> GRID) ===")
            for a in ATTACKS + ["link_pred"]:
                if a in base and a in grid:
                    print(f"  {names[a]:10s} {base[a]:.3f} -> {grid[a]:.3f}  ({grid[a]-base[a]:+.3f})")
                    f.write(f"{ds},{names[a]},{base[a]:.4f},{grid[a]:.4f},{grid[a]-base[a]:+.4f}\n")
    print("\nwrote", csv)


if __name__ == "__main__":
    main()
