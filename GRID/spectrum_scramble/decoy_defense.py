#!/usr/bin/env python3
"""todo #2 — decoy-graph defense: steps (a) accuracy + (b) link-stealing AUC.

Pipeline, per dataset (on the largest connected component):
  * rank edges by spectral cost (within-community = cheap), build decoy graphs
    = remove cheapest r|E| real edges + add cheapest r|E| decoy non-edges
    (spectral), or random edges (contrast).
  * TRAIN a 2-layer GCN on each graph variant -> posteriors + node-test accuracy.
  * ATTACK: positives are always the REAL edges; target_pred = each variant's
    posteriors -> does the attacker still recover the REAL links when the model
    was trained on the DECOY graph? (Attack-0/1/6, 5 runs.) Shadow = other
    dataset's real-trained box.

Reports node accuracy (a) and link-stealing AUC (b): real-trained vs decoy-trained.
"""
import os, sys, json, pickle, subprocess
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(GRIDROOT, "attack")
DSDIR = os.path.join(ATTACK, "data", "dataset", "original")
PRED = os.path.join(ATTACK, "data", "pred")
ADS = os.path.join(ATTACK, "data", "attack_dataset", "")
RES = os.path.join(HERE, "results", "attacks")
PY = sys.executable
sys.path.insert(0, GRIDROOT); sys.path.insert(0, ATTACK)
from lsa.data import load_data_original, preprocess_adj, preprocess_features
os.makedirs(RES, exist_ok=True)

DIRS = {"cora": "citeseer", "citeseer": "cora"}
CFGS = [("real", None, None), ("spec0.3", "spec", 0.3), ("spec0.5", "spec", 0.5), ("rand0.5", "rand", 0.5)]
KBAND, HID, DROP, LR, WD, EPOCHS, PAT = 64, 16, 0.5, 0.01, 5e-4, 200, 30
ATT = ["0", "1", "6"]
rng = np.random.RandomState(0)


def load_lcc(ds):
    adj, feat, *rest = load_data_original(DSDIR + os.sep, ds)
    y = np.asarray(rest[2]).argmax(1)                      # y_test covers all nodes
    X = feat.toarray() if sp.issparse(feat) else np.asarray(feat)
    A = sp.csr_matrix((adj + adj.T)); A.data[:] = 1.0; A = A.tolil(); A.setdiag(0)
    A = sp.csr_matrix(A); A.eliminate_zeros()
    nc, lab = connected_components(A, directed=False)
    idx = np.where(lab == np.bincount(lab).argmax())[0]
    A = sp.csr_matrix(A[idx][:, idx])
    return A, X[idx].astype(np.float64), y[idx].astype(int)


def spectral_rank(A):
    n = A.shape[0]
    L = sp.diags(np.asarray(A.sum(1)).ravel()) - A
    w, U = np.linalg.eigh(L.toarray())
    Ub = U[:, 1:KBAND + 1]
    r, c = sp.triu(A, 1).nonzero(); E = np.stack([r, c], 1)
    imp_E = ((Ub[E[:, 0]] - Ub[E[:, 1]]) ** 2).sum(1)
    Eset = set(map(tuple, E.tolist())); NE = []
    while len(NE) < len(E):
        i, j = rng.randint(n), rng.randint(n)
        if i > j: i, j = j, i
        if i != j and (i, j) not in Eset: NE.append((i, j))
    NE = np.array(NE); imp_NE = ((Ub[NE[:, 0]] - Ub[NE[:, 1]]) ** 2).sum(1)
    return E, NE, np.argsort(imp_E), np.argsort(imp_NE)


def build_adj(n, E, NE, oE, oNE, mode, r):
    if mode is None:
        keep = E
    else:
        m = int(r * len(E))
        if mode == "spec":
            rm = set(map(tuple, E[oE[:m]].tolist())); add = NE[oNE[:m]]
        else:
            rm = set(map(tuple, E[rng.permutation(len(E))[:m]].tolist())); add = NE[rng.permutation(len(NE))[:m]]
        keep = np.array([e for e in E.tolist() if tuple(e) not in rm] + add.tolist())
    A = sp.coo_matrix((np.ones(len(keep)), (keep[:, 0], keep[:, 1])), shape=(n, n))
    A = ((A + A.T) > 0).astype(np.float64).tolil(); A.setdiag(0)
    return sp.csr_matrix(A)


class GCN(nn.Module):
    def __init__(s, d, c):
        super().__init__(); s.W0 = nn.Linear(d, HID, bias=False); s.W1 = nn.Linear(HID, c, bias=False)
    def forward(s, X, A):
        h = F.relu(torch.sparse.mm(A, s.W0(X))); h = F.dropout(h, DROP, training=s.training)
        return torch.sparse.mm(A, s.W1(h))


class MLP(nn.Module):
    def __init__(s, d, c):
        super().__init__(); s.W0 = nn.Linear(d, HID); s.W1 = nn.Linear(HID, c)
    def forward(s, X): return s.W1(F.dropout(F.relu(s.W0(X)), DROP, training=s.training))


def splits(n, seed=0):
    p = np.random.RandomState(seed).permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    return p[:a], p[a:b], p[b:]


def train_gcn(A, X, y, sp_):
    support = preprocess_adj(A)
    Xt = torch.tensor(preprocess_features(sp.csr_matrix(X)), dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    tr, va, te = (torch.tensor(s) for s in sp_)
    torch.manual_seed(0); m = GCN(Xt.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    best, bs, bad = 1e9, None, 0
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad(); F.cross_entropy(m(Xt, support)[tr], yt[tr]).backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = float(F.cross_entropy(m(Xt, support)[va], yt[va]))
        if vl < best - 1e-4: best, bs, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= PAT: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): post = F.softmax(m(Xt, support), 1).numpy()
    acc = float((post.argmax(1)[te] == y[te]).mean())
    return post.astype(np.float64), acc


def train_mlp(X, y, sp_):
    Xt = torch.tensor(preprocess_features(sp.csr_matrix(X)), dtype=torch.float32); yt = torch.tensor(y, dtype=torch.long)
    tr = torch.tensor(sp_[0]); torch.manual_seed(0); m = MLP(Xt.shape[1], int(y.max()) + 1)
    opt = torch.optim.Adam(m.parameters(), lr=LR, weight_decay=WD)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad(); F.cross_entropy(m(Xt)[tr], yt[tr]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return F.softmax(m(Xt), 1).numpy().astype(np.float64)


def save_npz(name, A, X, y):
    r, c = sp.triu(A, 1).nonzero()
    np.savez(os.path.join(DSDIR, f"{name}.npz"), n_nodes=A.shape[0],
             edge_index=np.stack([r, c]).astype(np.int64), features=X, labels=y)


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write(p.stdout[-1500:] + p.stderr[-1500:]); raise SystemExit("fail " + " ".join(cmd))


def pgg(dataset, target_pred, ref, name):
    sh([PY, "partial_graph_generation.py", "--dataset", dataset, "--datapath", "data/dataset/original/",
        "--prediction_path", "data/pred/", "--saving_path", "data/attack_dataset/",
        "--target_pred", target_pred, "--reference_pred", ref, "--name", name, "--seed", "1"])


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "0,1,6", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES, "--runs", "5",
        "--tag", tag, "--link_prediction", "--quiet"])
    return out


def parse(p):
    d = json.load(open(p)); o = {a: d["results"][a].get("auc_mean", d["results"][a].get("auc")) for a in ATT if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict): o["link_pred"] = lp.get("auc_mean", lp.get("auc"))
    return o


def main():
    accs = {}
    for ds in DIRS:
        A, X, y = load_lcc(ds); n = A.shape[0]
        sp_ = splits(n)
        E, NE, oE, oNE = spectral_rank(A)
        print(f"[{ds}] LCC n={n} |E|={len(E)} classes={int(y.max())+1}", flush=True)
        save_npz(f"{ds}_sc_real", A, X, y)                       # positives source (real edges)
        ref = f"{ds}_sc_ref_pred.pkl"
        pickle.dump(train_mlp(X, y, sp_), open(os.path.join(PRED, ref), "wb"))
        for cfg, mode, r in CFGS:
            Acfg = build_adj(n, E, NE, oE, oNE, mode, r)
            post, acc = train_gcn(Acfg, X, y, sp_)
            accs[(ds, cfg)] = acc
            pickle.dump(post, open(os.path.join(PRED, f"{ds}_{cfg}_pred.pkl"), "wb"))
            ec = sp.triu(Acfg, 1).nnz
            print(f"  [{ds}/{cfg}] |E'|={ec} node_test_acc={acc*100:.1f}%", flush=True)
        # shadow = this dataset's real-trained attack dataset
        pgg(f"{ds}_sc_real", f"{ds}_real_pred.pkl", ref, f"{ds}_shadow")

    res = {}
    for ds, shadow in DIRS.items():
        for cfg, _, _ in CFGS:
            pgg(f"{ds}_sc_real", f"{ds}_{cfg}_pred.pkl", f"{ds}_sc_ref_pred.pkl", f"{ds}_{cfg}_atk")
            res[(ds, cfg)] = parse(run_attack(f"{ds}_{cfg}_atk", f"{shadow}_shadow", f"{ds}_{cfg}"))

    names = {"0": "Attack-0", "1": "Attack-1", "6": "Attack-6", "link_pred": "link_pred"}
    csv = os.path.join(RES, "decoy_defense_auc.csv")
    with open(csv, "w") as f:
        f.write("dataset,config,node_test_acc,attack,auc,auc_drop_vs_real\n")
        for ds in DIRS:
            base = res[(ds, "real")]
            print(f"\n===== {ds}  (node acc | link-steal AUC, drop vs real-trained) =====", flush=True)
            for cfg, _, _ in CFGS:
                r = res[(ds, cfg)]
                print(f"  [{cfg:8s}] node_acc={accs[(ds,cfg)]*100:4.1f}%  "
                      + "  ".join(f"{names[a]}={r[a]:.3f}({r[a]-base[a]:+.3f})" for a in ATT if a in r), flush=True)
                for a in ATT + ["link_pred"]:
                    if a in r:
                        f.write(f"{ds},{cfg},{accs[(ds,cfg)]:.4f},{names[a]},{r[a]:.4f},{r[a]-base.get(a,r[a]):+.4f}\n")
    for fn in os.listdir(ADS):
        if "_sc_real_attack_" in fn or "_shadow_attack_" in fn or "_atk_attack_" in fn:
            os.remove(os.path.join(ADS, fn))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
