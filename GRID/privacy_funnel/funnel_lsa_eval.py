#!/usr/bin/env python3
"""Put the v2 funnel on the SAME axis as GRID / Fisher-Rao: run the real He et al.
attacks (Attack-0/1/6) under random AND near3 negatives, on the funnel's served
posteriors, baseline (lambda=0) vs funnel (lambda=3). Reuses the GRID attack
harness (run_attack.py) via standard attack-dataset JSONs.
"""
import os, sys, json, math, subprocess
import numpy as np
import scipy.sparse as sp
import torch, torch.nn as nn, torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
GRIDROOT = os.path.dirname(HERE)
ATTACK = os.path.join(GRIDROOT, "attack")
ADS = os.path.join(ATTACK, "data", "attack_dataset", "")
RES = os.path.join(HERE, "results", "lsa_eval")
PY = sys.executable
sys.path.insert(0, HERE); sys.path.insert(0, GRIDROOT); sys.path.insert(0, ATTACK)
import funnel as Fb
import funnel_v2 as V2
from lsa.data import preprocess_adj, preprocess_features
os.makedirs(RES, exist_ok=True)
torch.set_default_dtype(torch.float32)

DIRS = {"cora": "citeseer", "citeseer": "cora"}
LAM = 3.0
ATT = ["0", "1", "6"]
EPOCHS = 300


def train_P(ds, lam_max):
    A, X, y = Fb.load_lcc(ds); n = A.shape[0]; C = int(y.max()) + 1
    support = preprocess_adj(A); Xt = torch.tensor(preprocess_features(sp.csr_matrix(X))); yt = torch.tensor(y, dtype=torch.long)
    tr = torch.tensor(Fb.splits(n)[0]); te = Fb.splits(n)[2]
    r0, c0 = sp.triu(A, 1).nonzero(); E = np.stack([r0, c0], 1)
    torch.manual_seed(0); enc = Fb.GCN(Xt.shape[1], C)
    disc = nn.Sequential(nn.Linear(2 * C, 64), nn.ReLU(), nn.Linear(64, 1))
    opt = torch.optim.Adam(list(enc.parameters()) + list(disc.parameters()), lr=0.01, weight_decay=5e-4)
    for ep in range(EPOCHS):
        lam = lam_max * (2.0 / (1.0 + math.exp(-10 * ep / EPOCHS)) - 1.0)
        pos, neg = Fb.same_class_pairs(E, y, n, 1, seed=ep)
        pi = torch.tensor(np.r_[pos[:, 0], neg[:, 0]]); pj = torch.tensor(np.r_[pos[:, 1], neg[:, 1]])
        lab = torch.tensor(np.r_[np.ones(len(pos)), np.zeros(len(neg))], dtype=torch.float32)
        enc.train(); disc.train(); opt.zero_grad()
        logits = enc(Xt, support); p = F.softmax(logits, 1)
        feat = torch.cat([(p[pi] - p[pj]).abs(), p[pi] * p[pj]], 1)
        dloss = F.binary_cross_entropy_with_logits(disc(V2.grl(feat, lam)).squeeze(1), lab)
        (F.cross_entropy(logits[tr], yt[tr]) + dloss).backward(); opt.step()
    enc.eval()
    with torch.no_grad(): P = F.softmax(enc(Xt, support), 1).numpy()
    return A, X, y, P, float((P.argmax(1)[te] == y[te]).mean())


def train_g(X, y):
    Xt = torch.tensor(preprocess_features(sp.csr_matrix(X))); yt = torch.tensor(y, dtype=torch.long)
    tr = torch.tensor(Fb.splits(len(y))[0]); torch.manual_seed(0)
    m = nn.Sequential(nn.Linear(Xt.shape[1], 16), nn.ReLU(), nn.Linear(16, int(y.max()) + 1))
    opt = torch.optim.Adam(m.parameters(), lr=0.01, weight_decay=5e-4)
    for _ in range(200):
        m.train(); opt.zero_grad(); F.cross_entropy(m(Xt)[tr], yt[tr]).backward(); opt.step()
    m.eval()
    with torch.no_grad(): return F.softmax(m(Xt), 1).numpy()


def nbrs_of(A):
    A = A.tocsr(); return [A.indices[A.indptr[i]:A.indptr[i + 1]].tolist() for i in range(A.shape[0])]


def exact_k(nbrs, s, k):
    seen = {s}; fr = {s}
    for _ in range(k):
        nx = set()
        for u in fr: nx.update(int(v) for v in nbrs[u])
        nx -= seen; seen |= nx; fr = nx
        if not fr: break
    return list(fr)


def sample_pairs(A, nbrs, n, mode, seed):
    import random
    rng = random.Random(seed)
    r, c = sp.triu(A, 1).nonzero(); E = list(zip(r.tolist(), c.tolist())); eset = set(E)
    links = list(E); rng.shuffle(links); npos = len(links); negs = []; seen = set()
    if mode == "near3":
        order = list(range(n)); rng.shuffle(order); i = 0
        while len(negs) < npos and i < n * 8:
            s = order[i % n]; i += 1; far = exact_k(nbrs, s, 3)
            if not far: continue
            t = int(far[rng.randrange(len(far))]); a, b = (s, t) if s < t else (t, s)
            if a != b and (a, b) not in eset and (a, b) not in seen: negs.append((a, b)); seen.add((a, b))
    else:
        while len(negs) < npos:
            a, b = rng.randrange(n), rng.randrange(n)
            if a > b: a, b = b, a
            if a != b and (a, b) not in eset and (a, b) not in seen: negs.append((a, b)); seen.add((a, b))
    m = min(len(links), len(negs)); return np.array(links[:m]), np.array(negs[:m])


def write_ds(name, links, negs, f, g, feat):
    def rec(lb, i, j): return {"label": lb, "gcn_pred0": f[i], "gcn_pred1": f[j], "dense_pred0": g[i],
                               "dense_pred1": g[j], "feature_arr0": feat[i], "feature_arr1": feat[j], "id_pair": [int(i), int(j)]}
    allr, tr, te = [], [], []; cut = int(0.5 * len(links))
    for k in range(len(links)):
        p = rec(1, links[k][0], links[k][1]); q = rec(0, negs[k][0], negs[k][1]); allr += [p, q]
        (tr if k < cut else te).extend([p, q])
    for rows, suf in [(allr, "all"), (tr, "train"), (te, "test")]:
        with open(os.path.join(ADS, f"{name}_attack_{suf}.json"), "w") as w:
            for r in rows: w.write(json.dumps(r) + "\n")


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode: sys.stderr.write(p.stdout[-1200:] + p.stderr[-1200:]); raise SystemExit("fail")


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "0,1,6", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES, "--runs", "5", "--tag", tag, "--quiet"])
    return out


def parse(p):
    d = json.load(open(p)); return {a: d["results"][a].get("auc_mean", d["results"][a].get("auc")) for a in ATT if a in d.get("results", {})}


def main():
    D = {}
    for ds in DIRS:
        A, X, y, P0, a0 = train_P(ds, 0.0)
        _, _, _, Pf, af = train_P(ds, LAM)
        g = train_g(X, y)
        D[ds] = dict(A=A, nbrs=nbrs_of(A), feat=X.tolist(), g=g.tolist(), P0=P0.tolist(), Pf=Pf.tolist(), a0=a0, af=af)
        print(f"[{ds}] node acc  baseline={a0*100:.1f}%  funnel(λ={LAM})={af*100:.1f}%", flush=True)

    res = {}
    for mode in ("random", "near3"):
        for ds in DIRS:
            d = D[ds]; n = d["A"].shape[0]
            links, negs = sample_pairs(d["A"], d["nbrs"], n, mode, seed=1234)
            write_ds(f"{ds}_{mode}_b", links, negs, d["P0"], d["g"], d["feat"])
            write_ds(f"{ds}_{mode}_f", links, negs, d["Pf"], d["g"], d["feat"])
        for ds, shadow in DIRS.items():
            res[(mode, ds, "base")] = parse(run_attack(f"{ds}_{mode}_b", f"{shadow}_{mode}_b", f"{ds}_{mode}_base"))
            res[(mode, ds, "fun")] = parse(run_attack(f"{ds}_{mode}_f", f"{shadow}_{mode}_b", f"{ds}_{mode}_fun"))

    names = {"0": "Attack-0", "1": "Attack-1", "6": "Attack-6"}
    csv = os.path.join(RES, "funnel_lsa.csv")
    with open(csv, "w") as fcsv:
        fcsv.write("eval,dataset,attack,baseline_auc,funnel_auc,drop\n")
        for mode in ("random", "near3"):
            for ds in DIRS:
                b, fn = res[(mode, ds, "base")], res[(mode, ds, "fun")]
                print(f"\n=== {ds} / {mode} eval  (baseline acc {D[ds]['a0']*100:.0f}% -> funnel {D[ds]['af']*100:.0f}%) ===", flush=True)
                for a in ATT:
                    if a in b and a in fn:
                        print(f"  {names[a]:9s} {b[a]:.3f} -> {fn[a]:.3f}  ({fn[a]-b[a]:+.3f})", flush=True)
                        fcsv.write(f"{mode},{ds},{names[a]},{b[a]:.4f},{fn[a]:.4f},{fn[a]-b[a]:+.4f}\n")
    for fnm in os.listdir(ADS):
        if any(t in fnm for t in ("_random_b_attack_", "_random_f_attack_", "_near3_b_attack_", "_near3_f_attack_")):
            os.remove(os.path.join(ADS, fnm))
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
