#!/usr/bin/env python3
"""Step 3: does the evaluation negative-sampling (the paper's D4) explain the
GRID reproduction gap?

Runs the REAL He et al. attacks (0 unsupervised, 1 shadow, 6 gray-box) on the
paper-faithful SOFT GCN (cora/citeseer, adjacent cosine ~0.93-0.97), baseline vs
GRID (theta=0.4, n=3), under three evaluation negative-sampling schemes:

  random : uniform unlinked pairs over all nodes   (literal He et al.; our setup)
  d1     : pairs restricted to a 40% node subgraph  (literal paper D4, Sec 5.1.3)
  near3  : negatives = unlinked pairs at graph distance EXACTLY 3
           (= GRID's "n-hop indirect neighbour" disguise floor; the mechanistic
           probe suggested by step-1)

Base and GRID share IDENTICAL evaluation pairs (same seed); only the served
posteriors differ, so any AUC change is purely the defence. Reuses run_attack.py
unchanged by writing standard {name}_attack_{all,train,test}.json files.
"""
import os, sys, json, pickle, random, subprocess
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
PRED = os.path.join(ATTACK, "data", "pred")
DS = os.path.join(ATTACK, "data", "dataset", "original", "")
ADS = os.path.join(ATTACK, "data", "attack_dataset", "")
RES = os.path.join(HERE, "results", "d4")
PY = sys.executable
sys.path.insert(0, ATTACK)
from lsa.data import load_data

DIRS = {"cora": "citeseer", "citeseer": "cora"}
MODES = ["random", "d1", "near3"]
ATTACKS = ["0", "1", "6"]
FRAC_D1 = 0.40
RUNS = 5
os.makedirs(RES, exist_ok=True)
os.makedirs(ADS, exist_ok=True)


def nbrs_of(adj):
    adj = adj.tocsr()
    return [adj.indices[adj.indptr[i]:adj.indptr[i + 1]].tolist() for i in range(adj.shape[0])]


def exact_k(nbrs, src, k):
    seen = {src}; frontier = {src}
    for _ in range(k):
        nxt = set()
        for u in frontier:
            nxt.update(int(v) for v in nbrs[u])
        nxt -= seen; seen |= nxt; frontier = nxt
        if not frontier:
            break
    return list(frontier)


def sample_pairs(adj, nbrs, N, mode, seed):
    """Return (links, negs) as equal-length lists of (i,j), i<j. Deterministic
    in `seed` and independent of the posteriors, so base and GRID reuse them."""
    rng = random.Random(seed)
    rows, cols = adj.nonzero()
    existing = set((int(r), int(c)) for r, c in zip(rows.tolist(), cols.tolist()) if r < c)

    subset = None
    if mode == "d1":
        srng = np.random.RandomState(1000 + seed)
        subset = set(int(x) for x in srng.choice(N, int(FRAC_D1 * N), replace=False))

    links = [(r, c) for (r, c) in existing
             if subset is None or (r in subset and c in subset)]
    rng.shuffle(links)
    npos = len(links)

    negs, seen = [], set()
    if mode == "near3":
        order = list(range(N)); rng.shuffle(order)
        i = 0
        while len(negs) < npos and i < N * 8:
            s = order[i % N]; i += 1
            far = exact_k(nbrs, s, 3)
            if not far:
                continue
            t = int(far[rng.randrange(len(far))])
            a, b = (s, t) if s < t else (t, s)
            if a != b and (a, b) not in existing and (a, b) not in seen:
                negs.append((a, b)); seen.add((a, b))
    else:
        pool = list(subset) if subset is not None else None
        guard = 0
        while len(negs) < npos and guard < npos * 200 + 10000:
            guard += 1
            a = rng.choice(pool) if pool else rng.randrange(N)
            b = rng.choice(pool) if pool else rng.randrange(N)
            if a > b:
                a, b = b, a
            if a != b and (a, b) not in existing and (a, b) not in seen:
                negs.append((a, b)); seen.add((a, b))
    m = min(len(links), len(negs))
    return links[:m], negs[:m]


def rec(label, i, j, f, g, feat):
    return {"label": label, "gcn_pred0": f[i], "gcn_pred1": f[j],
            "dense_pred0": g[i], "dense_pred1": g[j],
            "feature_arr0": feat[i], "feature_arr1": feat[j],
            "id_pair": [int(i), int(j)]}


def write_ds(name, links, negs, f, g, feat, split=0.5):
    allrows, train, test = [], [], []
    ntr = int(len(links) * split)
    for k in range(len(links)):
        p = rec(1, links[k][0], links[k][1], f, g, feat)
        n = rec(0, negs[k][0], negs[k][1], f, g, feat)
        allrows += [p, n]
        (train if k < ntr else test).extend([p, n])
    for rows, suf in [(allrows, "all"), (train, "train"), (test, "test")]:
        with open(os.path.join(ADS, f"{name}_attack_{suf}.json"), "w") as w:
            for r in rows:
                w.write(json.dumps(r) + "\n")


def load_ds(ds):
    adj, feat, *_ = load_data(DS, ds)
    feat = feat.toarray() if not isinstance(feat, np.ndarray) else feat
    g = pickle.load(open(os.path.join(PRED, f"{ds}_gkron_pred.pkl"), "rb"))
    fb = pickle.load(open(os.path.join(PRED, f"{ds}_softgcn_whole_pred.pkl"), "rb"))
    fg = pickle.load(open(os.path.join(PRED, f"{ds}_softgcn_whole_grid_pred.pkl"), "rb"))
    return (adj.tocsr(), nbrs_of(adj), feat.tolist(),
            np.asarray(g).tolist(), np.asarray(fb).tolist(), np.asarray(fg).tolist())


def sh(cmd):
    p = subprocess.run(cmd, cwd=ATTACK, capture_output=True, text=True)
    if p.returncode:
        sys.stderr.write("\n".join(cmd) + "\n" + p.stdout[-1500:] + p.stderr[-1500:])
        raise SystemExit("cmd failed")


def run_attack(target, shadow, tag):
    out = os.path.join(RES, f"{target}__{shadow}__{tag}.json")
    sh([PY, "run_attack.py", "--attack", "0,1,6", "--dataset", target, "--shadow", shadow,
        "--attack_dataset_path", "data/attack_dataset/", "--result_path", RES,
        "--runs", str(RUNS), "--tag", tag, "--link_prediction", "--quiet"])
    return out


def auc(e):
    return e.get("auc_mean", e.get("auc"))


def parse(p):
    d = json.load(open(p))
    out = {a: auc(d["results"][a]) for a in ATTACKS if a in d.get("results", {})}
    lp = d.get("link_prediction")
    if isinstance(lp, dict):
        out["link_pred"] = auc(lp)
    return out


def rm_mode(mode):
    for fn in os.listdir(ADS):
        if f"_{mode}_attack_" in fn or f"_{mode}_grid_attack_" in fn:
            os.remove(os.path.join(ADS, fn))


def main():
    data = {ds: load_ds(ds) for ds in DIRS}
    rowlog = []
    for mode in MODES:
        print(f"\n########## negative mode = {mode} ##########", flush=True)
        for ds in DIRS:
            adj, nbrs, feat, g, fb, fg = data[ds]
            N = adj.shape[0]
            links, negs = sample_pairs(adj, nbrs, N, mode, seed=hash((ds, mode)) % (2**31))
            print(f"  [{ds}/{mode}] pairs: {len(links)} linked + {len(negs)} unlinked", flush=True)
            write_ds(f"{ds}_{mode}", links, negs, fb, g, feat)
            write_ds(f"{ds}_{mode}_grid", links, negs, fg, g, feat)
        res = {}
        for ds, shadow in DIRS.items():
            b = run_attack(f"{ds}_{mode}", f"{shadow}_{mode}", f"{ds}_{mode}_base")
            gr = run_attack(f"{ds}_{mode}_grid", f"{shadow}_{mode}", f"{ds}_{mode}_grid")
            res[ds] = (parse(b), parse(gr))
        names = {"0": "Attack-0", "1": "Attack-1", "6": "Attack-6", "link_pred": "link_pred"}
        for ds in DIRS:
            base, grid = res[ds]
            print(f"\n  === {ds} / {mode} (no-defense -> GRID) ===", flush=True)
            for a in ["0", "1", "6", "link_pred"]:
                if a in base and a in grid:
                    print(f"    {names[a]:10s} {base[a]:.3f} -> {grid[a]:.3f}  ({grid[a]-base[a]:+.3f})", flush=True)
                    rowlog.append((mode, ds, names[a], base[a], grid[a], grid[a] - base[a]))
        rm_mode(mode)

    csv = os.path.join(RES, "negmode_results.csv")
    with open(csv, "w") as f:
        f.write("neg_mode,dataset,attack,auc_no_defense,auc_grid,delta\n")
        for mode, ds, a, b, g, d in rowlog:
            f.write(f"{mode},{ds},{a},{b:.4f},{g:.4f},{d:+.4f}\n")
    print(f"\nwrote {csv}", flush=True)


if __name__ == "__main__":
    main()
