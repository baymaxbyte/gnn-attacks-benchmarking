#!/usr/bin/env python3
"""Step-1 diagnostic (author hint #2): compare the full SIMILARITY DISTRIBUTIONS
of adjacent vs n-hop-indirect vs random-unlinked node pairs, BEFORE and AFTER
GRID -- rather than relying only on the average cosine.

Self-contained: reads the served posteriors the attack actually used
(attack/data/pred/{ds}_kronbb_whole_pred.pkl = before,
 ..._grid_pred.pkl = after, theta=0.4 n=3) and the graph adjacency. No attack
datasets / no retraining needed.

Outputs, per dataset:
  * histograms (cosine sim) of adj_before, adj_after, nhop, random-unlinked
  * a "similarity-only AUC": how well a single similarity score separates
    LINKED (adjacent) from UNLINKED pairs -- the ceiling a similarity-based
    attack could reach -- computed with (a) random-unlinked negatives [what the
    He et al. attack samples] and (b) n-hop negatives [GRID's design floor],
    before and after GRID, for both cosine and GRID's corr+cos metric.
"""
import os, sys, pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid_defense as G  # build_adjacency, n_hop_nodes

PRED = os.path.join(HERE, "attack", "data", "pred")
OUT = os.path.join(HERE, "results", "diag")
os.makedirs(OUT, exist_ok=True)
EPS = 1e-12
rng = np.random.RandomState(0)


def cosine(A, B):
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + EPS)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + EPS)
    return (A * B).sum(1)


def corr_plus_cos(A, B):
    Ac = A - A.mean(1, keepdims=True)
    Bc = B - B.mean(1, keepdims=True)
    corr = (Ac * Bc).sum(1) / (np.linalg.norm(Ac, axis=1) * np.linalg.norm(Bc, axis=1) + EPS)
    return corr + cosine(A, B)


def auc(pos, neg):
    """P(score(linked) > score(unlinked)); tie-aware (Mann-Whitney U)."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    r = rankdata(np.concatenate([pos, neg]))
    n1 = len(pos)
    U = r[:n1].sum() - n1 * (n1 + 1) / 2.0
    return U / (n1 * len(neg))


def pct(x):
    return (f"mean={np.mean(x):.3f} med={np.median(x):.3f} "
            f"p10={np.percentile(x,10):.3f} p90={np.percentile(x,90):.3f}")


def run(ds, max_pairs=4000):
    V = np.asarray(pickle.load(open(os.path.join(PRED, f"{ds}_kronbb_whole_pred.pkl"), "rb")), float)
    R = np.asarray(pickle.load(open(os.path.join(PRED, f"{ds}_kronbb_whole_grid_pred.pkl"), "rb")), float)
    _, nbrs = G.build_adjacency(ds)
    n = V.shape[0]
    assert len(nbrs) == n == R.shape[0], (len(nbrs), n, R.shape)

    adjset = set()
    adj = []
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            if j > i:
                adj.append((i, j)); adjset.add((i, j))
    adj = np.array(adj)

    # exactly-3-hop indirect pairs
    nh = []
    order = rng.permutation(n)
    for s in order:
        if len(nh) >= max_pairs:
            break
        far = G.n_hop_nodes(nbrs, int(s), 3)
        if far:
            nh.append((int(s), int(far[rng.randint(len(far))])))
    nh = np.array(nh)

    # random UNLINKED pairs (what the attack samples as negatives)
    ru = []
    while len(ru) < max_pairs:
        i, j = rng.randint(n), rng.randint(n)
        if i == j:
            continue
        a, b = (i, j) if i < j else (j, i)
        if (a, b) in adjset:
            continue
        ru.append((a, b))
    ru = np.array(ru)

    def sims(pairs, M, fn):
        return fn(M[pairs[:, 0]], M[pairs[:, 1]])

    res = {}
    for tag, fn in [("cos", cosine), ("corrcos", corr_plus_cos)]:
        res[tag] = dict(
            adj_b=sims(adj, V, fn), adj_a=sims(adj, R, fn),
            nh_b=sims(nh, V, fn), nh_a=sims(nh, R, fn),
            ru_b=sims(ru, V, fn), ru_a=sims(ru, R, fn),
        )

    print(f"\n================  {ds.upper()}  (whole KRON box, theta=0.4, n=3)  ================")
    print(f"pairs: adjacent={len(adj)}  3-hop={len(nh)}  random-unlinked={len(ru)}  nodes={n}")
    for tag in ("cos", "corrcos"):
        d = res[tag]
        name = "cosine" if tag == "cos" else "corr+cos (GRID metric)"
        print(f"\n--- {name} ---")
        print(f"  adjacent   before : {pct(d['adj_b'])}")
        print(f"  adjacent   AFTER  : {pct(d['adj_a'])}")
        print(f"  3-hop      before : {pct(d['nh_b'])}   (GRID's target floor)")
        print(f"  rand-unlink before: {pct(d['ru_b'])}   (attack's negatives)")
        auc_ru_b = auc(d['adj_b'], d['ru_b']); auc_ru_a = auc(d['adj_a'], d['ru_a'])
        auc_nh_b = auc(d['adj_b'], d['nh_b']); auc_nh_a = auc(d['adj_a'], d['nh_a'])
        print(f"  similarity-only AUC  linked-vs-RANDOM-unlinked : {auc_ru_b:.3f} -> {auc_ru_a:.3f}  (after GRID)")
        print(f"  similarity-only AUC  linked-vs-3hop            : {auc_nh_b:.3f} -> {auc_nh_a:.3f}  (after GRID)")

    # ---- plot (cosine) ----
    d = res["cos"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(d['ru_b'].min(), d['adj_a'].min()) - 0.02, 1.0, 60)
    ax.hist(d['adj_b'], bins=bins, density=True, alpha=0.5, color="#d62728", label="adjacent (before)")
    ax.hist(d['adj_a'], bins=bins, density=True, alpha=0.5, color="#ff7f0e", label="adjacent (AFTER GRID)")
    ax.hist(d['nh_b'],  bins=bins, density=True, alpha=0.45, color="#2ca02c", label="3-hop indirect (floor)")
    ax.hist(d['ru_b'],  bins=bins, density=True, alpha=0.40, color="#1f77b4", label="random unlinked (attack neg.)")
    ax.axvline(np.mean(d['nh_b']), color="#2ca02c", ls="--", lw=1)
    ax.set_xlabel("cosine similarity of posterior pair")
    ax.set_ylabel("density")
    ax.set_title(f"{ds} (whole KRON box): similarity distributions, GRID theta=0.4 n=3")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(OUT, f"sim_dist_{ds}_whole.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    print(f"  wrote {os.path.relpath(p, HERE)}")
    return res


if __name__ == "__main__":
    for ds in ("cora", "citeseer"):
        run(ds)
