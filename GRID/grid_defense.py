#!/usr/bin/env python3
"""GRID: Graph Link Disguise  (Lou et al., arXiv 2501.10985v2).

A post-hoc defence against similarity-based link-stealing attacks. It adds a
noise vector s to the served posterior v of a *subset* of nodes (the "core
nodes", a vertex cover over the high-similarity edges) so that the similarity
sim(v_i, v_j) of adjacent nodes is dragged down to the level of n-hop indirect
neighbours -- without changing any predicted label.

Implements Algorithm 1 (threshold-based core-node selection), the OPT-GRID
objective (minimise the adjacent-vs-n-hop similarity gap D_i), and Algorithm 3
(ConstraintCheck projection), solved by projected gradient descent (Algorithm 2
with the KKT multipliers folded into the projection).

sim(a,b) = Pearson-correlation(a,b) + cosine(a,b)      [paper Eqn. 3]

Guarantees enforced per node (verified empirically after running):
  * zero label loss        argmax(v) == argmax(v+s)        [L1 / Eqn. 5]
  * valid distribution      v+s >= 0  and  sum(v+s) == 1    [Eqn. 6]
  * distortion budget       ||s||_1 <= theta                [Eqn. 7]

Only the *target* (served) posteriors are perturbed; the attacker's reference /
shadow models are left untouched, matching the paper's threat model.
"""
import argparse
import os
import pickle
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ATTACK = os.path.join(HERE, "attack")
sys.path.insert(0, ATTACK)
from lsa.data import load_data  # noqa: E402  (reuse the exact attack-side loader)

DATAPATH = os.path.join(ATTACK, "data", "dataset", "original") + os.sep
POSTERIORS = os.path.join(HERE, "artifacts", "posteriors")
PRED = os.path.join(ATTACK, "data", "pred")
EPS = 1e-12


# ----------------------------------------------------------------- similarity
def _sim_rows(x, M):
    """Per-row sim(x, M[r]) = pearson(x,M[r]) + cosine(x,M[r]). x:[C], M:[m,C]."""
    xc = x - x.mean()
    Mc = M - M.mean(dim=1, keepdim=True)
    corr = (Mc * xc).sum(1) / (Mc.norm(dim=1) * xc.norm() + EPS)
    cos = (M * x).sum(1) / (M.norm(dim=1) * x.norm() + EPS)
    return corr + cos


def _sim_np(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    ac, bc = a - a.mean(), b - b.mean()
    corr = float((ac * bc).sum() / (np.linalg.norm(ac) * np.linalg.norm(bc) + EPS))
    cos = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + EPS))
    return corr + cos


# -------------------------------------------------------- graph neighbourhoods
def build_adjacency(dataset):
    adj = load_data(DATAPATH, dataset)[0]          # scipy csr, node order == posterior rows
    adj = adj.tocsr()
    indptr, indices = adj.indptr, adj.indices
    nbrs = [indices[indptr[i]:indptr[i + 1]] for i in range(adj.shape[0])]
    return adj, nbrs


def n_hop_nodes(nbrs, src, n):
    """Nodes whose shortest-path distance from src is exactly n (BFS)."""
    seen = {src}
    frontier = {src}
    for _ in range(n):
        nxt = set()
        for u in frontier:
            nxt.update(int(v) for v in nbrs[u])
        nxt -= seen
        seen |= nxt
        frontier = nxt
        if not frontier:
            break
    return list(frontier)


# ------------------------------------------------- Algorithm 1: core selection
def select_core_nodes(V, nbrs, delta):
    """Vertex cover over edges whose endpoint-similarity >= delta; of each such
    edge's two endpoints, keep the higher-(similarity-weighted-)degree one."""
    n = V.shape[0]
    deg = np.zeros(n)
    edges = []
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            if j <= i:
                continue
            w = _sim_np(V[i], V[j])
            deg[i] += w
            deg[j] += w
            edges.append((w, i, j))
    edges.sort(key=lambda e: e[0], reverse=True)   # descending weight
    core = set()
    for w, i, j in edges:
        if w < delta:
            continue                               # low-similarity edge: skip
        if i in core or j in core:
            continue                               # edge already covered
        core.add(i if deg[i] >= deg[j] else j)
    return sorted(core), deg, edges


def estimate_delta(V, nbrs, n_hop, max_pairs=1000, seed=0):
    """delta = mean similarity over sampled exactly-n-hop indirect node pairs."""
    rng = np.random.RandomState(seed)
    n = V.shape[0]
    sims, tries = [], 0
    order = rng.permutation(n)
    while len(sims) < max_pairs and tries < n * 4:
        src = int(order[tries % n]); tries += 1
        far = n_hop_nodes(nbrs, src, n_hop)
        if not far:
            continue
        k = int(far[rng.randint(len(far))])
        sims.append(_sim_np(V[src], V[k]))
    return (float(np.mean(sims)) if sims else 0.0), len(sims)


# ------------------------------------------- Algorithm 3: ConstraintCheck
def constraint_check(s, v, theta):
    C = int(torch.argmax(v))
    # L2: zero-sum, then box 0 <= v+s <= 1
    s = s - s.mean()
    s = torch.clamp(s, min=-v, max=1.0 - v)
    # L3: L1 distortion budget
    l1 = s.abs().sum()
    if l1 > theta:
        s = s * (theta / (l1 + EPS))
    # L1: keep the predicted label (argmax) unchanged
    for a in range(v.numel()):
        if a != C and (v[C] + s[C]) - (v[a] + s[a]) < 0:
            d = ((v[a] + s[a]) - (v[C] + s[C])) / 2.0
            s[a] = s[a] - d
            s[C] = s[C] + d
    return s


def _finalise(v, s, theta):
    """Hard guarantee of ALL constraints at once: argmax(v) preserved,
    v+s a valid probability distribution (>=0, sums to 1), and ||s||_1 <= theta.

    Works because the final s is scaled toward 0, i.e. r = v + a*(r0 - v) is a
    convex combination (a = theta/||.|| < 1) of v and a valid, label-preserving
    r0 -- so it inherits non-negativity, sum 1 (s sums to 0 -> scaling keeps it)
    and argmax C from both endpoints."""
    C = int(torch.argmax(v))
    r = torch.clamp(v + s, 0.0, 1.0)
    r = r / r.sum() if r.sum() > 0 else v.clone()
    if int(torch.argmax(r)) != C:                  # restore the original label
        r = r.clone(); r[C] = float(r.max()) + 1e-4; r = r / r.sum()
    s = r - v                                      # sums to 0
    l1 = float(s.abs().sum())
    if l1 > theta:                                 # hard budget cap (keeps sum 0)
        r = v + s * (theta / l1)
    return r


# ---------------------------------------------- Algorithm 2: optimise one node
def optimise_node(vi, P, delta, theta, iters, lr):
    """Push the similarity of node i to each adjacent node down to the n-hop
    floor delta (hinge), under the OPT-GRID constraints. Disguises adjacent
    nodes as n-hop indirect neighbours rather than driving similarity to zero,
    which is what defends against adaptive attacks."""
    v = torch.tensor(vi, dtype=torch.float64)
    s = torch.zeros_like(v, requires_grad=True)
    if not len(P):
        return v.numpy()
    Pt = torch.tensor(P, dtype=torch.float64)
    for _ in range(iters):
        r = v + s
        loss = torch.clamp(_sim_rows(r, Pt) - delta, min=0.0).sum()
        g, = torch.autograd.grad(loss, s)
        with torch.no_grad():
            s_new = constraint_check(s - lr * g, v, theta)
        s = s_new.detach().requires_grad_(True)
    return _finalise(v, s.detach(), theta).numpy()


# ----------------------------------------------------------------- driver
def load_pred(path):
    """Served posteriors as the attacker sees them: pickled [N,C] array."""
    with open(path, "rb") as f:
        return np.asarray(pickle.load(f), dtype=np.float64)


def apply_grid(dataset, setting, theta=0.4, n_hop=3, iters=60, lr=0.25,
               max_q=50, seed=0, verbose=True):
    V = np.load(os.path.join(POSTERIORS, f"{dataset}_{setting}.npy")).astype(np.float64)
    adj, nbrs = build_adjacency(dataset)
    return grid_posteriors(V, nbrs, dsname=f"{dataset}/{setting}", theta=theta,
                           n_hop=n_hop, iters=iters, lr=lr, seed=seed, verbose=verbose)


def grid_posteriors(V, nbrs, dsname="", theta=0.4, n_hop=3, iters=60, lr=0.25,
                    seed=0, verbose=True):
    """Apply GRID to a posterior matrix V given the graph neighbour lists.
    Perturbs only the core nodes; returns (R, stats)."""
    V = np.asarray(V, dtype=np.float64)
    V = V / V.sum(1, keepdims=True)
    n, C = V.shape
    assert len(nbrs) == n, f"graph nodes {len(nbrs)} != posteriors {n}"

    delta, n_samp = estimate_delta(V, nbrs, n_hop, seed=seed)
    core, deg, edges = select_core_nodes(V, nbrs, delta)
    if verbose:
        print(f"[{dsname}] N={n} C={C} edges={len(edges)} "
              f"delta={delta:.4f} (from {n_samp} {n_hop}-hop pairs) "
              f"core_nodes={len(core)} ({100*len(core)/n:.1f}%)")

    R = V.copy()
    rng = np.random.RandomState(seed)
    for idx, i in enumerate(core):
        P = V[[int(j) for j in nbrs[i]]] if len(nbrs[i]) else np.zeros((0, C))
        R[i] = optimise_node(V[i], P, delta, theta, iters, lr)
        if verbose and (idx + 1) % 400 == 0:
            print(f"    optimised {idx+1}/{len(core)} core nodes")

    stats = _validate(V, R, core, nbrs, theta, n_hop, rng)
    stats["core"] = len(core)
    if verbose:
        print(f"    label_changes={stats['label_changes']} (must be 0)  "
              f"max||s||1={stats['max_l1']:.3f} (budget {theta})  "
              f"rowsum_err={stats['rowsum_err']:.2e}")
        print(f"    adjacent sim  {stats['adj_before']:.4f} -> {stats['adj_after']:.4f}"
              f"   ({n_hop}-hop sim {stats['nhop_after']:.4f})")
    return R, stats


def _validate(V, R, core, nbrs, theta, n_hop, rng):
    label_changes = int((V.argmax(1) != R.argmax(1)).sum())
    l1 = np.abs(R - V).sum(1)
    rowsum_err = float(np.abs(R.sum(1) - 1).max())
    neg = float(R.min())
    # adjacent-pair similarity, before vs after (sample edges)
    pairs = []
    for i in range(V.shape[0]):
        for j in nbrs[i]:
            if int(j) > i:
                pairs.append((i, int(j)))
    if len(pairs) > 4000:
        pairs = [pairs[k] for k in rng.choice(len(pairs), 4000, replace=False)]
    adj_b = np.mean([_sim_np(V[i], V[j]) for i, j in pairs]) if pairs else 0.0
    adj_a = np.mean([_sim_np(R[i], R[j]) for i, j in pairs]) if pairs else 0.0
    # n-hop similarity after (reference level GRID targets)
    nh = []
    order = rng.permutation(V.shape[0])[:400]
    for s in order:
        far = n_hop_nodes(nbrs, int(s), n_hop)
        if far:
            nh.append(_sim_np(R[int(s)], R[int(far[rng.randint(len(far))])]))
    return dict(label_changes=label_changes, max_l1=float(l1.max()),
                mean_l1=float(l1[core].mean() if core else 0.0),
                rowsum_err=rowsum_err, min_val=neg,
                adj_before=float(adj_b), adj_after=float(adj_a),
                nhop_after=float(np.mean(nh) if nh else 0.0))


def save_pred(dataset, setting, R, bb_tag="kronbb", suffix="grid"):
    name = f"{dataset}_{bb_tag}_{setting}_{suffix}"
    path = os.path.join(PRED, f"{name}_pred.pkl")
    with open(path, "wb") as f:
        pickle.dump(R.astype(np.float64), f)
    return name, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--setting", default="whole")
    ap.add_argument("--theta", type=float, default=0.4)
    ap.add_argument("--nhop", type=int, default=3)
    ap.add_argument("--iters", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.25)
    ap.add_argument("--max_q", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bb_tag", default="kronbb")
    args = ap.parse_args()
    t0 = time.time()
    R, stats = apply_grid(args.dataset, args.setting, theta=args.theta,
                          n_hop=args.nhop, iters=args.iters, lr=args.lr,
                          max_q=args.max_q, seed=args.seed)
    name, path = save_pred(args.dataset, args.setting, R, bb_tag=args.bb_tag)
    print(f"  wrote {os.path.relpath(path, HERE)}  ({time.time()-t0:.1f}s)")
    if stats["label_changes"] != 0:
        raise SystemExit("ERROR: GRID changed predicted labels (utility violated)")


if __name__ == "__main__":
    main()
