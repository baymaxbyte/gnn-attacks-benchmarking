"""Is GRID's optimizer under-converged (bug) or budget-limited (fundamental)?
Compare, on cora-whole core nodes:
  (a) default optimise_node (iters=60, lr=0.25)
  (b) a HEAVY run (iters=800, lr=0.6) -- if (a)~=(b) the optimizer converged
  (c) an analytic 'best single-class move' upper bound under the SAME constraints
and report how far the confident peak actually moves (theory: <= theta/2).
"""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import grid_defense as G

THETA = 0.4
V = G.load_pred(os.path.join(HERE, "attack", "data", "pred", "cora_kronbb_whole_pred.pkl"))
V = V / V.sum(1, keepdims=True)
adj, nbrs = G.build_adjacency("cora")
delta, _ = G.estimate_delta(V, nbrs, 3, seed=0)
core, deg, edges = G.select_core_nodes(V, nbrs, delta)
rng = np.random.RandomState(0)
sample = [int(x) for x in rng.choice(core, 120, replace=False)]
print(f"cora whole: delta={delta:.3f}, core={len(core)}, sampling {len(sample)} nodes\n")


def mean_nbr_sim(vec, P):
    import numpy as np
    return float(np.mean([G._sim_np(vec, p) for p in P])) if len(P) else 0.0


def analytic_best(vi, P, theta):
    """Upper bound: move m=theta/2 from argmax class c onto the single other
    class that minimises mean neighbour similarity, keeping c the argmax."""
    C = len(vi); c = int(vi.argmax()); m = theta / 2.0
    best = mean_nbr_sim(vi, P)
    for k in range(C):
        if k == c: continue
        s = np.zeros(C); s[c] = -m; s[k] = +m
        r = vi + s
        if r.min() < -1e-9: continue
        if r.argmax() != c: continue            # argmax must stay c
        best = min(best, mean_nbr_sim(r, P))
    return best


d_def, d_heavy, d_base, d_ana = [], [], [], []
peak_drop, l1_used = [], []
for i in sample:
    P = V[[int(j) for j in nbrs[i]]]
    base = mean_nbr_sim(V[i], P); d_base.append(base)
    r_def = G.optimise_node(V[i], P, delta, THETA, iters=60,  lr=0.25)
    r_hvy = G.optimise_node(V[i], P, delta, THETA, iters=800, lr=0.6)
    d_def.append(mean_nbr_sim(r_def, P))
    d_heavy.append(mean_nbr_sim(r_hvy, P))
    d_ana.append(analytic_best(V[i], P, THETA))
    c = int(V[i].argmax())
    peak_drop.append(float(V[i][c] - r_def[c]))
    l1_used.append(float(np.abs(r_def - V[i]).sum()))

d_base, d_def, d_heavy, d_ana = map(np.array, (d_base, d_def, d_heavy, d_ana))
print("mean adjacent similarity (lower = better defence):")
print(f"  no defence (baseline)      {d_base.mean():.4f}")
print(f"  GRID default (60, 0.25)    {d_def.mean():.4f}   (reduced {d_base.mean()-d_def.mean():.4f})")
print(f"  GRID heavy   (800, 0.6)    {d_heavy.mean():.4f}   (reduced {d_base.mean()-d_heavy.mean():.4f})")
print(f"  analytic best-move bound   {d_ana.mean():.4f}   (reduced {d_base.mean()-d_ana.mean():.4f})")
print(f"\n  default vs heavy gap       {abs(d_def.mean()-d_heavy.mean()):.4f}  (small => converged)")
print(f"\nbudget use on confident nodes:")
print(f"  mean L1 used               {np.mean(l1_used):.3f} / {THETA}")
print(f"  mean peak-prob drop        {np.mean(peak_drop):.3f}  (theory max ~theta/2={THETA/2})")
pk = V[sample].max(1)
print(f"  mean original peak prob    {pk.mean():.3f}  (median {np.median(pk):.3f})")
