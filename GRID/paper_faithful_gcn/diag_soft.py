"""On the SOFT cora posteriors: is the residual gap to the paper from my solver
being suboptimal, or is theta=0.4 fundamentally budget-limited even here?"""
import os, sys, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); GRIDROOT = os.path.dirname(HERE)
sys.path.insert(0, GRIDROOT)
import grid_defense as G
THETA = 0.4
V = G.load_pred(os.path.join(HERE, "attack", "data", "pred", "cora_softgcn_whole_pred.pkl"))
V = V / V.sum(1, keepdims=True)
adj, nbrs = G.build_adjacency("cora")
delta, _ = G.estimate_delta(V, nbrs, 3, seed=0)
core, deg, edges = G.select_core_nodes(V, nbrs, delta)
rng = np.random.RandomState(0)
sample = [int(x) for x in rng.choice(core, 120, replace=False)]

def msim(vec, P): return float(np.mean([G._sim_np(vec, p) for p in P])) if len(P) else 0.0
def analytic(vi, P, th):
    C=len(vi); c=int(vi.argmax()); m=th/2; best=msim(vi,P)
    for k in range(C):
        if k==c: continue
        s=np.zeros(C); s[c]=-m; s[k]=+m; r=vi+s
        if r.min()<-1e-9 or r.argmax()!=c: continue
        best=min(best, msim(r,P))
    return best

base=[]; d_def=[]; d_hvy=[]; d_ana=[]; peak=[]
for i in sample:
    P=V[[int(j) for j in nbrs[i]]]; base.append(msim(V[i],P))
    d_def.append(msim(G.optimise_node(V[i],P,delta,THETA,60,0.25),P))
    d_hvy.append(msim(G.optimise_node(V[i],P,delta,THETA,800,0.6),P))
    d_ana.append(analytic(V[i],P,THETA))
    peak.append(float(V[i].max()))
base,d_def,d_hvy,d_ana=map(np.array,(base,d_def,d_hvy,d_ana))
print(f"soft cora, delta(floor)={delta:.3f}, mean original peak={np.mean(peak):.3f}")
print(f"  baseline adjacent sim      {base.mean():.3f}")
print(f"  GRID default               {d_def.mean():.3f}  (reduced {base.mean()-d_def.mean():.3f})")
print(f"  GRID heavy (800,0.6)       {d_hvy.mean():.3f}  (reduced {base.mean()-d_hvy.mean():.3f})")
print(f"  analytic best single-move  {d_ana.mean():.3f}  (reduced {base.mean()-d_ana.mean():.3f})")
print(f"  floor (n-hop delta)        {delta:.3f}")
